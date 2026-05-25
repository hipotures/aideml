import os
import re
import gc
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRanker

warnings.filterwarnings("ignore")

SEED = 392
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"

os.makedirs(WORK_DIR, exist_ok=True)


def sanitize_columns(columns):
    seen = {}
    out = []
    for col in columns:
        name = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
        name = name or "feature"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def build_features(df):
    x = df.copy()
    rp = x["RaceProgress"].astype(float).clip(lower=1e-4)
    lap = x["LapNumber"].astype(float)
    tyre = x["TyreLife"].astype(float).clip(lower=1.0)
    est_laps = (lap / rp).clip(lower=1.0, upper=120.0)

    x["EstimatedRaceLaps"] = est_laps
    x["EstimatedLapsRemaining"] = (est_laps - lap).clip(lower=-2.0, upper=100.0)
    x["TyreFracOfRace"] = tyre / est_laps
    x["TyreLifeToRaceProgress"] = tyre / rp
    x["DegradationPerTyreLap"] = x["Cumulative_Degradation"] / np.sqrt(tyre)
    x["DeltaPerTyreLap"] = x["LapTime_Delta"] / np.sqrt(tyre)
    x["LapTimePerProgress"] = x["LapTime (s)"] * rp
    x["AbsPositionChange"] = x["Position_Change"].abs()
    x["LateRace"] = (rp >= 0.65).astype(np.int8)
    x["LongStint"] = (tyre >= 12).astype(np.int8)
    x["StintTyreInteraction"] = x["Stint"].astype(float) * tyre
    x["PositionRaceProgress"] = x["Position"].astype(float) * rp
    return x


def make_rank_training_data(x_part, y_part):
    rp_bins = (
        pd.cut(
            x_part["RaceProgress"],
            bins=np.linspace(0.0, 1.000001, 21),
            labels=False,
            include_lowest=True,
        )
        .fillna(-1)
        .astype(int)
    )

    tyre_bins = (
        pd.cut(
            x_part["TyreLife"],
            bins=[-np.inf, 3, 6, 9, 12, 16, 21, 28, 40, np.inf],
            labels=False,
            include_lowest=True,
        )
        .fillna(-1)
        .astype(int)
    )

    key = (
        pd.DataFrame(
            {
                "Year": x_part["Year"].astype(str).values,
                "Race": x_part["Race"].astype(str).values,
                "Compound": x_part["Compound"].astype(str).values,
                "rp_bin": rp_bins.values,
                "tyre_bin": tyre_bins.values,
            }
        )
        .astype(str)
        .agg("_".join, axis=1)
    )

    stats = (
        pd.DataFrame({"key": key, "target": y_part})
        .groupby("key")["target"]
        .agg(["sum", "count"])
    )
    mixed_keys = stats.index[(stats["sum"] > 0) & (stats["sum"] < stats["count"])]
    hard_mask = key.isin(mixed_keys).values

    if hard_mask.sum() < 1000 or len(mixed_keys) < 10:
        return None, None, None

    selected = np.where(hard_mask)[0]
    selected_keys = key.iloc[selected].values
    order = np.argsort(selected_keys)
    idx = selected[order]
    sorted_keys = selected_keys[order]
    _, group_sizes = np.unique(sorted_keys, return_counts=True)

    return x_part.iloc[idx].copy(), y_part[idx].astype(int), group_sizes.tolist()


def base_params(y_train, n_estimators):
    pos = max(1, int(np.sum(y_train)))
    neg = max(1, len(y_train) - pos)
    return dict(
        objective="binary",
        metric="auc",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.2,
        scale_pos_weight=float(np.sqrt(neg / pos)),
        random_state=SEED,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
    )


def fit_base(x_tr, y_tr, cat_cols, x_va=None, y_va=None, n_estimators=1200):
    model = LGBMClassifier(**base_params(y_tr, n_estimators))
    if x_va is not None:
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(x_tr, y_tr, categorical_feature=cat_cols)
    return model


def fit_rank_specialist(x_tr, y_tr, cat_cols, seed):
    rank_x, rank_y, groups = make_rank_training_data(x_tr, y_tr)
    if rank_x is None:
        return None, None

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=450,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.8,
        label_gain=[0, 1],
        random_state=seed,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
    )

    ranker.fit(rank_x, rank_y, group=groups, categorical_feature=cat_cols)
    train_scores = ranker.predict(x_tr).reshape(-1, 1)

    calibrator = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
    calibrator.fit(train_scores, y_tr.astype(int))
    return ranker, calibrator


def specialist_predict(ranker, calibrator, x):
    if ranker is None or calibrator is None:
        return None
    scores = ranker.predict(x).reshape(-1, 1)
    return calibrator.predict_proba(scores)[:, 1]


def blend_predictions(base_pred, spec_pred, x, train_base_pred):
    if spec_pred is None:
        return np.clip(base_pred, 1e-7, 1 - 1e-7)

    low = np.quantile(train_base_pred, 0.65)
    high = np.quantile(train_base_pred, 0.997)
    uncertain = (base_pred >= low) & (base_pred <= high)

    late_stint = (
        (x["RaceProgress"].values >= 0.45)
        & (x["TyreLife"].values >= 7)
        & (x["PitStop"].values == 0)
    )

    gate = np.maximum(uncertain.astype(float), 0.75 * late_stint.astype(float))
    alpha = 0.28
    pred = base_pred * (1.0 - alpha * gate) + spec_pred * (alpha * gate)
    return np.clip(pred, 1e-7, 1 - 1e-7)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
n_train = len(train)

raw_all = pd.concat(
    [
        train.drop(columns=[TARGET, ID_COL]),
        test.drop(columns=[ID_COL]),
    ],
    axis=0,
    ignore_index=True,
)

features = build_features(raw_all)
original_cols = features.columns.tolist()
safe_cols = sanitize_columns(original_cols)
name_map = dict(zip(original_cols, safe_cols))
features.columns = safe_cols

cat_cols = [
    name_map[c] for c in ["Driver", "Race", "Compound", "Year"] if c in name_map
]
for col in cat_cols:
    features[col] = features[col].astype("category")

for col in features.columns:
    if col in cat_cols:
        continue
    if pd.api.types.is_float_dtype(features[col]):
        features[col] = features[col].astype("float32")
    elif pd.api.types.is_integer_dtype(features[col]):
        features[col] = features[col].astype("int32")

X = features.iloc[:n_train].reset_index(drop=True)
X_test = features.iloc[n_train:].reset_index(drop=True)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_base = np.zeros(n_train, dtype=np.float32)
oof_blend = np.zeros(n_train, dtype=np.float32)
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    x_tr = X.iloc[tr_idx].reset_index(drop=True)
    x_va = X.iloc[va_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    base = fit_base(x_tr, y_tr, cat_cols, x_va, y_va, n_estimators=1200)
    best_iter = getattr(base, "best_iteration_", None) or 1200
    best_iters.append(int(best_iter))

    base_tr_pred = base.predict_proba(x_tr)[:, 1]
    base_va_pred = base.predict_proba(x_va)[:, 1]

    ranker, calibrator = fit_rank_specialist(x_tr, y_tr, cat_cols, SEED + fold)
    spec_va_pred = specialist_predict(ranker, calibrator, x_va)

    blend_va_pred = blend_predictions(base_va_pred, spec_va_pred, x_va, base_tr_pred)

    oof_base[va_idx] = base_va_pred
    oof_blend[va_idx] = blend_va_pred

    fold_base_auc = roc_auc_score(y_va, base_va_pred)
    fold_blend_auc = roc_auc_score(y_va, blend_va_pred)
    print(
        f"Fold {fold}: base_auc={fold_base_auc:.6f}, blended_auc={fold_blend_auc:.6f}"
    )

    del base, ranker, calibrator, x_tr, x_va
    gc.collect()

base_auc = roc_auc_score(y, oof_base)
blend_auc = roc_auc_score(y, oof_blend)

final_estimators = int(np.clip(np.mean(best_iters), 250, 1200))
print(f"Mean best base iterations: {final_estimators}")

final_base = fit_base(X, y, cat_cols, n_estimators=final_estimators)
final_base_train_pred = final_base.predict_proba(X)[:, 1]
final_base_test_pred = final_base.predict_proba(X_test)[:, 1]

final_ranker, final_calibrator = fit_rank_specialist(X, y, cat_cols, SEED + 999)
final_spec_test_pred = specialist_predict(final_ranker, final_calibrator, X_test)
test_pred = blend_predictions(
    final_base_test_pred, final_spec_test_pred, X_test, final_base_train_pred
)

submission = sample.copy()
submission[TARGET] = test_pred.astype(float)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

oof_df = pd.DataFrame(
    {
        "row": np.arange(n_train, dtype=np.int32),
        "target": y.astype(int),
        "prediction": oof_blend.astype(float),
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

print(f"Base OOF ROC AUC: {base_auc:.6f}")
print(f"Blended OOF ROC AUC: {blend_auc:.6f}")
print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_auc": float(blend_auc),
            "base_cv_auc": float(base_auc),
            "research_hypotheses_llm_claimed_used": ["000392"],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
            "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
            "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
        }
    )
)
