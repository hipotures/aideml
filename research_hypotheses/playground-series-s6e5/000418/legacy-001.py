import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
N_FOLDS = 5
EMBARGO_RACES = 1

os.makedirs(WORK_DIR, exist_ok=True)
np.random.seed(SEED)

train_raw = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test_raw = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train_raw[TARGET].astype(int).to_numpy()


def add_features(df):
    df = df.copy()

    race_progress = df["RaceProgress"].clip(lower=1e-3, upper=1.0)
    est_total_laps = (df["LapNumber"] / race_progress).replace(
        [np.inf, -np.inf], np.nan
    )
    est_total_laps = est_total_laps.fillna(df["LapNumber"].max()).clip(
        lower=1.0, upper=110.0
    )

    compound = df["Compound"].astype(str).str.upper()
    compound_wear = {
        "SOFT": 1.18,
        "MEDIUM": 1.00,
        "HARD": 0.82,
        "INTERMEDIATE": 1.08,
        "WET": 1.12,
    }
    expected_stint = {
        "SOFT": 18.0,
        "MEDIUM": 25.0,
        "HARD": 34.0,
        "INTERMEDIATE": 16.0,
        "WET": 14.0,
    }

    wear_rate = compound.map(compound_wear).fillna(1.0).astype(float)
    expected_len = compound.map(expected_stint).fillna(24.0).astype(float)
    clipped_deg = df["Cumulative_Degradation"].clip(lower=-100.0, upper=2500.0)

    df["est_total_laps"] = est_total_laps
    df["laps_remaining"] = (est_total_laps - df["LapNumber"]).clip(
        lower=0.0, upper=110.0
    )
    df["compound_wear_rate"] = wear_rate
    df["latent_wear"] = df["TyreLife"] * wear_rate + 0.015 * clipped_deg
    df["expected_stint_len"] = expected_len
    df["stop_debt"] = (df["TyreLife"] - expected_len).clip(lower=-30.0, upper=80.0)
    df["stint_progress"] = (df["TyreLife"] / expected_len.clip(lower=1.0)).clip(
        0.0, 5.0
    )
    df["degradation_per_lap"] = (
        df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1.0)
    ).clip(-50.0, 80.0)
    df["lap_delta_abs"] = df["LapTime_Delta"].abs()
    df["position_pressure"] = ((21.0 - df["Position"]) / 20.0).clip(0.0, 1.0) * df[
        "RaceProgress"
    ]
    df["late_first_stint_pressure"] = np.maximum(0.0, df["RaceProgress"] - 0.55) * (
        df["Stint"] <= 1
    ).astype(float)
    df["current_stop_cooldown"] = df["PitStop"].astype(float) * (
        1.0 + df["RaceProgress"]
    )
    df["lap_time_log"] = np.log1p(df["LapTime (s)"].clip(lower=0.0))
    df["stint_x_wear"] = df["Stint"] * wear_rate

    return df


train = add_features(train_raw.drop(columns=[TARGET]))
test = add_features(test_raw)

feature_cols = [c for c in train.columns if c != ID_COL]
cat_cols = [c for c in feature_cols if train[c].dtype == "object"]

for col in cat_cols:
    cats = pd.concat([train[col], test[col]], ignore_index=True).astype(str).unique()
    train[col] = pd.Categorical(train[col].astype(str), categories=cats)
    test[col] = pd.Categorical(test[col].astype(str), categories=cats)

race_key = train["Year"].astype(str) + "|" + train["Race"].astype(str)
query_key = race_key + "|" + train["Driver"].astype(str)


def make_purged_race_folds():
    first_id = (
        pd.DataFrame({"race_key": race_key, "id": train[ID_COL]})
        .groupby("race_key")["id"]
        .min()
        .sort_values()
    )
    ordered_groups = first_id.index.to_numpy()
    blocks = [b for b in np.array_split(ordered_groups, N_FOLDS) if len(b) > 0]
    group_pos = {g: i for i, g in enumerate(ordered_groups)}
    all_groups = set(ordered_groups.tolist())

    folds = []
    for val_groups in blocks:
        lo = min(group_pos[g] for g in val_groups)
        hi = max(group_pos[g] for g in val_groups) + 1
        purged = set(
            ordered_groups[
                max(0, lo - EMBARGO_RACES) : min(
                    len(ordered_groups), hi + EMBARGO_RACES
                )
            ].tolist()
        )

        train_groups = all_groups - purged
        val_mask = race_key.isin(val_groups)
        train_mask = race_key.isin(train_groups)

        if train_mask.sum() == 0 or y[train_mask.to_numpy()].sum() == 0:
            train_mask = ~val_mask

        folds.append(
            (
                np.flatnonzero(train_mask.to_numpy()),
                np.flatnonzero(val_mask.to_numpy()),
            )
        )
    return folds


def ordered_query_index(idx):
    tmp = pd.DataFrame({"idx": idx, "q": query_key.iloc[idx].to_numpy()})
    tmp = tmp.sort_values(["q", "idx"], kind="mergesort")
    groups = tmp.groupby("q", sort=False).size().astype(int).tolist()
    return tmp["idx"].to_numpy(dtype=np.int64), groups


def clean_pred(pred):
    pred = np.asarray(pred, dtype=np.float64)
    finite = np.isfinite(pred)
    fill = float(np.median(pred[finite])) if finite.any() else 0.0
    return np.nan_to_num(pred, nan=fill, posinf=fill, neginf=fill)


def rankify(pred):
    return (
        pd.Series(clean_pred(pred))
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )


def safe_auc(target, pred):
    if np.unique(target).size < 2:
        return float("nan")
    return float(roc_auc_score(target, pred))


def numeric_frame(df, cols, medians=None):
    x = df[cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    if medians is None:
        medians = x.median()
    return x.fillna(medians).astype(np.float32), medians


folds = make_purged_race_folds()
n_jobs = min(8, os.cpu_count() or 1)
base_names = ["lgbm_gbdt", "lgbm_lambdarank", "xgb_monotone_physics"]
oof_raw = np.zeros((len(train), len(base_names)), dtype=np.float32)
oof_rank = np.zeros_like(oof_raw)
test_rank_sum = np.zeros((len(test), len(base_names)), dtype=np.float32)

specialist_cols = [
    "TyreLife",
    "RaceProgress",
    "LapNumber",
    "laps_remaining",
    "latent_wear",
    "compound_wear_rate",
    "stop_debt",
    "stint_progress",
    "PitStop",
    "degradation_per_lap",
    "Cumulative_Degradation",
    "late_first_stint_pressure",
]
specialist_cols = [c for c in specialist_cols if c in train.columns]
mono_constraints = {
    "TyreLife": 1,
    "RaceProgress": 1,
    "LapNumber": 1,
    "laps_remaining": -1,
    "latent_wear": 1,
    "compound_wear_rate": 1,
    "stop_debt": 1,
    "stint_progress": 1,
    "PitStop": -1,
    "degradation_per_lap": 1,
    "Cumulative_Degradation": 1,
    "late_first_stint_pressure": 1,
}
mono = "(" + ",".join(str(mono_constraints[c]) for c in specialist_cols) + ")"

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    X_tr = train.iloc[tr_idx][feature_cols]
    X_va = train.iloc[va_idx][feature_cols]
    X_te = test[feature_cols]
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    pos = max(1, int(y_tr.sum()))
    neg = max(1, len(y_tr) - pos)
    spw = min(200.0, neg / pos)

    gbdt = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        random_state=SEED + fold,
        n_jobs=n_jobs,
        verbosity=-1,
        force_col_wise=True,
    )
    gbdt.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    pred_va = clean_pred(
        gbdt.predict_proba(X_va, num_iteration=gbdt.best_iteration_ or None)[:, 1]
    )
    pred_te = clean_pred(
        gbdt.predict_proba(X_te, num_iteration=gbdt.best_iteration_ or None)[:, 1]
    )
    oof_raw[va_idx, 0] = pred_va
    oof_rank[va_idx, 0] = rankify(pred_va)
    test_rank_sum[:, 0] += rankify(pred_te)

    sorted_idx, rank_groups = ordered_query_index(tr_idx)
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_estimators=420,
        learning_rate=0.045,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=SEED + 100 + fold,
        n_jobs=n_jobs,
        verbosity=-1,
        force_col_wise=True,
    )
    ranker.fit(
        train.iloc[sorted_idx][feature_cols],
        y[sorted_idx],
        group=rank_groups,
        categorical_feature=cat_cols,
    )
    pred_va = clean_pred(ranker.predict(X_va))
    pred_te = clean_pred(ranker.predict(X_te))
    oof_raw[va_idx, 1] = pred_va
    oof_rank[va_idx, 1] = rankify(pred_va)
    test_rank_sum[:, 1] += rankify(pred_te)

    X_trn, medians = numeric_frame(train.iloc[tr_idx], specialist_cols)
    X_van, _ = numeric_frame(train.iloc[va_idx], specialist_cols, medians)
    X_ten, _ = numeric_frame(test, specialist_cols, medians)

    mono_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=650,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=40,
        subsample=0.85,
        colsample_bytree=0.95,
        reg_alpha=0.05,
        reg_lambda=2.0,
        scale_pos_weight=spw,
        tree_method="hist",
        max_bin=256,
        monotone_constraints=mono,
        random_state=SEED + 200 + fold,
        n_jobs=n_jobs,
        early_stopping_rounds=60,
    )
    mono_model.fit(X_trn, y_tr, eval_set=[(X_van, y_va)], verbose=False)
    pred_va = clean_pred(mono_model.predict_proba(X_van)[:, 1])
    pred_te = clean_pred(mono_model.predict_proba(X_ten)[:, 1])
    oof_raw[va_idx, 2] = pred_va
    oof_rank[va_idx, 2] = rankify(pred_va)
    test_rank_sum[:, 2] += rankify(pred_te)

    fold_scores = {
        base_names[i]: safe_auc(y_va, oof_raw[va_idx, i])
        for i in range(len(base_names))
    }
    print(
        f"fold {fold}/{len(folds)} rows train={len(tr_idx)} valid={len(va_idx)} auc={json.dumps(fold_scores)}"
    )

test_stack = test_rank_sum / len(folds)
mean_rank_oof = oof_rank.mean(axis=1)

meta_oof = np.zeros(len(train), dtype=np.float32)
for tr_idx, va_idx in folds:
    meta = LogisticRegression(
        C=0.5,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )
    meta.fit(oof_rank[tr_idx], y[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(oof_rank[va_idx])[:, 1]

final_meta = LogisticRegression(
    C=0.5,
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=SEED,
)
final_meta.fit(oof_rank, y)
test_pred = final_meta.predict_proba(test_stack)[:, 1]
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)
meta_oof = np.clip(meta_oof, 1e-6, 1 - 1e-6)

submission = sample.copy()
target_col = submission.columns[1]
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y,
        "prediction": meta_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

base_oof_auc = {
    base_names[i]: safe_auc(y, oof_raw[:, i]) for i in range(len(base_names))
}
result = {
    "research_hypotheses_llm_claimed_used": ["000418"],
    "metric": "roc_auc",
    "rank_stack_cv_auc": safe_auc(y, meta_oof),
    "mean_rank_oof_auc": safe_auc(y, mean_rank_oof),
    "base_oof_auc": base_oof_auc,
    "n_folds": len(folds),
    "embargo_races": EMBARGO_RACES,
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
}
print(json.dumps(result, indent=2))
