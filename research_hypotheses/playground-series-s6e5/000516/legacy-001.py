import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

from lightgbm import LGBMClassifier, early_stopping

warnings.filterwarnings("ignore")

RANDOM_STATE = 2026
N_SPLITS = 5
RULE_BLEND_WEIGHT = 0.15
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

TARGET = "PitNextLap"
ID_COL = "id"
y = train[TARGET].astype(int).to_numpy()


def safe_name(name):
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    if not name:
        name = "feature"
    if name[0].isdigit():
        name = "f_" + name
    return name


orig_numeric = [
    "LapNumber",
    "LapTime (s)",
    "LapTime_Delta",
    "Position",
    "Position_Change",
    "PitStop",
    "RaceProgress",
    "Stint",
    "TyreLife",
    "Cumulative_Degradation",
    "Year",
]
orig_cats = ["Compound", "Driver", "Race"]
num_map = {c: safe_name(c) for c in orig_numeric}
cat_map = {c: safe_name(c) for c in orig_cats}


def build_base(df):
    out = pd.DataFrame(index=df.index)
    for src, dst in num_map.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce").astype("float32")

    for src, dst in cat_map.items():
        out[dst] = df[src].astype("string").fillna("__NA__").astype(str)

    out["RaceYear"] = (
        df["Race"].astype("string").fillna("__NA__").astype(str)
        + "_"
        + df["Year"].astype(str)
    )

    eps_lap = np.maximum(out["LapNumber"].to_numpy(dtype=np.float32), 1.0)
    eps_tyre = np.maximum(out["TyreLife"].to_numpy(dtype=np.float32), 1.0)

    out["TyreLife_x_RaceProgress"] = out["TyreLife"] * out["RaceProgress"]
    out["TyreLife_x_Stint"] = out["TyreLife"] * out["Stint"]
    out["TyreLife_x_Cumulative_Degradation"] = (
        out["TyreLife"] * out["Cumulative_Degradation"]
    )
    out["RaceProgress_x_Stint"] = out["RaceProgress"] * out["Stint"]
    out["RaceProgress_x_Cumulative_Degradation"] = (
        out["RaceProgress"] * out["Cumulative_Degradation"]
    )
    out["LapNumber_x_RaceProgress"] = out["LapNumber"] * out["RaceProgress"]
    out["LapTime_Delta_x_TyreLife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["TyreLife_per_LapNumber"] = out["TyreLife"].to_numpy(dtype=np.float32) / eps_lap
    out["Cumulative_Degradation_per_TyreLife"] = (
        out["Cumulative_Degradation"].to_numpy(dtype=np.float32) / eps_tyre
    )

    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = (
        out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    )
    return out


base_train = build_base(train)
base_test = build_base(test)

numeric_cols = [num_map[c] for c in orig_numeric]
interaction_cols = [
    "TyreLife_x_RaceProgress",
    "TyreLife_x_Stint",
    "TyreLife_x_Cumulative_Degradation",
    "RaceProgress_x_Stint",
    "RaceProgress_x_Cumulative_Degradation",
    "LapNumber_x_RaceProgress",
    "LapTime_Delta_x_TyreLife",
    "TyreLife_per_LapNumber",
    "Cumulative_Degradation_per_TyreLife",
]
cat_cols = [cat_map[c] for c in orig_cats] + ["RaceYear"]
rule_cols = numeric_cols + interaction_cols
main_base_cols = rule_cols + cat_cols

for col in cat_cols:
    all_values = pd.concat([base_train[col], base_test[col]], ignore_index=True).astype(
        str
    )
    dtype = pd.CategoricalDtype(categories=pd.Index(all_values.unique()))
    base_train[col] = base_train[col].astype(str).astype(dtype)
    base_test[col] = base_test[col].astype(str).astype(dtype)


def balanced_weights(labels):
    labels = np.asarray(labels)
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    weights = np.ones(labels.shape[0], dtype=np.float32)
    if pos > 0:
        weights[labels == 1] = neg / pos
    return weights


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def blend_probs(main_p, rule_p, rule_weight=RULE_BLEND_WEIGHT):
    return sigmoid((1.0 - rule_weight) * logit(main_p) + rule_weight * logit(rule_p))


def fit_rule_model(X, labels, seed):
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=12,
        max_bins=255,
        min_samples_leaf=80,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
        random_state=seed,
    )
    model.fit(X, labels, sample_weight=balanced_weights(labels))
    return model


def extract_pdp_thresholds(
    model,
    X,
    feature_names,
    seed,
    sample_size=4000,
    grid_size=25,
    per_feature=2,
    max_flags=32,
):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    base = X[sample_idx].copy()
    thresholds = []

    for j, feature in enumerate(feature_names):
        values = X[:, j]
        values = values[np.isfinite(values)]
        if values.size < 20:
            continue

        unique_values = np.unique(values)
        if unique_values.size <= 2:
            continue
        if unique_values.size <= grid_size:
            grid = np.sort(unique_values).astype(np.float32)
        else:
            grid = np.unique(
                np.quantile(values, np.linspace(0.02, 0.98, grid_size))
            ).astype(np.float32)

        if grid.size < 3:
            continue

        work = base.copy()
        pdp = np.empty(grid.size, dtype=np.float32)
        for k, val in enumerate(grid):
            work[:, j] = val
            pdp[k] = model.predict_proba(work)[:, 1].mean()

        jumps = np.abs(np.diff(pdp))
        if (
            jumps.size == 0
            or not np.isfinite(jumps).any()
            or float(np.nanmax(jumps)) <= 0
        ):
            continue

        spread = float(grid[-1] - grid[0])
        min_gap = max(spread * 0.05, 1e-6)
        chosen = []
        for jump_idx in np.argsort(jumps)[::-1]:
            impact = float(jumps[jump_idx])
            if impact <= 0:
                break
            threshold = float((grid[jump_idx] + grid[jump_idx + 1]) / 2.0)
            if all(abs(threshold - old) >= min_gap for old in chosen):
                thresholds.append(
                    {"feature": feature, "threshold": threshold, "impact": impact}
                )
                chosen.append(threshold)
            if len(chosen) >= per_feature:
                break

    thresholds.sort(key=lambda d: d["impact"], reverse=True)
    return thresholds[:max_flags]


def add_threshold_flags(frame, thresholds):
    out = frame[main_base_cols].copy()
    for i, item in enumerate(thresholds):
        col = f"thr_{i:02d}_{safe_name(item['feature'])}"
        out[col] = (
            frame[item["feature"]].astype("float32") >= item["threshold"]
        ).astype("int8")
    return out


groups = (
    train["Race"].astype(str)
    + "_"
    + train["Year"].astype(str)
    + "_"
    + train["Driver"].astype(str)
).to_numpy()

if HAS_SGK:
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(cv.split(np.zeros(len(y)), y, groups))
else:
    cv = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(np.zeros(len(y)), y, groups))

oof = np.zeros(len(train), dtype=np.float64)
test_pred = np.zeros(len(test), dtype=np.float64)
fold_aucs = []
threshold_counts = []

X_rule_test = base_test[rule_cols].to_numpy(dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(splits, start=1):
    y_tr, y_val = y[tr_idx], y[val_idx]

    X_rule_tr = base_train.iloc[tr_idx][rule_cols].to_numpy(dtype=np.float32)
    X_rule_val = base_train.iloc[val_idx][rule_cols].to_numpy(dtype=np.float32)

    rule_model = fit_rule_model(X_rule_tr, y_tr, RANDOM_STATE + fold)
    thresholds = extract_pdp_thresholds(
        rule_model, X_rule_tr, rule_cols, RANDOM_STATE + 100 * fold
    )
    threshold_counts.append(len(thresholds))

    rule_val = rule_model.predict_proba(X_rule_val)[:, 1]
    rule_test = rule_model.predict_proba(X_rule_test)[:, 1]

    X_main_tr = add_threshold_flags(base_train.iloc[tr_idx], thresholds)
    X_main_val = add_threshold_flags(base_train.iloc[val_idx], thresholds)
    X_main_test = add_threshold_flags(base_test, thresholds)

    pos = max(float((y_tr == 1).sum()), 1.0)
    neg = float((y_tr == 0).sum())

    main_model = LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=1400,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE + fold,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )

    main_model.fit(
        X_main_tr,
        y_tr,
        eval_set=[(X_main_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
    )

    main_val = main_model.predict_proba(X_main_val)[:, 1]
    main_test = main_model.predict_proba(X_main_test)[:, 1]

    val_pred = blend_probs(main_val, rule_val)
    fold_test_pred = blend_probs(main_test, rule_test)

    oof[val_idx] = val_pred
    test_pred += fold_test_pred / len(splits)

    fold_auc = roc_auc_score(y_val, val_pred)
    fold_aucs.append(fold_auc)
    print(
        f"Fold {fold} ROC AUC: {fold_auc:.6f} using {len(thresholds)} PDP threshold flags"
    )

cv_auc = roc_auc_score(y, oof)
test_pred = np.clip(test_pred, 0.0, 1.0)

target_col = [c for c in sample.columns if c != ID_COL][0]

submission = sample[[ID_COL]].copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y.astype(int),
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_predictions = sample[[ID_COL]].copy()
test_predictions[target_col] = test_pred
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(f"5-fold grouped OOF ROC AUC: {cv_auc:.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000516"],
            "metric": "roc_auc",
            "cv_auc": float(cv_auc),
            "fold_auc": [float(v) for v in fold_aucs],
            "mean_pdp_threshold_flags": float(np.mean(threshold_counts)),
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
            "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
            "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
        },
        sort_keys=True,
    )
)
