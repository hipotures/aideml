import os
import re
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT_DIR = Path("./input")
WORKING_DIR = Path("./working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 20260524
N_SPLITS = 5


def add_current_counted_dry_obligation_features(df):
    out = df.copy()
    sort_cols = ["Year", "Race", "Driver", "LapNumber", ID_COL]
    order = out.sort_values(sort_cols, kind="mergesort").index.to_numpy()

    years = out["Year"].to_numpy()
    races = out["Race"].astype(str).to_numpy()
    drivers = out["Driver"].astype(str).to_numpy()
    compounds = out["Compound"].astype(str).to_numpy()

    slick_bit = {"SOFT": 1, "MEDIUM": 2, "HARD": 4}
    popcount = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int8)

    n = len(out)
    diversity = np.zeros(n, dtype=np.int8)
    debt = np.zeros(n, dtype=np.int8)
    current_is_new = np.zeros(n, dtype=np.int8)

    prev_key = None
    seen_mask = 0

    for row in order:
        key = (years[row], races[row], drivers[row])
        if key != prev_key:
            prev_key = key
            seen_mask = 0

        bit = slick_bit.get(compounds[row], 0)
        if bit:
            current_is_new[row] = 1 if (seen_mask & bit) == 0 else 0
            seen_mask |= bit

        div = int(popcount[seen_mask])
        diversity[row] = div
        debt[row] = max(0, 2 - div)

    race_progress = out["RaceProgress"].to_numpy()
    out["slick_diversity_current_counted"] = diversity
    out["dry_compound_debt_current_counted"] = debt
    out["current_slick_compound_is_new"] = current_is_new
    out["late_race_limited_slick_diversity_65"] = (
        (race_progress >= 0.65) & (diversity < 2)
    ).astype(np.int8)
    out["late_race_limited_slick_diversity_75"] = (
        (race_progress >= 0.75) & (diversity < 2)
    ).astype(np.int8)
    out["very_late_dry_compound_debt_85"] = (
        (race_progress >= 0.85) & (debt > 0)
    ).astype(np.int8)
    return out


def sanitize_columns(columns):
    used = {}
    safe_cols = []
    mapping = {}
    for col in columns:
        safe = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
        if not safe:
            safe = "feature"
        if safe[0].isdigit():
            safe = "f_" + safe
        base = safe
        k = used.get(base, 0)
        while safe in used:
            k += 1
            safe = f"{base}_{k}"
        used[safe] = 1
        used[base] = max(used.get(base, 0), k)
        mapping[col] = safe
        safe_cols.append(safe)
    return safe_cols, mapping


def make_model(seed, n_estimators, scale_pos_weight):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=max(1, os.cpu_count() or 1),
        force_col_wise=True,
        deterministic=True,
        verbosity=-1,
    )


train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
train_features = train.drop(columns=[TARGET])
full = pd.concat([train_features, test], axis=0, ignore_index=True, sort=False)
full = add_current_counted_dry_obligation_features(full)

features = full.drop(columns=[ID_COL])
safe_cols, col_map = sanitize_columns(features.columns)
features.columns = safe_cols

cat_cols = [col_map[c] for c in ["Compound", "Race", "Driver"] if c in col_map]
for col in cat_cols:
    features[col] = features[col].astype("category")

features = features.replace([np.inf, -np.inf], np.nan)

X = features.iloc[: len(train)].reset_index(drop=True)
X_test = features.iloc[len(train) :].reset_index(drop=True)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train), dtype=np.float64)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    pos = max(1, int(y_tr.sum()))
    neg = max(1, int(len(y_tr) - y_tr.sum()))
    model = make_model(SEED + fold, 1200, neg / pos)

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(period=0)],
    )

    best_iter = getattr(model, "best_iteration_", None) or 1200
    best_iterations.append(int(best_iter))

    va_pred = model.predict_proba(X_va, num_iteration=best_iter)[:, 1]
    oof[va_idx] = va_pred
    fold_auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(fold_auc)
    print(f"fold {fold} roc_auc: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold oof roc_auc: {cv_auc:.6f}")

final_n_estimators = max(100, int(np.mean(best_iterations)))
full_pos = max(1, int(y.sum()))
full_neg = max(1, int(len(y) - y.sum()))
final_model = make_model(SEED, final_n_estimators, full_neg / full_pos)
final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y,
        "prediction": oof,
    }
).to_csv(WORKING_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

target_col = [c for c in sample.columns if c != ID_COL][0]
pred_by_id = pd.Series(test_pred, index=test[ID_COL].to_numpy())

submission = sample.copy()
submission[target_col] = submission[ID_COL].map(pred_by_id).astype(float)
if submission[target_col].isna().any():
    raise ValueError("Some sample submission ids were not found in test predictions.")

submission.to_csv(WORKING_DIR / "submission.csv", index=False)
submission.to_csv(
    WORKING_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000018"],
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": [float(x) for x in fold_scores],
            "final_n_estimators": int(final_n_estimators),
            "submission_path": str(WORKING_DIR / "submission.csv"),
            "oof_path": str(WORKING_DIR / "oof_predictions.csv.gz"),
            "test_predictions_path": str(WORKING_DIR / "test_predictions.csv.gz"),
        },
        sort_keys=True,
    )
)
