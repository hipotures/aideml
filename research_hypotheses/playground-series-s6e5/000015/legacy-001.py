import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SLICK_LIFE_LAPS = {"SOFT": 24.0, "MEDIUM": 34.0, "HARD": 44.0}


def clean_feature_name(name, idx):
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return f"f{idx}_{cleaned}"


def add_final_window_features(df):
    df = df.copy()

    progress = df["RaceProgress"].astype(float).clip(lower=1e-4)
    lap_number = df["LapNumber"].astype(float)
    row_total_laps = (
        (lap_number / progress).replace([np.inf, -np.inf], np.nan).clip(1, 120)
    )

    group_keys = [df["Year"], df["Race"]]
    group_total = row_total_laps.groupby(group_keys).transform("median")
    group_max_lap = lap_number.groupby(group_keys).transform("max")

    total_laps = group_total.fillna(row_total_laps)
    total_laps = np.maximum(total_laps, group_max_lap)
    total_laps = np.rint(total_laps).clip(lower=lap_number, upper=120)

    remaining_now = (total_laps - lap_number).clip(lower=0)
    remaining_next = (remaining_now - 1.0).clip(lower=0)

    df["est_total_laps"] = total_laps.astype(float)
    df["est_laps_remaining_now"] = remaining_now.astype(float)
    df["est_laps_remaining_next"] = remaining_next.astype(float)

    newly_viable_cols = []
    margin_now_arrays = []

    for compound, life in SLICK_LIFE_LAPS.items():
        cname = compound.lower()
        margin_now = life - remaining_now
        margin_next = life - remaining_next
        newly_viable = ((margin_now < 0.0) & (margin_next >= 0.0)).astype(np.int8)

        df[f"{cname}_fresh_final_margin_now"] = margin_now.astype(float)
        df[f"{cname}_fresh_final_margin_next"] = margin_next.astype(float)
        df[f"{cname}_newly_viable_nextlap"] = newly_viable

        newly_viable_cols.append(f"{cname}_newly_viable_nextlap")
        margin_now_arrays.append(np.asarray(margin_now, dtype=float))

    newly = df[newly_viable_cols].astype(np.int8)
    df["final_window_any_newly_viable"] = (newly.sum(axis=1) > 0).astype(np.int8)
    df["final_window_count_newly_viable"] = newly.sum(axis=1).astype(np.int8)

    df["final_window_softest_newly_viable"] = np.select(
        [
            df["soft_newly_viable_nextlap"].eq(1),
            df["medium_newly_viable_nextlap"].eq(1),
            df["hard_newly_viable_nextlap"].eq(1),
        ],
        [3, 2, 1],
        default=0,
    ).astype(np.int8)

    margin_matrix = np.vstack(margin_now_arrays)
    nearest_idx = np.argmin(np.abs(margin_matrix), axis=0)
    df["final_window_nearest_boundary_margin"] = margin_matrix[
        nearest_idx, np.arange(margin_matrix.shape[1])
    ].astype(float)
    df["final_window_nearest_boundary_abs_margin"] = np.min(
        np.abs(margin_matrix), axis=0
    ).astype(float)

    return df


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["__is_train"] = 1
test["__is_train"] = 0
test[TARGET] = np.nan

all_df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
all_df = add_final_window_features(all_df)

cat_cols_original = all_df.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols_original:
    all_df[col] = all_df[col].astype("string").fillna("__NA__").astype("category")

feature_cols = [c for c in all_df.columns if c not in [TARGET, ID_COL, "__is_train"]]
rename_map = {c: clean_feature_name(c, i) for i, c in enumerate(feature_cols)}

train_mask = all_df["__is_train"].eq(1).to_numpy()
X = (
    all_df.loc[train_mask, feature_cols]
    .rename(columns=rename_map)
    .reset_index(drop=True)
)
X_test = (
    all_df.loc[~train_mask, feature_cols]
    .rename(columns=rename_map)
    .reset_index(drop=True)
)
y = all_df.loc[train_mask, TARGET].astype(int).to_numpy()

cat_cols = [rename_map[c] for c in cat_cols_original if c in rename_map]
num_cols = [c for c in X.columns if c not in cat_cols]

for col in num_cols:
    combined = pd.concat([X[col], X_test[col]], ignore_index=True)
    fill_value = combined.replace([np.inf, -np.inf], np.nan).median()
    if not np.isfinite(fill_value):
        fill_value = 0.0
    X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    X_test[col] = X_test[col].replace([np.inf, -np.inf], np.nan).fillna(fill_value)

pos = max(float(y.sum()), 1.0)
neg = float(len(y) - y.sum())
scale_pos_weight = neg / pos


def make_model(seed, n_estimators):
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=os.cpu_count() or 1,
        verbosity=-1,
        force_row_wise=True,
    )


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(len(X), dtype=float)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
    model = make_model(SEED + fold, 2000)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))
    best_iterations.append(int(model.best_iteration_ or model.n_estimators))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold OOF ROC AUC: {cv_auc:.6f}")

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=int),
        "target": y.astype(int),
        "prediction": np.clip(oof, 0.0, 1.0),
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_rounds = int(np.clip(np.mean(best_iterations) * 1.05, 100, 2500))
final_model = make_model(SEED, final_rounds)
final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = np.clip(final_model.predict_proba(X_test)[:, 1], 0.0, 1.0)

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

test_predictions = sample[[ID_COL]].copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "research_hypotheses_llm_claimed_used": ["000015"],
    "metric": "roc_auc",
    "cv_folds": 5,
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
    "compound_life_laps": SLICK_LIFE_LAPS,
    "final_model_iterations": final_rounds,
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
}
print(json.dumps(review, indent=2))
