import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42


def preprocess(df):
    df = df.copy()

    compound = df["Compound"].astype(str).str.upper().str.strip()
    valid = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
    df["Compound_Normalized"] = compound.where(compound.isin(valid), "UNKNOWN")

    lap = df["LapNumber"].astype(float).clip(lower=1)
    progress = df["RaceProgress"].astype(float).clip(lower=1e-4, upper=1.0)
    total_laps = (lap / progress).clip(lower=40, upper=90)
    laps_remaining = (total_laps - lap).clip(lower=0)

    life_map = {
        "SOFT": 18.0,
        "MEDIUM": 28.0,
        "HARD": 40.0,
        "INTERMEDIATE": 22.0,
        "WET": 20.0,
        "UNKNOWN": 28.0,
    }
    expected_life = df["Compound_Normalized"].map(life_map).astype(float).fillna(28.0)
    tyre_life = df["TyreLife"].astype(float).clip(lower=0)
    tyre_age_at_finish = tyre_life + laps_remaining

    df["EstimatedTotalLaps"] = total_laps
    df["EstimatedLapsRemaining"] = laps_remaining
    df["CompoundExpectedLife"] = expected_life
    df["CurrentTyreAgeAtFinish"] = tyre_age_at_finish
    df["CurrentTyreFinishMargin"] = expected_life - tyre_age_at_finish
    df["CurrentTyreFinishPressure"] = (
        tyre_age_at_finish - expected_life
    ) / expected_life.clip(lower=1.0)
    df["CurrentTyreCannotFinish"] = (tyre_age_at_finish > expected_life).astype(np.int8)

    return df


def make_safe_columns(columns):
    safe, seen = [], {}
    for col in columns:
        name = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_")
        if not name:
            name = "feature"
        if name[0].isdigit():
            name = "f_" + name
        base = name
        k = seen.get(base, 0)
        if k:
            name = f"{base}_{k}"
        seen[base] = k + 1
        safe.append(name)
    return safe


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = preprocess(train)
test = preprocess(test)

y = train[TARGET].astype(int).values
X = train.drop(columns=[TARGET, ID_COL])
X_test = test.drop(columns=[ID_COL])

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    combined = pd.concat([X[col], X_test[col]], axis=0).astype(str).fillna("missing")
    categories = sorted(combined.unique())
    X[col] = pd.Categorical(X[col].astype(str).fillna("missing"), categories=categories)
    X_test[col] = pd.Categorical(
        X_test[col].astype(str).fillna("missing"), categories=categories
    )

safe_cols = make_safe_columns(X.columns)
rename_map = dict(zip(X.columns, safe_cols))
X = X.rename(columns=rename_map)
X_test = X_test.rename(columns=rename_map)
cat_cols = [rename_map[c] for c in cat_cols]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
medians = X[num_cols].replace([np.inf, -np.inf], np.nan).median()
X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(medians)
X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan).fillna(medians)

params = dict(
    objective="binary",
    n_estimators=1400,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(X), dtype=float)
fold_scores, best_iterations = [], []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    model = LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))
    best_iterations.append(int(model.best_iteration_ or params["n_estimators"]))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
final_estimators = int(np.median(best_iterations))
final_params = params.copy()
final_params["n_estimators"] = final_estimators

final_model = LGBMClassifier(**final_params)
final_model.fit(
    X,
    y,
    categorical_feature=cat_cols,
    callbacks=[log_evaluation(0)],
)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample[[ID_COL]].copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
    "final_n_estimators": final_estimators,
    "research_hypotheses_llm_claimed_used": ["000011"],
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
print(f"OOF ROC AUC: {cv_auc:.6f}")
print(json.dumps(result, indent=2))
