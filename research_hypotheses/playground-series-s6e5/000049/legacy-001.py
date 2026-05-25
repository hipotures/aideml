import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESES_USED = ["000049"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values

base_features = [c for c in test.columns if c != ID_COL]
num_cols = [
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "TyreLife",
    "Position",
    "Position_Change",
    "RaceProgress",
    "LapNumber",
    "Stint",
]
cat_cols = ["Driver", "Race", "Compound", "Year"]

all_df = pd.concat(
    [
        train[base_features].copy(),
        test[base_features].copy(),
    ],
    axis=0,
    ignore_index=True,
)

# Hypothesis 000049: target-free group-relative numeric medians.
# Same-lap race context avoids within-driver-race future summaries.
same_lap_group = ["Year", "Race", "LapNumber"]
same_lap_numeric = [
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "TyreLife",
    "Position",
]

for col in same_lap_numeric:
    med = all_df.groupby(same_lap_group, observed=True)[col].transform("median")
    all_df[f"{col}_minus_race_lap_median"] = all_df[col] - med

# Broad non-sequential global baselines.
for group_col in ["Compound", "Year"]:
    for col in num_cols:
        med = all_df.groupby(group_col, observed=True)[col].transform("median")
        all_df[f"{col}_minus_{group_col}_median"] = all_df[col] - med

# A few stable interaction-like ratios against target-free broad medians.
for group_col in ["Compound", "Year"]:
    for col in ["TyreLife", "Cumulative_Degradation", "LapTime_Delta"]:
        med = all_df.groupby(group_col, observed=True)[col].transform("median")
        all_df[f"{col}_ratio_{group_col}_median"] = all_df[col] / (med.abs() + 1e-6)

# Encode categoricals consistently across train and test.
for col in cat_cols:
    all_df[col] = all_df[col].astype("category").cat.codes.astype("int32")

all_df = all_df.replace([np.inf, -np.inf], np.nan)
for col in all_df.columns:
    if all_df[col].isna().any():
        all_df[col] = all_df[col].fillna(all_df[col].median())

X = all_df.iloc[: len(train)].copy()
X_test = all_df.iloc[len(train) :].copy()

oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=49)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    model = LGBMClassifier(
        objective="binary",
        n_estimators=2500,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=4900 + fold,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        callbacks=[],
    )

    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits

    fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
    fold_scores.append(fold_auc)
    print(f"fold_{fold}_roc_auc={fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"cv_roc_auc={cv_auc:.6f}")
print(f"mean_fold_roc_auc={np.mean(fold_scores):.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_predictions = sample.copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "mean_fold_roc_auc": float(np.mean(fold_scores)),
            "research_hypotheses_llm_claimed_used": HYPOTHESES_USED,
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        }
    )
)
