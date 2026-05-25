import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID = "id"
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}

train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")


def add_physical_features(df):
    df = df.copy()
    df["Compound"] = df["Compound"].astype(str).str.upper()
    expected_life = {
        "SOFT": 18.0,
        "MEDIUM": 28.0,
        "HARD": 38.0,
        "INTERMEDIATE": 16.0,
        "WET": 14.0,
    }
    hardness = {
        "SOFT": 1.0,
        "MEDIUM": 2.0,
        "HARD": 3.0,
        "INTERMEDIATE": 0.5,
        "WET": 0.25,
    }

    df["ExpectedTyreLife"] = df["Compound"].map(expected_life).fillna(28.0)
    df["CompoundHardness"] = df["Compound"].map(hardness).fillna(1.0)

    race_progress = df["RaceProgress"].clip(1e-4, 1.0)
    df["EstimatedRaceLaps"] = (df["LapNumber"] / race_progress).clip(
        lower=df["LapNumber"], upper=120
    )
    df["LapsRemaining"] = (df["EstimatedRaceLaps"] - df["LapNumber"]).clip(lower=0)

    df["TyreLife_ExpectedLife_Ratio"] = df["TyreLife"] / df["ExpectedTyreLife"]
    df["ExpectedLife_Remaining"] = df["ExpectedTyreLife"] - df["TyreLife"]
    df["LapsRemaining_AfterExpectedLife"] = (
        df["TyreLife"] + df["LapsRemaining"] - df["ExpectedTyreLife"]
    ).clip(lower=0)

    positive_degradation = df["Cumulative_Degradation"].clip(lower=0)
    df["Wear_Pressure"] = positive_degradation * df["TyreLife_ExpectedLife_Ratio"].clip(
        lower=0
    )
    df["LapTime_Loss"] = df["LapTime_Delta"].clip(lower=0)

    df["PastExpectedLife"] = (df["TyreLife"] > df["ExpectedTyreLife"]).astype(np.int8)
    df["CanReachEnd_OnExpectedLife"] = (
        df["TyreLife"] + df["LapsRemaining"] <= df["ExpectedTyreLife"]
    ).astype(np.int8)
    df["Abs_Position_Change"] = df["Position_Change"].abs()
    df["LateRace"] = (df["RaceProgress"] >= 0.80).astype(np.int8)
    df["RaceGroup"] = df["Year"].astype(str) + "__" + df["Race"].astype(str)

    return df.replace([np.inf, -np.inf], np.nan)


train_fe = add_physical_features(train)
test_fe = add_physical_features(test)

cat_features = ["Compound", "Driver", "Race"]
for col in cat_features:
    cats = pd.Index(
        pd.concat([train_fe[col], test_fe[col]], ignore_index=True)
        .astype(str)
        .fillna("__MISSING__")
        .unique()
    )
    train_fe[col] = pd.Categorical(
        train_fe[col].astype(str).fillna("__MISSING__"), categories=cats
    )
    test_fe[col] = pd.Categorical(
        test_fe[col].astype(str).fillna("__MISSING__"), categories=cats
    )

feature_cols = [
    "Year",
    "LapNumber",
    "Position",
    "Position_Change",
    "Stint",
    "TyreLife",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "RaceProgress",
    "PitStop",
    "ExpectedTyreLife",
    "CompoundHardness",
    "EstimatedRaceLaps",
    "LapsRemaining",
    "ExpectedLife_Remaining",
    "TyreLife_ExpectedLife_Ratio",
    "LapsRemaining_AfterExpectedLife",
    "Wear_Pressure",
    "LapTime_Loss",
    "PastExpectedLife",
    "CanReachEnd_OnExpectedLife",
    "Abs_Position_Change",
    "LateRace",
    "Compound",
    "Driver",
    "Race",
]

monotone_positive = {
    "TyreLife_ExpectedLife_Ratio",
    "Wear_Pressure",
    "LapTime_Loss",
    "LapsRemaining_AfterExpectedLife",
}
monotone_constraints = [1 if c in monotone_positive else 0 for c in feature_cols]

dry_train_mask = train_fe["Compound"].astype(str).isin(DRY_COMPOUNDS)
dry_test_mask = test_fe["Compound"].astype(str).isin(DRY_COMPOUNDS)

X_dry = train_fe.loc[dry_train_mask, feature_cols]
y_dry = train_fe.loc[dry_train_mask, TARGET].astype(int).reset_index(drop=True)
dry_rows = train_fe.index[dry_train_mask].to_numpy()
groups_dry = train_fe.loc[dry_train_mask, "RaceGroup"].reset_index(drop=True)
X_dry = X_dry.reset_index(drop=True)

n_splits = min(5, groups_dry.nunique())
cv = GroupKFold(n_splits=n_splits)

base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.035,
    n_estimators=1400,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=3.0,
    monotone_constraints=monotone_constraints,
    random_state=958,
    n_jobs=max(1, os.cpu_count() or 1),
    verbosity=-1,
)

oof = np.zeros(len(X_dry), dtype=float)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_dry, y_dry, groups_dry), 1):
    model = LGBMClassifier(**base_params)
    model.fit(
        X_dry.iloc[tr_idx],
        y_dry.iloc[tr_idx],
        categorical_feature=cat_features,
        eval_set=[(X_dry.iloc[va_idx], y_dry.iloc[va_idx])],
        eval_metric="auc",
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )
    pred = model.predict_proba(X_dry.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y_dry.iloc[va_idx], pred)
    fold_scores.append(float(auc))
    best_iterations.append(int(model.best_iteration_ or base_params["n_estimators"]))
    print(f"fold_{fold}_dry_group_auc={auc:.6f}")

cv_auc = roc_auc_score(y_dry, oof)
print(f"dry_grouped_{n_splits}fold_roc_auc={cv_auc:.6f}")

final_params = dict(base_params)
final_params["n_estimators"] = int(np.median(best_iterations))
final_model = LGBMClassifier(**final_params)
final_model.fit(X_dry, y_dry, categorical_feature=cat_features)

global_rate = float(train_fe[TARGET].mean())
non_dry_rate = train_fe.loc[~dry_train_mask, TARGET].mean()
if not np.isfinite(non_dry_rate):
    non_dry_rate = global_rate

test_pred = np.full(len(test_fe), float(non_dry_rate), dtype=float)
if dry_test_mask.any():
    test_pred[dry_test_mask.to_numpy()] = final_model.predict_proba(
        test_fe.loc[dry_test_mask, feature_cols]
    )[:, 1]
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = sample[[ID]].copy()
submission[TARGET] = test_pred
submission.to_csv(WORK / "submission.csv", index=False)

test_predictions = sample[[ID]].copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    WORK / "test_predictions.csv.gz", index=False, compression="gzip"
)

oof_df = pd.DataFrame(
    {
        "row": dry_rows,
        "target": y_dry.to_numpy(),
        "prediction": oof,
    }
)
oof_df.to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

review = {
    "metric": f"dry_grouped_{n_splits}fold_roc_auc",
    "score": float(cv_auc),
    "fold_scores": fold_scores,
    "research_hypotheses_llm_claimed_used": ["000958"],
    "submission_path": str(WORK / "submission.csv"),
    "oof_path": str(WORK / "oof_predictions.csv.gz"),
    "test_predictions_path": str(WORK / "test_predictions.csv.gz"),
}
print(json.dumps(review, sort_keys=True))
