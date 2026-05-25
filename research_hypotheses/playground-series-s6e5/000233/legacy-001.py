import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values


def add_features(df, reference=None):
    out = df.copy()

    # Estimate event length from available laps per race/year; test-only pairs fall back safely.
    if reference is None:
        reference = out
    race_len = (
        reference.groupby(["Year", "Race"])["LapNumber"]
        .max()
        .rename("EstimatedRaceLaps")
    )
    out = out.merge(race_len, on=["Year", "Race"], how="left")
    out["EstimatedRaceLaps"] = out["EstimatedRaceLaps"].fillna(
        reference["LapNumber"].max()
    )

    out["LapsRemaining"] = (out["EstimatedRaceLaps"] - out["LapNumber"]).clip(lower=0)
    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["Degradation_per_TyreLife"] = out["Cumulative_Degradation"] / (
        out["TyreLife"] + 1.0
    )
    out["LapTimeDelta_abs"] = out["LapTime_Delta"].abs()
    out["PositionLoss"] = np.maximum(out["Position_Change"], 0)
    out["PositionGain"] = np.maximum(-out["Position_Change"], 0)
    out["LateStint"] = (out["TyreLife"] >= 10).astype(int)
    out["VeryOldTyre"] = (out["TyreLife"] >= 18).astype(int)
    return out


full_ref = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
train_fe = add_features(train.drop(columns=[TARGET]), full_ref)
test_fe = add_features(test, full_ref)


def candidate_window(df):
    enough_laps_left = df["LapsRemaining"] >= 1
    not_final_lap = df["RaceProgress"] <= 0.995
    after_opening = df["LapNumber"] >= 2
    mature_stint = df["TyreLife"] >= 3
    progress_band = df["RaceProgress"].between(0.03, 0.99)
    strategic_signal = (
        (df["TyreLife"] >= 6)
        | (df["Stint"] >= 2)
        | (df["Cumulative_Degradation"] > 0)
        | (df["LapTime_Delta"] > 0)
        | (df["PitStop"] == 1)
    )
    return (
        enough_laps_left
        & not_final_lap
        & after_opening
        & mature_stint
        & progress_band
        & strategic_signal
    ).astype(bool)


train_candidate = candidate_window(train_fe)
test_candidate = candidate_window(test_fe)

positive_recall = y[train_candidate.values].sum() / max(1, y.sum())
candidate_share = train_candidate.mean()
print(f"Stage-1 candidate positive recall: {positive_recall:.6f}")
print(f"Stage-1 candidate row share: {candidate_share:.6f}")

features = [c for c in train_fe.columns if c != ID_COL]
cat_cols = [c for c in features if train_fe[c].dtype == "object"]

for c in cat_cols:
    combined = pd.concat([train_fe[c], test_fe[c]], axis=0).astype("category")
    cats = combined.cat.categories
    train_fe[c] = pd.Categorical(train_fe[c], categories=cats)
    test_fe[c] = pd.Categorical(test_fe[c], categories=cats)

X = train_fe[features]
X_test = test_fe[features]

global_rate = y.mean()
non_candidate_rate = (
    y[~train_candidate.values].mean()
    if (~train_candidate.values).any()
    else global_rate
)
candidate_rate = (
    y[train_candidate.values].mean() if train_candidate.values.any() else global_rate
)
fallback_non_candidate = max(1e-6, min(0.05, non_candidate_rate))
fallback_candidate = max(1e-6, candidate_rate)

oof = np.full(len(train), fallback_non_candidate, dtype=float)
test_pred_folds = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    tr_mask = train_candidate.iloc[tr_idx].values
    va_mask = train_candidate.iloc[va_idx].values

    model = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1,
    )

    X_tr = X.iloc[tr_idx[tr_mask]]
    y_tr = y[tr_idx[tr_mask]]

    if len(np.unique(y_tr)) < 2:
        oof[va_idx[va_mask]] = fallback_candidate
        fold_test = np.full(len(test), fallback_non_candidate, dtype=float)
        fold_test[test_candidate.values] = fallback_candidate
    else:
        model.fit(
            X_tr,
            y_tr,
            categorical_feature=cat_cols,
            eval_set=(
                [(X.iloc[va_idx[va_mask]], y[va_idx[va_mask]])]
                if va_mask.any()
                else None
            ),
            eval_metric="auc",
        )

        if va_mask.any():
            oof[va_idx[va_mask]] = model.predict_proba(X.iloc[va_idx[va_mask]])[:, 1]

        fold_test = np.full(len(test), fallback_non_candidate, dtype=float)
        if test_candidate.any():
            fold_test[test_candidate.values] = model.predict_proba(
                X_test.loc[test_candidate.values]
            )[:, 1]

    fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")
    test_pred_folds.append(fold_test)

cv_auc = roc_auc_score(y, oof)
test_pred = np.mean(test_pred_folds, axis=0)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame({"row": np.arange(len(train)), "target": y, "prediction": oof})
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = submission[[ID_COL, TARGET]].copy()
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(f"5-fold ROC AUC: {cv_auc:.6f}")
print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_score": float(cv_auc),
            "stage1_candidate_positive_recall": float(positive_recall),
            "stage1_candidate_row_share": float(candidate_share),
            "research_hypotheses_llm_claimed_used": ["000233"],
        }
    )
)
