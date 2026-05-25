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
RANDOM_STATE = 42
N_SPLITS = 5
BLEND_BASELINE_WEIGHT = 0.70

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values
test_ids = sample[ID_COL].values


def add_features(df):
    df = df.copy()

    expected_life_map = {
        "SOFT": 18.0,
        "MEDIUM": 26.0,
        "HARD": 34.0,
        "INTERMEDIATE": 22.0,
        "WET": 18.0,
    }
    expected = df["Compound"].map(expected_life_map).fillna(25.0)

    laps_remaining = (
        df["LapNumber"] / df["RaceProgress"].clip(0.01, 1.0) - df["LapNumber"]
    ).clip(lower=0)
    finish_window_start = np.maximum(0, expected - laps_remaining)

    df["TyreLife_ExpectedLife_Ratio"] = df["TyreLife"] / expected
    df["Compound_LapsBeyondFinishWindow"] = (df["TyreLife"] - finish_window_start).clip(
        lower=0
    )
    df["LapsRemaining_AfterExpectedLife"] = (
        df["TyreLife"] + laps_remaining - expected
    ).clip(lower=0)

    cooldown = np.where(df["PitStop"].values == 1, 1.0, 0.0)
    df["PitCooldown_Suppression"] = cooldown * np.exp(
        -df["TyreLife"].clip(lower=0) / 3.0
    )

    degradation_signal = df["Cumulative_Degradation"].rank(
        pct=True
    ).values + np.maximum(df["LapTime_Delta"].values, 0) / (
        np.nanstd(df["LapTime_Delta"].values) + 1e-6
    )
    readiness = (
        0.45 * df["TyreLife_ExpectedLife_Ratio"].values
        + 0.25 * df["Compound_LapsBeyondFinishWindow"].values / 10.0
        + 0.20 * df["LapsRemaining_AfterExpectedLife"].values / 20.0
        + 0.10 * degradation_signal
    )
    df["PitReadiness_CooldownAdjusted"] = (
        readiness - df["PitCooldown_Suppression"].values
    )

    df["VeryShortRemainingLaps"] = np.maximum(0, 4.0 - laps_remaining)
    df["LapsRemaining"] = laps_remaining
    df["LapTimeDelta_Positive"] = np.maximum(df["LapTime_Delta"], 0)
    df["TyreLife_x_RaceProgress"] = df["TyreLife"] * df["RaceProgress"]
    df["Stint_x_TyreLife"] = df["Stint"] * df["TyreLife"]

    return df


all_df = pd.concat(
    [train.drop(columns=[TARGET]), test],
    axis=0,
    ignore_index=True,
)
all_df = add_features(all_df)

cat_cols = [c for c in all_df.columns if all_df[c].dtype == "object"]
for c in cat_cols:
    all_df[c] = all_df[c].astype("category")

X = all_df.iloc[: len(train)].drop(columns=[ID_COL])
X_test = all_df.iloc[len(train) :].drop(columns=[ID_COL])

mono_cols = [
    "TyreLife_ExpectedLife_Ratio",
    "Compound_LapsBeyondFinishWindow",
    "LapsRemaining_AfterExpectedLife",
    "PitReadiness_CooldownAdjusted",
    "PitCooldown_Suppression",
    "VeryShortRemainingLaps",
]
mono_constraints = [1, 1, 1, 1, -1, -1]

X_mono = X[mono_cols]
X_test_mono = X_test[mono_cols]

baseline_oof = np.zeros(len(train))
mono_oof = np.zeros(len(train))
baseline_test = np.zeros(len(test))
mono_test = np.zeros(len(test))

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    baseline = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbose=-1,
    )
    baseline.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[],
    )

    mono = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=16,
        max_depth=5,
        min_child_samples=120,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=2.0,
        monotone_constraints=mono_constraints,
        random_state=RANDOM_STATE + 100 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    mono.fit(
        X_mono.iloc[tr_idx],
        y_tr,
        eval_set=[(X_mono.iloc[val_idx], y_val)],
        eval_metric="auc",
        callbacks=[],
    )

    baseline_oof[val_idx] = baseline.predict_proba(X_val)[:, 1]
    mono_oof[val_idx] = mono.predict_proba(X_mono.iloc[val_idx])[:, 1]
    baseline_test += baseline.predict_proba(X_test)[:, 1] / N_SPLITS
    mono_test += mono.predict_proba(X_test_mono)[:, 1] / N_SPLITS

    fold_pred = (
        BLEND_BASELINE_WEIGHT * baseline_oof[val_idx]
        + (1.0 - BLEND_BASELINE_WEIGHT) * mono_oof[val_idx]
    )
    print(f"fold {fold} roc_auc: {roc_auc_score(y_val, fold_pred):.6f}")

blend_oof = (
    BLEND_BASELINE_WEIGHT * baseline_oof + (1.0 - BLEND_BASELINE_WEIGHT) * mono_oof
)
blend_test = (
    BLEND_BASELINE_WEIGHT * baseline_test + (1.0 - BLEND_BASELINE_WEIGHT) * mono_test
)
blend_test = np.clip(blend_test, 0.0, 1.0)

baseline_auc = roc_auc_score(y, baseline_oof)
mono_auc = roc_auc_score(y, mono_oof)
blend_auc = roc_auc_score(y, blend_oof)

submission = sample.copy()
submission[TARGET] = blend_test
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": blend_oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: blend_test,
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(blend_auc),
    "baseline_cv_roc_auc": float(baseline_auc),
    "monotonic_expert_cv_roc_auc": float(mono_auc),
    "research_hypotheses_llm_claimed_used": ["000598"],
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
print(json.dumps(result, indent=2))
