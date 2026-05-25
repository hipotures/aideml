import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("lightgbm is required for this script.") from e


INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42

os.makedirs(WORKING_DIR, exist_ok=True)


def add_features(df):
    df = df.copy()

    compound_life_scale = {
        "SOFT": 0.75,
        "MEDIUM": 1.00,
        "HARD": 1.25,
        "INTERMEDIATE": 0.90,
        "WET": 0.85,
    }
    scale = df["Compound"].map(compound_life_scale).fillna(1.0).astype(float)

    tyre_life = df["TyreLife"].astype(float).clip(lower=1.0)
    race_progress = df["RaceProgress"].astype(float).clip(0.0, 1.0)
    lap_delta = df["LapTime_Delta"].astype(float)
    cum_deg = df["Cumulative_Degradation"].astype(float)

    df["compound_scaled_tyre_age"] = tyre_life / scale
    df["positive_laptime_delta"] = np.maximum(lap_delta, 0.0)
    df["degradation_per_tyre_life"] = cum_deg / tyre_life
    df["finish_pressure"] = tyre_life * race_progress / scale

    df["age_x_positive_laptime_delta"] = (
        df["compound_scaled_tyre_age"] * df["positive_laptime_delta"]
    )
    df["age_x_cumulative_degradation"] = df["compound_scaled_tyre_age"] * cum_deg
    df["age_x_deg_per_life"] = (
        df["compound_scaled_tyre_age"] * df["degradation_per_tyre_life"]
    )
    df["finish_pressure_x_positive_delta"] = (
        df["finish_pressure"] * df["positive_laptime_delta"]
    )
    df["finish_pressure_x_deg_per_life"] = (
        df["finish_pressure"] * df["degradation_per_tyre_life"]
    )

    return df.replace([np.inf, -np.inf], np.nan)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values

train_fe = add_features(train.drop(columns=[TARGET]))
test_fe = add_features(test)

features = [c for c in train_fe.columns if c != ID_COL]
cat_cols = [
    c
    for c in features
    if train_fe[c].dtype == "object" or c in ["Year", "Stint", "PitStop"]
]

combined = pd.concat([train_fe[features], test_fe[features]], axis=0, ignore_index=True)
for c in cat_cols:
    combined[c] = combined[c].astype("category")

X = combined.iloc[: len(train_fe)].reset_index(drop=True)
X_test = combined.iloc[len(train_fe) :].reset_index(drop=True)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.035,
    "num_leaves": 48,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "n_estimators": 2500,
    "random_state": RANDOM_STATE,
    "n_jobs": max(1, os.cpu_count() or 1),
    "verbosity": -1,
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(X), dtype=float)
test_pred = np.zeros(len(X_test), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    te_pred = model.predict_proba(X_test)[:, 1]

    oof[va_idx] = va_pred
    test_pred += te_pred / skf.n_splits

    fold_auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(fold_auc)
    print(f"fold {fold} roc_auc: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"mean_fold_roc_auc: {np.mean(fold_scores):.6f}")
print(f"oof_roc_auc: {cv_auc:.6f}")

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

submission = pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: test_pred,
    }
)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "oof_roc_auc": float(cv_auc),
            "mean_fold_roc_auc": float(np.mean(fold_scores)),
            "fold_roc_auc": [float(x) for x in fold_scores],
            "research_hypotheses_llm_claimed_used": ["000031"],
        }
    )
)
