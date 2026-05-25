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
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

# Hypothesis 000238: curated circuit-energy/severity prior by race.
# 0=low, 1=medium, 2=high, 3=very high tyre energy/degradation.
TRACK_ENERGY = {
    "Monaco Grand Prix": 0,
    "Las Vegas Grand Prix": 0,
    "Italian Grand Prix": 0,
    "Azerbaijan Grand Prix": 0,
    "Canadian Grand Prix": 1,
    "Saudi Arabian Grand Prix": 1,
    "Miami Grand Prix": 1,
    "Australian Grand Prix": 1,
    "Bahrain Grand Prix": 2,
    "Spanish Grand Prix": 2,
    "Emilia Romagna Grand Prix": 2,
    "Dutch Grand Prix": 2,
    "Belgian Grand Prix": 2,
    "Mexico City Grand Prix": 2,
    "Abu Dhabi Grand Prix": 2,
    "United States Grand Prix": 2,
    "Qatar Grand Prix": 3,
    "British Grand Prix": 3,
    "Hungarian Grand Prix": 3,
    "Japanese Grand Prix": 3,
    "Brazilian Grand Prix": 3,
    "Austrian Grand Prix": 3,
    "Singapore Grand Prix": 3,
    "Chinese Grand Prix": 2,
    "Pre-Season Testing": 1,
}

COMPOUND_BASE_LIFE = {
    "SOFT": 18.0,
    "MEDIUM": 28.0,
    "HARD": 38.0,
    "INTERMEDIATE": 22.0,
    "WET": 18.0,
}

ENERGY_LIFE_MULT = {
    0: 1.25,
    1: 1.08,
    2: 0.92,
    3: 0.76,
}

ENERGY_DEG_MULT = {
    0: 0.75,
    1: 0.95,
    2: 1.15,
    3: 1.40,
}


def add_features(df):
    out = df.copy()

    out["track_energy_bin"] = out["Race"].map(TRACK_ENERGY).fillna(1).astype("int8")
    out["compound_base_life"] = out["Compound"].map(COMPOUND_BASE_LIFE).fillna(26.0)
    out["energy_life_mult"] = (
        out["track_energy_bin"].map(ENERGY_LIFE_MULT).astype(float)
    )
    out["energy_deg_mult"] = out["track_energy_bin"].map(ENERGY_DEG_MULT).astype(float)

    max_lap_by_race = out.groupby(["Year", "Race"])["LapNumber"].transform("max")
    progress_est_total = out["LapNumber"] / out["RaceProgress"].clip(0.01, 1.0)
    out["est_total_laps"] = np.maximum(max_lap_by_race, progress_est_total).clip(20, 90)
    out["laps_remaining_est"] = (out["est_total_laps"] - out["LapNumber"]).clip(0, 90)

    out["severity_expected_life"] = out["compound_base_life"] * out["energy_life_mult"]
    out["severity_excess_wear"] = out["TyreLife"] - out["severity_expected_life"]
    out["severity_wear_ratio"] = out["TyreLife"] / (
        out["severity_expected_life"] + 1e-6
    )
    out["severity_adjusted_degradation"] = (
        out["Cumulative_Degradation"] * out["energy_deg_mult"]
    )
    out["degradation_per_tyre_lap"] = out["Cumulative_Degradation"] / out[
        "TyreLife"
    ].clip(lower=1)
    out["severity_degradation_per_lap"] = out["severity_adjusted_degradation"] / out[
        "TyreLife"
    ].clip(lower=1)

    out["late_race_stop_margin"] = (
        out["laps_remaining_est"] - out["severity_expected_life"]
    )
    out["severity_late_stop_margin"] = out["late_race_stop_margin"] * (
        1.0 + 0.18 * out["track_energy_bin"]
    )
    out["life_remaining_expected"] = out["severity_expected_life"] - out["TyreLife"]
    out["pit_window_pressure"] = out["severity_wear_ratio"] + out["RaceProgress"] * (
        1.0 + 0.25 * out["track_energy_bin"]
    )
    out["energy_x_tyre_life"] = out["track_energy_bin"] * out["TyreLife"]
    out["energy_x_deg"] = out["track_energy_bin"] * out["Cumulative_Degradation"]
    out["energy_x_laps_remaining"] = out["track_energy_bin"] * out["laps_remaining_est"]

    out["is_current_pitstop"] = out["PitStop"].astype(int)
    out["stint_progress"] = out["TyreLife"] / (
        out["TyreLife"] + out["laps_remaining_est"] + 1e-6
    )

    for col in ["Compound", "Driver", "Race"]:
        out[col] = out[col].astype("category")

    return out


train_fe = add_features(train)
test_fe = add_features(test)

y = train_fe[TARGET].astype(int)
drop_cols = [ID_COL, TARGET]
features = [c for c in train_fe.columns if c not in drop_cols]

X = train_fe[features]
X_test = test_fe[features]
cat_features = [c for c in ["Compound", "Driver", "Race"] if c in features]

oof = np.zeros(len(train_fe), dtype=float)
test_pred = np.zeros(len(test_fe), dtype=float)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1800,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[],
    )

    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / N_SPLITS

    fold_auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold CV ROC AUC: {cv_auc:.6f}")

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y.values,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample[[ID_COL]].copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "research_hypotheses_llm_claimed_used": ["000238"],
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(result, f, indent=2)
