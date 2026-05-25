import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
N_THREADS = min(8, os.cpu_count() or 1)
INPUT = Path("./input")
WORKING = Path("./working")
WORKING.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

TARGET = "PitNextLap"
ID_COL = "id"
SUB_TARGET = [c for c in sample.columns if c != ID_COL][0]
n_train = len(train)

y = train[TARGET].astype(int).to_numpy()
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

full = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
train_mask_all = np.zeros(len(full), dtype=bool)
train_mask_all[:n_train] = True

compound_life = {
    "SOFT": 16.0,
    "MEDIUM": 24.0,
    "HARD": 34.0,
    "INTERMEDIATE": 20.0,
    "WET": 18.0,
}
full["CompoundLifePrior"] = full["Compound"].map(compound_life).fillna(24.0)

eps = 1e-6
full["RaceTotalLapsEst"] = (
    (full["LapNumber"] / full["RaceProgress"].clip(lower=0.01))
    .replace([np.inf, -np.inf], np.nan)
    .clip(1, 100)
)
full["LapsRemaining"] = (full["RaceTotalLapsEst"] - full["LapNumber"]).clip(lower=0)
full["TyreAgeRatio"] = full["TyreLife"] / full["CompoundLifePrior"].clip(lower=1)
full["TyreLifePctRace"] = full["TyreLife"] / full["RaceTotalLapsEst"].clip(lower=1)
full["RemainingTyreCapacity"] = (full["CompoundLifePrior"] - full["TyreLife"]).clip(
    lower=1
)
full["FinishStress"] = full["LapsRemaining"] / full["RemainingTyreCapacity"]
full["CanFinishCurrentTyre"] = (
    full["RemainingTyreCapacity"] >= full["LapsRemaining"]
).astype("int8")
full["DegPerLap"] = full["Cumulative_Degradation"] / full["TyreLife"].clip(lower=1)
full["AbsLapTimeDelta"] = full["LapTime_Delta"].abs()
full["RaceProgress2"] = full["RaceProgress"] ** 2
full["LateRaceTyrePressure"] = full["RaceProgress"] * full["TyreAgeRatio"]

race_lap = full.groupby(["Year", "Race", "LapNumber"], sort=False)
full["FieldLapMedian"] = race_lap["LapTime (s)"].transform("median")
full["FieldLapStd"] = race_lap["LapTime (s)"].transform("std").fillna(0)
full["FieldLapCount"] = race_lap["LapTime (s)"].transform("count")
full["RelLapTime"] = full["LapTime (s)"] - full["FieldLapMedian"]
full["RelLapTimeZ"] = full["RelLapTime"] / full["FieldLapStd"].replace(0, np.nan)
full["FieldDegMedian"] = race_lap["Cumulative_Degradation"].transform("median")
full["RelDeg"] = full["Cumulative_Degradation"] - full["FieldDegMedian"]
full["FieldTyreLifeMedian"] = race_lap["TyreLife"].transform("median")
full["RelTyreLife"] = full["TyreLife"] - full["FieldTyreLifeMedian"]
full["PositionPct"] = (full["Position"] - 1) / (full["FieldLapCount"] - 1).clip(lower=1)

compound_lap = full.groupby(["Year", "Race", "LapNumber", "Compound"], sort=False)
full["CompoundLapMedian"] = compound_lap["LapTime (s)"].transform("median")
full["CompoundPaceVsField"] = full["CompoundLapMedian"] - full["FieldLapMedian"]

ordered = full.sort_values(["Year", "Race", "Driver", "LapNumber", ID_COL]).copy()
driver_lap = ordered.groupby(["Year", "Race", "Driver"], sort=False, group_keys=False)

ordered["PrevLapTimeDelta"] = driver_lap["LapTime_Delta"].shift(1)
ordered["PrevRelLapTimeZ"] = driver_lap["RelLapTimeZ"].shift(1)
ordered["PrevPosition"] = driver_lap["Position"].shift(1)
ordered["DriverDeltaMean3"] = driver_lap["LapTime_Delta"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).mean()
)
ordered["DriverRelPaceMean3"] = driver_lap["RelLapTimeZ"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).mean()
)
ordered["DriverDegMean3"] = driver_lap["DegPerLap"].transform(
    lambda s: s.shift(1).rolling(3, min_periods=1).mean()
)

next_lap_num = driver_lap["LapNumber"].shift(-1)
has_next_lap = next_lap_num.eq(ordered["LapNumber"] + 1)
for col in [
    "Position",
    "PitStop",
    "RelLapTimeZ",
    "DegPerLap",
    "FinishStress",
    "TyreLife",
]:
    ordered[f"Next_{col}"] = driver_lap[col].shift(-1).where(has_next_lap)

new_cols = [
    "PrevLapTimeDelta",
    "PrevRelLapTimeZ",
    "PrevPosition",
    "DriverDeltaMean3",
    "DriverRelPaceMean3",
    "DriverDegMean3",
    "Next_Position",
    "Next_PitStop",
    "Next_RelLapTimeZ",
    "Next_DegPerLap",
    "Next_FinishStress",
    "Next_TyreLife",
]
full.loc[ordered.index, new_cols] = ordered[new_cols]


def robust_z(values, ref_mask):
    arr = np.asarray(values, dtype=np.float64)
    ref = arr[ref_mask & np.isfinite(arr)]
    if len(ref) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    med = np.nanmedian(ref)
    q25, q75 = np.nanpercentile(ref, [25, 75])
    scale = max(q75 - q25, 1e-6)
    return np.clip((arr - med) / scale, -6, 6).astype(np.float32)


valid_aux = (
    train_mask_all
    & full["Next_Position"].notna().to_numpy()
    & full["Next_RelLapTimeZ"].notna().to_numpy()
)

next_rel_pace = np.clip(
    full["Next_RelLapTimeZ"].fillna(0).to_numpy(dtype=np.float64), -6, 6
)
next_pos_loss = np.clip(
    (
        full["Next_Position"].fillna(full["Position"]).to_numpy(dtype=np.float64)
        - full["Position"].to_numpy(dtype=np.float64)
    )
    / 5.0,
    -3,
    3,
)
next_deg_z = robust_z(full["Next_DegPerLap"], train_mask_all)
next_finish_z = robust_z(full["Next_FinishStress"], train_mask_all)
next_pit_loss = full["Next_PitStop"].fillna(0).to_numpy(dtype=np.float64)
tyre_reset_z = robust_z(full["TyreLife"] - full["Next_TyreLife"], train_mask_all)

strategy_value = (
    -0.45 * next_rel_pace
    - 0.20 * next_pos_loss
    - 0.15 * next_deg_z
    - 0.15 * next_finish_z
    - 0.10 * next_pit_loss
    + 0.25 * tyre_reset_z
).astype(np.float32)

categorical_cols = ["Compound", "Driver", "Race"]
for c in categorical_cols:
    full[c] = pd.Categorical(full[c].astype(str).fillna("__missing__")).codes.astype(
        "int32"
    )

feature_cols = [
    "Compound",
    "Driver",
    "Race",
    "Year",
    "LapNumber",
    "LapTime (s)",
    "LapTime_Delta",
    "PitStop",
    "Position",
    "Position_Change",
    "RaceProgress",
    "Stint",
    "TyreLife",
    "Cumulative_Degradation",
    "CompoundLifePrior",
    "RaceTotalLapsEst",
    "LapsRemaining",
    "TyreAgeRatio",
    "TyreLifePctRace",
    "RemainingTyreCapacity",
    "FinishStress",
    "CanFinishCurrentTyre",
    "DegPerLap",
    "AbsLapTimeDelta",
    "RaceProgress2",
    "LateRaceTyrePressure",
    "FieldLapMedian",
    "FieldLapStd",
    "FieldLapCount",
    "RelLapTime",
    "RelLapTimeZ",
    "FieldDegMedian",
    "RelDeg",
    "FieldTyreLifeMedian",
    "RelTyreLife",
    "PositionPct",
    "CompoundLapMedian",
    "CompoundPaceVsField",
    "PrevLapTimeDelta",
    "PrevRelLapTimeZ",
    "PrevPosition",
    "DriverDeltaMean3",
    "DriverRelPaceMean3",
    "DriverDegMean3",
]

for c in feature_cols:
    full[c] = pd.to_numeric(full[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if c not in categorical_cols:
        fill_value = full.loc[: n_train - 1, c].median()
        if not np.isfinite(fill_value):
            fill_value = 0.0
        full[c] = full[c].fillna(fill_value).astype("float32")

X_base_train = full.iloc[:n_train][feature_cols].copy()
X_base_test = full.iloc[n_train:][feature_cols].copy()


def make_folds():
    if StratifiedGroupKFold is not None:
        try:
            cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            return list(cv.split(X_base_train, y, groups))
        except Exception:
            pass
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(cv.split(X_base_train, y))


folds = make_folds()


def regressor(seed):
    return LGBMRegressor(
        objective="regression_l2",
        n_estimators=450,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=N_THREADS,
        verbosity=-1,
    )


aux_value_train = strategy_value[:n_train]
valid_aux_train = valid_aux[:n_train]

pit_mean = float(np.nanmean(aux_value_train[valid_aux_train & (y == 1)]))
wait_mean = float(np.nanmean(aux_value_train[valid_aux_train & (y == 0)]))
if not np.isfinite(pit_mean):
    pit_mean = float(np.nanmean(aux_value_train[valid_aux_train]))
if not np.isfinite(wait_mean):
    wait_mean = float(np.nanmean(aux_value_train[valid_aux_train]))

pit_oof = np.full(n_train, pit_mean, dtype=np.float32)
wait_oof = np.full(n_train, wait_mean, dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    for action, store, fallback in [(1, pit_oof, pit_mean), (0, wait_oof, wait_mean)]:
        mask = np.zeros(n_train, dtype=bool)
        mask[tr_idx] = True
        mask &= valid_aux_train & (y == action)
        if mask.sum() < 100 or np.nanstd(aux_value_train[mask]) < 1e-8:
            store[va_idx] = fallback
            continue
        m = regressor(SEED + 100 * fold + action)
        m.fit(
            X_base_train.loc[mask, feature_cols],
            aux_value_train[mask],
            categorical_feature=[c for c in categorical_cols if c in feature_cols],
        )
        store[va_idx] = m.predict(X_base_train.iloc[va_idx][feature_cols]).astype(
            np.float32
        )

pit_full_model = regressor(SEED + 901)
wait_full_model = regressor(SEED + 902)

pit_full_mask = valid_aux_train & (y == 1)
wait_full_mask = valid_aux_train & (y == 0)

if pit_full_mask.sum() >= 100:
    pit_full_model.fit(
        X_base_train.loc[pit_full_mask, feature_cols],
        aux_value_train[pit_full_mask],
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
    )
    pit_test = pit_full_model.predict(X_base_test[feature_cols]).astype(np.float32)
else:
    pit_test = np.full(len(test), pit_mean, dtype=np.float32)

if wait_full_mask.sum() >= 100:
    wait_full_model.fit(
        X_base_train.loc[wait_full_mask, feature_cols],
        aux_value_train[wait_full_mask],
        categorical_feature=[c for c in categorical_cols if c in feature_cols],
    )
    wait_test = wait_full_model.predict(X_base_test[feature_cols]).astype(np.float32)
else:
    wait_test = np.full(len(test), wait_mean, dtype=np.float32)

X_train = X_base_train.copy()
X_test = X_base_test.copy()

X_train["pit_now_value"] = pit_oof
X_train["wait_1_value"] = wait_oof
X_train["pit_wait_value_gap"] = pit_oof - wait_oof

X_test["pit_now_value"] = pit_test
X_test["wait_1_value"] = wait_test
X_test["pit_wait_value_gap"] = pit_test - wait_test

final_features = feature_cols + ["pit_now_value", "wait_1_value", "pit_wait_value_gap"]
cat_features = [c for c in categorical_cols if c in final_features]


def classifier(seed, n_estimators=1200, spw=None):
    if spw is None:
        spw = (len(y) - y.sum()) / max(y.sum(), 1)
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=64,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=N_THREADS,
        verbosity=-1,
    )


oof = np.zeros(n_train, dtype=np.float32)
fold_scores = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    spw = (len(tr_idx) - y[tr_idx].sum()) / max(y[tr_idx].sum(), 1)
    clf = classifier(SEED + fold, spw=spw)
    clf.fit(
        X_train.iloc[tr_idx][final_features],
        y[tr_idx],
        eval_set=[(X_train.iloc[va_idx][final_features], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )
    pred = clf.predict_proba(X_train.iloc[va_idx][final_features])[:, 1]
    oof[va_idx] = pred
    score = roc_auc_score(y[va_idx], pred)
    fold_scores.append(score)
    best_iters.append(getattr(clf, "best_iteration_", None) or clf.n_estimators)
    print(f"fold {fold} roc_auc={score:.6f}")

oof_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {oof_auc:.6f}")

final_n_estimators = int(np.median(best_iters)) if best_iters else 900
final_clf = classifier(SEED + 999, n_estimators=max(100, final_n_estimators))
final_clf.fit(
    X_train[final_features],
    y,
    categorical_feature=cat_features,
)

test_pred = final_clf.predict_proba(X_test[final_features])[:, 1]
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = sample.copy()
submission[SUB_TARGET] = test_pred
submission.to_csv(WORKING / "submission.csv", index=False)
submission.to_csv(WORKING / "test_predictions.csv.gz", index=False, compression="gzip")

oof_df = pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(WORKING / "oof_predictions.csv.gz", index=False, compression="gzip")

review = {
    "metric": "roc_auc",
    "oof_roc_auc": float(oof_auc),
    "fold_roc_auc": [float(s) for s in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000843"],
    "submission_path": str(WORKING / "submission.csv"),
}
print(json.dumps(review, indent=2))
