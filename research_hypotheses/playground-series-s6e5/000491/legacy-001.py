import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 491
N_SPLITS = 5

os.makedirs(WORKING_DIR, exist_ok=True)


def meta(street, high_deg, low_warmup_overcut, overtake_diff, pit_low, pit_high, wet):
    return {
        "meta_street_temporary": street,
        "meta_high_degradation": high_deg,
        "meta_low_warmup_overcut": low_warmup_overcut,
        "meta_overtaking_difficulty": overtake_diff,
        "meta_pit_loss_low_s": pit_low,
        "meta_pit_loss_high_s": pit_high,
        "meta_wet_prone": wet,
    }


DEFAULT_META = meta(0.0, 0.45, 0.40, 0.50, 20.0, 23.0, 0.25)

RACE_META = {
    "Bahrain Grand Prix": meta(0.0, 0.85, 0.20, 0.35, 22.0, 24.0, 0.05),
    "Saudi Arabian Grand Prix": meta(1.0, 0.35, 0.25, 0.55, 19.0, 21.0, 0.02),
    "Australian Grand Prix": meta(1.0, 0.45, 0.30, 0.55, 20.0, 22.0, 0.20),
    "Emilia Romagna Grand Prix": meta(0.0, 0.35, 0.45, 0.75, 26.0, 29.0, 0.35),
    "Miami Grand Prix": meta(1.0, 0.55, 0.35, 0.55, 20.0, 22.0, 0.35),
    "Spanish Grand Prix": meta(0.0, 0.85, 0.35, 0.65, 22.0, 24.0, 0.15),
    "Monaco Grand Prix": meta(1.0, 0.20, 0.60, 0.98, 19.0, 21.0, 0.30),
    "Azerbaijan Grand Prix": meta(1.0, 0.35, 0.25, 0.45, 20.0, 22.0, 0.20),
    "Canadian Grand Prix": meta(0.7, 0.40, 0.35, 0.45, 18.0, 20.0, 0.45),
    "British Grand Prix": meta(0.0, 0.65, 0.40, 0.35, 21.0, 23.0, 0.55),
    "Austrian Grand Prix": meta(0.0, 0.55, 0.20, 0.35, 19.0, 21.0, 0.35),
    "Styrian Grand Prix": meta(0.0, 0.55, 0.20, 0.35, 19.0, 21.0, 0.35),
    "French Grand Prix": meta(0.0, 0.55, 0.40, 0.45, 22.0, 24.0, 0.20),
    "Hungarian Grand Prix": meta(0.0, 0.45, 0.55, 0.80, 20.0, 22.0, 0.35),
    "Belgian Grand Prix": meta(0.0, 0.40, 0.40, 0.30, 20.0, 22.0, 0.65),
    "Dutch Grand Prix": meta(0.0, 0.45, 0.50, 0.80, 21.0, 23.0, 0.50),
    "Italian Grand Prix": meta(0.0, 0.25, 0.25, 0.25, 23.0, 25.0, 0.25),
    "Singapore Grand Prix": meta(1.0, 0.85, 0.65, 0.90, 27.0, 30.0, 0.65),
    "Japanese Grand Prix": meta(0.0, 0.65, 0.45, 0.45, 20.0, 22.0, 0.55),
    "Qatar Grand Prix": meta(0.0, 0.95, 0.30, 0.55, 24.0, 26.0, 0.02),
    "United States Grand Prix": meta(0.0, 0.65, 0.35, 0.45, 20.0, 22.0, 0.20),
    "Mexico City Grand Prix": meta(0.0, 0.35, 0.65, 0.45, 20.0, 22.0, 0.25),
    "Sao Paulo Grand Prix": meta(0.0, 0.55, 0.40, 0.30, 20.0, 22.0, 0.55),
    "Las Vegas Grand Prix": meta(1.0, 0.20, 0.95, 0.35, 20.0, 22.0, 0.05),
    "Abu Dhabi Grand Prix": meta(0.0, 0.45, 0.35, 0.45, 21.0, 23.0, 0.05),
    "Chinese Grand Prix": meta(0.0, 0.70, 0.40, 0.40, 21.0, 23.0, 0.30),
    "Portuguese Grand Prix": meta(0.0, 0.55, 0.45, 0.45, 20.0, 22.0, 0.25),
    "Turkish Grand Prix": meta(0.0, 0.60, 0.45, 0.50, 20.0, 22.0, 0.35),
    "Russian Grand Prix": meta(0.4, 0.30, 0.35, 0.55, 21.0, 23.0, 0.25),
    "Pre-Season Testing": DEFAULT_META,
}


def nearby_laptime_density(s, radius=1.0):
    arr = s.to_numpy(dtype=np.float64, copy=False)
    order = np.argsort(arr)
    sorted_arr = arr[order]
    left = np.searchsorted(sorted_arr, sorted_arr - radius, side="left")
    right = np.searchsorted(sorted_arr, sorted_arr + radius, side="right")
    counts = (right - left - 1).astype(np.float32)
    out = np.empty(len(arr), dtype=np.float32)
    out[order] = counts
    return pd.Series(out, index=s.index)


def build_features(train_df, test_df):
    n_train = len(train_df)
    full = pd.concat(
        [train_df.drop(columns=[TARGET]), test_df],
        axis=0,
        ignore_index=True,
    )

    for col in ["Race", "Driver", "Compound"]:
        full[col] = full[col].astype(str).fillna("Unknown")

    full["RaceYear"] = full["Race"] + "_" + full["Year"].astype(str)
    full["RaceCompound"] = full["Race"] + "_" + full["Compound"]

    meta_df = pd.DataFrame([RACE_META.get(r, DEFAULT_META) for r in full["Race"]])
    full = pd.concat([full, meta_df], axis=1)
    full["meta_pit_loss_mid_s"] = (
        full["meta_pit_loss_low_s"] + full["meta_pit_loss_high_s"]
    ) / 2.0
    full["meta_pit_loss_width_s"] = (
        full["meta_pit_loss_high_s"] - full["meta_pit_loss_low_s"]
    )

    group_keys = ["RaceYear", "LapNumber"]
    full["LapsTotal_Est"] = (
        full.groupby("RaceYear", sort=False)["LapNumber"]
        .transform("max")
        .astype(np.float32)
    )
    full["LapsRemaining_Est"] = (
        (full["LapsTotal_Est"] - full["LapNumber"]).clip(lower=0).astype(np.float32)
    )
    full["LapFraction_Est"] = full["LapNumber"] / full["LapsTotal_Est"].replace(
        0, np.nan
    )
    full["TyreLifeRaceFrac"] = full["TyreLife"] / full["LapsTotal_Est"].replace(
        0, np.nan
    )
    full["TyreLifeOfRemaining"] = full["TyreLife"] / (
        full["TyreLife"] + full["LapsRemaining_Est"] + 1.0
    )

    gb = full.groupby(group_keys, sort=False)
    full["FieldSize"] = gb["Position"].transform("count").astype(np.float32)
    full["PositionPctInField"] = (full["Position"] - 1) / (
        full["FieldSize"] - 1
    ).replace(0, np.nan)
    field_mean = gb["LapTime (s)"].transform("mean")
    field_std = gb["LapTime (s)"].transform("std").fillna(0.0)
    full["LapTimeVsFieldMean"] = full["LapTime (s)"] - field_mean
    full["LapTimeFieldZ"] = full["LapTimeVsFieldMean"] / (field_std + 1e-3)
    full["FieldLapTimeStd"] = field_std
    full["FieldDensity_1s"] = (
        gb["LapTime (s)"].transform(nearby_laptime_density).astype(np.float32)
    )

    ordered = full[[*group_keys, "Position", "LapTime (s)", "TyreLife"]].sort_values(
        group_keys + ["Position"]
    )
    ogb = ordered.groupby(group_keys, sort=False)
    ordered["LapTimeAhead"] = ogb["LapTime (s)"].shift(1)
    ordered["LapTimeBehind"] = ogb["LapTime (s)"].shift(-1)
    ordered["TyreLifeAhead"] = ogb["TyreLife"].shift(1)
    ordered["TyreLifeBehind"] = ogb["TyreLife"].shift(-1)
    ordered["LapTimeGapAhead"] = ordered["LapTime (s)"] - ordered["LapTimeAhead"]
    ordered["LapTimeGapBehind"] = ordered["LapTimeBehind"] - ordered["LapTime (s)"]
    ordered["TyreLifeDeltaAhead"] = ordered["TyreLife"] - ordered["TyreLifeAhead"]
    ordered["TyreLifeDeltaBehind"] = ordered["TyreLife"] - ordered["TyreLifeBehind"]
    gap_cols = [
        "LapTimeGapAhead",
        "LapTimeGapBehind",
        "TyreLifeDeltaAhead",
        "TyreLifeDeltaBehind",
    ]
    full[gap_cols] = ordered[gap_cols].sort_index()

    for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        full[f"Compound_{comp}"] = (full["Compound"] == comp).astype(np.float32)

    full["TyreLife_x_high_deg"] = full["TyreLife"] * full["meta_high_degradation"]
    full["TyreLife_x_low_warmup"] = full["TyreLife"] * full["meta_low_warmup_overcut"]
    full["TyreLife_x_overtake_diff"] = (
        full["TyreLife"] * full["meta_overtaking_difficulty"]
    )
    full["TyreLife_x_pit_loss"] = full["TyreLife"] * full["meta_pit_loss_mid_s"]
    full["LapsRemaining_x_high_deg"] = (
        full["LapsRemaining_Est"] * full["meta_high_degradation"]
    )
    full["LapsRemaining_x_low_warmup"] = (
        full["LapsRemaining_Est"] * full["meta_low_warmup_overcut"]
    )
    full["LapsRemaining_x_pit_loss"] = (
        full["LapsRemaining_Est"] * full["meta_pit_loss_mid_s"]
    )
    full["Position_x_overtake_diff"] = (
        full["Position"] * full["meta_overtaking_difficulty"]
    )
    full["PositionPct_x_overtake_diff"] = (
        full["PositionPctInField"] * full["meta_overtaking_difficulty"]
    )
    full["PositionPct_x_pit_loss"] = (
        full["PositionPctInField"] * full["meta_pit_loss_mid_s"]
    )
    full["FieldDensity_x_overtake_diff"] = (
        full["FieldDensity_1s"] * full["meta_overtaking_difficulty"]
    )
    full["FieldDensity_x_street"] = (
        full["FieldDensity_1s"] * full["meta_street_temporary"]
    )
    full["LapTimeFieldZ_x_overtake_diff"] = (
        full["LapTimeFieldZ"] * full["meta_overtaking_difficulty"]
    )
    full["HighDeg_x_SOFT"] = full["meta_high_degradation"] * full["Compound_SOFT"]
    full["HighDeg_x_MEDIUM"] = full["meta_high_degradation"] * full["Compound_MEDIUM"]
    full["WarmupOvercut_x_HARD"] = (
        full["meta_low_warmup_overcut"] * full["Compound_HARD"]
    )
    full["WarmupOvercut_x_MEDIUM"] = (
        full["meta_low_warmup_overcut"] * full["Compound_MEDIUM"]
    )
    full["WetProne_x_INTERMEDIATE"] = (
        full["meta_wet_prone"] * full["Compound_INTERMEDIATE"]
    )
    full["WetProne_x_WET"] = full["meta_wet_prone"] * full["Compound_WET"]

    full = full.drop(columns=[ID_COL])
    cat_cols = ["Race", "Driver", "Compound", "RaceYear", "RaceCompound"]

    for col in full.columns:
        if col not in cat_cols:
            full[col] = pd.to_numeric(full[col], errors="coerce")

    train_x = full.iloc[:n_train].reset_index(drop=True)
    test_x = full.iloc[n_train:].reset_index(drop=True)

    for col in cat_cols:
        cats = pd.Index(
            pd.concat([train_x[col], test_x[col]], ignore_index=True)
            .astype(str)
            .unique()
        )
        train_x[col] = pd.Categorical(train_x[col].astype(str), categories=cats)
        test_x[col] = pd.Categorical(test_x[col].astype(str), categories=cats)

    num_cols = [c for c in train_x.columns if c not in cat_cols]
    train_x[num_cols] = (
        train_x[num_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    )
    test_x[num_cols] = (
        test_x[num_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    )

    return train_x, test_x, cat_cols


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
X, X_test, categorical_cols = build_features(train, test)
groups = X["RaceYear"].astype(str).to_numpy()

try:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X, y, groups))
except Exception:
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(X, y, groups))

oof = np.zeros(len(X), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)
fold_scores = []
n_jobs = min(8, os.cpu_count() or 1)

for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    pos = max(float(y_tr.sum()), 1.0)
    neg = float(len(y_tr) - y_tr.sum())

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=2000,
        learning_rate=0.035,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=5.0,
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE + fold,
        n_jobs=n_jobs,
        verbosity=-1,
        force_col_wise=True,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    va_pred = model.predict_proba(X_va)[:, 1]
    oof[va_idx] = va_pred.astype(np.float32)
    test_pred += (model.predict_proba(X_test)[:, 1] / len(splits)).astype(np.float32)

    fold_auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(fold_auc)
    print(f"fold {fold} ROC AUC: {fold_auc:.6f}")

oof_auc = roc_auc_score(y, oof)
test_pred = np.clip(test_pred, 0.0, 1.0)
oof = np.clip(oof, 0.0, 1.0)

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int32),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(f"OOF ROC AUC: {oof_auc:.6f}")
print(
    json.dumps(
        {
            "validation_metric": "roc_auc",
            "oof_roc_auc": float(oof_auc),
            "fold_auc_mean": float(np.mean(fold_scores)),
            "fold_auc_std": float(np.std(fold_scores)),
            "research_hypotheses_llm_claimed_used": ["000491"],
        }
    )
)
