import os
import json
import re
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP = True
except Exception:
    HAS_STRATIFIED_GROUP = False

import lightgbm as lgb

SEED = 2026
INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)


def build_features(df):
    df = df.copy()
    eps = 1e-6

    race_norm = (
        df["Race"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9]+", " ", regex=True)
        .str.strip()
    )
    df["RaceNorm"] = race_norm
    df["YearCat"] = df["Year"].astype(str)
    df["RaceYear"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["DriverYear"] = df["Driver"].astype(str) + "_" + df["Year"].astype(str)
    df["CompoundStint"] = df["Compound"].astype(str) + "_" + df["Stint"].astype(str)

    df["is_wet_weather_compound"] = (
        df["Compound"].isin(["WET", "INTERMEDIATE"]).astype(np.int8)
    )
    df["is_monaco"] = race_norm.str.contains("monaco", na=False).astype(np.int8)
    df["is_early_race"] = (df["RaceProgress"] < 0.25).astype(np.int8)
    df["is_mid_race"] = (
        (df["RaceProgress"] >= 0.25) & (df["RaceProgress"] < 0.75)
    ).astype(np.int8)
    df["is_late_race"] = (df["RaceProgress"] >= 0.75).astype(np.int8)

    race_progress = df["RaceProgress"].clip(lower=0.01)
    df["est_total_laps"] = (df["LapNumber"] / race_progress).clip(1, 120)
    df["laps_remaining_est"] = (df["est_total_laps"] - df["LapNumber"]).clip(0, 120)
    df["tyre_life_frac"] = df["TyreLife"] / (df["est_total_laps"] + eps)
    df["tyre_life_to_lap"] = df["TyreLife"] / (df["LapNumber"].clip(lower=1) + eps)
    df["degradation_per_tyre_lap"] = df["Cumulative_Degradation"] / (
        df["TyreLife"].clip(lower=1) + eps
    )
    df["abs_laptime_delta"] = df["LapTime_Delta"].abs()
    df["position_change_abs"] = df["Position_Change"].abs()
    df["stint_progress_interaction"] = df["Stint"] * df["RaceProgress"]
    df["tyre_life_x_progress"] = df["TyreLife"] * df["RaceProgress"]

    sort_cols = ["Year", "Race", "Driver", "LapNumber", "id"]
    sdf = df.sort_values(sort_cols).copy()
    g = sdf.groupby(["Year", "Race", "Driver"], sort=False, observed=False)

    sdf["pit_prev_lap"] = g["PitStop"].shift(1).fillna(0)
    sdf["prior_pit_count"] = (g["PitStop"].cumsum() - sdf["PitStop"]).clip(lower=0)
    sdf["pit_count_so_far"] = g["PitStop"].cumsum()
    sdf["had_prior_stop"] = (sdf["prior_pit_count"] > 0).astype(np.int8)
    sdf["prev_laptime_delta"] = g["LapTime_Delta"].shift(1).fillna(0)
    sdf["prev_position_change"] = g["Position_Change"].shift(1).fillna(0)
    prev_compound = g["Compound"].shift(1)
    sdf["compound_changed_from_prev"] = (
        prev_compound.notna() & sdf["Compound"].ne(prev_compound)
    ).astype(np.int8)

    seq_cols = [
        "pit_prev_lap",
        "prior_pit_count",
        "pit_count_so_far",
        "had_prior_stop",
        "prev_laptime_delta",
        "prev_position_change",
        "compound_changed_from_prev",
    ]
    for col in seq_cols:
        df[col] = sdf[col].sort_index()

    ry = df.groupby(["Year", "Race"], observed=False)
    df["race_year_max_lap_seen"] = ry["LapNumber"].transform("max")
    df["race_year_laptime_median"] = ry["LapTime_s"].transform("median")
    df["race_year_degradation_median"] = ry["Cumulative_Degradation"].transform(
        "median"
    )
    df["lap_time_vs_race_median"] = df["LapTime_s"] - df["race_year_laptime_median"]
    df["degradation_vs_race_median"] = (
        df["Cumulative_Degradation"] - df["race_year_degradation_median"]
    )

    # Hypothesis 000022: conservative Monaco 2025 mandatory multi-stop proxy.
    affected = (
        (df["is_monaco"] == 1)
        & (df["Year"] == 2025)
        & (df["is_wet_weather_compound"] == 0)
    ).to_numpy()

    stop_debt = np.maximum(2 - df["pit_count_so_far"].fillna(0).to_numpy(), 0)
    set_debt = np.maximum(3 - df["Stint"].to_numpy(), 0)
    phase_pressure = np.clip((df["RaceProgress"].to_numpy() - 0.25) / 0.65, 0, 1)
    laps_remaining = df["laps_remaining_est"].to_numpy()
    finish_feasible = (laps_remaining >= stop_debt + 1) & (
        df["RaceProgress"].to_numpy() < 0.985
    )

    df["monaco_2025_two_stop_rule"] = affected.astype(np.int8)
    df["monaco_2025_stop_debt"] = np.where(affected, stop_debt, 0)
    df["monaco_2025_set_debt"] = np.where(affected, set_debt, 0)
    df["monaco_2025_debt_pressure"] = np.where(
        affected & finish_feasible,
        stop_debt * phase_pressure,
        0,
    )
    df["monaco_2025_last_chance"] = (
        affected
        & (stop_debt > 0)
        & (laps_remaining > 1)
        & (laps_remaining <= stop_debt * 9 + 2)
    ).astype(np.int8)
    df["monaco_2025_current_or_prior_stop"] = np.where(
        affected,
        df["PitStop"].to_numpy() + df["had_prior_stop"].to_numpy(),
        0,
    )

    cat_cols = [
        "Compound",
        "Driver",
        "Race",
        "RaceNorm",
        "YearCat",
        "RaceYear",
        "DriverYear",
        "CompoundStint",
    ]
    for col in cat_cols:
        df[col] = df[col].astype(str).astype("category")

    num_cols = [c for c in df.columns if c not in cat_cols and c != "id"]
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df, cat_cols


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz")).rename(
    columns={"LapTime (s)": "LapTime_s"}
)
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz")).rename(
    columns={"LapTime (s)": "LapTime_s"}
)
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train["PitNextLap"].astype(int).to_numpy()
train_base = train.drop(columns=["PitNextLap"])
full = pd.concat([train_base, test], axis=0, ignore_index=True)

full_features, categorical_features = build_features(full)
feature_cols = [c for c in full_features.columns if c != "id"]

X = full_features.iloc[: len(train)][feature_cols].copy()
X_test = full_features.iloc[len(train) :][feature_cols].copy()

groups = (
    train_base["Year"].astype(str)
    + "|"
    + train_base["Race"].astype(str)
    + "|"
    + train_base["Driver"].astype(str)
).to_numpy()

if HAS_STRATIFIED_GROUP:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y, groups))
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y))

oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []
n_jobs = max(1, min(16, os.cpu_count() or 1))

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    pos = max(1, int(y_tr.sum()))
    neg = max(1, len(y_tr) - pos)
    scale_pos_weight = neg / pos

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=1800,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED + fold,
        n_jobs=n_jobs,
        verbosity=-1,
        metric="auc",
        force_col_wise=True,
        deterministic=True,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    va_pred = model.predict_proba(X_va)[:, 1]
    oof[va_idx] = va_pred
    fold_auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

    test_pred += model.predict_proba(X_test)[:, 1] / len(splits)

cv_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {cv_auc:.6f}")
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")

test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)
oof = np.clip(oof, 1e-6, 1 - 1e-6)

submission = sample.copy()
submission["PitNextLap"] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

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

metadata = {
    "research_hypotheses_llm_claimed_used": ["000022"],
    "evaluation_metric": "roc_auc",
    "oof_roc_auc": float(cv_auc),
    "mean_fold_roc_auc": float(np.mean(fold_scores)),
    "fold_roc_auc": [float(v) for v in fold_scores],
}
with open(os.path.join(WORKING_DIR, "result_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
