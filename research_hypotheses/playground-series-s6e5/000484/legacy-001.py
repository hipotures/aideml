import os
import re
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

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

RANDOM_STATE = 484
INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

target_col = "PitNextLap"
y = train[target_col].astype(int).reset_index(drop=True)

train_x = train.drop(columns=[target_col]).copy()
test_x = test.copy()
train_x["_is_train"] = 1
test_x["_is_train"] = 0
all_df = pd.concat([train_x, test_x], axis=0, ignore_index=True)


def sanitize_columns(df):
    seen = {}
    new_cols = []
    for col in df.columns:
        name = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
        if not name:
            name = "feature"
        if name[0].isdigit():
            name = "f_" + name
        base = name
        k = seen.get(base, 0)
        if k:
            name = f"{base}_{k}"
        seen[base] = k + 1
        new_cols.append(name)
    df = df.copy()
    df.columns = new_cols
    return df


def add_stint_geometry_features(df):
    df = df.copy()

    for col in ["Race", "Driver", "Compound"]:
        df[col] = df[col].astype("string").fillna("missing")

    lap = pd.to_numeric(df["LapNumber"], errors="coerce").astype("float32")
    tyre_life = pd.to_numeric(df["TyreLife"], errors="coerce").astype("float32")
    stint = pd.to_numeric(df["Stint"], errors="coerce").fillna(1).astype("float32")
    progress = (
        pd.to_numeric(df["RaceProgress"], errors="coerce")
        .astype("float32")
        .clip(lower=0.001)
    )

    raw_total = (lap / progress).replace([np.inf, -np.inf], np.nan)
    raw_total = raw_total.clip(lower=lap, upper=130)
    df["RawEstimatedTotalLaps"] = raw_total.astype("float32")

    group_est = df.groupby(["Year", "Race"], observed=True)[
        "RawEstimatedTotalLaps"
    ].transform("median")
    race_est = df.groupby("Race", observed=True)["RawEstimatedTotalLaps"].transform(
        "median"
    )
    overall_est = float(np.nanmedian(raw_total))
    est = group_est.fillna(race_est).fillna(overall_est).to_numpy(dtype=np.float32)
    est = np.maximum(est, lap.to_numpy(dtype=np.float32))
    est = np.clip(np.rint(est), 1, 130).astype(np.float32)
    est_s = pd.Series(est, index=df.index)

    comp = df["Compound"].astype(str).str.upper()
    race = df["Race"].astype(str)

    is_wet = comp.isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    is_preseason = race.str.contains("Pre-Season", case=False, na=False).astype(np.int8)
    dry_rule = ((is_wet == 0) & (is_preseason == 0)).astype(np.int8)

    stint_start = (lap - tyre_life + 1).clip(lower=1)
    laps_remaining = (est_s - lap).clip(lower=0)
    projected_stint_len = (est_s - stint_start + 1).clip(lower=1)

    df["EstimatedTotalLaps"] = est_s.astype("float32")
    df["StintStartLap"] = stint_start.astype("float32")
    df["StintProgressWithinRace"] = (tyre_life / est_s.clip(lower=1)).astype("float32")
    df["ProjectedCurrentStintLengthNoStop"] = projected_stint_len.astype("float32")
    df["ProjectedCurrentStintFractionNoStop"] = (
        projected_stint_len / est_s.clip(lower=1)
    ).astype("float32")
    df["ProjectedCurrentStintRemainingNoStop"] = (
        (projected_stint_len - tyre_life).clip(lower=0).astype("float32")
    )
    df["StintStartRaceFraction"] = (stint_start / est_s.clip(lower=1)).astype("float32")
    df["LapsRemaining"] = laps_remaining.astype("float32")
    df["LapNext"] = (lap + 1).astype("float32")
    df["LapToRaceEndRatio"] = (lap / est_s.clip(lower=1)).astype("float32")
    df["Is_WetWeather"] = is_wet
    df["Is_PreSeasonTesting"] = is_preseason
    df["DryRuleApplies"] = dry_rule
    df["RemainingMandatoryStopDebt"] = (
        (dry_rule == 1) & (stint <= 1) & (lap < est_s)
    ).astype(np.int8)
    df["RemainingSpecDebt"] = (
        (dry_rule == 1) & (stint <= 1) & comp.isin(["SOFT", "MEDIUM", "HARD"])
    ).astype(np.int8)
    df["RemainingRuleDebt"] = (
        df["RemainingMandatoryStopDebt"] + df["RemainingSpecDebt"]
    ).astype(np.int8)

    n_rows = len(df)
    lap_next = lap + 1
    stint_arr = np.rint(stint.to_numpy(dtype=np.float32)).astype(int)
    compounds = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

    for n_stops in [1, 2, 3]:
        signed_dist_names = []
        for stop_idx in range(1, n_stops + 1):
            ideal = est_s * stop_idx / (n_stops + 1)
            signed = lap_next - ideal
            prefix = f"T{n_stops}_Stop{stop_idx}"
            df[f"{prefix}_IdealLap"] = ideal.astype("float32")
            df[f"{prefix}_SignedDist"] = signed.astype("float32")
            df[f"{prefix}_AbsDist"] = signed.abs().astype("float32")
            signed_dist_names.append(f"{prefix}_SignedDist")

        dist_mat = df[signed_dist_names].to_numpy(dtype=np.float32)
        abs_mat = np.abs(dist_mat)
        nearest_idx = abs_mat.argmin(axis=1)
        current_idx = np.minimum(np.maximum(stint_arr, 1), n_stops) - 1

        nearest_signed = dist_mat[np.arange(n_rows), nearest_idx]
        nearest_abs = np.abs(nearest_signed)
        current_signed = dist_mat[np.arange(n_rows), current_idx]
        current_abs = np.abs(current_signed)
        valid_current = (stint.to_numpy(dtype=np.float32) <= n_stops).astype(np.int8)

        df[f"T{n_stops}_NearestSignedDist"] = nearest_signed.astype("float32")
        df[f"T{n_stops}_NearestAbsDist"] = nearest_abs.astype("float32")
        df[f"T{n_stops}_CurrentSignedDist"] = current_signed.astype("float32")
        df[f"T{n_stops}_CurrentAbsDist"] = current_abs.astype("float32")
        df[f"T{n_stops}_CurrentTemplateValid"] = valid_current
        df[f"T{n_stops}_DryTemplateValid"] = (valid_current * dry_rule).astype(np.int8)
        df[f"T{n_stops}_CurrentAbsDistDryGated"] = np.where(
            df[f"T{n_stops}_DryTemplateValid"] == 1, current_abs, 99.0
        ).astype("float32")
        df[f"T{n_stops}_NearestAbsDistDryGated"] = np.where(
            dry_rule == 1, nearest_abs, 99.0
        ).astype("float32")
        df[f"T{n_stops}_CurrentWindowScore"] = (
            1.0 / (1.0 + df[f"T{n_stops}_CurrentAbsDistDryGated"])
        ).astype("float32")
        df[f"T{n_stops}_NearestWindowScore"] = (
            1.0 / (1.0 + df[f"T{n_stops}_NearestAbsDistDryGated"])
        ).astype("float32")

        for dist_name in [
            f"T{n_stops}_CurrentSignedDist",
            f"T{n_stops}_CurrentAbsDist",
            f"T{n_stops}_NearestSignedDist",
            f"T{n_stops}_NearestAbsDist",
        ]:
            base = df[dist_name].astype("float32")
            df[f"{dist_name}_x_Stint"] = (base * stint).astype("float32")
            df[f"{dist_name}_x_WetWeather"] = (base * is_wet).astype("float32")
            df[f"{dist_name}_x_MandatoryDebt"] = (
                base * df["RemainingMandatoryStopDebt"]
            ).astype("float32")
            df[f"{dist_name}_x_SpecDebt"] = (base * df["RemainingSpecDebt"]).astype(
                "float32"
            )
            for c in compounds:
                df[f"{dist_name}_x_{c}"] = (base * comp.eq(c).astype(np.int8)).astype(
                    "float32"
                )

    for col in ["Race", "Driver", "Compound", "Year", "Stint"]:
        df[col] = df[col].astype("category")

    for col in df.columns:
        if str(df[col].dtype) == "category":
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")
        elif pd.api.types.is_integer_dtype(df[col]) and col != "id":
            df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


all_features = sanitize_columns(add_stint_geometry_features(all_df))
drop_cols = ["id", "_is_train"]
feature_cols = [c for c in all_features.columns if c not in drop_cols]

X = all_features.iloc[: len(train)][feature_cols].reset_index(drop=True)
X_test = all_features.iloc[len(train) :][feature_cols].reset_index(drop=True)
cat_features = [c for c in feature_cols if str(X[c].dtype) == "category"]

groups = (
    train["Year"].astype(str).fillna("NA")
    + "_"
    + train["Race"].astype(str).fillna("NA")
).reset_index(drop=True)

if StratifiedGroupKFold is not None and groups.nunique() >= 5:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y, groups))
    if any(y.iloc[val_idx].nunique() < 2 for _, val_idx in splits):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X, y))
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))

pos = int(y.sum())
neg = int(len(y) - pos)
scale_pos_weight = max(1.0, neg / max(pos, 1))

base_params = dict(
    objective="binary",
    metric="auc",
    n_estimators=2200,
    learning_rate=0.035,
    num_leaves=80,
    min_child_samples=80,
    subsample=0.86,
    colsample_bytree=0.86,
    reg_alpha=0.15,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=min(8, os.cpu_count() or 1),
    verbosity=-1,
)

oof = np.zeros(len(X), dtype=np.float32)
fold_scores = []
best_iters = []

for fold, (tr_idx, val_idx) in enumerate(splits, 1):
    model = LGBMClassifier(**base_params)
    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )

    val_pred = model.predict_proba(X.iloc[val_idx])[:, 1]
    oof[val_idx] = val_pred.astype(np.float32)
    score = roc_auc_score(y.iloc[val_idx], val_pred)
    fold_scores.append(float(score))
    best_iters.append(int(model.best_iteration_ or base_params["n_estimators"]))
    print(f"fold {fold} roc_auc: {score:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"cv roc_auc: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int32),
        "target": y.astype(int),
        "prediction": oof,
    }
).to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

final_estimators = int(
    np.clip(round(np.mean(best_iters) * 1.08), 100, base_params["n_estimators"])
)
final_params = dict(base_params)
final_params["n_estimators"] = final_estimators
final_model = LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_features)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission["PitNextLap"] = test_pred
submission.to_csv(WORK_DIR / "submission.csv", index=False)
submission.to_csv(WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip")

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
    "final_model_n_estimators": final_estimators,
    "research_hypotheses_llm_claimed_used": ["000484"],
}
print(json.dumps(result, sort_keys=True))
