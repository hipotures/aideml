import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

ID_COL = "id"
TARGET = "PitNextLap"
RANDOM_STATE = 42
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
n_train = len(train)

cap_source = train.copy()
cap_map = cap_source.groupby("Compound")["TyreLife"].quantile(0.90).to_dict()
cap_map = {k: max(5.0, float(v)) for k, v in cap_map.items()}
global_cap = max(5.0, float(cap_source["TyreLife"].quantile(0.90)))
fresh_dry_cap = max([cap_map.get(c, global_cap) for c in DRY_COMPOUNDS] + [global_cap])

all_df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)


def cumulative_rule_state(df):
    group_cols = [c for c in ["Year", "Race", "Driver"] if c in df.columns]
    sort_cols = group_cols + ["LapNumber", ID_COL]
    s = df.sort_values(sort_cols, kind="mergesort")
    keys = (
        [s[c].astype(str).to_numpy() for c in group_cols]
        if group_cols
        else np.zeros(len(s), dtype=np.int8)
    )

    pit = pd.to_numeric(s["PitStop"], errors="coerce").fillna(0).astype(np.int16)
    stops = pit.groupby(keys, sort=False).cumsum() if group_cols else pit.cumsum()

    dry_used = pd.Series(0, index=s.index, dtype=np.int16)
    for comp in DRY_COMPOUNDS:
        seen_now = s["Compound"].eq(comp).astype(np.int8)
        seen = (
            seen_now.groupby(keys, sort=False).cummax()
            if group_cols
            else seen_now.cummax()
        )
        dry_used += seen.astype(np.int16)

    wet_now = s["Compound"].isin(WET_COMPOUNDS).astype(np.int8)
    wet_seen = (
        wet_now.groupby(keys, sort=False).cummax() if group_cols else wet_now.cummax()
    )

    return (
        stops.reindex(df.index).fillna(0).astype(np.int16),
        dry_used.reindex(df.index).fillna(0).astype(np.int16),
        wet_seen.reindex(df.index).fillna(0).astype(np.int8),
    )


def add_hypothesis_000023_features(df):
    df = df.copy()

    stops_so_far, dry_used_so_far, wet_seen_so_far = cumulative_rule_state(df)
    df["StopsSoFar"] = stops_so_far
    df["DryCompoundsUsedSoFar"] = dry_used_so_far
    df["WetOrInterSeenSoFar"] = wet_seen_so_far

    lap = pd.to_numeric(df["LapNumber"], errors="coerce").astype("float32")
    progress = pd.to_numeric(df["RaceProgress"], errors="coerce").astype("float32")
    invalid_progress = (~np.isfinite(progress)) | (progress <= 0) | (progress > 1.25)

    progress_safe = progress.mask(invalid_progress).clip(0.01, 1.0)
    if {"Year", "Race"}.issubset(df.columns):
        fallback_total = (
            df.groupby(["Year", "Race"])["LapNumber"].transform("max").astype("float32")
        )
    else:
        fallback_total = pd.Series(
            float(np.nanmax(lap)), index=df.index, dtype="float32"
        )

    est_total_laps = (lap / progress_safe).fillna(fallback_total)
    est_total_laps = np.maximum(est_total_laps, lap).clip(upper=110.0)
    laps_remaining = (est_total_laps - lap).clip(lower=0.0, upper=100.0)
    denom = laps_remaining.clip(lower=1.0, upper=80.0) + 0.5

    df["EstimatedTotalLaps"] = est_total_laps.astype("float32")
    df["EstimatedLapsRemaining"] = laps_remaining.astype("float32")
    df["DebtUrgencyDenominator"] = denom.astype("float32")
    df["InvalidRaceProgressFlag"] = invalid_progress.astype(np.int8)
    df["EarlyProgressEstimateFlag"] = (progress_safe.fillna(0) < 0.05).astype(np.int8)

    tyre_life = (
        pd.to_numeric(df["TyreLife"], errors="coerce").fillna(0).astype("float32")
    )
    current_cap = df["Compound"].map(cap_map).fillna(global_cap).astype("float32")
    remaining_after_next_stop = (laps_remaining - 1.0).clip(lower=0.0)

    df["CurrentTyreCanFinish"] = ((tyre_life + laps_remaining) <= current_cap).astype(
        np.int8
    )
    df["NextStopCanFinish"] = (remaining_after_next_stop <= fresh_dry_cap).astype(
        np.int8
    )
    df["CurrentTyreFinishMargin"] = (current_cap - tyre_life - laps_remaining).astype(
        "float32"
    )
    df["NextStopFinishMargin"] = (fresh_dry_cap - remaining_after_next_stop).astype(
        "float32"
    )

    dry_now = df["Compound"].isin(DRY_COMPOUNDS)
    has_future_laps = laps_remaining > 1.0

    stop_debt = ((df["StopsSoFar"] == 0) & has_future_laps).astype(np.int8)
    compound_debt = (
        dry_now
        & (df["WetOrInterSeenSoFar"] == 0)
        & (df["DryCompoundsUsedSoFar"] < 2)
        & has_future_laps
    ).astype(np.int8)

    df["MandatoryStopDebt"] = stop_debt
    df["DryCompoundDebt"] = compound_debt
    df["FirstStopCompoundDebt"] = (
        (compound_debt == 1) & (df["StopsSoFar"] == 0)
    ).astype(np.int8)
    df["CompoundSwitchDebt"] = ((compound_debt == 1) & (df["StopsSoFar"] > 0)).astype(
        np.int8
    )
    df["AnyRuleDebt"] = np.maximum(stop_debt, compound_debt).astype(np.int8)
    df["RuleDebtCount"] = (stop_debt + compound_debt).astype(np.int8)

    debt_cols = [
        "MandatoryStopDebt",
        "DryCompoundDebt",
        "FirstStopCompoundDebt",
        "CompoundSwitchDebt",
        "AnyRuleDebt",
        "RuleDebtCount",
    ]

    for col in debt_cols:
        debt = df[col].astype("float32")
        urgency = (debt / denom).astype("float32")
        df[f"{col}_UrgencyPerLap"] = urgency
        for k in (2, 3, 5, 8):
            df[f"{col}_Few{k}LapsLeftWithDebt"] = (
                (debt > 0) & (laps_remaining <= k)
            ).astype(np.int8)
        df[f"{col}_Urgency_x_CurrentCanFinish"] = (
            urgency * df["CurrentTyreCanFinish"]
        ).astype("float32")
        df[f"{col}_Urgency_x_CurrentCannotFinish"] = (
            urgency * (1 - df["CurrentTyreCanFinish"])
        ).astype("float32")
        df[f"{col}_Urgency_x_NextStopCanFinish"] = (
            urgency * df["NextStopCanFinish"]
        ).astype("float32")
        df[f"{col}_Urgency_x_NextStopCannotFinish"] = (
            urgency * (1 - df["NextStopCanFinish"])
        ).astype("float32")

    for c in df.columns:
        if c == ID_COL:
            continue
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")

    return df


features = add_hypothesis_000023_features(all_df)
feature_cols = [c for c in features.columns if c != ID_COL]

for c in feature_cols:
    if features[c].dtype == "object":
        features[c] = features[c].astype("category")

X = features.iloc[:n_train][feature_cols].copy()
X_test = features.iloc[n_train:][feature_cols].copy()
cat_cols = [c for c in feature_cols if str(features[c].dtype) == "category"]

if {"Year", "Race", "Driver"}.issubset(train.columns):
    groups = (
        train["Year"].astype(str)
        + "|"
        + train["Race"].astype(str)
        + "|"
        + train["Driver"].astype(str)
    )
else:
    groups = np.arange(len(train))

pos = max(1, int(y.sum()))
neg = len(y) - pos
scale_pos_weight = float(np.clip(neg / pos, 1.0, 100.0))
n_jobs = max(1, min(8, os.cpu_count() or 1))

base_params = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_estimators=2500,
    learning_rate=0.035,
    num_leaves=64,
    min_child_samples=80,
    subsample=0.90,
    subsample_freq=1,
    colsample_bytree=0.90,
    reg_alpha=0.05,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=n_jobs,
    verbosity=-1,
    force_col_wise=True,
)

if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    split_iter = splitter.split(X, y, groups=groups)
    cv_name = "StratifiedGroupKFold"
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    split_iter = splitter.split(X, y)
    cv_name = "StratifiedKFold"

oof = np.zeros(len(train), dtype=np.float32)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
    model = lgb.LGBMClassifier(**base_params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iter = getattr(model, "best_iteration_", None) or base_params["n_estimators"]
    best_iterations.append(int(best_iter))
    pred = model.predict_proba(X.iloc[va_idx], num_iteration=best_iter)[:, 1]
    oof[va_idx] = pred.astype(np.float32)
    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))
    print(f"Fold {fold} ROC AUC: {auc:.6f} best_iter={best_iter}")

cv_auc = float(roc_auc_score(y, oof))
final_trees = int(
    np.clip(round(np.mean(best_iterations)), 100, base_params["n_estimators"])
)

final_params = dict(base_params)
final_params["n_estimators"] = final_trees
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

target_col = (
    TARGET
    if TARGET in sample.columns
    else [c for c in sample.columns if c != ID_COL][0]
)
submission = sample.copy()
submission[target_col] = test_pred
submission[[ID_COL, target_col]].to_csv(
    os.path.join(WORK_DIR, "submission.csv"), index=False
)
submission[[ID_COL, target_col]].to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int32),
        "target": y.astype(np.int8),
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_strategy": cv_name,
    "cv_roc_auc": cv_auc,
    "fold_roc_auc": fold_scores,
    "final_model_trees": final_trees,
    "research_hypotheses_llm_claimed_used": ["000023"],
}

with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"5-fold {cv_name} ROC AUC: {cv_auc:.6f}")
print(json.dumps(result))
