import os
import re
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
WET_COMPOUNDS = ("INTERMEDIATE", "WET")

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
train_x = train.drop(columns=[TARGET]).copy()
test_x = test.copy()

train_x["__dataset"] = "train"
train_x["__row"] = np.arange(len(train_x))
test_x["__dataset"] = "test"
test_x["__row"] = np.arange(len(test_x))
full = pd.concat([train_x, test_x], axis=0, ignore_index=True)


def add_rule_state_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    compound_upper = df["Compound"].astype(str).str.upper()
    race_text = df["Race"].astype(str)

    df["Rule_IsDryCompound"] = compound_upper.isin(DRY_COMPOUNDS).astype("int8")
    df["Rule_IsWetCompound"] = compound_upper.isin(WET_COMPOUNDS).astype("int8")
    df["Rule_IsMonaco2025"] = (
        (df["Year"].astype(int) == 2025)
        & race_text.str.contains("Monaco", case=False, na=False)
    ).astype("int8")

    lap = df["LapNumber"].astype(float)
    progress = df["RaceProgress"].astype(float).clip(lower=0.01)
    est_total_laps = np.maximum(lap, np.minimum(120.0, np.rint(lap / progress)))
    df["LapsRemaining_Est"] = np.maximum(0.0, est_total_laps - lap).astype("float32")

    sort_cols = ["Year", "Race", "Driver", "LapNumber", ID_COL]
    group_cols = ["Year", "Race", "Driver"]
    ordered = df.sort_values(sort_cols, kind="mergesort")
    g = ordered.groupby(group_cols, sort=False)

    pit_now = ordered["PitStop"].fillna(0).astype("int16")
    pit_cum = g["PitStop"].cumsum().astype("float32")
    pit_prior = pit_cum - pit_now.astype("float32")

    wet_now = ordered["Rule_IsWetCompound"].astype("int16")
    wet_cum = g["Rule_IsWetCompound"].cumsum().astype("int16")
    wet_seen = wet_cum.gt(0).astype("int8")
    wet_seen_prior = (wet_cum - wet_now).gt(0).astype("int8")

    ordered_compound = ordered["Compound"].astype(str).str.upper()
    distinct_dry = pd.Series(0, index=ordered.index, dtype="int16")
    distinct_dry_prior = pd.Series(0, index=ordered.index, dtype="int16")

    for comp in DRY_COMPOUNDS:
        flag = ordered_compound.eq(comp).astype("int16")
        cum = flag.groupby([ordered[c] for c in group_cols], sort=False).cumsum()
        seen = cum.gt(0).astype("int8")
        seen_prior = (cum - flag).gt(0).astype("int8")
        df.loc[ordered.index, f"Rule_SeenDry_{comp}"] = seen.to_numpy()
        df.loc[ordered.index, f"Rule_PriorSeenDry_{comp}"] = seen_prior.to_numpy()
        distinct_dry += seen.astype("int16")
        distinct_dry_prior += seen_prior.astype("int16")

    dry_pit_event = (
        ordered["PitStop"].fillna(0).astype(int).eq(1)
        & ordered["Rule_IsDryCompound"].astype(int).eq(1)
    ).astype("int16")
    dry_pit_cum = dry_pit_event.groupby(
        [ordered[c] for c in group_cols], sort=False
    ).cumsum()

    stint = ordered["Stint"].astype(float).fillna(1.0)
    sets_used = np.maximum(stint.to_numpy(), 1.0 + pit_cum.to_numpy())
    stops_so_far = np.maximum(pit_cum.to_numpy(), sets_used - 1.0)

    wet_seen_np = wet_seen.to_numpy()
    distinct_dry_np = distinct_dry.to_numpy(dtype=np.float32)

    std_dry_compound_debt = np.where(
        wet_seen_np == 1,
        0.0,
        np.maximum(0.0, 2.0 - distinct_dry_np),
    )
    std_remaining_stops = np.where(
        wet_seen_np == 1, 0.0, (distinct_dry_np < 2).astype(float)
    )

    monaco_stop_debt = np.maximum(0.0, 2.0 - stops_so_far)
    monaco_set_debt = np.maximum(0.0, 3.0 - sets_used)
    monaco_compound_debt = std_dry_compound_debt.copy()
    monaco_remaining_stops = np.maximum.reduce(
        [monaco_stop_debt, monaco_set_debt, monaco_compound_debt]
    )

    is_monaco = ordered["Rule_IsMonaco2025"].to_numpy(dtype=np.int8)
    remaining_mandatory = np.where(
        is_monaco == 1, monaco_remaining_stops, std_remaining_stops
    )

    laps_remaining = ordered["LapsRemaining_Est"].astype(float).to_numpy()
    inv_laps_remaining = 1.0 / (laps_remaining + 1.0)

    assign = {
        "Rule_PitStopsPrior": pit_prior.to_numpy(),
        "Rule_PitStopsSoFar": pit_cum.to_numpy(),
        "Rule_DryPitStopsSoFar": dry_pit_cum.to_numpy(dtype=np.float32),
        "Rule_WetExemptPrior": wet_seen_prior.to_numpy(dtype=np.float32),
        "Rule_WetExemptSeen": wet_seen.to_numpy(dtype=np.float32),
        "Rule_DistinctDryCompoundsPrior": distinct_dry_prior.to_numpy(dtype=np.float32),
        "Rule_DistinctDryCompoundsSoFar": distinct_dry.to_numpy(dtype=np.float32),
        "Rule_SetsUsedSoFar": sets_used,
        "Rule_StopsSoFar": stops_so_far,
        "Rule_DryStopDebt": std_dry_compound_debt,
        "Rule_MonacoStopDebt": monaco_stop_debt,
        "Rule_MonacoSetDebt": monaco_set_debt,
        "Rule_MonacoCompoundDebt": monaco_compound_debt,
        "Rule_RemainingMandatoryStops": remaining_mandatory,
        "Rule_RemainingMandatoryStopsPerLap": remaining_mandatory * inv_laps_remaining,
        "Rule_DryStopDebtPerLap": std_dry_compound_debt * inv_laps_remaining,
        "Rule_MonacoStopDebtPerLap": monaco_stop_debt * inv_laps_remaining,
        "Rule_MonacoSetDebtPerLap": monaco_set_debt * inv_laps_remaining,
        "Rule_MandatoryDeadlinePressure": (
            remaining_mandatory >= np.maximum(1.0, laps_remaining)
        ).astype(float),
        "Rule_NearMandatoryWindow": (
            (remaining_mandatory > 0) & (laps_remaining <= (remaining_mandatory + 3.0))
        ).astype(float),
    }

    for col, values in assign.items():
        df.loc[ordered.index, col] = np.asarray(values, dtype=np.float32)

    for comp in DRY_COMPOUNDS + WET_COMPOUNDS:
        flag = compound_upper.eq(comp).astype("float32")
        df[f"Rule_RemStops_x_Current_{comp}"] = (
            df["Rule_RemainingMandatoryStops"].astype("float32") * flag
        )
        df[f"Rule_LapsRemaining_x_Current_{comp}"] = (
            df["LapsRemaining_Est"].astype("float32") * flag
        )

    return df


full = add_rule_state_features(full)
full["RaceYear"] = full["Year"].astype(str) + "_" + full["Race"].astype(str)

cat_cols = ["Compound", "Driver", "Race", "RaceYear"]
for col in cat_cols:
    full[col] = full[col].astype("category")

drop_cols = {ID_COL, "__dataset", "__row"}
feature_cols = [c for c in full.columns if c not in drop_cols]

train_mask = full["__dataset"].eq("train")
X = full.loc[train_mask, feature_cols].reset_index(drop=True)
X_test = full.loc[~train_mask, feature_cols].reset_index(drop=True)
groups = full.loc[train_mask, "RaceYear"].astype(str).to_numpy()


def sanitize_columns(columns):
    seen = {}
    out = []
    for col in columns:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_")
        if not safe:
            safe = "feature"
        base = safe
        i = seen.get(base, 0)
        if i:
            safe = f"{base}_{i}"
        seen[base] = i + 1
        out.append(safe)
    return dict(zip(columns, out))


rename_map = sanitize_columns(feature_cols)
X = X.rename(columns=rename_map)
X_test = X_test.rename(columns=rename_map)
cat_cols = [rename_map[c] for c in cat_cols]

X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)


def make_model(n_estimators, scale_pos_weight):
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        scale_pos_weight=scale_pos_weight,
        random_state=20260524,
        n_jobs=-1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


cv = GroupKFold(n_splits=5)
oof = np.zeros(len(X), dtype=np.float32)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    y_tr, y_va = y[tr_idx], y[va_idx]
    pos = max(1, int(y_tr.sum()))
    neg = max(1, len(y_tr) - pos)
    model = make_model(n_estimators=1800, scale_pos_weight=neg / pos)

    model.fit(
        X.iloc[tr_idx],
        y_tr,
        eval_set=[(X.iloc[va_idx], y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(stopping_rounds=120, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = int(model.best_iteration_ or model.n_estimators)
    best_iterations.append(best_iter)
    pred = model.predict_proba(X.iloc[va_idx], num_iteration=best_iter)[:, 1]
    oof[va_idx] = pred.astype(np.float32)
    auc = roc_auc_score(y_va, pred)
    fold_scores.append(float(auc))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(y), dtype=np.int32),
        "target": y.astype(np.int8),
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_n_estimators = int(np.clip(round(np.mean(best_iterations)), 100, 1800))
pos = max(1, int(y.sum()))
neg = max(1, len(y) - pos)
final_model = make_model(n_estimators=final_n_estimators, scale_pos_weight=neg / pos)
final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_oof_roc_auc": float(cv_auc),
            "fold_roc_auc": fold_scores,
            "final_model_iterations": final_n_estimators,
            "research_hypotheses_llm_claimed_used": ["000379"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        }
    )
)
