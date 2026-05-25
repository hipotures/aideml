import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold

    SPLITTER_KIND = "StratifiedGroupKFold"
except ImportError:
    from sklearn.model_selection import GroupKFold

    StratifiedGroupKFold = None
    SPLITTER_KIND = "GroupKFold"

import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
SEED = 2026
N_SPLITS = 5
N_THREADS = min(16, os.cpu_count() or 1)

os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

n_train = len(train)
train["_is_train"] = 1
test["_is_train"] = 0
test[TARGET] = np.nan

all_df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
all_df["_orig_pos"] = np.arange(len(all_df))


def add_rule_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_compound_upper"] = df["Compound"].astype(str).str.upper()
    race_text = df["Race"].astype(str)

    dry_specs = ["HARD", "MEDIUM", "SOFT"]
    wet_specs = ["INTERMEDIATE", "WET"]
    all_specs = dry_specs + wet_specs

    df["is_race_session"] = (
        ~race_text.str.contains("Testing", case=False, na=False)
    ).astype(np.int8)
    df["is_monaco_2025"] = (
        (df["Year"].astype(int) == 2025)
        & race_text.str.contains("Monaco", case=False, na=False)
        & (df["is_race_session"] == 1)
    ).astype(np.int8)

    df["current_is_dry"] = df["_compound_upper"].isin(dry_specs).astype(np.int8)
    df["current_is_wet_or_inter"] = (
        df["_compound_upper"].isin(wet_specs).astype(np.int8)
    )

    for spec in all_specs:
        df[f"current_{spec.lower()}"] = (df["_compound_upper"] == spec).astype(np.int8)

    sort_cols = ["Year", "Race", "Driver", "LapNumber", "id", "_orig_pos"]
    sdf = df.sort_values(sort_cols, kind="mergesort").copy()
    group_cols = ["Year", "Race", "Driver"]
    grp = sdf.groupby(group_cols, sort=False)

    for spec in all_specs:
        cur_col = f"current_{spec.lower()}"
        seen_col = f"seen_{spec.lower()}_so_far"
        prior_col = f"prior_seen_{spec.lower()}"
        sdf[seen_col] = grp[cur_col].cummax().astype(np.int8)
        sdf[prior_col] = grp[seen_col].shift(1).fillna(0).astype(np.int8)

    dry_seen_cols = [f"seen_{s.lower()}_so_far" for s in dry_specs]
    wet_seen_cols = [f"seen_{s.lower()}_so_far" for s in wet_specs]
    prior_dry_cols = [f"prior_seen_{s.lower()}" for s in dry_specs]
    prior_wet_cols = [f"prior_seen_{s.lower()}" for s in wet_specs]

    sdf["dry_compounds_used_so_far"] = sdf[dry_seen_cols].sum(axis=1).astype(np.int8)
    sdf["prior_dry_compounds_used"] = sdf[prior_dry_cols].sum(axis=1).astype(np.int8)
    sdf["prior_compounds_used"] = (
        sdf[prior_dry_cols + prior_wet_cols].sum(axis=1)
    ).astype(np.int8)

    sdf["wet_rule_exempt"] = (
        (sdf["is_race_session"] == 1) & (sdf[wet_seen_cols].sum(axis=1) > 0)
    ).astype(np.int8)
    sdf["prior_wet_rule_exempt"] = (
        (sdf["is_race_session"] == 1) & (sdf[prior_wet_cols].sum(axis=1) > 0)
    ).astype(np.int8)

    sdf["dry_rule_prior_unsatisfied"] = (
        (sdf["is_race_session"] == 1)
        & (sdf["prior_wet_rule_exempt"] == 0)
        & (sdf["prior_dry_compounds_used"] < 2)
    ).astype(np.int8)
    sdf["dry_rule_still_unsatisfied"] = (
        (sdf["is_race_session"] == 1)
        & (sdf["wet_rule_exempt"] == 0)
        & (sdf["dry_compounds_used_so_far"] < 2)
    ).astype(np.int8)

    new_dry_spec = np.zeros(len(sdf), dtype=np.int8)
    for spec in dry_specs:
        new_dry_spec |= (
            (sdf[f"current_{spec.lower()}"].values == 1)
            & (sdf[f"prior_seen_{spec.lower()}"].values == 0)
        ).astype(np.int8)
    sdf["new_dry_spec_this_lap"] = new_dry_spec
    sdf["dry_rule_satisfied_this_lap"] = (
        (sdf["dry_rule_prior_unsatisfied"] == 1)
        & (sdf["dry_rule_still_unsatisfied"] == 0)
        & (sdf["new_dry_spec_this_lap"] == 1)
    ).astype(np.int8)

    completed_stops = np.maximum(sdf["Stint"].astype(float).fillna(1).values - 1, 0)
    monaco_remaining = np.maximum(2 - completed_stops, 0) * sdf["is_monaco_2025"].values
    dry_remaining = sdf["dry_rule_still_unsatisfied"].values.astype(float)

    sdf["monaco_remaining_required_stops_min"] = monaco_remaining.astype(np.float32)
    sdf["monaco_2025_forced_extra_stop"] = (
        (sdf["is_monaco_2025"].values == 1) & (monaco_remaining > dry_remaining)
    ).astype(np.int8)
    sdf["remaining_required_stops_min"] = np.maximum(
        dry_remaining, monaco_remaining
    ).astype(np.float32)

    progress = sdf["RaceProgress"].astype(float).clip(0, 1).values
    late = np.clip((progress - 0.65) / 0.35, 0, 1)
    endgame = np.clip((progress - 0.82) / 0.18, 0, 1)
    req = sdf["remaining_required_stops_min"].values

    sdf["rule_pressure"] = req * progress
    sdf["late_rule_pressure"] = req * late
    sdf["endgame_rule_pressure"] = req * endgame
    sdf["dry_unsatisfied_progress"] = (
        sdf["dry_rule_still_unsatisfied"].values * progress
    )
    sdf["monaco_forced_progress"] = (
        sdf["monaco_2025_forced_extra_stop"].values * progress
    )
    sdf["wet_exempt_progress"] = sdf["wet_rule_exempt"].values * progress
    sdf["current_dry_rule_pressure"] = (
        sdf["current_is_dry"].values * sdf["rule_pressure"].values
    )
    sdf["current_wet_rule_pressure"] = (
        sdf["current_is_wet_or_inter"].values * sdf["rule_pressure"].values
    )
    sdf["stint_remaining_required"] = sdf["Stint"].astype(float).values * req
    sdf["tyrelife_remaining_required"] = sdf["TyreLife"].astype(float).values * req
    sdf["prior_dry_count_progress"] = sdf["prior_dry_compounds_used"].values * progress
    sdf["dry_count_progress"] = sdf["dry_compounds_used_so_far"].values * progress

    sdf = sdf.sort_index()
    return sdf.drop(columns=["_compound_upper"])


all_df = add_rule_regime_features(all_df)

group_keys_all = (
    all_df["Year"].astype(str)
    + "_"
    + all_df["Race"].astype(str)
    + "_"
    + all_df["Driver"].astype(str)
)

cat_cols = ["Compound", "Driver", "Race"]
for col in cat_cols:
    all_df[col] = all_df[col].astype(str).fillna("missing")
    all_df[col] = pd.Categorical(all_df[col], categories=pd.Index(all_df[col].unique()))

drop_cols = {TARGET, "id", "_is_train", "_orig_pos"}
feature_cols = [c for c in all_df.columns if c not in drop_cols]

train_df = all_df.iloc[:n_train].copy()
test_df = all_df.iloc[n_train:].copy()

X = train_df[feature_cols]
y = train_df[TARGET].astype(int).values
X_test = test_df[feature_cols]
groups = group_keys_all.iloc[:n_train].values

if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y, groups))
else:
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(X, y, groups))

pos = y.sum()
neg = len(y) - pos
scale_pos_weight = neg / max(pos, 1)

params = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_estimators=2500,
    learning_rate=0.035,
    num_leaves=63,
    min_child_samples=90,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=1.5,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    n_jobs=N_THREADS,
    verbosity=-1,
)

oof = np.zeros(len(train_df), dtype=np.float32)
test_pred = np.zeros(len(test_df), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    val_pred = model.predict_proba(X.iloc[va_idx], num_iteration=model.best_iteration_)[
        :, 1
    ]
    oof[va_idx] = val_pred.astype(np.float32)
    fold_auc = roc_auc_score(y[va_idx], val_pred)
    fold_scores.append(fold_auc)

    test_pred += (
        model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1].astype(
            np.float32
        )
        / N_SPLITS
    )

    print(
        f"Fold {fold} ROC AUC: {fold_auc:.6f} | best_iteration={model.best_iteration_}"
    )

oof_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC ({SPLITTER_KIND}, {N_SPLITS} folds): {oof_auc:.6f}")
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")

oof_path = os.path.join(WORKING_DIR, "oof_predictions.csv.gz")
pd.DataFrame(
    {
        "row": np.arange(len(train_df)),
        "target": y,
        "prediction": oof,
    }
).to_csv(oof_path, index=False, compression="gzip")

test_pred_df = pd.DataFrame({"id": test["id"].values, TARGET: np.clip(test_pred, 0, 1)})
submission = sample[["id"]].merge(test_pred_df, on="id", how="left")
submission[TARGET] = submission[TARGET].fillna(float(np.nanmean(test_pred)))

submission_path = os.path.join(WORKING_DIR, "submission.csv")
test_pred_path = os.path.join(WORKING_DIR, "test_predictions.csv.gz")
submission.to_csv(submission_path, index=False)
submission.to_csv(test_pred_path, index=False, compression="gzip")

review = {
    "metric": "roc_auc",
    "oof_roc_auc": float(oof_auc),
    "mean_fold_roc_auc": float(np.mean(fold_scores)),
    "std_fold_roc_auc": float(np.std(fold_scores)),
    "cv": SPLITTER_KIND,
    "n_splits": N_SPLITS,
    "submission_path": submission_path,
    "oof_predictions_path": oof_path,
    "test_predictions_path": test_pred_path,
    "research_hypotheses_llm_claimed_used": ["000850"],
}
print(json.dumps(review, sort_keys=True))
