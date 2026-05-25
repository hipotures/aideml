import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)


def safe_column_names(columns):
    used = {}
    mapping = {}
    for c in columns:
        s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_")
        if not s:
            s = "feature"
        if s[0].isdigit():
            s = "f_" + s
        base = s
        i = 1
        while s in used:
            i += 1
            s = f"{base}_{i}"
        used[s] = True
        mapping[c] = s
    return mapping


def add_monaco_2025_rule_state(df):
    df = df.copy()
    df["_row_order"] = np.arange(len(df))

    group_cols = ["Year", "Race", "Driver"]
    sort_cols = group_cols + ["LapNumber", "id", "_row_order"]
    s = df.sort_values(sort_cols, kind="mergesort").copy()

    compound_upper = s["Compound"].astype(str).str.upper()
    s["_is_wet_or_inter"] = compound_upper.isin(["WET", "INTERMEDIATE"]).astype(np.int8)
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        s[f"_is_{comp.lower()}"] = (compound_upper == comp).astype(np.int8)

    g = s.groupby(group_cols, sort=False)

    s["used_wet_or_inter"] = g["_is_wet_or_inter"].cummax().astype(np.int8)
    s["dry_rule_active"] = (1 - s["used_wet_or_inter"]).astype(np.int8)

    slick_used_cols = []
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        col = f"used_{comp.lower()}_so_far"
        s[col] = g[f"_is_{comp.lower()}"].cummax().astype(np.int8)
        slick_used_cols.append(col)

    s["slick_compounds_used"] = s[slick_used_cols].sum(axis=1).astype(np.int8)
    s["sets_used"] = g["Stint"].cummax().fillna(1).clip(lower=1).astype(np.int16)
    s["stops_completed"] = g["PitStop"].cumsum().fillna(0).astype(np.int16)
    s["stops_completed"] = np.maximum(s["stops_completed"], s["sets_used"] - 1).astype(
        np.int16
    )

    race_upper = s["Race"].astype(str).str.upper()
    s["monaco_2025_rule"] = (
        race_upper.str.contains("MONACO", na=False) & (s["Year"] >= 2025)
    ).astype(np.int8)

    ordinary_remaining = np.maximum.reduce(
        [
            np.zeros(len(s), dtype=np.int16),
            1 - s["stops_completed"].to_numpy(),
            2 - s["slick_compounds_used"].to_numpy(),
        ]
    )
    monaco_remaining = np.maximum.reduce(
        [
            np.zeros(len(s), dtype=np.int16),
            2 - s["stops_completed"].to_numpy(),
            3 - s["sets_used"].to_numpy(),
            2 - s["slick_compounds_used"].to_numpy(),
        ]
    )

    ordinary_remaining = np.where(
        s["dry_rule_active"].to_numpy() == 1, ordinary_remaining, 0
    )
    monaco_remaining = np.where(
        s["dry_rule_active"].to_numpy() == 1, monaco_remaining, 0
    )

    s["required_stops_remaining"] = np.where(
        s["monaco_2025_rule"].to_numpy() == 1,
        monaco_remaining,
        ordinary_remaining,
    ).astype(np.int8)
    s["monaco_required_stops_remaining"] = (
        s["monaco_2025_rule"].to_numpy() * monaco_remaining
    ).astype(np.int8)
    s["monaco_extra_required_stops"] = (
        s["monaco_2025_rule"].to_numpy()
        * np.maximum(0, monaco_remaining - ordinary_remaining)
    ).astype(np.int8)

    race_progress = s["RaceProgress"].astype(float).clip(lower=0.001)
    total_laps_est = (s["LapNumber"].astype(float) / race_progress).replace(
        [np.inf, -np.inf], np.nan
    )
    s["LapsRemaining_Est"] = (
        (total_laps_est - s["LapNumber"].astype(float))
        .clip(lower=0, upper=120)
        .fillna(0)
    )

    denom = s["LapsRemaining_Est"] + 1.0
    s["required_stop_pressure"] = s["required_stops_remaining"] / denom
    s["monaco_stop_pressure"] = s["monaco_required_stops_remaining"] / denom
    s["finish_margin_laps"] = s["LapsRemaining_Est"] - s["required_stops_remaining"]
    s["monaco_finish_margin_laps"] = s["monaco_2025_rule"] * (
        s["LapsRemaining_Est"] - s["monaco_required_stops_remaining"]
    )
    s["monaco_must_pit_soon"] = (
        (s["monaco_required_stops_remaining"] > 0)
        & (s["LapsRemaining_Est"] <= (s["monaco_required_stops_remaining"] + 1.0))
    ).astype(np.int8)
    s["monaco_pressure_x_tyre_life"] = s["monaco_stop_pressure"] * s["TyreLife"].astype(
        float
    )
    s["monaco_pressure_x_race_progress"] = s["monaco_stop_pressure"] * s[
        "RaceProgress"
    ].astype(float)

    drop_cols = ["_is_wet_or_inter", "_is_soft", "_is_medium", "_is_hard"]
    s = s.drop(columns=drop_cols)
    s = s.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"])
    return s


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target = train["PitNextLap"].astype(int).to_numpy()
n_train = len(train)

all_df = pd.concat(
    [train.drop(columns=["PitNextLap"]), test],
    axis=0,
    ignore_index=True,
)
all_df = add_monaco_2025_rule_state(all_df)

for c in all_df.select_dtypes(include=["object"]).columns:
    all_df[c] = all_df[c].astype("category")

rename_map = safe_column_names(all_df.columns)
all_df = all_df.rename(columns=rename_map)

id_col = rename_map["id"]
feature_cols = [c for c in all_df.columns if c != id_col]

X_all = all_df[feature_cols].copy()
cat_cols = X_all.select_dtypes(include=["category"]).columns.tolist()

for c in cat_cols:
    X_all[c] = X_all[c].cat.add_categories("__MISSING__").fillna("__MISSING__")

num_cols = X_all.select_dtypes(include=[np.number]).columns.tolist()
X_all[num_cols] = X_all[num_cols].replace([np.inf, -np.inf], np.nan)
medians = X_all.iloc[:n_train][num_cols].median()
X_all[num_cols] = X_all[num_cols].fillna(medians).fillna(0)

X = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

groups_raw = train[["Year", "Race", "Driver"]].astype(str).agg("|".join, axis=1)
n_pos = int(target.sum())
n_neg = int(len(target) - n_pos)
scale_pos_weight = n_neg / max(n_pos, 1)

params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=2000,
    learning_rate=0.035,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    n_jobs=max(1, min(8, os.cpu_count() or 1)),
    verbosity=-1,
    scale_pos_weight=scale_pos_weight,
)

oof = np.zeros(len(X), dtype=float)
fold_aucs = []
best_iterations = []

try:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(X, target, groups_raw))
except Exception:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(X, target))

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        target[tr_idx],
        eval_set=[(X.iloc[va_idx], target[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    pred = model.predict_proba(X.iloc[va_idx], num_iteration=model.best_iteration_)[
        :, 1
    ]
    oof[va_idx] = pred
    auc = roc_auc_score(target[va_idx], pred)
    fold_aucs.append(auc)
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None and best_iter > 0:
        best_iterations.append(int(best_iter))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

overall_auc = roc_auc_score(target, oof)
mean_auc = float(np.mean(fold_aucs))
std_auc = float(np.std(fold_aucs))
print(f"OOF ROC AUC: {overall_auc:.6f}")
print(f"Mean fold ROC AUC: {mean_auc:.6f} +/- {std_auc:.6f}")

final_estimators = int(np.median(best_iterations)) if best_iterations else 800
final_params = params.copy()
final_params["n_estimators"] = max(50, final_estimators)

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(
    X,
    target,
    categorical_feature=cat_cols,
)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission["PitNextLap"] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(target)),
        "target": target,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample.copy()
test_pred_df["PitNextLap"] = test_pred
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "oof_roc_auc": float(overall_auc),
    "mean_fold_roc_auc": mean_auc,
    "std_fold_roc_auc": std_auc,
    "research_hypotheses_llm_claimed_used": ["000336"],
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
print(json.dumps(result, indent=2))
