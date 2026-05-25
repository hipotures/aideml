import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

import lightgbm as lgb

SEED = 42
N_FOLDS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
ID_COL = "id"
TARGET = "PitNextLap"

os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
all_raw = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)


def sanitize_columns(cols):
    seen, new_cols = {}, []
    for c in cols:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_") or "feature"
        if base[0].isdigit():
            base = "f_" + base
        n = seen.get(base, 0)
        seen[base] = n + 1
        new_cols.append(base if n == 0 else f"{base}_{n}")
    return new_cols


def add_rule_pressure_features(df):
    out = df.copy()
    out["_orig_order"] = np.arange(len(out))

    out["Race"] = out["Race"].astype(str)
    out["Driver"] = out["Driver"].astype(str)
    out["Compound"] = out["Compound"].astype(str).str.upper()

    race_lower = out["Race"].str.lower()
    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["is_Monaco"] = race_lower.str.contains("monaco", regex=False).astype(np.int8)
    out["is_Qatar"] = race_lower.str.contains("qatar", regex=False).astype(np.int8)
    out["is_Monaco_2025"] = ((out["is_Monaco"] == 1) & (out["Year"] == 2025)).astype(
        np.int8
    )
    out["is_Qatar_2023"] = ((out["is_Qatar"] == 1) & (out["Year"] == 2023)).astype(
        np.int8
    )
    out["is_Qatar_2025"] = ((out["is_Qatar"] == 1) & (out["Year"] == 2025)).astype(
        np.int8
    )

    out["PriorPitCount"] = np.maximum(out["Stint"].astype(float) - 1.0, 0.0)
    out["PitCount_Including_Current"] = out["PriorPitCount"] + out["PitStop"].astype(
        float
    )

    sort_cols = ["Year", "Race", "Driver", "LapNumber", ID_COL]
    ordered = out.sort_values(sort_cols).copy()
    group_cols = ["Year", "Race", "Driver"]

    comp = ordered["Compound"].astype(str).str.upper()
    wet_now = comp.isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    ordered["_wet_now"] = wet_now
    ordered["wet_rule_exemption_seen"] = (
        ordered.groupby(group_cols)["_wet_now"].cummax().astype(np.int8)
    )

    dry_seen_cols = []
    for dry_comp in ["SOFT", "MEDIUM", "HARD"]:
        col = f"seen_dry_{dry_comp}"
        ordered[col] = (comp == dry_comp).astype(np.int8)
        ordered[col] = ordered.groupby(group_cols)[col].cummax().astype(np.int8)
        dry_seen_cols.append(col)

    ordered["dry_specs_seen_count"] = ordered[dry_seen_cols].sum(axis=1).astype(np.int8)
    ordered["normal_two_spec_debt"] = (
        (ordered["wet_rule_exemption_seen"] == 0)
        & (ordered["dry_specs_seen_count"] < 2)
    ).astype(np.int8)

    qatar_cap = np.select(
        [ordered["is_Qatar_2023"] == 1, ordered["is_Qatar_2025"] == 1],
        [18.0, 25.0],
        default=0.0,
    )
    ordered["Qatar_stint_cap"] = qatar_cap
    raw_laps_to_cap = qatar_cap - ordered["TyreLife"].astype(float)
    ordered["laps_to_Qatar_stint_cap"] = np.where(qatar_cap > 0, raw_laps_to_cap, 999.0)
    ordered["laps_to_Qatar_stint_cap_clipped"] = np.where(
        qatar_cap > 0, np.clip(raw_laps_to_cap, -5.0, 30.0), 30.0
    )
    ordered["qatar_stint_cap_due_next"] = (
        (qatar_cap > 0) & (raw_laps_to_cap <= 1.0)
    ).astype(np.int8)
    ordered["qatar_stint_cap_overdue"] = (
        (qatar_cap > 0) & (raw_laps_to_cap <= 0.0)
    ).astype(np.int8)
    ordered["qatar_cap_pressure"] = np.where(
        qatar_cap > 0, 1.0 / (1.0 + np.maximum(raw_laps_to_cap, 0.0)), 0.0
    )

    ordered["Monaco_three_set_debt"] = np.where(
        ordered["is_Monaco_2025"] == 1,
        np.maximum(0.0, 3.0 - ordered["Stint"].astype(float)),
        0.0,
    )

    ordered["required_remaining_stops_by_rule"] = np.maximum.reduce(
        [
            ordered["normal_two_spec_debt"].astype(float),
            ordered["Monaco_three_set_debt"].astype(float),
            ordered["qatar_stint_cap_due_next"].astype(float),
        ]
    )

    ordered["TyreLife_x_required_remaining_stops_by_rule"] = (
        ordered["TyreLife"].astype(float) * ordered["required_remaining_stops_by_rule"]
    )
    ordered["Stint_x_required_remaining_stops_by_rule"] = (
        ordered["Stint"].astype(float) * ordered["required_remaining_stops_by_rule"]
    )
    ordered["PriorPitCount_x_required_remaining_stops_by_rule"] = (
        ordered["PriorPitCount"] * ordered["required_remaining_stops_by_rule"]
    )
    ordered["TyreLife_x_Monaco_three_set_debt"] = (
        ordered["TyreLife"].astype(float) * ordered["Monaco_three_set_debt"]
    )
    ordered["Stint_x_Monaco_three_set_debt"] = (
        ordered["Stint"].astype(float) * ordered["Monaco_three_set_debt"]
    )
    ordered["TyreLife_x_qatar_cap_pressure"] = (
        ordered["TyreLife"].astype(float) * ordered["qatar_cap_pressure"]
    )
    ordered["PriorPitCount_x_wet_rule_exemption_seen"] = (
        ordered["PriorPitCount"] * ordered["wet_rule_exemption_seen"]
    )

    ordered = ordered.sort_values("_orig_order")
    return ordered.drop(columns=["_orig_order", "_wet_now"])


all_feat = add_rule_pressure_features(all_raw)

feature_cols = [c for c in all_feat.columns if c != ID_COL]
X_all = all_feat[feature_cols].copy()

cat_cols_raw = X_all.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols_raw:
    X_all[c] = X_all[c].astype("category")

old_cols = list(X_all.columns)
new_cols = sanitize_columns(old_cols)
rename_map = dict(zip(old_cols, new_cols))
X_all.columns = new_cols
cat_cols = [rename_map[c] for c in cat_cols_raw]

X_train = X_all.iloc[: len(train)].reset_index(drop=True)
X_test = X_all.iloc[len(train) :].reset_index(drop=True)

groups = (
    train["Year"].astype(str)
    + "|"
    + train["Race"].astype(str)
    + "|"
    + train["Driver"].astype(str)
).to_numpy()

pos = max(1, int(y.sum()))
neg = max(1, int(len(y) - y.sum()))
scale_pos_weight = neg / pos


def make_model(n_estimators=2500, seed=SEED):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


if StratifiedGroupKFold is not None:
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = cv.split(X_train, y, groups)
else:
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = cv.split(X_train, y)

oof = np.zeros(len(train), dtype=float)
fold_scores, best_iters = [], []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = make_model(seed=SEED + fold)
    model.fit(
        X_train.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X_train.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    pred = model.predict_proba(X_train.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    fold_auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(fold_auc)
    best_iters.append(model.best_iteration_ or model.n_estimators)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

oof_auc = roc_auc_score(y, oof)
final_estimators = int(np.median(best_iters)) if best_iters else 800
final_estimators = max(50, final_estimators)

final_model = make_model(n_estimators=final_estimators, seed=SEED + 999)
final_model.fit(X_train, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

target_col = (
    TARGET
    if TARGET in sample.columns
    else [c for c in sample.columns if c != ID_COL][0]
)
submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission[[ID_COL, target_col]].to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "oof_roc_auc": float(oof_auc),
    "fold_roc_auc": [float(x) for x in fold_scores],
    "final_n_estimators": int(final_estimators),
    "research_hypotheses_llm_claimed_used": ["000507"],
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"OOF ROC AUC: {oof_auc:.6f}")
print(json.dumps(result))
