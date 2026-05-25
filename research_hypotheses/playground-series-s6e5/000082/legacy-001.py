import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
SEED = 42
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
base_cols = [c for c in train.columns if c != TARGET]
combined = pd.concat(
    [train[base_cols].copy(), test[base_cols].copy()],
    axis=0,
    ignore_index=True,
)
n_train = len(train)

race_keys = ["Year", "Race", "LapNumber"]
race_group_keys = ["Year", "Race"]

lap_stats = (
    combined.groupby(race_keys, observed=True)["PitStop"]
    .agg(pit_wave_count="sum", pit_wave_field_size="size", pit_wave_rate="mean")
    .reset_index()
)

for lag in [1, 2]:
    shifted = lap_stats.copy()
    shifted["LapNumber"] = shifted["LapNumber"] + lag
    shifted = shifted.rename(
        columns={
            "pit_wave_count": f"prev{lag}_pit_wave_count",
            "pit_wave_field_size": f"prev{lag}_pit_wave_field_size",
            "pit_wave_rate": f"prev{lag}_pit_wave_rate",
        }
    )
    combined = combined.merge(shifted, on=race_keys, how="left")

wave_cols = [c for c in combined.columns if c.startswith("prev")]
combined[wave_cols] = combined[wave_cols].fillna(0.0)

combined["prev1_any_pit_wave"] = (combined["prev1_pit_wave_count"] > 0).astype(int)
combined["prev2_any_pit_wave"] = (combined["prev2_pit_wave_count"] > 0).astype(int)
combined["pit_wave_count_delta"] = (
    combined["prev1_pit_wave_count"] - combined["prev2_pit_wave_count"]
)
combined["pit_wave_rate_delta"] = (
    combined["prev1_pit_wave_rate"] - combined["prev2_pit_wave_rate"]
)
combined["weighted_recent_pit_wave_rate"] = (
    combined["prev1_pit_wave_rate"] + 0.5 * combined["prev2_pit_wave_rate"]
)

combined["degradation_per_tyre_lap"] = combined["Cumulative_Degradation"] / (
    combined["TyreLife"] + 1.0
)
combined["tyre_pressure"] = (
    combined["TyreLife"] * (1.0 + combined["RaceProgress"])
    + np.clip(combined["degradation_per_tyre_lap"], -50, 50) / 10.0
)
combined["late_race"] = (combined["RaceProgress"] >= 0.75).astype(int)
combined["early_race"] = (combined["RaceProgress"] <= 0.25).astype(int)
combined["next_stop_feasible"] = (
    (combined["PitStop"] == 0)
    & (combined["TyreLife"] >= 2)
    & (combined["RaceProgress"] < 0.985)
).astype(int)

combined["prev1_wave_x_position"] = (
    combined["prev1_pit_wave_rate"] * combined["Position"]
)
combined["prev1_wave_x_front_pressure"] = combined["prev1_pit_wave_rate"] * (
    21 - combined["Position"]
)
combined["prev1_wave_x_tyre_pressure"] = (
    combined["prev1_pit_wave_rate"] * combined["tyre_pressure"]
)
combined["prev1_wave_x_feasible"] = (
    combined["prev1_pit_wave_rate"] * combined["next_stop_feasible"]
)
combined["weighted_wave_x_tyre_pressure"] = (
    combined["weighted_recent_pit_wave_rate"] * combined["tyre_pressure"]
)
combined["weighted_wave_x_feasible"] = (
    combined["weighted_recent_pit_wave_rate"] * combined["next_stop_feasible"]
)
combined["count_delta_x_position"] = (
    combined["pit_wave_count_delta"] * combined["Position"]
)

feature_df = combined.drop(columns=[ID_COL])


def clean_name(name):
    name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    return name or "feature"


new_cols, seen = [], {}
for col in feature_df.columns:
    base = clean_name(col)
    name = base
    i = 1
    while name in seen:
        i += 1
        name = f"{base}_{i}"
    seen[name] = True
    new_cols.append(name)
feature_df.columns = new_cols

cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    feature_df[col] = feature_df[col].astype("category")

X = feature_df.iloc[:n_train].reset_index(drop=True)
X_test = feature_df.iloc[n_train:].reset_index(drop=True)

groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = splitter.split(X, y, groups)
else:
    splitter = GroupKFold(n_splits=5)
    splits = splitter.split(X, y, groups)

base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=1800,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
    force_col_wise=True,
)

oof = np.zeros(n_train, dtype=float)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    pos = max(y_tr.sum(), 1)
    neg = len(y_tr) - y_tr.sum()
    params = base_params.copy()
    params["scale_pos_weight"] = float(neg / pos)

    model = LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    pred = model.predict_proba(X_va)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y_va, pred)
    fold_scores.append(auc)
    best_iterations.append(model.best_iteration_ or params["n_estimators"])
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold grouped CV ROC AUC: {cv_auc:.6f}")

pd.DataFrame({"row": np.arange(n_train), "target": y, "prediction": oof}).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_params = base_params.copy()
final_params["n_estimators"] = int(np.mean(best_iterations))
pos = max(y.sum(), 1)
neg = len(y) - y.sum()
final_params["scale_pos_weight"] = float(neg / pos)

final_model = LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_auc": float(cv_auc),
    "fold_auc": [float(v) for v in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000082"],
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
