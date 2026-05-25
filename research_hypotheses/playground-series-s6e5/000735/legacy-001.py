import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGKF = True
except Exception:
    from sklearn.model_selection import GroupKFold

    HAS_SGKF = False

from catboost import CatBoostClassifier, Pool

SEED = 735
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"


def add_features(df):
    df = df.copy()
    eps = 1e-6

    for c in ["Driver", "Race", "Compound"]:
        df[c] = df[c].astype(str).fillna("NA")

    df["Year_cat"] = df["Year"].astype(str)
    df["Driver_Race"] = df["Driver"] + "__" + df["Race"]
    df["Race_Year"] = df["Race"] + "__" + df["Year_cat"]
    df["Driver_Compound"] = df["Driver"] + "__" + df["Compound"]

    lap = df["LapNumber"].astype(float)
    progress = df["RaceProgress"].astype(float).clip(eps, 1.0)
    tyre = df["TyreLife"].astype(float).clip(lower=0)
    deg = df["Cumulative_Degradation"].astype(float)
    delta = df["LapTime_Delta"].astype(float)

    est_total_laps = (lap / progress).replace([np.inf, -np.inf], np.nan).clip(1, 90)
    remaining_laps = (est_total_laps - lap).clip(lower=0)

    mid_window = (1.0 - (progress - 0.50).abs() / 0.38).clip(lower=0)
    positive_deg = deg.clip(lower=0)
    positive_delta = delta.clip(lower=0)

    df["EstimatedTotalLaps"] = est_total_laps
    df["RemainingLaps"] = remaining_laps
    df["TyreLifeShare"] = tyre / (lap + 1.0)
    df["DegradationPerTyreLap"] = deg / (tyre + 1.0)
    df["PositiveLapDelta"] = positive_delta
    df["LatentWear"] = tyre + positive_deg / 55.0 + positive_delta / 35.0
    df["ServiceWindowPressure"] = tyre * mid_window
    df["CannotFinishPressure"] = tyre * np.sqrt((1.0 - progress).clip(lower=0) + 0.02)
    df["FreshTyreRelief"] = (df["PitStop"].astype(float) > 0).astype(int) * (
        1.0 / (tyre + 1.0)
    )

    return df.replace([np.inf, -np.inf], np.nan)


train_fe = add_features(train)
test_fe = add_features(test)

cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "Year_cat",
    "Driver_Race",
    "Race_Year",
    "Driver_Compound",
]

num_cols = [
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
    "EstimatedTotalLaps",
    "RemainingLaps",
    "TyreLifeShare",
    "DegradationPerTyreLap",
    "PositiveLapDelta",
    "LatentWear",
    "ServiceWindowPressure",
    "CannotFinishPressure",
    "FreshTyreRelief",
]

features = cat_cols + num_cols

for c in cat_cols:
    train_fe[c] = train_fe[c].astype(str).fillna("NA")
    test_fe[c] = test_fe[c].astype(str).fillna("NA")

for c in num_cols:
    med = train_fe[c].median()
    train_fe[c] = pd.to_numeric(train_fe[c], errors="coerce").fillna(med)
    test_fe[c] = pd.to_numeric(test_fe[c], errors="coerce").fillna(med)

X = train_fe[features]
X_test = test_fe[features]
y = train_fe[target_col].astype(int).values
groups = train_fe["Year"].astype(str) + "__" + train_fe["Race"].astype(str)

monotone_positive = {
    "TyreLife",
    "Cumulative_Degradation",
    "DegradationPerTyreLap",
    "PositiveLapDelta",
    "LatentWear",
    "ServiceWindowPressure",
    "CannotFinishPressure",
}
monotone_negative = {"PitStop", "FreshTyreRelief"}

monotone_constraints = []
for f in features:
    if f in monotone_positive:
        monotone_constraints.append(1)
    elif f in monotone_negative:
        monotone_constraints.append(-1)
    else:
        monotone_constraints.append(0)

threads = max(1, min(8, os.cpu_count() or 1))
common_params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    bootstrap_type="Bernoulli",
    subsample=0.82,
    random_strength=0.8,
    allow_writing_files=False,
    thread_count=threads,
    verbose=False,
)

base_params = dict(
    **common_params,
    iterations=520,
    learning_rate=0.055,
    depth=6,
    l2_leaf_reg=6.0,
    od_type="Iter",
    od_wait=70,
    random_seed=SEED,
)

specialist_params = dict(
    **common_params,
    iterations=420,
    learning_rate=0.055,
    depth=5,
    l2_leaf_reg=10.0,
    od_type="Iter",
    od_wait=70,
    random_seed=SEED + 1000,
    monotone_constraints=monotone_constraints,
)

if HAS_SGKF:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y, groups))
else:
    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(X, y, groups))

oof_base = np.zeros(len(train_fe), dtype=float)
oof_specialist = np.zeros(len(train_fe), dtype=float)
test_base = np.zeros(len(test_fe), dtype=float)
test_specialist = np.zeros(len(test_fe), dtype=float)

cat_feature_indices = [features.index(c) for c in cat_cols]
test_pool = Pool(X_test, cat_features=cat_feature_indices)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_feature_indices)
    valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_feature_indices)

    base_model = CatBoostClassifier(**base_params)
    base_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    oof_base[va_idx] = base_model.predict_proba(valid_pool)[:, 1]
    test_base += base_model.predict_proba(test_pool)[:, 1] / len(splits)

    specialist_model = CatBoostClassifier(**specialist_params)
    specialist_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    oof_specialist[va_idx] = specialist_model.predict_proba(valid_pool)[:, 1]
    test_specialist += specialist_model.predict_proba(test_pool)[:, 1] / len(splits)

    fold_blend = 0.65 * oof_base[va_idx] + 0.35 * oof_specialist[va_idx]
    print(f"fold {fold} ROC AUC: {roc_auc_score(y[va_idx], fold_blend):.6f}")

oof_pred = np.clip(0.65 * oof_base + 0.35 * oof_specialist, 0, 1)
test_pred = np.clip(0.65 * test_base + 0.35 * test_specialist, 0, 1)

auc_base = roc_auc_score(y, oof_base)
auc_specialist = roc_auc_score(y, oof_specialist)
auc_blend = roc_auc_score(y, oof_pred)

print(f"base CV ROC AUC: {auc_base:.6f}")
print(f"monotone specialist CV ROC AUC: {auc_specialist:.6f}")
print(f"blended CV ROC AUC: {auc_blend:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission = sample.copy()
submission[target_col] = test_pred
submission[[id_col, target_col]].to_csv(
    os.path.join(WORK_DIR, "submission.csv"), index=False
)
submission[[id_col, target_col]].to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "research_hypotheses_llm_claimed_used": ["000735"],
    "metric": "roc_auc",
    "cv_roc_auc": float(auc_blend),
    "base_cv_roc_auc": float(auc_base),
    "monotone_specialist_cv_roc_auc": float(auc_specialist),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)
print(json.dumps(review, sort_keys=True))
