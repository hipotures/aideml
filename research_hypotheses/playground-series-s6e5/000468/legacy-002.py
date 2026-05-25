import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP = True
except Exception:
    HAS_STRATIFIED_GROUP = False

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

n_train = len(train)
y = train[TARGET].astype(int).to_numpy()

all_data = pd.concat(
    [train.drop(columns=[TARGET]), test],
    axis=0,
    ignore_index=True,
    sort=False,
)


def add_monaco_2025_rule_debt_features(df):
    df = df.copy()

    monaco_2025 = (
        df["Race"].astype(str).eq("Monaco Grand Prix") & df["Year"].astype(int).eq(2025)
    ).astype(np.int8)

    df["Monaco2025"] = monaco_2025
    df["RequiredStopCount"] = np.where(monaco_2025 == 1, 2, 1).astype(np.int8)
    df["RequiredTyreSets"] = np.where(monaco_2025 == 1, 3, 2).astype(np.int8)

    progress = df["RaceProgress"].astype(float).clip(1e-4, 1.0)
    estimated_total_laps = df["LapNumber"].astype(float) / progress
    df["LapsRemaining_Est"] = (
        (estimated_total_laps - df["LapNumber"].astype(float))
        .clip(0, 120)
        .astype(np.float32)
    )

    sort_cols = ["Year", "Race", "Driver", "LapNumber", ID_COL]
    ordered = df.sort_values(sort_cols, kind="mergesort")
    pit_counts = ordered.groupby(["Year", "Race", "Driver"], sort=False)[
        "PitStop"
    ].cumsum()
    df.loc[ordered.index, "PitCountAfterCurrent"] = pit_counts.astype(
        np.float32
    ).to_numpy()

    df["CurrentTyreSetCount_Est"] = (df["PitCountAfterCurrent"] + 1.0).astype(
        np.float32
    )
    df["StopDebtToRule"] = (
        (df["RequiredStopCount"].astype(float) - df["PitCountAfterCurrent"])
        .clip(lower=0)
        .astype(np.float32)
    )
    df["RuleDebtAfterCurrent"] = (
        (df["RequiredTyreSets"].astype(float) - df["CurrentTyreSetCount_Est"])
        .clip(lower=0)
        .astype(np.float32)
    )

    late_race = (
        (df["RaceProgress"].astype(float) >= 0.60)
        | (df["LapsRemaining_Est"].astype(float) <= 20)
    ).astype(np.int8)

    df["LateRace_MonacoDebt"] = (
        df["Monaco2025"].astype(float)
        * df["RuleDebtAfterCurrent"].astype(float)
        * late_race.astype(float)
        / (df["LapsRemaining_Est"].astype(float) + 1.0)
    ).astype(np.float32)

    df["RuleDebt_x_PitCountAfterCurrent"] = (
        df["RuleDebtAfterCurrent"].astype(float)
        * df["PitCountAfterCurrent"].astype(float)
    ).astype(np.float32)
    df["RuleDebt_x_LapsRemaining_Est"] = (
        df["RuleDebtAfterCurrent"].astype(float) * df["LapsRemaining_Est"].astype(float)
    ).astype(np.float32)
    df["StopDebtPressure"] = (
        df["StopDebtToRule"].astype(float)
        / (df["LapsRemaining_Est"].astype(float) + 1.0)
    ).astype(np.float32)

    return df


all_data = add_monaco_2025_rule_debt_features(all_data)

categorical_cols = [c for c in ["Compound", "Driver", "Race"] if c in all_data.columns]
for col in categorical_cols:
    all_data[col] = all_data[col].astype("category")

feature_cols = [c for c in all_data.columns if c != ID_COL]


def safe_feature_names(cols):
    mapping = {}
    used = set()
    for i, col in enumerate(cols):
        name = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
        if not name:
            name = f"feature_{i}"
        if name[0].isdigit():
            name = f"f_{name}"
        base = name
        j = 2
        while name in used:
            name = f"{base}_{j}"
            j += 1
        used.add(name)
        mapping[col] = name
    return mapping


rename_map = safe_feature_names(feature_cols)
cat_features = [rename_map[c] for c in categorical_cols]

X_all = all_data[feature_cols].rename(columns=rename_map)
for col in X_all.select_dtypes(include=["float64"]).columns:
    X_all[col] = X_all[col].astype(np.float32)
for col in X_all.select_dtypes(include=["int64"]).columns:
    X_all[col] = X_all[col].astype(np.int32)

X = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

groups = (train["Year"].astype(str) + "_" + train["Race"].astype(str)).to_numpy()


def make_cv_splits():
    if HAS_STRATIFIED_GROUP:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=468)
        splits = list(cv.split(X, y, groups))
        if all(len(np.unique(y[val_idx])) == 2 for _, val_idx in splits):
            return splits, "StratifiedGroupKFold_by_YearRace"

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=468)
    return list(cv.split(X, y)), "StratifiedKFold_fallback"


splits, cv_name = make_cv_splits()

base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=2500,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.88,
    subsample_freq=1,
    colsample_bytree=0.86,
    reg_alpha=0.05,
    reg_lambda=1.5,
    random_state=468,
    n_jobs=max(1, os.cpu_count() or 1),
    verbose=-1,
)

oof = np.zeros(n_train, dtype=np.float32)
fold_scores = []
best_iterations = []

for fold, (tr_idx, val_idx) in enumerate(splits, start=1):
    fold_params = dict(base_params)
    fold_params["random_state"] = 468 + fold

    model = LGBMClassifier(**fold_params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[val_idx], y[val_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[early_stopping(100), log_evaluation(200)],
    )

    best_iter = model.best_iteration_ or fold_params["n_estimators"]
    best_iterations.append(best_iter)

    val_pred = model.predict_proba(X.iloc[val_idx], num_iteration=best_iter)[:, 1]
    oof[val_idx] = val_pred.astype(np.float32)

    fold_auc = roc_auc_score(y[val_idx], val_pred)
    fold_scores.append(fold_auc)
    print(f"fold {fold} auc: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"mean fold auc: {np.mean(fold_scores):.6f}")
print(f"oof auc: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_estimators = int(np.median(best_iterations)) if best_iterations else 500
final_params = dict(base_params)
final_params["n_estimators"] = max(50, final_estimators)
final_params["random_state"] = 10468

final_model = LGBMClassifier(**final_params)
final_model.fit(
    X,
    y,
    categorical_feature=cat_features,
)

test_pred = final_model.predict_proba(X_test)[:, 1].astype(np.float32)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

submission[[ID_COL, TARGET]].to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_strategy": cv_name,
    "fold_auc": [round(float(s), 6) for s in fold_scores],
    "mean_fold_auc": round(float(np.mean(fold_scores)), 6),
    "oof_auc": round(float(cv_auc), 6),
    "final_model_estimators": int(final_params["n_estimators"]),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "research_hypotheses_llm_claimed_used": ["000468"],
}
print(json.dumps(result, indent=2))
