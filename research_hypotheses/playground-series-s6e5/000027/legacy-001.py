import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

test_ids = test[ID_COL].copy()
sample_id_col = sample.columns[0]
sample_target_col = [c for c in sample.columns if c != sample_id_col][0]

EXPECTED_LIFE = {
    "SOFT": 18.0,
    "MEDIUM": 28.0,
    "HARD": 38.0,
    "INTERMEDIATE": 22.0,
    "WET": 30.0,
}
DEFAULT_EXPECTED_LIFE = float(np.median(list(EXPECTED_LIFE.values())))


def add_compound_scaled_tyre_features(df):
    df = df.copy()
    compound = df["Compound"].astype(str).str.upper()
    expected = (
        compound.map(EXPECTED_LIFE).fillna(DEFAULT_EXPECTED_LIFE).astype("float32")
    )
    tyre_life = df["TyreLife"].astype("float32")
    race_progress = df["RaceProgress"].astype("float32")

    ratio = (tyre_life / np.maximum(expected, 1e-6)).clip(0, 5).astype("float32")
    remaining = (expected - tyre_life).clip(-100, 100).astype("float32")
    over_expected = np.maximum(tyre_life - expected, 0).astype("float32")

    df["CompoundExpectedLife"] = expected
    df["TyreLifeToExpectedLife"] = ratio
    df["CompoundExpectedLifeRemaining"] = remaining
    df["TyreLifeOverExpectedLife"] = over_expected
    df["OldForCompound80"] = (ratio >= 0.80).astype("int8")
    df["OldForCompound100"] = (ratio >= 1.00).astype("int8")
    df["OldForCompound120"] = (ratio >= 1.20).astype("int8")
    df["TyreLifeRatio_x_RaceProgress"] = (ratio * race_progress).astype("float32")
    df["ExpectedRemaining_x_RaceProgress"] = (remaining * race_progress).astype(
        "float32"
    )
    df["OverExpected_x_RaceProgress"] = (over_expected * race_progress).astype(
        "float32"
    )
    return df


def make_safe_column_mapping(columns):
    mapping = {}
    used = {}
    for col in columns:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_")
        if not base:
            base = "col"
        if base[0].isdigit():
            base = "f_" + base
        name = base
        i = used.get(base, 0)
        while name in used:
            i += 1
            name = f"{base}_{i}"
        used[base] = i
        used[name] = 0
        mapping[col] = name
    return mapping


train_fe = add_compound_scaled_tyre_features(train)
test_fe = add_compound_scaled_tyre_features(test)

all_columns = list(dict.fromkeys(list(train_fe.columns) + list(test_fe.columns)))
column_mapping = make_safe_column_mapping(all_columns)
train_fe = train_fe.rename(columns=column_mapping)
test_fe = test_fe.rename(columns=column_mapping)

target_col = column_mapping[TARGET]
id_col = column_mapping[ID_COL]
categorical_cols = [
    column_mapping[c]
    for c in ["Compound", "Driver", "Race"]
    if column_mapping[c] in train_fe.columns
]

for col in categorical_cols:
    combined = (
        pd.concat([train_fe[col], test_fe[col]], axis=0)
        .astype("string")
        .fillna("__MISSING__")
    )
    categories = pd.Index(combined.unique())
    train_fe[col] = pd.Categorical(
        train_fe[col].astype("string").fillna("__MISSING__"), categories=categories
    )
    test_fe[col] = pd.Categorical(
        test_fe[col].astype("string").fillna("__MISSING__"), categories=categories
    )

feature_cols = [c for c in train_fe.columns if c not in [target_col, id_col]]
X = train_fe[feature_cols]
y = train_fe[target_col].astype("int8")
X_test = test_fe[feature_cols]

base_params = {
    "objective": "binary",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 64,
    "min_child_samples": 80,
    "subsample": 0.90,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "random_state": RANDOM_STATE,
    "n_jobs": max(1, (os.cpu_count() or 2) - 1),
    "verbosity": -1,
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(train_fe), dtype=np.float32)
fold_aucs = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    params = base_params.copy()
    params["scale_pos_weight"] = neg / max(pos, 1)

    model = LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(period=0)],
    )

    pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
    oof[va_idx] = pred.astype(np.float32)
    auc = roc_auc_score(y_va, pred)
    fold_aucs.append(float(auc))
    best_iter = int(model.best_iteration_ or base_params["n_estimators"])
    best_iterations.append(best_iter)
    print(f"Fold {fold} ROC AUC: {auc:.6f} best_iteration={best_iter}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold CV ROC AUC: {cv_auc:.6f}")

usable_best = [b for b in best_iterations if b > 0]
final_estimators = (
    int(np.ceil(np.mean(usable_best) * 1.10))
    if usable_best
    else base_params["n_estimators"]
)
final_estimators = int(np.clip(final_estimators, 100, base_params["n_estimators"]))

final_params = base_params.copy()
final_params["n_estimators"] = final_estimators
final_params["scale_pos_weight"] = (len(y) - int(y.sum())) / max(int(y.sum()), 1)

final_model = LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=categorical_cols)

test_pred = final_model.predict_proba(X_test)[:, 1].astype(np.float32)
test_pred = np.clip(test_pred, 0.0, 1.0)

pred_by_id = pd.Series(test_pred, index=test_ids.values)
submission = pd.DataFrame({sample_id_col: sample[sample_id_col].values})
submission[sample_target_col] = (
    submission[sample_id_col].map(pred_by_id).astype(np.float32)
)

if submission[sample_target_col].isna().any():
    raise ValueError("Some sample submission ids were not found in test predictions.")

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(y), dtype=np.int32),
        "target": y.astype(np.int8).values,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_mean_auc": float(cv_auc),
            "cv_fold_auc": fold_aucs,
            "final_n_estimators": final_estimators,
            "research_hypotheses_llm_claimed_used": ["000027"],
        }
    )
)
