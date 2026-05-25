import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
GROUP_COLS = ["Year", "Race", "Driver"]
LAP_COL = "LapNumber"
PIT_COL = "PitStop"


def add_shifted_next_pit(df):
    out = df.copy()
    sort_cols = GROUP_COLS + [LAP_COL]
    if ID_COL in out.columns:
        sort_cols.append(ID_COL)

    tmp = out[
        GROUP_COLS + [LAP_COL, PIT_COL] + ([ID_COL] if ID_COL in out.columns else [])
    ].copy()
    tmp["_row"] = np.arange(len(tmp))
    tmp = tmp.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    g = tmp.groupby(GROUP_COLS, sort=False, observed=False)
    tmp["_next_pit"] = g[PIT_COL].shift(-1)
    tmp["_next_lap"] = g[LAP_COL].shift(-1)
    tmp["_dup_lap"] = tmp.duplicated(GROUP_COLS + [LAP_COL], keep=False)
    tmp["_group_has_dup_lap"] = (
        tmp.groupby(GROUP_COLS, sort=False, observed=False)["_dup_lap"]
        .transform("max")
        .astype(bool)
    )

    valid = (
        tmp["_next_pit"].notna()
        & (~tmp["_group_has_dup_lap"])
        & (tmp["_next_lap"] == tmp[LAP_COL] + 1)
    )

    shifted = np.full(len(out), np.nan, dtype=float)
    valid_mask = np.zeros(len(out), dtype=bool)
    rows = tmp["_row"].to_numpy()

    shifted[rows] = tmp["_next_pit"].to_numpy(dtype=float)
    valid_mask[rows] = valid.to_numpy(dtype=bool)

    out["shifted_next_pit"] = shifted
    out["shift_valid"] = valid_mask
    return out


def make_model(y_train, seed=2026):
    pos = float(np.sum(y_train))
    neg = float(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1.0)

    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=64,
        min_child_samples=50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def apply_shift_overlay(base_pred, frame):
    pred = np.clip(np.asarray(base_pred, dtype=float).copy(), 0.002, 0.998)
    mask = frame["shift_valid"].to_numpy(dtype=bool)
    if mask.any():
        shifted = frame.loc[mask, "shifted_next_pit"].to_numpy(dtype=float)
        pred[mask] = np.where(shifted >= 0.5, 0.999, 0.001)
    return pred


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = add_shifted_next_pit(train)
test = add_shifted_next_pit(test)

y = train[TARGET].astype(int).to_numpy()
shift_mask = train["shift_valid"].to_numpy(dtype=bool)
shift_coverage = float(shift_mask.mean())

if shift_mask.any():
    shifted_train = train.loc[shift_mask, "shifted_next_pit"].astype(int).to_numpy()
    target_train = y[shift_mask]
    shift_alignment = float(np.mean(shifted_train == target_train))
    shift_auc = (
        float(roc_auc_score(target_train, shifted_train))
        if len(np.unique(target_train)) == 2
        else None
    )
else:
    shift_alignment = None
    shift_auc = None

feature_cols = [
    c
    for c in train.columns
    if c not in [ID_COL, TARGET, "shifted_next_pit", "shift_valid"]
]
cat_cols = [
    c for c in feature_cols if train[c].dtype == "object" or test[c].dtype == "object"
]

for c in cat_cols:
    all_values = (
        pd.concat([train[c], test[c]], axis=0).astype("string").fillna("__MISSING__")
    )
    cats = pd.Index(all_values.unique())
    train[c] = pd.Categorical(
        train[c].astype("string").fillna("__MISSING__"), categories=cats
    )
    test[c] = pd.Categorical(
        test[c].astype("string").fillna("__MISSING__"), categories=cats
    )

X = train[feature_cols]
X_test = test[feature_cols]

tr_idx, va_idx = train_test_split(
    np.arange(len(train)),
    test_size=0.2,
    random_state=2026,
    stratify=y,
)

model = make_model(y[tr_idx], seed=2026)
model.fit(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_cols)

valid_base = model.predict_proba(X.iloc[va_idx])[:, 1]
valid_pred = apply_shift_overlay(valid_base, train.iloc[va_idx])
valid_auc = float(roc_auc_score(y[va_idx], valid_pred))

validation_predictions = pd.DataFrame(
    {
        "row": va_idx,
        "target": y[va_idx],
        "prediction": valid_pred,
    }
)
validation_predictions.to_csv(
    os.path.join(WORK_DIR, "validation_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_model = make_model(y, seed=2027)
final_model.fit(X, y, categorical_feature=cat_cols)

test_base = final_model.predict_proba(X_test)[:, 1]
test_pred = apply_shift_overlay(test_base, test)

sample_id_col = sample.columns[0]
sample_target_col = sample.columns[1]
submission = sample[[sample_id_col]].copy()

if np.array_equal(sample[sample_id_col].to_numpy(), test[ID_COL].to_numpy()):
    submission[sample_target_col] = test_pred
else:
    pred_map = pd.Series(test_pred, index=test[ID_COL])
    submission[sample_target_col] = sample[sample_id_col].map(pred_map).to_numpy()

if submission[sample_target_col].isna().any():
    raise RuntimeError(
        "Some test ids from sample_submission were not found in test predictions."
    )

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

report = {
    "research_hypotheses_llm_claimed_used": ["000946"],
    "validation_metric": "roc_auc",
    "validation_roc_auc": valid_auc,
    "train_shift_coverage": shift_coverage,
    "train_shift_alignment": shift_alignment,
    "train_shift_auc_on_covered_rows": shift_auc,
    "test_shift_coverage": float(test["shift_valid"].mean()),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "validation_predictions_path": os.path.join(
        WORK_DIR, "validation_predictions.csv.gz"
    ),
    "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
}

print(json.dumps(report, indent=2))
