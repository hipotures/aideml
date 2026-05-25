import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP = True
except ImportError:
    HAS_STRATIFIED_GROUP = False

warnings.filterwarnings("ignore")

SEED = 976
N_FOLDS = 5
INNER_FOLDS = 2
N_JOBS = min(8, os.cpu_count() or 1)
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
AUX_COL = "AuxLogLapsToPitPred"
AUX_MIN = np.log1p(1.0)
AUX_MAX = np.log1p(90.0)

os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

group_cols = [c for c in ["Year", "Race", "Driver"] if c in train.columns]
groups = train[group_cols].astype(str).agg("|".join, axis=1).to_numpy()
y = train[TARGET].astype(np.int8).to_numpy()


def make_laps_until_next_pit(df, group_cols, target_col):
    out = pd.Series(np.nan, index=df.index, dtype="float32")
    sort_cols = [c for c in ["LapNumber", "id"] if c in df.columns]
    for _, g in df.groupby(group_cols, sort=False):
        order = g.sort_values(sort_cols).index.to_numpy()
        laps = df.loc[order, "LapNumber"].to_numpy(dtype=np.float32)
        target = df.loc[order, target_col].to_numpy(dtype=np.int8)
        vals = np.empty(len(order), dtype=np.float32)
        next_pit_lap = np.inf
        last_lap = float(laps[-1])
        for i in range(len(order) - 1, -1, -1):
            if target[i] == 1:
                next_pit_lap = float(laps[i]) + 1.0
            vals[i] = (
                next_pit_lap - float(laps[i])
                if np.isfinite(next_pit_lap)
                else last_lap - float(laps[i]) + 2.0
            )
            vals[i] = max(vals[i], 1.0)
        out.loc[order] = vals
    return np.clip(out.to_numpy(dtype=np.float32), 1.0, 90.0)


def safe_name_map(cols):
    mapping, seen = {}, {}
    for col in cols:
        base = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col)).strip("_") or "col"
        if base[0].isdigit():
            base = "f_" + base
        n = seen.get(base, 0)
        mapping[col] = base if n == 0 else f"{base}_{n}"
        seen[base] = n + 1
    return mapping


def cv_splits(y_values, group_values, n_splits, seed):
    actual = min(n_splits, len(pd.unique(pd.Series(group_values))))
    if actual < 2:
        raise ValueError("Need at least two groups for grouped validation.")
    if HAS_STRATIFIED_GROUP:
        splitter = StratifiedGroupKFold(
            n_splits=actual, shuffle=True, random_state=seed
        )
    else:
        splitter = GroupKFold(n_splits=actual)
    return list(splitter.split(np.zeros(len(y_values)), y_values, group_values))


def clip_aux(values):
    return np.clip(np.asarray(values, dtype=np.float32), AUX_MIN, AUX_MAX)


def make_aux_model(seed):
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=-1,
        force_col_wise=True,
    )


def make_hazard_model(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=90,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.95,
        reg_alpha=0.2,
        reg_lambda=8.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=-1,
        force_col_wise=True,
    )


aux_target = np.log1p(make_laps_until_next_pit(train, group_cols, TARGET))

all_cols = list(dict.fromkeys(list(train.columns) + list(test.columns)))
name_map = safe_name_map(all_cols)
train_m = train.rename(columns=name_map)
test_m = test.rename(columns=name_map)

target_m = name_map[TARGET]
id_m = name_map[ID_COL]
base_features = [name_map[c] for c in train.columns if c not in [ID_COL, TARGET]]
cat_original = [
    c
    for c in train.columns
    if c not in [ID_COL, TARGET]
    and (
        train[c].dtype == "object" or (c in test.columns and test[c].dtype == "object")
    )
]
cat_cols = [name_map[c] for c in cat_original]

for c in cat_cols:
    cats = pd.Index(pd.concat([train_m[c], test_m[c]], axis=0).astype(str).unique())
    train_m[c] = pd.Categorical(train_m[c].astype(str), categories=cats)
    test_m[c] = pd.Categorical(test_m[c].astype(str), categories=cats)

X_base = train_m[base_features]
X_test_base = test_m[base_features]
outer_splits = cv_splits(y, groups, N_FOLDS, SEED)

oof_pred = np.zeros(len(train_m), dtype=np.float32)
aux_oof = np.zeros(len(train_m), dtype=np.float32)
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(outer_splits, 1):
    inner_aux_pred = np.zeros(len(tr_idx), dtype=np.float32)
    inner_splits = cv_splits(y[tr_idx], groups[tr_idx], INNER_FOLDS, SEED + fold)

    for inner_fold, (itr_rel, iva_rel) in enumerate(inner_splits, 1):
        itr_idx = tr_idx[itr_rel]
        iva_idx = tr_idx[iva_rel]
        reg = make_aux_model(SEED + 1000 * fold + inner_fold)
        reg.fit(X_base.iloc[itr_idx], aux_target[itr_idx], categorical_feature=cat_cols)
        inner_aux_pred[iva_rel] = clip_aux(reg.predict(X_base.iloc[iva_idx]))

    outer_reg = make_aux_model(SEED + 1000 * fold + 99)
    outer_reg.fit(X_base.iloc[tr_idx], aux_target[tr_idx], categorical_feature=cat_cols)
    val_aux_pred = clip_aux(outer_reg.predict(X_base.iloc[va_idx]))
    aux_oof[va_idx] = val_aux_pred

    X_tr = X_base.iloc[tr_idx].copy()
    X_va = X_base.iloc[va_idx].copy()
    X_tr[AUX_COL] = inner_aux_pred
    X_va[AUX_COL] = val_aux_pred

    clf = make_hazard_model(SEED + 2000 * fold)
    clf.fit(X_tr, y[tr_idx], categorical_feature=cat_cols)
    pred = clf.predict_proba(X_va)[:, 1].astype(np.float32)
    oof_pred[va_idx] = pred

    fold_auc = roc_auc_score(y[va_idx], pred)
    fold_aucs.append(float(fold_auc))
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof_pred)
print(f"OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train_m)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_aux = make_aux_model(SEED + 9001)
final_aux.fit(X_base, aux_target, categorical_feature=cat_cols)
test_aux = clip_aux(final_aux.predict(X_test_base))

X_full = X_base.copy()
X_test = X_test_base.copy()
X_full[AUX_COL] = aux_oof
X_test[AUX_COL] = test_aux

final_clf = make_hazard_model(SEED + 9002)
final_clf.fit(X_full, y, categorical_feature=cat_cols)
test_pred = np.clip(final_clf.predict_proba(X_test)[:, 1], 0.0, 1.0)

id_out = sample.columns[0]
target_out = sample.columns[1]
submission = pd.DataFrame(
    {
        id_out: sample[id_out].to_numpy(),
        target_out: test_pred,
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": fold_aucs,
            "research_hypotheses_llm_claimed_used": ["000976"],
        }
    )
)
