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
except Exception:
    StratifiedGroupKFold = None

from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 20260524
TARGET = "PitNextLap"
ID_COL = "id"
HORIZONS = [2, 3, 5]
MAX_LAPS_TARGET = 8
OUTER_SPLITS = 5
INNER_SPLITS = 3
AUX_TREES = 100
FINAL_TREES = 520
FULL_FINAL_TREES = 620
N_JOBS = min(8, os.cpu_count() or 1)

os.makedirs("./working", exist_ok=True)

train = pd.read_csv("./input/train.csv.gz")
test = pd.read_csv("./input/test.csv.gz")
sample = pd.read_csv("./input/sample_submission.csv.gz")

cat_cols = [c for c in ["Compound", "Driver", "Race"] if c in train.columns]
for c in cat_cols:
    cats = pd.Index(
        pd.unique(pd.concat([train[c], test[c]], ignore_index=True).astype(str))
    )
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

y = train[TARGET].astype(int).reset_index(drop=True)
groups = (train["Year"].astype(str) + "__" + train["Race"].astype(str)).reset_index(
    drop=True
)

feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]


def safe_name(i, c):
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
    return f"f{i:02d}_{s or 'col'}"


rename = {c: safe_name(i, c) for i, c in enumerate(feature_cols)}
cat_features = [rename[c] for c in cat_cols]

X_train = train[feature_cols].rename(columns=rename).reset_index(drop=True)
X_test = test[feature_cols].rename(columns=rename).reset_index(drop=True)

for c in X_train.columns:
    if c not in cat_features:
        X_train[c] = X_train[c].astype(np.float32)
        X_test[c] = X_test[c].astype(np.float32)

aux_targets = [f"pit_within_{h}" for h in HORIZONS] + ["laps_until_next_pit"]
aux_features = [f"aux_{c}" for c in aux_targets]
aux_kind = {c: "binary" for c in aux_targets}
aux_kind["laps_until_next_pit"] = "regression"


class ConstantBinary:
    def __init__(self, p):
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, X):
        p = np.full(len(X), self.p, dtype=np.float32)
        return np.column_stack([1.0 - p, p])


class ConstantRegressor:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=np.float32)


def make_group_splits(y_values, group_values, n_splits, seed):
    y_arr = np.asarray(y_values).astype(int)
    g_arr = np.asarray(group_values)
    n_splits = min(n_splits, pd.Series(g_arr).nunique())
    if n_splits < 2:
        raise ValueError("Need at least two race groups for grouped validation.")
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    else:
        splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(y_arr)), y_arr, g_arr))


def make_auxiliary_targets(df):
    out = pd.DataFrame(index=df.index)
    dist_series = pd.Series(MAX_LAPS_TARGET + 1, index=df.index, dtype=np.float32)
    ordered = df.sort_values(
        ["Year", "Race", "Driver", "LapNumber", ID_COL], kind="mergesort"
    )

    for _, g in ordered.groupby(["Year", "Race", "Driver"], sort=False, observed=True):
        idx = g.index.to_numpy()
        labels = g[TARGET].astype(int).to_numpy()
        dist = np.empty(len(g), dtype=np.float32)
        d = MAX_LAPS_TARGET + 1
        for i in range(len(g) - 1, -1, -1):
            if labels[i] == 1:
                d = 1
            else:
                d = min(d + 1, MAX_LAPS_TARGET + 1)
            dist[i] = d
        dist_series.loc[idx] = dist

    for h in HORIZONS:
        out[f"pit_within_{h}"] = (dist_series <= h).astype(np.int8)
    out["laps_until_next_pit"] = dist_series.astype(np.float32)
    return out


def train_binary_model(X, target, seed, n_estimators):
    target = np.asarray(target).astype(int)
    if np.unique(target).size < 2:
        return ConstantBinary(target.mean())

    pos = target.sum()
    neg = len(target) - pos
    scale_pos_weight = float(neg / max(pos, 1))

    model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=47,
        min_child_samples=90,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_row_wise=True,
    )
    model.fit(X, target, categorical_feature=cat_features)
    return model


def train_reg_model(X, target, seed):
    target = np.asarray(target).astype(np.float32)
    if float(np.std(target)) < 1e-9:
        return ConstantRegressor(target.mean())

    model = LGBMRegressor(
        objective="regression",
        n_estimators=AUX_TREES,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_row_wise=True,
    )
    model.fit(X, target, categorical_feature=cat_features)
    return model


def predict_aux(model, X, kind):
    if kind == "binary":
        return np.clip(model.predict_proba(X)[:, 1], 0.0, 1.0).astype(np.float32)
    return np.clip(model.predict(X), 1.0, MAX_LAPS_TARGET + 1).astype(np.float32)


def add_aux_features(X, aux_df):
    out = X.copy()
    for c in aux_features:
        out[c] = np.asarray(aux_df[c], dtype=np.float32)
    return out


n = len(train)
outer_splits = make_group_splits(y, groups, OUTER_SPLITS, SEED)
aux_oof = pd.DataFrame(
    np.nan, index=np.arange(n), columns=aux_features, dtype=np.float32
)
oof_pred = np.zeros(n, dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(outer_splits, 1):
    print(f"Outer fold {fold}/{len(outer_splits)}")
    aux_tr_targets = make_auxiliary_targets(train.iloc[tr_idx])

    inner_aux = pd.DataFrame(
        np.nan, index=tr_idx, columns=aux_features, dtype=np.float32
    )
    inner_splits = make_group_splits(
        y.iloc[tr_idx], groups.iloc[tr_idx], INNER_SPLITS, SEED + fold
    )

    for inner_fold, (itr_rel, iva_rel) in enumerate(inner_splits, 1):
        itr_idx = tr_idx[itr_rel]
        iva_idx = tr_idx[iva_rel]
        for target_col, feat_col in zip(aux_targets, aux_features):
            kind = aux_kind[target_col]
            if kind == "binary":
                m = train_binary_model(
                    X_train.iloc[itr_idx],
                    aux_tr_targets.loc[itr_idx, target_col],
                    SEED + 100 * fold + inner_fold,
                    AUX_TREES,
                )
            else:
                m = train_reg_model(
                    X_train.iloc[itr_idx],
                    aux_tr_targets.loc[itr_idx, target_col],
                    SEED + 100 * fold + inner_fold,
                )
            inner_aux.loc[iva_idx, feat_col] = predict_aux(
                m, X_train.iloc[iva_idx], kind
            )

    if inner_aux.isna().any().any():
        raise RuntimeError("Inner OOF auxiliary predictions contain missing values.")

    val_aux = pd.DataFrame(index=va_idx, columns=aux_features, dtype=np.float32)
    for target_col, feat_col in zip(aux_targets, aux_features):
        kind = aux_kind[target_col]
        if kind == "binary":
            m = train_binary_model(
                X_train.iloc[tr_idx],
                aux_tr_targets.loc[tr_idx, target_col],
                SEED + 10 * fold,
                AUX_TREES,
            )
        else:
            m = train_reg_model(
                X_train.iloc[tr_idx],
                aux_tr_targets.loc[tr_idx, target_col],
                SEED + 10 * fold,
            )
        preds = predict_aux(m, X_train.iloc[va_idx], kind)
        val_aux[feat_col] = preds
        aux_oof.loc[va_idx, feat_col] = preds

    X_fold_train = add_aux_features(X_train.iloc[tr_idx], inner_aux)
    X_fold_val = add_aux_features(X_train.iloc[va_idx], val_aux)

    final_model = train_binary_model(
        X_fold_train, y.iloc[tr_idx], SEED + fold, FINAL_TREES
    )
    oof_pred[va_idx] = final_model.predict_proba(X_fold_val)[:, 1].astype(np.float32)

    fold_auc = roc_auc_score(y.iloc[va_idx], oof_pred[va_idx])
    fold_scores.append(float(fold_auc))
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

if aux_oof.isna().any().any():
    raise RuntimeError("Outer OOF auxiliary predictions contain missing values.")

cv_auc = roc_auc_score(y, oof_pred)
print(f"5-fold race-group CV ROC AUC: {cv_auc:.6f}")

full_aux_targets = make_auxiliary_targets(train)
test_aux = pd.DataFrame(
    index=np.arange(len(test)), columns=aux_features, dtype=np.float32
)

for target_col, feat_col in zip(aux_targets, aux_features):
    kind = aux_kind[target_col]
    print(f"Training full auxiliary model for {target_col}")
    if kind == "binary":
        m = train_binary_model(
            X_train, full_aux_targets[target_col], SEED + 900, AUX_TREES
        )
    else:
        m = train_reg_model(X_train, full_aux_targets[target_col], SEED + 900)
    test_aux[feat_col] = predict_aux(m, X_test, kind)

X_full_train = add_aux_features(X_train, aux_oof)
X_full_test = add_aux_features(X_test, test_aux)

full_model = train_binary_model(X_full_train, y, SEED + 999, FULL_FINAL_TREES)
test_pred = np.clip(full_model.predict_proba(X_full_test)[:, 1], 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv("./working/submission.csv", index=False)
submission.to_csv("./working/test_predictions.csv.gz", index=False, compression="gzip")

pd.DataFrame(
    {
        "row": np.arange(n),
        "target": y.astype(int).to_numpy(),
        "prediction": oof_pred,
    }
).to_csv("./working/oof_predictions.csv.gz", index=False, compression="gzip")

result = {
    "metric": "roc_auc",
    "validation": f"{len(outer_splits)}-fold race-event grouped CV",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
    "research_hypotheses_llm_claimed_used": ["000321"],
    "submission_path": "./working/submission.csv",
    "oof_path": "./working/oof_predictions.csv.gz",
    "test_predictions_path": "./working/test_predictions.csv.gz",
}
print(json.dumps(result))
