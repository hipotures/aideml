import os
import json
import gc
import warnings
import numpy as np
import pandas as pd

from pandas.api.types import CategoricalDtype
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

train["RaceYear"] = train["Race"].astype(str) + "_" + train["Year"].astype(str)
test["RaceYear"] = test["Race"].astype(str) + "_" + test["Year"].astype(str)

y = train[target_col].astype(int).to_numpy()
global_mean = float(train[target_col].mean())

cat_cols = ["Driver", "Race", "Compound", "RaceYear"]
for c in cat_cols:
    cats = (
        pd.concat([train[c], test[c]], axis=0, ignore_index=True).astype(str).unique()
    )
    dtype = CategoricalDtype(categories=cats)
    train[c] = train[c].astype(str).astype(dtype)
    test[c] = test[c].astype(str).astype(dtype)

key_defs = {
    "Driver": ["Driver"],
    "DriverYear": ["Driver", "Year"],
    "DriverCompound": ["Driver", "Compound"],
    "DriverRace": ["Driver", "Race"],
}


def make_key(df, keys):
    if len(keys) == 1:
        return df[keys[0]].astype(str)
    return df[keys].astype(str).agg("__".join, axis=1)


def add_count_frequency_features(tr, te, key_defs):
    n_all = len(tr) + len(te)
    for name, keys in key_defs.items():
        all_keys = make_key(
            pd.concat([tr[keys], te[keys]], axis=0, ignore_index=True), keys
        )
        counts = all_keys.value_counts(sort=False)

        tr_key = make_key(tr, keys)
        te_key = make_key(te, keys)

        tr[f"{name}_count"] = tr_key.map(counts).fillna(0).astype(float)
        te[f"{name}_count"] = te_key.map(counts).fillna(0).astype(float)
        tr[f"{name}_freq"] = tr[f"{name}_count"] / n_all
        te[f"{name}_freq"] = te[f"{name}_count"] / n_all


add_count_frequency_features(train, test, key_defs)

train_keys = {name: make_key(train, keys) for name, keys in key_defs.items()}
test_keys = {name: make_key(test, keys) for name, keys in key_defs.items()}


def fit_target_encoding(key_values, targets, smoothing=30.0, min_count=3):
    stats = (
        pd.DataFrame({"key": np.asarray(key_values), "target": targets})
        .groupby("key", sort=False)["target"]
        .agg(["sum", "count"])
    )
    enc = (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)
    enc = enc.where(stats["count"] >= min_count, global_mean)
    return enc


def map_target_encoding(key_values, enc):
    return key_values.map(enc).fillna(global_mean).astype(float).to_numpy()


def add_nested_target_encodings(X_tr, X_va, X_te, tr_idx, va_idx, groups):
    outer_groups = groups[tr_idx]
    inner_splits = min(5, len(np.unique(outer_groups)))

    for name in key_defs:
        col = f"{name}_te"
        tr_encoded = np.full(len(tr_idx), global_mean, dtype=float)

        if inner_splits >= 2:
            inner_cv = GroupKFold(n_splits=inner_splits)
            local_index = np.arange(len(tr_idx))
            for inner_fit_pos, inner_oof_pos in inner_cv.split(
                local_index, y[tr_idx], outer_groups
            ):
                fit_idx = tr_idx[inner_fit_pos]
                oof_idx = tr_idx[inner_oof_pos]
                enc = fit_target_encoding(train_keys[name].iloc[fit_idx], y[fit_idx])
                tr_encoded[inner_oof_pos] = map_target_encoding(
                    train_keys[name].iloc[oof_idx], enc
                )

        outer_enc = fit_target_encoding(train_keys[name].iloc[tr_idx], y[tr_idx])
        X_tr[col] = tr_encoded
        X_va[col] = map_target_encoding(train_keys[name].iloc[va_idx], outer_enc)
        X_te[col] = map_target_encoding(test_keys[name], outer_enc)

    return X_tr, X_va, X_te


raw_cols = [c for c in test.columns if c != id_col]
num_cols = [c for c in raw_cols if c not in cat_cols]
freq_cols = []
for name in key_defs:
    freq_cols.extend([f"{name}_count", f"{name}_freq"])

base_feature_cols = num_cols + cat_cols + freq_cols
te_cols = [f"{name}_te" for name in key_defs]
te_feature_cols = base_feature_cols + te_cols

X_base = train[base_feature_cols].copy()
X_test_base = test[base_feature_cols].copy()

groups = train["RaceYear"].astype(str).to_numpy()
outer_cv = GroupKFold(n_splits=5)

model_params = dict(
    objective="binary",
    learning_rate=0.045,
    n_estimators=1200,
    num_leaves=48,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=2.0,
    random_state=73,
    n_jobs=-1,
    verbose=-1,
)

freq_oof = np.zeros(len(train), dtype=float)
te_oof = np.zeros(len(train), dtype=float)
test_preds = np.zeros(len(test), dtype=float)
freq_fold_aucs = []
te_fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X_base, y, groups), 1):
    X_tr_base = X_base.iloc[tr_idx].copy()
    X_va_base = X_base.iloc[va_idx].copy()

    freq_model = LGBMClassifier(**model_params)
    freq_model.fit(
        X_tr_base,
        y[tr_idx],
        eval_set=[(X_va_base, y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[],
    )
    freq_pred = freq_model.predict_proba(X_va_base)[:, 1]
    freq_oof[va_idx] = freq_pred
    freq_auc = roc_auc_score(y[va_idx], freq_pred)
    freq_fold_aucs.append(freq_auc)

    X_tr_te = X_tr_base.copy()
    X_va_te = X_va_base.copy()
    X_test_te = X_test_base.copy()
    X_tr_te, X_va_te, X_test_te = add_nested_target_encodings(
        X_tr_te, X_va_te, X_test_te, tr_idx, va_idx, groups
    )

    te_model = LGBMClassifier(**model_params)
    te_model.fit(
        X_tr_te[te_feature_cols],
        y[tr_idx],
        eval_set=[(X_va_te[te_feature_cols], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[],
    )
    te_pred = te_model.predict_proba(X_va_te[te_feature_cols])[:, 1]
    te_oof[va_idx] = te_pred
    test_preds += te_model.predict_proba(X_test_te[te_feature_cols])[:, 1] / 5.0
    te_auc = roc_auc_score(y[va_idx], te_pred)
    te_fold_aucs.append(te_auc)

    print(
        f"Fold {fold} ROC AUC - freq_only: {freq_auc:.6f}, "
        f"fold_safe_driver_te: {te_auc:.6f}"
    )

    del X_tr_base, X_va_base, X_tr_te, X_va_te, X_test_te, freq_model, te_model
    gc.collect()

freq_mean_auc = float(np.mean(freq_fold_aucs))
freq_oof_auc = float(roc_auc_score(y, freq_oof))
te_mean_auc = float(np.mean(te_fold_aucs))
te_oof_auc = float(roc_auc_score(y, te_oof))

print(f"Frequency-only Mean CV ROC AUC: {freq_mean_auc:.6f}")
print(f"Frequency-only OOF ROC AUC: {freq_oof_auc:.6f}")
print(f"Fold-safe driver TE Mean CV ROC AUC: {te_mean_auc:.6f}")
print(f"Fold-safe driver TE OOF ROC AUC: {te_oof_auc:.6f}")

submission = sample.copy()
submission[target_col] = np.clip(test_preds, 0.0, 1.0)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": te_oof,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

test_pred_df = submission[[id_col, target_col]].copy()
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "mean_cv_roc_auc": te_mean_auc,
    "oof_roc_auc": te_oof_auc,
    "freq_only_mean_cv_roc_auc": freq_mean_auc,
    "freq_only_oof_roc_auc": freq_oof_auc,
    "research_hypotheses_llm_claimed_used": ["000073"],
}
print(json.dumps(result))
