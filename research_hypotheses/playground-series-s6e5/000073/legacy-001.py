import os
import json
import warnings
import numpy as np
import pandas as pd

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

y = train[target_col].astype(int).values
global_mean = float(train[target_col].mean())

train["RaceYear"] = train["Race"].astype(str) + "_" + train["Year"].astype(str)
test["RaceYear"] = test["Race"].astype(str) + "_" + test["Year"].astype(str)

base_cols = [c for c in test.columns if c != id_col]
cat_cols = ["Driver", "Race", "Compound", "RaceYear"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

key_defs = {
    "Driver": ["Driver"],
    "DriverYear": ["Driver", "Year"],
    "DriverCompound": ["Driver", "Compound"],
    "DriverRace": ["Driver", "Race"],
}


def add_count_frequency_features(tr, te, key_defs):
    n_train = len(tr)
    all_df = pd.concat([tr.drop(columns=[target_col]), te], axis=0, ignore_index=True)
    for name, keys in key_defs.items():
        key_frame = all_df[keys].astype(str)
        key_series = key_frame.agg("__".join, axis=1)
        vc = key_series.value_counts()
        tr_key = tr[keys].astype(str).agg("__".join, axis=1)
        te_key = te[keys].astype(str).agg("__".join, axis=1)
        tr[f"{name}_count"] = tr_key.map(vc).astype(float)
        te[f"{name}_count"] = te_key.map(vc).astype(float)
        tr[f"{name}_freq"] = tr[f"{name}_count"] / len(all_df)
        te[f"{name}_freq"] = te[f"{name}_count"] / len(all_df)


add_count_frequency_features(train, test, key_defs)


def smoothed_map(stats, smoothing=30.0):
    return (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)


def make_oof_target_encoding(
    tr, te, keys, groups, n_splits=5, smoothing=30.0, min_count=3
):
    oof = np.full(len(tr), global_mean, dtype=float)
    test_accum = np.zeros(len(te), dtype=float)
    gkf = GroupKFold(n_splits=n_splits)

    for train_idx, valid_idx in gkf.split(tr, y, groups):
        fold_tr = tr.iloc[train_idx]
        fold_va = tr.iloc[valid_idx]

        stats = fold_tr.groupby(keys, observed=True)[target_col].agg(["sum", "count"])
        enc = smoothed_map(stats, smoothing)
        enc = enc.where(stats["count"] >= min_count, global_mean)

        va_keyed = fold_va.set_index(keys)
        oof[valid_idx] = (
            va_keyed.index.map(enc).fillna(global_mean).astype(float).values
        )

        te_keyed = te.set_index(keys)
        test_accum += (
            te_keyed.index.map(enc).fillna(global_mean).astype(float).values / n_splits
        )

    full_stats = tr.groupby(keys, observed=True)[target_col].agg(["sum", "count"])
    full_enc = smoothed_map(full_stats, smoothing)
    full_enc = full_enc.where(full_stats["count"] >= min_count, global_mean)

    return oof, test_accum, full_enc


groups = train["RaceYear"].astype(str).values
te_cols = []

for name, keys in key_defs.items():
    col = f"{name}_te"
    tr_oof, te_pred, _ = make_oof_target_encoding(train, test, keys, groups)
    train[col] = tr_oof
    test[col] = te_pred
    te_cols.append(col)

num_cols = [c for c in base_cols if c not in cat_cols and c != id_col]
freq_cols = [c for c in train.columns if c.endswith("_count") or c.endswith("_freq")]
feature_cols = num_cols + cat_cols + freq_cols + te_cols

X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

fold_preds = np.zeros(len(train), dtype=float)
test_preds = np.zeros(len(test), dtype=float)
auc_scores = []

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

gkf = GroupKFold(n_splits=5)
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
    model = LGBMClassifier(**model_params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[],
    )

    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    fold_preds[va_idx] = va_pred
    test_preds += model.predict_proba(X_test)[:, 1] / 5.0

    auc = roc_auc_score(y[va_idx], va_pred)
    auc_scores.append(auc)
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

mean_auc = float(np.mean(auc_scores))
overall_auc = float(roc_auc_score(y, fold_preds))
print(f"Mean CV ROC AUC: {mean_auc:.6f}")
print(f"OOF ROC AUC: {overall_auc:.6f}")

submission = sample.copy()
submission[target_col] = test_preds
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": fold_preds,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = submission[[id_col, target_col]].copy()
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "mean_cv_roc_auc": mean_auc,
    "oof_roc_auc": overall_auc,
    "research_hypotheses_llm_claimed_used": ["000073"],
}
print(json.dumps(result))
