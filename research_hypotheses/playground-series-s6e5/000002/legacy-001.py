import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 2026
N_SPLITS = 5
N_JOBS = min(8, os.cpu_count() or 1)

train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
feature_cols = [
    c for c in train.columns if c not in [TARGET, ID_COL] and c in test.columns
]

X = train[feature_cols].copy()
X_test = test[feature_cols].copy()

cat_cols = [
    c for c in feature_cols if X[c].dtype.name in ("object", "category", "string")
]

for c in cat_cols:
    tr = X[c].astype("string").fillna("__MISSING__")
    te = X_test[c].astype("string").fillna("__MISSING__")
    cats = pd.Index(pd.concat([tr, te], axis=0).unique())
    X[c] = pd.Categorical(tr, categories=cats)
    X_test[c] = pd.Categorical(te, categories=cats)

for c in feature_cols:
    if c not in cat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")
        if X[c].isna().any() or X_test[c].isna().any():
            med = X[c].median()
            X[c] = X[c].fillna(med)
            X_test[c] = X_test[c].fillna(med)
        if pd.api.types.is_float_dtype(X[c]):
            X[c] = X[c].astype("float32")
            X_test[c] = X_test[c].astype("float32")


def as_group_string(s):
    return s.astype("string").fillna("__MISSING__")


def choose_cv(df, target):
    groups = None
    source = "StratifiedKFold"

    if "Race_Year" in df.columns:
        groups = as_group_string(df["Race_Year"]).to_numpy()
        source = "Race_Year"
    elif "Race" in df.columns and "Year" in df.columns:
        groups = (
            as_group_string(df["Race"]) + "_" + as_group_string(df["Year"])
        ).to_numpy()
        source = "Race_Year_constructed"
    elif "Race" in df.columns:
        groups = as_group_string(df["Race"]).to_numpy()
        source = "Race"

    if groups is not None and len(pd.unique(groups)) >= N_SPLITS:
        return source, GroupKFold(n_splits=N_SPLITS), groups

    return (
        "StratifiedKFold",
        StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        None,
    )


def make_model(y_fit, seed, n_estimators=1500):
    pos = max(float(np.sum(y_fit)), 1.0)
    neg = max(float(len(y_fit) - np.sum(y_fit)), 1.0)
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        scale_pos_weight=neg / pos,
        random_state=seed,
        n_jobs=N_JOBS,
        force_col_wise=True,
        verbosity=-1,
    )


def fit_model(model, X_fit, y_fit, valid=None):
    kwargs = {}
    if cat_cols:
        kwargs["categorical_feature"] = cat_cols
    if valid is not None:
        X_valid, y_valid = valid
        kwargs["eval_set"] = [(X_valid, y_valid)]
        kwargs["eval_metric"] = "auc"
        kwargs["callbacks"] = [
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(period=0),
        ]
    model.fit(X_fit, y_fit, **kwargs)
    return model


def safe_auc(y_true, pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, pred)


cv_source, splitter, groups = choose_cv(train, y)
split_iter = (
    splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
)

oof = np.zeros(len(train), dtype=np.float32)
fold_aucs = []
best_iterations = []

print(f"Feature family: base_raw_features")
print(f"Grouped validation source: {cv_source}")

for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
    model = make_model(y[tr_idx], RANDOM_STATE + fold)
    model = fit_model(
        model,
        X.iloc[tr_idx],
        y[tr_idx],
        valid=(X.iloc[va_idx], y[va_idx]),
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred.astype("float32")
    auc = safe_auc(y[va_idx], pred)
    fold_aucs.append(auc)

    best_iter = getattr(model, "best_iteration_", None)
    if isinstance(best_iter, (int, np.integer)) and best_iter > 0:
        best_iterations.append(int(best_iter))

    n_groups = len(pd.unique(groups[va_idx])) if groups is not None else 0
    print(f"Fold {fold} AUC: {auc:.6f} rows={len(va_idx)} groups={n_groups}")

fold_aucs_arr = np.array(fold_aucs, dtype=float)
group_mean_auc = float(np.nanmean(fold_aucs_arr))
group_std_auc = float(np.nanstd(fold_aucs_arr))
group_worst_auc = float(np.nanmin(fold_aucs_arr))
oof_auc = float(safe_auc(y, oof))

hold_tr, hold_va = train_test_split(
    np.arange(len(y)),
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)
hold_model = make_model(y[hold_tr], RANDOM_STATE + 100)
hold_model = fit_model(
    hold_model,
    X.iloc[hold_tr],
    y[hold_tr],
    valid=(X.iloc[hold_va], y[hold_va]),
)
hold_pred = hold_model.predict_proba(X.iloc[hold_va])[:, 1]
holdout_auc = float(safe_auc(y[hold_va], hold_pred))

pd.DataFrame(
    {
        "row": np.arange(len(y)),
        "target": y,
        "prediction": oof,
    }
).to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

pd.DataFrame(
    {
        "row": hold_va,
        "target": y[hold_va],
        "prediction": hold_pred,
    }
).to_csv(WORK_DIR / "validation_predictions.csv.gz", index=False, compression="gzip")

final_estimators = 900
if best_iterations:
    final_estimators = int(np.clip(round(np.median(best_iterations) * 1.05), 100, 1500))

final_model = make_model(y, RANDOM_STATE + 999, n_estimators=final_estimators)
final_model = fit_model(final_model, X, y)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

sample_target_col = [c for c in sample.columns if c != ID_COL][0]
submission = sample.copy()
submission[sample_target_col] = test_pred
submission.to_csv(WORK_DIR / "submission.csv", index=False)
submission[[ID_COL, sample_target_col]].to_csv(
    WORK_DIR / "test_predictions.csv.gz",
    index=False,
    compression="gzip",
)

print(f"Grouped CV mean AUC: {group_mean_auc:.6f}")
print(f"Grouped CV std AUC: {group_std_auc:.6f}")
print(f"Grouped CV worst-fold AUC: {group_worst_auc:.6f}")
print(f"Grouped CV overall OOF AUC: {oof_auc:.6f}")
print(f"Random holdout AUC: {holdout_auc:.6f}")
print(f"Final model estimators: {final_estimators}")
print(f"Saved submission to {WORK_DIR / 'submission.csv'}")

review = {
    "research_hypotheses_llm_claimed_used": ["000002"],
    "feature_family": "base_raw_features",
    "validation_group_source": cv_source,
    "group_cv_mean_auc": group_mean_auc,
    "group_cv_std_auc": group_std_auc,
    "group_cv_worst_fold_auc": group_worst_auc,
    "group_cv_oof_auc": oof_auc,
    "random_holdout_auc": holdout_auc,
    "final_estimators": final_estimators,
}
print(json.dumps(review, sort_keys=True))
