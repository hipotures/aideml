import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP = True
except Exception:
    HAS_STRATIFIED_GROUP = False

from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
ROW_COL = "row"

os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

test = test.set_index(ID_COL).loc[sample[ID_COL].values].reset_index()
train[ROW_COL] = np.arange(len(train))

feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL, ROW_COL]]
cat_cols = [c for c in feature_cols if train[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]
sort_cols = [c for c in ["Year", "Race", "Driver", "LapNumber"] if c in train.columns]

for df in (train, test):
    for c in cat_cols:
        df[c] = df[c].where(df[c].notna(), "__NA__").astype(str)

train_sorted = train.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
y_sorted = train_sorted[TARGET].astype(int).values
groups = train_sorted["Race"].astype(str).values

if HAS_STRATIFIED_GROUP:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
else:
    splitter = GroupKFold(n_splits=N_SPLITS)

splits = list(splitter.split(train_sorted[feature_cols], y_sorted, groups))
cat_feature_indices = [feature_cols.index(c) for c in cat_cols]

cat_params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 600,
    "learning_rate": 0.06,
    "depth": 6,
    "l2_leaf_reg": 6.0,
    "random_seed": SEED,
    "boosting_type": "Ordered",
    "has_time": True,
    "allow_writing_files": False,
    "thread_count": max(1, os.cpu_count() or 1),
    "verbose": False,
}


def baseline_matrix(train_part, valid_part):
    x_tr = train_part[num_cols].copy()
    x_va = valid_part[num_cols].copy()
    for c in cat_cols:
        freq = train_part[c].value_counts(normalize=True)
        name = f"{c}_freq"
        x_tr[name] = train_part[c].map(freq).fillna(0).astype("float32")
        x_va[name] = valid_part[c].map(freq).fillna(0).astype("float32")
    return x_tr, x_va


cat_oof = np.zeros(len(train), dtype=np.float32)
cat_fold_aucs = []
baseline_fold_aucs = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    tr = train_sorted.iloc[tr_idx].sort_values(sort_cols, kind="mergesort")
    va = train_sorted.iloc[va_idx].sort_values(sort_cols, kind="mergesort")

    x_tr_base, x_va_base = baseline_matrix(tr, va)
    base_model = LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    base_model.fit(
        x_tr_base,
        tr[TARGET].astype(int),
        eval_set=[(x_va_base, va[TARGET].astype(int))],
        eval_metric="auc",
        callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
    )
    base_pred = base_model.predict_proba(x_va_base)[:, 1]
    base_auc = roc_auc_score(va[TARGET].astype(int), base_pred)
    baseline_fold_aucs.append(base_auc)

    train_pool = Pool(
        tr[feature_cols], tr[TARGET].astype(int), cat_features=cat_feature_indices
    )
    valid_pool = Pool(
        va[feature_cols], va[TARGET].astype(int), cat_features=cat_feature_indices
    )

    cat_model = CatBoostClassifier(**cat_params)
    cat_model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=80,
        verbose=False,
    )
    cat_pred = cat_model.predict_proba(valid_pool)[:, 1]
    cat_oof[va[ROW_COL].values] = cat_pred.astype(np.float32)
    cat_auc = roc_auc_score(va[TARGET].astype(int), cat_pred)
    cat_fold_aucs.append(cat_auc)

    best_iter = cat_model.get_best_iteration()
    if best_iter is not None:
        best_iterations.append(best_iter + 1)

    print(f"Fold {fold}: CatBoost AUC={cat_auc:.6f}, baseline GBDT AUC={base_auc:.6f}")

cat_cv_auc = roc_auc_score(train[TARGET].astype(int), cat_oof)
baseline_mean_auc = float(np.mean(baseline_fold_aucs))
cat_mean_auc = float(np.mean(cat_fold_aucs))

oof_path = os.path.join(WORKING_DIR, "oof_predictions.csv.gz")
pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": train[TARGET].astype(int).values,
        "prediction": cat_oof,
    }
).to_csv(oof_path, index=False, compression="gzip")

final_iterations = (
    int(np.median(best_iterations)) if best_iterations else cat_params["iterations"]
)
final_iterations = max(50, min(final_iterations, cat_params["iterations"]))

final_params = dict(cat_params)
final_params["iterations"] = final_iterations

full_train = train.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
full_pool = Pool(
    full_train[feature_cols],
    full_train[TARGET].astype(int),
    cat_features=cat_feature_indices,
)
test_pool = Pool(test[feature_cols], cat_features=cat_feature_indices)

final_model = CatBoostClassifier(**final_params)
final_model.fit(full_pool, verbose=False)

test_pred = final_model.predict_proba(test_pool)[:, 1]
test_pred = np.clip(test_pred, 0, 1)

submission = sample.copy()
submission[TARGET] = test_pred
submission_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

test_pred_path = os.path.join(WORKING_DIR, "test_predictions.csv.gz")
submission.to_csv(test_pred_path, index=False, compression="gzip")

result = {
    "metric": "roc_auc",
    "catboost_oof_auc": float(cat_cv_auc),
    "catboost_fold_auc_mean": cat_mean_auc,
    "catboost_fold_auc_std": float(np.std(cat_fold_aucs)),
    "baseline_gbdt_fold_auc_mean": baseline_mean_auc,
    "baseline_gbdt_fold_auc_std": float(np.std(baseline_fold_aucs)),
    "final_catboost_iterations": int(final_iterations),
    "research_hypotheses_llm_claimed_used": ["000951"],
    "submission_path": submission_path,
    "oof_predictions_path": oof_path,
    "test_predictions_path": test_pred_path,
}

print(f"CatBoost race-grouped CV ROC AUC: {cat_cv_auc:.6f}")
print(f"Baseline frequency-encoded GBDT mean ROC AUC: {baseline_mean_auc:.6f}")
print(json.dumps(result, indent=2))
