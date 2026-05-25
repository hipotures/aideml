import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

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
train_ids = train[id_col].values
test_ids = sample[id_col].values

features = [c for c in train.columns if c not in [target_col, id_col]]
X = train[features].copy()
X_test = test[features].copy()

cat_cols = [c for c in ["Driver", "Race", "Compound", "Year", "Stint"] if c in features]
for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

cat_idx = [features.index(c) for c in cat_cols]
groups = train["Race"].astype(str).values

params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 1200,
    "learning_rate": 0.045,
    "depth": 8,
    "l2_leaf_reg": 6.0,
    "random_seed": 234,
    "bootstrap_type": "Bayesian",
    "bagging_temperature": 0.6,
    "one_hot_max_size": 2,
    "max_ctr_complexity": 3,
    "auto_class_weights": "Balanced",
    "od_type": "Iter",
    "od_wait": 80,
    "allow_writing_files": False,
    "verbose": 100,
    "thread_count": max(1, os.cpu_count() or 1),
}

gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(train), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float64)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
    train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_idx)
    valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_idx)
    test_pool = Pool(X_test, cat_features=cat_idx)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    va_pred = model.predict_proba(valid_pool)[:, 1]
    fold_auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(fold_auc)
    oof[va_idx] = va_pred.astype(np.float32)

    test_pred += model.predict_proba(test_pool)[:, 1] / gkf.n_splits
    print(f"fold {fold} roc_auc: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"CV ROC AUC: {cv_auc:.6f}")
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        id_col: test_ids,
        target_col: test_pred,
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(x) for x in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000234"],
}
with open(os.path.join(WORKING_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
