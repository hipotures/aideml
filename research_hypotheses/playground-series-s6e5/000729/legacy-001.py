import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]


def safe_name(name):
    out = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    return out if out else "feature"


safe_map = {}
used = set()
for c in feature_cols:
    base = safe_name(c)
    name = base
    k = 1
    while name in used:
        k += 1
        name = f"{base}_{k}"
    safe_map[c] = name
    used.add(name)

X_train_raw = train[feature_cols].rename(columns=safe_map)
X_test_raw = test[feature_cols].rename(columns=safe_map)
all_X = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

cat_cols = all_X.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    all_X[c] = all_X[c].astype("category")

X = all_X.iloc[: len(train)].reset_index(drop=True)
X_test = all_X.iloc[len(train) :].reset_index(drop=True)
y = train[TARGET].astype(int).values

groups = (
    train["Year"].astype(str).fillna("NA")
    + "_"
    + train["Race"].astype(str).fillna("NA")
).values

n_jobs = max(1, min(os.cpu_count() or 1, 8))

domain_y = np.r_[np.zeros(len(X), dtype=int), np.ones(len(X_test), dtype=int)]
domain_model = LGBMClassifier(
    objective="binary",
    n_estimators=250,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=300,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=3.0,
    random_state=SEED,
    n_jobs=n_jobs,
    verbosity=-1,
)
domain_model.fit(all_X, domain_y, categorical_feature=cat_cols)

p_test_given_x = domain_model.predict_proba(X)[:, 1]
p_test_given_x = np.clip(p_test_given_x, 1e-4, 1 - 1e-4)
density_ratio = (p_test_given_x / (1.0 - p_test_given_x)) * (len(X) / len(X_test))
importance_w = np.clip(density_ratio, 0.2, 5.0)
importance_w = importance_w / np.mean(importance_w)

try:
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y, groups))
except Exception:
    from sklearn.model_selection import GroupKFold

    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(X, y, groups))

base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=900,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=n_jobs,
    verbosity=-1,
)

oof = np.zeros(len(X), dtype=float)
fold_weighted_auc = []
fold_auc = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = LGBMClassifier(**base_params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        sample_weight=importance_w[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_sample_weight=[importance_w[va_idx]],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )

    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred

    w_auc = roc_auc_score(y[va_idx], pred, sample_weight=importance_w[va_idx])
    auc = roc_auc_score(y[va_idx], pred)
    fold_weighted_auc.append(w_auc)
    fold_auc.append(auc)
    best_iters.append(
        getattr(model, "best_iteration_", None) or base_params["n_estimators"]
    )

    print(
        f"fold={fold} weighted_auc={w_auc:.6f} auc={auc:.6f} best_iter={best_iters[-1]}"
    )

weighted_cv_auc = roc_auc_score(y, oof, sample_weight=importance_w)
plain_cv_auc = roc_auc_score(y, oof)
final_estimators = int(np.clip(np.mean(best_iters), 100, base_params["n_estimators"]))

final_params = base_params.copy()
final_params["n_estimators"] = final_estimators
final_model = LGBMClassifier(**final_params)
final_model.fit(
    X,
    y,
    sample_weight=importance_w,
    categorical_feature=cat_cols,
)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
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

test_predictions = sample.copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "research_hypotheses_llm_claimed_used": ["000729"],
    "metric": "importance_weighted_grouped_5fold_roc_auc",
    "cv_weighted_auc": float(weighted_cv_auc),
    "cv_unweighted_auc": float(plain_cv_auc),
    "fold_weighted_auc": [float(v) for v in fold_weighted_auc],
    "fold_unweighted_auc": [float(v) for v in fold_auc],
    "final_n_estimators": int(final_estimators),
    "importance_weight_min": float(np.min(importance_w)),
    "importance_weight_mean": float(np.mean(importance_w)),
    "importance_weight_max": float(np.max(importance_w)),
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
print(json.dumps(result, indent=2))
