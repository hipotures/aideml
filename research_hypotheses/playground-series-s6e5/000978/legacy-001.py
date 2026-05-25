import os
import json
import gc
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None
from sklearn.model_selection import GroupKFold

from catboost import CatBoostClassifier, CatBoostRanker, Pool

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
BLEND_CLS_WEIGHT = 0.70

os.makedirs("./working", exist_ok=True)

train = pd.read_csv("./input/train.csv.gz")
test = pd.read_csv("./input/test.csv.gz")
sample = pd.read_csv("./input/sample_submission.csv.gz")

target_col = "PitNextLap"
id_col = "id"
features = [c for c in test.columns if c != id_col]
y = train[target_col].astype(int).values

cat_cols = [
    c for c in features if train[c].dtype == "object" or test[c].dtype == "object"
]
num_cols = [c for c in features if c not in cat_cols]

for c in cat_cols:
    train[c] = train[c].astype(str).fillna("__NA__")
    test[c] = test[c].astype(str).fillna("__NA__")

for c in num_cols:
    med = train[c].replace([np.inf, -np.inf], np.nan).median()
    train[c] = train[c].replace([np.inf, -np.inf], np.nan).fillna(med)
    test[c] = test[c].replace([np.inf, -np.inf], np.nan).fillna(med)


def race_year_group(df):
    return (df["Year"].astype(str) + "|" + df["Race"].astype(str)).values


def query_group(df):
    return (
        df["Year"].astype(str)
        + "|"
        + df["Race"].astype(str)
        + "|"
        + df["LapNumber"].astype(str)
    ).values


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def make_rank_pool(df, labels):
    qid = query_group(df)
    order = np.argsort(qid, kind="mergesort")
    return Pool(
        df.iloc[order][features],
        label=np.asarray(labels)[order],
        cat_features=cat_cols,
        group_id=qid[order],
    )


cv_groups = race_year_group(train)

if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    split_iter = splitter.split(np.zeros(len(train)), y, groups=cv_groups)
else:
    splitter = GroupKFold(n_splits=N_SPLITS)
    split_iter = splitter.split(np.zeros(len(train)), y, groups=cv_groups)

oof_cls = np.zeros(len(train), dtype=float)
oof_rank = np.zeros(len(train), dtype=float)
test_cls = np.zeros(len(test), dtype=float)
test_rank = np.zeros(len(test), dtype=float)

test_pool = Pool(test[features], cat_features=cat_cols)

for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
    x_tr = train.iloc[tr_idx]
    x_va = train.iloc[va_idx]
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    train_pool = Pool(x_tr[features], y_tr, cat_features=cat_cols)
    valid_pool = Pool(x_va[features], y_va, cat_features=cat_cols)

    clf = CatBoostClassifier(
        iterations=700,
        learning_rate=0.055,
        depth=6,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        bootstrap_type="Bernoulli",
        subsample=0.85,
        random_strength=1.0,
        random_seed=SEED + fold,
        od_type="Iter",
        od_wait=80,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    clf.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    oof_cls[va_idx] = clf.predict_proba(valid_pool)[:, 1]
    test_cls += clf.predict_proba(test_pool)[:, 1] / N_SPLITS

    rank_train_pool = make_rank_pool(x_tr, y_tr)
    ranker = CatBoostRanker(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=8.0,
        loss_function="QuerySoftMax",
        random_seed=1000 + SEED + fold,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    ranker.fit(rank_train_pool)

    oof_rank[va_idx] = sigmoid(
        ranker.predict(Pool(x_va[features], cat_features=cat_cols))
    )
    test_rank += sigmoid(ranker.predict(test_pool)) / N_SPLITS

    fold_blend = (
        BLEND_CLS_WEIGHT * oof_cls[va_idx] + (1.0 - BLEND_CLS_WEIGHT) * oof_rank[va_idx]
    )
    print(
        f"fold {fold} auc: "
        f"classifier={roc_auc_score(y_va, oof_cls[va_idx]):.6f}, "
        f"ranker={roc_auc_score(y_va, oof_rank[va_idx]):.6f}, "
        f"blend={roc_auc_score(y_va, fold_blend):.6f}"
    )

    del clf, ranker, train_pool, valid_pool, rank_train_pool
    gc.collect()

oof_pred = BLEND_CLS_WEIGHT * oof_cls + (1.0 - BLEND_CLS_WEIGHT) * oof_rank
test_pred = BLEND_CLS_WEIGHT * test_cls + (1.0 - BLEND_CLS_WEIGHT) * test_rank
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

auc_cls = roc_auc_score(y, oof_cls)
auc_rank = roc_auc_score(y, oof_rank)
auc_blend = roc_auc_score(y, oof_pred)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv("./working/oof_predictions.csv.gz", index=False, compression="gzip")

target_sub_col = [c for c in sample.columns if c != id_col][0]
submission = sample.copy()
submission[target_sub_col] = test_pred
submission.to_csv("./working/submission.csv", index=False)
submission.to_csv("./working/test_predictions.csv.gz", index=False, compression="gzip")

review = {
    "research_hypotheses_llm_claimed_used": ["000978"],
    "metric": "roc_auc",
    "cv_auc_classifier": float(auc_cls),
    "cv_auc_ranker": float(auc_rank),
    "cv_auc_blend": float(auc_blend),
    "blend_classifier_weight": BLEND_CLS_WEIGHT,
    "submission_path": "./working/submission.csv",
}
with open("./working/result_review.json", "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review, indent=2))
