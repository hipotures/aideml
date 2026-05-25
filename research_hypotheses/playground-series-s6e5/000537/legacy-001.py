import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, CatBoostRanker, Pool

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
RANDOM_STATE = 537

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
features = [c for c in train.columns if c not in [TARGET, ID_COL]]
cat_cols = [c for c in features if train[c].dtype == "object"]
cat_idx = [features.index(c) for c in cat_cols]

X = train[features].copy()
X_test = test[features].copy()

for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

group_cols = ["Year", "Race", "LapNumber"]
train_group = train[group_cols].astype(str).agg("_".join, axis=1)
test_group = test[group_cols].astype(str).agg("_".join, axis=1)

oof_cls = np.zeros(len(train))
oof_rank = np.zeros(len(train))
test_cls_folds = []
test_rank_folds = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    cls_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=1200,
        learning_rate=0.035,
        depth=7,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        auto_class_weights="Balanced",
        early_stopping_rounds=100,
        random_seed=RANDOM_STATE + fold,
        verbose=False,
        allow_writing_files=False,
    )

    cls_model.fit(
        Pool(X_tr, y_tr, cat_features=cat_idx),
        eval_set=Pool(X_va, y_va, cat_features=cat_idx),
        use_best_model=True,
    )

    oof_cls[va_idx] = cls_model.predict_proba(X_va)[:, 1]
    test_cls_folds.append(cls_model.predict_proba(X_test)[:, 1])

    tr_order = np.argsort(train_group.iloc[tr_idx].values, kind="stable")
    va_order = np.argsort(train_group.iloc[va_idx].values, kind="stable")

    tr_sorted_idx = tr_idx[tr_order]
    va_sorted_idx = va_idx[va_order]

    rank_train_pool = Pool(
        X.iloc[tr_sorted_idx],
        y[tr_sorted_idx],
        cat_features=cat_idx,
        group_id=train_group.iloc[tr_sorted_idx].values,
    )
    rank_valid_pool = Pool(
        X.iloc[va_sorted_idx],
        y[va_sorted_idx],
        cat_features=cat_idx,
        group_id=train_group.iloc[va_sorted_idx].values,
    )

    rank_model = CatBoostRanker(
        loss_function="PairLogit",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=0.7,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        early_stopping_rounds=100,
        random_seed=RANDOM_STATE + 100 + fold,
        verbose=False,
        allow_writing_files=False,
    )

    rank_model.fit(rank_train_pool, eval_set=rank_valid_pool, use_best_model=True)

    rank_va_scores = rank_model.predict(X.iloc[va_sorted_idx])
    oof_rank[va_sorted_idx] = rank_va_scores
    test_rank_folds.append(rank_model.predict(X_test))

    fold_cls_auc = roc_auc_score(y_va, oof_cls[va_idx])
    fold_rank_auc = roc_auc_score(y_va, oof_rank[va_idx])
    print(
        f"fold {fold}: classifier_auc={fold_cls_auc:.6f}, ranker_auc={fold_rank_auc:.6f}"
    )


def minmax_scale(a):
    a = np.asarray(a, dtype=float)
    lo, hi = np.min(a), np.max(a)
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


oof_rank_s = minmax_scale(oof_rank)
test_cls = np.mean(test_cls_folds, axis=0)
test_rank = minmax_scale(np.mean(test_rank_folds, axis=0))

weights = np.linspace(0.0, 0.45, 46)
best_w, best_auc = 0.0, -1.0
for w in weights:
    pred = (1.0 - w) * oof_cls + w * oof_rank_s
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_auc = auc
        best_w = float(w)

final_oof = (1.0 - best_w) * oof_cls + best_w * oof_rank_s
final_test = (1.0 - best_w) * test_cls + best_w * test_rank
final_test = np.clip(final_test, 0.0, 1.0)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": final_oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample[[ID_COL]].copy()
test_pred_df[TARGET] = final_test
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
test_pred_df.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

result = {
    "metric": "roc_auc",
    "cv_auc": float(roc_auc_score(y, final_oof)),
    "classifier_oof_auc": float(roc_auc_score(y, oof_cls)),
    "ranker_oof_auc": float(roc_auc_score(y, oof_rank_s)),
    "ranker_blend_weight": best_w,
    "research_hypotheses_llm_claimed_used": ["000537"],
}
print(json.dumps(result, indent=2))
