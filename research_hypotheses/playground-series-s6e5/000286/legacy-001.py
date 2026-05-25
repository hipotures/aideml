import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

from catboost import CatBoostClassifier, CatBoostRanker, Pool

HYPOTHESIS_ID = "000286"
INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

y = train[target_col].astype(int).values
features = [c for c in train.columns if c not in [target_col, id_col]]

X = train[features].copy()
X_test = test[features].copy()

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype(str).fillna("NA")
    X_test[c] = X_test[c].astype(str).fillna("NA")

cat_idx = [X.columns.get_loc(c) for c in cat_cols]


def make_group_id(df):
    return (
        df["Race"].astype(str)
        + "_"
        + df["Year"].astype(str)
        + "_"
        + df["LapNumber"].astype(str)
    )


def rank01(a):
    return (rankdata(a, method="average") - 1) / max(len(a) - 1, 1)


group_all = make_group_id(train)
test_group = make_group_id(test)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_cls = np.zeros(len(train))
oof_rank = np.zeros(len(train))
test_cls = np.zeros(len(test))
test_rank = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=8,
        random_seed=1000 + fold,
        auto_class_weights="Balanced",
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
        thread_count=max(1, min(8, os.cpu_count() or 1)),
    )
    clf.fit(
        Pool(X_tr, y_tr, cat_features=cat_idx),
        eval_set=Pool(X_va, y_va, cat_features=cat_idx),
        use_best_model=True,
    )
    oof_cls[va_idx] = clf.predict_proba(Pool(X_va, cat_features=cat_idx))[:, 1]
    test_cls += (
        clf.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1] / skf.n_splits
    )

    rank_tr = train.iloc[tr_idx].copy()
    rank_va = train.iloc[va_idx].copy()
    rank_tr["_group"] = make_group_id(rank_tr)
    rank_va["_group"] = make_group_id(rank_va)

    tr_order = rank_tr.sort_values("_group").index
    va_order = rank_va.sort_values("_group").index

    Xr_tr = X.loc[tr_order].copy()
    yr_tr = train.loc[tr_order, target_col].astype(float).values
    gr_tr = make_group_id(train.loc[tr_order]).values

    Xr_va = X.loc[va_order].copy()
    yr_va = train.loc[va_order, target_col].astype(float).values
    gr_va = make_group_id(train.loc[va_order]).values

    ranker = CatBoostRanker(
        loss_function="QueryCrossEntropy",
        eval_metric="QueryAUC",
        iterations=700,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=8,
        random_seed=2000 + fold,
        od_type="Iter",
        od_wait=80,
        verbose=False,
        allow_writing_files=False,
        thread_count=max(1, min(8, os.cpu_count() or 1)),
    )
    ranker.fit(
        Pool(Xr_tr, yr_tr, group_id=gr_tr, cat_features=cat_idx),
        eval_set=Pool(Xr_va, yr_va, group_id=gr_va, cat_features=cat_idx),
        use_best_model=True,
    )

    va_pred_sorted = ranker.predict(Pool(Xr_va, group_id=gr_va, cat_features=cat_idx))
    (
        oof_rank[
            pd.Index(va_order).map(lambda idx: np.where(va_order == idx)[0][0]).values
        ]
        if False
        else None
    )
    oof_rank[va_order] = va_pred_sorted

    test_rank_df = test.copy()
    test_rank_df["_group"] = test_group.values
    test_order = test_rank_df.sort_values("_group").index
    Xt_sorted = X_test.loc[test_order].copy()
    gt_sorted = test_group.loc[test_order].values
    pred_sorted = ranker.predict(
        Pool(Xt_sorted, group_id=gt_sorted, cat_features=cat_idx)
    )
    fold_test_rank = np.zeros(len(test))
    fold_test_rank[test_order] = pred_sorted
    test_rank += fold_test_rank / skf.n_splits

    fold_blend = 0.65 * rank01(oof_cls[va_idx]) + 0.35 * rank01(oof_rank[va_idx])
    print(f"fold {fold} rank-average ROC AUC: {roc_auc_score(y_va, fold_blend):.6f}")

oof_blend = 0.65 * rank01(oof_cls) + 0.35 * rank01(oof_rank)
test_blend = 0.65 * rank01(test_cls) + 0.35 * rank01(test_rank)

cv_auc = roc_auc_score(y, oof_blend)
print(f"CV ROC AUC: {cv_auc:.6f}")
print({"research_hypotheses_llm_claimed_used": [HYPOTHESIS_ID]})

submission = sample[[id_col]].copy()
submission[target_col] = test_blend
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_blend,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        id_col: sample[id_col].values,
        target_col: test_blend,
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
