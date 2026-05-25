import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier, CatBoostRanker, Pool

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
features = [c for c in train.columns if c not in [target_col, id_col]]
cat_features = [c for c in features if train[c].dtype == "object"]

for c in cat_features:
    train[c] = train[c].astype(str).fillna("missing")
    test[c] = test[c].astype(str).fillna("missing")

for c in features:
    if c not in cat_features:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

event_group = train["Year"].astype(str) + "_" + train["Race"].astype(str)
test_event_group = test["Year"].astype(str) + "_" + test["Race"].astype(str)

rank_group = (
    train["Year"].astype(str)
    + "_"
    + train["Race"].astype(str)
    + "_lap"
    + train["LapNumber"].astype(str)
)
test_rank_group = (
    test["Year"].astype(str)
    + "_"
    + test["Race"].astype(str)
    + "_lap"
    + test["LapNumber"].astype(str)
)

X = train[features]
X_test = test[features]

n_splits = 5
gkf = GroupKFold(n_splits=n_splits)

oof_cls = np.zeros(len(train))
oof_rank = np.zeros(len(train))
test_cls_folds = []
test_rank_folds = []

cat_idx = [features.index(c) for c in cat_features]

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=event_group), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    cls_train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    cls_valid_pool = Pool(X_va, y_va, cat_features=cat_idx)
    cls_test_pool = Pool(X_test, cat_features=cat_idx)

    cls = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=8,
        random_seed=2026 + fold,
        auto_class_weights="Balanced",
        allow_writing_files=False,
        verbose=False,
        early_stopping_rounds=80,
    )
    cls.fit(cls_train_pool, eval_set=cls_valid_pool, use_best_model=True)

    oof_cls[va_idx] = cls.predict_proba(cls_valid_pool)[:, 1]
    test_cls_folds.append(cls.predict_proba(cls_test_pool)[:, 1])

    rank_train = X.iloc[tr_idx].copy()
    rank_train["__target__"] = y_tr
    rank_train["__group__"] = rank_group.iloc[tr_idx].values
    rank_train = rank_train.sort_values(
        ["__group__", "LapNumber", "Position"]
    ).reset_index(drop=True)

    rank_valid = X.iloc[va_idx].copy()
    rank_valid["__target__"] = y_va
    rank_valid["__group__"] = rank_group.iloc[va_idx].values
    rank_valid["__orig_idx__"] = va_idx
    rank_valid = rank_valid.sort_values(
        ["__group__", "LapNumber", "Position"]
    ).reset_index(drop=True)

    rank_test = X_test.copy()
    rank_test["__group__"] = test_rank_group.values
    rank_test["__orig_idx__"] = np.arange(len(test))
    rank_test = rank_test.sort_values(
        ["__group__", "LapNumber", "Position"]
    ).reset_index(drop=True)

    rank_train_pool = Pool(
        rank_train[features],
        rank_train["__target__"].values,
        group_id=rank_train["__group__"].values,
        cat_features=cat_idx,
    )
    rank_valid_pool = Pool(
        rank_valid[features],
        rank_valid["__target__"].values,
        group_id=rank_valid["__group__"].values,
        cat_features=cat_idx,
    )
    rank_test_pool = Pool(
        rank_test[features],
        group_id=rank_test["__group__"].values,
        cat_features=cat_idx,
    )

    ranker = CatBoostRanker(
        loss_function="YetiRank",
        iterations=700,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=10,
        random_seed=3026 + fold,
        allow_writing_files=False,
        verbose=False,
    )
    ranker.fit(rank_train_pool, eval_set=rank_valid_pool, use_best_model=False)

    valid_rank_pred_sorted = ranker.predict(rank_valid_pool)
    tmp_valid = pd.DataFrame(
        {"idx": rank_valid["__orig_idx__"].values, "pred": valid_rank_pred_sorted}
    )
    oof_rank[tmp_valid["idx"].values] = tmp_valid["pred"].values

    test_rank_pred_sorted = ranker.predict(rank_test_pool)
    tmp_test = pd.DataFrame(
        {"idx": rank_test["__orig_idx__"].values, "pred": test_rank_pred_sorted}
    ).sort_values("idx")
    test_rank_folds.append(tmp_test["pred"].values)

    fold_stack_raw = 0.5 * oof_cls[va_idx] + 0.5 * (
        1.0 / (1.0 + np.exp(-np.clip(oof_rank[va_idx], -30, 30)))
    )
    print(f"Fold {fold} classifier AUC: {roc_auc_score(y_va, oof_cls[va_idx]):.6f}")
    print(f"Fold {fold} rough stacked AUC: {roc_auc_score(y_va, fold_stack_raw):.6f}")

stack_X = np.column_stack([oof_cls, oof_rank])
stacker = LogisticRegression(
    C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"
)
stacker.fit(stack_X, y)
oof_stack = stacker.predict_proba(stack_X)[:, 1]

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(oof_stack, y)
oof_pred = calibrator.predict(oof_stack)

auc = roc_auc_score(y, oof_pred)
print(f"OOF ROC AUC: {auc:.6f}")

test_cls = np.mean(test_cls_folds, axis=0)
test_rank = np.mean(test_rank_folds, axis=0)
test_stack = stacker.predict_proba(np.column_stack([test_cls, test_rank]))[:, 1]
test_pred = calibrator.predict(test_stack)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample[[id_col]].copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_out = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
)
oof_out.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_out = sample[[id_col]].copy()
test_out[target_col] = test_pred
test_out.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "oof_roc_auc": float(auc),
    "research_hypotheses_llm_claimed_used": ["000816"],
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
print(json.dumps(result, indent=2))
