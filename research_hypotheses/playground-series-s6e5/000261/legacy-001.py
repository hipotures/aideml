import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGKF = True
except Exception:
    from sklearn.model_selection import GroupKFold

    HAS_SGKF = False

from catboost import CatBoostClassifier, CatBoostRanker, Pool

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

SEED = 2026
N_SPLITS = 5
THREADS = max(1, min(8, os.cpu_count() or 1))
TARGET = "PitNextLap"
ID_COL = "id"
GROUP_COLS = ["Race", "Year", "LapNumber"]
RELATIVE_COLS = ["TyreLife", "LapTime_Delta"]


def make_group_key(df):
    return (
        df["Race"].astype(str)
        + "|"
        + df["Year"].astype(str)
        + "|"
        + df["LapNumber"].astype(str)
    )


def percentile_by_group(values, groups):
    tmp = pd.DataFrame({"value": np.asarray(values), "group": np.asarray(groups)})
    rank = tmp.groupby("group")["value"].rank(method="average").to_numpy()
    count = tmp.groupby("group")["value"].transform("count").to_numpy()
    return np.where(count > 1, (rank - 1.0) / (count - 1.0), 0.5).astype("float32")


def add_relative_group_features(df):
    df = df.copy()
    group_key = make_group_key(df)
    for col in RELATIVE_COLS:
        df[f"{col}_grp_pct"] = percentile_by_group(df[col].to_numpy(), group_key)
        med = df.groupby(group_key)[col].transform("median")
        df[f"{col}_grp_med_gap"] = (df[col] - med).astype("float32")
    for col in df.select_dtypes(include=["float64"]).columns:
        if col != TARGET:
            df[col] = df[col].astype("float32")
    return df, group_key


def make_rank_pool(X, y, idx, group_key, cat_features):
    idx = np.asarray(idx)
    keys = group_key.iloc[idx].to_numpy()
    order = np.argsort(keys, kind="mergesort")
    sorted_idx = idx[order]
    pool = Pool(
        X.iloc[sorted_idx],
        y[sorted_idx],
        group_id=group_key.iloc[sorted_idx].to_numpy(),
        cat_features=cat_features,
    )
    return pool, sorted_idx


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train_fe, train_group_key = add_relative_group_features(train)
test_fe, test_group_key = add_relative_group_features(test)

y = train_fe[TARGET].astype(int).to_numpy()
features = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
cat_cols = [c for c in features if train_fe[c].dtype == "object"]

for c in cat_cols:
    train_fe[c] = train_fe[c].astype(str).fillna("__NA__")
    test_fe[c] = test_fe[c].astype(str).fillna("__NA__")

X = train_fe[features]
X_test = test_fe[features]
cat_features = [features.index(c) for c in cat_cols]

if HAS_SGKF:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y, groups=train_group_key))
else:
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(X, y, groups=train_group_key))

oof_cls = np.zeros(len(train_fe), dtype="float32")
oof_rank = np.zeros(len(train_fe), dtype="float32")
test_cls = np.zeros(len(test_fe), dtype="float32")
test_rank = np.zeros(len(test_fe), dtype="float32")

test_pool_cls = Pool(X_test, cat_features=cat_features)
test_rank_order = np.argsort(test_group_key.to_numpy(), kind="mergesort")
test_rank_pool = Pool(
    X_test.iloc[test_rank_order],
    group_id=test_group_key.iloc[test_rank_order].to_numpy(),
    cat_features=cat_features,
)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    clf = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=6,
        random_strength=0.5,
        auto_class_weights="Balanced",
        random_seed=SEED + fold,
        thread_count=THREADS,
        allow_writing_files=False,
        verbose=False,
    )

    train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
    valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_features)

    clf.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=80,
        verbose=False,
    )

    oof_cls[va_idx] = clf.predict_proba(valid_pool)[:, 1]
    test_cls += clf.predict_proba(test_pool_cls)[:, 1].astype("float32") / N_SPLITS

    ranker = CatBoostRanker(
        loss_function="YetiRankPairwise",
        eval_metric="NDCG:top=10",
        iterations=450,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=8,
        random_strength=1.0,
        random_seed=SEED + 100 + fold,
        thread_count=THREADS,
        allow_writing_files=False,
        verbose=False,
    )

    rank_train_pool, _ = make_rank_pool(X, y, tr_idx, train_group_key, cat_features)
    rank_valid_pool, va_sorted_idx = make_rank_pool(
        X, y, va_idx, train_group_key, cat_features
    )

    ranker.fit(
        rank_train_pool,
        eval_set=rank_valid_pool,
        use_best_model=True,
        early_stopping_rounds=60,
        verbose=False,
    )

    valid_rank_raw = ranker.predict(rank_valid_pool)
    oof_rank[va_sorted_idx] = percentile_by_group(
        valid_rank_raw,
        train_group_key.iloc[va_sorted_idx].to_numpy(),
    )

    test_rank_raw_sorted = ranker.predict(test_rank_pool)
    test_rank_raw = np.empty(len(test_fe), dtype="float32")
    test_rank_raw[test_rank_order] = test_rank_raw_sorted
    test_rank += (
        percentile_by_group(test_rank_raw, test_group_key.to_numpy()) / N_SPLITS
    )

    fold_blend = 0.7 * oof_cls[va_idx] + 0.3 * oof_rank[va_idx]
    print(f"Fold {fold} ROC AUC: {roc_auc_score(y[va_idx], fold_blend):.6f}")

cls_auc = roc_auc_score(y, oof_cls)
rank_auc = roc_auc_score(y, oof_rank)

weights = np.linspace(0.0, 1.0, 21)
blend_scores = []
for w in weights:
    pred = w * oof_cls + (1.0 - w) * oof_rank
    blend_scores.append((w, roc_auc_score(y, pred)))

best_w, best_auc = max(blend_scores, key=lambda x: x[1])
test_pred = best_w * test_cls + (1.0 - best_w) * test_rank
test_pred = np.clip(test_pred, 0.0, 1.0)
oof_pred = best_w * oof_cls + (1.0 - best_w) * oof_rank

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

oof_df = pd.DataFrame(
    {
        "row": train[ID_COL].to_numpy(),
        "target": y,
        "prediction": oof_pred,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(best_auc),
    "classifier_oof_roc_auc": float(cls_auc),
    "ranker_oof_rank_roc_auc": float(rank_auc),
    "blend_classifier_weight": float(best_w),
    "research_hypotheses_llm_claimed_used": ["000261"],
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"Classifier OOF ROC AUC: {cls_auc:.6f}")
print(f"Ranker-rank OOF ROC AUC: {rank_auc:.6f}")
print(f"Blended CV ROC AUC: {best_auc:.6f}")
print(json.dumps(result, indent=2))
