import os
import gc
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier, XGBRanker

warnings.filterwarnings("ignore")

SEED = 2026
TARGET = "PitNextLap"
ID_COL = "id"
GROUP_COLS = ["Year", "Race", "LapNumber"]
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
n_train = len(train)

numeric_cols = [
    c
    for c in train.columns
    if c not in [ID_COL, TARGET] and pd.api.types.is_numeric_dtype(train[c])
]
relative_cols = [c for c in numeric_cols if c not in ["Year", "LapNumber"]]


def make_group_key(df):
    return (
        df["Year"].astype(str)
        + "|"
        + df["Race"].astype(str)
        + "|"
        + df["LapNumber"].astype(str)
    )


def add_features(df):
    df = df.copy()
    g = df.groupby(GROUP_COLS, sort=False)

    for c in relative_cols:
        mean = g[c].transform("mean")
        std = g[c].transform("std").replace(0, np.nan)
        df[f"{c}_field_diff"] = df[c] - mean
        df[f"{c}_field_z"] = ((df[c] - mean) / (std + 1e-6)).fillna(0)
        df[f"{c}_field_rank"] = g[c].rank(method="average", pct=True)

    df["TyreLife_per_LapNumber"] = df["TyreLife"] / (df["LapNumber"] + 1e-6)
    df["Deg_per_TyreLife"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1e-6)
    df["LapTime_per_TyreLife"] = df["LapTime (s)"] / (df["TyreLife"] + 1e-6)
    df["Stint_x_TyreLife"] = df["Stint"] * df["TyreLife"]
    df["Progress_x_TyreLife"] = df["RaceProgress"] * df["TyreLife"]
    df["Position_x_Progress"] = df["Position"] * df["RaceProgress"]
    df["RemainingRace"] = 1.0 - df["RaceProgress"]
    return df


train_key = make_group_key(train)
test_key = make_group_key(test)
train_qid = pd.factorize(train_key, sort=True)[0].astype(np.int32)

train_fe = add_features(train.drop(columns=[TARGET]))
test_fe = add_features(test)

feature_cols = [c for c in train_fe.columns if c != ID_COL]
cat_cols = [
    c
    for c in feature_cols
    if train_fe[c].dtype == "object" or test_fe[c].dtype == "object"
]

X_train_df = train_fe[feature_cols].copy()
X_test_df = test_fe[feature_cols].copy()

if cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    combined_cat = (
        pd.concat([X_train_df[cat_cols], X_test_df[cat_cols]], axis=0)
        .astype(str)
        .fillna("__NA__")
    )
    enc.fit(combined_cat)
    X_train_df[cat_cols] = enc.transform(
        X_train_df[cat_cols].astype(str).fillna("__NA__")
    )
    X_test_df[cat_cols] = enc.transform(
        X_test_df[cat_cols].astype(str).fillna("__NA__")
    )

X_all = pd.concat([X_train_df, X_test_df], axis=0)
X_all = X_all.replace([np.inf, -np.inf], np.nan)
medians = X_all.iloc[:n_train].median(axis=0)
X_all = X_all.fillna(medians).fillna(0)

X_train = X_all.iloc[:n_train].to_numpy(dtype=np.float32, copy=True)
X_test = X_all.iloc[n_train:].to_numpy(dtype=np.float32, copy=True)

del X_train_df, X_test_df, X_all, train_fe, test_fe
gc.collect()

try:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X_train, y, groups=train_qid))
except Exception:
    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(X_train, y, groups=train_qid))

n_jobs = max(1, min(8, os.cpu_count() or 1))
pos = max(1, int(y.sum()))
neg = max(1, len(y) - pos)
scale_pos_weight = neg / pos

oof_clf = np.zeros(n_train, dtype=np.float32)
oof_pairwise_raw = np.zeros(n_train, dtype=np.float32)
oof_ndcg_raw = np.zeros(n_train, dtype=np.float32)

test_clf = np.zeros(len(test), dtype=np.float32)
test_pairwise_raw = np.zeros(len(test), dtype=np.float32)
test_ndcg_raw = np.zeros(len(test), dtype=np.float32)


def fit_ranker(model, X, y_fold, qid):
    order = np.argsort(qid, kind="mergesort")
    Xs = X[order]
    ys = y_fold[order]
    qids = qid[order]
    try:
        model.fit(Xs, ys, qid=qids, verbose=False)
    except TypeError:
        _, counts = np.unique(qids, return_counts=True)
        model.fit(Xs, ys, group=counts, verbose=False)
    return model


for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    clf = XGBClassifier(
        n_estimators=320,
        learning_rate=0.045,
        max_depth=4,
        min_child_weight=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=4.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED + fold,
        n_jobs=n_jobs,
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_train[tr_idx], y[tr_idx], verbose=False)
    oof_clf[va_idx] = clf.predict_proba(X_train[va_idx])[:, 1]
    test_clf += clf.predict_proba(X_test)[:, 1] / len(splits)

    for objective, oof_arr, test_arr in [
        ("rank:pairwise", oof_pairwise_raw, test_pairwise_raw),
        ("rank:ndcg", oof_ndcg_raw, test_ndcg_raw),
    ]:
        ranker = XGBRanker(
            n_estimators=240,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=15,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            objective=objective,
            eval_metric="ndcg",
            tree_method="hist",
            random_state=SEED + fold,
            n_jobs=n_jobs,
        )
        ranker = fit_ranker(
            ranker,
            X_train[tr_idx],
            y[tr_idx],
            train_qid[tr_idx],
        )
        oof_arr[va_idx] = ranker.predict(X_train[va_idx]).astype(np.float32)
        test_arr += ranker.predict(X_test).astype(np.float32) / len(splits)

    fold_auc = roc_auc_score(y[va_idx], oof_clf[va_idx])
    print(f"Fold {fold} classifier ROC AUC: {fold_auc:.6f}")

    del clf, ranker
    gc.collect()


def within_group_rank(values, keys):
    return (
        pd.Series(values)
        .groupby(pd.Series(np.asarray(keys)), sort=False)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )


pairwise_oof = within_group_rank(oof_pairwise_raw, train_key)
ndcg_oof = within_group_rank(oof_ndcg_raw, train_key)
pairwise_test = within_group_rank(test_pairwise_raw, test_key)
ndcg_test = within_group_rank(test_ndcg_raw, test_key)

rank_weight = 0.20
candidates = {
    "classifier_plus_rank_pairwise": (
        (1.0 - rank_weight) * oof_clf + rank_weight * pairwise_oof,
        (1.0 - rank_weight) * test_clf + rank_weight * pairwise_test,
    ),
    "classifier_plus_rank_ndcg": (
        (1.0 - rank_weight) * oof_clf + rank_weight * ndcg_oof,
        (1.0 - rank_weight) * test_clf + rank_weight * ndcg_test,
    ),
}

classifier_auc = roc_auc_score(y, oof_clf)
pairwise_auc = roc_auc_score(y, pairwise_oof)
ndcg_auc = roc_auc_score(y, ndcg_oof)

print(f"5-fold OOF ROC AUC classifier: {classifier_auc:.6f}")
print(f"5-fold OOF ROC AUC rank:pairwise score only: {pairwise_auc:.6f}")
print(f"5-fold OOF ROC AUC rank:ndcg score only: {ndcg_auc:.6f}")

best_name = None
best_auc = -np.inf
best_oof = None
best_test = None

for name, (oof_pred, test_pred) in candidates.items():
    auc = roc_auc_score(y, oof_pred)
    print(f"5-fold OOF ROC AUC {name}: {auc:.6f}")
    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_oof = oof_pred
        best_test = test_pred

best_test = np.clip(best_test, 0.0, 1.0)

submission = sample.copy()
if np.array_equal(sample[ID_COL].to_numpy(), test[ID_COL].to_numpy()):
    submission[TARGET] = best_test
else:
    pred_by_id = pd.Series(best_test, index=test[ID_COL].to_numpy())
    submission[TARGET] = sample[ID_COL].map(pred_by_id).to_numpy()

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": np.clip(best_oof, 0.0, 1.0),
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

test_pred_df = sample.copy()
test_pred_df[TARGET] = submission[TARGET].to_numpy()
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(f"Selected blend: {best_name}")
print(f"Selected 5-fold OOF ROC AUC: {best_auc:.6f}")

print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000577"],
            "metric": "roc_auc",
            "cv_folds": 5,
            "rank_objectives_tested": ["rank:pairwise", "rank:ndcg"],
            "selected_model": best_name,
            "oof_auc": float(best_auc),
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
            "oof_predictions_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
            "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
        },
        sort_keys=True,
    )
)
