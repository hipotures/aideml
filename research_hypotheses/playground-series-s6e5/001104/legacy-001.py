import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]
QID_COLS = ["Year", "Race", "LapNumber"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values

features = [c for c in train.columns if c not in [TARGET, ID_COL]]
all_df = pd.concat([train[features], test[features]], axis=0, ignore_index=True)

for c in CAT_COLS:
    all_df[c] = all_df[c].astype("category")
    all_df[c] = all_df[c].cat.codes.astype("int32")

for c in all_df.columns:
    if all_df[c].dtype == "float64":
        all_df[c] = all_df[c].astype("float32")
    elif all_df[c].dtype == "int64":
        all_df[c] = all_df[c].astype("int32")

X = all_df.iloc[: len(train)].reset_index(drop=True)
X_test = all_df.iloc[len(train) :].reset_index(drop=True)

train_qid_raw = train[QID_COLS].astype(str).agg("|".join, axis=1)
test_qid_raw = test[QID_COLS].astype(str).agg("|".join, axis=1)
qid_codes = pd.factorize(pd.concat([train_qid_raw, test_qid_raw], ignore_index=True))[0]
train_qid = qid_codes[: len(train)]
test_qid = qid_codes[len(train) :]

oof_clf = np.zeros(len(train), dtype=np.float32)
oof_rank = np.zeros(len(train), dtype=np.float32)
test_clf_folds = []
test_rank_folds = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=104)

try:
    from xgboost import XGBClassifier, XGBRanker

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        clf = XGBClassifier(
            n_estimators=650,
            max_depth=5,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=8,
            reg_lambda=4.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=1000 + fold,
            n_jobs=max(1, os.cpu_count() or 1),
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_clf[va_idx] = clf.predict_proba(X_va)[:, 1]
        test_clf_folds.append(clf.predict_proba(X_test)[:, 1])

        rank_df = pd.DataFrame({"idx": tr_idx, "qid": train_qid[tr_idx], "y": y_tr})
        grp_stats = rank_df.groupby("qid")["y"].agg(["sum", "count"])
        mixed_qids = grp_stats[
            (grp_stats["sum"] > 0) & (grp_stats["sum"] < grp_stats["count"])
        ].index
        rank_df = rank_df[rank_df["qid"].isin(mixed_qids)].sort_values(
            "qid", kind="mergesort"
        )
        rank_idx = rank_df["idx"].values
        group_sizes = rank_df.groupby("qid", sort=False).size().values

        ranker = XGBRanker(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.045,
            subsample=0.90,
            colsample_bytree=0.90,
            min_child_weight=4,
            reg_lambda=3.0,
            objective="rank:ndcg",
            eval_metric="ndcg",
            tree_method="hist",
            random_state=2000 + fold,
            n_jobs=max(1, os.cpu_count() or 1),
        )
        ranker.fit(X.iloc[rank_idx], y[rank_idx], group=group_sizes, verbose=False)

        oof_rank[va_idx] = ranker.predict(X_va)
        test_rank_folds.append(ranker.predict(X_test))

        print(f"fold {fold} classifier_auc={roc_auc_score(y_va, oof_clf[va_idx]):.6f}")

except Exception as e:
    raise RuntimeError(
        "This solution requires xgboost with XGBClassifier and XGBRanker support."
    ) from e


def rank_normalize(values):
    s = pd.Series(values)
    return ((s.rank(method="average").values - 0.5) / len(s)).astype(np.float32)


oof_clf_r = rank_normalize(oof_clf)
oof_rank_r = rank_normalize(oof_rank)

best_auc = -1.0
best_w = 0.0
for w in np.linspace(0.0, 1.0, 41):
    pred = (1.0 - w) * oof_clf_r + w * oof_rank_r
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_auc = auc
        best_w = float(w)

test_clf = np.mean(test_clf_folds, axis=0)
test_rank = np.mean(test_rank_folds, axis=0)
test_pred = (1.0 - best_w) * rank_normalize(test_clf) + best_w * rank_normalize(
    test_rank
)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

oof_pred = (1.0 - best_w) * oof_clf_r + best_w * oof_rank_r

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: test_pred,
    }
)
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
test_pred_df.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(best_auc),
            "best_ranker_blend_weight": best_w,
            "research_hypotheses_llm_claimed_used": ["001104"],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        },
        indent=2,
    )
)
