import os
import json
import warnings
import numpy as np
import pandas as pd

from catboost import CatBoostRanker, Pool
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

for df in (train, test):
    df["RaceYear"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["query_group"] = df["RaceYear"] + "_lap_" + df["LapNumber"].astype(str)

feature_cols = [
    c for c in train.columns if c not in [target_col, id_col, "query_group"]
]
cat_cols = [c for c in feature_cols if train[c].dtype == "object"]

for c in cat_cols:
    train[c] = train[c].astype(str).fillna("missing")
    test[c] = test[c].astype(str).fillna("missing")

num_cols = [c for c in feature_cols if c not in cat_cols]
for c in num_cols:
    med = train[c].median()
    train[c] = train[c].fillna(med)
    test[c] = test[c].fillna(med)

y = train[target_col].astype(int).values
groups = train["query_group"].astype(str).values

try:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(train, y, groups))
except Exception:
    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(train, y, groups))


def sorted_indices_by_group(df, indices):
    return (
        df.iloc[indices][["query_group", id_col]]
        .sort_values(["query_group", id_col])
        .index.to_numpy()
    )


def make_rank_pool(df, indices, labels=None):
    ordered_idx = sorted_indices_by_group(df, indices)
    pool = Pool(
        data=df.loc[ordered_idx, feature_cols],
        label=None if labels is None else labels[ordered_idx],
        group_id=df.loc[ordered_idx, "query_group"].astype(str).values,
        cat_features=cat_cols,
    )
    return pool, ordered_idx


test_pool = Pool(test[feature_cols], cat_features=cat_cols)

oof_raw = np.zeros(len(train), dtype=np.float32)
test_raw_folds = np.zeros((len(test), N_SPLITS), dtype=np.float32)

params = {
    "loss_function": "QueryCrossEntropy",
    "iterations": 700,
    "learning_rate": 0.045,
    "depth": 6,
    "l2_leaf_reg": 6.0,
    "random_seed": SEED,
    "od_type": "Iter",
    "od_wait": 60,
    "allow_writing_files": False,
    "thread_count": max(1, min(8, os.cpu_count() or 1)),
    "verbose": 100,
}

for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
    train_pool, tr_order = make_rank_pool(train, tr_idx, y)
    valid_pool, va_order = make_rank_pool(train, va_idx, y)

    model = CatBoostRanker(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    oof_raw[va_order] = model.predict(valid_pool).astype(np.float32)
    test_raw_folds[:, fold - 1] = model.predict(test_pool).astype(np.float32)

    fold_auc = roc_auc_score(y[va_order], oof_raw[va_order])
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

raw_auc = roc_auc_score(y, oof_raw)

cal_oof = np.zeros(len(train), dtype=np.float32)
cal_splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + 1)

for cal_tr, cal_va in cal_splitter.split(oof_raw.reshape(-1, 1), y):
    calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
    calibrator.fit(oof_raw[cal_tr].reshape(-1, 1), y[cal_tr])
    cal_oof[cal_va] = calibrator.predict_proba(oof_raw[cal_va].reshape(-1, 1))[:, 1]

cal_auc = roc_auc_score(y, cal_oof)

final_calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
final_calibrator.fit(oof_raw.reshape(-1, 1), y)

test_raw_mean = test_raw_folds.mean(axis=1)
test_pred = final_calibrator.predict_proba(test_raw_mean.reshape(-1, 1))[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": cal_oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

print(f"OOF ROC AUC raw ranker score: {raw_auc:.6f}")
print(f"OOF ROC AUC calibrated probability: {cal_auc:.6f}")
print(
    json.dumps(
        {
            "metric": "roc_auc",
            "oof_roc_auc_raw": float(raw_auc),
            "oof_roc_auc_calibrated": float(cal_auc),
            "research_hypotheses_llm_claimed_used": ["000666"],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        }
    )
)
