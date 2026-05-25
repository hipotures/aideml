import os
import json
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 144

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_catboost_features(df):
    df = df.copy()
    for col in ["Driver", "Race", "Compound", "Year"]:
        df[col] = df[col].astype("string").fillna("__NA__").astype(str)

    df["Driver__Race"] = df["Driver"] + "__" + df["Race"]
    df["Driver__Compound"] = df["Driver"] + "__" + df["Compound"]
    df["Race__Compound"] = df["Race"] + "__" + df["Compound"]
    df["Race__Year__Compound"] = df["Race"] + "__" + df["Year"] + "__" + df["Compound"]
    return df


train = add_catboost_features(train)
test = add_catboost_features(test)

cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "Year",
    "Driver__Race",
    "Driver__Compound",
    "Race__Compound",
    "Race__Year__Compound",
]

feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]
sort_cols = ["Year", "Race", "Driver", "LapNumber", ID_COL]
groups = train["Year"] + "__" + train["Race"]
y = train[TARGET].astype(int).to_numpy()

params = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "iterations": 800,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "max_ctr_complexity": 2,
    "has_time": True,
    "random_seed": SEED,
    "allow_writing_files": False,
    "thread_count": min(16, os.cpu_count() or 1),
    "od_type": "Iter",
    "od_wait": 80,
    "verbose": False,
}

oof = np.zeros(len(train), dtype=float)
fold_aucs = []
best_iters = []

gkf = GroupKFold(n_splits=5)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train, y, groups), start=1):
    tr = train.iloc[tr_idx].sort_values(sort_cols)
    va = train.iloc[va_idx].sort_values(sort_cols)

    tr_pool = Pool(tr[feature_cols], tr[TARGET].astype(int), cat_features=cat_cols)
    va_pool = Pool(va[feature_cols], va[TARGET].astype(int), cat_features=cat_cols)

    model = CatBoostClassifier(**params)
    model.fit(tr_pool, eval_set=va_pool, use_best_model=True)

    pred = model.predict_proba(va_pool)[:, 1]
    oof[va.index.to_numpy()] = pred

    fold_auc = roc_auc_score(va[TARGET].astype(int), pred)
    fold_aucs.append(fold_auc)

    best_iter = model.get_best_iteration()
    if best_iter is not None and best_iter >= 0:
        best_iters.append(best_iter + 1)

    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

oof_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {oof_auc:.6f}")
print(f"Mean fold ROC AUC: {np.mean(fold_aucs):.6f} +/- {np.std(fold_aucs):.6f}")

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_params = params.copy()
final_params.pop("od_type", None)
final_params.pop("od_wait", None)
if best_iters:
    final_params["iterations"] = int(
        np.clip(np.median(best_iters), 100, params["iterations"])
    )

full_train = train.sort_values(sort_cols)
full_pool = Pool(
    full_train[feature_cols], full_train[TARGET].astype(int), cat_features=cat_cols
)

final_model = CatBoostClassifier(**final_params)
final_model.fit(full_pool)

test_sorted = test.sort_values(sort_cols)
test_pool = Pool(test_sorted[feature_cols], cat_features=cat_cols)
test_pred_sorted = final_model.predict_proba(test_pool)[:, 1]
test_pred = pd.Series(test_pred_sorted, index=test_sorted.index).sort_index().to_numpy()
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "oof_roc_auc": float(oof_auc),
            "fold_roc_auc": [float(v) for v in fold_aucs],
            "research_hypotheses_llm_claimed_used": ["000144"],
        }
    )
)
