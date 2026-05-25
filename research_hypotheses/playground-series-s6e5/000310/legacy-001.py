import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 2026

NUMERIC_CORE = [
    "LapNumber",
    "TyreLife",
    "RaceProgress",
    "Position",
    "Position_Change",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "PitStop",
]

BASE_CATS = ["Driver", "Race", "Compound", "Stint", "Year"]
STRATEGY_CATS = [
    "Driver_x_Race",
    "Race_x_Year_x_Compound",
    "Compound_x_Stint",
    "Driver_x_Compound_x_Stint",
]
CAT_COLS = BASE_CATS + STRATEGY_CATS
FEATURES = NUMERIC_CORE + CAT_COLS


def prepare_features(df):
    out = pd.DataFrame(index=df.index)

    for col in NUMERIC_CORE:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    cats = {}
    for col in BASE_CATS:
        cats[col] = df[col].astype("string").fillna("__NA__")
        out[col] = cats[col].astype(str)

    out["Driver_x_Race"] = (cats["Driver"] + "|" + cats["Race"]).astype(str)
    out["Race_x_Year_x_Compound"] = (
        cats["Race"] + "|" + cats["Year"] + "|" + cats["Compound"]
    ).astype(str)
    out["Compound_x_Stint"] = (cats["Compound"] + "|" + cats["Stint"]).astype(str)
    out["Driver_x_Compound_x_Stint"] = (
        cats["Driver"] + "|" + cats["Compound"] + "|" + cats["Stint"]
    ).astype(str)

    return out[FEATURES]


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

X = prepare_features(train)
X_test = prepare_features(test)
y = train[TARGET].astype(int).values
groups = (train["Race"].astype(str) + "|" + train["Year"].astype(str)).values
cat_idx = [X.columns.get_loc(c) for c in CAT_COLS]

n_groups = pd.Series(groups).nunique()
n_splits = min(5, n_groups)
if n_splits < 2:
    raise ValueError("Need at least two Race-Year groups for grouped validation.")

if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    splits = splitter.split(X, y, groups)
else:
    splitter = GroupKFold(n_splits=n_splits)
    splits = splitter.split(X, y, groups)

common_params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    boosting_type="Ordered",
    iterations=650,
    learning_rate=0.07,
    depth=6,
    l2_leaf_reg=8.0,
    random_strength=0.6,
    bootstrap_type="Bernoulli",
    subsample=0.85,
    rsm=0.90,
    one_hot_max_size=2,
    max_ctr_complexity=2,
    auto_class_weights="SqrtBalanced",
    fold_permutation_block=128,
    random_seed=RANDOM_STATE,
    allow_writing_files=False,
    thread_count=max(1, os.cpu_count() or 1),
    verbose=100,
)

oof = np.zeros(len(train), dtype=np.float32)
fold_aucs = []
best_tree_counts = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_idx)
    valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_idx)

    model = CatBoostClassifier(
        **common_params,
        od_type="Iter",
        od_wait=80,
    )
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    pred = model.predict_proba(valid_pool)[:, 1]
    oof[va_idx] = pred.astype(np.float32)

    fold_auc = roc_auc_score(y[va_idx], pred)
    fold_aucs.append(fold_auc)
    best_tree_counts.append(int(model.tree_count_))

    print(
        f"fold={fold} auc={fold_auc:.6f} trees={model.tree_count_} "
        f"valid_size={len(va_idx)}"
    )

cv_auc = roc_auc_score(y, oof)
final_iterations = int(
    np.clip(round(np.mean(best_tree_counts)), 100, common_params["iterations"])
)

print(f"OOF ROC AUC: {cv_auc:.6f}")
print(f"Training final model with {final_iterations} trees")

final_params = dict(common_params)
final_params["iterations"] = final_iterations
final_params["verbose"] = 100

final_model = CatBoostClassifier(**final_params)
final_model.fit(Pool(X, y, cat_features=cat_idx))

test_pred = final_model.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1]
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

submission = sample.copy()
target_col = [c for c in submission.columns if c != ID_COL][0]
submission[target_col] = test_pred

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result_review = {
    "research_hypotheses_llm_claimed_used": ["000310"],
    "metric": "roc_auc",
    "oof_auc": float(cv_auc),
    "fold_aucs": [float(v) for v in fold_aucs],
    "final_iterations": int(final_iterations),
}
print(json.dumps(result_review, sort_keys=True))
