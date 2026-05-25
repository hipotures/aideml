import os
import gc
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

SEED = 42
N_FOLDS = 5
MAX_ITERS = int(os.environ.get("CATBOOST_ITERS", "550"))
WORKDIR = "./working"
os.makedirs(WORKDIR, exist_ok=True)

train = pd.read_csv("./input/train.csv.gz")
test = pd.read_csv("./input/test.csv.gz")
sample = pd.read_csv("./input/sample_submission.csv.gz")

target_col = "PitNextLap"
id_col = "id"
y = train[target_col].astype(int).to_numpy()

base_cat_cols = ["Driver", "Race", "Compound"]
numeric_cols = [
    "LapNumber",
    "TyreLife",
    "RaceProgress",
    "Cumulative_Degradation",
    "LapTime (s)",
    "LapTime_Delta",
    "Position",
    "Position_Change",
]
cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "Year_cat",
    "Stint_cat",
    "PitStop_cat",
    "LapBucket",
    "Driver_Compound",
    "Race_Compound",
    "Race_LapBucket",
]
feature_cols = cat_cols + numeric_cols


def make_features(df):
    out = df.copy()

    for c in base_cat_cols:
        out[c] = out[c].astype(str).fillna("__NA__")

    out["Year_cat"] = out["Year"].astype(str).fillna("__NA__")
    out["Stint_cat"] = out["Stint"].astype(str).fillna("__NA__")
    out["PitStop_cat"] = out["PitStop"].astype(str).fillna("__NA__")

    lap = pd.to_numeric(out["LapNumber"], errors="coerce").fillna(-1)
    out["LapBucket"] = (
        np.floor(((lap - 1).clip(lower=0)) / 5).astype(np.int16).astype(str)
    )

    out["Driver_Compound"] = out["Driver"] + "__" + out["Compound"]
    out["Race_Compound"] = out["Race"] + "__" + out["Compound"]
    out["Race_LapBucket"] = out["Race"] + "__lb" + out["LapBucket"]

    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    return out[feature_cols]


X = make_features(train.drop(columns=[target_col]))
X_test = make_features(test)
groups = train["Year"].astype(str) + "__" + train["Race"].astype(str)

if HAS_SGK:
    try:
        splitter = StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=SEED
        )
        splits = list(splitter.split(X, y, groups))
        split_name = "StratifiedGroupKFold by Year/Race"
    except Exception:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X, y))
        split_name = "StratifiedKFold fallback"
else:
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y))
    split_name = "StratifiedKFold"

print(f"Using {split_name}")


def build_model(eval_metric, iterations, seed):
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.06,
        depth=6,
        loss_function="Logloss",
        eval_metric=eval_metric,
        boosting_type="Ordered",
        l2_leaf_reg=8.0,
        random_strength=1.0,
        od_type="Iter",
        od_wait=70,
        random_seed=seed,
        thread_count=max(1, os.cpu_count() or 1),
        allow_writing_files=False,
        verbose=False,
    )


def fit_fold(eval_metric, fold_id, tr_idx, va_idx):
    train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_cols)
    valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_cols)

    model = build_model(eval_metric, MAX_ITERS, SEED + fold_id)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    pred = model.predict_proba(valid_pool)[:, 1].astype("float32")
    auc = roc_auc_score(y[va_idx], pred)
    trees = int(getattr(model, "tree_count_", MAX_ITERS))

    del train_pool, valid_pool, model
    gc.collect()

    return pred, auc, trees


first_tr, first_va = splits[0]
comparison = {}
stored_first_fold = {}

for metric_name in ["Logloss", "AUC"]:
    print(f"First-fold comparison with eval_metric={metric_name}")
    pred, auc, trees = fit_fold(metric_name, 0, first_tr, first_va)
    comparison[metric_name] = {"fold_auc": float(auc), "trees": int(trees)}
    stored_first_fold[metric_name] = (pred, auc, trees)
    print(f"  {metric_name} first-fold ROC AUC: {auc:.6f}, trees: {trees}")

selected_metric = max(comparison, key=lambda m: comparison[m]["fold_auc"])
print(f"Selected early-stopping metric: {selected_metric}")

oof = np.zeros(len(train), dtype="float32")
fold_aucs = []
tree_counts = []

first_pred, first_auc, first_trees = stored_first_fold[selected_metric]
oof[first_va] = first_pred
fold_aucs.append(float(first_auc))
tree_counts.append(int(first_trees))

for fold_id, (tr_idx, va_idx) in enumerate(splits[1:], start=1):
    print(f"Training fold {fold_id + 1}/{N_FOLDS} with eval_metric={selected_metric}")
    pred, auc, trees = fit_fold(selected_metric, fold_id, tr_idx, va_idx)
    oof[va_idx] = pred
    fold_aucs.append(float(auc))
    tree_counts.append(int(trees))
    print(f"  Fold {fold_id + 1} ROC AUC: {auc:.6f}, trees: {trees}")

cv_auc = roc_auc_score(y, oof)
final_iterations = int(np.clip(np.median(tree_counts), 120, MAX_ITERS))

print(f"5-fold CV ROC AUC: {cv_auc:.6f}")
print(f"Fold ROC AUCs: {[round(v, 6) for v in fold_aucs]}")
print(f"Final training iterations: {final_iterations}")

pd.DataFrame({"row": np.arange(len(train)), "target": y, "prediction": oof}).to_csv(
    os.path.join(WORKDIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

full_pool = Pool(X, y, cat_features=cat_cols)
test_pool = Pool(X_test, cat_features=cat_cols)

final_model = build_model(selected_metric, final_iterations, SEED + 999)
final_model.fit(full_pool, use_best_model=False)

test_pred = final_model.predict_proba(test_pool)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORKDIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKDIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "validation_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(v) for v in fold_aucs],
    "first_fold_comparison": comparison,
    "selected_early_stopping_metric": selected_metric,
    "final_iterations": int(final_iterations),
    "research_hypotheses_llm_claimed_used": ["000279"],
}

for name in ["result_review.json", "result.json"]:
    with open(os.path.join(WORKDIR, name), "w") as f:
        json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
