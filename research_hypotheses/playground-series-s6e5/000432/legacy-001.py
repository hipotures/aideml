import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

INPUT_DIR = Path("./input")
WORKING_DIR = Path("./working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "id"
TARGET_COL = "PitNextLap"
RAW_CAT_COLS = ["Driver", "Race", "Year", "Compound", "Stint"]
CROSS_SPECS = [
    ("Driver", "Compound", "Driver__Compound"),
    ("Race", "Compound", "Race__Compound"),
    ("Race", "Stint", "Race__Stint"),
]
RANDOM_SEED = 432
N_SPLITS = 5
N_THREADS = max(1, min(16, os.cpu_count() or 1))


def add_strategy_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in RAW_CAT_COLS:
        out[col] = out[col].astype("string").fillna("NA").astype(str)
    for left, right, name in CROSS_SPECS:
        out[name] = out[left] + "__" + out[right]
    return out


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(ar, br)[0, 1])


def run_frequency_lgbm_baseline(train_df, y, feature_cols, cat_cols, folds):
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    except Exception as exc:
        print(
            f"Skipping baseline rank comparison because LightGBM import failed: {exc}"
        )
        return None, None

    num_cols = [c for c in feature_cols if c not in cat_cols]
    oof = np.zeros(len(train_df), dtype=np.float32)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        tr = train_df.iloc[tr_idx]
        va = train_df.iloc[va_idx]

        x_tr = tr[num_cols].copy()
        x_va = va[num_cols].copy()

        for col in cat_cols:
            freqs = tr[col].value_counts(normalize=True)
            x_tr[f"{col}__freq"] = tr[col].map(freqs).astype("float32")
            x_va[f"{col}__freq"] = va[col].map(freqs).fillna(0).astype("float32")

        x_tr = x_tr.replace([np.inf, -np.inf], np.nan).fillna(-999)
        x_va = x_va.replace([np.inf, -np.inf], np.nan).fillna(-999)
        safe_cols = [f"f{i}" for i in range(x_tr.shape[1])]
        x_tr.columns = safe_cols
        x_va.columns = safe_cols

        model = LGBMClassifier(
            objective="binary",
            n_estimators=250,
            learning_rate=0.06,
            num_leaves=63,
            min_child_samples=60,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            random_state=RANDOM_SEED + fold,
            n_jobs=N_THREADS,
            verbosity=-1,
        )
        model.fit(
            x_tr,
            y[tr_idx],
            eval_set=[(x_va, y[va_idx])],
            eval_metric="auc",
            callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
        )
        pred = model.predict_proba(x_va)[:, 1]
        oof[va_idx] = pred
        score = roc_auc_score(y[va_idx], pred)
        fold_scores.append(score)
        print(f"Frequency baseline fold {fold} ROC AUC: {score:.6f}")

    return oof, float(roc_auc_score(y, oof))


train_raw = pd.read_csv(INPUT_DIR / "train.csv.gz")
test_raw = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

train = add_strategy_categoricals(train_raw)
test = add_strategy_categoricals(test_raw)

cat_cols = RAW_CAT_COLS + [name for _, _, name in CROSS_SPECS]
feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
cat_feature_indices = [feature_cols.index(c) for c in cat_cols]

y = train[TARGET_COL].astype(int).to_numpy()
groups = train_raw["Year"].astype(str) + "__" + train_raw["Race"].astype(str)

folds = list(GroupKFold(n_splits=N_SPLITS).split(train[feature_cols], y, groups))
test_pool = Pool(test[feature_cols], cat_features=cat_feature_indices)

oof = np.zeros(len(train), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float64)
fold_aucs = []

cat_params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=900,
    learning_rate=0.045,
    depth=6,
    l2_leaf_reg=10.0,
    random_strength=1.5,
    bootstrap_type="Bayesian",
    bagging_temperature=0.8,
    boosting_type="Ordered",
    one_hot_max_size=2,
    max_ctr_complexity=2,
    ctr_leaf_count_limit=64,
    counter_calc_method="SkipTest",
    od_type="Iter",
    od_wait=80,
    random_seed=RANDOM_SEED,
    thread_count=N_THREADS,
    allow_writing_files=False,
)

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    train_pool = Pool(
        train.iloc[tr_idx][feature_cols], y[tr_idx], cat_features=cat_feature_indices
    )
    valid_pool = Pool(
        train.iloc[va_idx][feature_cols], y[va_idx], cat_features=cat_feature_indices
    )

    model = CatBoostClassifier(**cat_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=100)

    val_pred = model.predict_proba(valid_pool)[:, 1]
    oof[va_idx] = val_pred.astype(np.float32)
    test_pred += model.predict_proba(test_pool)[:, 1] / N_SPLITS

    fold_auc = roc_auc_score(y[va_idx], val_pred)
    fold_aucs.append(float(fold_auc))
    print(f"CatBoost fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = float(roc_auc_score(y, oof))
print(f"Grouped 5-fold CatBoost ROC AUC: {cv_auc:.6f}")

baseline_oof, baseline_auc = run_frequency_lgbm_baseline(
    train, y, feature_cols, cat_cols, folds
)
baseline_rank_corr = None
if baseline_oof is not None:
    baseline_rank_corr = rank_corr(oof, baseline_oof)
    print(f"Frequency baseline grouped 5-fold ROC AUC: {baseline_auc:.6f}")
    print(
        f"CatBoost vs frequency baseline OOF rank correlation: {baseline_rank_corr:.6f}"
    )

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(WORKING_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

pred_col = [c for c in sample.columns if c != ID_COL][0]
test_pred_by_id = pd.Series(test_pred, index=test_raw[ID_COL].to_numpy())
submission = sample[[ID_COL]].copy()
submission[pred_col] = submission[ID_COL].map(test_pred_by_id).astype(float).clip(0, 1)
submission.to_csv(WORKING_DIR / "submission.csv", index=False)
submission.to_csv(
    WORKING_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "validation_scheme": "5-fold GroupKFold by Year__Race",
    "catboost_cv_auc": cv_auc,
    "catboost_fold_auc": fold_aucs,
    "frequency_baseline_cv_auc": baseline_auc,
    "catboost_vs_frequency_baseline_oof_rank_corr": baseline_rank_corr,
    "research_hypotheses_llm_claimed_used": ["000432"],
    "submission_path": str(WORKING_DIR / "submission.csv"),
    "oof_predictions_path": str(WORKING_DIR / "oof_predictions.csv.gz"),
    "test_predictions_path": str(WORKING_DIR / "test_predictions.csv.gz"),
}
with open(WORKING_DIR / "result_review.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
