import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_JOBS = min(8, os.cpu_count() or 1)
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

y = train[target_col].astype(int).values
groups = train["Race"].astype(str) + "_" + train["Year"].astype(str)

combined = pd.concat(
    [train[feature_cols], test[feature_cols]], axis=0, ignore_index=True
)
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    combined[c] = combined[c].astype("category")

X_train_all = combined.iloc[: len(train)].reset_index(drop=True)
X_test_all = combined.iloc[len(train) :].reset_index(drop=True)

domain_y = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
domain_oof = np.zeros(len(combined), dtype=float)

adv_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr_idx, va_idx) in enumerate(adv_cv.split(combined, domain_y), 1):
    adv_model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=RANDOM_STATE + fold,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    adv_model.fit(
        combined.iloc[tr_idx],
        domain_y[tr_idx],
        eval_set=[(combined.iloc[va_idx], domain_y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    domain_oof[va_idx] = adv_model.predict_proba(combined.iloc[va_idx])[:, 1]

adv_auc = roc_auc_score(domain_y, domain_oof)

train_testlike = np.clip(domain_oof[: len(train)], 1e-4, 1 - 1e-4)
weights = train_testlike / (1.0 - train_testlike)
lo, hi = np.quantile(weights, [0.01, 0.99])
weights = np.clip(weights, lo, hi)
weights = weights / weights.mean()

oof = np.zeros(len(train), dtype=float)
fold_aucs = []
best_iterations = []

target_cv = GroupKFold(n_splits=5)
for fold, (tr_idx, va_idx) in enumerate(target_cv.split(X_train_all, y, groups), 1):
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=47,
        min_child_samples=70,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=2.0,
        random_state=RANDOM_STATE + 100 + fold,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    model.fit(
        X_train_all.iloc[tr_idx],
        y[tr_idx],
        sample_weight=weights[tr_idx],
        eval_set=[(X_train_all.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
    )
    pred = model.predict_proba(X_train_all.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    fold_aucs.append(float(auc))
    best_iterations.append(model.best_iteration_ or model.n_estimators)
    print(f"Fold {fold} GroupKFold ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
final_n_estimators = int(np.clip(round(np.mean(best_iterations)), 100, 1200))

final_model = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    n_estimators=final_n_estimators,
    learning_rate=0.035,
    num_leaves=47,
    min_child_samples=70,
    subsample=0.90,
    colsample_bytree=0.90,
    reg_lambda=2.0,
    random_state=RANDOM_STATE + 999,
    n_jobs=N_JOBS,
    verbose=-1,
)
final_model.fit(
    X_train_all,
    y,
    sample_weight=weights,
    categorical_feature=cat_cols,
)

test_pred = final_model.predict_proba(X_test_all)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000443"],
    "adversarial_validation_auc": float(adv_auc),
    "weighted_groupkfold_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_aucs,
    "mean_best_iteration": final_n_estimators,
    "weight_min": float(weights.min()),
    "weight_mean": float(weights.mean()),
    "weight_max": float(weights.max()),
}

print(f"Adversarial validation ROC AUC: {adv_auc:.6f}")
print(f"Weighted 5-fold Race_Year GroupKFold ROC AUC: {cv_auc:.6f}")
print(json.dumps(result, indent=2))
