import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = Path("./input")
WORKING_DIR = Path("./working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

target_col = "PitNextLap"
id_col = "id"
features = [c for c in train.columns if c not in [target_col, id_col]]
y = train[target_col].astype(int).to_numpy()

groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

cat_cols = (
    train[features].select_dtypes(include=["object", "category"]).columns.tolist()
)
combined = pd.concat([train[features], test[features]], axis=0, ignore_index=True)

for col in cat_cols:
    combined[col] = combined[col].astype("category")

feature_map = {c: f"f{i}" for i, c in enumerate(features)}
X = combined.iloc[: len(train)].copy().rename(columns=feature_map)
X_test = (
    combined.iloc[len(train) :]
    .copy()
    .reset_index(drop=True)
    .rename(columns=feature_map)
)
cat_features = [feature_map[c] for c in cat_cols]

pos = max(int(y.sum()), 1)
neg = len(y) - pos
scale_pos_weight = neg / pos
n_jobs = max(1, min(os.cpu_count() or 1, 8))


def make_model(seed, n_estimators=900):
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float32)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups), 1):
    model = make_model(1000 + fold)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=75, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = getattr(model, "best_iteration_", None) or model.n_estimators
    best_iterations.append(int(best_iter))
    pred = model.predict_proba(X.iloc[va_idx], num_iteration=best_iter)[:, 1]
    oof[va_idx] = pred.astype(np.float32)

    fold_auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(fold_auc))
    print(f"Fold {fold} Year_Race-heldout ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"OOF Year_Race-heldout ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(WORKING_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

final_estimators = int(np.median(best_iterations)) if best_iterations else 600
final_estimators = max(100, final_estimators)
final_model = make_model(2026, n_estimators=final_estimators)
final_model.fit(X, y, categorical_feature=cat_features)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(WORKING_DIR / "submission.csv", index=False)
submission.to_csv(
    WORKING_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "validation": "5-fold StratifiedGroupKFold grouped by Year_Race",
    "oof_auc": float(cv_auc),
    "fold_auc": fold_scores,
    "final_n_estimators": int(final_estimators),
    "research_hypotheses_llm_claimed_used": ["000615"],
    "files_written": [
        str(WORKING_DIR / "submission.csv"),
        str(WORKING_DIR / "oof_predictions.csv.gz"),
        str(WORKING_DIR / "test_predictions.csv.gz"),
    ],
}

with open(WORKING_DIR / "result_review.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
