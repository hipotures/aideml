import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

y = train[target_col].astype(int).values
train_ids = train[id_col].values
test_ids = sample[id_col].values

train_x = train.drop(columns=[target_col])
test_x = test.copy()

interaction_specs = {
    "Race_Year": ["Race", "Year"],
    "Race_Compound": ["Race", "Compound"],
    "Driver_Compound": ["Driver", "Compound"],
    "Driver_Race": ["Driver", "Race"],
    "Compound_Stint": ["Compound", "Stint"],
}

combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

created_interactions = []
for new_col, cols in interaction_specs.items():
    if all(c in combined.columns for c in cols):
        combined[new_col] = combined[cols].astype(str).agg("__".join, axis=1)
        freq = combined[new_col].value_counts(normalize=True)
        combined[new_col + "_freq"] = combined[new_col].map(freq).astype("float32")
        created_interactions.append(new_col)

train_x = combined.iloc[: len(train_x)].copy()
test_x = combined.iloc[len(train_x) :].copy()

drop_cols = [id_col]
features = [c for c in train_x.columns if c not in drop_cols]

cat_cols = [
    c for c in features if train_x[c].dtype == "object" or c in created_interactions
]

for c in cat_cols:
    all_values = (
        pd.concat([train_x[c], test_x[c]], axis=0).astype(str).fillna("__MISSING__")
    )
    categories = pd.Categorical(all_values).categories
    train_x[c] = pd.Categorical(
        train_x[c].astype(str).fillna("__MISSING__"), categories=categories
    )
    test_x[c] = pd.Categorical(
        test_x[c].astype(str).fillna("__MISSING__"), categories=categories
    )

X = train_x[features]
X_test = test_x[features]

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
except Exception as e:
    raise RuntimeError("This script requires lightgbm to be installed.") from e

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=48)
oof = np.zeros(len(X), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.035,
        n_estimators=3000,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=4800 + fold,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(100), log_evaluation(100)],
    )

    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = va_pred.astype(np.float32)
    fold_auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(fold_auc)

    test_pred += model.predict_proba(X_test)[:, 1].astype(np.float32) / skf.n_splits
    print(f"fold {fold} auc: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold cv auc: {cv_auc:.6f}")
print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_auc": float(cv_auc),
            "fold_auc": [float(x) for x in fold_scores],
            "research_hypotheses_llm_claimed_used": ["000048"],
        }
    )
)

pd.DataFrame(
    {
        "row": np.arange(len(train_x), dtype=np.int64),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_predictions = sample.copy()
test_predictions[target_col] = np.clip(test_pred, 0, 1)
test_predictions.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

submission = sample.copy()
submission[target_col] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
