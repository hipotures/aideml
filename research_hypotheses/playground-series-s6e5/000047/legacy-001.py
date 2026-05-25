import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

y = train[target_col].astype(int)
train_ids = train[id_col].copy()
test_ids = sample[id_col].copy()

X_train = train.drop(columns=[target_col])
X_test = test.copy()

cat_cols = ["Driver", "Race", "Compound"]
for col in cat_cols:
    X_train[col] = X_train[col].astype(str).str.strip().str.upper()
    X_test[col] = X_test[col].astype(str).str.strip().str.upper()

X_train["Year"] = X_train["Year"].astype(str)
X_test["Year"] = X_test["Year"].astype(str)
X_train["Race_Year"] = X_train["Race"] + "_" + X_train["Year"]
X_test["Race_Year"] = X_test["Race"] + "_" + X_test["Year"]

freq_cols = ["Driver", "Race", "Compound", "Year", "Race_Year"]
combined = pd.concat([X_train[freq_cols], X_test[freq_cols]], axis=0, ignore_index=True)

for col in freq_cols:
    freq = combined[col].value_counts(normalize=True)
    X_train[f"{col}_freq"] = X_train[col].map(freq).astype("float32")
    X_test[f"{col}_freq"] = X_test[col].map(freq).astype("float32")
    if col in ["Driver", "Race_Year"]:
        X_train[f"{col}_log_freq"] = np.log1p(X_train[f"{col}_freq"]).astype("float32")
        X_test[f"{col}_log_freq"] = np.log1p(X_test[f"{col}_freq"]).astype("float32")

model_cat_cols = ["Driver", "Race", "Compound", "Year", "Race_Year"]
for col in model_cat_cols:
    categories = pd.Index(pd.concat([X_train[col], X_test[col]], axis=0).unique())
    X_train[col] = pd.Categorical(X_train[col], categories=categories)
    X_test[col] = pd.Categorical(X_test[col], categories=categories)

drop_cols = [id_col]
features = [c for c in X_train.columns if c not in drop_cols]

params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.04,
    num_leaves=64,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=3000,
    random_state=47,
    n_jobs=-1,
    verbosity=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
oof = np.zeros(len(X_train), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train.iloc[tr_idx][features],
        y.iloc[tr_idx],
        eval_set=[(X_train.iloc[va_idx][features], y.iloc[va_idx])],
        eval_metric="auc",
        categorical_feature=model_cat_cols,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )

    va_pred = model.predict_proba(X_train.iloc[va_idx][features])[:, 1]
    oof[va_idx] = va_pred
    fold_auc = roc_auc_score(y.iloc[va_idx], va_pred)
    fold_scores.append(float(fold_auc))

    test_pred += model.predict_proba(X_test[features])[:, 1] / skf.n_splits
    print(f"fold {fold} roc_auc: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)

submission = sample.copy()
submission[target_col] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y.values,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        id_col: test_ids.values,
        target_col: submission[target_col].values,
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": fold_scores,
            "research_hypotheses_llm_claimed_used": ["000047"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        },
        indent=2,
    )
)
