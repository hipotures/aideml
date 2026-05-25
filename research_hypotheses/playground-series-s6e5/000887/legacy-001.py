import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
features = [c for c in train.columns if c not in [TARGET, ID_COL]]

for c in CAT_COLS:
    all_vals = pd.concat([train[c], test[c]], axis=0).astype(str)
    cats = pd.Categorical(all_vals).categories
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

X = train[features].copy()
X_test = test[features].copy()
groups = train["Year"].astype(str) + "|" + train["Race"].astype(str)


def make_model(seed=42):
    return LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.045,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


logo = LeaveOneGroupOut()
oof_logo = np.zeros(len(train), dtype=float)
logo_scores = []

for fold, (tr_idx, va_idx) in enumerate(logo.split(X, y, groups), 1):
    model = make_model(1000 + fold)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        categorical_feature=CAT_COLS,
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof_logo[va_idx] = pred
    if len(np.unique(y[va_idx])) == 2:
        logo_scores.append(roc_auc_score(y[va_idx], pred))

logo_auc = roc_auc_score(y, oof_logo)

group_order = (
    train[[ID_COL, "Year", "Race"]]
    .assign(group=groups)
    .groupby("group", as_index=False)
    .agg(year=("Year", "min"), first_id=(ID_COL, "min"))
    .sort_values(["year", "first_id"])["group"]
    .tolist()
)

blocked_pred = np.full(len(train), np.nan, dtype=float)
blocked_scores = []
min_train_groups = max(5, int(0.25 * len(group_order)))

for fold, valid_group in enumerate(group_order[min_train_groups:], 1):
    train_groups = set(group_order[: min_train_groups + fold - 1])
    tr_idx = np.where(groups.isin(train_groups).values)[0]
    va_idx = np.where((groups == valid_group).values)[0]

    if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
        continue

    model = make_model(2000 + fold)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        categorical_feature=CAT_COLS,
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    blocked_pred[va_idx] = pred
    blocked_scores.append(roc_auc_score(y[va_idx], pred))

blocked_mask = ~np.isnan(blocked_pred)
blocked_auc = roc_auc_score(y[blocked_mask], blocked_pred[blocked_mask])

val_out = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_logo,
    }
)
val_out.to_csv(
    os.path.join(WORKING_DIR, "validation_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_model = make_model(777)
final_model.fit(X, y, categorical_feature=CAT_COLS)
test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(f"LeaveOneGroupOut Year|Race ROC AUC: {logo_auc:.6f}")
print(f"Mean per-held-race LOGO ROC AUC: {np.mean(logo_scores):.6f}")
print(f"Blocked rolling Year|Race ROC AUC: {blocked_auc:.6f}")
print(f"Mean per-block ROC AUC: {np.mean(blocked_scores):.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000887"],
            "logo_auc": float(logo_auc),
            "blocked_rolling_auc": float(blocked_auc),
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        },
        indent=2,
    )
)
