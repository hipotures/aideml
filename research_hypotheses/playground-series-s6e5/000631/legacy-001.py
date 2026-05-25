import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int)
test_ids = sample[ID_COL].copy()


def add_features(df):
    out = df.copy()
    out["TyreLife_x_RaceProgress"] = out["TyreLife"] * out["RaceProgress"]
    out["TyreLife_per_Stint"] = out["TyreLife"] / (out["Stint"].clip(lower=1))
    out["Degradation_per_TyreLife"] = out["Cumulative_Degradation"] / (
        out["TyreLife"].clip(lower=1)
    )
    out["LapProgress_Left"] = 1.0 - out["RaceProgress"]
    out["LapNumber_x_Stint"] = out["LapNumber"] * out["Stint"]
    return out


train_fe = add_features(train.drop(columns=[TARGET]))
test_fe = add_features(test)

# Keep id out of modeling: it is mostly a split marker, not a racing signal.
features = [c for c in train_fe.columns if c != ID_COL]
cat_cols = [c for c in features if train_fe[c].dtype == "object"]
cat_idx = [features.index(c) for c in cat_cols]

X_train = train_fe[features].copy()
X_test = test_fe[features].copy()

for c in cat_cols:
    X_train[c] = X_train[c].astype(str).fillna("missing")
    X_test[c] = X_test[c].astype(str).fillna("missing")

# Hypothesis 000631: adversarial train-test weighting.
adv_X = pd.concat([X_train, X_test], axis=0, ignore_index=True)
adv_y = np.r_[np.zeros(len(X_train), dtype=int), np.ones(len(X_test), dtype=int)]

adv_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.08,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=RANDOM_STATE,
    verbose=False,
    allow_writing_files=False,
)

adv_model.fit(Pool(adv_X, adv_y, cat_features=cat_idx))
train_test_likeness = adv_model.predict_proba(Pool(X_train, cat_features=cat_idx))[:, 1]

# Stabilize weights so a few extreme rows do not dominate training.
lo, hi = np.quantile(train_test_likeness, [0.02, 0.98])
weights = np.clip(train_test_likeness, lo, hi)
weights = weights / weights.mean()

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
    model = CatBoostClassifier(
        iterations=900,
        learning_rate=0.045,
        depth=7,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE + fold,
        verbose=False,
        allow_writing_files=False,
    )

    train_pool = Pool(
        X_train.iloc[tr_idx],
        y.iloc[tr_idx],
        cat_features=cat_idx,
        weight=weights[tr_idx],
    )
    valid_pool = Pool(
        X_train.iloc[va_idx],
        y.iloc[va_idx],
        cat_features=cat_idx,
    )

    model.fit(
        train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=80
    )

    oof[va_idx] = model.predict_proba(valid_pool)[:, 1]
    test_pred += (
        model.predict_proba(Pool(X_test, cat_features=cat_idx))[:, 1] / N_SPLITS
    )

    fold_auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])
    fold_scores.append(fold_auc)
    print(f"fold {fold} roc_auc: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"mean fold roc_auc: {np.mean(fold_scores):.6f}")
print(f"overall oof roc_auc: {cv_auc:.6f}")
print(
    f"adversarial train-test classifier train_auc: {roc_auc_score(adv_y, adv_model.predict_proba(Pool(adv_X, cat_features=cat_idx))[:, 1]):.6f}"
)

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0.0, 1.0)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y.values,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: submission[TARGET].values,
    }
)
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "mean_fold_roc_auc": float(np.mean(fold_scores)),
    "fold_roc_auc": [float(x) for x in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000631"],
}
with open(os.path.join(WORKING_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
