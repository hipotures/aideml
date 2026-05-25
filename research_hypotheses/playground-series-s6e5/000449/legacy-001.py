import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"

os.makedirs(WORK_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_group_splits(X, y, groups, n_splits=5):
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED
        )
        return list(splitter.split(X, y, groups))
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(X, y, groups))


def make_shift_frame(train_df, test_df):
    both = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    train_drivers = set(train_df["Driver"].astype(str))
    test_drivers = set(test_df["Driver"].astype(str))

    rp = pd.cut(
        both["RaceProgress"].clip(0, 1),
        bins=np.linspace(0, 1, 21),
        include_lowest=True,
    ).astype(str)

    return pd.DataFrame(
        {
            "Year": both["Year"].astype(str),
            "Race": both["Race"].astype(str),
            "Compound": both["Compound"].astype(str),
            "RaceProgressBin": rp,
            "DriverInTrain": both["Driver"].astype(str).isin(train_drivers).astype(str),
            "DriverInTest": both["Driver"].astype(str).isin(test_drivers).astype(str),
        }
    )


def estimate_importance_weights(train_df, test_df):
    n_train = len(train_df)
    n_test = len(test_df)

    shift_X = make_shift_frame(train_df, test_df)
    domain_y = np.r_[np.zeros(n_train, dtype=int), np.ones(n_test, dtype=int)]
    train_domain_oof = np.zeros(n_train, dtype=float)

    domain_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in domain_cv.split(shift_X, domain_y):
        pipe = Pipeline(
            steps=[
                (
                    "ohe",
                    ColumnTransformer([("cat", make_ohe(), shift_X.columns.tolist())]),
                ),
                (
                    "lr",
                    LogisticRegression(
                        C=0.5,
                        max_iter=500,
                        solver="lbfgs",
                        n_jobs=min(4, os.cpu_count() or 1),
                    ),
                ),
            ]
        )
        pipe.fit(shift_X.iloc[tr_idx], domain_y[tr_idx])

        train_va = va_idx[va_idx < n_train]
        if len(train_va):
            train_domain_oof[train_va] = pipe.predict_proba(shift_X.iloc[train_va])[
                :, 1
            ]

    missing = train_domain_oof <= 0
    if missing.any():
        pipe.fit(shift_X, domain_y)
        train_domain_oof[missing] = pipe.predict_proba(
            shift_X.iloc[:n_train].iloc[missing]
        )[:, 1]

    p_test = np.clip(train_domain_oof, 1e-4, 1 - 1e-4)
    weights = (p_test / (1 - p_test)) * (n_train / n_test)
    weights = np.clip(weights, 0.05, 20.0)
    weights = weights / np.mean(weights)
    return weights


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
groups = train["Race"].astype(str) + "_" + train["Year"].astype(str)

importance_weights = estimate_importance_weights(train, test)

features = [c for c in train.columns if c not in [TARGET, ID_COL]]
combined_X = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
cat_cols = combined_X.select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols:
    combined_X[col] = combined_X[col].astype("category")

X_train = combined_X.iloc[: len(train)].reset_index(drop=True)
X_test = combined_X.iloc[len(train) :].reset_index(drop=True)

splits = make_group_splits(X_train, y, groups, N_SPLITS)
oof_raw = np.zeros(len(train), dtype=float)
test_raw = np.zeros(len(test), dtype=float)
fold_weighted_auc = []

params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 1600,
    "learning_rate": 0.035,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.2,
    "reg_lambda": 8.0,
    "random_state": SEED,
    "n_jobs": os.cpu_count() or 1,
    "verbosity": -1,
}

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train.iloc[tr_idx],
        y[tr_idx],
        sample_weight=importance_weights[tr_idx],
        eval_set=[(X_train.iloc[va_idx], y[va_idx])],
        eval_sample_weight=[importance_weights[va_idx]],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    va_raw = model.predict(X_train.iloc[va_idx], raw_score=True)
    oof_raw[va_idx] = va_raw
    test_raw += model.predict(X_test, raw_score=True) / len(splits)

    auc = roc_auc_score(y[va_idx], va_raw, sample_weight=importance_weights[va_idx])
    fold_weighted_auc.append(float(auc))
    print(f"Fold {fold} weighted ROC AUC: {auc:.6f}")

cal_oof = np.zeros(len(train), dtype=float)
cal_splits = make_group_splits(pd.DataFrame({"raw": oof_raw}), y, groups, N_SPLITS)

for tr_idx, va_idx in cal_splits:
    calibrator = LogisticRegression(C=1000.0, max_iter=1000, solver="lbfgs")
    calibrator.fit(
        oof_raw[tr_idx].reshape(-1, 1),
        y[tr_idx],
        sample_weight=importance_weights[tr_idx],
    )
    cal_oof[va_idx] = calibrator.predict_proba(oof_raw[va_idx].reshape(-1, 1))[:, 1]

final_calibrator = LogisticRegression(C=1000.0, max_iter=1000, solver="lbfgs")
final_calibrator.fit(
    oof_raw.reshape(-1, 1),
    y,
    sample_weight=importance_weights,
)

test_pred = final_calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

weighted_auc = roc_auc_score(y, cal_oof, sample_weight=importance_weights)
unweighted_auc = roc_auc_score(y, cal_oof)

print(f"Weighted race-year CV ROC AUC: {weighted_auc:.6f}")
print(f"Unweighted race-year CV ROC AUC: {unweighted_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": cal_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "weighted_race_year_group_cv_roc_auc",
    "weighted_race_year_group_cv_roc_auc": float(weighted_auc),
    "unweighted_race_year_group_cv_roc_auc": float(unweighted_auc),
    "fold_weighted_roc_auc": fold_weighted_auc,
    "research_hypotheses_llm_claimed_used": ["000449"],
}
print(json.dumps(result, sort_keys=True))
