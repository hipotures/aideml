import os
import gc
import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
ID_COL = "id"
TARGET = "PitNextLap"
SEED = 20260523
N_JOBS = min(8, os.cpu_count() or 1)
REGIMES = ("dry", "wet", "testing")
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

os.makedirs(WORKING_DIR, exist_ok=True)


def add_regime(df):
    df = df.copy()
    race = df["Race"].astype(str)
    compound = df["Compound"].astype(str)
    is_testing = race.eq("Pre-Season Testing")
    is_wet = compound.isin(WET_COMPOUNDS)
    df["Regime"] = np.where(is_testing, "testing", np.where(is_wet, "wet", "dry"))
    df["IsTesting"] = is_testing.astype(np.int8)
    df["IsWetCompound"] = is_wet.astype(np.int8)
    return df


def align_categories(train_df, test_df, feature_cols):
    cat_cols = (
        train_df[feature_cols]
        .select_dtypes(include=["object", "category"])
        .columns.tolist()
    )
    for col in cat_cols:
        both = pd.concat(
            [train_df[col].astype(str), test_df[col].astype(str)], ignore_index=True
        )
        cats = pd.Index(both.unique())
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
        test_df[col] = pd.Categorical(test_df[col].astype(str), categories=cats)
    return train_df, test_df, cat_cols


def make_target_model(seed):
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=420,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=4.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
    )


def make_domain_model(seed):
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=140,
        learning_rate=0.05,
        num_leaves=24,
        min_child_samples=250,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_lambda=8.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
    )


def fit_lgbm(model, X, y, weights, cat_cols):
    model.fit(X, y, sample_weight=weights, categorical_feature=cat_cols)
    return model


def compute_shift_weights(X_source, X_target, cat_cols, seed):
    n_source, n_target = len(X_source), len(X_target)
    domain_X = pd.concat([X_source, X_target], axis=0, ignore_index=True)
    domain_y = np.r_[
        np.zeros(n_source, dtype=np.int8), np.ones(n_target, dtype=np.int8)
    ]

    model = make_domain_model(seed)
    model.fit(domain_X, domain_y, categorical_feature=cat_cols)

    p_test_domain = model.predict_proba(domain_X.iloc[:n_source])[:, 1]
    p_test_domain = np.clip(p_test_domain, 1e-4, 1 - 1e-4)
    weights = (p_test_domain / (1.0 - p_test_domain)) * (n_source / n_target)
    weights = np.clip(weights, 0.25, 4.0)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def fit_regime_system(X, y, regimes, weights, cat_cols, seed):
    pooled = fit_lgbm(make_target_model(seed), X, y, weights, cat_cols)
    models = {}

    regime_values = regimes.astype(str).to_numpy()
    for i, regime in enumerate(REGIMES):
        mask = regime_values == regime
        y_regime = y[mask]
        enough_rows = mask.sum() >= 1500
        enough_classes = len(np.unique(y_regime)) == 2 and y_regime.sum() >= 20

        if enough_rows and enough_classes:
            models[regime] = fit_lgbm(
                make_target_model(seed + 101 + i),
                X.iloc[mask],
                y_regime,
                weights[mask],
                cat_cols,
            )
        else:
            models[regime] = None

    return pooled, models


def predict_regime_system(pooled, models, X, regimes):
    preds = pooled.predict_proba(X)[:, 1]
    regime_values = regimes.astype(str).to_numpy()

    for regime, model in models.items():
        mask = regime_values == regime
        if model is not None and mask.any():
            preds[mask] = model.predict_proba(X.iloc[mask])[:, 1]

    return np.clip(preds, 1e-6, 1 - 1e-6)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = add_regime(train)
test = add_regime(test)

feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET)]
train, test, cat_cols = align_categories(train, test, feature_cols)

X = train[feature_cols]
X_test = test[feature_cols]
y = train[TARGET].astype(int).to_numpy()
groups = train["Year"].to_numpy()

logo = LeaveOneGroupOut()
oof = np.zeros(len(train), dtype=np.float32)
fold_auc = {}

for fold, (tr_idx, va_idx) in enumerate(logo.split(X, y, groups=groups), 1):
    val_year = int(groups[va_idx][0])
    print(f"Fold {fold}: validating Year={val_year}, rows={len(va_idx)}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    r_tr, r_va = train["Regime"].iloc[tr_idx], train["Regime"].iloc[va_idx]

    weights = compute_shift_weights(X_tr, X_test, cat_cols, SEED + fold)
    pooled, models = fit_regime_system(
        X_tr, y_tr, r_tr, weights, cat_cols, SEED + 10 * fold
    )
    preds = predict_regime_system(pooled, models, X_va, r_va)

    oof[va_idx] = preds
    fold_auc[str(val_year)] = float(roc_auc_score(y_va, preds))
    print(f"  Year {val_year} ROC AUC: {fold_auc[str(val_year)]:.6f}")

    del weights, pooled, models
    gc.collect()

overall_auc = float(roc_auc_score(y, oof))
print(f"Leave-one-year-out ROC AUC: {overall_auc:.6f}")

oof_path = os.path.join(WORKING_DIR, "oof_predictions.csv.gz")
pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(oof_path, index=False, compression="gzip")

full_weights = compute_shift_weights(X, X_test, cat_cols, SEED + 999)
pooled, models = fit_regime_system(
    X, y, train["Regime"], full_weights, cat_cols, SEED + 5000
)
test_preds = predict_regime_system(pooled, models, X_test, test["Regime"])

submission = sample.copy()
submission[TARGET] = test_preds
submission_path = os.path.join(WORKING_DIR, "submission.csv")
test_pred_path = os.path.join(WORKING_DIR, "test_predictions.csv.gz")

submission.to_csv(submission_path, index=False)
submission.to_csv(test_pred_path, index=False, compression="gzip")

regime_auc = {}
train_regimes = train["Regime"].astype(str).to_numpy()
for regime in REGIMES:
    mask = train_regimes == regime
    if mask.sum() > 0 and len(np.unique(y[mask])) == 2:
        regime_auc[regime] = float(roc_auc_score(y[mask], oof[mask]))

review = {
    "metric": "roc_auc",
    "validation_strategy": "LeaveOneGroupOut by Year",
    "validation_auc": overall_auc,
    "fold_auc": fold_auc,
    "regime_auc": regime_auc,
    "research_hypotheses_llm_claimed_used": ["000324"],
    "files_written": {
        "submission": submission_path,
        "oof_predictions": oof_path,
        "test_predictions": test_pred_path,
    },
}

print(json.dumps(review, indent=2))
