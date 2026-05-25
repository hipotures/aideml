import os
import json
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None


SEED = 42
N_SPLITS = 5
INNER_SPLITS = 4
TARGET = "PitNextLap"
ID_COL = "id"
TE_COL = "te_RaceYear_Compound"
MIN_COUNT = 30
SMOOTHING = 75.0

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)


def as_key(values):
    return (
        pd.Series(values, copy=False)
        .fillna("__NA__")
        .astype(str)
        .reset_index(drop=True)
    )


def add_derived(df):
    df = df.copy()
    race = as_key(df["Race"])
    year = as_key(df["Year"])
    compound = as_key(df["Compound"])
    driver = as_key(df["Driver"])

    df["Race_Year"] = race + "|" + year
    df["RaceYear_Compound"] = df["Race_Year"] + "|" + compound
    df["cv_group"] = df["Race_Year"] + "|" + driver
    return df


def add_ordinal_codes(train, test, cols):
    code_cols = []
    for col in cols:
        all_values = pd.concat(
            [as_key(train[col]), as_key(test[col])], ignore_index=True
        )
        uniques = pd.Index(pd.unique(all_values))
        mapping = pd.Series(np.arange(len(uniques), dtype=np.int32), index=uniques)

        code_col = f"{col}_code"
        train[code_col] = (
            as_key(train[col]).map(mapping).fillna(-1).astype(np.int32).values
        )
        test[code_col] = (
            as_key(test[col]).map(mapping).fillna(-1).astype(np.int32).values
        )
        code_cols.append(code_col)
    return code_cols


def make_grouped_splits(y, groups, n_splits, seed):
    groups = as_key(groups)
    n_splits = min(int(n_splits), int(groups.nunique()))
    if n_splits < 2:
        raise ValueError("Need at least two groups for grouped validation.")

    x_dummy = np.zeros(len(y), dtype=np.int8)

    if StratifiedGroupKFold is not None:
        try:
            cv = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            return list(cv.split(x_dummy, y, groups))
        except Exception as exc:
            warnings.warn(
                f"Falling back to GroupKFold because StratifiedGroupKFold failed: {exc}"
            )

    cv = GroupKFold(n_splits=n_splits)
    return list(cv.split(x_dummy, y, groups))


def fit_smoothed_te(keys, y, min_count=MIN_COUNT, smoothing=SMOOTHING):
    keys = as_key(keys)
    y = np.asarray(y, dtype=np.float32)
    prior = float(np.mean(y))

    tmp = pd.DataFrame({"key": keys, "target": y})
    stats = tmp.groupby("key", sort=False)["target"].agg(["count", "mean"])
    encoded = (stats["count"] * stats["mean"] + smoothing * prior) / (
        stats["count"] + smoothing
    )
    encoded = encoded.where(stats["count"] >= min_count, prior)
    return encoded.astype(np.float32), prior


def transform_smoothed_te(keys, mapping, prior):
    return as_key(keys).map(mapping).fillna(prior).astype(np.float32).to_numpy()


def make_oof_te(keys, y, groups, n_splits, seed):
    keys = as_key(keys)
    groups = as_key(groups)
    y = np.asarray(y, dtype=np.float32)

    oof = np.full(len(y), float(np.mean(y)), dtype=np.float32)
    splits = make_grouped_splits(y, groups, n_splits, seed)

    for tr_idx, va_idx in splits:
        mapping, prior = fit_smoothed_te(keys.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = transform_smoothed_te(keys.iloc[va_idx], mapping, prior)

    return oof


def make_model(seed, n_estimators=1600):
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=int(n_estimators),
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=max(1, os.cpu_count() or 1),
        force_col_wise=True,
        verbosity=-1,
    )


def fit_lgbm(model, x_tr, y_tr, cat_cols, x_va=None, y_va=None):
    fit_kwargs = {"categorical_feature": cat_cols}

    if x_va is not None:
        fit_kwargs.update(
            {
                "eval_set": [(x_va, y_va)],
                "eval_metric": "auc",
                "callbacks": [
                    lgb.early_stopping(100, verbose=False),
                    lgb.log_evaluation(0),
                ],
            }
        )

    model.fit(x_tr, y_tr, **fit_kwargs)
    return model


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = add_derived(train)
test = add_derived(test)

cat_cols = ["Compound", "Driver", "Race", "Race_Year"]
cat_code_cols = add_ordinal_codes(train, test, cat_cols)

exclude = {
    ID_COL,
    TARGET,
    "Race_Year",
    "RaceYear_Compound",
    "cv_group",
    *cat_cols,
    *cat_code_cols,
}
numeric_cols = [
    c
    for c in train.columns
    if c not in exclude and pd.api.types.is_numeric_dtype(train[c])
]
base_cols = numeric_cols + cat_code_cols
feature_cols = base_cols + [TE_COL]

x_train_base = train[base_cols].copy()
x_test_base = test[base_cols].copy()

y = train[TARGET].astype(int).to_numpy()
groups = train["cv_group"]

outer_splits = make_grouped_splits(y, groups, N_SPLITS, SEED)
oof_pred = np.zeros(len(train), dtype=np.float32)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(outer_splits, start=1):
    x_tr = x_train_base.iloc[tr_idx].copy()
    x_va = x_train_base.iloc[va_idx].copy()

    x_tr[TE_COL] = make_oof_te(
        train["RaceYear_Compound"].iloc[tr_idx],
        y[tr_idx],
        train["cv_group"].iloc[tr_idx],
        INNER_SPLITS,
        SEED + 100 + fold,
    )

    fold_mapping, fold_prior = fit_smoothed_te(
        train["RaceYear_Compound"].iloc[tr_idx],
        y[tr_idx],
    )
    x_va[TE_COL] = transform_smoothed_te(
        train["RaceYear_Compound"].iloc[va_idx],
        fold_mapping,
        fold_prior,
    )

    model = make_model(SEED + fold)
    model = fit_lgbm(
        model,
        x_tr[feature_cols],
        y[tr_idx],
        cat_code_cols,
        x_va[feature_cols],
        y[va_idx],
    )

    pred = model.predict_proba(x_va[feature_cols])[:, 1]
    oof_pred[va_idx] = pred.astype(np.float32)

    fold_auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(fold_auc))
    best_iterations.append(
        int(getattr(model, "best_iteration_", None) or model.n_estimators)
    )
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

oof_auc = roc_auc_score(y, oof_pred)
print(f"OOF ROC AUC: {oof_auc:.6f}")

x_full = x_train_base.copy()
x_full[TE_COL] = make_oof_te(
    train["RaceYear_Compound"],
    y,
    train["cv_group"],
    N_SPLITS,
    SEED + 999,
)

full_mapping, full_prior = fit_smoothed_te(train["RaceYear_Compound"], y)
x_test = x_test_base.copy()
x_test[TE_COL] = transform_smoothed_te(
    test["RaceYear_Compound"], full_mapping, full_prior
)

final_estimators = max(100, int(np.median(best_iterations)))
final_model = make_model(SEED + 1000, n_estimators=final_estimators)
final_model = fit_lgbm(final_model, x_full[feature_cols], y, cat_code_cols)

test_pred = final_model.predict_proba(x_test[feature_cols])[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int32),
        "target": y.astype(int),
        "prediction": oof_pred,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

pred_by_id = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_pred})
submission = sample[[ID_COL]].merge(pred_by_id, on=ID_COL, how="left")
if submission[TARGET].isna().any():
    submission[TARGET] = submission[TARGET].fillna(float(np.mean(test_pred)))

submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "oof_roc_auc": float(oof_auc),
    "mean_fold_roc_auc": float(np.mean(fold_scores)),
    "fold_roc_auc": fold_scores,
    "n_outer_folds": len(outer_splits),
    "cv_group": "Race_Year|Driver",
    "target_encoding_features_used": ["Race_Year x Compound"],
    "target_encoding_min_count": MIN_COUNT,
    "target_encoding_smoothing": SMOOTHING,
    "final_n_estimators": final_estimators,
    "research_hypotheses_llm_claimed_used": ["000007"],
}

print(json.dumps(result, indent=2))
