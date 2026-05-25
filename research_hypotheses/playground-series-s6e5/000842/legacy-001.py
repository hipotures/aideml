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

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 842
N_FOLDS = 5

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values


def add_group_columns(df):
    df = df.copy()
    df["Stint"] = df["Stint"].astype(str)
    df["Year"] = df["Year"].astype(str)
    df["Race_Year"] = df["Race"].astype(str) + "__" + df["Year"].astype(str)
    df["Driver_Compound"] = df["Driver"].astype(str) + "__" + df["Compound"].astype(str)
    df["Race_Compound"] = df["Race"].astype(str) + "__" + df["Compound"].astype(str)
    return df


train_fe = add_group_columns(train.drop(columns=[TARGET]))
test_fe = add_group_columns(test)

pool_cols = [
    "Driver",
    "Race",
    "Compound",
    "Stint",
    "Race_Year",
    "Driver_Compound",
    "Race_Compound",
]
cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "Year",
    "Stint",
    "Race_Year",
    "Driver_Compound",
    "Race_Compound",
]


def eb_features(train_part, valid_part, target, cols, prior_strength=40.0):
    prior = float(np.mean(target))
    prior = np.clip(prior, 1e-5, 1 - 1e-5)
    out = pd.DataFrame(index=valid_part.index)

    tmp = train_part[cols].copy()
    tmp["_target_"] = target

    for col in cols:
        stats = tmp.groupby(col, observed=True)["_target_"].agg(["sum", "count"])
        post = (stats["sum"] + prior_strength * prior) / (
            stats["count"] + prior_strength
        )
        post = post.clip(1e-5, 1 - 1e-5)

        mapped = valid_part[col].map(post).fillna(prior).astype(float)
        out[f"eb_{col}_mean"] = mapped.values
        out[f"eb_{col}_logodds"] = np.log(mapped.values / (1.0 - mapped.values))

    return out


def make_base_features(df):
    drop_cols = [ID_COL]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    for col in cat_cols:
        X[col] = X[col].astype("category")
    return X


base_train = make_base_features(train_fe)
base_test = make_base_features(test_fe)

oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
fold_scores = []

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.035,
    "num_leaves": 48,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 2.0,
    "verbosity": -1,
    "seed": RANDOM_STATE,
    "num_threads": max(1, os.cpu_count() or 1),
}

for fold, (tr_idx, va_idx) in enumerate(cv.split(base_train, y), 1):
    tr_raw = train_fe.iloc[tr_idx].reset_index(drop=True)
    va_raw = train_fe.iloc[va_idx].reset_index(drop=True)
    te_raw = test_fe.reset_index(drop=True)

    tr_y = y[tr_idx]
    va_y = y[va_idx]

    tr_eb_inner = np.zeros((len(tr_idx), len(pool_cols) * 2), dtype=float)
    inner_cv = StratifiedKFold(
        n_splits=4, shuffle=True, random_state=RANDOM_STATE + fold
    )
    eb_col_names = None

    for inner_tr, inner_va in inner_cv.split(tr_raw, tr_y):
        inner_eb = eb_features(
            tr_raw.iloc[inner_tr],
            tr_raw.iloc[inner_va],
            tr_y[inner_tr],
            pool_cols,
        )
        if eb_col_names is None:
            eb_col_names = list(inner_eb.columns)
        tr_eb_inner[inner_va] = inner_eb.values

    tr_eb = pd.DataFrame(
        tr_eb_inner, columns=eb_col_names, index=base_train.iloc[tr_idx].index
    )
    va_eb = eb_features(tr_raw, va_raw, tr_y, pool_cols)
    va_eb.index = base_train.iloc[va_idx].index
    te_eb = eb_features(tr_raw, te_raw, tr_y, pool_cols)
    te_eb.index = base_test.index

    X_tr = pd.concat([base_train.iloc[tr_idx], tr_eb], axis=1)
    X_va = pd.concat([base_train.iloc[va_idx], va_eb], axis=1)
    X_te = pd.concat([base_test, te_eb], axis=1)

    model = lgb.LGBMClassifier(**params, n_estimators=4000)
    model.fit(
        X_tr,
        tr_y,
        eval_set=[(X_va, va_y)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )

    va_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
    te_pred = model.predict_proba(X_te, num_iteration=model.best_iteration_)[:, 1]

    oof[va_idx] = va_pred
    test_pred += te_pred / N_FOLDS

    auc = roc_auc_score(va_y, va_pred)
    fold_scores.append(auc)
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: submission[TARGET].values,
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "oof_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(v) for v in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000842"],
}
print(json.dumps(result, indent=2))
