import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
n_threads = max(1, os.cpu_count() or 1)

train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")


def add_features(df):
    out = df.copy()
    for c in ["Driver", "Race", "Compound"]:
        out[c] = out[c].astype("string").fillna("__NA__").astype(str)

    year_s = out["Year"].astype(str)
    stint_s = out["Stint"].astype(str)

    out["Year_cat"] = year_s
    out["Year_Race"] = year_s + "__" + out["Race"]
    out["Race_Compound"] = out["Race"] + "__" + out["Compound"]
    out["Driver_Compound"] = out["Driver"] + "__" + out["Compound"]
    out["Race_Driver"] = out["Race"] + "__" + out["Driver"]
    out["Compound_Stint"] = out["Compound"] + "__" + stint_s
    return out


train_fe = add_features(train)
test_fe = add_features(test)
y = train_fe[TARGET].astype(int).to_numpy()

base_cat_cols = ["Driver", "Race", "Compound", "Year_cat"]
sidecar_cat_cols = base_cat_cols + [
    "Year_Race",
    "Race_Compound",
    "Driver_Compound",
    "Race_Driver",
    "Compound_Stint",
]
cat_set = set(sidecar_cat_cols)

numeric_cols = [
    c
    for c in train_fe.columns
    if c not in {ID_COL, TARGET}
    and c not in cat_set
    and pd.api.types.is_numeric_dtype(train_fe[c])
]

lgb_feature_cols = numeric_cols + base_cat_cols
cb_feature_cols = numeric_cols + sidecar_cat_cols

X_lgb_train = train_fe[lgb_feature_cols].copy()
X_lgb_test = test_fe[lgb_feature_cols].copy()
X_cb_train = train_fe[cb_feature_cols].copy()
X_cb_test = test_fe[cb_feature_cols].copy()

for c in numeric_cols:
    X_lgb_train[c] = X_lgb_train[c].astype(np.float32)
    X_lgb_test[c] = X_lgb_test[c].astype(np.float32)
    X_cb_train[c] = X_cb_train[c].astype(np.float32)
    X_cb_test[c] = X_cb_test[c].astype(np.float32)

for c in base_cat_cols:
    all_values = pd.concat([X_lgb_train[c], X_lgb_test[c]], ignore_index=True).astype(
        str
    )
    dtype = pd.CategoricalDtype(categories=pd.Index(all_values.unique()))
    X_lgb_train[c] = X_lgb_train[c].astype(str).astype(dtype)
    X_lgb_test[c] = X_lgb_test[c].astype(str).astype(dtype)

for c in sidecar_cat_cols:
    X_cb_train[c] = X_cb_train[c].astype(str)
    X_cb_test[c] = X_cb_test[c].astype(str)


def make_blocked_folds(df, n_splits=5):
    groups = (df["Year"].astype(str) + "__" + df["Race"].astype(str)).to_numpy()
    group_order = (
        pd.DataFrame({"group": groups, "id": df[ID_COL].to_numpy()})
        .groupby("group", sort=False)["id"]
        .min()
        .sort_values()
        .index.to_numpy()
    )
    folds = []
    group_series = pd.Series(groups)
    for valid_groups in np.array_split(group_order, n_splits):
        valid_mask = group_series.isin(valid_groups).to_numpy()
        tr_idx = np.flatnonzero(~valid_mask)
        va_idx = np.flatnonzero(valid_mask)
        folds.append((tr_idx, va_idx))
    return folds, groups


folds, groups = make_blocked_folds(train_fe, N_SPLITS)


def safe_auc(y_true, pred):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, pred))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log1p(-p)


lgb_oof = np.zeros(len(train_fe), dtype=np.float64)
cb_oof = np.zeros(len(train_fe), dtype=np.float64)
lgb_test_preds = []
cb_test_preds = []

cb_test_pool = Pool(X_cb_test, cat_features=sidecar_cat_cols)

for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
    y_tr = y[tr_idx]
    pos = max(1, int(y_tr.sum()))
    neg = max(1, len(y_tr) - pos)
    scale_pos_weight = min(50.0, neg / pos)

    lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.035,
        "num_leaves": 63,
        "max_depth": 7,
        "min_data_in_leaf": 80,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 4.0,
        "cat_smooth": 20.0,
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "seed": SEED + fold,
        "num_threads": n_threads,
        "force_col_wise": True,
    }

    lgb_train = lgb.Dataset(
        X_lgb_train.iloc[tr_idx],
        label=y[tr_idx],
        categorical_feature=base_cat_cols,
        free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        X_lgb_train.iloc[va_idx],
        label=y[va_idx],
        categorical_feature=base_cat_cols,
        reference=lgb_train,
        free_raw_data=False,
    )

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=1200,
        valid_sets=[lgb_valid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    best_iter = lgb_model.best_iteration or 1200
    lgb_oof[va_idx] = lgb_model.predict(
        X_lgb_train.iloc[va_idx], num_iteration=best_iter
    )
    lgb_test_preds.append(lgb_model.predict(X_lgb_test, num_iteration=best_iter))

    cb_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 700,
        "learning_rate": 0.045,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "random_strength": 1.0,
        "boosting_type": "Ordered",
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "one_hot_max_size": 2,
        "max_ctr_complexity": 1,
        "scale_pos_weight": scale_pos_weight,
        "od_type": "Iter",
        "od_wait": 80,
        "random_seed": SEED + 100 + fold,
        "thread_count": n_threads,
        "allow_writing_files": False,
        "verbose": False,
    }

    cb_train_pool = Pool(
        X_cb_train.iloc[tr_idx], y[tr_idx], cat_features=sidecar_cat_cols
    )
    cb_valid_pool = Pool(
        X_cb_train.iloc[va_idx], y[va_idx], cat_features=sidecar_cat_cols
    )

    cb_model = CatBoostClassifier(**cb_params)
    cb_model.fit(
        cb_train_pool, eval_set=cb_valid_pool, use_best_model=True, verbose=False
    )

    cb_oof[va_idx] = cb_model.predict_proba(cb_valid_pool)[:, 1]
    cb_test_preds.append(cb_model.predict_proba(cb_test_pool)[:, 1])

    print(
        f"Fold {fold}: "
        f"LGB AUC={safe_auc(y[va_idx], lgb_oof[va_idx]):.6f}, "
        f"CatBoost sidecar AUC={safe_auc(y[va_idx], cb_oof[va_idx]):.6f}, "
        f"valid_rows={len(va_idx)}, valid_groups={len(np.unique(groups[va_idx]))}"
    )

lgb_test = np.mean(np.vstack(lgb_test_preds), axis=0)
cb_test = np.mean(np.vstack(cb_test_preds), axis=0)

meta_train = np.column_stack([logit(lgb_oof), logit(cb_oof)])
meta_test = np.column_stack([logit(lgb_test), logit(cb_test)])

blend_oof = np.zeros(len(train_fe), dtype=np.float64)
for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
    blender = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    blender.fit(meta_train[tr_idx], y[tr_idx])
    blend_oof[va_idx] = blender.predict_proba(meta_train[va_idx])[:, 1]

final_blender = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
final_blender.fit(meta_train, y)
blend_test = final_blender.predict_proba(meta_test)[:, 1]
blend_test = np.clip(blend_test, 0.0, 1.0)

lgb_auc = safe_auc(y, lgb_oof)
cb_auc = safe_auc(y, cb_oof)
blend_auc = safe_auc(y, blend_oof)

sample_id_col = sample.columns[0]
sample_target_col = sample.columns[1]

if np.array_equal(sample[sample_id_col].to_numpy(), test[ID_COL].to_numpy()):
    ordered_test_pred = blend_test
else:
    pred_map = pd.Series(blend_test, index=test[ID_COL].to_numpy())
    ordered_test_pred = sample[sample_id_col].map(pred_map).to_numpy()

submission = pd.DataFrame(
    {
        sample_id_col: sample[sample_id_col].to_numpy(),
        sample_target_col: ordered_test_pred,
    }
)
submission.to_csv(WORK / "submission.csv", index=False)
submission.to_csv(WORK / "test_predictions.csv.gz", index=False, compression="gzip")

oof = pd.DataFrame(
    {
        "row": np.arange(len(train_fe), dtype=np.int64),
        "target": y.astype(int),
        "prediction": blend_oof,
    }
)
oof.to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

components = pd.DataFrame(
    {
        "row": np.arange(len(train_fe), dtype=np.int64),
        "target": y.astype(int),
        "lgb_prediction": lgb_oof,
        "catboost_sidecar_prediction": cb_oof,
        "prediction": blend_oof,
    }
)
components.to_csv(
    WORK / "oof_component_predictions.csv.gz", index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000546"],
    "metric": "roc_auc",
    "cv_scheme": "5-fold contiguous blocked CV by Year/Race ordered by first id",
    "lightgbm_oof_auc": lgb_auc,
    "catboost_sidecar_oof_auc": cb_auc,
    "sigmoid_blocked_blend_oof_auc": blend_auc,
    "submission_path": str(WORK / "submission.csv"),
    "oof_predictions_path": str(WORK / "oof_predictions.csv.gz"),
    "test_predictions_path": str(WORK / "test_predictions.csv.gz"),
}

for name in ["result.json", "review.json"]:
    with open(WORK / name, "w") as f:
        json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
