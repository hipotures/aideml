import os
import json
import shutil
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
y = train[target_col].astype(int).values
features = [c for c in train.columns if c not in [target_col, id_col]]
cat_cols = (
    train[features].select_dtypes(include=["object", "category"]).columns.tolist()
)
num_cols = [c for c in features if c not in cat_cols]

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
splits = list(cv.split(train, y))
model_oof = {}
model_test = {}


def rank01(x):
    return pd.Series(x).rank(method="average").to_numpy(dtype=np.float64) / (
        len(x) + 1.0
    )


def safe_auc(name, preds):
    auc = roc_auc_score(y, preds)
    print(f"{name} OOF ROC AUC: {auc:.6f}")
    return auc


def add_predictions(name, oof, test_pred):
    model_oof[name] = np.asarray(oof, dtype=np.float64)
    model_test[name] = np.asarray(test_pred, dtype=np.float64)
    safe_auc(name, model_oof[name])


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def smooth_map(values, target, smoothing=20.0):
    df = pd.DataFrame({"v": values, "y": target})
    stat = df.groupby("v")["y"].agg(["sum", "count"])
    prior = float(np.mean(target))
    enc = (stat["sum"] + smoothing * prior) / (stat["count"] + smoothing)
    return enc, prior


def make_te_features(x_tr, y_tr, x_va, x_te, cat_columns, base_num_columns):
    tr_out = x_tr[base_num_columns].copy()
    va_out = x_va[base_num_columns].copy()
    te_out = x_te[base_num_columns].copy()

    inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    for col in cat_columns:
        tr_encoded = pd.Series(np.nan, index=x_tr.index, dtype=np.float64)
        for a, b in inner.split(x_tr, y_tr):
            idx_a = x_tr.index[a]
            idx_b = x_tr.index[b]
            enc, prior = smooth_map(x_tr.loc[idx_a, col], y_tr[a])
            tr_encoded.loc[idx_b] = (
                x_tr.loc[idx_b, col].map(enc).fillna(prior).astype(float)
            )

        enc_full, prior_full = smooth_map(x_tr[col], y_tr)
        tr_out[col + "_te"] = tr_encoded.fillna(prior_full).values
        va_out[col + "_te"] = (
            x_va[col].map(enc_full).fillna(prior_full).astype(float).values
        )
        te_out[col + "_te"] = (
            x_te[col].map(enc_full).fillna(prior_full).astype(float).values
        )

    return tr_out, va_out, te_out


def train_autogluon_variant():
    try:
        from autogluon.tabular import TabularPredictor
    except Exception as e:
        print(f"AutoGluon unavailable, skipping AutoGluon GBM/XGB variant: {e}")
        return

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    ag_root = os.path.join(WORK_DIR, "autogluon_rank_blend")
    shutil.rmtree(ag_root, ignore_errors=True)

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        fold_path = os.path.join(ag_root, f"fold_{fold}")
        tr_df = train.iloc[tr_idx][features + [target_col]].copy()
        va_df = train.iloc[va_idx][features + [target_col]].copy()

        predictor = TabularPredictor(
            label=target_col,
            eval_metric="roc_auc",
            path=fold_path,
            verbosity=0,
        )
        try:
            predictor.fit(
                train_data=tr_df,
                tuning_data=va_df,
                hyperparameters={"GBM": {}, "XGB": {}},
                time_limit=90,
                presets="medium_quality",
                num_cpus=max(1, min(8, os.cpu_count() or 1)),
                verbosity=0,
            )
            va_proba = predictor.predict_proba(va_df[features], as_multiclass=False)
            te_proba = predictor.predict_proba(test[features], as_multiclass=False)
            oof[va_idx] = np.asarray(va_proba, dtype=np.float64)
            test_pred += np.asarray(te_proba, dtype=np.float64) / N_SPLITS
        except Exception as e:
            print(f"AutoGluon fold {fold} failed: {e}")
            return

    add_predictions("autogluon_gbm_xgb", oof, test_pred)


def train_catboost_variant():
    from catboost import CatBoostClassifier, Pool

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)
    cat_idx = [features.index(c) for c in cat_cols]

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        x_tr = train.iloc[tr_idx][features].copy()
        x_va = train.iloc[va_idx][features].copy()
        x_te = test[features].copy()

        for c in cat_cols:
            x_tr[c] = x_tr[c].astype(str)
            x_va[c] = x_va[c].astype(str)
            x_te[c] = x_te[c].astype(str)

        model = CatBoostClassifier(
            iterations=700,
            learning_rate=0.045,
            depth=6,
            l2_leaf_reg=6.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=SEED + fold,
            verbose=False,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=60,
        )
        model.fit(
            Pool(x_tr, y[tr_idx], cat_features=cat_idx),
            eval_set=Pool(x_va, y[va_idx], cat_features=cat_idx),
            use_best_model=True,
        )
        oof[va_idx] = model.predict_proba(x_va)[:, 1]
        test_pred += model.predict_proba(x_te)[:, 1] / N_SPLITS

    add_predictions("catboost_native_cat", oof, test_pred)


def train_lgbm_te_variant():
    import lightgbm as lgb

    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.045,
        "num_leaves": 64,
        "min_data_in_leaf": 90,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 2.0,
        "verbosity": -1,
        "seed": SEED,
        "num_threads": max(1, min(12, os.cpu_count() or 1)),
    }

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        x_tr, x_va, x_te = make_te_features(
            train.iloc[tr_idx][features],
            y[tr_idx],
            train.iloc[va_idx][features],
            test[features],
            cat_cols,
            num_cols,
        )

        dtrain = lgb.Dataset(x_tr, label=y[tr_idx])
        dvalid = lgb.Dataset(x_va, label=y[va_idx], reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=900,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va_idx] = model.predict(x_va, num_iteration=model.best_iteration)
        test_pred += model.predict(x_te, num_iteration=model.best_iteration) / N_SPLITS

    add_predictions("lightgbm_oof_target_encoding", oof, test_pred)


def train_logistic_variant():
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", make_ohe(), cat_cols),
        ],
        remainder="drop",
    )

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        clf = Pipeline(
            steps=[
                ("pre", pre),
                (
                    "lr",
                    LogisticRegression(
                        C=0.7,
                        solver="saga",
                        penalty="l2",
                        class_weight="balanced",
                        max_iter=250,
                        random_state=SEED + fold,
                        n_jobs=max(1, min(8, os.cpu_count() or 1)),
                    ),
                ),
            ]
        )
        clf.fit(train.iloc[tr_idx][features], y[tr_idx])
        oof[va_idx] = clf.predict_proba(train.iloc[va_idx][features])[:, 1]
        test_pred += clf.predict_proba(test[features])[:, 1] / N_SPLITS

    add_predictions("regularized_logistic_ohe", oof, test_pred)


def weight_grid(n_models, units=5):
    out = []

    def rec(prefix, remaining, k):
        if k == 1:
            out.append(prefix + [remaining])
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, k - 1)

    rec([], units, n_models)
    return np.asarray(out, dtype=np.float64) / units


train_autogluon_variant()
train_catboost_variant()
train_lgbm_te_variant()
train_logistic_variant()

if len(model_oof) < 2:
    raise RuntimeError(
        "Need at least two successful model variants for hypothesis 000634 rank blending."
    )

names = list(model_oof)
oof_rank = np.column_stack([rank01(model_oof[n]) for n in names])
test_rank = np.column_stack([rank01(model_test[n]) for n in names])

candidates = weight_grid(len(names), units=5)
try:
    from joblib import Parallel, delayed

    workers = min(16, os.cpu_count() or 1)
    print(f"Evaluating {len(candidates)} blend candidates with {workers} workers")
    scores = Parallel(n_jobs=workers, prefer="threads")(
        delayed(roc_auc_score)(y, oof_rank.dot(w)) for w in candidates
    )
except Exception:
    workers = 1
    print(f"Evaluating {len(candidates)} blend candidates with {workers} workers")
    scores = [roc_auc_score(y, oof_rank.dot(w)) for w in candidates]

best_i = int(np.argmax(scores))
best_w = candidates[best_i]
blend_oof_rank = oof_rank.dot(best_w)
blend_test_rank = test_rank.dot(best_w)

cal = LogisticRegression(C=1.0, solver="lbfgs")
cal.fit(blend_oof_rank.reshape(-1, 1), y)
blend_oof = cal.predict_proba(blend_oof_rank.reshape(-1, 1))[:, 1]
blend_test = cal.predict_proba(blend_test_rank.reshape(-1, 1))[:, 1]

final_auc = roc_auc_score(y, blend_oof)
print("Models:", names)
print("Best rank-blend weights:", {n: float(w) for n, w in zip(names, best_w)})
print(f"Final 5-fold OOF ROC AUC: {final_auc:.6f}")

submission = sample.copy()
submission[target_col] = blend_test
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": blend_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample.copy()
test_pred_df[target_col] = blend_test
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000634"],
    "metric": "roc_auc",
    "cv_roc_auc": float(final_auc),
    "models": names,
    "best_rank_blend_weights": {n: float(w) for n, w in zip(names, best_w)},
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
