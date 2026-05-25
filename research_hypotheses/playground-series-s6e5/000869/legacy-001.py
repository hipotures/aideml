import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

try:
    from scipy.special import ndtr
except Exception:
    from math import erf

    ndtr = np.vectorize(lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0))))

warnings.filterwarnings("ignore")

SEED = 2026
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
group_cols = ["Year", "Race", "Driver"]
features = [c for c in train.columns if c not in [target_col, id_col]]

y = train[target_col].astype(np.float32).values
cat_cols = (
    train[features].select_dtypes(include=["object", "category"]).columns.tolist()
)
num_cols = [c for c in features if c not in cat_cols]


def make_ohe():
    try:
        return OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def make_time_to_next_pit(df):
    work = df[group_cols + ["LapNumber", "PitStop"]].copy()
    work["_row"] = np.arange(len(df))
    work = work.sort_values(group_cols + ["LapNumber", "_row"])

    lower = np.ones(len(df), dtype=np.float32)
    upper = np.full(len(df), np.inf, dtype=np.float32)

    for _, g in work.groupby(group_cols, sort=False):
        rows = g["_row"].to_numpy()
        laps = g["LapNumber"].to_numpy(dtype=np.float32)
        pits = laps[g["PitStop"].to_numpy() == 1]
        max_lap = float(np.max(laps))

        if len(pits):
            pos = np.searchsorted(pits, laps, side="right")
            observed = pos < len(pits)
            duration = np.maximum(pits[np.minimum(pos, len(pits) - 1)] - laps, 1.0)
        else:
            observed = np.zeros(len(g), dtype=bool)
            duration = np.ones(len(g), dtype=np.float32)

        censor_duration = np.maximum(max_lap + 1.0 - laps, 1.0)
        duration = np.where(observed, duration, censor_duration).astype(np.float32)

        lower[rows] = duration
        upper[rows] = np.where(observed, duration, np.inf).astype(np.float32)

    positives = df[target_col].values == 1
    lower[positives] = 1.0
    upper[positives] = 1.0
    return lower, upper


def predict_booster(model, dmat, output_margin=False):
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and best_iter >= 0:
        try:
            return model.predict(
                dmat,
                iteration_range=(0, best_iter + 1),
                output_margin=output_margin,
            )
        except TypeError:
            best_ntree = getattr(model, "best_ntree_limit", 0)
            if best_ntree:
                return model.predict(
                    dmat, ntree_limit=best_ntree, output_margin=output_margin
                )
    return model.predict(dmat, output_margin=output_margin)


def aft_next_lap_risk(log_time_location, sigma):
    z = (np.log(1.0) - log_time_location) / sigma
    return np.clip(ndtr(z), 1e-6, 1.0 - 1e-6)


lower_bound, upper_bound = make_time_to_next_pit(train)

race_groups = (train["Year"].astype(str) + "__" + train["Race"].astype(str)).values

if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(train, y, groups=race_groups))
else:
    splitter = GroupKFold(n_splits=5)
    folds = list(splitter.split(train, y, groups=race_groups))

oof_clf = np.zeros(len(train), dtype=np.float32)
oof_aft = np.zeros(len(train), dtype=np.float32)
test_clf = np.zeros(len(test), dtype=np.float64)
test_aft = np.zeros(len(test), dtype=np.float64)

nthread = max(1, os.cpu_count() or 1)
pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1.0))
aft_sigma = 1.15

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    pre = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        sparse_threshold=1.0,
        remainder="drop",
    )

    X_tr = pre.fit_transform(train.iloc[tr_idx][features]).astype(np.float32)
    X_va = pre.transform(train.iloc[va_idx][features]).astype(np.float32)
    X_te = pre.transform(test[features]).astype(np.float32)

    dtr_clf = xgb.DMatrix(X_tr, label=y[tr_idx], nthread=nthread)
    dva_clf = xgb.DMatrix(X_va, label=y[va_idx], nthread=nthread)
    dte = xgb.DMatrix(X_te, nthread=nthread)

    clf_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "learning_rate": 0.045,
        "max_depth": 5,
        "min_child_weight": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 2.0,
        "alpha": 0.1,
        "scale_pos_weight": pos_weight,
        "seed": SEED + fold,
        "nthread": nthread,
        "verbosity": 0,
    }

    clf = xgb.train(
        clf_params,
        dtr_clf,
        num_boost_round=450,
        evals=[(dva_clf, "valid")],
        early_stopping_rounds=45,
        verbose_eval=False,
    )

    oof_clf[va_idx] = predict_booster(clf, dva_clf).astype(np.float32)
    test_clf += predict_booster(clf, dte) / len(folds)

    dtr_aft = xgb.DMatrix(X_tr, nthread=nthread)
    dva_aft = xgb.DMatrix(X_va, nthread=nthread)
    dtr_aft.set_float_info("label_lower_bound", lower_bound[tr_idx])
    dtr_aft.set_float_info("label_upper_bound", upper_bound[tr_idx])
    dva_aft.set_float_info("label_lower_bound", lower_bound[va_idx])
    dva_aft.set_float_info("label_upper_bound", upper_bound[va_idx])

    aft_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": aft_sigma,
        "tree_method": "hist",
        "learning_rate": 0.045,
        "max_depth": 4,
        "min_child_weight": 25,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 2.0,
        "alpha": 0.1,
        "seed": SEED + 100 + fold,
        "nthread": nthread,
        "verbosity": 0,
    }

    aft = xgb.train(
        aft_params,
        dtr_aft,
        num_boost_round=450,
        evals=[(dva_aft, "valid")],
        early_stopping_rounds=45,
        verbose_eval=False,
    )

    mu_va = predict_booster(aft, dva_aft, output_margin=True)
    mu_te = predict_booster(aft, dte, output_margin=True)
    oof_aft[va_idx] = aft_next_lap_risk(mu_va, aft_sigma).astype(np.float32)
    test_aft += aft_next_lap_risk(mu_te, aft_sigma) / len(folds)

    fold_blend = np.clip(
        0.70 * oof_clf[va_idx] + 0.30 * oof_aft[va_idx], 1e-6, 1 - 1e-6
    )
    print(f"fold {fold} blended ROC AUC: {roc_auc_score(y[va_idx], fold_blend):.6f}")

oof_blend = np.clip(0.70 * oof_clf + 0.30 * oof_aft, 1e-6, 1.0 - 1e-6)
test_blend = np.clip(0.70 * test_clf + 0.30 * test_aft, 1e-6, 1.0 - 1e-6)

clf_auc = roc_auc_score(y, oof_clf)
aft_auc = roc_auc_score(y, oof_aft)
blend_auc = roc_auc_score(y, oof_blend)

submission = sample.copy()
submission[target_col] = test_blend
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_blend,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        id_col: sample[id_col].values,
        target_col: test_blend,
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(f"OOF classifier ROC AUC: {clf_auc:.6f}")
print(f"OOF AFT next-lap risk ROC AUC: {aft_auc:.6f}")
print(f"OOF blended ROC AUC: {blend_auc:.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000869"],
            "metric": "grouped_5fold_oof_roc_auc",
            "classifier_auc": float(clf_auc),
            "aft_auc": float(aft_auc),
            "blended_auc": float(blend_auc),
        }
    )
)
