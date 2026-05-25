import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
group_cols = ["Year", "Race", "Driver"]

y = train[target_col].astype(int).values
feature_cols = [c for c in train.columns if c not in [target_col, id_col]]
cat_cols = [
    c for c in feature_cols if train[c].dtype == "object" or test[c].dtype == "object"
]
num_cols = [c for c in feature_cols if c not in cat_cols]


def make_aft_bounds(df):
    lower = np.empty(len(df), dtype=np.float32)
    upper = np.empty(len(df), dtype=np.float32)

    for _, idx in df.groupby(group_cols, sort=False).indices.items():
        idx = np.asarray(idx)
        laps0 = df.loc[idx, "LapNumber"].to_numpy()
        ids0 = df.loc[idx, id_col].to_numpy()
        order = np.lexsort((ids0, laps0))
        sidx = idx[order]

        laps = df.loc[sidx, "LapNumber"].to_numpy(dtype=np.float32)
        pit = df.loc[sidx, "PitStop"].to_numpy()
        pit_laps = laps[pit == 1]
        max_lap = float(np.max(laps))

        lb = np.maximum(max_lap - laps + 1.0, 1.0).astype(np.float32)
        ub = np.full(len(laps), np.inf, dtype=np.float32)

        if len(pit_laps):
            pos = np.searchsorted(pit_laps, laps, side="right")
            has_future = pos < len(pit_laps)
            exact = np.maximum(pit_laps[pos[has_future]] - laps[has_future], 1.0)
            lb[has_future] = exact.astype(np.float32)
            ub[has_future] = exact.astype(np.float32)

        lower[sidx] = lb
        upper[sidx] = ub

    return lower, upper


aft_lower, aft_upper = make_aft_bounds(train)

all_cat = (
    pd.concat([train[cat_cols], test[cat_cols]], axis=0)
    .fillna("__MISSING__")
    .astype(str)
)
try:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
except TypeError:
    enc = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)
enc.fit(all_cat)

train_cat = enc.transform(train[cat_cols].fillna("__MISSING__").astype(str))
test_cat = enc.transform(test[cat_cols].fillna("__MISSING__").astype(str))

all_num = pd.concat([train[num_cols], test[num_cols]], axis=0)
medians = all_num.median(numeric_only=True)
train_num = sparse.csr_matrix(train[num_cols].fillna(medians).astype(np.float32).values)
test_num = sparse.csr_matrix(test[num_cols].fillna(medians).astype(np.float32).values)

X = sparse.hstack([train_num, train_cat], format="csr")
X_test = sparse.hstack([test_num, test_cat], format="csr")

groups = train[group_cols].astype(str).agg("||".join, axis=1).values
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

clf_oof = np.zeros(len(train), dtype=np.float32)
aft_raw_oof = np.zeros(len(train), dtype=np.float32)

clf_params = dict(
    objective="binary",
    n_estimators=700,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=max(1, os.cpu_count() or 1),
    verbosity=-1,
)

aft_params = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.5,
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 30,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "lambda": 2.0,
    "alpha": 0.05,
    "seed": SEED,
    "verbosity": 0,
    "nthread": max(1, os.cpu_count() or 1),
}

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(X[tr_idx], y[tr_idx])
    clf_oof[va_idx] = clf.predict_proba(X[va_idx])[:, 1]

    dtrain = xgb.DMatrix(X[tr_idx])
    dtrain.set_float_info("label_lower_bound", aft_lower[tr_idx])
    dtrain.set_float_info("label_upper_bound", aft_upper[tr_idx])

    aft = xgb.train(aft_params, dtrain, num_boost_round=420, verbose_eval=False)
    pred_time = aft.predict(xgb.DMatrix(X[va_idx]))
    pred_time = np.nan_to_num(pred_time, nan=999.0, posinf=999.0, neginf=999.0)
    aft_raw_oof[va_idx] = -np.maximum(pred_time, 0.0)

    print(
        f"fold {fold} classifier_auc={roc_auc_score(y[va_idx], clf_oof[va_idx]):.6f} "
        f"aft_auc={roc_auc_score(y[va_idx], aft_raw_oof[va_idx]):.6f}"
    )

aft_score_oof = rankdata(aft_raw_oof, method="average").astype(np.float32) / (
    len(aft_raw_oof) + 1.0
)
blend_weight_aft = 0.15
blend_oof = (1.0 - blend_weight_aft) * clf_oof + blend_weight_aft * aft_score_oof

clf_auc = roc_auc_score(y, clf_oof)
aft_auc = roc_auc_score(y, aft_score_oof)
blend_auc = roc_auc_score(y, blend_oof)

print(f"CV ROC AUC classifier: {clf_auc:.6f}")
print(f"CV ROC AUC AFT score: {aft_auc:.6f}")
print(f"CV ROC AUC fixed blend: {blend_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": blend_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_clf = lgb.LGBMClassifier(**clf_params)
final_clf.fit(X, y)
test_clf = final_clf.predict_proba(X_test)[:, 1]

dfull = xgb.DMatrix(X)
dfull.set_float_info("label_lower_bound", aft_lower)
dfull.set_float_info("label_upper_bound", aft_upper)
final_aft = xgb.train(aft_params, dfull, num_boost_round=420, verbose_eval=False)

test_time = final_aft.predict(xgb.DMatrix(X_test))
test_time = np.nan_to_num(test_time, nan=999.0, posinf=999.0, neginf=999.0)
test_aft_raw = -np.maximum(test_time, 0.0)

sorted_oof_raw = np.sort(aft_raw_oof)
test_aft_score = np.searchsorted(sorted_oof_raw, test_aft_raw, side="right").astype(
    np.float32
)
test_aft_score /= len(sorted_oof_raw) + 1.0

test_pred = (1.0 - blend_weight_aft) * test_clf + blend_weight_aft * test_aft_score
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

test_predictions = sample.copy()
test_predictions[target_col] = test_pred
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000297"],
    "metric": "roc_auc",
    "cv_roc_auc_classifier": float(clf_auc),
    "cv_roc_auc_aft_score": float(aft_auc),
    "cv_roc_auc_fixed_blend": float(blend_auc),
    "blend_weight_aft": float(blend_weight_aft),
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
