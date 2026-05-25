import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGKF = True
except Exception:
    from sklearn.model_selection import GroupKFold

    HAS_SGKF = False

from catboost import CatBoostClassifier, Pool

try:
    import lightgbm as lgb

    HAS_LGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAS_LGB = False


RANDOM_STATE = 137
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"

os.makedirs("./working", exist_ok=True)

train = pd.read_csv("./input/train.csv.gz")
test = pd.read_csv("./input/test.csv.gz")
sample = pd.read_csv("./input/sample_submission.csv.gz")

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values


def add_interactions(df):
    df = df.copy()
    df["Driver_Year"] = df["Driver"].astype(str) + "_" + df["Year"].astype(str)
    df["Race_Compound"] = df["Race"].astype(str) + "_" + df["Compound"].astype(str)
    df["Driver_Compound"] = df["Driver"].astype(str) + "_" + df["Compound"].astype(str)
    return df


train_fe = add_interactions(train.drop(columns=[TARGET]))
test_fe = add_interactions(test)

groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)

feature_cols = [c for c in train_fe.columns if c != ID_COL]
cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "Year",
    "Driver_Year",
    "Race_Compound",
    "Driver_Compound",
]
cat_cols = [c for c in cat_cols if c in feature_cols]

X_cat = train_fe[feature_cols].copy()
T_cat = test_fe[feature_cols].copy()
for c in cat_cols:
    X_cat[c] = X_cat[c].astype(str).fillna("__NA__")
    T_cat[c] = T_cat[c].astype(str).fillna("__NA__")

num_cols = [c for c in feature_cols if c not in cat_cols]
X_num = train_fe[num_cols].copy()
T_num = test_fe[num_cols].copy()

for c in cat_cols:
    all_vals = pd.concat([train_fe[c], test_fe[c]], axis=0).astype(str).fillna("__NA__")
    freq = all_vals.value_counts(normalize=True)
    X_num[c + "_freq"] = (
        train_fe[c].astype(str).fillna("__NA__").map(freq).astype("float32")
    )
    T_num[c + "_freq"] = (
        test_fe[c].astype(str).fillna("__NA__").map(freq).astype("float32")
    )

X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0)
T_num = T_num.replace([np.inf, -np.inf], np.nan).fillna(0)

cat_oof = np.zeros(len(train))
num_oof = np.zeros(len(train))
cat_test = np.zeros(len(test))
num_test = np.zeros(len(test))
fold_rows = []

if HAS_SGKF:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = splitter.split(X_cat, y, groups)
else:
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = splitter.split(X_cat, y, groups)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr_cat, X_va_cat = X_cat.iloc[tr_idx], X_cat.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_pool = Pool(X_tr_cat, y_tr, cat_features=cat_cols)
    valid_pool = Pool(X_va_cat, y_va, cat_features=cat_cols)
    test_pool = Pool(T_cat, cat_features=cat_cols)

    cb = CatBoostClassifier(
        iterations=900,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="SqrtBalanced",
        random_seed=RANDOM_STATE + fold,
        one_hot_max_size=2,
        allow_writing_files=False,
        verbose=False,
    )
    cb.fit(
        train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=80
    )
    cat_oof[va_idx] = cb.predict_proba(valid_pool)[:, 1]
    cat_test += cb.predict_proba(test_pool)[:, 1] / N_SPLITS

    X_tr_num, X_va_num = X_num.iloc[tr_idx], X_num.iloc[va_idx]

    if HAS_LGB:
        neg = max((y_tr == 0).sum(), 1)
        pos = max((y_tr == 1).sum(), 1)
        lgbm = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.035,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            scale_pos_weight=neg / pos,
            random_state=RANDOM_STATE + 100 + fold,
            n_jobs=max(1, os.cpu_count() or 1),
            verbosity=-1,
        )
        lgbm.fit(
            X_tr_num,
            y_tr,
            eval_set=[(X_va_num, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        num_oof[va_idx] = lgbm.predict_proba(X_va_num)[:, 1]
        num_test += lgbm.predict_proba(T_num)[:, 1] / N_SPLITS
    else:
        neg = max((y_tr == 0).sum(), 1)
        pos = max((y_tr == 1).sum(), 1)
        weights = np.where(y_tr == 1, neg / pos, 1.0)
        hgb = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=RANDOM_STATE + 100 + fold,
        )
        hgb.fit(X_tr_num, y_tr, sample_weight=weights)
        num_oof[va_idx] = hgb.predict_proba(X_va_num)[:, 1]
        num_test += hgb.predict_proba(T_num)[:, 1] / N_SPLITS

    fold_rows.append(
        {
            "fold": fold,
            "catboost_auc": float(roc_auc_score(y_va, cat_oof[va_idx])),
            "numeric_auc": float(roc_auc_score(y_va, num_oof[va_idx])),
        }
    )
    print(json.dumps(fold_rows[-1]))

cat_auc = roc_auc_score(y, cat_oof)
num_auc = roc_auc_score(y, num_oof)

best_w = 1.0
best_auc = -1.0
for w in np.linspace(0.0, 1.0, 21):
    pred = w * cat_oof + (1.0 - w) * num_oof
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_auc = auc
        best_w = float(w)

blend_oof = best_w * cat_oof + (1.0 - best_w) * num_oof
blend_test = best_w * cat_test + (1.0 - best_w) * num_test
blend_test = np.clip(blend_test, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = blend_test
submission.to_csv("./working/submission.csv", index=False)
submission.to_csv("./working/test_predictions.csv.gz", index=False, compression="gzip")

oof = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": blend_oof,
    }
)
oof.to_csv("./working/oof_predictions.csv.gz", index=False, compression="gzip")

result = {
    "metric": "roc_auc",
    "cv_auc": float(best_auc),
    "catboost_oof_auc": float(cat_auc),
    "numeric_oof_auc": float(num_auc),
    "catboost_blend_weight": best_w,
    "folds": fold_rows,
    "submission_path": "./working/submission.csv",
    "oof_path": "./working/oof_predictions.csv.gz",
    "test_predictions_path": "./working/test_predictions.csv.gz",
    "research_hypotheses_llm_claimed_used": ["000137"],
}
print(json.dumps(result, indent=2))
