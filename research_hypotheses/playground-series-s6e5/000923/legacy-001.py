import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

RANDOM_STATE = 923
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target = "PitNextLap"
id_col = "id"


def add_features(df):
    df = df.copy()
    for c in ["Race", "Driver", "Compound", "Year", "Stint", "PitStop"]:
        df[c] = df[c].astype(str)
    df["Race_Compound"] = df["Race"] + "__" + df["Compound"]
    df["Driver_Race"] = df["Driver"] + "__" + df["Race"]
    df["Year_Race"] = df["Year"] + "__" + df["Race"]
    df["Compound_Stint"] = df["Compound"] + "__" + df["Stint"]
    df["Driver_Compound"] = df["Driver"] + "__" + df["Compound"]
    return df


train = add_features(train)
test = add_features(test)

cat_cols = [
    "Compound",
    "Driver",
    "Race",
    "Year",
    "Stint",
    "PitStop",
    "Race_Compound",
    "Driver_Race",
    "Year_Race",
    "Compound_Stint",
    "Driver_Compound",
]
features = [c for c in train.columns if c not in [target, id_col]]

for c in cat_cols:
    cats = pd.Index(pd.concat([train[c], test[c]], axis=0).astype(str).unique())
    dtype = pd.CategoricalDtype(categories=cats)
    train[c] = train[c].astype(str).astype(dtype)
    test[c] = test[c].astype(str).astype(dtype)

y = train[target].astype(int).values
groups = train["Year_Race"].astype(str).values
folds = list(GroupKFold(n_splits=N_SPLITS).split(train, y, groups))

pos = max(1, int(y.sum()))
neg = max(1, int(len(y) - y.sum()))
scale_pos_weight = neg / pos
threads = max(1, os.cpu_count() or 1)

cat_oof = np.zeros(len(train))
gbm_oof = np.zeros(len(train))
cat_test = np.zeros(len(test))
gbm_test = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    X_tr, X_va = train.iloc[tr_idx][features], train.iloc[va_idx][features]
    y_tr, y_va = y[tr_idx], y[va_idx]

    cat_model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=8.0,
        random_seed=RANDOM_STATE + fold,
        auto_class_weights="Balanced",
        max_ctr_complexity=2,
        one_hot_max_size=2,
        early_stopping_rounds=80,
        allow_writing_files=False,
        thread_count=threads,
        verbose=False,
    )
    cat_model.fit(
        Pool(X_tr, y_tr, cat_features=cat_cols),
        eval_set=Pool(X_va, y_va, cat_features=cat_cols),
        use_best_model=True,
    )
    cat_oof[va_idx] = cat_model.predict_proba(Pool(X_va, cat_features=cat_cols))[:, 1]
    cat_test += (
        cat_model.predict_proba(Pool(test[features], cat_features=cat_cols))[:, 1]
        / N_SPLITS
    )

    gbm_model = LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.3,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE + fold,
        n_jobs=threads,
        verbosity=-1,
    )
    gbm_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )
    gbm_oof[va_idx] = gbm_model.predict_proba(X_va)[:, 1]
    gbm_test += gbm_model.predict_proba(test[features])[:, 1] / N_SPLITS


def rank_norm(a):
    return pd.Series(a).rank(method="average").to_numpy() / len(a)


cat_auc = roc_auc_score(y, cat_oof)
gbm_auc = roc_auc_score(y, gbm_oof)

cat_oof_r = rank_norm(cat_oof)
gbm_oof_r = rank_norm(gbm_oof)
cat_test_r = rank_norm(cat_test)
gbm_test_r = rank_norm(gbm_test)

best_auc, best_w = -1.0, None
for w in np.linspace(0, 1, 21):
    pred = w * cat_oof_r + (1 - w) * gbm_oof_r
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_auc, best_w = auc, float(w)

oof_blend = best_w * cat_oof_r + (1 - best_w) * gbm_oof_r
test_blend = best_w * cat_test_r + (1 - best_w) * gbm_test_r

sub = sample.copy()
sub[target] = test_blend
sub.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_blend,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

sub.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_auc_blend": float(best_auc),
            "cv_auc_catboost": float(cat_auc),
            "cv_auc_gbm": float(gbm_auc),
            "catboost_blend_weight": best_w,
            "research_hypotheses_llm_claimed_used": ["000923"],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        },
        indent=2,
    )
)
