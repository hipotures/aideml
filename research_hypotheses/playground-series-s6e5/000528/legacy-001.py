import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
BASE_CAT_COLS = ["Compound", "Driver", "Race"]
COMBO_COLS = ["Race_Compound", "Driver_Compound", "Race_Stint", "Year_Race"]
CAT_COLS = BASE_CAT_COLS + COMBO_COLS

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.sort_values(ID_COL).reset_index(drop=True)
test = test.sort_values(ID_COL).reset_index(drop=True)


def add_features(df):
    df = df.copy()
    for c in BASE_CAT_COLS:
        df[c] = df[c].astype(str).fillna("__NA__")
    df["Race_Compound"] = df["Race"] + "__" + df["Compound"]
    df["Driver_Compound"] = df["Driver"] + "__" + df["Compound"]
    df["Race_Stint"] = df["Race"] + "__" + df["Stint"].astype(str)
    df["Year_Race"] = df["Year"].astype(str) + "__" + df["Race"]
    for c in COMBO_COLS:
        df[c] = df[c].astype(str).fillna("__NA__")
    return df


train = add_features(train)
test = add_features(test)

y = train[TARGET].astype(int).values
features = [c for c in train.columns if c not in [ID_COL, TARGET]]
num_cols = [c for c in features if c not in CAT_COLS]

group_order = train.groupby("Year_Race")[ID_COL].min().sort_values().index.to_numpy()

tscv = TimeSeriesSplit(n_splits=5)
folds = []
for tr_g_idx, va_g_idx in tscv.split(group_order):
    tr_groups = set(group_order[tr_g_idx])
    va_groups = set(group_order[va_g_idx])
    tr_idx = np.flatnonzero(train["Year_Race"].isin(tr_groups).values)
    va_idx = np.flatnonzero(train["Year_Race"].isin(va_groups).values)
    folds.append((tr_idx, va_idx))

cat_params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=600,
    learning_rate=0.06,
    depth=6,
    l2_leaf_reg=8.0,
    random_seed=SEED,
    boosting_type="Ordered",
    has_time=True,
    max_ctr_complexity=1,
    auto_class_weights="Balanced",
    allow_writing_files=False,
    verbose=False,
    thread_count=max(1, os.cpu_count() or 1),
)

oof = np.full(len(train), np.nan, dtype=np.float32)
cat_fold_auc = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    X_tr, X_va = train.iloc[tr_idx][features], train.iloc[va_idx][features]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = CatBoostClassifier(**cat_params, od_type="Iter", od_wait=60)
    model.fit(
        Pool(X_tr, y_tr, cat_features=CAT_COLS),
        eval_set=Pool(X_va, y_va, cat_features=CAT_COLS),
        use_best_model=True,
    )

    pred = model.predict_proba(Pool(X_va, cat_features=CAT_COLS))[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y_va, pred)
    cat_fold_auc.append(float(auc))
    best_iter = model.get_best_iteration()
    best_iters.append(cat_params["iterations"] if best_iter is None else best_iter + 1)
    print(f"CatBoost fold {fold} ROC AUC: {auc:.6f}")

valid_mask = ~np.isnan(oof)
cat_cv_auc = roc_auc_score(y[valid_mask], oof[valid_mask])
print(f"CatBoost chronological grouped 5-fold ROC AUC: {cat_cv_auc:.6f}")

baseline_auc = None
try:
    import lightgbm as lgb

    base_oof = np.full(len(train), np.nan, dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
        X_tr = tr_df[num_cols].copy()
        X_va = va_df[num_cols].copy()

        for c in CAT_COLS:
            freq = tr_df[c].value_counts(normalize=True)
            X_tr[f"{c}_freq"] = tr_df[c].map(freq).fillna(0).astype("float32")
            X_va[f"{c}_freq"] = va_df[c].map(freq).fillna(0).astype("float32")

        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            is_unbalance=True,
            random_state=SEED + fold,
            n_jobs=max(1, os.cpu_count() or 1),
            verbosity=-1,
        )
        clf.fit(
            X_tr,
            y[tr_idx],
            eval_set=[(X_va, y[va_idx])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )
        base_oof[va_idx] = clf.predict_proba(X_va)[:, 1]

    baseline_auc = roc_auc_score(y[valid_mask], base_oof[valid_mask])
    print(f"Manual frequency LightGBM comparison ROC AUC: {baseline_auc:.6f}")
except Exception as e:
    print(f"Manual frequency LightGBM comparison skipped: {repr(e)}")

pd.DataFrame(
    {
        "row": np.flatnonzero(valid_mask),
        "target": y[valid_mask],
        "prediction": oof[valid_mask],
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_iters = int(np.clip(np.median(best_iters), 100, cat_params["iterations"]))
final_params = dict(cat_params)
final_params["iterations"] = final_iters

final_model = CatBoostClassifier(**final_params)
final_model.fit(Pool(train[features], y, cat_features=CAT_COLS))

test_pred = final_model.predict_proba(Pool(test[features], cat_features=CAT_COLS))[:, 1]
submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "catboost_chronological_grouped_cv_auc": float(cat_cv_auc),
    "catboost_fold_auc": cat_fold_auc,
    "manual_frequency_lgbm_cv_auc": (
        None if baseline_auc is None else float(baseline_auc)
    ),
    "final_catboost_iterations": final_iters,
    "research_hypotheses_llm_claimed_used": ["000528"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print(json.dumps(result, indent=2))
