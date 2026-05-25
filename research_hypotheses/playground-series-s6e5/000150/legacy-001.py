import os
import gc
import json
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"

os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_features(df):
    df = df.copy()
    for c in ["Driver", "Race", "Compound"]:
        df[c] = df[c].fillna("__NA__").astype(str)

    df["Race_Year"] = df["Race"] + "_" + df["Year"].astype(str)
    df["Driver_Race"] = df["Driver"] + "_" + df["Race"]
    df["Driver_Compound"] = df["Driver"] + "_" + df["Compound"]
    df["Race_Compound"] = df["Race"] + "_" + df["Compound"]
    df["Driver_Stint"] = df["Driver"] + "_" + df["Stint"].astype(str)

    race_progress = df["RaceProgress"].clip(lower=1e-3)
    est_total_laps = df["LapNumber"] / race_progress
    df["EstimatedTotalLaps"] = est_total_laps
    df["EstimatedLapsLeft"] = est_total_laps - df["LapNumber"]
    df["TyreLifeRaceShare"] = df["TyreLife"] / est_total_laps.replace(0, np.nan)
    df["TyreLifeLapRatio"] = df["TyreLife"] / (df["LapNumber"] + 1.0)
    df["DegradationPerTyreLife"] = df["Cumulative_Degradation"] / df[
        "TyreLife"
    ].replace(0, np.nan)
    df["AbsLapTimeDelta"] = df["LapTime_Delta"].abs()
    df["PositionChangeAbs"] = df["Position_Change"].abs()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


train_fe = add_features(train)
test_fe = add_features(test)
y = train_fe[TARGET].astype(int).to_numpy()

cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "Race_Year",
    "Driver_Race",
    "Driver_Compound",
    "Race_Compound",
    "Driver_Stint",
]
low_card_cols = ["Compound", "Race", "Year", "Stint"]
freq_cols = [
    "Driver",
    "Race_Year",
    "Driver_Race",
    "Driver_Compound",
    "Race_Compound",
    "Driver_Stint",
]

exclude = {ID_COL, TARGET, *cat_cols}
num_cols = [
    c
    for c in train_fe.columns
    if c not in exclude and pd.api.types.is_numeric_dtype(train_fe[c])
]
cb_features = num_cols + cat_cols

for c in cat_cols:
    train_fe[c] = train_fe[c].fillna("__NA__").astype(str)
    test_fe[c] = test_fe[c].fillna("__NA__").astype(str)

for c in num_cols:
    median = train_fe[c].median()
    train_fe[c] = train_fe[c].fillna(median).astype(np.float32)
    test_fe[c] = test_fe[c].fillna(median).astype(np.float32)

try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)

ohe.fit(pd.concat([train_fe[low_card_cols], test_fe[low_card_cols]], axis=0))
train_ohe = ohe.transform(train_fe[low_card_cols])
test_ohe = ohe.transform(test_fe[low_card_cols])


def frequency_block(reference_df, apply_df, columns):
    blocks = []
    for col in columns:
        freq = reference_df[col].value_counts(normalize=True)
        vals = (
            apply_df[col]
            .map(freq)
            .fillna(0)
            .astype(np.float32)
            .to_numpy()
            .reshape(-1, 1)
        )
        blocks.append(vals)
    return np.hstack(blocks).astype(np.float32)


def make_lgb_matrix(reference_df, apply_df, apply_ohe):
    numeric = apply_df[num_cols].to_numpy(dtype=np.float32)
    freqs = frequency_block(reference_df, apply_df, freq_cols)
    dense = sparse.csr_matrix(np.hstack([numeric, freqs]).astype(np.float32))
    return sparse.hstack([dense, apply_ohe], format="csr")


groups = train_fe["Race_Year"].to_numpy()
if StratifiedGroupKFold is not None:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(train_fe, y, groups))
    cv_name = "StratifiedGroupKFold"
else:
    cv = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(train_fe, y, groups))
    cv_name = "GroupKFold"

oof_cat = np.zeros(len(train_fe), dtype=np.float32)
oof_lgb = np.zeros(len(train_fe), dtype=np.float32)
test_cat = np.zeros(len(test_fe), dtype=np.float32)
test_lgb = np.zeros(len(test_fe), dtype=np.float32)

threads = max(1, os.cpu_count() or 1)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr_cb = train_fe.iloc[tr_idx][cb_features]
    X_va_cb = train_fe.iloc[va_idx][cb_features]
    X_te_cb = test_fe[cb_features]

    train_pool = Pool(X_tr_cb, y[tr_idx], cat_features=cat_cols)
    valid_pool = Pool(X_va_cb, y[va_idx], cat_features=cat_cols)
    test_pool = Pool(X_te_cb, cat_features=cat_cols)

    cat_model = CatBoostClassifier(
        iterations=350,
        learning_rate=0.07,
        depth=6,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED + fold,
        auto_class_weights="Balanced",
        allow_writing_files=False,
        thread_count=threads,
        verbose=False,
    )
    cat_model.fit(
        train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=60
    )
    oof_cat[va_idx] = cat_model.predict_proba(valid_pool)[:, 1]
    test_cat += cat_model.predict_proba(test_pool)[:, 1] / N_SPLITS

    ref_df = train_fe.iloc[tr_idx]
    X_tr_lgb = make_lgb_matrix(ref_df, train_fe.iloc[tr_idx], train_ohe[tr_idx])
    X_va_lgb = make_lgb_matrix(ref_df, train_fe.iloc[va_idx], train_ohe[va_idx])
    X_te_lgb = make_lgb_matrix(ref_df, test_fe, test_ohe)

    pos = y[tr_idx].sum()
    neg = len(tr_idx) - pos
    scale_pos_weight = float(neg / max(pos, 1))

    lgb_model = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        random_state=SEED + fold,
        n_jobs=threads,
        scale_pos_weight=scale_pos_weight,
        verbosity=-1,
    )
    lgb_model.fit(
        X_tr_lgb,
        y[tr_idx],
        eval_set=[(X_va_lgb, y[va_idx])],
        eval_metric="auc",
        callbacks=[early_stopping(75, verbose=False), log_evaluation(0)],
    )
    oof_lgb[va_idx] = lgb_model.predict_proba(X_va_lgb)[:, 1]
    test_lgb += lgb_model.predict_proba(X_te_lgb)[:, 1] / N_SPLITS

    fold_cat_auc = roc_auc_score(y[va_idx], oof_cat[va_idx])
    fold_lgb_auc = roc_auc_score(y[va_idx], oof_lgb[va_idx])
    print(
        f"fold {fold}: catboost_auc={fold_cat_auc:.6f} frequency_lgb_auc={fold_lgb_auc:.6f}"
    )

    del train_pool, valid_pool, test_pool, cat_model, lgb_model
    del X_tr_lgb, X_va_lgb, X_te_lgb
    gc.collect()

cat_auc = roc_auc_score(y, oof_cat)
lgb_auc = roc_auc_score(y, oof_lgb)

best_auc = -1.0
best_w = 0.0
best_oof = None
for w in np.linspace(0, 1, 21):
    pred = w * oof_cat + (1.0 - w) * oof_lgb
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_auc = auc
        best_w = float(w)
        best_oof = pred

test_pred = np.clip(best_w * test_cat + (1.0 - best_w) * test_lgb, 0, 1)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {"row": np.arange(len(train_fe)), "target": y, "prediction": best_oof}
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample.copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv": cv_name,
    "catboost_oof_roc_auc": float(cat_auc),
    "frequency_lgb_oof_roc_auc": float(lgb_auc),
    "blended_oof_roc_auc": float(best_auc),
    "best_catboost_weight": float(best_w),
    "research_hypotheses_llm_claimed_used": ["000150"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print(f"CatBoost OOF ROC AUC: {cat_auc:.6f}")
print(f"Frequency LightGBM OOF ROC AUC: {lgb_auc:.6f}")
print(f"Blended OOF ROC AUC: {best_auc:.6f} with CatBoost weight {best_w:.2f}")
print(json.dumps(result, sort_keys=True))
