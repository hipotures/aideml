import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORKING = Path("./working")
WORKING.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 20260523
N_SPLITS = 5

train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values
test_ids = sample[ID_COL].values


def add_shared_features(df):
    out = df.copy()
    out["Year"] = out["Year"].astype(str)
    for c in ["Driver", "Race", "Compound", "Year"]:
        out[c] = out[c].astype(str).fillna("missing")

    out["Year_Race"] = out["Year"] + "_" + out["Race"]
    out["Driver_Race"] = out["Driver"] + "_" + out["Race"]
    out["Driver_Compound"] = out["Driver"] + "_" + out["Compound"]
    out["Race_Compound"] = out["Race"] + "_" + out["Compound"]

    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["LapNumber_x_Progress"] = out["LapNumber"] * out["RaceProgress"]
    out["Deg_per_TyreLife"] = out["Cumulative_Degradation"] / (out["TyreLife"] + 1.0)
    out["LapTime_per_Lap"] = out["LapTime (s)"] / (out["LapNumber"] + 1.0)
    out["Abs_Position_Change"] = out["Position_Change"].abs()
    out["IsWetCompound"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    return out


train_fe = add_shared_features(train.drop(columns=[TARGET]))
test_fe = add_shared_features(test)

cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "Year",
    "Year_Race",
    "Driver_Race",
    "Driver_Compound",
    "Race_Compound",
]
base_drop = [ID_COL]
num_cols = [c for c in train_fe.columns if c not in base_drop + cat_cols]
feature_cols = cat_cols + num_cols

groups = train_fe["Year_Race"].values
gkf = GroupKFold(n_splits=N_SPLITS)


def percentile_rank(x):
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rank(method="average", pct=True).values


def add_count_features(fit_df, apply_df, cols):
    res = pd.DataFrame(index=apply_df.index)
    n = len(fit_df)
    for c in cols:
        vc = fit_df[c].value_counts(dropna=False)
        res[f"{c}_count"] = apply_df[c].map(vc).fillna(0).astype(float)
        res[f"{c}_freq"] = res[f"{c}_count"] / max(n, 1)
    return res


def add_target_features(fit_df, fit_y, apply_df, cols):
    res = pd.DataFrame(index=apply_df.index)
    global_mean = float(np.mean(fit_y))
    tmp = fit_df.copy()
    tmp["_target"] = fit_y
    for c in cols:
        stats = tmp.groupby(c)["_target"].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * 20.0) / (
            stats["count"] + 20.0
        )
        res[f"{c}_te"] = apply_df[c].map(smooth).fillna(global_mean).astype(float)
    return res


def make_lgb_features(fit_df, fit_y, apply_df):
    x_num = apply_df[num_cols].copy()
    x_count = add_count_features(fit_df, apply_df, cat_cols)
    x_te = add_target_features(fit_df, fit_y, apply_df, cat_cols)
    return pd.concat(
        [
            x_num.reset_index(drop=True),
            x_count.reset_index(drop=True),
            x_te.reset_index(drop=True),
        ],
        axis=1,
    )


try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
except Exception as e:
    raise RuntimeError("lightgbm is required for the feature-heavy tree model") from e

try:
    from catboost import CatBoostClassifier, Pool
except Exception as e:
    raise RuntimeError("catboost is required for hypothesis 000901") from e

oof_lgb = np.zeros(len(train_fe), dtype=float)
oof_cat = np.zeros(len(train_fe), dtype=float)
oof_blend = np.zeros(len(train_fe), dtype=float)

test_lgb_rank_folds = []
test_cat_rank_folds = []
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_fe, y, groups), 1):
    x_tr_raw = train_fe.iloc[tr_idx].reset_index(drop=True)
    x_va_raw = train_fe.iloc[va_idx].reset_index(drop=True)
    y_tr, y_va = y[tr_idx], y[va_idx]

    x_tr_lgb = make_lgb_features(x_tr_raw, y_tr, x_tr_raw)
    x_va_lgb = make_lgb_features(x_tr_raw, y_tr, x_va_raw)
    x_te_lgb = make_lgb_features(x_tr_raw, y_tr, test_fe)

    lgb = LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=RANDOM_STATE + fold,
        n_jobs=max(1, os.cpu_count() or 1),
        verbose=-1,
        class_weight="balanced",
    )
    lgb.fit(
        x_tr_lgb,
        y_tr,
        eval_set=[(x_va_lgb, y_va)],
        eval_metric="auc",
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    pred_lgb_va = lgb.predict_proba(x_va_lgb)[:, 1]
    pred_lgb_te = lgb.predict_proba(x_te_lgb)[:, 1]

    x_tr_cat = x_tr_raw[feature_cols].copy()
    x_va_cat = x_va_raw[feature_cols].copy()
    x_te_cat = test_fe[feature_cols].copy()
    for c in cat_cols:
        x_tr_cat[c] = x_tr_cat[c].astype(str)
        x_va_cat[c] = x_va_cat[c].astype(str)
        x_te_cat[c] = x_te_cat[c].astype(str)

    cat = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=6.0,
        random_seed=RANDOM_STATE + fold,
        auto_class_weights="Balanced",
        allow_writing_files=False,
        verbose=False,
        od_type="Iter",
        od_wait=100,
        thread_count=max(1, os.cpu_count() or 1),
    )
    cat.fit(
        Pool(x_tr_cat, y_tr, cat_features=cat_cols),
        eval_set=Pool(x_va_cat, y_va, cat_features=cat_cols),
        use_best_model=True,
    )
    pred_cat_va = cat.predict_proba(Pool(x_va_cat, cat_features=cat_cols))[:, 1]
    pred_cat_te = cat.predict_proba(Pool(x_te_cat, cat_features=cat_cols))[:, 1]

    rank_lgb_va = percentile_rank(pred_lgb_va)
    rank_cat_va = percentile_rank(pred_cat_va)
    pred_blend_va = 0.5 * rank_lgb_va + 0.5 * rank_cat_va

    oof_lgb[va_idx] = pred_lgb_va
    oof_cat[va_idx] = pred_cat_va
    oof_blend[va_idx] = pred_blend_va

    test_lgb_rank_folds.append(percentile_rank(pred_lgb_te))
    test_cat_rank_folds.append(percentile_rank(pred_cat_te))

    fold_auc = roc_auc_score(y_va, pred_blend_va)
    fold_scores.append(fold_auc)
    print(f"fold={fold} blend_rank_auc={fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof_blend)
print(f"OOF rank-blend ROC AUC: {cv_auc:.6f}")
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")

test_pred = 0.5 * np.mean(test_lgb_rank_folds, axis=0) + 0.5 * np.mean(
    test_cat_rank_folds, axis=0
)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(WORKING / "submission.csv", index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof_blend,
    }
).to_csv(WORKING / "oof_predictions.csv.gz", index=False, compression="gzip")

pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: test_pred,
    }
).to_csv(WORKING / "test_predictions.csv.gz", index=False, compression="gzip")

result = {
    "metric": "roc_auc",
    "oof_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(v) for v in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000901"],
    "submission_path": str(WORKING / "submission.csv"),
}
with open(WORKING / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
