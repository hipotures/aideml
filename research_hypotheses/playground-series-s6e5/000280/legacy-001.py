import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

ID_COL = "id"
TARGET = "PitNextLap"
MODEL_NAMES = ["base", "cat", "wet", "late"]


def add_features(df):
    out = df.copy()
    eps = 1e-6
    compound = out["Compound"].astype(str)
    race = out["Race"].astype(str)
    rp = out["RaceProgress"].astype(float).clip(0, 1)
    tyre = out["TyreLife"].astype(float).clip(lower=1)
    lap = out["LapNumber"].astype(float).clip(lower=1)

    out["IsWetCompound"] = compound.isin(["WET", "INTERMEDIATE"]).astype("int8")
    out["IsTesting"] = (race == "Pre-Season Testing").astype("int8")
    out["RacePhase"] = np.where(rp < 0.33, "early", np.where(rp < 0.66, "mid", "late"))
    out["DriverRace"] = out["Driver"].astype(str) + "_" + race
    out["CompoundPhase"] = compound + "_" + out["RacePhase"].astype(str)

    out["DegPerTyreLap"] = out["Cumulative_Degradation"].astype(float) / (tyre + eps)
    out["LapDeltaPerLap"] = out["LapTime_Delta"].astype(float) / (lap + eps)
    out["TyreLifeToLap"] = tyre / (lap + eps)
    out["ProgressTyreLife"] = rp * tyre
    out["ProgressPosition"] = rp * out["Position"].astype(float)
    out["StintTyreLife"] = out["Stint"].astype(float) * tyre
    out["RemainingProgress"] = 1.0 - rp

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return out


def percentile_rank(x):
    return (
        pd.Series(np.asarray(x))
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )


def regime_weights(df):
    n = len(df)
    compound = df["Compound"].astype(str).to_numpy()
    race = df["Race"].astype(str).to_numpy()
    rp = df["RaceProgress"].astype(float).to_numpy()

    wet = np.isin(compound, ["WET", "INTERMEDIATE"])
    testing = race == "Pre-Season Testing"
    early = rp < 0.33
    mid = (rp >= 0.33) & (rp < 0.66)
    late = rp >= 0.66

    w = np.zeros((n, len(MODEL_NAMES)), dtype=np.float32)
    b, c, ww, l = 0, 1, 2, 3

    w[:, b] = 0.44
    w[:, c] = 0.34
    w[:, ww] = 0.05
    w[:, l] = 0.17

    w[wet, b] -= 0.08
    w[wet, c] -= 0.04
    w[wet, ww] += 0.34
    w[wet, l] -= 0.02
    w[~wet, ww] = 0.02

    w[testing, b] += 0.08
    w[testing, c] += 0.08
    w[testing, ww] -= 0.03
    w[testing, l] -= 0.05

    w[early, b] += 0.08
    w[early, l] -= 0.10
    w[mid, c] += 0.03
    w[late, l] += 0.24
    w[late, b] -= 0.07
    w[late, c] -= 0.04

    w = np.clip(w, 0.01, None)
    return w / w.sum(axis=1, keepdims=True)


def blend_rank_predictions(pred_dict, df):
    ranks = np.column_stack([percentile_rank(pred_dict[name]) for name in MODEL_NAMES])
    weights = regime_weights(df)
    return percentile_rank((ranks * weights).sum(axis=1))


def valid_subset(idx, mask, y, min_rows):
    sub = idx[mask[idx]]
    if len(sub) >= min_rows and np.unique(y[sub]).size == 2:
        return sub
    return idx


def fit_lgb_model(X, y, train_idx, eval_idx, params, cat_cols):
    model = lgb.LGBMClassifier(**params)
    callbacks = [
        lgb.early_stopping(80, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    try:
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_set=[(X.iloc[eval_idx], y[eval_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=callbacks,
        )
    except TypeError:
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            eval_set=[(X.iloc[eval_idx], y[eval_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
        )
    return model


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train_fe = add_features(train)
test_fe = add_features(test)
y = train_fe[TARGET].astype(int).to_numpy()

feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
cat_cols = [
    "Compound",
    "Driver",
    "Race",
    "RacePhase",
    "DriverRace",
    "CompoundPhase",
    "Year",
    "Stint",
    "PitStop",
]
cat_cols = [c for c in cat_cols if c in feature_cols]

X_lgb = train_fe[feature_cols].copy()
T_lgb = test_fe[feature_cols].copy()
X_cb = train_fe[feature_cols].copy()
T_cb = test_fe[feature_cols].copy()

for c in cat_cols:
    cats = pd.Index(pd.concat([train_fe[c], test_fe[c]], axis=0).astype(str).unique())
    X_lgb[c] = pd.Categorical(train_fe[c].astype(str), categories=cats)
    T_lgb[c] = pd.Categorical(test_fe[c].astype(str), categories=cats)
    X_cb[c] = train_fe[c].astype(str).fillna("__NA__")
    T_cb[c] = test_fe[c].astype(str).fillna("__NA__")

base_params = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_estimators=900,
    learning_rate=0.035,
    num_leaves=80,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.10,
    reg_lambda=2.00,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)

wet_params = base_params.copy()
wet_params.update(
    n_estimators=650,
    learning_rate=0.045,
    num_leaves=48,
    min_child_samples=30,
    subsample=0.90,
    colsample_bytree=0.90,
    reg_lambda=1.50,
)

late_params = base_params.copy()
late_params.update(
    n_estimators=750,
    learning_rate=0.040,
    num_leaves=64,
    min_child_samples=50,
    subsample=0.88,
    colsample_bytree=0.88,
    reg_lambda=2.50,
)

groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)
if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(train_fe, y, groups))
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(train_fe, y))

oof_blend = np.zeros(len(train_fe), dtype=np.float32)
oof_parts = {name: np.zeros(len(train_fe), dtype=np.float32) for name in MODEL_NAMES}
test_blends = []
fold_aucs = []

wet_mask = train_fe["Compound"].astype(str).isin(["WET", "INTERMEDIATE"]).to_numpy()
late_mask = (train_fe["RaceProgress"].astype(float).to_numpy() >= 0.58) | (
    train_fe["TyreLife"].astype(float).to_numpy() >= 14
)

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    print(f"Fold {fold}/5")

    val_pred = {}
    test_pred = {}

    base = fit_lgb_model(X_lgb, y, tr_idx, va_idx, base_params, cat_cols)
    val_pred["base"] = base.predict_proba(X_lgb.iloc[va_idx])[:, 1]
    test_pred["base"] = base.predict_proba(T_lgb)[:, 1]
    del base
    gc.collect()

    cb = CatBoostClassifier(
        iterations=650,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED + fold,
        bootstrap_type="Bayesian",
        bagging_temperature=0.5,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    train_pool = Pool(X_cb.iloc[tr_idx], y[tr_idx], cat_features=cat_cols)
    valid_pool = Pool(X_cb.iloc[va_idx], y[va_idx], cat_features=cat_cols)
    cb.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    val_pred["cat"] = cb.predict_proba(valid_pool)[:, 1]
    test_pred["cat"] = cb.predict_proba(Pool(T_cb, cat_features=cat_cols))[:, 1]
    del cb, train_pool, valid_pool
    gc.collect()

    wet_tr = valid_subset(tr_idx, wet_mask, y, min_rows=1500)
    wet_va = valid_subset(va_idx, wet_mask, y, min_rows=200)
    wet_model = fit_lgb_model(X_lgb, y, wet_tr, wet_va, wet_params, cat_cols)
    val_pred["wet"] = wet_model.predict_proba(X_lgb.iloc[va_idx])[:, 1]
    test_pred["wet"] = wet_model.predict_proba(T_lgb)[:, 1]
    del wet_model
    gc.collect()

    late_tr = valid_subset(tr_idx, late_mask, y, min_rows=3000)
    late_va = valid_subset(va_idx, late_mask, y, min_rows=400)
    late_model = fit_lgb_model(X_lgb, y, late_tr, late_va, late_params, cat_cols)
    val_pred["late"] = late_model.predict_proba(X_lgb.iloc[va_idx])[:, 1]
    test_pred["late"] = late_model.predict_proba(T_lgb)[:, 1]
    del late_model
    gc.collect()

    for name in MODEL_NAMES:
        oof_parts[name][va_idx] = percentile_rank(val_pred[name])

    fold_blend = blend_rank_predictions(val_pred, train_fe.iloc[va_idx])
    oof_blend[va_idx] = fold_blend
    test_blends.append(blend_rank_predictions(test_pred, test_fe))

    fold_auc = roc_auc_score(y[va_idx], fold_blend)
    fold_aucs.append(float(fold_auc))
    print(f"Fold {fold} rank-gated blend ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof_blend)
for name in MODEL_NAMES:
    print(f"OOF {name} rank ROC AUC: {roc_auc_score(y, oof_parts[name]):.6f}")
print(f"OOF rank-gated blend ROC AUC: {cv_auc:.6f}")
print(f"Fold ROC AUC mean/std: {np.mean(fold_aucs):.6f} / {np.std(fold_aucs):.6f}")

test_final = percentile_rank(np.mean(np.vstack(test_blends), axis=0))
test_final = np.clip(test_final, 1e-6, 1 - 1e-6)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": np.clip(oof_blend, 1e-6, 1 - 1e-6),
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_out = sample[[ID_COL]].copy()
test_out[TARGET] = test_final
test_out.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
test_out.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

report = {
    "metric": "roc_auc",
    "cv_auc": float(cv_auc),
    "fold_auc": fold_aucs,
    "research_hypotheses_llm_claimed_used": ["000280"],
    "artifacts": {
        "submission": os.path.join(WORK_DIR, "submission.csv"),
        "oof_predictions": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
        "test_predictions": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    },
}
for name in ["result.json", "review.json"]:
    with open(os.path.join(WORK_DIR, name), "w") as f:
        json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
