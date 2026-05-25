import os
import json
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
os.environ.setdefault("MKL_NUM_THREADS", str(min(8, os.cpu_count() or 1)))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID = "id"
CAT_COLS = ["Driver", "Race", "Compound", "Year", "Stint"]
CAT_SEEDS = [13, 47]
N_SPLITS = 5
AUTOGLUON_TIME_PER_FOLD = 120
CAT_WEIGHT = 0.70
AG_WEIGHT = 0.30


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=np.abs(b) > 1e-9)


def add_features(df):
    out = df.copy()

    def num(name, default=0.0):
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").astype(float)
        return pd.Series(default, index=out.index, dtype=float)

    lap = num("LapNumber")
    progress = num("RaceProgress").clip(lower=1e-4)
    tyre = num("TyreLife").clip(lower=1.0)
    deg = num("Cumulative_Degradation")
    lap_time = num("LapTime (s)")
    lap_delta = num("LapTime_Delta")
    pos_change = num("Position_Change")
    position = num("Position")
    stint = num("Stint").clip(lower=1.0)

    est_laps = safe_div(lap, progress)
    out["EstimatedRaceLaps"] = est_laps
    out["LapsRemaining"] = est_laps - lap
    out["TyreLifeToLap"] = safe_div(tyre, lap.clip(lower=1.0))
    out["TyreLifeToRace"] = safe_div(tyre, np.maximum(est_laps, 1.0))
    out["DegradationPerTyreLife"] = safe_div(deg, tyre)
    out["DegradationPerLap"] = safe_div(deg, lap.clip(lower=1.0))
    out["LapTimePerTyreLife"] = safe_div(lap_time, tyre)
    out["LapDeltaAbs"] = lap_delta.abs()
    out["PositionChangeAbs"] = pos_change.abs()
    out["InversePosition"] = safe_div(1.0, position.clip(lower=1.0))
    out["ProgressXTyreLife"] = progress * tyre
    out["StintXTyreLife"] = stint * tyre
    out["LateRaceOldTyre"] = progress * tyre

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def positive_proba(proba):
    if isinstance(proba, pd.DataFrame):
        if 1 in proba.columns:
            return proba[1].to_numpy(dtype=float)
        if "1" in proba.columns:
            return proba["1"].to_numpy(dtype=float)
        return proba.iloc[:, -1].to_numpy(dtype=float)
    arr = np.asarray(proba, dtype=float)
    return arr[:, -1] if arr.ndim == 2 else arr


def clean_proba(p):
    p = np.asarray(p, dtype=float)
    finite = np.isfinite(p)
    fill = float(np.mean(p[finite])) if finite.any() else 0.0
    return np.clip(np.nan_to_num(p, nan=fill, posinf=1.0, neginf=0.0), 0.0, 1.0)


train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
groups = train["Race"].astype(str) + "_" + train["Year"].astype(str)

full = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
full = add_features(full)

for c in CAT_COLS:
    if c in full.columns:
        full[c] = full[c].fillna("__MISSING__").astype(str)

features = [c for c in full.columns if c != ID]
cat_features = [c for c in CAT_COLS if c in features]

X = full.iloc[: len(train)][features].copy()
X_test = full.iloc[len(train) :][features].copy()

if StratifiedGroupKFold is not None:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=2026)
    splits = list(cv.split(X, y, groups))
else:
    cv = GroupKFold(n_splits=N_SPLITS)
    splits = list(cv.split(X, y, groups))

try:
    from autogluon.tabular import TabularPredictor

    HAVE_AG = True
except Exception as e:
    print(f"AutoGluon unavailable, using CatBoost-only fallback: {e}")
    HAVE_AG = False

oof_cat = np.zeros(len(X), dtype=float)
oof_ag = np.full(len(X), np.nan, dtype=float)
test_cat_folds = []
test_ag_folds = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    pos = max(int(y_tr.sum()), 1)
    neg = int(len(y_tr) - y_tr.sum())
    scale_pos_weight = neg / pos

    fold_cat_val = np.zeros(len(va_idx), dtype=float)
    fold_cat_test = np.zeros(len(X_test), dtype=float)

    for seed in CAT_SEEDS:
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.06,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            scale_pos_weight=scale_pos_weight,
            max_ctr_complexity=3,
            early_stopping_rounds=80,
            thread_count=min(8, os.cpu_count() or 1),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            Pool(X_tr, y_tr, cat_features=cat_features),
            eval_set=Pool(X_va, y_va, cat_features=cat_features),
            use_best_model=True,
        )
        fold_cat_val += positive_proba(
            model.predict_proba(Pool(X_va, cat_features=cat_features))
        ) / len(CAT_SEEDS)
        fold_cat_test += positive_proba(
            model.predict_proba(Pool(X_test, cat_features=cat_features))
        ) / len(CAT_SEEDS)

    fold_cat_val = clean_proba(fold_cat_val)
    fold_cat_test = clean_proba(fold_cat_test)
    oof_cat[va_idx] = fold_cat_val
    test_cat_folds.append(fold_cat_test)

    ag_val = None
    ag_test = None
    if HAVE_AG:
        try:
            ag_train = X_tr.copy()
            ag_train[TARGET] = y_tr
            ag_valid = X_va.copy()
            ag_valid[TARGET] = y_va
            ag_path = WORK / f"autogluon_fold{fold}_{int(time.time())}_{os.getpid()}"

            predictor = TabularPredictor(
                label=TARGET,
                problem_type="binary",
                eval_metric="roc_auc",
                path=str(ag_path),
                verbosity=0,
            )
            predictor.fit(
                train_data=ag_train,
                tuning_data=ag_valid,
                hyperparameters={"GBM": {}, "XGB": {}},
                presets="medium_quality",
                time_limit=AUTOGLUON_TIME_PER_FOLD,
                refit_full=False,
            )
            ag_val = clean_proba(positive_proba(predictor.predict_proba(X_va)))
            ag_test = clean_proba(positive_proba(predictor.predict_proba(X_test)))
            oof_ag[va_idx] = ag_val
            test_ag_folds.append(ag_test)
        except Exception as e:
            print(f"Fold {fold}: AutoGluon failed, continuing with CatBoost only: {e}")

    fold_pred = (
        clean_proba(CAT_WEIGHT * fold_cat_val + AG_WEIGHT * ag_val)
        if ag_val is not None
        else fold_cat_val
    )
    fold_auc = (
        roc_auc_score(y_va, fold_pred) if len(np.unique(y_va)) == 2 else float("nan")
    )
    cat_auc = (
        roc_auc_score(y_va, fold_cat_val) if len(np.unique(y_va)) == 2 else float("nan")
    )
    print(
        f"Fold {fold} ROC AUC: blend={fold_auc:.6f} catboost={cat_auc:.6f} positives={int(y_va.sum())}"
    )

test_cat = clean_proba(np.mean(np.vstack(test_cat_folds), axis=0))
have_full_ag = np.isfinite(oof_ag).all() and len(test_ag_folds) == N_SPLITS

if have_full_ag:
    test_ag = clean_proba(np.mean(np.vstack(test_ag_folds), axis=0))
    oof_pred = clean_proba(CAT_WEIGHT * oof_cat + AG_WEIGHT * oof_ag)
    test_pred = clean_proba(CAT_WEIGHT * test_cat + AG_WEIGHT * test_ag)
else:
    oof_pred = clean_proba(oof_cat)
    test_pred = clean_proba(test_cat)

cv_auc = roc_auc_score(y, oof_pred)
print(f"5-fold Race-Year grouped CV ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = test_pred
submission[[ID, TARGET]].to_csv(WORK / "submission.csv", index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

submission[[ID, TARGET]].to_csv(
    WORK / "test_predictions.csv.gz", index=False, compression="gzip"
)

review = {
    "metric": "roc_auc",
    "cv_auc": float(cv_auc),
    "validation": "5-fold grouped CV by Race-Year",
    "used_autogluon_gbm_xgb_blend": bool(have_full_ag),
    "research_hypotheses_llm_claimed_used": ["000622"],
}
print(json.dumps(review, sort_keys=True))
