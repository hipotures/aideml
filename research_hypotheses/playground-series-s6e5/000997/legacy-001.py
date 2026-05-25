import os
import gc
import json
import math
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GroupKFold, StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

try:
    from scipy.special import ndtr

    def norm_cdf(z):
        return ndtr(z)

except Exception:
    _erf = np.vectorize(math.erf, otypes=[float])

    def norm_cdf(z):
        z = np.asarray(z, dtype=np.float64)
        return 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))


warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID = "id"
SEED = 42
N_SPLITS = 5
BASE_ROUNDS = 400
AFT_ROUNDS = 300
AFT_SCALE = 0.85
BASE_BLEND_WEIGHT = 0.75

os.makedirs(WORKING_DIR, exist_ok=True)


def numeric_col(df, name, default=0.0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def make_features(train_df, test_df):
    all_df = pd.concat(
        [train_df.drop(columns=[TARGET]), test_df], axis=0, ignore_index=True
    )
    X = all_df.drop(columns=[ID], errors="ignore").copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        s = X[c].astype("string").fillna("__NA__").astype(str)
        freq = s.value_counts(normalize=True)
        X[f"{c}_freq"] = s.map(freq).astype(float)
        X[c] = s

    compound = (
        X["Compound"].astype(str)
        if "Compound" in X.columns
        else pd.Series("", index=X.index)
    )

    lap = numeric_col(X, "LapNumber").clip(lower=1.0)
    tyre = numeric_col(X, "TyreLife").clip(lower=0.0)
    progress = numeric_col(X, "RaceProgress").clip(lower=1e-4, upper=1.5)
    stint = numeric_col(X, "Stint").clip(lower=1.0)
    laptime = numeric_col(X, "LapTime (s)")
    delta = numeric_col(X, "LapTime_Delta")
    degr = numeric_col(X, "Cumulative_Degradation")
    position = numeric_col(X, "Position")
    pos_change = numeric_col(X, "Position_Change")
    pitstop = numeric_col(X, "PitStop")

    estimated_total = (lap / progress).clip(lower=1.0, upper=120.0)
    remaining = (estimated_total - lap).clip(lower=0.0, upper=120.0)

    X["estimated_total_laps"] = estimated_total
    X["estimated_laps_remaining"] = remaining
    X["tyre_to_lap_clock"] = tyre / (lap + 1.0)
    X["tyre_to_remaining_clock"] = tyre / (remaining + 1.0)
    X["stint_progress_proxy"] = tyre / (tyre + remaining + 1.0)
    X["clock_product"] = lap * tyre
    X["clock_gap"] = lap - tyre
    X["tyre_life_sq"] = tyre * tyre
    X["lap_number_sq"] = lap * lap
    X["degradation_per_tyre_lap"] = degr / (tyre + 1.0)
    X["delta_per_tyre_lap"] = delta / (tyre + 1.0)
    X["abs_laptime_delta"] = np.abs(delta)
    X["position_x_progress"] = position * progress
    X["position_change_abs"] = np.abs(pos_change)
    X["pitstop_or_fresh_tyre"] = ((pitstop > 0) | (tyre <= 2)).astype(np.int8)
    X["stint_x_tyre"] = stint * tyre
    X["is_wet_family"] = compound.isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    X["laptime_per_lap_clock"] = laptime / (lap + 1.0)

    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])

    X = X.replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    X = X.astype(np.float32)

    n_train = len(train_df)
    return X.iloc[:n_train].to_numpy(dtype=np.float32), X.iloc[n_train:].to_numpy(
        dtype=np.float32
    )


def make_aft_bounds(df, y):
    n = len(df)
    lower = np.ones(n, dtype=np.float32)
    upper = np.full(n, np.inf, dtype=np.float32)

    key_cols = [c for c in ["Year", "Race", "Driver", "Stint"] if c in df.columns]
    work = pd.DataFrame(
        {
            "_row": np.arange(n, dtype=np.int64),
            "_target": y.astype(np.int8),
            "_lap": pd.to_numeric(df["LapNumber"], errors="coerce")
            .fillna(0)
            .astype(float),
        }
    )
    for c in key_cols:
        work[c] = df[c].astype(str).values

    sort_cols = key_cols + ["_lap", "_row"]
    work = work.sort_values(sort_cols, kind="mergesort")

    grouped = work.groupby(key_cols, sort=False) if key_cols else [(None, work)]
    for _, g in grouped:
        laps = g["_lap"].to_numpy(dtype=np.float32)
        rows = g["_row"].to_numpy(dtype=np.int64)
        targets = g["_target"].to_numpy(dtype=np.int8)
        if len(g) == 0:
            continue

        next_event_lap = np.nan
        last_lap = float(laps[-1])

        for i in range(len(g) - 1, -1, -1):
            if targets[i] == 1:
                next_event_lap = float(laps[i])

            row = rows[i]
            if np.isfinite(next_event_lap):
                tte = max(1.0, next_event_lap - float(laps[i]) + 1.0)
                lower[row] = tte
                upper[row] = tte
            else:
                lower[row] = max(2.0, last_lap - float(laps[i]) + 2.0)
                upper[row] = np.inf

    return lower, upper


def make_groups(df):
    key_cols = [c for c in ["Year", "Race", "Driver"] if c in df.columns]
    if not key_cols:
        return np.arange(len(df))
    return df[key_cols].astype(str).agg("|".join, axis=1).to_numpy()


def get_splits(X, y, groups):
    if StratifiedGroupKFold is not None:
        try:
            sgkf = StratifiedGroupKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=SEED
            )
            splits = list(sgkf.split(X, y, groups))
            if all(np.unique(y[va]).size == 2 for _, va in splits):
                return splits
        except Exception:
            pass

    try:
        gkf = GroupKFold(n_splits=N_SPLITS)
        splits = list(gkf.split(X, y, groups))
        if all(np.unique(y[va]).size == 2 for _, va in splits):
            return splits
    except Exception:
        pass

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(X, y))


def aft_next_lap_hazard(model, dmat):
    mu = np.asarray(model.predict(dmat, output_margin=True), dtype=np.float64)
    z_hi = (np.log(1.5) - mu) / AFT_SCALE
    z_lo = (np.log(0.5) - mu) / AFT_SCALE
    cdf_hi = norm_cdf(z_hi)
    cdf_lo = norm_cdf(z_lo)
    hazard = (cdf_hi - cdf_lo) / np.maximum(1.0 - cdf_lo, 1e-9)
    return np.clip(hazard, 1e-6, 1.0 - 1e-6).astype(np.float32)


def safe_auc(y_true, pred):
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, pred))


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
X_train, X_test = make_features(train, test)
aft_lower, aft_upper = make_aft_bounds(train, y)
groups = make_groups(train)
splits = get_splits(X_train, y, groups)

oof_base = np.zeros(len(train), dtype=np.float32)
oof_aft = np.zeros(len(train), dtype=np.float32)
test_base = np.zeros(len(test), dtype=np.float64)
test_aft = np.zeros(len(test), dtype=np.float64)

dtest = xgb.DMatrix(X_test)
nthread = max(1, min(16, os.cpu_count() or 1))

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    y_tr = y[tr_idx]
    pos = float(np.sum(y_tr == 1))
    neg = float(np.sum(y_tr == 0))
    scale_pos_weight = min(50.0, neg / max(pos, 1.0))

    dtr_base = xgb.DMatrix(X_train[tr_idx], label=y_tr)
    dva = xgb.DMatrix(X_train[va_idx])

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "eta": 0.04,
        "max_depth": 5,
        "min_child_weight": 25,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "lambda": 2.0,
        "alpha": 0.05,
        "scale_pos_weight": scale_pos_weight,
        "seed": SEED + fold,
        "nthread": nthread,
    }
    base_model = xgb.train(
        base_params, dtr_base, num_boost_round=BASE_ROUNDS, verbose_eval=False
    )

    dtr_aft = xgb.DMatrix(X_train[tr_idx])
    dtr_aft.set_float_info("label_lower_bound", aft_lower[tr_idx])
    dtr_aft.set_float_info("label_upper_bound", aft_upper[tr_idx])

    aft_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "tree_method": "hist",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": AFT_SCALE,
        "eta": 0.05,
        "max_depth": 4,
        "min_child_weight": 30,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "lambda": 2.0,
        "alpha": 0.05,
        "seed": SEED + 100 + fold,
        "nthread": nthread,
    }
    aft_model = xgb.train(
        aft_params, dtr_aft, num_boost_round=AFT_ROUNDS, verbose_eval=False
    )

    base_va = np.clip(base_model.predict(dva), 1e-6, 1.0 - 1e-6).astype(np.float32)
    aft_va = aft_next_lap_hazard(aft_model, dva)

    oof_base[va_idx] = base_va
    oof_aft[va_idx] = aft_va

    test_base += base_model.predict(dtest) / len(splits)
    test_aft += aft_next_lap_hazard(aft_model, dtest) / len(splits)

    fold_blend = BASE_BLEND_WEIGHT * base_va + (1.0 - BASE_BLEND_WEIGHT) * aft_va
    print(f"Fold {fold} ROC AUC: {safe_auc(y[va_idx], fold_blend):.6f}")

    del dtr_base, dtr_aft, dva, base_model, aft_model
    gc.collect()

oof_pred = BASE_BLEND_WEIGHT * oof_base + (1.0 - BASE_BLEND_WEIGHT) * oof_aft
test_pred = BASE_BLEND_WEIGHT * test_base + (1.0 - BASE_BLEND_WEIGHT) * test_aft
test_pred = np.clip(test_pred, 1e-6, 1.0 - 1e-6)

baseline_auc = safe_auc(y, oof_base)
aft_auc = safe_auc(y, oof_aft)
blend_auc = safe_auc(y, oof_pred)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample[[ID]].copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

submission = sample[[ID]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

print(f"OOF ROC AUC baseline: {baseline_auc:.6f}")
print(f"OOF ROC AUC recurrent_event_aft: {aft_auc:.6f}")
print(f"OOF ROC AUC blended: {blend_auc:.6f}")

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": blend_auc,
            "baseline_cv_roc_auc": baseline_auc,
            "recurrent_event_aft_cv_roc_auc": aft_auc,
            "research_hypotheses_llm_claimed_used": ["000997"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        },
        indent=2,
    )
)
