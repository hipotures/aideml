import gc
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
EMBARGO_EVENTS = 1
TARGET = "PitNextLap"
ID_COL = "id"

INPUT = Path("./input")
WORKING = Path("./working")
WORKING.mkdir(parents=True, exist_ok=True)

from lightgbm import LGBMClassifier, early_stopping, log_evaluation


def clean_columns(cols):
    seen, out = {}, []
    for c in cols:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_") or "f"
        k = seen.get(base, 0)
        seen[base] = k + 1
        out.append(base if k == 0 else f"{base}_{k}")
    return out


def make_event_key(df):
    return df["Year"].astype(str) + "|" + df["Race"].astype(str)


def build_features(train, test):
    full = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)

    lap = pd.to_numeric(full["LapNumber"], errors="coerce").clip(lower=1)
    progress = pd.to_numeric(full["RaceProgress"], errors="coerce").clip(lower=1e-3)
    tyre = pd.to_numeric(full["TyreLife"], errors="coerce").clip(lower=1)
    stint = pd.to_numeric(full["Stint"], errors="coerce").clip(lower=1)
    deg = pd.to_numeric(full["Cumulative_Degradation"], errors="coerce")
    lap_delta = pd.to_numeric(full["LapTime_Delta"], errors="coerce")
    lap_time = pd.to_numeric(full["LapTime (s)"], errors="coerce")
    pos_change = pd.to_numeric(full["Position_Change"], errors="coerce")

    est_total_laps = (lap / progress).clip(lower=lap, upper=120)
    full["Event"] = full["Year"].astype(str) + "_" + full["Race"].astype(str)
    full["Driver_Race"] = full["Driver"].astype(str) + "_" + full["Race"].astype(str)
    full["Compound_Stint"] = (
        full["Compound"].astype(str) + "_" + full["Stint"].astype(str)
    )
    full["EstimatedRaceLaps"] = est_total_laps
    full["LapsRemaining"] = est_total_laps - lap
    full["TyreLifeFracRace"] = tyre / est_total_laps.replace(0, np.nan)
    full["TyreLifeFracLap"] = tyre / lap
    full["DegPerTyreLap"] = deg / tyre
    full["DegPerRaceProgress"] = deg / progress
    full["AbsLapTimeDelta"] = lap_delta.abs()
    full["AbsPositionChange"] = pos_change.abs()
    full["LogLapTime"] = np.log1p(lap_time.clip(lower=0))
    full["LateRaceTyreAge"] = progress * tyre
    full["StintTyreRatio"] = tyre / stint

    features = full.drop(columns=[ID_COL])
    cat_cols = features.select_dtypes(include=["object"]).columns.tolist()

    old_cols = features.columns.tolist()
    new_cols = clean_columns(old_cols)
    name_map = dict(zip(old_cols, new_cols))
    features.columns = new_cols
    cat_cols = [name_map[c] for c in cat_cols]

    for c in cat_cols:
        features[c] = features[c].astype("category")

    features = features.replace([np.inf, -np.inf], np.nan)
    return (
        features.iloc[: len(train)].copy(),
        features.iloc[len(train) :].copy(),
        cat_cols,
    )


def make_purged_event_folds(train):
    event_key = make_event_key(train)
    order = (
        pd.DataFrame({"event": event_key, "id": train[ID_COL].values})
        .groupby("event", sort=False)["id"]
        .min()
        .sort_values()
        .index.to_numpy()
    )
    chunks = np.array_split(np.arange(len(order)), N_SPLITS)
    folds = []

    for fold, chunk in enumerate(chunks):
        start, end = int(chunk[0]), int(chunk[-1]) + 1
        embargo_start = max(0, start - EMBARGO_EVENTS)
        embargo_end = min(len(order), end + EMBARGO_EVENTS)

        val_events = set(order[start:end])
        removed_events = set(order[embargo_start:embargo_end])

        val_idx = np.flatnonzero(event_key.isin(val_events).to_numpy())
        tr_idx = np.flatnonzero(~event_key.isin(removed_events).to_numpy())
        folds.append((tr_idx, val_idx, start, end))

    return folds


def rank01(x):
    return pd.Series(np.asarray(x)).rank(method="average").to_numpy(dtype=float) / (
        len(x) + 1.0
    )


def rank_blend(pred_list):
    ranks = [rank01(p) for p in pred_list]
    return np.mean(ranks, axis=0)


def fit_binned_isotonic(x, y, n_bins=256):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    bins = min(n_bins, max(8, np.unique(x).size))

    order = np.argsort(x, kind="mergesort")
    bin_id = np.empty(n, dtype=np.int32)
    bin_id[order] = np.minimum(np.arange(n) * bins // n, bins - 1)

    frame = pd.DataFrame({"bin": bin_id, "x": x, "y": y})
    grouped = frame.groupby("bin", sort=True).agg(
        x=("x", "mean"), y=("y", "mean"), n=("y", "size")
    )
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip")
    iso.fit(
        grouped["x"].to_numpy(),
        grouped["y"].to_numpy(),
        sample_weight=np.sqrt(grouped["n"].to_numpy()),
    )
    return iso


def make_model(params, seed_offset=0, n_estimators=None):
    p = dict(params)
    p["random_state"] = p.get("random_state", SEED) + seed_offset
    if n_estimators is not None:
        p["n_estimators"] = int(max(50, n_estimators))
    return LGBMClassifier(**p)


def auc_safe(y_true, pred):
    return float(roc_auc_score(y_true, pred))


train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
X, X_test, cat_cols = build_features(train, test)
folds = make_purged_event_folds(train)

threads = max(1, min(8, os.cpu_count() or 1))
model_specs = {
    "lgbm_regularized": {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 1600,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "min_child_samples": 90,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.05,
        "reg_lambda": 2.5,
        "n_jobs": threads,
        "verbose": -1,
        "force_col_wise": True,
    },
    "lgbm_conservative": {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 1800,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 160,
        "subsample": 0.90,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.20,
        "reg_lambda": 5.0,
        "n_jobs": threads,
        "verbose": -1,
        "force_col_wise": True,
    },
}

candidate_results = {}
for model_name, params in model_specs.items():
    oof = np.zeros(len(train), dtype=np.float32)
    fold_aucs, best_iters = [], []

    for fold_id, (tr_idx, va_idx, _, _) in enumerate(folds):
        model = make_model(params, seed_offset=fold_id)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = pred
        fold_aucs.append(auc_safe(y[va_idx], pred))
        best_iters.append(
            getattr(model, "best_iteration_", None) or params["n_estimators"]
        )
        del model
        gc.collect()

    candidate_results[model_name] = {
        "oof": oof,
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)),
        "best_iteration": int(np.median(best_iters)),
    }
    print(f"{model_name} purged folds AUC: {fold_aucs}")
    print(
        f"{model_name} purged mean/std AUC: {np.mean(fold_aucs):.6f} / {np.std(fold_aucs, ddof=1):.6f}"
    )

best_mean = max(v["mean_auc"] for v in candidate_results.values())
median_std = float(np.median([v["std_auc"] for v in candidate_results.values()]))
selected = [
    name
    for name, res in candidate_results.items()
    if res["mean_auc"] >= best_mean - 0.0025
    and res["std_auc"] <= max(0.01, 1.5 * median_std)
]
if not selected:
    selected = [max(candidate_results, key=lambda n: candidate_results[n]["mean_auc"])]

purged_raw = rank_blend([candidate_results[n]["oof"] for n in selected])
purged_raw_auc = auc_safe(y, purged_raw)

purged_cal = np.zeros(len(train), dtype=np.float32)
all_idx = np.arange(len(train))
for _, va_idx, _, _ in folds:
    cal_idx = np.setdiff1d(all_idx, va_idx, assume_unique=False)
    iso = fit_binned_isotonic(purged_raw[cal_idx], y[cal_idx])
    purged_cal[va_idx] = iso.predict(purged_raw[va_idx])
purged_cal_auc = auc_safe(y, purged_cal)

test_model_preds = []
for model_name in selected:
    params = model_specs[model_name]
    n_est = candidate_results[model_name]["best_iteration"]
    model = make_model(params, seed_offset=1000, n_estimators=n_est)
    model.fit(X, y, categorical_feature=cat_cols)
    test_model_preds.append(model.predict_proba(X_test)[:, 1])
    del model
    gc.collect()

test_raw = rank_blend(test_model_preds)
final_iso = fit_binned_isotonic(purged_raw, y)
test_pred = np.clip(final_iso.predict(test_raw), 1e-6, 1 - 1e-6)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
naive_oofs, naive_summary = [], {}
for model_name in selected:
    params = model_specs[model_name]
    oof = np.zeros(len(train), dtype=np.float32)
    fold_aucs = []
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = make_model(params, seed_offset=2000 + fold_id)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = pred
        fold_aucs.append(auc_safe(y[va_idx], pred))
        del model
        gc.collect()
    naive_oofs.append(oof)
    naive_summary[model_name] = {
        "fold_aucs": [float(v) for v in fold_aucs],
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)),
    }

naive_raw = rank_blend(naive_oofs)
naive_auc = auc_safe(y, naive_raw)

oof_df = pd.DataFrame(
    {"row": np.arange(len(train)), "target": y, "prediction": purged_cal}
)
oof_df.to_csv(WORKING / "oof_predictions.csv.gz", index=False, compression="gzip")

test_pred_df = sample.copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    WORKING / "test_predictions.csv.gz", index=False, compression="gzip"
)
test_pred_df.to_csv(WORKING / "submission.csv", index=False)

review = {
    "research_hypotheses_llm_claimed_used": ["000844"],
    "metric": "roc_auc",
    "selected_models": selected,
    "purged_rank_blend_auc": float(purged_raw_auc),
    "purged_calibrated_rank_blend_auc": float(purged_cal_auc),
    "naive_rank_blend_auc": float(naive_auc),
    "candidate_purged_summary": {
        name: {
            "fold_aucs": [float(v) for v in res["fold_aucs"]],
            "mean_auc": float(res["mean_auc"]),
            "std_auc": float(res["std_auc"]),
            "best_iteration": int(res["best_iteration"]),
        }
        for name, res in candidate_results.items()
    },
    "naive_selected_summary": naive_summary,
    "embargo_events_each_side": EMBARGO_EVENTS,
}
with open(WORKING / "result_review.json", "w") as f:
    json.dump(review, f, indent=2)

print(f"Selected stable models: {selected}")
print(f"Purged 5-fold rank-blend ROC AUC: {purged_raw_auc:.6f}")
print(f"Purged 5-fold calibrated ROC AUC: {purged_cal_auc:.6f}")
print(f"Naive shuffled 5-fold rank-blend ROC AUC: {naive_auc:.6f}")
print(json.dumps(review, indent=2))
