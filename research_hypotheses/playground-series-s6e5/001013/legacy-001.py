import gc
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
ID_COL = "id"
TARGET = "PitNextLap"
INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

y = train[TARGET].astype(np.int8).to_numpy()
raw_features = [c for c in test.columns if c != ID_COL]
cat_cols = [
    c for c in raw_features if train[c].dtype == "object" or test[c].dtype == "object"
]

groups = (
    train["Year"].astype(str)
    + "|"
    + train["Race"].astype(str)
    + "|"
    + train["Driver"].astype(str)
)

for c in cat_cols:
    cats = pd.Index(
        pd.concat([train[c], test[c]], ignore_index=True).astype(str).unique()
    )
    dtype = pd.CategoricalDtype(categories=cats)
    train[c] = train[c].astype(str).astype(dtype)
    test[c] = test[c].astype(str).astype(dtype)

for df in (train, test):
    for c in df.select_dtypes(include=["float64"]).columns:
        if c != TARGET:
            df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=["int64"]).columns:
        if c != ID_COL:
            df[c] = pd.to_numeric(df[c], downcast="integer")

ENGINEERED = [
    "expected_tyre_life",
    "estimated_total_laps",
    "laps_remaining",
    "remaining_life_est",
    "tyre_age_vs_expected",
    "laps_remaining_vs_expected_life",
    "finish_tyre_deficit",
    "pit_window_pressure",
    "old_tyre_late_race_pressure",
    "dry_tyre",
    "wet_tyre",
    "late_race",
    "old_tyre",
    "dry_pit_window_pressure",
    "wet_pit_window_pressure",
]
FEATURES = raw_features + ENGINEERED

MONO_POS = {
    "tyre_age_vs_expected",
    "laps_remaining_vs_expected_life",
    "finish_tyre_deficit",
    "pit_window_pressure",
    "old_tyre_late_race_pressure",
    "dry_pit_window_pressure",
    "wet_pit_window_pressure",
}
MONO_NEG = {"remaining_life_est"}
MONO_CONSTRAINTS = [
    1 if c in MONO_POS else -1 if c in MONO_NEG else 0 for c in FEATURES
]


class PitFeatureBuilder:
    def __init__(self, quantile=0.80):
        self.quantile = quantile
        self.global_expected = 25.0
        self.expected_by_compound = {}

    def fit(self, df):
        tyre = pd.to_numeric(df["TyreLife"], errors="coerce").astype(float)
        val = float(np.nanquantile(tyre, self.quantile))
        self.global_expected = float(
            np.clip(val if np.isfinite(val) else 25.0, 5.0, 80.0)
        )

        tmp = pd.DataFrame(
            {
                "compound": df["Compound"].astype(str).to_numpy(),
                "tyre": tyre.to_numpy(),
            }
        )
        q = tmp.groupby("compound", observed=True)["tyre"].quantile(self.quantile)
        self.expected_by_compound = {
            str(k): float(np.clip(v, 5.0, 80.0)) for k, v in q.dropna().items()
        }
        return self

    def transform(self, df):
        out = df.copy()
        compound = out["Compound"].astype(str)
        tyre = (
            pd.to_numeric(out["TyreLife"], errors="coerce")
            .astype("float32")
            .clip(lower=1.0)
        )
        lap = (
            pd.to_numeric(out["LapNumber"], errors="coerce")
            .astype("float32")
            .clip(lower=1.0)
        )
        progress = (
            pd.to_numeric(out["RaceProgress"], errors="coerce")
            .astype("float32")
            .clip(0.01, 1.0)
        )

        expected = compound.map(self.expected_by_compound).astype("float32")
        expected = expected.fillna(self.global_expected).clip(lower=5.0, upper=80.0)

        total_laps = (lap / progress).clip(lower=1.0, upper=120.0)
        laps_remaining = (total_laps - lap).clip(lower=0.0, upper=120.0)
        remaining_life = expected - tyre

        age_ratio = (
            (tyre / expected)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .clip(0.0, 5.0)
        )
        deficit_ratio = (tyre + laps_remaining - expected) / expected
        deficit_ratio = (
            deficit_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
        )
        finish_deficit = deficit_ratio.clip(lower=0.0)

        pit_pressure = (age_ratio - 0.65).clip(lower=0.0) * (
            0.5 + progress
        ) + finish_deficit * progress
        pit_pressure = pit_pressure.clip(0.0, 10.0)
        late_pressure = (age_ratio * progress).clip(0.0, 5.0)

        dry = (~compound.isin(["INTERMEDIATE", "WET"])).astype(np.float32)
        wet = compound.isin(["INTERMEDIATE", "WET"]).astype(np.float32)

        out["expected_tyre_life"] = expected
        out["estimated_total_laps"] = total_laps
        out["laps_remaining"] = laps_remaining
        out["remaining_life_est"] = remaining_life
        out["tyre_age_vs_expected"] = age_ratio
        out["laps_remaining_vs_expected_life"] = deficit_ratio
        out["finish_tyre_deficit"] = finish_deficit
        out["pit_window_pressure"] = pit_pressure
        out["old_tyre_late_race_pressure"] = late_pressure
        out["dry_tyre"] = dry
        out["wet_tyre"] = wet
        out["late_race"] = (progress >= 0.65).astype(np.float32)
        out["old_tyre"] = (age_ratio >= 0.90).astype(np.float32)
        out["dry_pit_window_pressure"] = pit_pressure * dry
        out["wet_pit_window_pressure"] = pit_pressure * wet

        out[ENGINEERED] = (
            out[ENGINEERED]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )
        return out[FEATURES]


def make_model(seed, n_estimators, y_train):
    pos = max(float(np.sum(y_train)), 1.0)
    neg = max(float(len(y_train) - pos), 1.0)
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=int(n_estimators),
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=120,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.90,
        reg_alpha=0.1,
        reg_lambda=4.0,
        max_bin=255,
        min_data_per_group=50,
        cat_smooth=20.0,
        scale_pos_weight=float(np.sqrt(neg / pos)),
        monotone_constraints=MONO_CONSTRAINTS,
        monotone_constraints_method="advanced",
        random_state=seed,
        n_jobs=min(16, os.cpu_count() or 1),
        force_col_wise=True,
        verbosity=-1,
    )


train_raw = train[raw_features]
test_raw = test[raw_features]

try:
    if StratifiedGroupKFold is None:
        raise RuntimeError("StratifiedGroupKFold unavailable")
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(train_raw, y, groups))
    cv_name = "StratifiedGroupKFold"
except Exception:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(train_raw, y))
    cv_name = "StratifiedKFold"

print(f"Running {cv_name} with {N_SPLITS} folds", flush=True)

oof = np.zeros(len(train), dtype=np.float32)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    builder = PitFeatureBuilder().fit(train_raw.iloc[tr_idx])
    X_tr = builder.transform(train_raw.iloc[tr_idx])
    X_va = builder.transform(train_raw.iloc[va_idx])

    model = make_model(SEED + fold, 1600, y[tr_idx])
    model.fit(
        X_tr,
        y[tr_idx],
        eval_set=[(X_va, y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            early_stopping(100, first_metric_only=True, verbose=False),
            log_evaluation(0),
        ],
    )

    best_iter = getattr(model, "best_iteration_", None) or 1600
    best_iterations.append(int(best_iter))
    pred = model.predict_proba(X_va, num_iteration=best_iter)[:, 1].astype(np.float32)
    oof[va_idx] = pred

    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))
    print(f"fold {fold} roc_auc={auc:.6f} best_iteration={int(best_iter)}", flush=True)

    del builder, X_tr, X_va, model
    gc.collect()

cv_auc = float(roc_auc_score(y, oof))
final_estimators = int(np.clip(np.median(best_iterations), 100, 1600))
print(f"OOF ROC AUC={cv_auc:.6f}", flush=True)
print(f"Training final constrained model with {final_estimators} trees", flush=True)

final_builder = PitFeatureBuilder().fit(train_raw)
X_full = final_builder.transform(train_raw)
X_test = final_builder.transform(test_raw)

final_model = make_model(SEED, final_estimators, y)
final_model.fit(X_full, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1].astype(float)
test_pred = np.clip(test_pred, 0.0, 1.0)

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y.astype(int),
        "prediction": np.clip(oof, 0.0, 1.0),
    }
).to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

submission = sample[[ID_COL]].copy()
if len(sample) == len(test) and np.array_equal(
    sample[ID_COL].to_numpy(), test[ID_COL].to_numpy()
):
    submission[TARGET] = test_pred
else:
    pred_by_id = pd.Series(test_pred, index=test[ID_COL].to_numpy())
    submission[TARGET] = submission[ID_COL].map(pred_by_id).astype(float).to_numpy()

submission.to_csv(WORK_DIR / "submission.csv", index=False)
submission.to_csv(WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip")

result = {
    "metric": "roc_auc",
    "cv_auc": cv_auc,
    "fold_auc": fold_scores,
    "cv": cv_name,
    "final_estimators": final_estimators,
    "research_hypotheses_llm_claimed_used": ["001013"],
    "submission_path": str(WORK_DIR / "submission.csv"),
    "oof_path": str(WORK_DIR / "oof_predictions.csv.gz"),
    "test_predictions_path": str(WORK_DIR / "test_predictions.csv.gz"),
}
with open(WORK_DIR / "result_review.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("RESULT_REVIEW_JSON=" + json.dumps(result, sort_keys=True), flush=True)
