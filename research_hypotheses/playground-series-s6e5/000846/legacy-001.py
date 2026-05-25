import os
import gc
import json
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 846
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

compound_limits = {
    "SOFT": 21.0,
    "MEDIUM": 30.0,
    "HARD": 39.0,
    "INTERMEDIATE": 19.0,
    "WET": 18.0,
}

cat_cols = ["Compound", "Driver", "Race", "Year"]

positive_mono = [
    "tyre_life_ratio",
    "service_pressure",
    "extreme_wear_pressure",
    "degradation_pressure",
    "finish_window_pressure",
    "cannot_finish",
    "cannot_finish_margin",
    "worn_finish_load",
]

negative_mono = [
    "fresh_after_pit",
    "fresh_stint_cooldown",
    "pitstop_cooldown",
    "new_stint_cooldown",
]

neutral_num = [
    "LapNumber",
    "RaceProgress",
    "Stint",
    "Position",
    "Position_Change",
    "estimated_total_laps",
    "laps_remaining",
    "compound_life_limit",
    "lap_time_log",
    "lap_time_delta_clip",
    "time_loss_pressure",
]

feature_cols = cat_cols + positive_mono + negative_mono + neutral_num


def add_features(df):
    out = df.copy()

    for c in cat_cols:
        out[c] = out[c].astype("object").where(out[c].notna(), "__NA__").astype(str)

    lap = pd.to_numeric(out["LapNumber"], errors="coerce").fillna(0).astype("float32")
    progress = (
        pd.to_numeric(out["RaceProgress"], errors="coerce")
        .fillna(0)
        .clip(0.01, 1.0)
        .astype("float32")
    )
    tyre = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(0).astype("float32")
    pitstop = (
        pd.to_numeric(out["PitStop"], errors="coerce")
        .fillna(0)
        .clip(0, 1)
        .astype("float32")
    )
    stint = pd.to_numeric(out["Stint"], errors="coerce").fillna(1).astype("float32")
    degradation = (
        pd.to_numeric(out["Cumulative_Degradation"], errors="coerce")
        .fillna(0)
        .astype("float32")
    )
    lap_time = (
        pd.to_numeric(out["LapTime (s)"], errors="coerce")
        .fillna(0)
        .clip(0, 600)
        .astype("float32")
    )
    lap_delta = (
        pd.to_numeric(out["LapTime_Delta"], errors="coerce")
        .fillna(0)
        .clip(-60, 60)
        .astype("float32")
    )

    limit = out["Compound"].map(compound_limits).fillna(28.0).astype("float32")
    estimated_total = (lap / progress).replace([np.inf, -np.inf], np.nan).fillna(lap)
    estimated_total = np.maximum(estimated_total, lap)
    estimated_total = np.minimum(estimated_total, 90.0).astype("float32")
    remaining = np.maximum(0.0, estimated_total - lap).astype("float32")

    out["LapNumber"] = lap
    out["RaceProgress"] = progress
    out["Stint"] = stint
    out["Position"] = (
        pd.to_numeric(out["Position"], errors="coerce").fillna(0).astype("float32")
    )
    out["Position_Change"] = (
        pd.to_numeric(out["Position_Change"], errors="coerce")
        .fillna(0)
        .astype("float32")
    )
    out["estimated_total_laps"] = estimated_total
    out["laps_remaining"] = remaining
    out["compound_life_limit"] = limit

    out["tyre_life_ratio"] = (tyre / limit).clip(0, 4).astype("float32")
    out["service_pressure"] = (
        ((tyre - 0.45 * limit) / (0.55 * limit)).clip(0, 3).astype("float32")
    )
    out["extreme_wear_pressure"] = ((tyre - limit) / limit).clip(0, 3).astype("float32")
    out["degradation_pressure"] = (
        (np.maximum(degradation, 0) / (15.0 * limit)).clip(0, 6).astype("float32")
    )

    late_window = ((progress - 0.55) / 0.35).clip(0, 1)
    enough_laps_left = ((remaining - 1.0) / 8.0).clip(0, 1)
    out["finish_window_pressure"] = (late_window * enough_laps_left).astype("float32")

    finish_load = tyre + remaining
    out["cannot_finish"] = ((finish_load > limit) & (remaining > 1.0)).astype("float32")
    out["cannot_finish_margin"] = (
        ((finish_load - limit) / limit).clip(0, 4).astype("float32")
    )
    out["worn_finish_load"] = (
        (out["tyre_life_ratio"] * out["finish_window_pressure"])
        .clip(0, 4)
        .astype("float32")
    )

    out["fresh_after_pit"] = ((4.0 - tyre) / 4.0).clip(0, 1).astype("float32")
    out["fresh_stint_cooldown"] = ((6.0 - tyre) / 6.0).clip(0, 1).astype("float32")
    out["pitstop_cooldown"] = pitstop.astype("float32")
    out["new_stint_cooldown"] = (((stint > 1) & (tyre <= 3)) | (pitstop > 0)).astype(
        "float32"
    )

    out["lap_time_log"] = np.log1p(lap_time).astype("float32")
    out["lap_time_delta_clip"] = lap_delta.astype("float32")
    out["time_loss_pressure"] = (
        (lap_delta.clip(lower=0) / 20.0).clip(0, 5).astype("float32")
    )

    for c in feature_cols:
        if c not in cat_cols:
            out[c] = (
                pd.to_numeric(out[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .astype("float32")
            )

    return out[feature_cols]


X = add_features(train)
X_test = add_features(test)
y = train[TARGET].astype(int).to_numpy()

cat_idx = [feature_cols.index(c) for c in cat_cols]
mono_map = {c: 1 for c in positive_mono}
mono_map.update({c: -1 for c in negative_mono})
monotone_constraints = [mono_map.get(c, 0) for c in feature_cols]

common_params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=500,
    learning_rate=0.055,
    depth=6,
    l2_leaf_reg=8.0,
    random_strength=0.6,
    bootstrap_type="Bernoulli",
    subsample=0.82,
    auto_class_weights="SqrtBalanced",
    allow_writing_files=False,
    verbose=False,
    thread_count=max(1, min(10, os.cpu_count() or 1)),
)


def make_model(seed, constrained=False, iterations=None):
    params = common_params.copy()
    params["random_seed"] = seed
    if iterations is not None:
        params["iterations"] = int(iterations)
    if constrained:
        params["monotone_constraints"] = monotone_constraints
    return CatBoostClassifier(**params)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_constrained = np.zeros(len(train), dtype=np.float32)
oof_baseline = np.zeros(len(train), dtype=np.float32)
fold_constrained_auc = []
fold_baseline_auc = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_idx)
    valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_idx)

    baseline = make_model(SEED + fold, constrained=False)
    baseline.fit(
        train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=75
    )
    base_pred = baseline.predict_proba(valid_pool)[:, 1]
    oof_baseline[va_idx] = base_pred
    base_auc = roc_auc_score(y[va_idx], base_pred)
    fold_baseline_auc.append(base_auc)

    constrained = make_model(SEED + 100 + fold, constrained=True)
    constrained.fit(
        train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=75
    )
    con_pred = constrained.predict_proba(valid_pool)[:, 1]
    oof_constrained[va_idx] = con_pred
    con_auc = roc_auc_score(y[va_idx], con_pred)
    fold_constrained_auc.append(con_auc)

    best_iter = constrained.get_best_iteration()
    best_iters.append(
        common_params["iterations"]
        if best_iter is None or best_iter <= 0
        else best_iter + 1
    )

    print(f"fold={fold} unconstrained_auc={base_auc:.6f} constrained_auc={con_auc:.6f}")

    del baseline, constrained, train_pool, valid_pool
    gc.collect()

baseline_cv_auc = roc_auc_score(y, oof_baseline)
constrained_cv_auc = roc_auc_score(y, oof_constrained)
print(f"unconstrained_baseline_5fold_roc_auc={baseline_cv_auc:.6f}")
print(f"monotone_constrained_5fold_roc_auc={constrained_cv_auc:.6f}")

oof_path = os.path.join(WORK_DIR, "oof_predictions.csv.gz")
pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y.astype(np.float32),
        "prediction": oof_constrained,
    }
).to_csv(oof_path, index=False, compression="gzip")

final_iterations = int(
    np.clip(np.mean(best_iters) * 1.10, 120, common_params["iterations"])
)
final_pool = Pool(X, y, cat_features=cat_idx)
test_pool = Pool(X_test, cat_features=cat_idx)

final_model = make_model(SEED + 999, constrained=True, iterations=final_iterations)
final_model.fit(final_pool)
test_pred = final_model.predict_proba(test_pool)[:, 1]
test_pred = np.clip(test_pred, 0, 1)

submission_target = (
    TARGET
    if TARGET in sample.columns
    else [c for c in sample.columns if c != ID_COL][0]
)
submission = sample.copy()
submission[submission_target] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000846"],
    "metric": "roc_auc",
    "cv_folds": 5,
    "unconstrained_baseline_cv_auc": float(baseline_cv_auc),
    "monotone_constrained_cv_auc": float(constrained_cv_auc),
    "final_model_iterations": int(final_iterations),
    "n_features": int(len(feature_cols)),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "oof_path": oof_path,
}
print(json.dumps(review, sort_keys=True))
