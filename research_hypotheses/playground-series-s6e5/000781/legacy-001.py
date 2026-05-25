import os
import re
import gc
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 2026
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()


def clean_name(name):
    name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if not name:
        name = "feature"
    if name[0].isdigit():
        name = "f_" + name
    return name


def make_unique(names):
    seen, out = {}, []
    for n in names:
        k = seen.get(n, 0)
        out.append(n if k == 0 else f"{n}_{k}")
        seen[n] = k + 1
    return out


def make_keys(df):
    keys = pd.DataFrame(index=df.index)
    keys["Year"] = df["Year"].astype(str)
    keys["Compound"] = df["Compound"].astype(str)
    keys["Race"] = df["Race"].astype(str)
    keys["Driver"] = df["Driver"].astype(str)
    keys["RaceYear"] = keys["Race"] + "|" + keys["Year"]
    keys["DriverCompound"] = keys["Driver"] + "|" + keys["Compound"]
    keys["DriverRaceYear"] = keys["Driver"] + "|" + keys["RaceYear"]
    return keys


def make_physical_features(df):
    num_cols = [
        "Year",
        "LapNumber",
        "LapTime (s)",
        "LapTime_Delta",
        "PitStop",
        "Position",
        "Position_Change",
        "RaceProgress",
        "Stint",
        "TyreLife",
        "Cumulative_Degradation",
    ]
    out = pd.DataFrame(index=df.index)
    for col in num_cols:
        out[clean_name(col)] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    lap = out["LapNumber"].clip(lower=1)
    progress = out["RaceProgress"].clip(lower=0.005)
    tyre = out["TyreLife"].clip(lower=1)
    est_total_laps = (lap / progress).clip(lower=1, upper=120)

    out["EstimatedTotalLaps"] = est_total_laps.astype("float32")
    out["EstimatedLapsRemaining"] = (
        (est_total_laps - lap).clip(lower=0, upper=120).astype("float32")
    )
    out["TyreLifeRaceFrac"] = (tyre / est_total_laps).clip(0, 2).astype("float32")
    out["TyreLifeLapFrac"] = (tyre / lap).clip(0, 5).astype("float32")
    out["DegradationPerTyreLap"] = (
        (out["Cumulative_Degradation"] / tyre)
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
    )
    out["AbsLapTimeDelta"] = out["LapTime_Delta"].abs().astype("float32")
    out["AbsPositionChange"] = out["Position_Change"].abs().astype("float32")
    out["IsFreshTyre"] = (out["TyreLife"] <= 2).astype("float32")
    out["IsLateRace"] = (out["RaceProgress"] >= 0.75).astype("float32")
    out["IsFirstStint"] = (out["Stint"] == 1).astype("float32")
    return out


def make_base_features(train_df, test_df):
    train_num = make_physical_features(train_df).reset_index(drop=True)
    test_num = make_physical_features(test_df).reset_index(drop=True)

    cat_cols = ["Compound", "Race", "Year"]
    combined = (
        pd.concat([train_df[cat_cols], test_df[cat_cols]], axis=0, ignore_index=True)
        .fillna("__NA__")
        .astype(str)
    )
    dummies = pd.get_dummies(combined, prefix=cat_cols, dtype=np.uint8)
    dummies.columns = make_unique([clean_name(c) for c in dummies.columns])

    train_cat = dummies.iloc[: len(train_df)].reset_index(drop=True)
    test_cat = dummies.iloc[len(train_df) :].reset_index(drop=True)

    train_base = pd.concat([train_num, train_cat], axis=1)
    test_base = pd.concat([test_num, test_cat], axis=1)
    train_base.columns = make_unique([clean_name(c) for c in train_base.columns])
    test_base.columns = train_base.columns

    train_base = (
        train_base.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    )
    test_base = test_base.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return train_base, test_base


TE_SPECS = [
    ("te_year_eb", "Year", None, 200.0),
    ("te_compound_eb", "Compound", None, 120.0),
    ("te_race_eb", "Race", None, 180.0),
    ("te_race_year_eb", "RaceYear", "te_race_eb", 120.0),
    ("te_driver_eb", "Driver", None, 160.0),
    ("te_driver_compound_eb", "DriverCompound", "te_driver_eb", 70.0),
    ("te_driver_race_year_eb", "DriverRaceYear", "te_race_year_eb", 70.0),
]
TE_COLS = [s[0] for s in TE_SPECS]


def fit_te_transform(fit_keys, y_fit, apply_keys):
    apply_index = apply_keys.index
    fit_keys = fit_keys.reset_index(drop=True)
    apply_keys = apply_keys.reset_index(drop=True)
    y_fit = np.asarray(y_fit, dtype=np.float32)

    global_mean = float(np.mean(y_fit))
    fit_enc = pd.DataFrame(index=fit_keys.index)
    apply_enc = pd.DataFrame(index=apply_keys.index)

    for out_col, key_col, parent_col, smoothing in TE_SPECS:
        if parent_col is None:
            fit_prior = pd.Series(global_mean, index=fit_keys.index, dtype="float32")
            apply_prior = pd.Series(
                global_mean, index=apply_keys.index, dtype="float32"
            )
        else:
            fit_prior = fit_enc[parent_col]
            apply_prior = apply_enc[parent_col]

        tmp = pd.DataFrame(
            {
                "key": fit_keys[key_col].to_numpy(),
                "target": y_fit,
                "prior": fit_prior.to_numpy(dtype=np.float32),
            }
        )
        stats = tmp.groupby("key", sort=False).agg(
            target_sum=("target", "sum"),
            count=("target", "size"),
            prior=("prior", "mean"),
        )
        mapping = (
            (stats["target_sum"] + smoothing * stats["prior"])
            / (stats["count"] + smoothing)
        ).astype("float32")

        fit_vals = fit_keys[key_col].map(mapping)
        apply_vals = apply_keys[key_col].map(mapping)
        fit_enc[out_col] = fit_vals.fillna(fit_prior).astype("float32").to_numpy()
        apply_enc[out_col] = apply_vals.fillna(apply_prior).astype("float32").to_numpy()

    apply_enc.index = apply_index
    return apply_enc


def safe_n_splits(y_values, requested):
    counts = np.bincount(np.asarray(y_values, dtype=int), minlength=2)
    return int(max(2, min(requested, counts.min())))


def crossfit_te(keys, y_values, n_splits, seed):
    keys = keys.reset_index(drop=True)
    y_values = np.asarray(y_values, dtype=int)
    out = pd.DataFrame(
        np.zeros((len(keys), len(TE_COLS)), dtype=np.float32),
        columns=TE_COLS,
    )
    cv = StratifiedKFold(
        n_splits=safe_n_splits(y_values, n_splits), shuffle=True, random_state=seed
    )
    for tr_idx, va_idx in cv.split(np.zeros(len(y_values)), y_values):
        enc = fit_te_transform(keys.iloc[tr_idx], y_values[tr_idx], keys.iloc[va_idx])
        out.iloc[va_idx, :] = enc.to_numpy(dtype=np.float32)
    return out


def make_model(seed, n_estimators):
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.035,
        n_estimators=int(n_estimators),
        num_leaves=31,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.5,
        random_state=seed,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )


train_keys = make_keys(train)
test_keys = make_keys(test)
base_train, base_test = make_base_features(train, test)

groups = train_keys["RaceYear"].to_numpy()
if StratifiedGroupKFold is not None:
    try:
        outer_cv = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=RANDOM_STATE
        )
        outer_splits = list(outer_cv.split(np.zeros(len(y)), y, groups))
        cv_name = "5-fold StratifiedGroupKFold grouped by RaceYear"
    except Exception:
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        outer_splits = list(outer_cv.split(np.zeros(len(y)), y))
        cv_name = "5-fold StratifiedKFold"
else:
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    outer_splits = list(outer_cv.split(np.zeros(len(y)), y))
    cv_name = "5-fold StratifiedKFold"

print(f"Evaluation: {cv_name}")

oof = np.zeros(len(train), dtype=np.float32)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(outer_splits, start=1):
    print(f"Fold {fold}: building leak-free hierarchical EB encodings")
    tr_te = crossfit_te(
        train_keys.iloc[tr_idx], y[tr_idx], n_splits=4, seed=RANDOM_STATE + fold
    )
    va_te = fit_te_transform(
        train_keys.iloc[tr_idx], y[tr_idx], train_keys.iloc[va_idx]
    )

    X_tr = pd.concat(
        [base_train.iloc[tr_idx].reset_index(drop=True), tr_te.reset_index(drop=True)],
        axis=1,
    )
    X_va = pd.concat(
        [base_train.iloc[va_idx].reset_index(drop=True), va_te.reset_index(drop=True)],
        axis=1,
    )

    model = make_model(RANDOM_STATE + fold, n_estimators=1400)
    model.fit(
        X_tr,
        y[tr_idx],
        eval_set=[(X_va, y[va_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    pred = model.predict_proba(X_va)[:, 1].astype(np.float32)
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))

    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None or best_iter <= 0:
        best_iter = model.get_params()["n_estimators"]
    best_iterations.append(int(best_iter))

    print(f"Fold {fold} ROC AUC: {auc:.6f} | best_iteration: {best_iter}")

    del tr_te, va_te, X_tr, X_va, model, pred
    gc.collect()

cv_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int32),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_n_estimators = max(100, int(np.mean(best_iterations)))
print(f"Training final single LightGBM model with {final_n_estimators} trees")

full_train_te = crossfit_te(train_keys, y, n_splits=5, seed=RANDOM_STATE + 777)
test_te = fit_te_transform(train_keys, y, test_keys)

X_full = pd.concat(
    [base_train.reset_index(drop=True), full_train_te.reset_index(drop=True)], axis=1
)
X_test = pd.concat(
    [base_test.reset_index(drop=True), test_te.reset_index(drop=True)], axis=1
)

final_model = make_model(RANDOM_STATE + 999, n_estimators=final_n_estimators)
final_model.fit(X_full, y)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

target_col = [c for c in sample.columns if c != ID_COL][0]
submission = sample.copy()
submission[target_col] = test_pred

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
    "research_hypotheses_llm_claimed_used": ["000781"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print(json.dumps(result, sort_keys=True))
