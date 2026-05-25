import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

import lightgbm as lgb
from lightgbm import LGBMClassifier

INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
HORIZONS = [1, 2, 3, 5]
SEED = 2026

os.makedirs(WORK_DIR, exist_ok=True)


class BetaOrSigmoidCalibrator:
    def __init__(self):
        self.kind = None
        self.model = None
        self.constant = None

    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=int)

        if np.unique(y).size < 2:
            self.kind = "constant"
            self.constant = float(np.mean(y))
            return self

        xb = np.column_stack([np.log(p), np.log1p(-p)])
        beta = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
        beta.fit(xb, y)
        a, b = beta.coef_[0]

        if a >= -1e-8 and b <= 1e-8:
            self.kind = "beta"
            self.model = beta
        else:
            xs = np.log(p / (1 - p)).reshape(-1, 1)
            sig = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
            sig.fit(xs, y)
            self.kind = "sigmoid"
            self.model = sig

        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)

        if self.kind == "constant":
            return np.full(len(p), self.constant, dtype=float)
        if self.kind == "beta":
            x = np.column_stack([np.log(p), np.log1p(-p)])
        else:
            x = np.log(p / (1 - p)).reshape(-1, 1)

        return self.model.predict_proba(x)[:, 1]


def make_horizon_targets(df):
    labels = {h: np.zeros(len(df), dtype=np.uint8) for h in HORIZONS}
    ordered = df.sort_values(
        ["Year", "Race", "Driver", "LapNumber", ID_COL], kind="mergesort"
    )

    for _, g in ordered.groupby(["Year", "Race", "Driver"], sort=False):
        idx = g.index.to_numpy()
        laps = g["LapNumber"].to_numpy(dtype=float)
        pit_laps = laps[g[TARGET].to_numpy(dtype=int) == 1] + 1.0

        if len(pit_laps) == 0:
            continue

        pos = np.searchsorted(pit_laps, laps, side="right")
        next_pit = np.full(len(g), np.inf, dtype=float)
        ok = pos < len(pit_laps)
        next_pit[ok] = pit_laps[pos[ok]]
        dist = next_pit - laps

        for h in HORIZONS:
            labels[h][idx] = ((dist > 0) & (dist <= h)).astype(np.uint8)

    return labels


def add_features(df):
    out = df.copy()

    lap = out["LapNumber"].astype(float)
    tyre = out["TyreLife"].astype(float).clip(lower=1)
    progress = out["RaceProgress"].astype(float).clip(lower=0.01)

    est_total = (lap / progress).clip(lower=1, upper=120)
    laps_left = (est_total - lap).clip(lower=0, upper=120)

    out["EstimatedTotalLaps"] = est_total
    out["LapsToFinish"] = laps_left
    out["TyreLifeToFinishRatio"] = tyre / (laps_left + 1.0)
    out["TyreLifeProgress"] = tyre * progress
    out["StintTyreLife"] = out["Stint"].astype(float) * tyre
    out["DegPerTyreLap"] = out["Cumulative_Degradation"].astype(float) / tyre
    out["LapDeltaPerTyreLap"] = out["LapTime_Delta"].astype(float) / tyre
    out["AbsLapTimeDelta"] = out["LapTime_Delta"].astype(float).abs()
    out["PositionLossFlag"] = (out["Position_Change"].astype(float) > 0).astype("int8")
    out["PositionGainFlag"] = (out["Position_Change"].astype(float) < 0).astype("int8")
    out["LateRaceOldTyre"] = progress * tyre
    out["TyreLifeSq"] = tyre**2
    out["RaceProgressSq"] = progress**2
    out["CompoundIsWet"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")
    out["CompoundIsSoft"] = (out["Compound"] == "SOFT").astype("int8")
    out["RaceYear"] = out["Year"].astype(str) + "_" + out["Race"].astype(str)
    out["DriverYear"] = out["Driver"].astype(str) + "_" + out["Year"].astype(str)
    out["DriverRace"] = out["Driver"].astype(str) + "_" + out["Race"].astype(str)

    return out


def horizon_to_hazard(preds):
    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    mono = np.maximum.accumulate(preds, axis=1)
    per_lap = 1.0 - np.power(1.0 - mono, 1.0 / np.asarray(HORIZONS, dtype=float))
    weights = np.array([0.55, 0.20, 0.15, 0.10], dtype=float)
    hazard = per_lap @ weights
    return np.clip(hazard, 1e-6, 1 - 1e-6)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz")).reset_index(drop=True)
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz")).reset_index(drop=True)
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
horizon_targets = make_horizon_targets(train)

train_fe = add_features(train)
test_fe = add_features(test)

drop_cols = {ID_COL, TARGET}
features = [c for c in train_fe.columns if c not in drop_cols]
cat_cols = ["Compound", "Driver", "Race", "RaceYear", "DriverYear", "DriverRace"]
cat_cols = [c for c in cat_cols if c in features]

combined = pd.concat([train_fe[features], test_fe[features]], axis=0, ignore_index=True)
num_cols = [c for c in combined.columns if c not in cat_cols]

for c in cat_cols:
    combined[c] = combined[c].astype("string").fillna("missing").astype("category")

for c in num_cols:
    combined[c] = pd.to_numeric(combined[c], errors="coerce")
    combined[c] = (
        combined[c].replace([np.inf, -np.inf], np.nan).fillna(-999.0).astype("float32")
    )

X = combined.iloc[: len(train)].copy()
X_test = combined.iloc[len(train) :].copy()

groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y, groups=groups))
else:
    splitter = GroupKFold(n_splits=5)
    folds = list(splitter.split(X, y, groups=groups))

oof_horizon = np.zeros((len(train), len(HORIZONS)), dtype=np.float32)
test_horizon = np.zeros((len(test), len(HORIZONS)), dtype=np.float32)

threads = max(1, min(8, os.cpu_count() or 1))

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    print(f"Fold {fold}/{len(folds)}")
    X_tr = X.iloc[tr_idx]
    X_va = X.iloc[va_idx]

    for j, horizon in enumerate(HORIZONS):
        yh = horizon_targets[horizon]
        y_tr = yh[tr_idx]
        y_va = yh[va_idx]

        if np.unique(y_tr).size < 2:
            pred = np.full(len(va_idx), float(np.mean(y_tr)), dtype=float)
            test_pred = np.full(len(test), float(np.mean(y_tr)), dtype=float)
        else:
            pos = max(1.0, float(y_tr.sum()))
            neg = max(1.0, float(len(y_tr) - y_tr.sum()))
            model = LGBMClassifier(
                objective="binary",
                n_estimators=1200,
                learning_rate=0.04,
                num_leaves=63,
                min_child_samples=100,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=3.0,
                scale_pos_weight=float(np.sqrt(neg / pos)),
                random_state=SEED + 97 * fold + horizon,
                n_jobs=threads,
                force_col_wise=True,
                verbosity=-1,
            )
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                categorical_feature=cat_cols,
                callbacks=[
                    lgb.early_stopping(80, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
            pred = model.predict_proba(X_va)[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]

        oof_horizon[va_idx, j] = pred
        test_horizon[:, j] += test_pred / len(folds)

raw_oof = horizon_to_hazard(oof_horizon)
raw_test = horizon_to_hazard(test_horizon)
raw_auc = roc_auc_score(y, raw_oof)

cal_oof = np.zeros(len(train), dtype=float)
cal_kinds = []

for tr_idx, va_idx in folds:
    cal = BetaOrSigmoidCalibrator().fit(raw_oof[tr_idx], y[tr_idx])
    cal_oof[va_idx] = cal.predict(raw_oof[va_idx])
    cal_kinds.append(cal.kind)

cal_auc = roc_auc_score(y, cal_oof)

final_cal = BetaOrSigmoidCalibrator().fit(raw_oof, y)
test_pred = final_cal.predict(raw_test)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

pred_col = TARGET if TARGET in sample.columns else sample.columns[-1]
submission = sample[[ID_COL]].copy()
submission[pred_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=int),
        "target": y.astype(int),
        "prediction": cal_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(f"Raw monotone hazard OOF ROC AUC: {raw_auc:.6f}")
print(f"Cross-fitted calibrated OOF ROC AUC: {cal_auc:.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000746"],
            "metric": "roc_auc",
            "raw_oof_roc_auc": float(raw_auc),
            "calibrated_oof_roc_auc": float(cal_auc),
            "fold_calibration_kinds": cal_kinds,
            "final_calibration_kind": final_cal.kind,
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
            "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
            "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
        }
    )
)
