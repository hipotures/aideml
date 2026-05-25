import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORKING = Path("./working")
WORKING.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 2026
N_SPLITS = 5
N_JOBS = max(1, min(8, os.cpu_count() or 1))


def clean_name(c):
    c = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
    return c or "col"


def clean_columns(df):
    out = df.copy()
    out.columns = [clean_name(c) for c in out.columns]
    return out


def add_features(df):
    df = df.copy()
    eps = 1e-6

    est_laps = df["LapNumber"] / df["RaceProgress"].clip(lower=eps)
    df["EstimatedRaceLaps"] = est_laps.clip(1, 120)
    df["LapsRemaining"] = (df["EstimatedRaceLaps"] - df["LapNumber"]).clip(0, 120)
    df["TyreLifeRaceFrac"] = df["TyreLife"] / df["EstimatedRaceLaps"].clip(lower=1)
    df["TyreLifeLapFrac"] = df["TyreLife"] / df["LapNumber"].clip(lower=1)
    df["DegradationPerTyreLap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(
        lower=1
    )
    df["LapDeltaPerTyreLap"] = df["LapTime_Delta"] / df["TyreLife"].clip(lower=1)
    df["AbsLapTimeDelta"] = df["LapTime_Delta"].abs()
    df["AbsPositionChange"] = df["Position_Change"].abs()
    df["Progress_x_TyreLife"] = df["RaceProgress"] * df["TyreLife"]
    df["Stint_x_TyreLife"] = df["Stint"] * df["TyreLife"]

    wet = df["Compound"].isin(["WET", "INTERMEDIATE"])
    dry = ~wet
    first = dry & (df["Stint"] <= 1)
    late = dry & (df["Stint"] > 1) & ((df["RaceProgress"] >= 0.65) | (df["Stint"] >= 3))

    df["IsWetCompound"] = wet.astype(np.int8)
    df["IsSoft"] = (df["Compound"] == "SOFT").astype(np.int8)
    df["IsMedium"] = (df["Compound"] == "MEDIUM").astype(np.int8)
    df["IsHard"] = (df["Compound"] == "HARD").astype(np.int8)
    df["FirstStint"] = (df["Stint"] <= 1).astype(np.int8)
    df["LateRace"] = (df["RaceProgress"] >= 0.65).astype(np.int8)
    df["RegimeCode"] = np.select([wet, first, late], [0, 1, 2], default=3).astype(
        np.int8
    )
    return df


def add_history_features(df):
    df = df.copy()
    df["_row_order"] = np.arange(len(df))
    ordered = df.sort_values(["Year", "Race", "Driver", "LapNumber", "id"])
    keys = ["Year", "Race", "Driver"]
    grp = ordered.groupby(keys, sort=False)

    lag_cols = [
        "LapTime_s",
        "LapTime_Delta",
        "Position",
        "Position_Change",
        "TyreLife",
        "Cumulative_Degradation",
        "PitStop",
    ]
    for col in lag_cols:
        ordered[f"Prev_{col}"] = grp[col].shift(1)
        if col != "PitStop":
            ordered[f"ChangeFromPrev_{col}"] = ordered[col] - ordered[f"Prev_{col}"]

    ordered["_LastPitLapTmp"] = ordered["LapNumber"].where(ordered["PitStop"].eq(1))
    ordered["LastPitLapObserved"] = grp["_LastPitLapTmp"].ffill()
    ordered["LapsSincePitObserved"] = (
        (ordered["LapNumber"] - ordered["LastPitLapObserved"])
        .fillna(ordered["LapNumber"])
        .clip(lower=0)
    )
    ordered["PitCountSoFar"] = grp["PitStop"].cumsum()

    ordered = ordered.sort_values("_row_order").drop(
        columns=["_row_order", "_LastPitLapTmp"]
    )
    return ordered


def pct_rank(a):
    a = np.asarray(a, dtype=float)
    if len(a) == 0:
        return a
    return pd.Series(a).rank(method="average").to_numpy(dtype=float) / (len(a) + 1.0)


def logit_clip(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def regime_weights(regime_code, available):
    base = np.array(
        [
            [0.75, 0.05, 0.05, 0.15],
            [0.03, 0.70, 0.07, 0.20],
            [0.03, 0.07, 0.70, 0.20],
            [0.05, 0.15, 0.15, 0.65],
        ],
        dtype=float,
    )
    w = base[np.asarray(regime_code, dtype=int)].copy()
    available = np.asarray(available, dtype=float)
    w *= available.reshape(1, -1)
    denom = w.sum(axis=1, keepdims=True)
    if available.sum() == 0:
        return np.full_like(w, 1.0 / w.shape[1])
    bad = denom[:, 0] <= 0
    if bad.any():
        w[bad] = available / available.sum()
        denom = w.sum(axis=1, keepdims=True)
    return w / denom


def blend_rank_matrix(rank_matrix, regime_code, available):
    w = regime_weights(regime_code, available)
    return (rank_matrix * w).sum(axis=1)


def make_model(y_train, seed):
    pos = float(np.sum(y_train))
    neg = float(len(y_train) - pos)
    spw = np.sqrt(neg / max(pos, 1.0))
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=70,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=5.0,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=N_JOBS,
        force_col_wise=True,
        verbosity=-1,
    )


def fit_predict_specialist(X_tr, y_tr, X_val_reg, y_val_reg, X_pred, cat_cols, seed):
    if len(y_tr) < 100 or np.unique(y_tr).size < 2:
        return np.full(len(X_pred), float(np.mean(y_tr)) if len(y_tr) else 0.01), False

    model = make_model(y_tr, seed)
    callbacks = [lgb.log_evaluation(period=0)]
    fit_kwargs = {"categorical_feature": cat_cols}

    if len(y_val_reg) >= 50 and np.unique(y_val_reg).size == 2:
        callbacks.append(lgb.early_stopping(stopping_rounds=60, verbose=False))
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val_reg, y_val_reg)],
            eval_metric="auc",
            callbacks=callbacks,
            **fit_kwargs,
        )
    else:
        model.fit(X_tr, y_tr, callbacks=callbacks, **fit_kwargs)

    return model.predict_proba(X_pred)[:, 1], True


train = clean_columns(pd.read_csv(INPUT / "train.csv.gz"))
test = clean_columns(pd.read_csv(INPUT / "test.csv.gz"))
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

target = train["PitNextLap"].astype(int).to_numpy()
train_x = train.drop(columns=["PitNextLap"])
n_train = len(train_x)

all_x = pd.concat([train_x, test], axis=0, ignore_index=True)
all_x = add_features(all_x)
all_x = add_history_features(all_x)

cat_cols = [c for c in ["Compound", "Driver", "Race"] if c in all_x.columns]
for c in cat_cols:
    all_x[c] = all_x[c].astype("category")

X = all_x.iloc[:n_train].reset_index(drop=True)
X_test = all_x.iloc[n_train:].reset_index(drop=True)
feature_cols = [c for c in X.columns if c != "id"]

regime_train = X["RegimeCode"].to_numpy()
regime_test = X_test["RegimeCode"].to_numpy()

specialists = [
    ("wet_intermediate", 0),
    ("dry_first_stint", 1),
    ("dry_late_later_stint", 2),
    ("dry_other", 3),
]

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_raw = np.zeros(n_train, dtype=float)
test_raw_folds = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, target), 1):
    X_tr_all = X.iloc[tr_idx][feature_cols]
    y_tr_all = target[tr_idx]
    X_val_all = X.iloc[val_idx][feature_cols]
    y_val_all = target[val_idx]

    val_rank_matrix = np.zeros((len(val_idx), len(specialists)), dtype=float)
    test_rank_matrix = np.zeros((len(X_test), len(specialists)), dtype=float)
    available = []

    for j, (name, code) in enumerate(specialists):
        tr_mask = regime_train[tr_idx] == code
        val_mask_same_regime = regime_train[val_idx] == code

        X_tr = X_tr_all.loc[tr_mask]
        y_tr = y_tr_all[tr_mask]
        X_val_reg = X_val_all.loc[val_mask_same_regime]
        y_val_reg = y_val_all[val_mask_same_regime]

        val_pred, ok = fit_predict_specialist(
            X_tr,
            y_tr,
            X_val_reg,
            y_val_reg,
            X_val_all,
            cat_cols,
            RANDOM_STATE + fold * 17 + j,
        )
        test_pred, _ = fit_predict_specialist(
            X_tr,
            y_tr,
            X_val_reg,
            y_val_reg,
            X_test[feature_cols],
            cat_cols,
            RANDOM_STATE + fold * 17 + j,
        )

        val_rank_matrix[:, j] = pct_rank(val_pred)
        test_rank_matrix[:, j] = pct_rank(test_pred)
        available.append(ok and np.nanstd(val_pred) > 1e-12)

    available = np.array(available, dtype=bool)
    fold_val = blend_rank_matrix(val_rank_matrix, regime_train[val_idx], available)
    fold_test = blend_rank_matrix(test_rank_matrix, regime_test, available)

    oof_raw[val_idx] = fold_val
    test_raw_folds.append(fold_test)

    fold_auc = roc_auc_score(y_val_all, fold_val)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

test_raw = np.mean(np.vstack(test_raw_folds), axis=0)
raw_auc = roc_auc_score(target, oof_raw)

calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
calibrator.fit(logit_clip(oof_raw), target)
oof_cal = calibrator.predict_proba(logit_clip(oof_raw))[:, 1]
cal_auc = roc_auc_score(target, oof_cal)

use_calibration = bool(cal_auc + 1e-12 >= raw_auc and calibrator.coef_[0, 0] > 0)
if use_calibration:
    oof_final = oof_cal
    test_final = calibrator.predict_proba(logit_clip(test_raw))[:, 1]
    final_auc = cal_auc
else:
    oof_final = oof_raw
    test_final = test_raw
    final_auc = raw_auc

print(f"OOF ROC AUC raw regime-rank blend: {raw_auc:.6f}")
print(f"OOF ROC AUC final: {final_auc:.6f} calibration_used={use_calibration}")

pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": target,
        "prediction": oof_final,
    }
).to_csv(WORKING / "oof_predictions.csv.gz", index=False, compression="gzip")

target_col = [c for c in sample.columns if c != "id"][0]
submission = sample.copy()
submission[target_col] = np.clip(test_final, 0, 1)
submission.to_csv(WORKING / "submission.csv", index=False)
submission.to_csv(WORKING / "test_predictions.csv.gz", index=False, compression="gzip")

print(
    json.dumps(
        {
            "roc_auc": float(final_auc),
            "raw_roc_auc": float(raw_auc),
            "platt_calibrated_roc_auc": float(cal_auc),
            "calibration_used": use_calibration,
            "research_hypotheses_llm_claimed_used": ["000999"],
        }
    )
)
