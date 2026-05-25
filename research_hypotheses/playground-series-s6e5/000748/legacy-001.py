import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 748
DRY = {"SOFT", "MEDIUM", "HARD"}
WET = {"INTERMEDIATE", "WET"}


def clean_columns(df):
    rename = {}
    used = set()
    for c in df.columns:
        nc = re.sub(r"[^0-9a-zA-Z_]+", "_", c).strip("_")
        if re.match(r"^[0-9]", nc):
            nc = "f_" + nc
        base, k = nc, 1
        while nc in used:
            k += 1
            nc = f"{base}_{k}"
        rename[c] = nc
        used.add(nc)
    return df.rename(columns=rename)


train = clean_columns(pd.read_csv(INPUT / "train.csv.gz"))
test = clean_columns(pd.read_csv(INPUT / "test.csv.gz"))
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")


def engineer_sequence(df, make_aux_target=False):
    df = df.copy()
    df["Compound"] = df["Compound"].astype(str)
    df["Race"] = df["Race"].astype(str)
    df["Driver"] = df["Driver"].astype(str)

    df["is_dry_compound"] = df["Compound"].isin(DRY).astype(np.int8)
    df["is_wet_compound"] = df["Compound"].isin(WET).astype(np.int8)
    df["race_is_monaco"] = (
        df["Race"].str.contains("Monaco", case=False, na=False).astype(np.int8)
    )
    df["race_is_testing"] = (
        df["Race"].str.contains("Testing", case=False, na=False).astype(np.int8)
    )

    progress = df["RaceProgress"].replace(0, np.nan).astype(float)
    total_laps = (df["LapNumber"].astype(float) / progress).replace(
        [np.inf, -np.inf], np.nan
    )
    total_laps = total_laps.fillna(df["LapNumber"]).clip(1, 120)
    df["estimated_total_laps"] = total_laps
    df["laps_remaining_est"] = (total_laps - df["LapNumber"]).clip(lower=0)
    df["tyre_life_frac_race"] = df["TyreLife"] / (total_laps + 1e-3)
    df["stint_life_to_remaining"] = df["TyreLife"] / (df["laps_remaining_est"] + 1.0)
    df["degradation_per_tyre_lap"] = df["Cumulative_Degradation"] / (
        df["TyreLife"] + 1.0
    )
    df["lap_time_delta_abs"] = df["LapTime_Delta"].abs()
    df["position_loss_flag"] = (df["Position_Change"] > 0).astype(np.int8)

    sort_cols = ["Year", "Race", "Driver", "LapNumber", ID_COL]
    group_cols = ["Year", "Race", "Driver"]
    df["__orig_order"] = np.arange(len(df))
    s = df.sort_values(sort_cols, kind="mergesort").copy()

    s["__dry_first_seen"] = (
        (~s.duplicated(group_cols + ["Compound"])) & (s["is_dry_compound"] == 1)
    ).astype(np.int8)

    grp = s.groupby(group_cols, sort=False, observed=True)
    s["dry_specs_seen_so_far"] = grp["__dry_first_seen"].cumsum().astype(np.int8)
    s["wet_seen_so_far"] = grp["is_wet_compound"].cummax().astype(np.int8)
    s["pit_count_so_far"] = grp["PitStop"].cumsum().astype(np.int16)
    s["current_dry_spec_seen_before"] = (
        (s["is_dry_compound"] == 1) & (s["__dry_first_seen"] == 0)
    ).astype(np.int8)

    only_one_dry = (s["dry_specs_seen_so_far"] <= 1).astype(np.int8)
    no_wet_yet = (s["wet_seen_so_far"] == 0).astype(np.int8)
    legal_base = (
        (s["is_dry_compound"] == 1)
        & (no_wet_yet == 1)
        & (only_one_dry == 1)
        & (s["race_is_monaco"] == 0)
        & (s["race_is_testing"] == 0)
        & (s["laps_remaining_est"] > 0.5)
    )

    s["only_dry_spec_so_far"] = only_one_dry
    s["legal_must_stop_guess"] = legal_base.astype(np.int8)
    s["legal_must_stop_stint1"] = (legal_base & (s["Stint"] <= 1)).astype(np.int8)
    s["legal_late_window_8"] = (legal_base & (s["laps_remaining_est"] <= 8)).astype(
        np.int8
    )
    s["legal_late_window_3"] = (legal_base & (s["laps_remaining_est"] <= 3)).astype(
        np.int8
    )
    s["legal_urgency"] = s["legal_must_stop_guess"] / (s["laps_remaining_est"] + 1.0)

    if make_aux_target:
        future_pits = grp["PitStop"].transform(
            lambda x: x.iloc[::-1].cumsum().iloc[::-1] - x
        )
        s["must_stop_before_finish_target"] = (future_pits > 0).astype(np.int8)

    s = s.sort_values("__orig_order", kind="mergesort").drop(
        columns=["__orig_order", "__dry_first_seen"]
    )
    return s.reset_index(drop=True)


train_fe = engineer_sequence(train, make_aux_target=True)

train_base = train.drop(columns=[TARGET]).copy()
train_base["__dataset"] = "train"
train_base["__row"] = np.arange(len(train_base))
test_base = test.copy()
test_base["__dataset"] = "test"
test_base["__row"] = np.arange(len(test_base))
combined = pd.concat([train_base, test_base], ignore_index=True)
combined_fe = engineer_sequence(combined, make_aux_target=False)
test_fe = (
    combined_fe[combined_fe["__dataset"] == "test"]
    .sort_values("__row", kind="mergesort")
    .drop(columns=["__dataset", "__row"])
    .reset_index(drop=True)
)

cat_cols = ["Compound", "Driver", "Race"]
for c in cat_cols:
    cats = pd.concat([train_fe[c], test_fe[c]], ignore_index=True).astype(str).unique()
    cats = sorted(cats)
    train_fe[c] = pd.Categorical(train_fe[c].astype(str), categories=cats)
    test_fe[c] = pd.Categorical(test_fe[c].astype(str), categories=cats)

drop_cols = {ID_COL, TARGET, "must_stop_before_finish_target", "__dataset", "__row"}
feature_cols = [c for c in train_fe.columns if c not in drop_cols]
feature_cols = [c for c in feature_cols if c in test_fe.columns]

y = train_fe[TARGET].astype(int).values
must_y = train_fe["must_stop_before_finish_target"].astype(int).values
groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)

if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(splitter.split(train_fe, y, groups))
else:
    splitter = GroupKFold(n_splits=5)
    folds = list(splitter.split(train_fe, y, groups))


class ConstantModel:
    def __init__(self, p):
        self.p = float(np.clip(p, 1e-6, 1 - 1e-6))

    def predict_proba(self, X):
        p = np.full(len(X), self.p, dtype=float)
        return np.column_stack([1.0 - p, p])


def fit_lgb(X_tr, y_tr, X_val=None, y_val=None, seed=RANDOM_STATE, n_estimators=900):
    y_tr = np.asarray(y_tr).astype(int)
    if len(np.unique(y_tr)) < 2:
        return ConstantModel(y_tr.mean() if len(y_tr) else 0.0)

    pos = max(1, int(y_tr.sum()))
    neg = max(1, len(y_tr) - pos)
    params = dict(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=70,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        scale_pos_weight=min(80.0, neg / pos),
        random_state=seed,
        n_jobs=max(1, (os.cpu_count() or 2) - 1),
        verbosity=-1,
    )
    model = LGBMClassifier(**params)
    fit_kwargs = dict(categorical_feature=[c for c in cat_cols if c in X_tr.columns])

    if X_val is not None and y_val is not None and len(np.unique(y_val)) == 2:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            **fit_kwargs,
        )
    else:
        model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(0)], **fit_kwargs)
    return model


stage1_oof = np.zeros(len(train_fe), dtype=float)
stage1_test_by_fold = []

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    model = fit_lgb(
        train_fe.iloc[tr_idx][feature_cols],
        must_y[tr_idx],
        train_fe.iloc[va_idx][feature_cols],
        must_y[va_idx],
        seed=RANDOM_STATE + fold,
        n_estimators=700,
    )
    stage1_oof[va_idx] = model.predict_proba(train_fe.iloc[va_idx][feature_cols])[:, 1]
    stage1_test_by_fold.append(model.predict_proba(test_fe[feature_cols])[:, 1])

stage1_auc = roc_auc_score(must_y, stage1_oof)

stage2_features = feature_cols + ["must_stop_before_finish_score"]
oof = np.zeros(len(train_fe), dtype=float)
test_pred_folds = []

train_stage = train_fe.copy()
train_stage["must_stop_before_finish_score"] = stage1_oof

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    test_stage = test_fe.copy()
    test_stage["must_stop_before_finish_score"] = stage1_test_by_fold[fold - 1]

    fold_val_pred = np.zeros(len(va_idx), dtype=float)
    fold_test_pred = np.zeros(len(test_fe), dtype=float)

    for dry_flag in [1, 0]:
        tr_mask = train_stage.iloc[tr_idx]["is_dry_compound"].values == dry_flag
        va_mask = train_stage.iloc[va_idx]["is_dry_compound"].values == dry_flag
        te_mask = test_stage["is_dry_compound"].values == dry_flag

        local_tr = tr_idx[tr_mask]
        local_va = va_idx[va_mask]

        if len(local_tr) == 0:
            p = y[tr_idx].mean()
            fold_val_pred[va_mask] = p
            fold_test_pred[te_mask] = p
            continue

        model = fit_lgb(
            train_stage.iloc[local_tr][stage2_features],
            y[local_tr],
            train_stage.iloc[local_va][stage2_features] if len(local_va) else None,
            y[local_va] if len(local_va) else None,
            seed=RANDOM_STATE + 100 + fold + dry_flag,
            n_estimators=900,
        )

        if len(local_va):
            fold_val_pred[va_mask] = model.predict_proba(
                train_stage.iloc[local_va][stage2_features]
            )[:, 1]
        if te_mask.any():
            fold_test_pred[te_mask] = model.predict_proba(
                test_stage.loc[te_mask, stage2_features]
            )[:, 1]

    oof[va_idx] = fold_val_pred
    test_pred_folds.append(fold_test_pred)

cv_auc = roc_auc_score(y, oof)
test_pred = np.clip(np.mean(test_pred_folds, axis=0), 1e-6, 1 - 1e-6)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(WORK / "submission.csv", index=False)
submission.to_csv(WORK / "test_predictions.csv.gz", index=False, compression="gzip")

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": np.clip(oof, 1e-6, 1 - 1e-6),
    }
).to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

print(f"Stage-1 must-stop-before-finish OOF ROC AUC: {stage1_auc:.6f}")
print(f"Stage-2 PitNextLap 5-fold OOF ROC AUC: {cv_auc:.6f}")
print(
    json.dumps(
        {
            "cv_roc_auc": float(cv_auc),
            "stage1_must_stop_roc_auc": float(stage1_auc),
            "research_hypotheses_llm_claimed_used": ["000748"],
            "submission_path": str(WORK / "submission.csv"),
        }
    )
)
