import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
REGIMES = ["dry-strategic", "wet_to_dry", "dry_to_wet_or_reactive"]
CAT_COLS = ["compound", "compound_family", "driver", "race"]
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
N_JOBS = max(1, os.cpu_count() or 1)


class ConstantHazard:
    def __init__(self, p):
        self.p = float(np.clip(p, 1e-7, 1 - 1e-7))
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = np.full(len(X), self.p, dtype=np.float32)
        return np.column_stack([1 - p, p])


def add_context(df):
    df = df.copy()
    df["Compound"] = df["Compound"].astype(str).str.upper()
    df["compound_family"] = np.where(df["Compound"].isin(WET_COMPOUNDS), "wet", "dry")
    df["is_wet_tyre"] = (df["compound_family"] == "wet").astype(np.int8)
    df["year_race"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)

    lap_group = ["Year", "Race", "LapNumber"]
    race_group = ["Year", "Race"]
    df["field_wet_frac"] = (
        df.groupby(lap_group)["is_wet_tyre"].transform("mean").astype(np.float32)
    )
    df["field_dry_frac"] = 1.0 - df["field_wet_frac"]
    df["field_pitstop_frac"] = (
        df.groupby(lap_group)["PitStop"].transform("mean").astype(np.float32)
    )
    df["race_wet_frac"] = (
        df.groupby(race_group)["is_wet_tyre"].transform("mean").astype(np.float32)
    )
    return df


def add_next_compound(df):
    df = df.copy()
    sort_cols = ["Year", "Race", "Driver", "LapNumber"]
    if ID_COL in df.columns:
        sort_cols.append(ID_COL)

    ordered = df.sort_values(sort_cols)
    g = ordered.groupby(["Year", "Race", "Driver"], sort=False)
    next_family = g["compound_family"].shift(-1)
    next_compound = g["Compound"].shift(-1)
    next_lap = g["LapNumber"].shift(-1)

    df["next_family"] = next_family.reindex(df.index)
    df["next_compound"] = next_compound.reindex(df.index)
    df["next_lap"] = next_lap.reindex(df.index)

    valid_next = df["next_lap"].notna() & (df["next_lap"] > df["LapNumber"])
    df.loc[~valid_next, "next_family"] = np.nan
    df.loc[~valid_next, "next_compound"] = np.nan
    df["next_family"] = df["next_family"].fillna(df["compound_family"])
    return df


def fit_thresholds(df):
    def abs_q(col, q):
        v = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .abs()
        )
        return float(v.quantile(q))

    lap_time = pd.to_numeric(df["LapTime (s)"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return {
        "lap_delta_abs": abs_q("LapTime_Delta", 0.97),
        "pos_change_abs": max(3.0, abs_q("Position_Change", 0.96)),
        "lap_time_hi": float(lap_time.quantile(0.99)),
    }


def infer_regimes(df, y, thresholds):
    wet_now = df["compound_family"].values == "wet"
    wet_next = df["next_family"].values == "wet"
    field_wet = df["field_wet_frac"].to_numpy(dtype=np.float32)

    lap_delta_abs = (
        pd.to_numeric(df["LapTime_Delta"], errors="coerce").abs().fillna(0).to_numpy()
    )
    pos_change_abs = (
        pd.to_numeric(df["Position_Change"], errors="coerce").abs().fillna(0).to_numpy()
    )
    lap_time = pd.to_numeric(df["LapTime (s)"], errors="coerce").fillna(0).to_numpy()

    reactive = (
        (lap_delta_abs >= thresholds["lap_delta_abs"])
        | (pos_change_abs >= thresholds["pos_change_abs"])
        | (lap_time >= thresholds["lap_time_hi"])
    )

    regime = np.zeros(len(df), dtype=np.int8)
    regime[(field_wet >= 0.05) | wet_now | reactive] = 2
    regime[wet_now & (field_wet <= 0.40)] = 1

    pos = np.asarray(y).astype(int) == 1
    wet_to_dry = pos & wet_now & (~wet_next)
    dry_to_wet_or_reactive = (
        pos
        & (~wet_to_dry)
        & (
            ((~wet_now) & wet_next)
            | (wet_now & wet_next)
            | (field_wet >= 0.05)
            | reactive
        )
    )

    regime[pos] = 0
    regime[dry_to_wet_or_reactive] = 2
    regime[wet_to_dry] = 1
    return regime


def build_features(df):
    lap = pd.to_numeric(df["LapNumber"], errors="coerce").astype(np.float32)
    race_progress = (
        pd.to_numeric(df["RaceProgress"], errors="coerce")
        .astype(np.float32)
        .clip(lower=1e-3)
    )
    tyre = pd.to_numeric(df["TyreLife"], errors="coerce").astype(np.float32)
    stint = pd.to_numeric(df["Stint"], errors="coerce").astype(np.float32)
    cum_deg = pd.to_numeric(df["Cumulative_Degradation"], errors="coerce").astype(
        np.float32
    )
    lap_delta = pd.to_numeric(df["LapTime_Delta"], errors="coerce").astype(np.float32)
    pos_change = pd.to_numeric(df["Position_Change"], errors="coerce").astype(
        np.float32
    )

    est_total_laps = (lap / race_progress).clip(lower=1, upper=120)
    laps_to_finish = (est_total_laps - lap).clip(lower=0, upper=120)
    tyre_safe = tyre.clip(lower=1.0)

    X = pd.DataFrame(index=df.index)
    X["year"] = pd.to_numeric(df["Year"], errors="coerce").astype(np.float32)
    X["lap_number"] = lap
    X["lap_time"] = pd.to_numeric(df["LapTime (s)"], errors="coerce").astype(np.float32)
    X["lap_time_delta"] = lap_delta
    X["abs_lap_time_delta"] = lap_delta.abs()
    X["pit_stop"] = pd.to_numeric(df["PitStop"], errors="coerce").astype(np.float32)
    X["position"] = pd.to_numeric(df["Position"], errors="coerce").astype(np.float32)
    X["position_change"] = pos_change
    X["abs_position_change"] = pos_change.abs()
    X["race_progress"] = race_progress
    X["stint"] = stint
    X["tyre_life"] = tyre
    X["log_tyre_life"] = np.log1p(tyre.clip(lower=0))
    X["cumulative_degradation"] = cum_deg
    X["degradation_per_lap"] = cum_deg / tyre_safe
    X["tyre_life_to_lap"] = tyre / lap.clip(lower=1)
    X["estimated_total_laps"] = est_total_laps
    X["laps_to_finish"] = laps_to_finish
    X["tyre_life_to_finish_ratio"] = tyre / (laps_to_finish + 1.0)
    X["field_wet_frac"] = df["field_wet_frac"].astype(np.float32)
    X["field_dry_frac"] = df["field_dry_frac"].astype(np.float32)
    X["field_pitstop_frac"] = df["field_pitstop_frac"].astype(np.float32)
    X["race_wet_frac"] = df["race_wet_frac"].astype(np.float32)
    X["is_wet_tyre"] = df["is_wet_tyre"].astype(np.float32)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    X["compound"] = df["Compound"].astype(str)
    X["compound_family"] = df["compound_family"].astype(str)
    X["driver"] = df["Driver"].astype(str)
    X["race"] = df["Race"].astype(str)
    return X


def align_categories(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    for col in CAT_COLS:
        cats = pd.Index(
            pd.concat([X_train[col], X_test[col]], ignore_index=True)
            .astype(str)
            .unique()
        )
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats)
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats)
    return X_train, X_test


def make_gate(seed):
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=140,
        learning_rate=0.06,
        num_leaves=15,
        max_depth=4,
        min_child_samples=300,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_col_wise=True,
    )


def make_hazard(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=420,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=90,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=4.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_col_wise=True,
    )


def gate_proba(model, X):
    raw = np.asarray(model.predict_proba(X))
    out = np.zeros((len(X), len(REGIMES)), dtype=np.float32)
    for j, cls in enumerate(model.classes_):
        cls = int(cls)
        if 0 <= cls < len(REGIMES):
            out[:, cls] = raw[:, j]
    row_sum = out.sum(axis=1)
    out[row_sum <= 0] = 1.0 / len(REGIMES)
    out /= out.sum(axis=1, keepdims=True)
    return out


def positive_proba(model, X):
    raw = np.asarray(model.predict_proba(X))
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return raw[:, classes.index(1)].astype(np.float32)
    return np.full(len(X), 1e-7, dtype=np.float32)


def train_hazard(X, y, seed):
    y = np.asarray(y).astype(int)
    prior = (y.sum() + 0.5) / (len(y) + 1.0)
    if len(y) < 100 or y.sum() < 5 or y.sum() == len(y):
        return ConstantHazard(prior)
    model = make_hazard(seed)
    model.fit(X, y, categorical_feature=CAT_COLS)
    return model


def mixture_predict(gate_model, hazard_models, X):
    gw = gate_proba(gate_model, X)
    hp = np.column_stack([positive_proba(m, X) for m in hazard_models]).astype(
        np.float32
    )
    return np.clip((gw * hp).sum(axis=1), 1e-7, 1 - 1e-7)


train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

train = add_next_compound(add_context(train))
test = add_context(test)

y = train[TARGET].astype(int).to_numpy()
thresholds = fit_thresholds(train)
regime = infer_regimes(train, y, thresholds)
groups = train["year_race"].to_numpy()

X_train = build_features(train)
X_test = build_features(test)
X_train, X_test = align_categories(X_train, X_test)

oof = np.zeros(len(train), dtype=np.float32)
fold_scores = []
gkf = GroupKFold(n_splits=5)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y, groups), 1):
    gate = make_gate(100 + fold)
    gate.fit(X_train.iloc[tr_idx], regime[tr_idx], categorical_feature=CAT_COLS)

    hazards = []
    for k, name in enumerate(REGIMES):
        sub_idx = tr_idx[regime[tr_idx] == k]
        hazards.append(
            train_hazard(X_train.iloc[sub_idx], y[sub_idx], 1000 + fold * 10 + k)
        )

    pred = mixture_predict(gate, hazards, X_train.iloc[va_idx])
    oof[va_idx] = pred

    if len(np.unique(y[va_idx])) == 2:
        auc = roc_auc_score(y[va_idx], pred)
        fold_scores.append(auc)
        print(f"Fold {fold} Year_Race-group ROC AUC: {auc:.6f}")
    else:
        print(f"Fold {fold} Year_Race-group ROC AUC: nan")

cv_auc = roc_auc_score(y, oof)
print(f"Overall grouped 5-fold ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y,
        "prediction": oof,
    }
).to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

gate_full = make_gate(999)
gate_full.fit(X_train, regime, categorical_feature=CAT_COLS)

hazards_full = []
for k, name in enumerate(REGIMES):
    sub_idx = np.flatnonzero(regime == k)
    hazards_full.append(train_hazard(X_train.iloc[sub_idx], y[sub_idx], 2000 + k))

test_pred = mixture_predict(gate_full, hazards_full, X_test)

target_col = [c for c in sample.columns if c != ID_COL][0]
pred_by_id = pd.Series(test_pred, index=test[ID_COL].values)
submission = sample.copy()
submission[target_col] = submission[ID_COL].map(pred_by_id).astype(float)
submission[target_col] = (
    submission[target_col].fillna(float(np.mean(test_pred))).clip(1e-7, 1 - 1e-7)
)

submission.to_csv(WORK / "submission.csv", index=False)
submission.to_csv(WORK / "test_predictions.csv.gz", index=False, compression="gzip")

print(
    json.dumps(
        {
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": [float(x) for x in fold_scores],
            "research_hypotheses_llm_claimed_used": ["000752"],
            "submission_path": str(WORK / "submission.csv"),
        }
    )
)
