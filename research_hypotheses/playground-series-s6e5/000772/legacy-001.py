import os
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

INPUT_DIR = Path("./input")
WORKING_DIR = Path("./working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESIS_ID = "000772"

STATIC_CAUTION = {
    "Monaco Grand Prix": 0.20,
    "Azerbaijan Grand Prix": 0.22,
    "Singapore Grand Prix": 0.19,
    "Saudi Arabian Grand Prix": 0.18,
    "Australian Grand Prix": 0.17,
    "Canadian Grand Prix": 0.15,
    "Sao Paulo Grand Prix": 0.16,
    "Dutch Grand Prix": 0.14,
    "Miami Grand Prix": 0.13,
    "Italian Grand Prix": 0.12,
    "Mexico City Grand Prix": 0.12,
    "Las Vegas Grand Prix": 0.13,
    "Emilia Romagna Grand Prix": 0.13,
    "Japanese Grand Prix": 0.12,
    "British Grand Prix": 0.11,
    "United States Grand Prix": 0.10,
    "Spanish Grand Prix": 0.08,
    "Bahrain Grand Prix": 0.08,
    "Hungarian Grand Prix": 0.09,
    "Belgian Grand Prix": 0.10,
    "Austrian Grand Prix": 0.08,
    "Qatar Grand Prix": 0.10,
    "Abu Dhabi Grand Prix": 0.08,
    "Chinese Grand Prix": 0.10,
    "French Grand Prix": 0.08,
    "Pre-Season Testing": 0.03,
}

STATIC_PIT_LOSS = {
    "Monaco Grand Prix": 19.0,
    "Azerbaijan Grand Prix": 21.0,
    "Singapore Grand Prix": 28.0,
    "Saudi Arabian Grand Prix": 20.5,
    "Australian Grand Prix": 22.0,
    "Canadian Grand Prix": 19.5,
    "Sao Paulo Grand Prix": 22.0,
    "Dutch Grand Prix": 21.5,
    "Miami Grand Prix": 22.5,
    "Italian Grand Prix": 24.0,
    "Mexico City Grand Prix": 22.0,
    "Las Vegas Grand Prix": 21.5,
    "Emilia Romagna Grand Prix": 25.0,
    "Japanese Grand Prix": 22.0,
    "British Grand Prix": 21.0,
    "United States Grand Prix": 21.5,
    "Spanish Grand Prix": 23.0,
    "Bahrain Grand Prix": 22.5,
    "Hungarian Grand Prix": 21.5,
    "Belgian Grand Prix": 20.0,
    "Austrian Grand Prix": 20.0,
    "Qatar Grand Prix": 24.0,
    "Abu Dhabi Grand Prix": 22.0,
    "Chinese Grand Prix": 22.0,
    "French Grand Prix": 23.0,
    "Pre-Season Testing": 22.0,
}

STATIC_TYRE_LIFE = {
    "SOFT": 18.0,
    "MEDIUM": 28.0,
    "HARD": 38.0,
    "INTERMEDIATE": 24.0,
    "WET": 26.0,
}


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


class DiscountedPitLossEVFeatures:
    def fit(self, df):
        self.global_green_loss = 22.0
        self.global_discount = 0.58
        self.global_caution = 0.10
        self.green_loss_map = {}
        self.discount_map = {}
        self.caution_map = {}
        self.tyre_life_map = STATIC_TYRE_LIFE.copy()

        clean = df[(df["PitStop"] == 0) & df["LapTime (s)"].between(50, 350)]
        race_clean = clean.groupby("Race")["LapTime (s)"].median()
        global_clean = float(clean["LapTime (s)"].median())

        pit = df[(df["PitStop"] == 1) & df["LapTime (s)"].between(50, 350)].copy()
        if len(pit):
            pit["clean_ref"] = pit["Race"].map(race_clean).fillna(global_clean)
            pit["loss"] = (pit["LapTime (s)"] - pit["clean_ref"]).clip(5, 80)
            learned_green = pit.groupby("Race")["loss"].quantile(0.70)
            pit_counts = pit.groupby("Race")["loss"].size()
            self.global_green_loss = float(np.clip(pit["loss"].quantile(0.70), 12, 45))

            for race in set(df["Race"].astype(str)):
                static_loss = STATIC_PIT_LOSS.get(race, self.global_green_loss)
                if race in learned_green.index and pit_counts.get(race, 0) >= 5:
                    self.green_loss_map[race] = float(
                        np.clip(0.70 * learned_green[race] + 0.30 * static_loss, 10, 50)
                    )
                else:
                    self.green_loss_map[race] = float(static_loss)

            pit["green_prior"] = (
                pit["Race"].map(self.green_loss_map).fillna(self.global_green_loss)
            )
            pit["loss_ratio"] = (pit["loss"] / pit["green_prior"]).clip(0.20, 2.00)
            cheap = pit[pit["loss_ratio"] < 0.75]
            cheap_rate = (
                pit.assign(is_cheap=pit["loss_ratio"] < 0.75)
                .groupby("Race")["is_cheap"]
                .mean()
            )
            cheap_ratio = cheap.groupby("Race")["loss_ratio"].median()

            lap_slow = (
                clean.groupby(["Race", "Year", "LapNumber"])["LapTime_Delta"]
                .median()
                .reset_index()
            )
            lap_slow["slow_lap"] = lap_slow["LapTime_Delta"] > 12.0
            slow_rate = lap_slow.groupby("Race")["slow_lap"].mean()

            for race in set(df["Race"].astype(str)):
                static_c = STATIC_CAUTION.get(race, self.global_caution)
                cheap_c = float(cheap_rate.get(race, 0.0))
                slow_c = float(slow_rate.get(race, 0.0))
                caution = (
                    0.60 * static_c + 0.25 * cheap_c + 0.15 * min(0.35, slow_c * 3.0)
                )
                self.caution_map[race] = float(np.clip(caution, 0.02, 0.30))
                self.discount_map[race] = float(
                    np.clip(cheap_ratio.get(race, self.global_discount), 0.35, 0.85)
                )
        else:
            for race in set(df["Race"].astype(str)):
                self.green_loss_map[race] = float(
                    STATIC_PIT_LOSS.get(race, self.global_green_loss)
                )
                self.caution_map[race] = float(
                    STATIC_CAUTION.get(race, self.global_caution)
                )
                self.discount_map[race] = self.global_discount

        pit_life = df[(df["PitStop"] == 1) & df["TyreLife"].between(1, 90)]
        if len(pit_life):
            learned_life = pit_life.groupby("Compound")["TyreLife"].median()
            for comp, val in learned_life.items():
                base = STATIC_TYRE_LIFE.get(str(comp), float(val))
                self.tyre_life_map[str(comp)] = float(
                    np.clip(0.65 * val + 0.35 * base, 6, 80)
                )

        self.lap_delta_scale = float(np.nanpercentile(np.abs(df["LapTime_Delta"]), 95))
        self.deg_scale = float(
            np.nanpercentile(np.abs(df["Cumulative_Degradation"]), 95)
        )
        self.lap_delta_scale = max(self.lap_delta_scale, 1.0)
        self.deg_scale = max(self.deg_scale, 1.0)
        return self

    def transform(self, df):
        out = df.copy()
        race = out["Race"].astype(str)
        comp = out["Compound"].astype(str)

        green = (
            race.map(self.green_loss_map).fillna(self.global_green_loss).astype(float)
        )
        caution = race.map(self.caution_map).fillna(self.global_caution).astype(float)
        discount = (
            race.map(self.discount_map).fillna(self.global_discount).astype(float)
        )
        tyre_life_prior = comp.map(self.tyre_life_map).fillna(28.0).astype(float)

        expected_loss = green * (1.0 - caution * (1.0 - discount))
        pit_cost_relief = ((green - expected_loss) / green).clip(0, 0.60)

        progress = out["RaceProgress"].clip(0.01, 1.0).astype(float)
        total_laps_est = (out["LapNumber"].astype(float) / progress).clip(5, 100)
        remaining_laps = (total_laps_est - out["LapNumber"].astype(float)).clip(0, 100)
        tyre_ratio = (out["TyreLife"].astype(float) / tyre_life_prior).clip(0, 4)
        tyre_budget_left = (tyre_life_prior - out["TyreLife"].astype(float)).clip(
            -80, 80
        )
        run_to_finish_gap = remaining_laps - tyre_budget_left

        age_pressure = ((tyre_ratio - 0.55) / 0.55).clip(0, 1)
        finish_pressure = sigmoid(run_to_finish_gap / 4.0)
        service_pressure = (0.60 * age_pressure + 0.40 * finish_pressure).clip(0, 1)

        front_value = ((21.0 - out["Position"].astype(float)) / 20.0).clip(0.05, 1.0)
        midpack_value = (
            1.0 - (out["Position"].astype(float) - 10.5).abs() / 10.0
        ).clip(0, 1)
        position_value = 0.60 * front_value + 0.40 * midpack_value

        pos_lap_delta = (
            out["LapTime_Delta"].astype(float).clip(lower=0) / self.lap_delta_scale
        ).clip(0, 3)
        pos_deg = (
            out["Cumulative_Degradation"].astype(float).clip(lower=0) / self.deg_scale
        ).clip(0, 3)
        borderline = np.exp(-np.square((service_pressure - 0.55) / 0.25))

        out["green_pit_loss_prior"] = green
        out["scvsc_discount_factor_prior"] = discount
        out["caution_risk_prior"] = caution
        out["expected_pit_loss"] = expected_loss
        out["pit_loss_discount_seconds"] = green - expected_loss
        out["pit_cost_relief_ratio"] = pit_cost_relief
        out["compound_life_prior"] = tyre_life_prior
        out["tyre_life_ratio"] = tyre_ratio
        out["estimated_total_laps"] = total_laps_est
        out["estimated_remaining_laps"] = remaining_laps
        out["run_to_finish_gap"] = run_to_finish_gap
        out["service_window_pressure"] = service_pressure
        out["position_value_for_stop"] = position_value
        out["caution_adjusted_undercut_value"] = (
            service_pressure
            * position_value
            * (1.0 + pos_lap_delta + pos_deg)
            * (green / expected_loss).clip(0.5, 2.5)
        )
        out["wait_for_cheapest_stop_pressure"] = (
            borderline
            * caution
            * pit_cost_relief
            * (0.50 + progress)
            * (0.75 + position_value)
        )
        out["cheap_stop_next_lap_ev"] = (
            service_pressure * caution * pit_cost_relief * (0.50 + progress)
        )
        out["expected_loss_x_service_pressure"] = expected_loss * service_pressure
        out["caution_x_progress"] = caution * progress
        out["caution_x_position_value"] = caution * position_value
        out["green_loss_x_progress"] = green * progress
        out["service_minus_wait_pressure"] = (
            service_pressure - out["wait_for_cheapest_stop_pressure"]
        )
        out["expected_loss_per_position_value"] = expected_loss / (
            0.25 + position_value
        )
        return out


def prepare_categories(train_df, valid_df, test_df, cat_cols):
    for col in cat_cols:
        cats = pd.Index(
            pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0)
            .astype(str)
            .unique()
        )
        train_df[col] = pd.Categorical(train_df[col].astype(str), categories=cats)
        valid_df[col] = pd.Categorical(valid_df[col].astype(str), categories=cats)
        test_df[col] = pd.Categorical(test_df[col].astype(str), categories=cats)
    return train_df, valid_df, test_df


train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

y = train[TARGET].astype(int).values
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

feature_cols_base = [c for c in train.columns if c not in [TARGET, ID_COL]]
cat_cols = [c for c in ["Compound", "Driver", "Race", "Year"] if c in feature_cols_base]

if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=772)
    split_iter = splitter.split(train, y, groups)
else:
    splitter = GroupKFold(n_splits=5)
    split_iter = splitter.split(train, y, groups)

oof = np.zeros(len(train), dtype=np.float64)
test_pred = np.zeros(len(test), dtype=np.float64)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(split_iter, 1):
    tr_raw = train.iloc[tr_idx].reset_index(drop=True)
    va_raw = train.iloc[va_idx].reset_index(drop=True)

    fe = DiscountedPitLossEVFeatures().fit(tr_raw)
    tr_fe = fe.transform(tr_raw)
    va_fe = fe.transform(va_raw)
    te_fe = fe.transform(test)

    feature_cols = [c for c in tr_fe.columns if c not in [TARGET, ID_COL]]
    tr_x = tr_fe[feature_cols].copy()
    va_x = va_fe[feature_cols].copy()
    te_x = te_fe[feature_cols].copy()

    tr_x, va_x, te_x = prepare_categories(tr_x, va_x, te_x, cat_cols)

    pos = max(1, int(y[tr_idx].sum()))
    neg = max(1, len(tr_idx) - pos)
    scale_pos_weight = neg / pos

    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=2500,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=1000 + fold,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )

    model.fit(
        tr_x,
        y[tr_idx],
        eval_set=[(va_x, y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(120, verbose=False)],
    )

    va_pred = model.predict_proba(va_x)[:, 1]
    te_pred = model.predict_proba(te_x)[:, 1]

    oof[va_idx] = va_pred
    test_pred += te_pred / 5.0

    fold_auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold grouped CV ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 1e-6, 1 - 1e-6)
submission.to_csv(WORKING_DIR / "submission.csv", index=False)
submission.to_csv(
    WORKING_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(oof, 1e-6, 1 - 1e-6),
    }
).to_csv(WORKING_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

review = {
    "metric": "roc_auc",
    "cv_auc": float(cv_auc),
    "fold_auc": [float(v) for v in fold_scores],
    "research_hypotheses_llm_claimed_used": [HYPOTHESIS_ID],
    "submission_path": str(WORKING_DIR / "submission.csv"),
}
print(json.dumps(review, sort_keys=True))
