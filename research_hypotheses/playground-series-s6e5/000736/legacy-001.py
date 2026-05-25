import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

warnings.filterwarnings("ignore")

SEED = 2026
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
N_JOBS = max(1, min(8, os.cpu_count() or 1))

REGIME_NAMES = ["dry_slick", "wet_intermediate", "high_energy", "late_race"]
CATEGORICAL_COLS = ["Compound", "Driver", "Race", "Year_cat"]

BASE_NUMERIC_COLS = [
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
ENGINEERED_NUMERIC_COLS = [
    "is_slick",
    "is_wet_compound",
    "compound_hardness",
    "race_total_laps_est",
    "laps_remaining_est",
    "tyre_life_frac",
    "stint_to_lap_ratio",
    "positive_degradation_per_lap",
    "degradation_per_lap",
    "abs_degradation_per_lap",
    "energy_pressure",
    "lap_delta_abs",
    "lap_delta_clipped",
    "lap_time_clipped",
    "front_position_score",
    "race_progress_x_tyre_life",
    "late_life_pressure",
]
FEATURE_COLS = BASE_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS + CATEGORICAL_COLS

GATE_NUMERIC_COLS = [
    "RaceProgress",
    "LapNumber",
    "TyreLife",
    "Stint",
    "compound_hardness",
    "is_slick",
    "is_wet_compound",
    "laps_remaining_est",
    "race_total_laps_est",
    "positive_degradation_per_lap",
    "energy_pressure",
    "Position",
]
GATE_CATEGORICAL_COLS = ["Compound", "Race", "Year_cat"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def make_ohe():
    try:
        return OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def add_features(df):
    df = df.copy()

    compound_hardness = {
        "WET": 0.3,
        "INTERMEDIATE": 0.7,
        "SOFT": 1.0,
        "MEDIUM": 2.0,
        "HARD": 3.0,
    }

    lap = df["LapNumber"].clip(lower=1).astype(float)
    progress = df["RaceProgress"].clip(lower=0.01, upper=1.0).astype(float)
    tyre = df["TyreLife"].clip(lower=1).astype(float)
    total_laps = (lap / progress).clip(upper=120)
    total_laps = np.maximum(total_laps, lap)
    laps_remaining = np.maximum(total_laps - lap, 0)

    df["Year_cat"] = df["Year"].astype(str)
    df["is_slick"] = df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(np.int8)
    df["is_wet_compound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    df["compound_hardness"] = (
        df["Compound"].map(compound_hardness).fillna(2.0).astype(np.float32)
    )

    df["race_total_laps_est"] = total_laps.astype(np.float32)
    df["laps_remaining_est"] = laps_remaining.astype(np.float32)
    df["tyre_life_frac"] = (tyre / (lap + 1.0)).astype(np.float32)
    df["stint_to_lap_ratio"] = (df["Stint"].astype(float) / (lap + 1.0)).astype(
        np.float32
    )

    cum_deg = df["Cumulative_Degradation"].astype(float)
    pos_deg = np.maximum(cum_deg, 0.0)
    df["positive_degradation_per_lap"] = (pos_deg / tyre).astype(np.float32)
    df["degradation_per_lap"] = (cum_deg / tyre).astype(np.float32)
    df["abs_degradation_per_lap"] = (np.abs(cum_deg) / tyre).astype(np.float32)
    df["energy_pressure"] = (
        df["positive_degradation_per_lap"] * np.log1p(tyre)
    ).astype(np.float32)

    delta = df["LapTime_Delta"].astype(float)
    df["lap_delta_abs"] = np.abs(delta).astype(np.float32)
    df["lap_delta_clipped"] = delta.clip(-60, 60).astype(np.float32)
    df["lap_time_clipped"] = (
        df["LapTime (s)"].astype(float).clip(50, 300).astype(np.float32)
    )

    df["front_position_score"] = ((21.0 - df["Position"].astype(float)) / 20.0).astype(
        np.float32
    )
    df["race_progress_x_tyre_life"] = (progress * tyre).astype(np.float32)
    df["late_life_pressure"] = (tyre / (tyre + laps_remaining + 1.0)).astype(np.float32)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


class SoftRegimeGate:
    def __init__(self, seed=SEED):
        self.seed = seed
        self.energy_mid_ = None
        self.energy_scale_ = None
        self.pipe_ = None

    def _prior(self, X):
        wet = X["is_wet_compound"].to_numpy(dtype=float)
        slick = X["is_slick"].to_numpy(dtype=float)
        progress = X["RaceProgress"].to_numpy(dtype=float)
        laps_remaining = X["laps_remaining_est"].to_numpy(dtype=float)
        energy_raw = X["positive_degradation_per_lap"].to_numpy(dtype=float)

        energy = sigmoid((energy_raw - self.energy_mid_) / self.energy_scale_)
        late = np.maximum(
            sigmoid((progress - 0.76) / 0.055),
            sigmoid((8.5 - laps_remaining) / 2.5),
        )

        dry_score = 0.20 + 1.80 * slick * (1.0 - 0.55 * late) * (1.0 - 0.50 * energy)
        wet_score = 0.08 + 2.50 * wet
        energy_score = 0.10 + 1.80 * energy * (1.0 - 0.65 * wet)
        late_score = 0.10 + 2.00 * late

        scores = np.vstack([dry_score, wet_score, energy_score, late_score]).T
        scores = np.maximum(scores, 1e-6)
        return scores / scores.sum(axis=1, keepdims=True)

    def fit(self, X):
        energy_values = X["positive_degradation_per_lap"].to_numpy(dtype=float)
        self.energy_mid_ = float(np.nanquantile(energy_values, 0.70))
        spread = float(
            np.nanquantile(energy_values, 0.88) - np.nanquantile(energy_values, 0.52)
        )
        self.energy_scale_ = max(spread, 1e-3)

        prior = self._prior(X)
        labels = prior.argmax(axis=1)

        if len(np.unique(labels)) >= 2:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="median"), GATE_NUMERIC_COLS),
                    ("cat", make_ohe(), GATE_CATEGORICAL_COLS),
                ]
            )
            clf = LogisticRegression(
                C=0.8,
                max_iter=300,
                class_weight="balanced",
                solver="lbfgs",
                random_state=self.seed,
            )
            self.pipe_ = Pipeline([("pre", preprocessor), ("clf", clf)])
            self.pipe_.fit(X[GATE_NUMERIC_COLS + GATE_CATEGORICAL_COLS], labels)

        return self

    def regime_prior(self, X):
        return self._prior(X)

    def predict_proba(self, X):
        prior = self._prior(X)
        if self.pipe_ is None:
            return prior

        raw = self.pipe_.predict_proba(X[GATE_NUMERIC_COLS + GATE_CATEGORICAL_COLS])
        learned = np.zeros((len(X), len(REGIME_NAMES)), dtype=np.float32)
        for j, cls in enumerate(self.pipe_.named_steps["clf"].classes_):
            learned[:, int(cls)] = raw[:, j]

        probs = 0.72 * learned + 0.28 * prior
        probs = np.maximum(probs, 1e-6)
        return probs / probs.sum(axis=1, keepdims=True)


def make_lgbm(seed):
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=320,
        learning_rate=0.045,
        num_leaves=64,
        min_child_samples=90,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=3.0,
        cat_smooth=20.0,
        force_col_wise=True,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
    )


def fit_lgbm(model, X, y, sample_weight):
    try:
        model.fit(
            X[FEATURE_COLS],
            y,
            sample_weight=sample_weight,
            categorical_feature=CATEGORICAL_COLS,
            callbacks=[lgb.log_evaluation(period=0)],
        )
    except TypeError:
        model.fit(
            X[FEATURE_COLS],
            y,
            sample_weight=sample_weight,
            categorical_feature=CATEGORICAL_COLS,
        )
    return model


def fit_moe(X, y, seed):
    gate = SoftRegimeGate(seed=seed).fit(X)
    regime_weights = gate.regime_prior(X)

    experts = []
    for k, name in enumerate(REGIME_NAMES):
        weights = 0.35 + 2.0 * regime_weights[:, k]
        weights = weights / np.mean(weights)
        model = make_lgbm(seed + 101 * (k + 1))
        experts.append(fit_lgbm(model, X, y, weights))

    return gate, experts


def predict_moe(gate, experts, X):
    gate_probs = gate.predict_proba(X)
    expert_probs = np.column_stack(
        [model.predict_proba(X[FEATURE_COLS])[:, 1] for model in experts]
    )
    preds = np.sum(gate_probs * expert_probs, axis=1)
    return np.clip(preds, 1e-6, 1.0 - 1e-6)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
    sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

    y = train[TARGET].astype(int).to_numpy()
    n_train = len(train)

    combined = pd.concat(
        [train.drop(columns=[TARGET]), test],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    combined = add_features(combined)

    for col in CATEGORICAL_COLS:
        combined[col] = combined[col].astype(str).astype("category")

    for col in FEATURE_COLS:
        if col not in combined.columns:
            raise ValueError(f"Missing expected feature column: {col}")

    X = combined.iloc[:n_train].copy()
    X_test = combined.iloc[n_train:].copy()

    groups = X["Year_cat"].astype(str) + "_" + X["Race"].astype(str)
    if StratifiedGroupKFold is not None and groups.nunique() >= N_SPLITS:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=SEED
        )
        splits = splitter.split(X, y, groups)
        cv_name = "StratifiedGroupKFold"
    else:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = splitter.split(X, y)
        cv_name = "StratifiedKFold"

    oof = np.zeros(n_train, dtype=np.float32)
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(splits, start=1):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y[tr_idx], y[val_idx]

        gate, experts = fit_moe(X_tr, y_tr, SEED + fold)
        val_pred = predict_moe(gate, experts, X_val)
        oof[val_idx] = val_pred.astype(np.float32)

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(float(fold_auc))
        print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

    cv_auc = roc_auc_score(y, oof)
    print(f"{cv_name} 5-fold ROC AUC: {cv_auc:.6f}")

    oof_df = pd.DataFrame(
        {
            "row": np.arange(n_train),
            "target": y,
            "prediction": oof,
        }
    )
    oof_df.to_csv(
        os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    final_gate, final_experts = fit_moe(X, y, SEED + 999)
    test_pred = predict_moe(final_gate, final_experts, X_test)

    submission = sample.copy()
    submission[TARGET] = test_pred
    submission[[ID_COL, TARGET]].to_csv(
        os.path.join(WORK_DIR, "submission.csv"), index=False
    )
    submission[[ID_COL, TARGET]].to_csv(
        os.path.join(WORK_DIR, "test_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    result_review = {
        "research_hypotheses_llm_claimed_used": ["000736"],
        "metric": "roc_auc",
        "cv_strategy": cv_name,
        "cv_roc_auc": float(cv_auc),
        "fold_auc": fold_scores,
        "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    }
    with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
        json.dump(result_review, f, indent=2)

    print(json.dumps(result_review, sort_keys=True))


if __name__ == "__main__":
    main()
