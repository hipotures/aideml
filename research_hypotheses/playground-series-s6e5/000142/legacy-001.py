import os
import re
import gc
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
EPS = 1e-6
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def clean_name(s):
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(s)).strip("_")
    return s if s else "feature"


def safe_auc(y, p):
    return roc_auc_score(y, np.clip(p, EPS, 1 - EPS))


def add_features(df):
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].astype(str).fillna("__MISSING__")

    df["is_wet_compound"] = df["Compound"].isin(["WET", "INTERMEDIATE"]).astype(np.int8)
    df["dry_compound"] = df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(np.int8)
    df["lap_remaining_est"] = np.where(
        df["RaceProgress"] > 0,
        df["LapNumber"] / np.clip(df["RaceProgress"], 0.01, 1.0) - df["LapNumber"],
        np.nan,
    )
    df["tyre_x_progress"] = df["TyreLife"] * df["RaceProgress"]
    df["degradation_per_tyre_lap"] = df["Cumulative_Degradation"] / np.clip(
        df["TyreLife"], 1, None
    )
    df["driver_race"] = df["Year"].astype(str) + "_" + df["Race"] + "_" + df["Driver"]

    sort_cols = ["Year", "Race", "Driver", "LapNumber", "id"]
    ordered = df.sort_values(sort_cols).copy()
    g = ordered.groupby(["Year", "Race", "Driver"], sort=False)

    for col in [
        "LapTime_s",
        "LapTime_Delta",
        "Position",
        "TyreLife",
        "Cumulative_Degradation",
    ]:
        if col in ordered.columns:
            ordered[f"{col}_lag1"] = g[col].shift(1)
            ordered[f"{col}_delta_from_lag1"] = ordered[col] - ordered[f"{col}_lag1"]
            ordered[f"{col}_roll3_mean"] = g[col].transform(
                lambda s: s.shift(1).rolling(3, min_periods=1).mean()
            )

    ordered["prev_pitstop"] = g["PitStop"].shift(1).fillna(0)
    ordered["stint_lap_frac"] = ordered["TyreLife"] / np.clip(
        ordered["LapNumber"], 1, None
    )
    return ordered.sort_index()


class IdentityCalibrator:
    name = "identity"

    def fit(self, p, y, regime=None):
        self.prior_ = float(np.mean(y))
        return self

    def predict(self, p, regime=None):
        return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


class SigmoidCalibrator:
    name = "sigmoid"

    def fit(self, p, y, regime=None):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        self.fallback_ = IdentityCalibrator().fit(p, y)
        if len(np.unique(y)) < 2 or np.std(p) < 1e-12:
            self.model_ = None
            return self
        x = np.log(p / (1 - p)).reshape(-1, 1)
        self.model_ = LogisticRegression(C=100.0, max_iter=1000, solver="lbfgs")
        self.model_.fit(x, y)
        if self.model_.coef_[0, 0] <= 0:
            self.model_ = None
        return self

    def predict(self, p, regime=None):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        if self.model_ is None:
            return self.fallback_.predict(p)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        return np.clip(self.model_.predict_proba(x)[:, 1], EPS, 1 - EPS)


class BetaCalibrator:
    name = "beta"

    def fit(self, p, y, regime=None):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        self.fallback_ = SigmoidCalibrator().fit(p, y)
        if len(np.unique(y)) < 2 or np.std(p) < 1e-12:
            self.model_ = None
            return self
        x = np.column_stack([np.log(p), np.log1p(-p)])
        self.model_ = LogisticRegression(C=100.0, max_iter=1000, solver="lbfgs")
        self.model_.fit(x, y)
        a, b = self.model_.coef_[0]
        if not (a >= -1e-8 and b <= 1e-8):
            self.model_ = None
        return self

    def predict(self, p, regime=None):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        if self.model_ is None:
            return self.fallback_.predict(p)
        x = np.column_stack([np.log(p), np.log1p(-p)])
        return np.clip(self.model_.predict_proba(x)[:, 1], EPS, 1 - EPS)


class RegimeIsotonicCalibrator:
    name = "regime_isotonic"

    def __init__(self, min_n=2500):
        self.min_n = min_n

    def fit(self, p, y, regime):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        regime = np.asarray(regime)
        self.global_ = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        self.models_ = {}
        for r in np.unique(regime):
            m = regime == r
            if m.sum() >= self.min_n and len(np.unique(y[m])) == 2:
                self.models_[r] = IsotonicRegression(out_of_bounds="clip").fit(
                    p[m], y[m]
                )
        return self

    def predict(self, p, regime):
        p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
        regime = np.asarray(regime)
        out = self.global_.predict(p)
        for r, model in self.models_.items():
            m = regime == r
            if m.any():
                out[m] = model.predict(p[m])
        return np.clip(out, EPS, 1 - EPS)


def fit_best_calibrator(p, y, regime):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    raw_auc = safe_auc(y, p)
    candidates = [
        SigmoidCalibrator().fit(p, y),
        BetaCalibrator().fit(p, y),
        RegimeIsotonicCalibrator().fit(p, y, regime),
    ]

    best = candidates[0]
    best_pred = best.predict(p, regime)
    best_brier = brier_score_loss(y, best_pred)
    best_auc = safe_auc(y, best_pred)

    for cal in candidates[1:]:
        pred = cal.predict(p, regime)
        auc = safe_auc(y, pred)
        brier = brier_score_loss(y, pred)
        unique_count = np.unique(np.round(pred, 12)).size
        min_unique = min(100, max(10, int(len(pred) * 0.001)))
        if (
            auc >= max(raw_auc, best_auc) - 1e-5
            and unique_count >= min_unique
            and brier < best_brier
        ):
            best, best_pred, best_brier, best_auc = cal, pred, brier, auc
    return best


def make_regime(df):
    wet = df["Compound"].astype(str).isin(["WET", "INTERMEDIATE"]).astype(int).values
    late = (df["RaceProgress"].values >= 0.55).astype(int)
    return wet * 2 + late


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.rename(columns={c: clean_name(c) for c in train.columns})
test = test.rename(columns={c: clean_name(c) for c in test.columns})
sample = sample.rename(columns={c: clean_name(c) for c in sample.columns})

target_col = "PitNextLap"
id_col = "id"
y = train[target_col].astype(int).values

all_x = pd.concat(
    [train.drop(columns=[target_col]), test],
    axis=0,
    ignore_index=True,
)
all_x = add_features(all_x)
cat_cols = all_x.select_dtypes(include=["object"]).columns.tolist()

for c in cat_cols:
    all_x[c] = all_x[c].astype("category")

num_cols = [c for c in all_x.columns if c not in cat_cols and c != id_col]
feature_cols = [c for c in all_x.columns if c != id_col]

X = all_x.iloc[: len(train)][feature_cols].reset_index(drop=True)
X_test = all_x.iloc[len(train) :][feature_cols].reset_index(drop=True)
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
regime_train = make_regime(train)
regime_test = make_regime(test)

folds = list(
    StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
        X, y, groups
    )
)
model_names = ["catboost", "lightgbm", "sequence_hgb"]
base_oof = np.zeros((len(X), len(model_names)), dtype=np.float32)
base_test = np.zeros((len(X_test), len(model_names)), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    pos = max(y_tr.sum(), 1)
    neg = max(len(y_tr) - y_tr.sum(), 1)
    scale_pos_weight = np.sqrt(neg / pos)

    cb = CatBoostClassifier(
        iterations=650,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED + fold,
        auto_class_weights="SqrtBalanced",
        allow_writing_files=False,
        verbose=False,
        thread_count=max(1, os.cpu_count() or 1),
    )
    cb.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_va, y_va),
        use_best_model=True,
        verbose=False,
    )
    base_oof[va_idx, 0] = cb.predict_proba(X_va)[:, 1]
    base_test[:, 0] += cb.predict_proba(X_test)[:, 1] / N_SPLITS

    lgm = LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED + fold,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )
    lgm.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    base_oof[va_idx, 1] = lgm.predict_proba(X_va)[:, 1]
    base_test[:, 1] += lgm.predict_proba(X_test)[:, 1] / N_SPLITS

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    seq_model = Pipeline(
        steps=[
            ("pre", pre),
            (
                "hgb",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=0.045,
                    max_iter=450,
                    max_leaf_nodes=31,
                    min_samples_leaf=45,
                    l2_regularization=0.08,
                    early_stopping=True,
                    validation_fraction=0.12,
                    n_iter_no_change=25,
                    random_state=SEED + fold,
                ),
            ),
        ]
    )
    seq_model.fit(X_tr, y_tr)
    base_oof[va_idx, 2] = seq_model.predict_proba(X_va)[:, 1]
    base_test[:, 2] += seq_model.predict_proba(X_test)[:, 1] / N_SPLITS

    fold_auc = safe_auc(y_va, base_oof[va_idx].mean(axis=1))
    print(f"fold {fold} raw mean base ROC AUC: {fold_auc:.6f}")
    gc.collect()

base_aucs = {name: safe_auc(y, base_oof[:, i]) for i, name in enumerate(model_names)}
print("base OOF ROC AUC:", json.dumps(base_aucs, sort_keys=True))

meta_oof = np.zeros(len(X), dtype=np.float32)
meta_calibrator_names = []

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    z_tr = np.zeros((len(tr_idx), len(model_names)), dtype=np.float32)
    z_va = np.zeros((len(va_idx), len(model_names)), dtype=np.float32)

    chosen = []
    for m, name in enumerate(model_names):
        cal = fit_best_calibrator(base_oof[tr_idx, m], y[tr_idx], regime_train[tr_idx])
        z_tr[:, m] = cal.predict(base_oof[tr_idx, m], regime_train[tr_idx])
        z_va[:, m] = cal.predict(base_oof[va_idx, m], regime_train[va_idx])
        chosen.append(getattr(cal, "name", type(cal).__name__))

    stacker = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced"
    )
    stacker.fit(z_tr, y[tr_idx])
    meta_oof[va_idx] = stacker.predict_proba(z_va)[:, 1]
    meta_calibrator_names.append(chosen)
    print(
        f"meta fold {fold} calibrators: {chosen}; stacked ROC AUC: {safe_auc(y[va_idx], meta_oof[va_idx]):.6f}"
    )

cv_auc = safe_auc(y, meta_oof)
print(f"grouped 5-fold calibrated stacked OOF ROC AUC: {cv_auc:.6f}")

z_all = np.zeros((len(X), len(model_names)), dtype=np.float32)
z_test = np.zeros((len(X_test), len(model_names)), dtype=np.float32)
final_calibrators = []

for m, name in enumerate(model_names):
    cal = fit_best_calibrator(base_oof[:, m], y, regime_train)
    z_all[:, m] = cal.predict(base_oof[:, m], regime_train)
    z_test[:, m] = cal.predict(base_test[:, m], regime_test)
    final_calibrators.append(getattr(cal, "name", type(cal).__name__))

final_stacker = LogisticRegression(
    C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced"
)
final_stacker.fit(z_all, y)
test_pred = np.clip(final_stacker.predict_proba(z_test)[:, 1], EPS, 1 - EPS)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(meta_oof, EPS, 1 - EPS),
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

sub = sample.copy()
sub[target_col] = test_pred
sub[[id_col, target_col]].to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
sub[[id_col, target_col]].to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000142"],
    "metric": "grouped_5fold_oof_roc_auc",
    "cv_auc": float(cv_auc),
    "base_oof_auc": {k: float(v) for k, v in base_aucs.items()},
    "final_calibrators": dict(zip(model_names, final_calibrators)),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
}
print(json.dumps(result, indent=2, sort_keys=True))
