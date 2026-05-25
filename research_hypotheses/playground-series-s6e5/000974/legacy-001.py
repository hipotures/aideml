import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

SEED = 42
N_FOLDS = 5
TARGET = "PitNextLap"
ID_COL = "id"
REGIMES = ["dry_early_mid", "dry_late", "wet"]


class ConstantModel:
    def __init__(self, p):
        self.p = float(np.clip(p, 1e-6, 1 - 1e-6))

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1 - p, p])


class BetaCalibrator:
    def __init__(self):
        self.model = None
        self.identity = False

    @staticmethod
    def _features(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    def fit(self, p, y):
        y = np.asarray(y, dtype=int)
        if len(y) < 30 or len(np.unique(y)) < 2:
            self.identity = True
            return self
        try:
            self.model = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
            self.model.fit(self._features(p), y)
        except Exception:
            self.identity = True
        return self

    def transform(self, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        if self.identity or self.model is None:
            return p
        return self.model.predict_proba(self._features(p))[:, 1]


def add_features(df):
    df = df.copy()
    compound = df["Compound"].astype(str).str.upper()
    wet = compound.isin(["INTERMEDIATE", "WET"]).astype(np.int8)

    est_laps = df["LapNumber"] / np.clip(df["RaceProgress"], 0.01, None)
    df["IsWetCompound"] = wet
    df["IsDryCompound"] = 1 - wet
    df["EstimatedRaceLaps"] = est_laps
    df["EstimatedLapsRemaining"] = est_laps - df["LapNumber"]
    df["TyreLifeRaceFrac"] = df["TyreLife"] / np.maximum(est_laps, 1)
    df["DegradationPerTyreLife"] = df["Cumulative_Degradation"] / np.maximum(
        df["TyreLife"], 1
    )
    df["AbsLapTimeDelta"] = np.abs(df["LapTime_Delta"])
    df["PositionLost"] = np.maximum(df["Position_Change"], 0)
    df["PositionGained"] = np.maximum(-df["Position_Change"], 0)
    df["TyreLife_x_Stint"] = df["TyreLife"] * df["Stint"]
    df["LateRace"] = (df["RaceProgress"] >= 0.65).astype(np.int8)
    return df.replace([np.inf, -np.inf], np.nan)


def make_regime(df):
    compound = df["Compound"].astype(str).str.upper()
    wet = compound.isin(["INTERMEDIATE", "WET"])
    late = df["RaceProgress"].values >= 0.65
    return np.where(wet, "wet", np.where(late, "dry_late", "dry_early_mid"))


def align_categories(train_df, test_df, cat_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in cat_cols:
        tr = train_df[c].astype(str).fillna("__NA__")
        te = test_df[c].astype(str).fillna("__NA__")
        cats = pd.Index(pd.concat([tr, te], axis=0).unique())
        train_df[c] = pd.Categorical(tr, categories=cats)
        test_df[c] = pd.Categorical(te, categories=cats)
    return train_df, test_df


def fit_lgbm(X_tr, y_tr, X_va=None, y_va=None, seed=SEED):
    y_tr = np.asarray(y_tr, dtype=int)
    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    if pos == 0 or neg == 0 or len(y_tr) < 50:
        return ConstantModel(pos / max(len(y_tr), 1))

    params = dict(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        scale_pos_weight=min(50.0, neg / max(pos, 1)),
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model = lgb.LGBMClassifier(**params)

    fit_kwargs = dict(categorical_feature=CAT_COLS)
    if X_va is not None and len(X_va) > 0 and len(np.unique(y_va)) == 2:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            **fit_kwargs,
        )
    else:
        model.fit(X_tr, y_tr, **fit_kwargs)
    return model


def predict_positive(model, X):
    return np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)


def crossfit_beta_calibrate(raw_oof, raw_test, y, fold_id, train_mask, test_mask):
    cal_oof = np.full_like(raw_oof, np.nan, dtype=float)

    for f in range(N_FOLDS):
        fit_mask = train_mask & (fold_id != f) & np.isfinite(raw_oof)
        val_mask = train_mask & (fold_id == f) & np.isfinite(raw_oof)
        if val_mask.any():
            calibrator = BetaCalibrator().fit(raw_oof[fit_mask], y[fit_mask])
            cal_oof[val_mask] = calibrator.transform(raw_oof[val_mask])

    full_mask = train_mask & np.isfinite(raw_oof)
    cal_test = np.full_like(raw_test, np.nan, dtype=float)
    if test_mask.any():
        calibrator = BetaCalibrator().fit(raw_oof[full_mask], y[full_mask])
        cal_test[test_mask] = calibrator.transform(raw_test[test_mask])

    return cal_oof, cal_test


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.rename(columns={"LapTime (s)": "LapTime_s"})
test = test.rename(columns={"LapTime (s)": "LapTime_s"})

y = train[TARGET].astype(int).values
train_regime = make_regime(train)
test_regime = make_regime(test)

train_fe = add_features(train.drop(columns=[TARGET]))
test_fe = add_features(test)

features = [c for c in train_fe.columns if c != ID_COL]
CAT_COLS = [c for c in features if train_fe[c].dtype == "object"] + ["Year"]
CAT_COLS = [c for c in CAT_COLS if c in features]

X = train_fe[features]
X_test = test_fe[features]
X, X_test = align_categories(X, X_test, CAT_COLS)

experts = ["global"] + REGIMES
raw_oof = {e: np.full(len(train), np.nan, dtype=float) for e in experts}
test_sum = {e: np.zeros(len(test), dtype=float) for e in experts}
test_count = {e: np.zeros(len(test), dtype=float) for e in experts}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_id = np.full(len(train), -1, dtype=int)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    fold_id[va_idx] = fold

    global_model = fit_lgbm(
        X.iloc[tr_idx],
        y[tr_idx],
        X.iloc[va_idx],
        y[va_idx],
        seed=SEED + fold,
    )
    raw_oof["global"][va_idx] = predict_positive(global_model, X.iloc[va_idx])
    test_sum["global"] += predict_positive(global_model, X_test)
    test_count["global"] += 1

    for regime in REGIMES:
        tr_reg = tr_idx[train_regime[tr_idx] == regime]
        va_reg = va_idx[train_regime[va_idx] == regime]
        te_mask = test_regime == regime

        if len(tr_reg) == 0:
            continue

        model = fit_lgbm(
            X.iloc[tr_reg],
            y[tr_reg],
            X.iloc[va_reg] if len(va_reg) else None,
            y[va_reg] if len(va_reg) else None,
            seed=SEED + 100 * (fold + 1) + REGIMES.index(regime),
        )

        if len(va_reg):
            raw_oof[regime][va_reg] = predict_positive(model, X.iloc[va_reg])
        if te_mask.any():
            test_sum[regime][te_mask] += predict_positive(model, X_test.loc[te_mask])
            test_count[regime][te_mask] += 1

raw_test = {}
for e in experts:
    raw_test[e] = np.divide(
        test_sum[e],
        np.maximum(test_count[e], 1),
        out=np.full(len(test), np.nan, dtype=float),
        where=test_count[e] > 0,
    )

cal_oof = {}
cal_test = {}

for e in experts:
    if e == "global":
        train_mask = np.ones(len(train), dtype=bool)
        test_mask = np.ones(len(test), dtype=bool)
    else:
        train_mask = train_regime == e
        test_mask = test_regime == e

    cal_oof[e], cal_test[e] = crossfit_beta_calibrate(
        raw_oof[e],
        raw_test[e],
        y,
        fold_id,
        train_mask,
        test_mask,
    )

global_oof = cal_oof["global"]
global_test = cal_test["global"]

regime_oof = np.full(len(train), np.nan, dtype=float)
regime_test = np.full(len(test), np.nan, dtype=float)

for regime in REGIMES:
    tr_mask = train_regime == regime
    te_mask = test_regime == regime
    regime_oof[tr_mask] = cal_oof[regime][tr_mask]
    regime_test[te_mask] = cal_test[regime][te_mask]

regime_oof = np.where(np.isfinite(regime_oof), regime_oof, global_oof)
regime_test = np.where(np.isfinite(regime_test), regime_test, global_test)

pred_oof = np.clip(0.35 * global_oof + 0.65 * regime_oof, 1e-6, 1 - 1e-6)
pred_test = np.clip(0.35 * global_test + 0.65 * regime_test, 1e-6, 1 - 1e-6)

auc = roc_auc_score(y, pred_oof)
brier = brier_score_loss(y, pred_oof)
global_auc = roc_auc_score(y, global_oof)
regime_auc = roc_auc_score(y, regime_oof)

target_col = [c for c in sample.columns if c != ID_COL][0]
submission = sample.copy()
submission[target_col] = pred_test
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {"row": np.arange(len(train)), "target": y, "prediction": pred_oof}
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(f"OOF ROC AUC (5-fold fold-safe calibrated MoE, hypothesis 000974): {auc:.6f}")
print(f"OOF Brier score: {brier:.6f}")
print(f"Global calibrated OOF ROC AUC: {global_auc:.6f}")
print(f"Regime expert calibrated OOF ROC AUC: {regime_auc:.6f}")
print(
    "RESULT_JSON:"
    + json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000974"],
            "validation_metric": "roc_auc",
            "validation_auc": float(auc),
            "brier_score": float(brier),
            "global_auc": float(global_auc),
            "regime_expert_auc": float(regime_auc),
            "n_folds": N_FOLDS,
        },
        sort_keys=True,
    )
)
