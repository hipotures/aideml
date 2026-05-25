import os
import re
import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 42
N_SPLITS = 5

train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
groups = train["Race"].astype(str).to_numpy()


def add_state_features(df):
    out = df.copy()
    comp = out["Compound"].astype(str).str.upper()
    race = out["Race"].astype(str)
    out["is_wet"] = comp.isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    out["monaco_like"] = race.str.contains(
        "Monaco|Singapore|Azerbaijan", case=False, regex=True
    ).astype(np.int8)
    out["race_phase"] = pd.cut(
        out["RaceProgress"].clip(0, 1.01),
        bins=[-0.01, 0.33, 0.67, 1.01],
        labels=[0, 1, 2],
    ).astype(np.int8)
    out["fresh_tyre"] = (out["TyreLife"] <= 3).astype(np.int8)
    out["old_tyre"] = (out["TyreLife"] >= 18).astype(np.int8)
    out["rule_debt"] = (
        (out["is_wet"] == 0) & (out["Stint"] <= 1) & (out["RaceProgress"] >= 0.45)
    ).astype(np.int8)
    return out


def make_regime(df):
    base = np.full(len(df), "dry", dtype=object)
    wet = df["is_wet"].to_numpy() == 1
    monaco = (df["monaco_like"].to_numpy() == 1) & (~wet)
    debt_old = (
        (df["rule_debt"].to_numpy() == 1)
        & (df["old_tyre"].to_numpy() == 1)
        & (~wet)
        & (~monaco)
    )
    fresh = (df["fresh_tyre"].to_numpy() == 1) & (~wet) & (~monaco) & (~debt_old)
    base[wet] = "wet"
    base[monaco] = "monaco_like"
    base[debt_old] = "debt_old"
    base[fresh] = "fresh"
    phase = df["race_phase"].astype(str).to_numpy()
    return np.array([f"{b}_p{p}" for b, p in zip(base, phase)], dtype=object)


train_fe = add_state_features(train)
test_fe = add_state_features(test)
train_regime = make_regime(train_fe)
test_regime = make_regime(test_fe)

feature_cols = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
cat_cols_raw = train_fe[feature_cols].select_dtypes(include=["object"]).columns.tolist()

for col in cat_cols_raw:
    cats = pd.Index(
        pd.concat([train_fe[col], test_fe[col]], ignore_index=True).astype(str).unique()
    )
    train_fe[col] = pd.Categorical(train_fe[col].astype(str), categories=cats)
    test_fe[col] = pd.Categorical(test_fe[col].astype(str), categories=cats)


def clean_name(c):
    return re.sub(r"[^A-Za-z0-9_]+", "_", c).strip("_")


rename = {c: clean_name(c) for c in feature_cols}
X = train_fe[feature_cols].rename(columns=rename)
X_test = test_fe[feature_cols].rename(columns=rename)
cat_cols = [rename[c] for c in cat_cols_raw]

try:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y, groups))
except Exception:
    splitter = GroupKFold(n_splits=N_SPLITS)
    folds = list(splitter.split(X, y, groups))

try:
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
except Exception as e:
    raise RuntimeError(
        "This script requires lightgbm, listed as an installed package."
    ) from e

pos = max(1, int(y.sum()))
neg = max(1, len(y) - pos)
scale_pos_weight = neg / pos

base_oof = np.zeros(len(train), dtype=np.float64)
test_base = np.zeros(len(test), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    model = LGBMClassifier(
        objective="binary",
        n_estimators=1600,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )
    base_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_base += model.predict_proba(X_test)[:, 1] / N_SPLITS
    fold_auc = roc_auc_score(y[va_idx], base_oof[va_idx])
    print(f"base_fold_{fold}_race_group_auc={fold_auc:.6f}")
    del model
    gc.collect()


class BetaCalibrator:
    def __init__(self, c=20.0):
        self.model = LogisticRegression(C=c, solver="lbfgs", max_iter=300)

    @staticmethod
    def _features(p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        return np.column_stack([np.log(p), np.log1p(-p)])

    def fit(self, p, y_true):
        self.model.fit(self._features(p), y_true)
        return self

    def predict(self, p):
        return self.model.predict_proba(self._features(p))[:, 1]


def fit_regime_calibrators(scores, y_true, regimes, min_count=1000, min_pos=20):
    global_cal = BetaCalibrator().fit(scores, y_true)
    calibrators = {}
    for r in np.unique(regimes):
        m = regimes == r
        n_pos = int(y_true[m].sum())
        n_neg = int(m.sum() - n_pos)
        if m.sum() >= min_count and n_pos >= min_pos and n_neg >= min_pos:
            calibrators[r] = BetaCalibrator().fit(scores[m], y_true[m])
    return global_cal, calibrators


def predict_regime_calibrators(scores, regimes, global_cal, calibrators):
    pred = global_cal.predict(scores)
    for r, cal in calibrators.items():
        m = regimes == r
        if np.any(m):
            pred[m] = cal.predict(scores[m])
    return np.clip(pred, 0, 1)


cal_oof = np.zeros(len(train), dtype=np.float64)
for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    global_cal, calibrators = fit_regime_calibrators(
        base_oof[tr_idx],
        y[tr_idx],
        train_regime[tr_idx],
    )
    cal_oof[va_idx] = predict_regime_calibrators(
        base_oof[va_idx],
        train_regime[va_idx],
        global_cal,
        calibrators,
    )

base_auc = roc_auc_score(y, base_oof)
cal_auc = roc_auc_score(y, cal_oof)

weights = np.linspace(0.0, 1.0, 21)
blend_aucs = []
for w in weights:
    blend_oof = (1.0 - w) * base_oof + w * cal_oof
    blend_aucs.append(roc_auc_score(y, blend_oof))

best_i = int(np.argmax(blend_aucs))
best_w = float(weights[best_i])
best_auc = float(blend_aucs[best_i])

if best_auc <= base_auc + 1e-7:
    best_w = 0.0
    final_auc = float(base_auc)
    final_oof = base_oof.copy()
else:
    final_auc = best_auc
    final_oof = (1.0 - best_w) * base_oof + best_w * cal_oof

global_cal, calibrators = fit_regime_calibrators(base_oof, y, train_regime)
test_cal = predict_regime_calibrators(test_base, test_regime, global_cal, calibrators)
test_pred = np.clip((1.0 - best_w) * test_base + best_w * test_cal, 0, 1)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(WORK / "submission.csv", index=False)
submission.to_csv(WORK / "test_predictions.csv.gz", index=False, compression="gzip")

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y,
        "prediction": final_oof,
    }
).to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

result = {
    "research_hypotheses_llm_claimed_used": ["000363"],
    "metric": "race_grouped_5fold_roc_auc",
    "base_auc": float(base_auc),
    "calibrated_auc": float(cal_auc),
    "selected_blend_weight": float(best_w),
    "final_auc": float(final_auc),
}
with open(WORK / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"base_race_grouped_5fold_roc_auc={base_auc:.6f}")
print(f"calibrated_race_grouped_5fold_roc_auc={cal_auc:.6f}")
print(f"selected_blend_weight={best_w:.2f}")
print(f"race_grouped_5fold_roc_auc={final_auc:.6f}")
print(json.dumps(result, sort_keys=True))
