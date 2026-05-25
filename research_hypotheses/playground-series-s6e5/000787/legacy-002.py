import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

SEED = 2026
N_SPLITS = 5
N_THREADS = max(1, min(8, os.cpu_count() or 1))
TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
EPS = 1e-6


def safe_name(c):
    c = re.sub(r"[^0-9a-zA-Z_]+", "_", c).strip("_")
    return c or "feature"


def build_features(train, test):
    full = pd.concat(
        [train.drop(columns=[TARGET], errors="ignore"), test],
        axis=0,
        ignore_index=True,
    )
    full = full.drop(columns=[ID_COL], errors="ignore").copy()
    full.columns = [safe_name(c) for c in full.columns]

    for c in CAT_COLS:
        full[c] = full[c].astype(str).fillna("missing")

    rp = full["RaceProgress"].clip(EPS, 1.0)
    tyre = full["TyreLife"].clip(1.0)
    lap = full["LapNumber"].clip(1.0)

    full["WetCompoundFlag"] = (
        full["Compound"].str.upper().isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    full["LapsRemainingEst"] = lap * (1.0 - rp) / rp
    full["TyreLifeOverLap"] = full["TyreLife"] / lap
    full["DegradationPerTyreLife"] = full["Cumulative_Degradation"] / tyre
    full["AbsLapTimeDelta"] = full["LapTime_Delta"].abs()
    full["RacePhaseEarly"] = (full["RaceProgress"] < 0.34).astype("int8")
    full["RacePhaseLate"] = (full["RaceProgress"] >= 0.67).astype("int8")
    full["PositionXStint"] = full["Position"] * full["Stint"]
    full["TyreLifeXProgress"] = full["TyreLife"] * full["RaceProgress"]

    for c in CAT_COLS:
        full[c] = full[c].astype("category")

    return (
        full.iloc[: len(train)].reset_index(drop=True),
        full.iloc[len(train) :].reset_index(drop=True),
    )


def make_regimes(df):
    wet = (
        df["Compound"].astype(str).str.upper().isin(["WET", "INTERMEDIATE"]).to_numpy()
    )
    progress = df["RaceProgress"].to_numpy()
    phase = np.select(
        [progress < 0.34, progress < 0.67], ["early", "mid"], default="late"
    )
    prefix = np.where(wet, "wet_", "slick_")
    return (
        pd.Series(prefix, dtype="object")
        .str.cat(pd.Series(phase, dtype="object"))
        .to_numpy()
    )


def pct_rank_train(a):
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = (np.arange(len(a), dtype=np.float64) + 0.5) / len(a)
    return ranks


def pct_rank_test(train_scores, test_scores):
    s = np.sort(train_scores)
    return (np.searchsorted(s, test_scores, side="right") + 0.5) / (len(s) + 1.0)


def ece_score(y, p, n_bins=15):
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 0, 1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


class BetaCalibrator:
    def fit(self, scores, y, batch_scores=None):
        y = np.asarray(y).astype(int)
        self.constant = None
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
            self.model = None
            return self
        p = np.clip(np.asarray(scores), EPS, 1 - EPS)
        x = np.column_stack([np.log(p), np.log1p(-p)])
        self.model = LogisticRegression(C=100.0, max_iter=1000, solver="lbfgs")
        self.model.fit(x, y)
        return self

    def predict(self, scores):
        if self.constant is not None:
            return np.full(len(scores), self.constant, dtype=np.float64)
        p = np.clip(np.asarray(scores), EPS, 1 - EPS)
        x = np.column_stack([np.log(p), np.log1p(-p)])
        return self.model.predict_proba(x)[:, 1]


class IsoCalibrator:
    def fit(self, scores, y, batch_scores=None):
        y = np.asarray(y).astype(int)
        self.constant = None
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
            self.iso = None
            return self
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso.fit(np.asarray(scores), y)
        return self

    def predict(self, scores):
        if self.constant is not None:
            return np.full(len(scores), self.constant, dtype=np.float64)
        return self.iso.predict(np.asarray(scores))


class VennAbersCalibrator:
    def __init__(self, max_grid=256):
        self.max_grid = max_grid

    def fit(self, scores, y, batch_scores=None):
        scores = np.asarray(scores, dtype=np.float64)
        y = np.asarray(y).astype(int)
        self.constant = None
        if len(np.unique(y)) < 2:
            self.constant = float(y.mean())
            self.iso0 = self.iso1 = None
            return self

        ref = np.asarray(
            batch_scores if batch_scores is not None and len(batch_scores) else scores,
            dtype=np.float64,
        )
        qn = min(self.max_grid, max(2, len(ref)))
        grid = np.unique(np.quantile(ref, np.linspace(0, 1, qn)))
        if len(grid) == 0:
            grid = np.array([float(np.mean(scores))])

        s_aug = np.concatenate([scores, grid])
        w_aug = np.concatenate(
            [np.ones(len(scores)), np.full(len(grid), 1.0 / len(grid))]
        )

        self.iso0 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso1 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.iso0.fit(
            s_aug, np.concatenate([y, np.zeros(len(grid))]), sample_weight=w_aug
        )
        self.iso1.fit(
            s_aug, np.concatenate([y, np.ones(len(grid))]), sample_weight=w_aug
        )
        return self

    def predict(self, scores):
        if self.constant is not None:
            return np.full(len(scores), self.constant, dtype=np.float64)
        p0 = np.clip(self.iso0.predict(np.asarray(scores)), 0, 1)
        p1 = np.clip(self.iso1.predict(np.asarray(scores)), 0, 1)
        den = 1.0 - p0 + p1
        return np.clip(np.divide(p1, den, out=(p0 + p1) / 2.0, where=den > EPS), 0, 1)


def make_calibrator(name):
    if name == "beta":
        return BetaCalibrator()
    if name == "isotonic":
        return IsoCalibrator()
    if name == "venn_abers":
        return VennAbersCalibrator()
    raise ValueError(name)


def calibrator_cv_predictions(method, scores, y, seed):
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y).astype(int)
    min_class = np.bincount(y, minlength=2).min()
    n_splits = int(min(3, min_class))
    if n_splits < 2:
        return np.full(len(y), y.mean(), dtype=np.float64)

    preds = np.zeros(len(y), dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(scores.reshape(-1, 1), y):
        cal = make_calibrator(method)
        batch = scores[va] if method == "venn_abers" else None
        cal.fit(scores[tr], y[tr], batch_scores=batch)
        preds[va] = cal.predict(scores[va])
    return np.clip(preds, 0, 1)


def evaluate_calibrators(scores, y, seed):
    results = {}
    for method in ["beta", "isotonic", "venn_abers"]:
        pred = calibrator_cv_predictions(method, scores, y, seed)
        auc = roc_auc_score(y, pred) if len(np.unique(y)) == 2 else 0.5
        results[method] = {
            "pred": pred,
            "auc": float(auc),
            "ece": float(ece_score(y, pred)),
            "brier": float(brier_score_loss(y, pred)),
        }
    return (
        sorted(
            results, key=lambda m: (results[m]["auc"], -results[m]["ece"]), reverse=True
        )[0],
        results,
    )


def regime_calibrate(oof_score, test_score, y, train_regime, test_regime):
    best_global, global_results = evaluate_calibrators(oof_score, y, SEED + 100)
    global_oof = global_results[best_global]["pred"]

    global_cal = make_calibrator(best_global)
    global_cal.fit(
        oof_score, y, batch_scores=test_score if best_global == "venn_abers" else None
    )
    global_test = global_cal.predict(test_score)

    cal_oof = global_oof.copy()
    cal_test = global_test.copy()
    report = {
        "global": {
            "method": best_global,
            "auc": global_results[best_global]["auc"],
            "ece": global_results[best_global]["ece"],
            "brier": global_results[best_global]["brier"],
        },
        "regimes": {},
    }

    for r in sorted(set(train_regime) | set(test_regime)):
        tr_idx = np.where(train_regime == r)[0]
        te_idx = np.where(test_regime == r)[0]
        yy = y[tr_idx]
        if len(tr_idx) < 2000 or np.bincount(yy.astype(int), minlength=2).min() < 20:
            report["regimes"][r] = {
                "method": "global_fallback",
                "n_train": int(len(tr_idx)),
                "n_test": int(len(te_idx)),
            }
            continue

        best, results = evaluate_calibrators(
            oof_score[tr_idx], yy, SEED + abs(hash(r)) % 10000
        )
        cal_oof[tr_idx] = results[best]["pred"]

        if len(te_idx):
            cal = make_calibrator(best)
            batch = test_score[te_idx] if best == "venn_abers" else None
            cal.fit(oof_score[tr_idx], yy, batch_scores=batch)
            cal_test[te_idx] = cal.predict(test_score[te_idx])

        report["regimes"][r] = {
            "method": best,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "auc": results[best]["auc"],
            "ece": results[best]["ece"],
            "brier": results[best]["brier"],
        }

    return np.clip(cal_oof, 0, 1), np.clip(cal_test, 0, 1), report


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
X, X_test = build_features(train, test)
cat_features = [c for c in CAT_COLS if c in X.columns]

X_lgb = X.copy()
X_test_lgb = X_test.copy()

X_cb = X.copy()
X_test_cb = X_test.copy()
for c in cat_features:
    X_cb[c] = X_cb[c].astype(str)
    X_test_cb[c] = X_test_cb[c].astype(str)

base_oof = {}
base_test = {}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
pos_weight = float((len(y) - y.sum()) / max(1, y.sum()))

try:
    import lightgbm as lgb

    oof = np.zeros(len(train), dtype=np.float64)
    pred_test = np.zeros(len(test), dtype=np.float64)

    for fold, (tr, va) in enumerate(cv.split(X_lgb, y), 1):
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.035,
            num_leaves=63,
            min_child_samples=60,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=6.0,
            scale_pos_weight=pos_weight,
            random_state=SEED + fold,
            n_jobs=N_THREADS,
            verbosity=-1,
        )
        model.fit(
            X_lgb.iloc[tr],
            y[tr],
            eval_set=[(X_lgb.iloc[va], y[va])],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )
        oof[va] = model.predict_proba(X_lgb.iloc[va])[:, 1]
        pred_test += model.predict_proba(X_test_lgb)[:, 1] / N_SPLITS
        print(f"lightgbm fold {fold} auc={roc_auc_score(y[va], oof[va]):.6f}")

    base_oof["lightgbm"] = oof
    base_test["lightgbm"] = pred_test
except Exception as e:
    print(f"LightGBM expert skipped: {e}")

try:
    from catboost import CatBoostClassifier, Pool

    oof = np.zeros(len(train), dtype=np.float64)
    pred_test = np.zeros(len(test), dtype=np.float64)
    cb_cat_idx = [X_cb.columns.get_loc(c) for c in cat_features]

    for fold, (tr, va) in enumerate(cv.split(X_cb, y), 1):
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=900,
            learning_rate=0.045,
            depth=6,
            l2_leaf_reg=7.0,
            random_strength=0.7,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            scale_pos_weight=pos_weight,
            random_seed=SEED + 50 + fold,
            od_type="Iter",
            od_wait=80,
            allow_writing_files=False,
            thread_count=N_THREADS,
            verbose=False,
        )
        tr_pool = Pool(X_cb.iloc[tr], y[tr], cat_features=cb_cat_idx)
        va_pool = Pool(X_cb.iloc[va], y[va], cat_features=cb_cat_idx)
        te_pool = Pool(X_test_cb, cat_features=cb_cat_idx)
        model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
        oof[va] = model.predict_proba(va_pool)[:, 1]
        pred_test += model.predict_proba(te_pool)[:, 1] / N_SPLITS
        print(f"catboost fold {fold} auc={roc_auc_score(y[va], oof[va]):.6f}")

    base_oof["catboost"] = oof
    base_test["catboost"] = pred_test
except Exception as e:
    print(f"CatBoost expert skipped: {e}")

if len(base_oof) < 2:
    raise RuntimeError(
        "Hypothesis 000787 requires at least two diverse experts for the OOF blend."
    )

expert_names = list(base_oof)
oof_mat = np.column_stack([np.clip(base_oof[n], EPS, 1 - EPS) for n in expert_names])
test_mat = np.column_stack([np.clip(base_test[n], EPS, 1 - EPS) for n in expert_names])

rank_oof = np.column_stack(
    [pct_rank_train(oof_mat[:, i]) for i in range(oof_mat.shape[1])]
)
rank_test = np.column_stack(
    [pct_rank_test(oof_mat[:, i], test_mat[:, i]) for i in range(test_mat.shape[1])]
)

meta_X = np.column_stack([oof_mat, rank_oof, rank_oof.mean(axis=1)])
meta_test = np.column_stack([test_mat, rank_test, rank_test.mean(axis=1)])

stack_oof = np.zeros(len(train), dtype=np.float64)

for fold, (tr, va) in enumerate(cv.split(meta_X, y), 1):
    stacker = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    stacker.fit(meta_X[tr], y[tr])
    stack_oof[va] = stacker.predict_proba(meta_X[va])[:, 1]
    print(f"stack fold {fold} auc={roc_auc_score(y[va], stack_oof[va]):.6f}")

final_stacker = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
final_stacker.fit(meta_X, y)
stack_test = final_stacker.predict_proba(meta_test)[:, 1]

train_regime = make_regimes(train)
test_regime = make_regimes(test)
cal_oof, cal_test, calibration_report = regime_calibrate(
    stack_oof, stack_test, y, train_regime, test_regime
)

expert_aucs = {name: float(roc_auc_score(y, base_oof[name])) for name in expert_names}
stack_auc = float(roc_auc_score(y, stack_oof))
cal_auc = float(roc_auc_score(y, cal_oof))
cal_ece = float(ece_score(y, cal_oof))
cal_brier = float(brier_score_loss(y, cal_oof))

print("Expert OOF ROC AUCs:", json.dumps(expert_aucs, sort_keys=True))
print(f"Stack OOF ROC AUC: {stack_auc:.6f}")
print(f"Regime calibrated OOF ROC AUC: {cal_auc:.6f}")
print(f"Regime calibrated OOF ECE: {cal_ece:.6f}")
print(f"Regime calibrated OOF Brier: {cal_brier:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(cal_test, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

oof_df = pd.DataFrame(
    {
        "row": (
            train[ID_COL].to_numpy()
            if ID_COL in train.columns
            else np.arange(len(train))
        ),
        "target": y,
        "prediction": cal_oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": cal_auc,
    "stack_oof_roc_auc": stack_auc,
    "calibrated_oof_ece": cal_ece,
    "calibrated_oof_brier": cal_brier,
    "expert_oof_roc_aucs": expert_aucs,
    "research_hypotheses_llm_claimed_used": ["000787"],
    "calibration_report": calibration_report,
    "files": {
        "submission": os.path.join(WORK_DIR, "submission.csv"),
        "oof_predictions": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
        "test_predictions": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    },
}

for name in ["result.json", "review.json"]:
    with open(os.path.join(WORK_DIR, name), "w") as f:
        json.dump(result, f, indent=2)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": cal_auc,
            "research_hypotheses_llm_claimed_used": ["000787"],
        },
        indent=2,
    )
)
