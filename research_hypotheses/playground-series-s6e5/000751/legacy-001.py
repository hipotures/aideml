import os
import gc
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

SEED = 20260524
N_SPLITS = 5
INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
N_THREADS = max(1, min(8, os.cpu_count() or 1))


class BetaCalibrator:
    def __init__(self, eps=1e-6, reg=1e-4):
        self.eps = eps
        self.reg = reg
        self.a_ = 1.0
        self.b_ = 1.0
        self.c_ = 0.0
        self.identity_ = False

    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=np.float64), self.eps, 1.0 - self.eps)
        y = np.asarray(y, dtype=np.float64)
        if np.unique(y).size < 2:
            self.identity_ = True
            return self

        lp = np.log(p)
        lq = -np.log1p(-p)

        def objective(theta):
            ua, ub, c = theta
            a = np.exp(ua)
            b = np.exp(ub)
            z = a * lp + b * lq + c
            loss = np.mean(np.logaddexp(0.0, z) - y * z)
            loss += 0.5 * self.reg * (ua * ua + ub * ub + 0.01 * c * c)

            pr = expit(z)
            g = pr - y
            grad = np.array(
                [
                    np.mean(g * a * lp) + self.reg * ua,
                    np.mean(g * b * lq) + self.reg * ub,
                    np.mean(g) + self.reg * 0.01 * c,
                ]
            )
            return loss, grad

        res = minimize(
            objective,
            x0=np.zeros(3, dtype=np.float64),
            jac=True,
            method="L-BFGS-B",
            bounds=[(-4.0, 4.0), (-4.0, 4.0), (-12.0, 12.0)],
            options={"maxiter": 200, "ftol": 1e-10},
        )
        theta = res.x if res.success else np.zeros(3, dtype=np.float64)
        self.a_ = float(np.exp(theta[0]))
        self.b_ = float(np.exp(theta[1]))
        self.c_ = float(theta[2])
        return self

    def transform(self, p):
        if self.identity_:
            return np.asarray(p, dtype=np.float64)
        p = np.clip(np.asarray(p, dtype=np.float64), self.eps, 1.0 - self.eps)
        z = self.a_ * np.log(p) + self.b_ * (-np.log1p(-p)) + self.c_
        return expit(z)


def safe_auc(y, p):
    if np.unique(y).size < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def add_features(df):
    out = df.copy()
    rp = out["RaceProgress"].clip(0.01, 1.0)
    est_total = out["LapNumber"] / rp
    est_total = np.maximum(est_total, out["LapNumber"])
    est_total = np.minimum(est_total, 100.0)

    out["EstimatedTotalLaps"] = est_total.astype("float32")
    out["LapsRemainingEst"] = np.maximum(est_total - out["LapNumber"], 0.0).astype(
        "float32"
    )
    out["TyreLifeFrac"] = (out["TyreLife"] / np.maximum(est_total, 1.0)).astype(
        "float32"
    )
    out["DegPerTyreLap"] = (
        out["Cumulative_Degradation"] / out["TyreLife"].clip(lower=1.0)
    ).astype("float32")
    out["LapTimeDeltaAbs"] = out["LapTime_Delta"].abs().astype("float32")
    out["WetFlag"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")
    out["DryFlag"] = (1 - out["WetFlag"]).astype("int8")
    return out


def make_folds(X, y, groups):
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=SEED
        )
        return list(splitter.split(X, y, groups))
    except Exception:
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=N_SPLITS)
        return list(splitter.split(X, y, groups))


def prepare_data():
    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
    sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

    train = add_features(train)
    test = add_features(test)

    y = train[TARGET].astype(int).to_numpy()
    groups = train["Year"].astype(str) + "__" + train["Race"].astype(str)

    train = train.drop(columns=[TARGET])
    feature_cols = [c for c in train.columns if c != ID_COL]
    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in cat_cols:
        tr = X[c].where(X[c].notna(), "__NA__").astype(str)
        te = X_test[c].where(X_test[c].notna(), "__NA__").astype(str)
        cats = pd.concat([tr, te], ignore_index=True).drop_duplicates().tolist()
        X[c] = pd.Categorical(tr, categories=cats)
        X_test[c] = pd.Categorical(te, categories=cats)

    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        med = X[c].median()
        X[c] = X[c].fillna(med).astype("float32")
        X_test[c] = X_test[c].fillna(med).astype("float32")

    return X, X_test, y, groups.to_numpy(), sample, cat_cols, num_cols


def class_weight_ratio(y):
    pos = float(np.sum(y))
    neg = float(len(y) - pos)
    return neg / max(pos, 1.0)


def fit_predict_lgb(X_tr, y_tr, X_va, X_te, cat_cols, num_cols, seed):
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=48,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=4.0,
        scale_pos_weight=class_weight_ratio(y_tr),
        random_state=seed,
        n_jobs=N_THREADS,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    va = model.predict_proba(X_va)[:, 1]
    te = model.predict_proba(X_te)[:, 1]
    return va, te


def make_ordinal_encoder():
    try:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float32,
        )
    except TypeError:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float32,
        )


def make_xgb_arrays(X_tr, X_va, X_te, cat_cols, num_cols):
    parts_tr = [X_tr[num_cols].to_numpy(dtype=np.float32, copy=False)]
    parts_va = [X_va[num_cols].to_numpy(dtype=np.float32, copy=False)]
    parts_te = [X_te[num_cols].to_numpy(dtype=np.float32, copy=False)]

    if cat_cols:
        enc = make_ordinal_encoder()
        parts_tr.append(enc.fit_transform(X_tr[cat_cols].astype(str)))
        parts_va.append(enc.transform(X_va[cat_cols].astype(str)))
        parts_te.append(enc.transform(X_te[cat_cols].astype(str)))

    return np.hstack(parts_tr), np.hstack(parts_va), np.hstack(parts_te)


def fit_predict_xgb(X_tr, y_tr, X_va, X_te, cat_cols, num_cols, seed):
    Xtr, Xva, Xte = make_xgb_arrays(X_tr, X_va, X_te, cat_cols, num_cols)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_bin=256,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=5.0,
        scale_pos_weight=class_weight_ratio(y_tr),
        random_state=seed,
        n_jobs=N_THREADS,
        verbosity=0,
    )
    model.fit(Xtr, y_tr, verbose=False)
    va = model.predict_proba(Xva)[:, 1]
    te = model.predict_proba(Xte)[:, 1]
    del Xtr, Xva, Xte, model
    gc.collect()
    return va, te


def catboost_frame(df, cat_cols):
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype(str)
    return out


def fit_predict_cat(X_tr, y_tr, X_va, X_te, cat_cols, num_cols, seed):
    Xtr = catboost_frame(X_tr, cat_cols)
    Xva = catboost_frame(X_va, cat_cols)
    Xte = catboost_frame(X_te, cat_cols)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=450,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=6.0,
        random_strength=0.5,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        auto_class_weights="Balanced",
        random_seed=seed,
        thread_count=N_THREADS,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(Xtr, y_tr, cat_features=cat_cols)
    va = model.predict_proba(Xva)[:, 1]
    te = model.predict_proba(Xte)[:, 1]
    del Xtr, Xva, Xte, model
    gc.collect()
    return va, te


def build_meta_features(preds, frame):
    preds = np.asarray(preds, dtype=np.float32)
    parts = [
        preds,
        preds.mean(axis=1, keepdims=True).astype("float32"),
        preds.std(axis=1, keepdims=True).astype("float32"),
        (preds.max(axis=1, keepdims=True) - preds.min(axis=1, keepdims=True)).astype(
            "float32"
        ),
    ]
    regime_cols = [c for c in ["WetFlag", "DryFlag"] if c in frame.columns]
    if regime_cols:
        parts.append(frame[regime_cols].to_numpy(dtype=np.float32, copy=False))
    return np.hstack(parts).astype("float32", copy=False)


def make_meta_model(seed):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.75,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def main():
    os.makedirs(WORKING_DIR, exist_ok=True)
    X, X_test, y, groups, sample, cat_cols, num_cols = prepare_data()
    folds = make_folds(X, y, groups)

    model_specs = [
        ("lgbm", fit_predict_lgb),
        ("xgb_hist", fit_predict_xgb),
        ("catboost", fit_predict_cat),
    ]
    model_names = [m[0] for m in model_specs]

    n, n_test = len(X), len(X_test)
    raw_oof = np.zeros((n, len(model_specs)), dtype=np.float32)
    raw_test = np.zeros((n_test, len(model_specs)), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        for m_idx, (name, fit_fn) in enumerate(model_specs):
            print(f"Training level-1 {name}, fold {fold}/{len(folds)}")
            va_pred, te_pred = fit_fn(
                X_tr, y_tr, X_va, X_test, cat_cols, num_cols, SEED + 101 * fold + m_idx
            )
            raw_oof[va_idx, m_idx] = va_pred.astype(np.float32)
            raw_test[:, m_idx] += te_pred.astype(np.float32) / len(folds)
            print(f"{name} fold {fold} raw ROC AUC: {safe_auc(y_va, va_pred):.6f}")
        gc.collect()

    print("Base raw OOF ROC AUCs:")
    base_auc_summary = {}
    for i, name in enumerate(model_names):
        auc = safe_auc(y, raw_oof[:, i])
        base_auc_summary[name] = auc
        print(f"  {name}: {auc:.6f}")

    meta_oof = np.zeros(n, dtype=np.float32)
    calibration_deltas = []

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        cal_tr = np.zeros((len(tr_idx), len(model_names)), dtype=np.float32)
        cal_va = np.zeros((len(va_idx), len(model_names)), dtype=np.float32)

        for m_idx, name in enumerate(model_names):
            calibrator = BetaCalibrator().fit(raw_oof[tr_idx, m_idx], y[tr_idx])
            tr_pred = calibrator.transform(raw_oof[tr_idx, m_idx])
            va_pred = calibrator.transform(raw_oof[va_idx, m_idx])

            raw_auc = safe_auc(y[va_idx], raw_oof[va_idx, m_idx])
            cal_auc = safe_auc(y[va_idx], va_pred)
            delta = cal_auc - raw_auc
            calibration_deltas.append(delta)

            if np.isfinite(delta) and delta < -1e-6:
                print(
                    f"Calibration AUC check failed for {name} fold {fold}; using raw scores."
                )
                tr_pred = raw_oof[tr_idx, m_idx]
                va_pred = raw_oof[va_idx, m_idx]

            cal_tr[:, m_idx] = tr_pred
            cal_va[:, m_idx] = va_pred

        F_tr = build_meta_features(cal_tr, X.iloc[tr_idx])
        F_va = build_meta_features(cal_va, X.iloc[va_idx])
        meta = make_meta_model(SEED + fold)
        meta.fit(F_tr, y[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(F_va)[:, 1].astype(np.float32)

        fold_auc = safe_auc(y[va_idx], meta_oof[va_idx])
        print(f"Meta fold {fold} ROC AUC: {fold_auc:.6f}")

    cv_auc = safe_auc(y, meta_oof)
    min_cal_delta = float(np.nanmin(calibration_deltas))
    print(f"Grouped 5-fold stacked ROC AUC: {cv_auc:.6f}")
    print(f"Minimum beta-calibration AUC delta: {min_cal_delta:+.3e}")

    final_cal_oof = np.zeros_like(raw_oof, dtype=np.float32)
    final_cal_test = np.zeros_like(raw_test, dtype=np.float32)

    for m_idx, name in enumerate(model_names):
        calibrator = BetaCalibrator().fit(raw_oof[:, m_idx], y)
        oof_pred = calibrator.transform(raw_oof[:, m_idx])
        test_pred = calibrator.transform(raw_test[:, m_idx])

        raw_auc = safe_auc(y, raw_oof[:, m_idx])
        cal_auc = safe_auc(y, oof_pred)
        if np.isfinite(cal_auc - raw_auc) and cal_auc + 1e-6 < raw_auc:
            print(f"Final calibration AUC check failed for {name}; using raw scores.")
            oof_pred = raw_oof[:, m_idx]
            test_pred = raw_test[:, m_idx]

        final_cal_oof[:, m_idx] = oof_pred
        final_cal_test[:, m_idx] = test_pred

    F_all = build_meta_features(final_cal_oof, X)
    F_test = build_meta_features(final_cal_test, X_test)
    final_meta = make_meta_model(SEED)
    final_meta.fit(F_all, y)
    test_prob = np.clip(final_meta.predict_proba(F_test)[:, 1], 0.0, 1.0)

    submission = sample.copy()
    submission[TARGET] = test_prob
    submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
    submission.to_csv(
        os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    oof_df = pd.DataFrame(
        {
            "row": np.arange(n, dtype=np.int64),
            "target": y.astype(int),
            "prediction": np.clip(meta_oof, 0.0, 1.0),
        }
    )
    oof_df.to_csv(
        os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    result = {
        "metric": "grouped_5fold_roc_auc",
        "score": cv_auc,
        "research_hypotheses_llm_claimed_used": ["000751"],
        "base_raw_oof_auc": base_auc_summary,
        "minimum_beta_calibration_auc_delta": min_cal_delta,
        "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        "oof_path": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
        "test_predictions_path": os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
