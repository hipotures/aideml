import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

RANDOM_STATE = 431
N_JOBS = min(12, os.cpu_count() or 1)
INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

try:
    import lightgbm as lgb
except ImportError as e:
    raise RuntimeError(
        "This script requires lightgbm, which is listed as available for this task."
    ) from e


def read_data():
    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
    sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))
    return train, test, sample


def prepare_features(train, test):
    target = train["PitNextLap"].astype(int).values
    feature_cols = [c for c in train.columns if c not in ["id", "PitNextLap"]]
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    cat_cols = [
        c
        for c in feature_cols
        if X_train[c].dtype == "object" or X_test[c].dtype == "object"
    ]
    for col in cat_cols:
        cats = pd.Index(
            pd.concat([X_train[col], X_test[col]], ignore_index=True)
            .astype(str)
            .unique()
        )
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats)
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats)

    for col in feature_cols:
        if col not in cat_cols:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

    return X_train, X_test, target, feature_cols, cat_cols


def fit_lgbm(model, X_tr, y_tr, cat_cols, sample_weight=None, X_val=None, y_val=None):
    kwargs = {
        "sample_weight": sample_weight,
        "categorical_feature": cat_cols,
    }
    if X_val is not None:
        kwargs.update(
            {
                "eval_set": [(X_val, y_val)],
                "eval_metric": "auc",
                "callbacks": [
                    lgb.early_stopping(60, verbose=False),
                    lgb.log_evaluation(0),
                ],
            }
        )
    try:
        model.fit(X_tr, y_tr, **kwargs)
    except TypeError:
        kwargs.pop("callbacks", None)
        model.fit(X_tr, y_tr, **kwargs)
    return model


def make_domain_model(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_col_wise=True,
    )


def make_main_model(seed, n_estimators=700):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_col_wise=True,
    )


def estimate_density_ratio_weights(X_train, X_test, cat_cols):
    n_train = len(X_train)
    n_test = len(X_test)
    X_domain = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_domain = np.r_[np.zeros(n_train, dtype=int), np.ones(n_test, dtype=int)]

    domain_oof = np.zeros(len(X_domain), dtype=float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_domain, y_domain), 1):
        model = make_domain_model(RANDOM_STATE + fold)
        fit_lgbm(
            model,
            X_domain.iloc[tr_idx],
            y_domain[tr_idx],
            cat_cols,
            X_val=X_domain.iloc[va_idx],
            y_val=y_domain[va_idx],
        )
        domain_oof[va_idx] = model.predict_proba(X_domain.iloc[va_idx])[:, 1]

    domain_auc = roc_auc_score(y_domain, domain_oof)

    p_test_given_x = np.clip(domain_oof[:n_train], 1e-4, 1 - 1e-4)
    density_ratio = (p_test_given_x / (1.0 - p_test_given_x)) * (n_train / n_test)
    density_ratio = np.clip(density_ratio, 0.20, 5.00)
    density_ratio = density_ratio / np.mean(density_ratio)

    return density_ratio.astype(np.float32), float(domain_auc)


def get_groups(train):
    if {"Year", "Race"}.issubset(train.columns):
        return train["Year"].astype(str) + "_" + train["Race"].astype(str)
    if "Race" in train.columns:
        return train["Race"].astype(str)
    return pd.Series(np.arange(len(train)) % 5)


def run_grouped_cv(train, X, y, cat_cols, weights):
    groups = get_groups(train)
    cv = GroupKFold(n_splits=5)

    weighted_oof = np.zeros(len(X), dtype=float)
    unweighted_oof = np.zeros(len(X), dtype=float)
    weighted_fold_auc = []
    unweighted_fold_auc = []
    best_iterations = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        unweighted = make_main_model(RANDOM_STATE + 100 + fold)
        fit_lgbm(unweighted, X_tr, y_tr, cat_cols, X_val=X_va, y_val=y_va)
        unweighted_oof[va_idx] = unweighted.predict_proba(X_va)[:, 1]

        weighted = make_main_model(RANDOM_STATE + 200 + fold)
        fit_lgbm(
            weighted,
            X_tr,
            y_tr,
            cat_cols,
            sample_weight=weights[tr_idx],
            X_val=X_va,
            y_val=y_va,
        )
        weighted_oof[va_idx] = weighted.predict_proba(X_va)[:, 1]

        uw_auc = roc_auc_score(y_va, unweighted_oof[va_idx])
        wt_auc = roc_auc_score(y_va, weighted_oof[va_idx])
        unweighted_fold_auc.append(uw_auc)
        weighted_fold_auc.append(wt_auc)

        best_iter = getattr(weighted, "best_iteration_", None)
        if best_iter is not None and best_iter > 0:
            best_iterations.append(int(best_iter))

        print(f"fold={fold} unweighted_auc={uw_auc:.6f} weighted_auc={wt_auc:.6f}")

    unweighted_auc = roc_auc_score(y, unweighted_oof)
    weighted_auc = roc_auc_score(y, weighted_oof)
    final_estimators = int(np.median(best_iterations)) if best_iterations else 700
    final_estimators = max(100, min(900, final_estimators))

    return {
        "weighted_oof": weighted_oof,
        "unweighted_oof": unweighted_oof,
        "weighted_auc": float(weighted_auc),
        "unweighted_auc": float(unweighted_auc),
        "weighted_fold_auc": [float(x) for x in weighted_fold_auc],
        "unweighted_fold_auc": [float(x) for x in unweighted_fold_auc],
        "final_estimators": final_estimators,
    }


def main():
    train, test, sample = read_data()
    X_train, X_test, y, feature_cols, cat_cols = prepare_features(train, test)

    weights, domain_auc = estimate_density_ratio_weights(X_train, X_test, cat_cols)
    print(f"train_vs_test_domain_auc={domain_auc:.6f}")
    print(
        "density_ratio_weights "
        f"min={weights.min():.4f} mean={weights.mean():.4f} max={weights.max():.4f}"
    )

    cv_result = run_grouped_cv(train, X_train, y, cat_cols, weights)
    print(f"grouped_cv_unweighted_roc_auc={cv_result['unweighted_auc']:.6f}")
    print(f"grouped_cv_weighted_roc_auc={cv_result['weighted_auc']:.6f}")

    oof_path = os.path.join(WORKING_DIR, "oof_predictions.csv.gz")
    pd.DataFrame(
        {
            "row": np.arange(len(train)),
            "target": y,
            "prediction": np.clip(cv_result["weighted_oof"], 1e-6, 1 - 1e-6),
        }
    ).to_csv(oof_path, index=False, compression="gzip")

    final_model = make_main_model(
        RANDOM_STATE + 999, n_estimators=cv_result["final_estimators"]
    )
    fit_lgbm(final_model, X_train, y, cat_cols, sample_weight=weights)
    test_pred = np.clip(final_model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)

    submission = sample.copy()
    submission["PitNextLap"] = test_pred
    submission_path = os.path.join(WORKING_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    test_pred_path = os.path.join(WORKING_DIR, "test_predictions.csv.gz")
    submission[["id", "PitNextLap"]].to_csv(
        test_pred_path, index=False, compression="gzip"
    )

    result = {
        "research_hypotheses_llm_claimed_used": ["000431"],
        "metric": "roc_auc",
        "grouped_cv_weighted_auc": cv_result["weighted_auc"],
        "grouped_cv_unweighted_auc": cv_result["unweighted_auc"],
        "train_vs_test_domain_auc": domain_auc,
        "weighted_fold_auc": cv_result["weighted_fold_auc"],
        "unweighted_fold_auc": cv_result["unweighted_fold_auc"],
        "final_model_estimators": cv_result["final_estimators"],
        "density_ratio_weight_min": float(weights.min()),
        "density_ratio_weight_mean": float(weights.mean()),
        "density_ratio_weight_max": float(weights.max()),
        "saved_files": {
            "submission": submission_path,
            "oof_predictions": oof_path,
            "test_predictions": test_pred_path,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
