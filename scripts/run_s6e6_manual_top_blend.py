from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import os
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


COMPETITION = "playground-series-s6e6"
RUN_ID = "manual-s6e6-top-preprocess-tabular-blend"
ID_COL = "id"
TARGET_COL = "class"
CLASS_ORDER = ["GALAXY", "QSO", "STAR"]

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings(
    "ignore",
    message=".*Falling back to prediction using DMatrix due to mismatched devices.*",
    category=UserWarning,
)


def log_blank() -> None:
    print("", flush=True)


REALMLP_PARAMS = {
    "random_state": 42,
    "verbosity": 2,
    "n_ens": 40,
    "n_epochs": 5,
    "batch_size": 256,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10,
    "early_stopping_multiplicative_patience": 1,
    "lr": 0.019,
    "wd": 0.01,
    "sq_mom": 0.99,
    "lr_sched": "lin_cos_log_15",
    "first_layer_lr_factor": 0.25,
    "embedding_size": 6,
    "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128],
    "act": "silu",
    "p_drop": 0.05,
    "p_drop_sched": "invsqrtp1e-3",
    "plr_hidden_1": 16,
    "plr_hidden_2": 8,
    "plr_act_name": "gelu",
    "plr_lr_factor": 0.1151,
    "plr_sigma": 2.33,
    "ls_eps": 0.01,
    "ls_eps_sched": "sqrt_cos",
    "add_front_scale": False,
    "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": [
        "one_hot",
        "median_center",
        "robust_scale",
        "smooth_clip",
        "embedding",
        "l2_normalize",
    ],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(data_dir: Path, stem: str) -> pd.DataFrame:
    for path in (data_dir / f"{stem}.csv", data_dir / f"{stem}.csv.gz"):
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Missing {stem}.csv or {stem}.csv.gz under {data_dir}")


def load_preprocess_function(path: Path):
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "preprocess":
            expr = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(expr)
            namespace = {"pd": pd, "np": np}
            exec(compile(expr, str(path), "exec"), namespace)
            return namespace["preprocess"]
    raise ValueError(f"No top-level preprocess function found in {path}")


def stratified_subsample(
    train: pd.DataFrame,
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(train) <= max_rows:
        return train
    frac = max_rows / float(len(train))
    return (
        train.groupby(TARGET_COL, group_keys=False)
        .apply(lambda part: part.sample(max(1, int(round(len(part) * frac))), random_state=seed))
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def make_features(
    *,
    data_dir: Path,
    aux_path: Path,
    preprocess_path: Path,
    max_train_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, LabelEncoder, list[str]]:
    train = read_csv(data_dir, "train")
    test = read_csv(data_dir, "test")
    sample = read_csv(data_dir, "sample_submission")
    aux = pd.read_csv(aux_path)
    train = stratified_subsample(train, max_train_rows, seed)

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_ORDER)
    y = pd.Series(label_encoder.transform(train[TARGET_COL]), name=TARGET_COL)

    train_features = train.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    test_features = test.drop(columns=[ID_COL], errors="ignore")
    combined = pd.concat([train_features, test_features], ignore_index=True, sort=False)

    preprocess = load_preprocess_function(preprocess_path)
    started_at = time.time()
    processed = preprocess(combined.copy(), aux.copy())
    preprocess_time = time.time() - started_at
    if not isinstance(processed, pd.DataFrame):
        raise TypeError("preprocess must return a pandas DataFrame")
    if len(processed) != len(combined):
        raise ValueError(f"preprocess changed row count: {len(processed)} != {len(combined)}")

    train_fe = processed.iloc[: len(train)].reset_index(drop=True)
    test_fe = processed.iloc[len(train) :].reset_index(drop=True)
    cat_cols = [
        col
        for col in train_fe.columns
        if pd.api.types.is_object_dtype(train_fe[col])
        or isinstance(train_fe[col].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(train_fe[col])
    ]
    print(
        f"Prepared features: train={train_fe.shape} test={test_fe.shape} "
        f"cat_cols={len(cat_cols)} preprocess_time={preprocess_time:.2f}s",
        flush=True,
    )
    return train_fe, test_fe, y, test[ID_COL], sample, label_encoder, cat_cols


def make_numeric_frames(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[bool]]:
    combined = pd.concat([train_fe, test_fe], ignore_index=True, sort=False)
    out = combined.copy()
    for col in cat_cols:
        ser = out[col].astype("string").fillna("__MISSING__")
        out[col] = pd.Categorical(ser).codes.astype("int32")
    for col in out.columns:
        if col not in cat_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    train_num = out.iloc[: len(train_fe)].copy()
    test_num = out.iloc[len(train_fe) :].copy()
    medians = train_num.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_num = train_num.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    test_num = test_num.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    cat_indicator = [col in set(cat_cols) for col in train_num.columns]
    return train_num, test_num, cat_indicator


def make_catboost_frames(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_cb = train_fe.copy()
    test_cb = test_fe.copy()
    for frame in (train_cb, test_cb):
        for col in cat_cols:
            frame[col] = frame[col].astype("string").fillna("__MISSING__").astype(str)
    return train_cb, test_cb, cat_cols


def normalize_proba(proba: np.ndarray, n_classes: int) -> np.ndarray:
    arr = np.asarray(proba, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != n_classes:
        raise ValueError(f"Expected proba shape (*, {n_classes}), got {arr.shape}")
    arr = np.clip(arr, 1e-12, np.inf)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


def fit_xgb_fold(X_tr, y_tr, X_va, y_va, X_te, seed: int, n_classes: int):
    return fit_xgb_fold_with_params(X_tr, y_tr, X_va, y_va, X_te, seed, n_classes, {})


def fit_xgb_fold_with_params(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    seed: int,
    n_classes: int,
    params_override: dict[str, Any],
):
    from xgboost import XGBClassifier

    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "eval_metric": "mlogloss",
        "n_estimators": 1600,
        "learning_rate": 0.035,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 8.0,
        "min_child_weight": 2.0,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": seed,
        "verbosity": 1,
        "early_stopping_rounds": 100,
    }
    params.update(params_override)
    model = XGBClassifier(**params)
    weights = compute_sample_weight(class_weight="balanced", y=y_tr)
    try:
        model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_va, y_va)], verbose=False)
    except Exception as exc:
        print(f"XGBoost CUDA fit failed, retrying on CPU: {exc}", flush=True)
        model.set_params(device="cpu")
        model.fit(X_tr, y_tr, sample_weight=weights, eval_set=[(X_va, y_va)], verbose=False)
    return normalize_proba(model.predict_proba(X_va), n_classes), normalize_proba(
        model.predict_proba(X_te), n_classes
    )


def fit_lgb_fold(X_tr, y_tr, X_va, y_va, X_te, seed: int, n_classes: int):
    return fit_lgb_fold_with_params(X_tr, y_tr, X_va, y_va, X_te, seed, n_classes, {})


def fit_lgb_fold_with_params(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    seed: int,
    n_classes: int,
    params_override: dict[str, Any],
):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "n_estimators": 2200,
        "learning_rate": 0.035,
        "num_leaves": 96,
        "max_depth": -1,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_lambda": 8.0,
        "class_weight": "balanced",
        "random_state": seed,
        "device": "cuda",
        "verbosity": -1,
    }
    params.update(params_override)
    model = LGBMClassifier(**params)
    try:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
    except Exception as exc:
        print(f"LightGBM CUDA fit failed, retrying on CPU: {exc}", flush=True)
        model.set_params(device="cpu")
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
    return normalize_proba(model.predict_proba(X_va), n_classes), normalize_proba(
        model.predict_proba(X_te), n_classes
    )


def fit_cat_fold(X_tr, y_tr, X_va, y_va, X_te, cat_cols: list[str], seed: int, n_classes: int):
    return fit_cat_fold_with_params(X_tr, y_tr, X_va, y_va, X_te, cat_cols, seed, n_classes, {})


def fit_cat_fold_with_params(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    cat_cols: list[str],
    seed: int,
    n_classes: int,
    params_override: dict[str, Any],
):
    from catboost import CatBoostClassifier, Pool

    params = {
        "loss_function": "MultiClass",
        "eval_metric": "TotalF1:average=Macro",
        "iterations": 2200,
        "learning_rate": 0.035,
        "depth": 6,
        "l2_leaf_reg": 8.0,
        "random_seed": seed,
        "auto_class_weights": "Balanced",
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.8,
        "od_type": "Iter",
        "od_wait": 120,
        "allow_writing_files": False,
        "verbose": 200,
    }
    params.update(params_override)
    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_cols)
    valid_pool = Pool(X_va, label=y_va, cat_features=cat_cols)
    test_pool = Pool(X_te, cat_features=cat_cols)
    try:
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    except Exception as exc:
        print(f"CatBoost GPU fit failed, retrying on CPU: {exc}", flush=True)
        params["task_type"] = "CPU"
        params.pop("devices", None)
        params.pop("gpu_ram_part", None)
        params["thread_count"] = os.cpu_count() or -1
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return normalize_proba(model.predict_proba(valid_pool), n_classes), normalize_proba(
        model.predict_proba(test_pool), n_classes
    )


def fit_realmlp_fold(
    X_tr,
    y_tr,
    X_va,
    y_va,
    X_te,
    cat_indicator: list[bool],
    seed: int,
    n_classes: int,
    n_ens: int,
):
    from pytabkit import RealMLP_TD_Classifier

    params = dict(REALMLP_PARAMS)
    params["random_state"] = seed
    params["n_ens"] = n_ens
    model = RealMLP_TD_Classifier(**params)
    model.fit(
        X_tr,
        y_tr,
        X_va,
        y_va,
        cat_indicator=np.asarray(cat_indicator, dtype=bool),
    )
    return normalize_proba(model.predict_proba(X_va), n_classes), normalize_proba(
        model.predict_proba(X_te), n_classes
    )


def train_model_oof(
    model_name: str,
    *,
    X_num: pd.DataFrame,
    X_test_num: pd.DataFrame,
    X_cb: pd.DataFrame,
    X_test_cb: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    cat_indicator: list[bool],
    folds: int,
    seed: int,
    realmlp_n_ens: int,
    tuned_params: dict[str, Any] | None = None,
    lgb_device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_classes = len(CLASS_ORDER)
    oof = np.zeros((len(y), n_classes), dtype=np.float64)
    test_preds = np.zeros((len(X_test_num), n_classes), dtype=np.float64)
    scores: list[float] = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    started_at = time.time()
    tuned_params = tuned_params or {}

    log_blank()
    print(f"[{model_name}] starting {folds}-fold OOF training", flush=True)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num, y), 1):
        print(f"[{model_name}] fold {fold}/{folds}", flush=True)
        y_tr = y.iloc[tr_idx].to_numpy()
        y_va = y.iloc[va_idx].to_numpy()
        if model_name == "xgb":
            val_proba, fold_test = fit_xgb_fold_with_params(
                X_num.iloc[tr_idx],
                y_tr,
                X_num.iloc[va_idx],
                y_va,
                X_test_num,
                seed + fold,
                n_classes,
                tuned_params,
            )
        elif model_name == "lgb":
            fold_params = dict(tuned_params)
            fold_params["device"] = lgb_device
            val_proba, fold_test = fit_lgb_fold_with_params(
                X_num.iloc[tr_idx],
                y_tr,
                X_num.iloc[va_idx],
                y_va,
                X_test_num,
                seed + fold,
                n_classes,
                fold_params,
            )
        elif model_name == "cat":
            val_proba, fold_test = fit_cat_fold_with_params(
                X_cb.iloc[tr_idx],
                y_tr,
                X_cb.iloc[va_idx],
                y_va,
                X_test_cb,
                cat_cols,
                seed + fold,
                n_classes,
                tuned_params,
            )
        elif model_name == "realmlp":
            val_proba, fold_test = fit_realmlp_fold(
                X_num.iloc[tr_idx],
                y_tr,
                X_num.iloc[va_idx],
                y_va,
                X_test_num,
                cat_indicator,
                seed + fold,
                n_classes,
                realmlp_n_ens,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        oof[va_idx] = val_proba
        test_preds += fold_test / folds
        fold_score = balanced_accuracy_score(y_va, val_proba.argmax(axis=1))
        scores.append(float(fold_score))
        print(f"[{model_name}] fold {fold} balanced_accuracy={fold_score:.6f}", flush=True)

    score = balanced_accuracy_score(y, oof.argmax(axis=1))
    log_blank()
    stats = {
        "model": model_name,
        "oof_balanced_accuracy": float(score),
        "fold_balanced_accuracy": scores,
        "fit_time": float(time.time() - started_at),
        "tuned_params": tuned_params,
    }
    print(f"[{model_name}] OOF balanced_accuracy={score:.6f}", flush=True)
    return oof, test_preds, stats


def suggest_xgb_params(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2600, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 9),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
    }


def suggest_lgb_params(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 2600, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 48, 192, step=16),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12]),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 120, step=10),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
    }


def suggest_cat_params(trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 1000, 2600, step=200),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 2.0, 30.0, log=True),
        "bootstrap_type": "Bayesian",
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "border_count": trial.suggest_categorical("border_count", [128, 254]),
        "verbose": False,
    }


def validate_hpo_session(session: str) -> None:
    if not session.strip():
        raise ValueError("--hpo-session cannot be empty")
    if any(part in session for part in ("/", "\\", "..")):
        raise ValueError("--hpo-session must be a simple name without path separators")


def run_model_hpo(
    model_name: str,
    *,
    session: str,
    hpo_dir: Path,
    n_trials: int,
    folds: int,
    X_num: pd.DataFrame,
    X_cb: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    seed: int,
    lgb_device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    import optuna

    if model_name not in {"xgb", "lgb", "cat"}:
        raise ValueError(f"Optuna HPO is not configured for model: {model_name}")
    if n_trials < 0:
        raise ValueError("--hpo-trials must be >= 0")
    if folds < 2:
        raise ValueError("--hpo-folds must be at least 2")

    validate_hpo_session(session)
    hpo_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"{session}-{model_name}"
    db_path = hpo_dir / f"{study_name}.db"
    storage_name = f"sqlite:///{db_path.resolve()}"
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )
    trials_before = len(study.trials)
    completed_before = sum(
        1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    )
    trials_to_run = max(0, n_trials - completed_before)
    log_blank()
    print(
        f"[hpo:{model_name}] study={study_name} storage={db_path} "
        f"existing_trials={trials_before} completed_trials={completed_before} "
        f"target_trials={n_trials} adding_trials={trials_to_run}",
        flush=True,
    )

    n_classes = len(CLASS_ORDER)
    y_arr = y.to_numpy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 1019)

    def objective(trial) -> float:
        if model_name == "xgb":
            params = suggest_xgb_params(trial)
        elif model_name == "lgb":
            params = suggest_lgb_params(trial)
            params["device"] = lgb_device
        elif model_name == "cat":
            params = suggest_cat_params(trial)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        scores: list[float] = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num, y_arr), 1):
            y_tr = y_arr[tr_idx]
            y_va = y_arr[va_idx]
            if model_name == "xgb":
                val_proba, _ = fit_xgb_fold_with_params(
                    X_num.iloc[tr_idx],
                    y_tr,
                    X_num.iloc[va_idx],
                    y_va,
                    X_num.iloc[va_idx],
                    seed + 7000 + trial.number * 100 + fold,
                    n_classes,
                    params,
                )
            elif model_name == "lgb":
                val_proba, _ = fit_lgb_fold_with_params(
                    X_num.iloc[tr_idx],
                    y_tr,
                    X_num.iloc[va_idx],
                    y_va,
                    X_num.iloc[va_idx],
                    seed + 7000 + trial.number * 100 + fold,
                    n_classes,
                    params,
                )
            else:
                val_proba, _ = fit_cat_fold_with_params(
                    X_cb.iloc[tr_idx],
                    y_tr,
                    X_cb.iloc[va_idx],
                    y_va,
                    X_cb.iloc[va_idx],
                    cat_cols,
                    seed + 7000 + trial.number * 100 + fold,
                    n_classes,
                    params,
                )
            score = balanced_accuracy_score(y_va, val_proba.argmax(axis=1))
            scores.append(float(score))

        mean_score = float(np.mean(scores))
        trial.set_user_attr("fold_balanced_accuracy", scores)
        print(
            f"[hpo:{model_name}] trial={trial.number} "
            f"balanced_accuracy={mean_score:.6f} folds={scores}",
            flush=True,
        )
        return mean_score

    started_at = time.time()
    if trials_to_run > 0:
        study.optimize(objective, n_trials=trials_to_run)
    trials_after = len(study.trials)
    completed_after = sum(
        1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    )
    try:
        best_trial = study.best_trial
    except ValueError:
        stats = {
            "enabled": False,
            "session": session,
            "study_name": study_name,
            "storage": str(db_path),
            "trials_before": trials_before,
            "trials_after": trials_after,
            "completed_trials_before": completed_before,
            "completed_trials_after": completed_after,
            "target_trials": n_trials,
            "trials_to_run": trials_to_run,
            "trials_added": trials_after - trials_before,
            "reason": "no completed trials in study",
            "fit_time": float(time.time() - started_at),
            "folds": folds,
        }
        log_blank()
        print(f"[hpo:{model_name}] no completed trials; using default parameters", flush=True)
        log_blank()
        return {}, stats
    best_params = dict(best_trial.params)
    stats = {
        "enabled": True,
        "session": session,
        "study_name": study_name,
        "storage": str(db_path),
        "trials_before": trials_before,
        "trials_after": trials_after,
        "completed_trials_before": completed_before,
        "completed_trials_after": completed_after,
        "target_trials": n_trials,
        "trials_to_run": trials_to_run,
        "trials_added": trials_after - trials_before,
        "best_trial_number": int(best_trial.number),
        "best_value": float(best_trial.value),
        "best_params": best_params,
        "fit_time": float(time.time() - started_at),
        "folds": folds,
    }
    if model_name == "lgb":
        stats["lgb_hpo_device"] = lgb_device
    log_blank()
    print(
        f"[hpo:{model_name}] best_trial={stats['best_trial_number']} "
        f"best_balanced_accuracy={stats['best_value']:.6f}",
        flush=True,
    )
    return best_params, stats


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def blend_arrays(mode: str, weights: np.ndarray, arrays: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(arrays, axis=0)
    if mode == "raw":
        blended = np.tensordot(weights, stacked, axes=(0, 0))
        return normalize_proba(blended, stacked.shape[2])
    if mode == "logit":
        logits = np.log(np.clip(stacked, 1e-8, 1.0))
        blended = np.tensordot(weights, logits, axes=(0, 0))
        return softmax(blended)
    raise ValueError(f"Unknown blend mode: {mode}")


def blend_candidates(n_models: int, step: float) -> list[np.ndarray]:
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    candidates: list[np.ndarray] = []

    def rec(prefix: list[float], remaining: int, total: float) -> None:
        if remaining == 1:
            last = 1.0 - total
            if last >= -1e-9:
                weights = np.array(prefix + [max(0.0, last)], dtype=np.float64)
                if np.count_nonzero(weights > 1e-9) > 0:
                    candidates.append(weights / weights.sum())
            return
        for value in grid:
            if total + value <= 1.0 + 1e-9:
                rec(prefix + [float(value)], remaining - 1, total + float(value))

    rec([], n_models, 0.0)
    return candidates


def select_blend(
    y: np.ndarray,
    arrays: list[np.ndarray],
    *,
    step: float,
) -> tuple[str, np.ndarray, float]:
    best_mode = "raw"
    best_weights = np.ones(len(arrays), dtype=np.float64) / len(arrays)
    best_score = -np.inf
    for mode in ("raw", "logit"):
        for weights in blend_candidates(len(arrays), step):
            pred = blend_arrays(mode, weights, arrays).argmax(axis=1)
            score = balanced_accuracy_score(y, pred)
            if score > best_score:
                best_score = float(score)
                best_mode = mode
                best_weights = weights
    return best_mode, best_weights, best_score


def nested_blend(
    y: pd.Series,
    model_names: list[str],
    oof_by_model: dict[str, np.ndarray],
    test_by_model: dict[str, np.ndarray],
    *,
    folds: int,
    seed: int,
    step: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_arr = y.to_numpy()
    oof_arrays = [oof_by_model[name] for name in model_names]
    test_arrays = [test_by_model[name] for name in model_names]
    meta_oof = np.zeros_like(oof_arrays[0])
    test_blend = np.zeros_like(test_arrays[0])
    selected: list[dict[str, Any]] = []
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 913)

    log_blank()
    print(f"[blend] starting nested blend for models={model_names}", flush=True)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(oof_arrays[0], y_arr), 1):
        mode, weights, train_score = select_blend(
            y_arr[tr_idx],
            [arr[tr_idx] for arr in oof_arrays],
            step=step,
        )
        meta_oof[va_idx] = blend_arrays(mode, weights, [arr[va_idx] for arr in oof_arrays])
        test_blend += blend_arrays(mode, weights, test_arrays) / folds
        valid_score = balanced_accuracy_score(y_arr[va_idx], meta_oof[va_idx].argmax(axis=1))
        selected.append(
            {
                "fold": fold,
                "mode": mode,
                "weights": {name: float(weight) for name, weight in zip(model_names, weights)},
                "train_balanced_accuracy": float(train_score),
                "valid_balanced_accuracy": float(valid_score),
            }
        )
        print(
            f"[blend] meta fold {fold}: mode={mode} valid={valid_score:.6f} "
            f"weights={selected[-1]['weights']}",
            flush=True,
        )

    nested_score = balanced_accuracy_score(y_arr, meta_oof.argmax(axis=1))
    full_mode, full_weights, full_score = select_blend(y_arr, oof_arrays, step=step)
    log_blank()
    stats = {
        "nested_blend_balanced_accuracy": float(nested_score),
        "full_oof_selected_balanced_accuracy": float(full_score),
        "full_oof_mode": full_mode,
        "full_oof_weights": {
            name: float(weight) for name, weight in zip(model_names, full_weights)
        },
        "nested_folds": selected,
    }
    print(f"[blend] nested OOF balanced_accuracy={nested_score:.6f}", flush=True)
    print(
        f"[blend] full OOF diagnostic balanced_accuracy={full_score:.6f} "
        f"mode={full_mode} weights={stats['full_oof_weights']}",
        flush=True,
    )
    return meta_oof, test_blend, stats


def proba_frame(
    *,
    ids: pd.Series | None,
    y_true: pd.Series | None,
    proba: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({f"proba_{label}": proba[:, idx] for idx, label in enumerate(labels)})
    frame.insert(0, "prediction", labels[proba.argmax(axis=1)])
    if y_true is not None:
        frame.insert(0, "target", labels[y_true.to_numpy()])
        frame.insert(0, "row", np.arange(len(frame), dtype=np.int64))
    if ids is not None:
        frame.insert(0, ID_COL, ids.reset_index(drop=True))
    return frame


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_artifacts(
    *,
    artifact_dir: Path,
    labels: np.ndarray,
    test_ids: pd.Series,
    sample: pd.DataFrame,
    y: pd.Series,
    oof_proba: np.ndarray,
    test_proba: np.ndarray,
    model_stats: list[dict[str, Any]],
    blend_stats: dict[str, Any],
    args: argparse.Namespace,
    exec_time: float,
) -> Path:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    solution_path = artifact_dir / "solution.py"
    shutil.copy2(Path(__file__), solution_path)

    test_labels = labels[test_proba.argmax(axis=1)]
    pred_frame = pd.DataFrame({ID_COL: test_ids.reset_index(drop=True), TARGET_COL: test_labels})
    submission = sample.copy()
    submission[TARGET_COL] = submission[[ID_COL]].merge(
        pred_frame,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )[TARGET_COL].to_numpy()
    submission_path = artifact_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    oof_path = artifact_dir / "oof_predictions.csv.gz"
    proba_frame(ids=None, y_true=y, proba=oof_proba, labels=labels).to_csv(
        oof_path,
        index=False,
        compression="gzip",
    )
    test_path = artifact_dir / "test_predictions.csv.gz"
    proba_frame(ids=test_ids, y_true=None, proba=test_proba, labels=labels).to_csv(
        test_path,
        index=False,
        compression="gzip",
    )

    local_score = float(blend_stats["nested_blend_balanced_accuracy"])
    timestamp = artifact_dir.name
    node_id = uuid.uuid4().hex
    manifest = {
        "schema_version": 1,
        "kind": "source_node",
        "run": RUN_ID,
        "timestamp": timestamp,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "status": "ok",
        "local_score": local_score,
        "metric_maximize": True,
        "is_buggy": False,
        "sha256": sha256_file(submission_path),
        "files": {
            "solution": {
                "path": "solution.py",
                "sha256": sha256_file(solution_path),
                "size": solution_path.stat().st_size,
            },
            "submission": {
                "path": "submission.csv",
                "sha256": sha256_file(submission_path),
                "size": submission_path.stat().st_size,
            },
            "oof_predictions": {
                "path": "oof_predictions.csv.gz",
                "sha256": sha256_file(oof_path),
                "size": oof_path.stat().st_size,
            },
            "test_predictions": {
                "path": "test_predictions.csv.gz",
                "sha256": sha256_file(test_path),
                "size": test_path.stat().st_size,
            },
        },
        "node": {
            "id": node_id,
            "step": 0,
            "ctime": time.time(),
            "parent_id": None,
            "status": "ok",
            "origin": "normal",
            "plan": (
                "Manual s6e6 runner using top AG preprocessing with "
                "RealMLP/PyTabKit-style OOF blend plus balanced GBDT experts."
            ),
            "analysis": (
                "RealMLP has no sample_weight/class_weight in the local PyTabKit API; "
                "class balancing is applied to XGBoost, LightGBM, and CatBoost."
            ),
            "validity_warning": None,
            "is_buggy": False,
            "metric": {"value": local_score, "maximize": True, "name": "balanced_accuracy"},
            "submission_validation": None,
        },
        "execution": {"exec_time": exec_time},
        "run_stats": {
            "eval_metric": "balanced_accuracy",
            "models": model_stats,
            "blend": blend_stats,
            "args": json_safe(vars(args)),
        },
        "submission_validation": None,
        "autogluon": {},
        "source": {
            "source_run": "2-hopping-sheep-from-camelot",
            "source_node_id": "57f1f399122b474da92cce640e6c4b91",
            "source_step": None,
            "source_timestamp": "20260603T195610",
            "source_preprocess_sha256": sha256_file(Path(args.preprocess_source)),
        },
    }
    write_json(artifact_dir / "aide_result.json", manifest)
    write_json(
        artifact_dir.parent.parent / "journal.json",
        {
            "nodes": [
                {
                    "id": node_id,
                    "step": 0,
                    "ctime": manifest["node"]["ctime"],
                    "code": solution_path.read_text(encoding="utf-8"),
                    "plan": manifest["node"]["plan"],
                    "analysis": manifest["node"]["analysis"],
                    "is_buggy": False,
                    "metric": {"value": local_score, "maximize": True, "name": "balanced_accuracy"},
                    "exec_time": exec_time,
                    "children": [],
                    "parent": None,
                }
            ],
            "node2parent": {},
        },
    )
    return submission_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("aide/example_tasks/playground-series-s6e6"),
    )
    parser.add_argument(
        "--aux-path",
        type=Path,
        default=Path("aide/example_tasks/playground-series-s6e6/original_sdss17/star_classification.csv"),
    )
    parser.add_argument(
        "--preprocess-source",
        type=Path,
        default=Path("logs/2-hopping-sheep-from-camelot/artifacts/20260603T195610/solution.py"),
    )
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["xgb", "lgb", "cat", "realmlp"],
        default=["xgb", "lgb", "cat", "realmlp"],
    )
    parser.add_argument("--realmlp-n-ens", type=int, default=40)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument(
        "--hpo-session",
        default=None,
        help="Persistent Optuna session prefix. Studies are named '<session>-xgb', etc.",
    )
    parser.add_argument(
        "--hpo-trials",
        type=int,
        default=20,
        help="Target number of completed Optuna trials per selected supported model.",
    )
    parser.add_argument(
        "--hpo-folds",
        type=int,
        default=3,
        help="CV folds used inside each Optuna objective.",
    )
    parser.add_argument(
        "--hpo-dir",
        type=Path,
        default=Path("logs/optuna/playground-series-s6e6"),
        help="Workspace directory for persistent Optuna SQLite databases.",
    )
    parser.add_argument(
        "--lgb-hpo-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="LightGBM device used during Optuna HPO. CPU avoids native CUDA crashes.",
    )
    parser.add_argument(
        "--lgb-device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="LightGBM device used for final OOF/test folds.",
    )
    parser.add_argument(
        "--artifact-suffix",
        default="",
        help="Optional suffix added to the artifact timestamp for smoke-test runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.hpo_session is not None:
        validate_hpo_session(args.hpo_session)
    started_at = time.time()
    train_fe, test_fe, y, test_ids, sample, label_encoder, cat_cols = make_features(
        data_dir=args.data_dir,
        aux_path=args.aux_path,
        preprocess_path=args.preprocess_source,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
    X_num, X_test_num, cat_indicator = make_numeric_frames(train_fe, test_fe, cat_cols)
    X_cb, X_test_cb, cb_cat_cols = make_catboost_frames(train_fe, test_fe, cat_cols)

    oof_by_model: dict[str, np.ndarray] = {}
    test_by_model: dict[str, np.ndarray] = {}
    model_stats: list[dict[str, Any]] = []
    for model_name in args.models:
        tuned_params: dict[str, Any] = {}
        hpo_stats: dict[str, Any] | None = None
        if args.hpo_session and model_name in {"xgb", "lgb", "cat"}:
            tuned_params, hpo_stats = run_model_hpo(
                model_name,
                session=args.hpo_session,
                hpo_dir=args.hpo_dir,
                n_trials=args.hpo_trials,
                folds=args.hpo_folds,
                X_num=X_num,
                X_cb=X_cb,
                y=y,
                cat_cols=cb_cat_cols,
                seed=args.seed,
                lgb_device=args.lgb_hpo_device,
            )
        elif args.hpo_session:
            hpo_stats = {
                "enabled": False,
                "reason": f"Optuna HPO is not configured for {model_name}",
            }
            print(f"[hpo:{model_name}] skipped: {hpo_stats['reason']}", flush=True)

        oof, test_pred, stats = train_model_oof(
            model_name,
            X_num=X_num,
            X_test_num=X_test_num,
            X_cb=X_cb,
            X_test_cb=X_test_cb,
            y=y,
            cat_cols=cb_cat_cols,
            cat_indicator=cat_indicator,
            folds=args.folds,
            seed=args.seed,
            realmlp_n_ens=args.realmlp_n_ens,
            tuned_params=tuned_params,
            lgb_device=args.lgb_device,
        )
        if hpo_stats is not None:
            stats["hpo"] = hpo_stats
        oof_by_model[model_name] = oof
        test_by_model[model_name] = test_pred
        model_stats.append(stats)

    oof_proba, test_proba, blend_stats = nested_blend(
        y,
        list(args.models),
        oof_by_model,
        test_by_model,
        folds=args.folds,
        seed=args.seed,
        step=args.blend_step,
    )
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S") + args.artifact_suffix
    artifact_dir = args.logs_dir / RUN_ID / "artifacts" / timestamp
    submission_path = write_artifacts(
        artifact_dir=artifact_dir,
        labels=label_encoder.classes_,
        test_ids=test_ids,
        sample=sample,
        y=y,
        oof_proba=oof_proba,
        test_proba=test_proba,
        model_stats=model_stats,
        blend_stats=blend_stats,
        args=args,
        exec_time=time.time() - started_at,
    )
    submission_sha = sha256_file(submission_path)
    log_blank()
    print(f"Saved submission: {submission_path}", flush=True)
    log_blank()
    print(
        "Submit command: "
        f"uv run python scripts/kaggle_submission_lab.py --sha {submission_sha[:10]}",
        flush=True,
    )
    log_blank()
    print(
        "AIDE_RESULT_JSON: "
        + json.dumps(
            {
                "is_bug": False,
                "metric": blend_stats["nested_blend_balanced_accuracy"],
                "eval_metric": "balanced_accuracy",
                "lower_is_better": False,
                "run_stats": {"models": model_stats, "blend": blend_stats},
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
