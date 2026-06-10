import os
import sys
import subprocess
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from aide_solution_helpers import (
    load_competition_data,
    working_dir,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    write_validation_predictions,
    aide_stage,
    log_stage,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 2026
N_FOLDS = 5
EPS = 1e-6


def safe_divide(a, b):
    return a / (np.abs(b) + 1e-3)


def softmax_from_logits(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    return exps / exps.sum(axis=1, keepdims=True)


def apply_biases(probs, biases):
    logits = np.log(np.clip(probs, EPS, 1.0)) + biases.reshape(1, -1)
    return softmax_from_logits(logits)


def probs_to_logits(probs):
    return np.log(np.clip(probs, EPS, 1.0))


def build_features(train, test):
    train_df = train.drop(columns=["class"]).copy()
    test_df = test.copy()

    # Covariate-only feature construction on concatenated train/test, with no target present.
    combined = pd.concat(
        [train_df.assign(_is_train=1), test_df.assign(_is_train=0)],
        axis=0,
        ignore_index=True,
    )

    for col in ["u", "g", "r", "i", "z", "alpha", "delta", "redshift"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    for col in ["spectral_type", "galaxy_population"]:
        combined[col] = combined[col].astype(str).fillna("missing")

    combined["u_g"] = combined["u"] - combined["g"]
    combined["g_r"] = combined["g"] - combined["r"]
    combined["r_i"] = combined["r"] - combined["i"]
    combined["i_z"] = combined["i"] - combined["z"]
    combined["u_r"] = combined["u"] - combined["r"]
    combined["g_i"] = combined["g"] - combined["i"]
    combined["r_z"] = combined["r"] - combined["z"]
    combined["u_z"] = combined["u"] - combined["z"]

    redshift_pos = np.clip(combined["redshift"].to_numpy(dtype=np.float64), 0.0, None)
    combined["redshift_pos"] = redshift_pos
    combined["redshift_log1p"] = np.log1p(redshift_pos)
    combined["redshift_sqrt"] = np.sqrt(redshift_pos)
    combined["redshift_sq"] = combined["redshift"].to_numpy(dtype=np.float64) ** 2
    combined["rz_x_redshift"] = combined["r_z"] * combined["redshift_log1p"]
    combined["ug_over_gr"] = safe_divide(
        combined["u_g"].to_numpy(dtype=np.float64),
        combined["g_r"].to_numpy(dtype=np.float64),
    )
    combined["ri_over_iz"] = safe_divide(
        combined["r_i"].to_numpy(dtype=np.float64),
        combined["i_z"].to_numpy(dtype=np.float64),
    )

    alpha_rad = np.deg2rad(combined["alpha"].to_numpy(dtype=np.float64))
    delta_rad = np.deg2rad(combined["delta"].to_numpy(dtype=np.float64))
    combined["alpha_sin"] = np.sin(alpha_rad)
    combined["alpha_cos"] = np.cos(alpha_rad)
    combined["delta_sin"] = np.sin(delta_rad)
    combined["delta_cos"] = np.cos(delta_rad)
    combined["sky_cross"] = combined["alpha_sin"] * combined["delta_cos"]

    for col in ["spectral_type", "galaxy_population"]:
        combined[col] = pd.Categorical(combined[col]).codes.astype(np.int16)

    feature_cols = [c for c in combined.columns if c not in ["id", "_is_train"]]
    features = combined[feature_cols].astype(np.float32)

    n_train = len(train)
    x_train = features.iloc[:n_train].to_numpy(dtype=np.float32, copy=True)
    x_test = features.iloc[n_train:].to_numpy(dtype=np.float32, copy=True)
    return x_train, x_test, feature_cols


def lightgbm_cuda_available():
    code = """
import numpy as np
from lightgbm import LGBMClassifier
rng = np.random.default_rng(0)
X = rng.normal(size=(256, 8)).astype("float32")
y = rng.integers(0, 3, size=256)
model = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=10,
    learning_rate=0.1,
    num_leaves=31,
    class_weight="balanced",
    device_type="cuda",
    verbosity=-1,
)
model.fit(X, y)
print("CUDA_OK", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(working_dir()),
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "CUDA_OK" in result.stdout


def fit_catboost(x_train, y_train, x_valid, y_valid, fold_id):
    log_stage(f"fold={fold_id}|model=catboost|event=fit_start")
    gpu_params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 1400,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 6.0,
        "random_seed": RANDOM_STATE + fold_id,
        "auto_class_weights": "Balanced",
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.8,
        "od_type": "Iter",
        "od_wait": 100,
        "allow_writing_files": False,
        "verbose": False,
    }
    try:
        model = CatBoostClassifier(**gpu_params)
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
        return model
    except Exception as exc:
        print(
            f"CatBoost GPU failed on fold {fold_id}: {exc}. Falling back to CPU.",
            flush=True,
        )
        cpu_params = dict(gpu_params)
        cpu_params.pop("task_type", None)
        cpu_params.pop("devices", None)
        cpu_params.pop("gpu_ram_part", None)
        model = CatBoostClassifier(**cpu_params)
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model=True)
        return model


def fit_lightgbm(x_train, y_train, x_valid, y_valid, fold_id, use_cuda):
    device_params = {"device_type": "cuda"} if use_cuda else {"device_type": "cpu"}
    log_stage(
        f"fold={fold_id}|model=lightgbm|event=fit_start|device={device_params['device_type']}"
    )
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE + fold_id,
        verbosity=-1,
        **device_params,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def fit_xgboost(x_train, y_train, x_valid, y_valid, fold_id):
    log_stage(f"fold={fold_id}|model=xgboost|event=fit_start|device=cuda")
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=1600,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        max_bin=256,
        early_stopping_rounds=100,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE + fold_id,
        n_jobs=max(1, (os.cpu_count() or 1) // 2),
        verbosity=0,
    )
    try:
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        return model
    except Exception as exc:
        print(
            f"XGBoost CUDA failed on fold {fold_id}: {exc}. Falling back to CPU.",
            flush=True,
        )
        cpu_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=1600,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=2.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            tree_method="hist",
            device="cpu",
            max_bin=256,
            early_stopping_rounds=100,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE + fold_id,
            n_jobs=max(1, (os.cpu_count() or 1) // 2),
            verbosity=0,
        )
        cpu_model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        return cpu_model


def evaluate_weight_candidate(candidate, prob_mats, y_true):
    blended = np.zeros_like(prob_mats[0], dtype=np.float64)
    for weight, mat in zip(candidate, prob_mats):
        blended += weight * mat
    preds = blended.argmax(axis=1)
    return balanced_accuracy_score(y_true, preds), candidate


def find_best_weights(prob_mats, y_true):
    weight_candidates = []
    for a in range(1, 9):
        for b in range(1, 9):
            c = 10 - a - b
            if 1 <= c <= 8:
                weight_candidates.append((a / 10.0, b / 10.0, c / 10.0))

    workers = min(16, os.cpu_count() or 1)
    print(
        f"Evaluating {len(weight_candidates)} blend candidates with {workers} workers",
        flush=True,
    )
    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=workers, prefer="threads")(
            delayed(evaluate_weight_candidate)(candidate, prob_mats, y_true)
            for candidate in weight_candidates
        )
    except Exception:
        results = [
            evaluate_weight_candidate(candidate, prob_mats, y_true)
            for candidate in weight_candidates
        ]

    best_score, best_weights = max(results, key=lambda x: x[0])
    return np.array(best_weights, dtype=np.float64), best_score


def evaluate_bias_candidate(candidate, probs, y_true):
    biases = np.array([candidate[0], candidate[1], 0.0], dtype=np.float64)
    adjusted = apply_biases(probs, biases)
    preds = adjusted.argmax(axis=1)
    return balanced_accuracy_score(y_true, preds), biases


def find_best_biases(probs, y_true):
    grid = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
    candidates = [(b0, b1) for b0 in grid for b1 in grid]

    workers = min(16, os.cpu_count() or 1)
    print(
        f"Evaluating {len(candidates)} bias candidates with {workers} workers",
        flush=True,
    )
    try:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=workers, prefer="threads")(
            delayed(evaluate_bias_candidate)(candidate, probs, y_true)
            for candidate in candidates
        )
    except Exception:
        results = [
            evaluate_bias_candidate(candidate, probs, y_true)
            for candidate in candidates
        ]

    best_score, best_biases = max(results, key=lambda x: x[0])
    return best_biases, best_score


def make_calibrator():
    return LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=1.5,
        max_iter=500,
        random_state=RANDOM_STATE,
    )


def calibrated_oof_predictions(blended_oof, y):
    logits = probs_to_logits(blended_oof)
    calibrated = np.zeros_like(blended_oof, dtype=np.float64)
    stage2_cv = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE + 99
    )

    for fold_id, (fit_idx, valid_idx) in enumerate(stage2_cv.split(logits, y), start=1):
        log_stage(f"fold={fold_id}|model=calibration+bias|event=fit_start")
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 500 + fold_id
        )
        inner_fit_rel, inner_tune_rel = next(
            splitter.split(logits[fit_idx], y[fit_idx])
        )
        inner_fit_idx = fit_idx[inner_fit_rel]
        inner_tune_idx = fit_idx[inner_tune_rel]

        inner_calibrator = make_calibrator()
        inner_calibrator.fit(logits[inner_fit_idx], y[inner_fit_idx])
        tune_probs = inner_calibrator.predict_proba(logits[inner_tune_idx])
        fold_biases, fold_bias_score = find_best_biases(tune_probs, y[inner_tune_idx])

        fold_calibrator = make_calibrator()
        fold_calibrator.fit(logits[fit_idx], y[fit_idx])
        valid_probs = fold_calibrator.predict_proba(logits[valid_idx])
        calibrated[valid_idx] = apply_biases(valid_probs, fold_biases)

        fold_preds = calibrated[valid_idx].argmax(axis=1)
        fold_score = balanced_accuracy_score(y[valid_idx], fold_preds)
        print(
            f"Calibration fold {fold_id}: bias_tune_bal_acc={fold_bias_score:.6f}, "
            f"valid_bal_acc={fold_score:.6f}",
            flush=True,
        )

    return calibrated


def fit_final_calibration(blended_oof, blended_test, y):
    oof_logits = probs_to_logits(blended_oof)
    test_logits = probs_to_logits(blended_test)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=RANDOM_STATE + 777
    )
    fit_idx, tune_idx = next(splitter.split(oof_logits, y))

    inner_calibrator = make_calibrator()
    inner_calibrator.fit(oof_logits[fit_idx], y[fit_idx])
    tune_probs = inner_calibrator.predict_proba(oof_logits[tune_idx])
    final_biases, _ = find_best_biases(tune_probs, y[tune_idx])

    final_calibrator = make_calibrator()
    final_calibrator.fit(oof_logits, y)

    calibrated_test = final_calibrator.predict_proba(test_logits)
    calibrated_test = apply_biases(calibrated_test, final_biases)
    return calibrated_test


def main():
    train, test, sample_sub = load_competition_data()

    with aide_stage("build_features_stage"):
        x_train, x_test, feature_cols = build_features(train, test)
        y_labels = train["class"].astype(str).to_numpy()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_labels)
        class_names = list(label_encoder.classes_)
        print(f"Using {len(feature_cols)} features", flush=True)

    with aide_stage("make_folds_stage"):
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(cv.split(x_train, y))
        use_lgb_cuda = lightgbm_cuda_available()
        print(f"LightGBM CUDA available: {use_lgb_cuda}", flush=True)

    model_names = ["catboost", "lightgbm", "xgboost"]
    oof_probs = {
        name: np.zeros((len(train), len(class_names)), dtype=np.float64)
        for name in model_names
    }
    test_probs = {
        name: np.zeros((len(test), len(class_names)), dtype=np.float64)
        for name in model_names
    }

    with aide_stage("fit_predict_fold_stage"):
        for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
            x_tr = x_train[train_idx]
            y_tr = y[train_idx]
            x_va = x_train[valid_idx]
            y_va = y[valid_idx]

            print(f"Starting fold {fold_id}/{N_FOLDS}", flush=True)

            cat_model = fit_catboost(x_tr, y_tr, x_va, y_va, fold_id)
            oof_probs["catboost"][valid_idx] = cat_model.predict_proba(x_va)
            test_probs["catboost"] += cat_model.predict_proba(x_test) / N_FOLDS
            cat_fold_score = balanced_accuracy_score(
                y_va, oof_probs["catboost"][valid_idx].argmax(axis=1)
            )
            print(
                f"Fold {fold_id} CatBoost balanced_accuracy={cat_fold_score:.6f}",
                flush=True,
            )

            lgb_model = fit_lightgbm(x_tr, y_tr, x_va, y_va, fold_id, use_lgb_cuda)
            oof_probs["lightgbm"][valid_idx] = lgb_model.predict_proba(x_va)
            test_probs["lightgbm"] += lgb_model.predict_proba(x_test) / N_FOLDS
            lgb_fold_score = balanced_accuracy_score(
                y_va, oof_probs["lightgbm"][valid_idx].argmax(axis=1)
            )
            print(
                f"Fold {fold_id} LightGBM balanced_accuracy={lgb_fold_score:.6f}",
                flush=True,
            )

            xgb_model = fit_xgboost(x_tr, y_tr, x_va, y_va, fold_id)
            oof_probs["xgboost"][valid_idx] = xgb_model.predict_proba(x_va)
            test_probs["xgboost"] += xgb_model.predict_proba(x_test) / N_FOLDS
            xgb_fold_score = balanced_accuracy_score(
                y_va, oof_probs["xgboost"][valid_idx].argmax(axis=1)
            )
            print(
                f"Fold {fold_id} XGBoost balanced_accuracy={xgb_fold_score:.6f}",
                flush=True,
            )

    with aide_stage("score_stage"):
        base_prob_list = [oof_probs[name] for name in model_names]
        weights, raw_blend_score = find_best_weights(base_prob_list, y)
        print(
            "Chosen blend weights: "
            + ", ".join(
                f"{name}={weight:.2f}" for name, weight in zip(model_names, weights)
            ),
            flush=True,
        )
        print(f"Raw blended OOF balanced_accuracy={raw_blend_score:.6f}", flush=True)

        blended_oof = np.zeros_like(base_prob_list[0], dtype=np.float64)
        blended_test = np.zeros_like(test_probs["catboost"], dtype=np.float64)
        for weight, name in zip(weights, model_names):
            blended_oof += weight * oof_probs[name]
            blended_test += weight * test_probs[name]

        calibrated_oof = calibrated_oof_predictions(blended_oof, y)
        final_score = balanced_accuracy_score(y, calibrated_oof.argmax(axis=1))
        print(f"balanced_accuracy_cv={final_score:.6f}", flush=True)

        calibrated_test = fit_final_calibration(blended_oof, blended_test, y)
        final_test_pred = label_encoder.inverse_transform(
            calibrated_test.argmax(axis=1)
        )
        final_oof_pred = label_encoder.inverse_transform(calibrated_oof.argmax(axis=1))

    with aide_stage("write_outputs_stage"):
        submission = sample_sub[["id"]].copy()
        submission["class"] = final_test_pred
        write_submission(submission)

        oof_frame = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=np.int64),
                "target": y_labels,
                "prediction": final_oof_pred,
            }
        )
        write_oof_predictions(oof_frame)

        test_pred_frame = sample_sub[["id"]].copy()
        for class_idx, class_name in enumerate(class_names):
            test_pred_frame[f"class_{class_name}"] = calibrated_test[:, class_idx]
        test_pred_frame["class"] = final_test_pred
        write_test_predictions(test_pred_frame)


if __name__ == "__main__":
    main()
