import glob
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from aide_solution_helpers import (
    aide_stage,
    load_competition_data,
    log_stage,
    working_dir,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
    write_validation_predictions,
)

RANDOM_STATE = 42
CLASS_NAMES = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
SHARED_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
PHOT_COLS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]


def clean_photometry(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            df[col] = values.mask(values <= -999)
    return df


def cast_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df


def cast_categorical(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
    return df


def align_proba(proba: np.ndarray, model_classes, target_classes) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(target_classes)), dtype=np.float32)
    class_to_idx = {cls: idx for idx, cls in enumerate(model_classes)}
    for j, cls in enumerate(target_classes):
        aligned[:, j] = proba[:, class_to_idx[cls]]
    return aligned


def build_catboost_params(random_seed: int, iterations: int, use_gpu: bool) -> dict:
    params = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": iterations,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 5.0,
        "min_data_in_leaf": 32,
        "auto_class_weights": "Balanced",
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    if use_gpu:
        params.update(
            {
                "task_type": "GPU",
                "devices": "0",
                "gpu_ram_part": 0.8,
            }
        )
    return params


def fit_catboost(
    train_pool: Pool,
    valid_pool: Pool | None,
    random_seed: int,
    iterations: int,
    model_name: str,
):
    last_error = None
    for use_gpu in (True, False):
        mode = "GPU" if use_gpu else "CPU"
        try:
            model = CatBoostClassifier(
                **build_catboost_params(
                    random_seed=random_seed, iterations=iterations, use_gpu=use_gpu
                )
            )
            if valid_pool is None:
                model.fit(train_pool)
            else:
                model.fit(
                    train_pool,
                    eval_set=valid_pool,
                    use_best_model=True,
                    early_stopping_rounds=100,
                )
            return model
        except Exception as exc:
            last_error = exc
            log_stage(f"{model_name} {mode} fit failed, trying fallback: {exc}")
    raise RuntimeError(f"All CatBoost fits failed for {model_name}") from last_error


with aide_stage("build_features_stage"):
    _ = working_dir()
    train, test, sample_sub = load_competition_data()
    target_col = "class"
    id_col = "id"

    aux_candidates = sorted(
        glob.glob(
            os.path.join(".", "input", "**", "star_classification.csv"), recursive=True
        )
    )
    if not aux_candidates:
        raise FileNotFoundError(
            "Could not find auxiliary file star_classification.csv under ./input"
        )
    aux_path = aux_candidates[0]
    aux = pd.read_csv(aux_path)

    train = clean_photometry(train, PHOT_COLS)
    test = clean_photometry(test, PHOT_COLS)
    aux = clean_photometry(aux, PHOT_COLS)

    comp_feature_cols = [c for c in train.columns if c not in (id_col, target_col)]
    numeric_cols = [c for c in comp_feature_cols if c not in CAT_COLS]

    train_features = cast_numeric(train[comp_feature_cols], numeric_cols)
    test_features = cast_numeric(test[comp_feature_cols], numeric_cols)
    train_features = cast_categorical(train_features, CAT_COLS)
    test_features = cast_categorical(test_features, CAT_COLS)

    aux_features = cast_numeric(aux[SHARED_COLS], SHARED_COLS)
    aux_target = aux[target_col].astype(str).values
    y = train[target_col].astype(str).values

with aide_stage("make_folds_stage"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(train_features, y))

with aide_stage("fit_predict_fold_stage"):
    log_stage(f"Training auxiliary CatBoost on {len(aux_features):,} SDSS rows")
    aux_pool = Pool(aux_features, label=aux_target)
    aux_model = fit_catboost(
        train_pool=aux_pool,
        valid_pool=None,
        random_seed=RANDOM_STATE,
        iterations=500,
        model_name="auxiliary_catboost",
    )

    # This auxiliary model is trained on a disjoint labeled domain, so its full-data predictions do not leak competition targets.
    aux_train_proba = align_proba(
        aux_model.predict_proba(aux_features=train_features[SHARED_COLS]),
        aux_model.classes_,
        CLASS_NAMES,
    )
    aux_test_proba = align_proba(
        aux_model.predict_proba(aux_features=test_features[SHARED_COLS]),
        aux_model.classes_,
        CLASS_NAMES,
    )

    stacked_train = train_features.copy()
    stacked_test = test_features.copy()
    for j, cls in enumerate(CLASS_NAMES):
        col = f"aux_proba_{cls.lower()}"
        stacked_train[col] = aux_train_proba[:, j].astype("float32")
        stacked_test[col] = aux_test_proba[:, j].astype("float32")

    oof_proba = np.zeros((len(train), len(CLASS_NAMES)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(CLASS_NAMES)), dtype=np.float32)
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        log_stage(f"Fold {fold}/5 - training competition_catboost")
        X_tr = stacked_train.iloc[train_idx]
        X_va = stacked_train.iloc[valid_idx]
        y_tr = y[train_idx]
        y_va = y[valid_idx]

        train_pool = Pool(X_tr, label=y_tr, cat_features=CAT_COLS)
        valid_pool = Pool(X_va, label=y_va, cat_features=CAT_COLS)
        test_pool = Pool(stacked_test, cat_features=CAT_COLS)

        model = fit_catboost(
            train_pool=train_pool,
            valid_pool=valid_pool,
            random_seed=RANDOM_STATE + fold,
            iterations=900,
            model_name=f"competition_catboost_fold_{fold}",
        )

        valid_proba = align_proba(
            model.predict_proba(valid_pool), model.classes_, CLASS_NAMES
        )
        fold_test_proba = align_proba(
            model.predict_proba(test_pool), model.classes_, CLASS_NAMES
        )

        oof_proba[valid_idx] = valid_proba
        test_proba += fold_test_proba / len(folds)

        valid_pred = CLASS_NAMES[valid_proba.argmax(axis=1)]
        fold_score = balanced_accuracy_score(y_va, valid_pred)
        fold_scores.append(fold_score)
        print(f"Fold {fold} balanced_accuracy: {fold_score:.6f}", flush=True)

with aide_stage("score_stage"):
    oof_pred = CLASS_NAMES[oof_proba.argmax(axis=1)]
    cv_score = balanced_accuracy_score(y, oof_pred)
    print(f"CV balanced_accuracy: {cv_score:.6f}", flush=True)

with aide_stage("write_outputs_stage"):
    test_pred = CLASS_NAMES[test_proba.argmax(axis=1)]

    oof_df = pd.DataFrame(
        {
            "row": np.arange(len(train), dtype=np.int64),
            "target": y,
            "prediction": oof_pred,
        }
    )
    write_oof_predictions(oof_df)

    test_pred_df = pd.DataFrame(
        {
            id_col: sample_sub[id_col].values,
            target_col: test_pred,
            f"{target_col}_GALAXY": test_proba[:, 0],
            f"{target_col}_QSO": test_proba[:, 1],
            f"{target_col}_STAR": test_proba[:, 2],
        }
    )
    write_test_predictions(test_pred_df)

    submission_df = pd.DataFrame(
        {
            id_col: sample_sub[id_col].values,
            target_col: test_pred,
        }
    )
    write_submission(submission_df)
