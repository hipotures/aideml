import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
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
)

RANDOM_STATE = 42
N_FOLDS = 5
N_JOBS = min(16, os.cpu_count() or 1)

ID_COL = "id"
TARGET_COL = "class"
CLASS_NAMES = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
CLASS_TO_INT = {label: idx for idx, label in enumerate(CLASS_NAMES)}

SHARED_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
PHOT_COLS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]


def find_auxiliary_path() -> Path:
    candidates = [
        Path("./input/star_classification.csv"),
        Path("./input/original_sdss17/star_classification.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Expected auxiliary data at ./input/star_classification.csv "
        "or ./input/original_sdss17/star_classification.csv"
    )


def clean_numeric_frame(frame: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in numeric_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        if col in PHOT_COLS:
            values = values.mask(values <= -999)
        frame[col] = values.astype(np.float32)
    return frame


def prepare_competition_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
    numeric_cols = [c for c in feature_cols if c not in CAT_COLS]

    # This concatenation is covariate-only and target-free; it is used only to align category levels.
    combined = pd.concat(
        [train[feature_cols].copy(), test[feature_cols].copy()],
        axis=0,
        ignore_index=True,
    )

    for col in numeric_cols:
        values = pd.to_numeric(combined[col], errors="coerce")
        if col in PHOT_COLS:
            values = values.mask(values <= -999)
        combined[col] = values.astype(np.float32)

    for col in CAT_COLS:
        combined[col] = combined[col].fillna("missing").astype(str).astype("category")

    train_features = combined.iloc[: len(train)].copy()
    test_features = combined.iloc[len(train) :].copy()
    return train_features, test_features, feature_cols


def prepare_auxiliary_features(aux: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    aux = aux.loc[aux[TARGET_COL].isin(CLASS_TO_INT)].copy()
    aux_x = clean_numeric_frame(aux[SHARED_COLS], SHARED_COLS)
    aux_y = aux[TARGET_COL].map(CLASS_TO_INT).to_numpy(dtype=np.int32)
    return aux_x, aux_y


def build_lgbm(seed: int, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASS_NAMES),
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
    )


with aide_stage("build_features_stage"):
    working_dir()
    train, test, sample_sub = load_competition_data()

    aux_path = find_auxiliary_path()
    aux = pd.read_csv(aux_path)

    y = train[TARGET_COL].map(CLASS_TO_INT).to_numpy(dtype=np.int32)

    train_features, test_features, feature_cols = prepare_competition_features(
        train, test
    )
    aux_features, aux_y = prepare_auxiliary_features(aux)

with aide_stage("make_folds_stage"):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(train_features, y))

with aide_stage("fit_predict_fold_stage"):
    log_stage(
        f"fold=0/{N_FOLDS}|model=aux_lgbm_shared|rows={len(aux_features)}|path={aux_path.name}"
    )
    aux_model = build_lgbm(seed=RANDOM_STATE, n_estimators=1200)
    aux_model.fit(aux_features, aux_y)

    # The auxiliary model is trained on a separate labeled domain, so these stack features do not leak competition targets.
    aux_train_proba = aux_model.predict_proba(train_features[SHARED_COLS])
    aux_test_proba = aux_model.predict_proba(test_features[SHARED_COLS])

    stacked_train = train_features.copy()
    stacked_test = test_features.copy()
    for class_idx, class_name in enumerate(CLASS_NAMES):
        col_name = f"aux_proba_{class_name.lower()}"
        stacked_train[col_name] = aux_train_proba[:, class_idx].astype(np.float32)
        stacked_test[col_name] = aux_test_proba[:, class_idx].astype(np.float32)

    oof_proba = np.zeros((len(train), len(CLASS_NAMES)), dtype=np.float32)
    test_proba = np.zeros((len(test), len(CLASS_NAMES)), dtype=np.float32)

    for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
        log_stage(f"fold={fold}/{N_FOLDS}|model=comp_lgbm_stacked")

        x_train = stacked_train.iloc[train_idx]
        x_valid = stacked_train.iloc[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        model = build_lgbm(seed=RANDOM_STATE + fold, n_estimators=3000)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="multi_logloss",
            categorical_feature=CAT_COLS,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        valid_proba = model.predict_proba(x_valid, num_iteration=model.best_iteration_)
        fold_test_proba = model.predict_proba(
            stacked_test, num_iteration=model.best_iteration_
        )

        oof_proba[valid_idx] = valid_proba.astype(np.float32)
        test_proba += fold_test_proba.astype(np.float32) / N_FOLDS

        valid_pred = valid_proba.argmax(axis=1)
        fold_score = balanced_accuracy_score(y_valid, valid_pred)
        print(f"Fold {fold} balanced_accuracy: {fold_score:.6f}", flush=True)

with aide_stage("score_stage"):
    oof_pred = oof_proba.argmax(axis=1)
    cv_score = balanced_accuracy_score(y, oof_pred)
    print(f"CV balanced_accuracy: {cv_score:.6f}", flush=True)

with aide_stage("write_outputs_stage"):
    oof_labels = CLASS_NAMES[oof_pred]
    test_pred = test_proba.argmax(axis=1)
    test_labels = CLASS_NAMES[test_pred]

    oof_df = pd.DataFrame(
        {
            "row": np.arange(len(train), dtype=np.int64),
            "target": CLASS_NAMES[y],
            "prediction": oof_labels,
        }
    )
    write_oof_predictions(oof_df)

    test_pred_df = pd.DataFrame(
        {
            ID_COL: sample_sub[ID_COL].values,
            TARGET_COL: test_labels,
            f"{TARGET_COL}_GALAXY": test_proba[:, 0],
            f"{TARGET_COL}_QSO": test_proba[:, 1],
            f"{TARGET_COL}_STAR": test_proba[:, 2],
        }
    )
    write_test_predictions(test_pred_df)

    submission_df = pd.DataFrame(
        {
            ID_COL: sample_sub[ID_COL].values,
            TARGET_COL: test_labels,
        }
    )
    write_submission(submission_df)
