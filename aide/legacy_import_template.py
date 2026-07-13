from __future__ import annotations

import pprint
from typing import Any

from .autogluon_preprocess import validate_preprocess_source
from .utils.config import Config, aux_file_name


TEMPLATE_NAME = "xgb_lgbm_cat_cv5_stack_gpu_balanced"


def build_legacy_stacking_template(preprocess_source: str, cfg: Config) -> str:
    """Wrap AutoGluon feature code in a standalone legacy ML solution."""
    validate_preprocess_source(preprocess_source)
    constants: dict[str, Any] = {
        "template": TEMPLATE_NAME,
        "n_splits": 5,
        "seed": 20260713,
        "models": ["xgboost", "lightgbm", "catboost"],
        "device": "gpu",
        "class_balance": "balanced",
        "stack_model": "logistic_regression",
        "stack_weight": 0.6,
        "base_ensemble_weight": 0.4,
    }
    aux_name = aux_file_name(cfg)
    if aux_name is not None:
        constants["aux_file"] = aux_name
    constants_literal = pprint.pformat(constants, sort_dicts=True, width=88)

    return f'''from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from aide_solution_helpers import (
    aide_stage,
    load_competition_data,
    log_stage,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)

warnings.filterwarnings("ignore")

AIDE_LEGACY_TEMPLATE_CONFIG = {constants_literal}


{preprocess_source.strip()}


def _read_aux() -> pd.DataFrame | None:
    name = AIDE_LEGACY_TEMPLATE_CONFIG.get("aux_file")
    if not name:
        return None
    path = Path("./input") / str(name)
    if not path.exists():
        raise FileNotFoundError(f"Configured auxiliary file not found: {{path}}")
    return pd.read_csv(path)


def _run_feature_code(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_features = train.drop(columns=[target_col], errors="ignore")
    combined = pd.concat([train_features, test.copy()], ignore_index=True, sort=False)
    aux = _read_aux()
    if len(inspect.signature(preprocess).parameters) == 2:
        transformed = preprocess(
            combined.copy(),
            pd.DataFrame() if aux is None else aux.copy(),
        )
    else:
        transformed = preprocess(combined.copy())
    if not isinstance(transformed, pd.DataFrame):
        raise TypeError("preprocess(df) must return a pandas DataFrame")
    if len(transformed) != len(combined):
        raise ValueError(
            f"preprocess(df) changed row count: {{len(transformed)}} != {{len(combined)}}"
        )
    if target_col in transformed.columns:
        raise ValueError(f"preprocess(df) created target column {{target_col!r}}")
    transformed = transformed.reset_index(drop=True)
    return (
        transformed.iloc[: len(train)].copy(),
        transformed.iloc[len(train) :].copy(),
    )


def _encode_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    id_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_features = train_features.drop(columns=[id_col], errors="ignore")
    test_features = test_features.drop(columns=[id_col], errors="ignore")
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError("Train and test feature columns differ after preprocess(df)")

    combined = pd.concat([train_features, test_features], ignore_index=True, sort=False)
    combined.columns = [str(column) for column in combined.columns]
    if combined.columns.duplicated().any():
        duplicates = combined.columns[combined.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature columns: {{duplicates}}")

    categorical = [
        column
        for column in combined.columns
        if not pd.api.types.is_numeric_dtype(combined[column])
    ]
    numeric = [column for column in combined.columns if column not in categorical]
    encoded_parts: list[np.ndarray] = []
    feature_names: list[str] = []

    if numeric:
        numeric_frame = combined[numeric].replace([np.inf, -np.inf], np.nan)
        numeric_values = SimpleImputer(strategy="median").fit_transform(numeric_frame)
        numeric_values = np.nan_to_num(numeric_values, nan=0.0, posinf=0.0, neginf=0.0)
        encoded_parts.append(numeric_values.astype(np.float32, copy=False))
        feature_names.extend(numeric)

    if categorical:
        categorical_frame = combined[categorical].copy()
        for column in categorical:
            categorical_frame[column] = (
                categorical_frame[column].astype("string").fillna("__MISSING__")
            )
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float32,
        )
        encoded_parts.append(encoder.fit_transform(categorical_frame))
        feature_names.extend(categorical)

    if not encoded_parts:
        raise ValueError("preprocess(df) produced no usable features")
    matrix = np.concatenate(encoded_parts, axis=1)
    return matrix[: len(train_features)], matrix[len(train_features) :], feature_names


def _model_factories(class_count: int):
    objective = "multi:softprob" if class_count > 2 else "binary:logistic"
    xgb_metric = "mlogloss" if class_count > 2 else "logloss"
    cat_loss = "MultiClass" if class_count > 2 else "Logloss"
    return {{
        "xgboost": lambda seed: XGBClassifier(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=2.0,
            objective=objective,
            eval_metric=xgb_metric,
            tree_method="hist",
            device="cuda",
            random_state=seed,
            n_jobs=-1,
        ),
        "lightgbm": lambda seed: LGBMClassifier(
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.05,
            reg_lambda=2.0,
            device="cuda",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        ),
        "catboost": lambda seed: CatBoostClassifier(
            iterations=700,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=5.0,
            loss_function=cat_loss,
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.8,
            random_seed=seed,
            allow_writing_files=False,
            verbose=False,
        ),
    }}


def _fit_base_layer(
    x_train: np.ndarray,
    y: np.ndarray,
    x_test: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    config = AIDE_LEGACY_TEMPLATE_CONFIG
    splitter = StratifiedKFold(
        n_splits=int(config["n_splits"]),
        shuffle=True,
        random_state=int(config["seed"]),
    )
    factories = _model_factories(class_count)
    model_names = list(config["models"])
    oof = np.zeros((len(y), len(model_names) * class_count), dtype=np.float32)
    test = np.zeros((len(x_test), len(model_names) * class_count), dtype=np.float64)
    fold_scores = {{name: [] for name in model_names}}

    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(x_train, y), start=1):
        fold_weights = compute_sample_weight(class_weight="balanced", y=y[fit_idx])
        for model_index, name in enumerate(model_names):
            log_stage(f"base_fold={{fold}}/{{config['n_splits']}}|model={{name}}|device=gpu")
            model = factories[name](int(config["seed"]) + 100 * fold + model_index)
            model.fit(x_train[fit_idx], y[fit_idx], sample_weight=fold_weights)
            start = model_index * class_count
            stop = start + class_count
            valid_probability = np.asarray(model.predict_proba(x_train[valid_idx]))
            test_probability = np.asarray(model.predict_proba(x_test))
            oof[valid_idx, start:stop] = valid_probability
            test[:, start:stop] += test_probability / int(config["n_splits"])
            score = balanced_accuracy_score(y[valid_idx], valid_probability.argmax(axis=1))
            fold_scores[name].append(float(score))
            print(
                f"Fold {{fold}}/{{config['n_splits']}} {{name}} balanced_accuracy={{score:.6f}}",
                flush=True,
            )
    return oof, test.astype(np.float32), fold_scores


def _fit_stack_layer(
    base_oof: np.ndarray,
    y: np.ndarray,
    base_test: np.ndarray,
    *,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    config = AIDE_LEGACY_TEMPLATE_CONFIG
    splitter = StratifiedKFold(
        n_splits=int(config["n_splits"]),
        shuffle=True,
        random_state=int(config["seed"]) + 1,
    )
    stack_oof = np.zeros((len(y), class_count), dtype=np.float32)
    stack_test = np.zeros((len(base_test), class_count), dtype=np.float64)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(base_oof, y), start=1):
        log_stage(f"stack_fold={{fold}}/{{config['n_splits']}}|model=logistic_regression")
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=int(config["seed"]) + fold,
        )
        model.fit(base_oof[fit_idx], y[fit_idx])
        stack_oof[valid_idx] = model.predict_proba(base_oof[valid_idx])
        stack_test += model.predict_proba(base_test) / int(config["n_splits"])
    return stack_oof, stack_test.astype(np.float32)


def _base_average(probabilities: np.ndarray, *, class_count: int) -> np.ndarray:
    model_count = len(AIDE_LEGACY_TEMPLATE_CONFIG["models"])
    return probabilities.reshape(len(probabilities), model_count, class_count).mean(axis=1)


def main() -> None:
    config = AIDE_LEGACY_TEMPLATE_CONFIG
    with aide_stage("legacy_load_and_features"):
        train, test, sample_submission = load_competition_data()
        id_col, target_col = sample_submission.columns[:2]
        if target_col not in train.columns:
            raise ValueError(f"Target column {{target_col!r}} not found in train data")
        train_features, test_features = _run_feature_code(
            train,
            test,
            target_col=target_col,
        )
        x_train, x_test, feature_names = _encode_features(
            train_features,
            test_features,
            id_col=id_col,
        )
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(train[target_col])
        class_count = len(label_encoder.classes_)
        if class_count < 2:
            raise ValueError("Classification requires at least two target classes")
        print(
            f"Legacy template built {{len(feature_names)}} encoded features for "
            f"{{len(train)}} train and {{len(test)}} test rows.",
            flush=True,
        )

    with aide_stage("legacy_cv5_base_models"):
        base_oof, base_test, fold_scores = _fit_base_layer(
            x_train,
            y,
            x_test,
            class_count=class_count,
        )

    with aide_stage("legacy_cv5_stacking"):
        stack_oof, stack_test = _fit_stack_layer(
            base_oof,
            y,
            base_test,
            class_count=class_count,
        )

    with aide_stage("legacy_weighted_ensemble"):
        base_average_oof = _base_average(base_oof, class_count=class_count)
        base_average_test = _base_average(base_test, class_count=class_count)
        final_oof = (
            float(config["stack_weight"]) * stack_oof
            + float(config["base_ensemble_weight"]) * base_average_oof
        )
        final_test = (
            float(config["stack_weight"]) * stack_test
            + float(config["base_ensemble_weight"]) * base_average_test
        )
        oof_indices = final_oof.argmax(axis=1)
        test_indices = final_test.argmax(axis=1)
        score = float(balanced_accuracy_score(y, oof_indices))
        oof_labels = label_encoder.inverse_transform(oof_indices)
        test_labels = label_encoder.inverse_transform(test_indices)

    with aide_stage("legacy_write_outputs"):
        submission = sample_submission[[id_col]].copy()
        submission[target_col] = test_labels
        write_submission(submission)

        oof_output = pd.DataFrame({{
            id_col: train[id_col].to_numpy() if id_col in train else np.arange(len(train)),
            "target": train[target_col].to_numpy(),
            "prediction": oof_labels,
        }})
        test_output = submission.copy()
        for class_index, class_label in enumerate(label_encoder.classes_):
            oof_output[f"prob_{{class_label}}"] = final_oof[:, class_index]
            test_output[f"prob_{{class_label}}"] = final_test[:, class_index]
        write_oof_predictions(oof_output)
        write_test_predictions(test_output)

    print(f"Validation balanced_accuracy: {{score:.6f}}")
    print("Submission saved successfully.")
    print("AIDE_RESULT_JSON: " + json.dumps({{
        "is_bug": False,
        "summary": "Legacy GPU CV5 stacked boosting template completed.",
        "metric": score,
        "eval_metric": "balanced_accuracy",
        "lower_is_better": False,
        "run_stats": {{
            "template": config["template"],
            "feature_count": len(feature_names),
            "models": list(config["models"]),
            "n_splits": int(config["n_splits"]),
            "class_balance": config["class_balance"],
            "stack_weight": float(config["stack_weight"]),
            "base_ensemble_weight": float(config["base_ensemble_weight"]),
            "base_fold_scores": fold_scores,
        }},
    }}, sort_keys=True))


main()
'''.strip() + "\n"
