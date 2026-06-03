from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import datetime as dt
import inspect
import joblib
import json
import math
import os
import signal
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aideml-matplotlib")
try:
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
except OSError:
    pass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from aide.autogluon_preprocess import extract_preprocess_source
from scripts import kaggle_submission_lab as lab


DEFAULT_PROJECT_NAME = "playground-series-s6e6"
DEFAULT_DATA_DIR = Path("aide/example_tasks") / DEFAULT_PROJECT_NAME
DEFAULT_OUTPUT_DIR = Path("logs/lazypredict")
DEFAULT_TUNE_TOP_K = 5
DEFAULT_TUNE_TRIALS = 50
DEFAULT_TUNE_TIMEOUT = None
DEFAULT_TUNE_BACKEND = "optuna"
DEFAULT_TUNE_N_JOBS = 16
DEFAULT_TUNE_METRIC = "roc_auc"
SUPPORTED_METRICS = {"roc_auc", "balanced_accuracy", "accuracy", "f1"}
METRIC_ALIASES = {
    "auc": "roc_auc",
    "roc auc": "roc_auc",
    "roc-auc": "roc_auc",
    "roc_auc": "roc_auc",
    "balanced accuracy": "balanced_accuracy",
    "balanced-accuracy": "balanced_accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "bal acc": "balanced_accuracy",
    "bal_acc": "balanced_accuracy",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "f1": "f1",
    "f1 score": "f1",
    "f1_score": "f1",
}
NATIVE_BOOSTING_MODELS = {
    "CatBoostClassifier",
    "LGBMClassifier",
    "XGBClassifier",
}
DEFAULT_EXCLUDED_MODELS = {
    "CategoricalNB",
    "FixedThresholdClassifier",
    "KNeighborsClassifier",
    "LabelPropagation",
    "LabelSpreading",
    "NuSVC",
    "SVC",
    "SelfTrainingClassifier",
    "StackingClassifier",
    "TunedThresholdClassifierCV",
}
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
WORKER_DATA: dict[str, Any] = {}


def load_project_env() -> dict[str, str]:
    load_dotenv(dotenv_path=Path(".env"), override=True)
    return {
        "project_name": os.getenv("AIDE_PROJECT_NAME", "").strip(),
        "project_metric": os.getenv("AIDE_PROJECT_METRIC", "").strip(),
        "data_dir": os.getenv("AIDE_PROJECT_DATA_DIR", "").strip(),
    }


def normalize_metric(value: str) -> str:
    normalized = " ".join(str(value).strip().lower().replace("_", " ").split())
    metric = METRIC_ALIASES.get(normalized)
    if metric is None:
        valid = ", ".join(sorted(SUPPORTED_METRICS))
        raise ValueError(f"Unsupported metric {value!r}; expected one of: {valid}")
    return metric


def metric_column(metric: str) -> str:
    return {
        "roc_auc": "ROC AUC",
        "balanced_accuracy": "Balanced Accuracy",
        "accuracy": "Accuracy",
        "f1": "F1 Score",
    }[metric]


def write_outputs(
    *,
    summary_path: Path,
    tuning_path: Path,
    details_path: Path,
    summary_rows: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
) -> None:
    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
        compression="gzip" if summary_path.name.endswith(".gz") else None,
    )
    if tuning_rows:
        pd.DataFrame(tuning_rows).to_csv(
            tuning_path,
            index=False,
            compression="gzip" if tuning_path.name.endswith(".gz") else None,
        )
    details_path.write_text(json.dumps(details, indent=2, default=str) + "\n")


def timestamp_now() -> str:
    return dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def read_csv(data_dir: Path, stem: str) -> pd.DataFrame:
    gz_path = data_dir / f"{stem}.csv.gz"
    csv_path = data_dir / f"{stem}.csv"
    if gz_path.exists():
        return pd.read_csv(gz_path)
    return pd.read_csv(csv_path)


def make_combined_frame(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train_df.copy(), test_df.copy()], ignore_index=True, sort=False)


@contextlib.contextmanager
def preprocess_timeout(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def raise_timeout(_signum, _frame):
        raise TimeoutError(f"preprocess(df) exceeded {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def validate_preprocessed_frame(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    target_col: str,
    train_rows: int,
    test_rows: int,
) -> pd.DataFrame:
    if not isinstance(after, pd.DataFrame):
        raise TypeError("preprocess(df) must return a pandas DataFrame")
    if len(after) != len(before):
        raise ValueError(f"preprocess changed row count: {len(after)} != {len(before)}")
    forbidden_columns = {target_col, "__is_train__", "__aide_row_id__"}
    forbidden = sorted(forbidden_columns.intersection(after.columns))
    if forbidden:
        raise ValueError(f"preprocess created forbidden column(s): {', '.join(forbidden)}")
    ordered = after.reset_index(drop=True)
    if len(ordered.iloc[:train_rows]) != train_rows:
        raise ValueError("preprocess changed number of train rows")
    if len(ordered.iloc[train_rows:]) != test_rows:
        raise ValueError("preprocess changed number of test rows")
    return ordered


def normalized_score(record: dict[str, Any]) -> float:
    score = record.get("local_score")
    if score is None:
        return float("-inf")
    value = float(score)
    if record.get("metric_maximize") is False:
        return -value
    return value


def record_is_ready(record: dict[str, Any]) -> bool:
    return (
        record.get("local_score") is not None
        and not record.get("is_buggy")
        and record.get("status") == "ok"
        and record.get("solution_path")
        and Path(record["solution_path"]).exists()
    )


def record_matches_competition(record: dict[str, Any], competition: str | None) -> bool:
    if not competition:
        return True
    record_competition = str(record.get("competition") or "").strip()
    return not record_competition or record_competition == competition


def select_records(
    records: list[dict[str, Any]],
    *,
    run: str | None,
    competition: str | None,
    limit: int,
    dedupe: bool,
) -> list[dict[str, Any]]:
    ready = [record for record in records if record_is_ready(record)]
    ready = [
        record
        for record in ready
        if record_matches_competition(record, competition)
    ]
    if run:
        ready = [record for record in ready if record.get("run") == run]
    ordered = sorted(
        ready,
        key=lambda record: (normalized_score(record), str(record.get("timestamp") or "")),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in ordered:
        key = str(record.get("sha256") or record.get("solution_path") or "")
        if dedupe and key in seen:
            continue
        if key:
            seen.add(key)
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def resolve_data_dir(record: dict[str, Any], explicit_data_dir: Path | None) -> Path:
    if explicit_data_dir is not None:
        return explicit_data_dir
    run = str(record.get("run") or "")
    if run:
        workspace_input = Path("workspaces") / run / "input"
        if data_dir_has_task_files(workspace_input):
            return workspace_input
    return DEFAULT_DATA_DIR


def data_dir_has_task_files(data_dir: Path) -> bool:
    return all(
        (data_dir / f"{stem}.csv").exists()
        or (data_dir / f"{stem}.csv.gz").exists()
        for stem in ("train", "test", "sample_submission")
    )


def load_task_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    train_df = read_csv(data_dir, "train")
    test_df = read_csv(data_dir, "test")
    sample_submission = read_csv(data_dir, "sample_submission")
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]
    if target_col not in train_df.columns:
        raise ValueError(f"Target column {target_col!r} not found in train data")
    return train_df, test_df, sample_submission, id_col, target_col


def build_preprocess_function(solution_path: Path):
    source = extract_preprocess_source(solution_path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"np": np, "pd": pd}
    exec(source, namespace)
    preprocess = namespace.get("preprocess")
    if not callable(preprocess):
        raise ValueError(f"No callable preprocess(df) in {solution_path}")
    return preprocess


def preprocess_accepts_aux(preprocess: Any) -> bool:
    signature = inspect.signature(preprocess)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    return len(positional) >= 2


def read_aux_csv(data_dir: Path, solution_path: Path) -> pd.DataFrame | None:
    config = lab.parse_autogluon_config(solution_path.read_text(encoding="utf-8")) or {}
    aux_file = config.get("aux_file")
    if not aux_file:
        return None
    aux_path = data_dir / str(aux_file)
    if not aux_path.exists():
        raise FileNotFoundError(f"Configured aux file not found: {aux_path}")
    return pd.read_csv(aux_path)


def run_preprocess(
    preprocess: Any,
    combined: pd.DataFrame,
    aux_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if preprocess_accepts_aux(preprocess):
        if aux_df is None:
            aux_df = pd.DataFrame()
        return preprocess(combined.copy(), aux_df.copy())
    return preprocess(combined.copy())


def run_artifact_preprocess(
    record: dict[str, Any],
    *,
    data_dir: Path,
    preprocess_time_limit: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, dict[str, Any]]:
    train_df, test_df, sample_submission, id_col, target_col = load_task_frames(data_dir)
    y = train_df[target_col].reset_index(drop=True)
    train_features = train_df.drop(columns=[target_col, id_col], errors="ignore")
    test_features = test_df.drop(columns=[id_col], errors="ignore")
    combined = make_combined_frame(train_features, test_features)

    solution_path = Path(record["solution_path"])
    preprocess = build_preprocess_function(solution_path)
    aux_df = read_aux_csv(data_dir, solution_path)
    started = time.time()
    with preprocess_timeout(preprocess_time_limit):
        preprocessed = run_preprocess(preprocess, combined, aux_df)
    preprocess_time = time.time() - started
    preprocessed = validate_preprocessed_frame(
        combined,
        preprocessed,
        target_col=target_col,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )
    train_fe = preprocessed.iloc[: len(train_df)].copy()
    test_fe = preprocessed.iloc[len(train_df):].copy()
    metadata = {
        "data_dir": str(data_dir),
        "target_col": target_col,
        "id_col": id_col,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "preprocessed_columns": int(len(train_fe.columns)),
        "preprocess_time": preprocess_time,
        "preprocess_accepts_aux": preprocess_accepts_aux(preprocess),
        "aux_rows": None if aux_df is None else int(len(aux_df)),
        "aux_columns": None if aux_df is None else int(len(aux_df.columns)),
    }
    return train_fe, test_fe, y, sample_submission, metadata


def clean_for_lazypredict(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = features.copy()
    numeric_cols = list(cleaned.select_dtypes(include=[np.number]).columns)
    inf_count = 0
    if numeric_cols:
        numeric_values = cleaned[numeric_cols].to_numpy(dtype=float, copy=False)
        inf_count = int(np.isinf(numeric_values).sum())
        if inf_count:
            cleaned[numeric_cols] = cleaned[numeric_cols].replace(
                [np.inf, -np.inf],
                np.nan,
            )

    for col in cleaned.columns:
        dtype = cleaned[col].dtype
        if (
            pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_categorical_dtype(dtype)
            or pd.api.types.is_datetime64_any_dtype(dtype)
            or pd.api.types.is_timedelta64_dtype(dtype)
        ):
            cleaned[col] = cleaned[col].astype("object")

    cleaned = cleaned.where(pd.notna(cleaned), np.nan)
    null_cells = int(cleaned.isna().sum().sum())
    metadata = {
        "feature_columns": int(len(cleaned.columns)),
        "numeric_columns": int(len(cleaned.select_dtypes(include=[np.number, "bool"]).columns)),
        "object_columns": int(len(cleaned.select_dtypes(include=["object"]).columns)),
        "null_cells": null_cells,
        "inf_cells_replaced": inf_count,
    }
    return cleaned, metadata


def sample_train_valid(
    features: pd.DataFrame,
    y: pd.Series,
    *,
    sample_train: int | None,
    sample_valid: int | None,
    valid_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    total = len(features)
    if not 0 < valid_fraction < 1:
        raise ValueError("--valid-fraction must be between 0 and 1")

    if sample_train is None and sample_valid is None:
        requested = total
        split_fraction = valid_fraction
    else:
        if sample_train is None:
            sample_valid = max(1, int(sample_valid or 1))
            sample_train = max(1, int(round(sample_valid * (1.0 - valid_fraction) / valid_fraction)))
        if sample_valid is None:
            sample_train = max(1, int(sample_train or 1))
            sample_valid = max(1, int(round(sample_train * valid_fraction / (1.0 - valid_fraction))))
        requested = min(total, max(2, int(sample_train) + int(sample_valid)))
        split_fraction = min(0.8, max(1, int(sample_valid)) / max(2, int(sample_train) + int(sample_valid)))

    stratify = y if y.nunique(dropna=True) <= 20 else None
    if requested < total:
        sample_idx, _ = train_test_split(
            np.arange(total),
            train_size=requested,
            random_state=random_state,
            stratify=stratify,
        )
        features = features.iloc[sample_idx].reset_index(drop=True)
        y = y.iloc[sample_idx].reset_index(drop=True)
        stratify = y if y.nunique(dropna=True) <= 20 else None

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        y,
        test_size=split_fraction,
        random_state=random_state,
        stratify=stratify,
    )
    metadata = {
        "source_rows": int(total),
        "sampled_rows": int(len(features)),
        "sample_fraction": float(len(features) / total),
        "valid_fraction": float(split_fraction),
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
    }
    return x_train, x_valid, y_train, y_valid, metadata


@contextlib.contextmanager
def model_timeout(seconds: float | None):
    if seconds is None or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def raise_timeout(_signum, _frame):
        raise TimeoutError(f"model exceeded timeout of {seconds:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def init_model_worker(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    categorical_encoder: str,
    random_state: int,
    n_jobs_per_model: int,
    model_timeout_seconds: float | None,
    eval_mode: str,
    cv_folds: int,
    native_boosting_data: bool,
) -> None:
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(max(1, int(n_jobs_per_model)))
    WORKER_DATA.clear()
    WORKER_DATA.update(
        {
            "x_train": x_train,
            "x_valid": x_valid,
            "y_train": y_train,
            "y_valid": y_valid,
            "categorical_encoder": categorical_encoder,
            "random_state": random_state,
            "n_jobs_per_model": max(1, int(n_jobs_per_model)),
            "model_timeout": model_timeout_seconds,
            "eval_mode": eval_mode,
            "cv_folds": cv_folds,
            "native_boosting_data": native_boosting_data,
        }
    )


def model_kwargs_for(model_cls: Any, *, random_state: int, n_jobs_per_model: int) -> dict[str, Any]:
    try:
        params = model_cls().get_params()
    except Exception:
        params = {}
    kwargs: dict[str, Any] = {}
    if "random_state" in params:
        kwargs["random_state"] = random_state
    if "n_jobs" in params:
        kwargs["n_jobs"] = n_jobs_per_model
    if "nthread" in params:
        kwargs["nthread"] = n_jobs_per_model
    if "thread_count" in params:
        kwargs["thread_count"] = n_jobs_per_model

    module = getattr(model_cls, "__module__", "") or ""
    if "catboost" in module:
        kwargs.setdefault("verbose", 0)
        kwargs.setdefault("allow_writing_files", False)
    if "lightgbm" in module:
        kwargs.setdefault("verbose", -1)
        kwargs.setdefault("verbosity", -1)
    if "xgboost" in module:
        kwargs.setdefault("verbosity", 0)
        kwargs.setdefault("enable_categorical", True)
        kwargs.setdefault("tree_method", "hist")
    return kwargs


def model_uses_native_data(model_name: str, native_boosting_data: bool) -> bool:
    return native_boosting_data and model_name in NATIVE_BOOSTING_MODELS


class NativeBoostingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        model_cls: Any,
        model_name: str,
        model_params: dict[str, Any] | None = None,
    ):
        self.model_cls = model_cls
        self.model_name = model_name
        self.model_params = model_params

    def _prepare_fit_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x.copy()
        self.feature_columns_ = list(frame.columns)
        self.categorical_columns_ = list(
            frame.select_dtypes(include=["object", "category", "string"]).columns
        )
        self.category_levels_ = {}
        for col in self.categorical_columns_:
            if self.model_name == "CatBoostClassifier":
                frame[col] = frame[col].astype("object").where(frame[col].notna(), "missing").astype(str)
            else:
                cat = frame[col].astype("category")
                self.category_levels_[col] = list(cat.cat.categories)
                frame[col] = cat
        return frame

    def _prepare_predict_frame(self, x: pd.DataFrame) -> pd.DataFrame:
        frame = x.copy()
        frame = frame.reindex(columns=self.feature_columns_)
        for col in self.categorical_columns_:
            if self.model_name == "CatBoostClassifier":
                frame[col] = frame[col].astype("object").where(frame[col].notna(), "missing").astype(str)
            else:
                dtype = pd.CategoricalDtype(categories=self.category_levels_.get(col))
                frame[col] = frame[col].astype(dtype)
        return frame

    def fit(self, x: pd.DataFrame, y: pd.Series):
        params = dict(self.model_params or {})
        self.model_ = self.model_cls(**params)
        x_fit = self._prepare_fit_frame(x)
        if self.model_name == "CatBoostClassifier" and self.categorical_columns_:
            self.model_.fit(x_fit, y, cat_features=self.categorical_columns_)
        else:
            self.model_.fit(x_fit, y)
        self.classes_ = getattr(self.model_, "classes_", np.array(sorted(pd.unique(y))))
        return self

    def predict(self, x: pd.DataFrame):
        return self.model_.predict(self._prepare_predict_frame(x))

    def predict_proba(self, x: pd.DataFrame):
        return self.model_.predict_proba(self._prepare_predict_frame(x))


def build_estimator(
    model_name: str,
    model_cls: Any,
    x_train: pd.DataFrame,
    *,
    categorical_encoder: str,
    random_state: int,
    n_jobs: int,
    native_boosting_data: bool,
):
    from lazypredict.Supervised import build_preprocessor

    kwargs = model_kwargs_for(
        model_cls,
        random_state=random_state,
        n_jobs_per_model=n_jobs,
    )
    if model_uses_native_data(model_name, native_boosting_data):
        return NativeBoostingClassifier(model_cls, model_name, kwargs)
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train, categorical_encoder)),
            ("classifier", model_cls(**kwargs)),
        ]
    )


def apply_tuned_params(estimator: Any, params: dict[str, Any]) -> Any:
    if not params:
        return estimator
    if hasattr(estimator, "model_params"):
        estimator.model_params = {
            **dict(getattr(estimator, "model_params") or {}),
            **params,
        }
    else:
        estimator.set_params(**{f"classifier__{key}": value for key, value in params.items()})
    return estimator


def compute_classifier_metrics(pipe: Pipeline, x_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, Any]:
    y_pred = pipe.predict(x_valid)
    roc_auc = None
    try:
        if hasattr(pipe, "predict_proba"):
            y_pred_proba = pipe.predict_proba(x_valid)
            if y_pred_proba.shape[1] == 2:
                roc_auc = roc_auc_score(y_valid, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(
                    y_valid,
                    y_pred_proba,
                    multi_class="ovr",
                    average="weighted",
                )
        elif hasattr(pipe, "decision_function"):
            roc_auc = roc_auc_score(y_valid, pipe.decision_function(x_valid))
        else:
            roc_auc = roc_auc_score(y_valid, y_pred)
    except Exception:
        roc_auc = None

    return {
        "accuracy": accuracy_score(y_valid, y_pred, normalize=True),
        "balanced_accuracy": balanced_accuracy_score(y_valid, y_pred),
        "roc_auc": roc_auc,
        "f1": f1_score(y_valid, y_pred, average="weighted"),
        "precision": precision_score(y_valid, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_valid, y_pred, average="weighted"),
    }


def compute_classifier_cv_metrics(
    pipe: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": make_scorer(f1_score, average="weighted"),
        "precision": make_scorer(precision_score, average="weighted", zero_division=0),
        "recall": make_scorer(recall_score, average="weighted"),
        "roc_auc": "roc_auc",
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipe,
                x_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
    except Exception:
        scoring.pop("roc_auc", None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(
                pipe,
                x_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
        scores["test_roc_auc"] = [np.nan]

    return {
        "accuracy": float(np.nanmean(scores["test_accuracy"])),
        "balanced_accuracy": float(np.nanmean(scores["test_balanced_accuracy"])),
        "roc_auc": finite_or_none(np.nanmean(scores["test_roc_auc"])),
        "f1": float(np.nanmean(scores["test_f1"])),
        "precision": float(np.nanmean(scores["test_precision"])),
        "recall": float(np.nanmean(scores["test_recall"])),
    }


def fit_one_model(model_item: tuple[str, Any]) -> dict[str, Any]:
    from lazypredict.Supervised import prepare_dataframes

    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None

    name, model_cls = model_item
    x_train = WORKER_DATA["x_train"]
    x_valid = WORKER_DATA["x_valid"]
    y_train = WORKER_DATA["y_train"]
    y_valid = WORKER_DATA["y_valid"]
    categorical_encoder = WORKER_DATA["categorical_encoder"]
    random_state = int(WORKER_DATA["random_state"])
    n_jobs_per_model = int(WORKER_DATA["n_jobs_per_model"])
    timeout = WORKER_DATA["model_timeout"]
    eval_mode = str(WORKER_DATA.get("eval_mode") or "valid")
    cv_folds = int(WORKER_DATA.get("cv_folds") or 5)
    native_boosting_data = bool(WORKER_DATA.get("native_boosting_data", True))

    started = time.time()
    try:
        x_train, x_valid = prepare_dataframes(x_train, x_valid)
        pipe = build_estimator(
            name,
            model_cls,
            x_train,
            categorical_encoder=categorical_encoder,
            random_state=random_state,
            n_jobs=n_jobs_per_model,
            native_boosting_data=native_boosting_data,
        )
        limiter = (
            threadpool_limits(limits=n_jobs_per_model)
            if threadpool_limits is not None
            else contextlib.nullcontext()
        )
        with model_timeout(timeout), limiter:
            if eval_mode == "cv":
                metrics = compute_classifier_cv_metrics(
                    pipe,
                    x_train,
                    y_train,
                    cv_folds=cv_folds,
                    random_state=random_state,
                )
            else:
                pipe.fit(x_train, y_train)
                metrics = compute_classifier_metrics(pipe, x_valid, y_valid)
        metrics["name"] = name
        metrics["time"] = time.time() - started
        return {"status": "ok", "name": name, "metrics": metrics}
    except Exception as exc:  # noqa: BLE001 - per-model failures are expected in LazyPredict
        return {
            "status": "error",
            "name": name,
            "error": repr(exc),
            "time": time.time() - started,
        }


def parse_best_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or pd.isna(value):
        return {}
    text = str(value)
    if not text or text.startswith("N/A"):
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        raise TypeError(f"Best Params is not a dict: {type(parsed).__name__}")
    return parsed


def safe_estimator_name(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    return safe.strip("_") or "model"


def build_tuned_estimators(
    tuned_scores: pd.DataFrame,
    x_reference: pd.DataFrame,
    *,
    max_models: int | None,
    exclude_models: set[str],
    categorical_encoder: str,
    random_state: int,
    n_jobs: int,
    native_boosting_data: bool,
    top_k: int | None,
) -> tuple[list[tuple[str, Pipeline]], list[dict[str, Any]]]:
    estimator_lookup = dict(
        estimator_items(max_models, exclude_models=exclude_models)
    )
    rows = tuned_scores.reset_index().to_dict(orient="records")
    rows = [row for row in rows if row.get("Status") == "ok"]
    rows = sorted(
        rows,
        key=lambda row: (
            float("-inf")
            if pd.isna(row.get("Tuned ROC AUC"))
            else float(row.get("Tuned ROC AUC"))
        ),
        reverse=True,
    )
    if top_k is not None:
        rows = rows[:top_k]

    estimators: list[tuple[str, Pipeline]] = []
    stack_models: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for row in rows:
        model_name = str(row.get("Model") or "")
        model_class = estimator_lookup.get(model_name)
        if model_class is None:
            continue
        params = parse_best_params(row.get("Best Params"))
        base_name = safe_estimator_name(model_name)
        name = base_name
        idx = 2
        while name in used_names:
            name = f"{base_name}_{idx}"
            idx += 1
        used_names.add(name)
        estimator = build_estimator(
            model_name,
            model_class,
            x_reference,
            categorical_encoder=categorical_encoder,
            random_state=random_state,
            n_jobs=n_jobs,
            native_boosting_data=native_boosting_data,
        )
        apply_tuned_params(estimator, params)
        estimators.append(
            (
                name,
                estimator,
            )
        )
        stack_models.append(
            {
                "name": model_name,
                "estimator_name": name,
                "params": params,
                "tuned_roc_auc": finite_or_none(row.get("Tuned ROC AUC")),
            }
        )
    return estimators, stack_models


def make_tuned_stacking_classifier(
    estimators: list[tuple[str, Pipeline]],
    *,
    random_state: int,
    stack_cv_folds: int,
    stack_n_jobs: int,
) -> StackingClassifier:
    if len(estimators) < 2:
        raise ValueError("Need at least two tuned models to build a stack")
    final_estimator = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=stack_n_jobs,
    )
    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=stack_cv_folds,
        stack_method="auto",
        n_jobs=stack_n_jobs,
        passthrough=False,
        verbose=0,
    )


def run_tuned_stacking(
    tuned_scores: pd.DataFrame,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    *,
    max_models: int | None,
    exclude_models: set[str],
    categorical_encoder: str,
    random_state: int,
    eval_mode: str,
    cv_folds: int,
    stack_cv_folds: int,
    stack_n_jobs: int,
    stack_timeout: float | None,
    stack_top_k: int | None,
    native_boosting_data: bool,
    console: Console,
) -> dict[str, Any]:
    from lazypredict.preprocessing import prepare_dataframes

    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None

    x_train, x_valid = prepare_dataframes(x_train, x_valid)
    estimators, stack_models = build_tuned_estimators(
        tuned_scores,
        x_train,
        max_models=max_models,
        exclude_models=exclude_models,
        categorical_encoder=categorical_encoder,
        random_state=random_state,
        n_jobs=stack_n_jobs,
        native_boosting_data=native_boosting_data,
        top_k=stack_top_k,
    )
    stack = make_tuned_stacking_classifier(
        estimators,
        random_state=random_state,
        stack_cv_folds=stack_cv_folds,
        stack_n_jobs=stack_n_jobs,
    )
    limiter = (
        threadpool_limits(limits=stack_n_jobs)
        if threadpool_limits is not None
        else contextlib.nullcontext()
    )
    started = time.time()
    with model_timeout(stack_timeout), limiter:
        if eval_mode == "cv":
            metrics = compute_classifier_cv_metrics(
                stack,
                x_train,
                y_train,
                cv_folds=cv_folds,
                random_state=random_state,
            )
        else:
            stack.fit(x_train, y_train)
            metrics = compute_classifier_metrics(stack, x_valid, y_valid)
    elapsed = time.time() - started
    console.print(
        "stacked tuned models: "
        + ", ".join(model["name"] for model in stack_models)
    )
    return {
        "status": "ok",
        "model": "StackingClassifier(tuned)",
        "roc_auc": finite_or_none(metrics.get("roc_auc")),
        "balanced_accuracy": finite_or_none(metrics.get("balanced_accuracy")),
        "accuracy": finite_or_none(metrics.get("accuracy")),
        "f1": finite_or_none(metrics.get("f1")),
        "precision": finite_or_none(metrics.get("precision")),
        "recall": finite_or_none(metrics.get("recall")),
        "time_taken": elapsed,
        "stack_models": stack_models,
        "stack_cv_folds": stack_cv_folds,
        "stack_n_jobs": stack_n_jobs,
    }


def write_tuned_stack_submission(
    tuned_scores: pd.DataFrame,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_path: Path,
    *,
    models_dir: Path | None,
    save_models: bool,
    max_models: int | None,
    exclude_models: set[str],
    categorical_encoder: str,
    random_state: int,
    stack_cv_folds: int,
    stack_n_jobs: int,
    stack_timeout: float | None,
    stack_top_k: int | None,
    native_boosting_data: bool,
) -> dict[str, Any]:
    from lazypredict.preprocessing import prepare_dataframes

    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None

    x_final = pd.concat([x_train, x_valid], ignore_index=True)
    y_final = pd.concat(
        [pd.Series(y_train), pd.Series(y_valid)],
        ignore_index=True,
    )
    x_final, x_test = prepare_dataframes(x_final, x_test)
    estimators, stack_models = build_tuned_estimators(
        tuned_scores,
        x_final,
        max_models=max_models,
        exclude_models=exclude_models,
        categorical_encoder=categorical_encoder,
        random_state=random_state,
        n_jobs=stack_n_jobs,
        native_boosting_data=native_boosting_data,
        top_k=stack_top_k,
    )
    stack = make_tuned_stacking_classifier(
        estimators,
        random_state=random_state,
        stack_cv_folds=stack_cv_folds,
        stack_n_jobs=stack_n_jobs,
    )
    limiter = (
        threadpool_limits(limits=stack_n_jobs)
        if threadpool_limits is not None
        else contextlib.nullcontext()
    )
    saved_base_models: list[dict[str, Any]] = []
    if save_models:
        if models_dir is None:
            models_dir = output_path.with_suffix("").parent / f"{output_path.stem}-models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for estimator_name, estimator in estimators:
            model_started = time.time()
            with model_timeout(stack_timeout), limiter:
                estimator.fit(x_final, y_final)
            model_path = models_dir / f"{estimator_name}.joblib"
            joblib.dump(estimator, model_path)
            saved_base_models.append(
                {
                    "estimator_name": estimator_name,
                    "path": str(model_path),
                    "fit_time": time.time() - model_started,
                }
            )
    started = time.time()
    with model_timeout(stack_timeout), limiter:
        stack.fit(x_final, y_final)
        proba = stack.predict_proba(x_test)
    elapsed = time.time() - started
    stack_model_path = None
    if save_models:
        if models_dir is None:
            models_dir = output_path.with_suffix("").parent / f"{output_path.stem}-models"
        models_dir.mkdir(parents=True, exist_ok=True)
        stack_model_path = models_dir / "stacking_classifier_tuned.joblib"
        joblib.dump(stack, stack_model_path)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError(f"Unexpected predict_proba shape: {proba.shape}")
    preds = np.asarray(proba[:, 1], dtype=float)
    if len(preds) != len(sample_submission):
        raise ValueError(
            f"Submission row count mismatch: {len(preds)} != {len(sample_submission)}"
        )
    if not np.isfinite(preds).all():
        raise ValueError("Submission predictions contain NaN or inf")

    submission = sample_submission.copy()
    target_col = submission.columns[1]
    submission[target_col] = preds
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return {
        "status": "ok",
        "path": str(output_path),
        "rows": int(len(submission)),
        "target_col": str(target_col),
        "prediction_min": float(preds.min()),
        "prediction_max": float(preds.max()),
        "prediction_mean": float(preds.mean()),
        "fit_rows": int(len(x_final)),
        "test_rows": int(len(x_test)),
        "time_taken": elapsed,
        "stack_models": stack_models,
        "model_dir": None if models_dir is None else str(models_dir),
        "saved_base_models": saved_base_models,
        "saved_stack_model": None if stack_model_path is None else str(stack_model_path),
    }


def positive_class_proba(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(x)
        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ValueError(f"Unexpected predict_proba shape: {proba.shape}")
        classes = list(getattr(estimator, "classes_", []))
        if 1 in classes:
            index = classes.index(1)
        elif "1" in classes:
            index = classes.index("1")
        else:
            index = 1
        return np.asarray(proba[:, index], dtype=float)
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    return np.asarray(estimator.predict(x), dtype=float)


def write_tuned_oof_predictions(
    tuned_scores: pd.DataFrame,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    x_test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_dir: Path,
    *,
    max_models: int | None,
    exclude_models: set[str],
    categorical_encoder: str,
    random_state: int,
    cv_folds: int,
    n_jobs: int,
    stack_top_k: int | None,
    native_boosting_data: bool,
    timeout: float | None,
    console: Console,
) -> dict[str, Any]:
    from lazypredict.preprocessing import prepare_dataframes

    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None

    x_final = pd.concat([x_train, x_valid], ignore_index=True)
    y_final = pd.concat(
        [pd.Series(y_train), pd.Series(y_valid)],
        ignore_index=True,
    )
    source = np.array(["train"] * len(x_train) + ["valid"] * len(x_valid))
    source_row = np.concatenate([np.arange(len(x_train)), np.arange(len(x_valid))])
    x_final, x_test = prepare_dataframes(x_final, x_test)
    estimators, stack_models = build_tuned_estimators(
        tuned_scores,
        x_final,
        max_models=max_models,
        exclude_models=exclude_models,
        categorical_encoder=categorical_encoder,
        random_state=random_state,
        n_jobs=n_jobs,
        native_boosting_data=native_boosting_data,
        top_k=stack_top_k,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    limiter = (
        threadpool_limits(limits=n_jobs)
        if threadpool_limits is not None
        else contextlib.nullcontext()
    )
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]
    manifest: list[dict[str, Any]] = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    with progress:
        task_id = progress.add_task(
            f"Saving tuned OOF/test predictions ({cv_folds}-fold)",
            total=len(estimators),
        )
        for (estimator_name, estimator), model_meta in zip(estimators, stack_models):
            started = time.time()
            try:
                oof = np.full(len(x_final), np.nan, dtype=float)
                for train_idx, valid_idx in cv.split(x_final, y_final):
                    fold_estimator = clone(estimator)
                    with model_timeout(timeout), limiter:
                        fold_estimator.fit(
                            x_final.iloc[train_idx],
                            y_final.iloc[train_idx],
                        )
                        oof[valid_idx] = positive_class_proba(
                            fold_estimator,
                            x_final.iloc[valid_idx],
                        )
                if not np.isfinite(oof).all():
                    raise ValueError("OOF predictions contain NaN or inf")
                full_estimator = clone(estimator)
                with model_timeout(timeout), limiter:
                    full_estimator.fit(x_final, y_final)
                    test_pred = positive_class_proba(full_estimator, x_test)
                if not np.isfinite(test_pred).all():
                    raise ValueError("Test predictions contain NaN or inf")

                oof_path = output_dir / f"{estimator_name}-oof.csv.gz"
                test_path = output_dir / f"{estimator_name}-test.csv.gz"
                pd.DataFrame(
                    {
                        "row": np.arange(len(x_final)),
                        "source": source,
                        "source_row": source_row,
                        "target": y_final.to_numpy(),
                        "prediction": oof,
                    }
                ).to_csv(oof_path, index=False, compression="gzip")
                pd.DataFrame(
                    {
                        id_col: sample_submission[id_col].to_numpy(),
                        target_col: test_pred,
                    }
                ).to_csv(test_path, index=False, compression="gzip")
                manifest.append(
                    {
                        "status": "ok",
                        "model": model_meta["name"],
                        "estimator_name": estimator_name,
                        "oof_path": str(oof_path),
                        "test_path": str(test_path),
                        "oof_roc_auc": float(roc_auc_score(y_final, oof)),
                        "rows": int(len(oof)),
                        "test_rows": int(len(test_pred)),
                        "time_taken": time.time() - started,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - keep other OOF models
                manifest.append(
                    {
                        "status": "error",
                        "model": model_meta["name"],
                        "estimator_name": estimator_name,
                        "error": repr(exc),
                        "time_taken": time.time() - started,
                    }
                )
            progress.update(task_id, advance=1)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    return {
        "status": "ok",
        "path": str(output_dir),
        "manifest_path": str(manifest_path),
        "models": manifest,
    }


def scores_dataframe(results: list[dict[str, Any]], *, score_metric: str) -> pd.DataFrame:
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "Model": metrics["name"],
                "Accuracy": metrics["accuracy"],
                "Balanced Accuracy": metrics["balanced_accuracy"],
                "ROC AUC": metrics["roc_auc"],
                "F1 Score": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Time Taken": metrics["time"],
            }
        )
    if not rows:
        return pd.DataFrame()
    scores = pd.DataFrame(rows).set_index("Model")
    column = metric_column(score_metric)
    return scores.sort_values(column, ascending=False, na_position="last")


def estimator_items(
    max_models: int | None,
    *,
    exclude_models: set[str],
    verbose: int = 0,
) -> list[tuple[str, Any]]:
    from lazypredict.Supervised import LazyClassifier

    clf = LazyClassifier(verbose=verbose)
    items = clf._get_estimator_list()
    if exclude_models:
        items = [(name, model) for name, model in items if name not in exclude_models]
    return items if max_models is None else items[:max_models]


def run_lazypredict_parallel(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    *,
    random_state: int,
    max_models: int | None,
    timeout: float | None,
    categorical_encoder: str,
    model_workers: int,
    n_jobs_per_model: int,
    eval_mode: str,
    cv_folds: int,
    native_boosting_data: bool,
    exclude_models: set[str],
    score_metric: str,
    verbose: int,
    console: Console,
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, Any]]]:
    items = estimator_items(
        max_models,
        exclude_models=exclude_models,
        verbose=verbose,
    )
    workers = max(1, min(int(model_workers), len(items)))
    results: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    failures: list[dict[str, Any]] = []

    if workers == 1:
        init_model_worker(
            x_train,
            x_valid,
            y_train,
            y_valid,
            categorical_encoder,
            random_state,
            n_jobs_per_model,
            timeout,
            eval_mode,
            cv_folds,
            native_boosting_data,
        )
        for item in items:
            result = fit_one_model(item)
            if result["status"] == "ok":
                results.append(result)
            else:
                errors[result["name"]] = result["error"]
                failures.append(
                    {
                        "name": result["name"],
                        "error": result["error"],
                        "time": result.get("time"),
                    }
                )
        return scores_dataframe(results, score_metric=score_metric), errors, failures

    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        disable=verbose <= 0,
    )
    with progress:
        task_id = progress.add_task(
            (
                f"LazyPredict models ({eval_mode}, {workers} workers, "
                f"{n_jobs_per_model} job/model)"
            ),
            total=len(items),
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_model_worker,
            initargs=(
                x_train,
                x_valid,
                y_train,
                y_valid,
                categorical_encoder,
                random_state,
                n_jobs_per_model,
                timeout,
                eval_mode,
                cv_folds,
                native_boosting_data,
            ),
        ) as executor:
            future_to_name = {
                executor.submit(fit_one_model, item): item[0]
                for item in items
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {"status": "error", "name": name, "error": repr(exc)}
                if result["status"] == "ok":
                    results.append(result)
                else:
                    errors[result["name"]] = result["error"]
                    failures.append(
                        {
                            "name": result["name"],
                            "error": result["error"],
                            "time": result.get("time"),
                        }
                    )
                progress.update(task_id, advance=1)
    return scores_dataframe(results, score_metric=score_metric), errors, failures


def run_lazy_tuning(
    scores: pd.DataFrame,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    *,
    max_models: int | None,
    exclude_models: set[str],
    categorical_encoder: str,
    random_state: int,
    tune_top_k: int,
    tune_trials: int,
    tune_timeout: float | None,
    tune_backend: str,
    tune_n_jobs: int,
    tune_metric: str,
    eval_mode: str,
    cv_folds: int,
    native_boosting_data: bool,
    verbose: int,
    console: Console,
) -> pd.DataFrame:
    from lazypredict.preprocessing import build_preprocessor, prepare_dataframes
    from lazypredict.search_spaces import get_search_space
    from lazypredict.tuning import (
        tune_supervised_flaml,
        tune_supervised_optuna,
        tune_supervised_sklearn,
    )
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        threadpool_limits = None

    estimator_lookup = dict(
        estimator_items(max_models, exclude_models=exclude_models)
    )
    top_model_names = [
        name
        for name in list(scores.index[:tune_top_k])
        if name in estimator_lookup
    ]
    x_train, x_valid = prepare_dataframes(x_train, x_valid)
    preprocessor = build_preprocessor(x_train, categorical_encoder)
    rows: list[dict[str, Any]] = []
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        disable=verbose <= 0,
    )
    with progress:
        task_id = progress.add_task(
            f"LazyPredict tuning top {len(top_model_names)} "
            f"({eval_mode}, {tune_backend}, {tune_trials} trials/model, "
            f"{tune_timeout:g}s/model, metric={tune_metric})",
            total=len(top_model_names),
        )
        for model_name in top_model_names:
            model_class = estimator_lookup[model_name]
            started = time.time()
            try:
                if get_search_space(model_name) is None:
                    rows.append(
                        {
                            "Model": model_name,
                            "Baseline ROC AUC": finite_or_none(scores.loc[model_name].get("ROC AUC")),
                            "Tuned ROC AUC": None,
                            "Delta ROC AUC": None,
                            "Baseline Balanced Accuracy": finite_or_none(scores.loc[model_name].get("Balanced Accuracy")),
                            "Tuned Balanced Accuracy": None,
                            "Delta Balanced Accuracy": None,
                            "Best Score": None,
                            "Best Params": "N/A (no search space)",
                            "Tune Time": time.time() - started,
                            "Search Time": None,
                            "Refit Time": None,
                            "Status": "skipped",
                            "Error": None,
                        }
                    )
                    progress.update(task_id, advance=1)
                    continue
                use_native = model_uses_native_data(model_name, native_boosting_data)
                if use_native and tune_backend != "optuna":
                    raise ValueError(
                        "Native boosting tuning currently supports only "
                        "--tune-backend optuna"
                    )
                if use_native:
                    import optuna

                    space_fn = get_search_space(model_name)

                    def objective(trial):
                        params = space_fn(trial)
                        kwargs = model_kwargs_for(
                            model_class,
                            random_state=random_state,
                            n_jobs_per_model=tune_n_jobs,
                        )
                        kwargs.update(params)
                        estimator = NativeBoostingClassifier(
                            model_class,
                            model_name,
                            kwargs,
                        )
                        if eval_mode == "cv":
                            cv = StratifiedKFold(
                                n_splits=cv_folds,
                                shuffle=True,
                                random_state=random_state,
                            )
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                return float(
                                    np.nanmean(
                                        cross_val_score(
                                            estimator,
                                            x_train,
                                            y_train,
                                            cv=cv,
                                            scoring=tune_metric,
                                            n_jobs=tune_n_jobs,
                                        )
                                    )
                                )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            estimator.fit(x_train, y_train)
                            metrics = compute_classifier_metrics(
                                estimator,
                                x_valid,
                                y_valid,
                            )
                        return metrics[tune_metric]

                    study = optuna.create_study(direction="maximize")
                    study.optimize(
                        objective,
                        n_trials=tune_trials,
                        timeout=tune_timeout,
                    )
                    best_params = study.best_params
                    best_score = study.best_value
                elif eval_mode == "valid":
                    if tune_backend != "optuna":
                        raise ValueError(
                            "--eval-mode valid tuning is currently supported only "
                            "with --tune-backend optuna"
                        )
                    import optuna

                    space_fn = get_search_space(model_name)

                    def objective(trial):
                        params = space_fn(trial)
                        kwargs = model_kwargs_for(
                            model_class,
                            random_state=random_state,
                            n_jobs_per_model=tune_n_jobs,
                        )
                        kwargs.update(params)
                        pipe = Pipeline(
                            steps=[
                                ("preprocessor", preprocessor),
                                ("classifier", model_class(**kwargs)),
                            ]
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            pipe.fit(x_train, y_train)
                            metrics = compute_classifier_metrics(
                                pipe,
                                x_valid,
                                y_valid,
                            )
                        return metrics[tune_metric]

                    study = optuna.create_study(direction="maximize")
                    study.optimize(
                        objective,
                        n_trials=tune_trials,
                        timeout=tune_timeout,
                    )
                    best_params = study.best_params
                    best_score = study.best_value
                elif tune_backend == "optuna":
                    best_params, best_score = tune_supervised_optuna(
                        model_name,
                        model_class,
                        x_train,
                        y_train,
                        preprocessor,
                        tune_metric,
                        cv=cv_folds,
                        n_trials=tune_trials,
                        timeout=tune_timeout,
                        random_state=random_state,
                        n_jobs=tune_n_jobs,
                        use_gpu=False,
                    )
                elif tune_backend == "sklearn":
                    best_params, best_score = tune_supervised_sklearn(
                        model_name,
                        model_class,
                        x_train,
                        y_train,
                        preprocessor,
                        tune_metric,
                        cv=cv_folds,
                        n_iter=tune_trials,
                        random_state=random_state,
                        n_jobs=tune_n_jobs,
                    )
                elif tune_backend == "flaml":
                    best_params, best_score = tune_supervised_flaml(
                        model_name,
                        model_class,
                        x_train,
                        y_train,
                        preprocessor,
                        tune_metric,
                        cv=cv_folds,
                        n_trials=tune_trials,
                        timeout=tune_timeout,
                        random_state=random_state,
                        n_jobs=tune_n_jobs,
                    )
                else:
                    raise ValueError(
                        "--tune-backend must be one of: optuna, sklearn, flaml"
                    )
                search_time = time.time() - started
                refit_started = time.time()
                pipe = build_estimator(
                    model_name,
                    model_class,
                    x_train,
                    categorical_encoder=categorical_encoder,
                    random_state=random_state,
                    n_jobs=tune_n_jobs,
                    native_boosting_data=native_boosting_data,
                )
                apply_tuned_params(pipe, best_params)
                limiter = (
                    threadpool_limits(limits=tune_n_jobs)
                    if threadpool_limits is not None
                    else contextlib.nullcontext()
                )
                with model_timeout(tune_timeout), limiter:
                    if eval_mode == "cv":
                        tuned_metrics = compute_classifier_cv_metrics(
                            pipe,
                            x_train,
                            y_train,
                            cv_folds=cv_folds,
                            random_state=random_state,
                        )
                    else:
                        pipe.fit(x_train, y_train)
                        tuned_metrics = compute_classifier_metrics(pipe, x_valid, y_valid)
                refit_time = time.time() - refit_started
                baseline_roc_auc = finite_or_none(scores.loc[model_name].get("ROC AUC"))
                tuned_roc_auc = finite_or_none(tuned_metrics.get("roc_auc"))
                baseline_bal_acc = finite_or_none(
                    scores.loc[model_name].get("Balanced Accuracy")
                )
                tuned_bal_acc = finite_or_none(tuned_metrics.get("balanced_accuracy"))
                rows.append(
                    {
                        "Model": model_name,
                        "Baseline ROC AUC": baseline_roc_auc,
                        "Tuned ROC AUC": tuned_roc_auc,
                        "Delta ROC AUC": (
                            None
                            if baseline_roc_auc is None or tuned_roc_auc is None
                            else tuned_roc_auc - baseline_roc_auc
                        ),
                        "Baseline Balanced Accuracy": baseline_bal_acc,
                        "Tuned Balanced Accuracy": tuned_bal_acc,
                        "Delta Balanced Accuracy": (
                            None
                            if baseline_bal_acc is None or tuned_bal_acc is None
                            else tuned_bal_acc - baseline_bal_acc
                        ),
                        "Best Score": best_score,
                        "Best Params": json.dumps(best_params, sort_keys=True),
                        "Tune Time": time.time() - started,
                        "Search Time": search_time,
                        "Refit Time": refit_time,
                        "Status": "ok",
                        "Error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - tuning should not kill preview
                baseline_roc_auc = finite_or_none(scores.loc[model_name].get("ROC AUC"))
                baseline_bal_acc = finite_or_none(
                    scores.loc[model_name].get("Balanced Accuracy")
                )
                rows.append(
                    {
                        "Model": model_name,
                        "Baseline ROC AUC": baseline_roc_auc,
                        "Tuned ROC AUC": None,
                        "Delta ROC AUC": None,
                        "Baseline Balanced Accuracy": baseline_bal_acc,
                        "Tuned Balanced Accuracy": None,
                        "Delta Balanced Accuracy": None,
                        "Best Score": None,
                        "Best Params": None,
                        "Tune Time": time.time() - started,
                        "Search Time": None,
                        "Refit Time": None,
                        "Status": "error",
                        "Error": repr(exc),
                    }
                )
            progress.update(task_id, advance=1)
    return pd.DataFrame(rows).set_index("Model") if rows else pd.DataFrame()


def render_selected(console: Console, records: list[dict[str, Any]]) -> None:
    table = Table(title="LazyPredict preview inputs")
    table.add_column("#", justify="right")
    table.add_column("cv", justify="right")
    table.add_column("run")
    table.add_column("step", justify="right")
    table.add_column("sha")
    table.add_column("solution")
    for idx, record in enumerate(records, start=1):
        table.add_row(
            str(idx),
            f"{float(record.get('local_score')):.6f}",
            str(record.get("run") or "-"),
            str(record.get("step") if record.get("step") is not None else "-"),
            str(record.get("sha256") or "")[:10],
            str(record.get("solution_path") or "-"),
        )
    console.print(table)


def render_result_preview(console: Console, rows: list[dict[str, Any]], top_models: int) -> None:
    table = Table(title="LazyPredict top models")
    table.add_column("#", justify="right")
    table.add_column("cv", justify="right")
    table.add_column("step", justify="right")
    table.add_column("sha")
    table.add_column("model")
    table.add_column("roc_auc", justify="right")
    table.add_column("bal_acc", justify="right")
    table.add_column("time", justify="right")
    for row in rows:
        if int(row["model_rank"]) > top_models:
            continue
        table.add_row(
            str(row["candidate_rank"]),
            f"{float(row['local_score']):.6f}",
            str(row.get("step") or "-"),
            str(row.get("sha256") or "")[:10],
            str(row["model"]),
            "" if row.get("roc_auc") is None else f"{float(row['roc_auc']):.5f}",
            "" if row.get("balanced_accuracy") is None else f"{float(row['balanced_accuracy']):.5f}",
            "" if row.get("time_taken") is None else f"{float(row['time_taken']):.2f}s",
        )
    console.print(table)


def render_slowest_models(
    console: Console,
    rows: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> None:
    slow_rows = sorted(
        [row for row in rows if row.get("time_taken") is not None],
        key=lambda row: float(row["time_taken"]),
        reverse=True,
    )[:limit]
    if not slow_rows:
        return
    table = Table(title="Slowest successful models")
    table.add_column("#", justify="right")
    table.add_column("step", justify="right")
    table.add_column("sha")
    table.add_column("model")
    table.add_column("time", justify="right")
    table.add_column("roc_auc", justify="right")
    for row in slow_rows:
        table.add_row(
            str(row["candidate_rank"]),
            str(row.get("step") or "-"),
            str(row.get("sha256") or "")[:10],
            str(row.get("model") or "-"),
            f"{float(row['time_taken']):.2f}s",
            "" if row.get("roc_auc") is None else f"{float(row['roc_auc']):.5f}",
        )
    console.print(table)


def render_failed_models(
    console: Console,
    failures: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> None:
    if not failures:
        return
    table = Table(title="Failed or timed out models")
    table.add_column("#", justify="right")
    table.add_column("step", justify="right")
    table.add_column("sha")
    table.add_column("model")
    table.add_column("time", justify="right")
    table.add_column("error")
    for row in failures[:limit]:
        error = str(row.get("error") or "")
        table.add_row(
            str(row.get("candidate_rank") or "-"),
            str(row.get("step") or "-"),
            str(row.get("sha256") or "")[:10],
            str(row.get("model") or "-"),
            "" if row.get("time_taken") is None else f"{float(row['time_taken']):.2f}s",
            error[:140],
        )
    console.print(table)


def render_tuning_results(
    console: Console,
    rows: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> None:
    if not rows:
        return
    table = Table(title="LazyPredict tuning results")
    table.add_column("#", justify="right")
    table.add_column("step", justify="right")
    table.add_column("sha")
    table.add_column("model")
    table.add_column("roc_before", justify="right")
    table.add_column("roc_after", justify="right")
    table.add_column("d_roc", justify="right")
    table.add_column("bal_before", justify="right")
    table.add_column("bal_after", justify="right")
    table.add_column("d_bal", justify="right")
    table.add_column("cv_best", justify="right")
    table.add_column("time", justify="right")
    table.add_column("status")
    for row in rows[:limit]:
        table.add_row(
            str(row.get("tune_rank") or "-"),
            str(row.get("step") or "-"),
            str(row.get("sha256") or "")[:10],
            str(row.get("model") or "-"),
            "" if row.get("baseline_roc_auc") is None else f"{float(row['baseline_roc_auc']):.5f}",
            "" if row.get("tuned_roc_auc") is None else f"{float(row['tuned_roc_auc']):.5f}",
            "" if row.get("delta_roc_auc") is None else f"{float(row['delta_roc_auc']):+.5f}",
            "" if row.get("baseline_balanced_accuracy") is None else f"{float(row['baseline_balanced_accuracy']):.5f}",
            "" if row.get("tuned_balanced_accuracy") is None else f"{float(row['tuned_balanced_accuracy']):.5f}",
            "" if row.get("delta_balanced_accuracy") is None else f"{float(row['delta_balanced_accuracy']):+.5f}",
            "" if row.get("best_score") is None else f"{float(row['best_score']):.5f}",
            "" if row.get("tune_time") is None else f"{float(row['tune_time']):.2f}s",
            str(row.get("status") or "-"),
        )
    console.print(table)


def print_compact_diagnostics(
    console: Console,
    summary_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
) -> None:
    slowest = sorted(
        [row for row in summary_rows if row.get("time_taken") is not None],
        key=lambda row: float(row["time_taken"]),
        reverse=True,
    )[:5]
    for row in slowest:
        roc_auc = ""
        if row.get("roc_auc") is not None:
            roc_auc = f"{float(row['roc_auc']):.5f}"
        console.print(
            "slowest: "
            f"candidate={row.get('candidate_rank')} "
            f"step={row.get('step')} "
            f"sha={str(row.get('sha256') or '')[:10]} "
            f"model={row.get('model')} "
            f"time={float(row['time_taken']):.2f}s "
            f"roc_auc={roc_auc}"
        )
    if failure_rows:
        console.print(f"failures: {len(failure_rows)} model(s)")
        for row in failure_rows[:10]:
            time_taken = ""
            if row.get("time_taken") is not None:
                time_taken = f"{float(row['time_taken']):.2f}s"
            console.print(
                "failed: "
                f"candidate={row.get('candidate_rank')} "
                f"step={row.get('step')} "
                f"sha={str(row.get('sha256') or '')[:10]} "
                f"model={row.get('model')} "
                f"time={time_taken} "
                f"error={str(row.get('error') or '')[:180]}"
            )
    else:
        console.print("failures: 0 model(s)")


def print_compact_tuning(
    console: Console,
    tuning_rows: list[dict[str, Any]],
) -> None:
    for row in tuning_rows:
        roc_before = row.get("baseline_roc_auc")
        roc_after = row.get("tuned_roc_auc")
        d_roc = row.get("delta_roc_auc")
        bal_before = row.get("baseline_balanced_accuracy")
        bal_after = row.get("tuned_balanced_accuracy")
        d_bal = row.get("delta_balanced_accuracy")
        cv_best = row.get("best_score")
        console.print(
            "tuned: "
            f"candidate={row.get('candidate_rank')} "
            f"step={row.get('step')} "
            f"sha={str(row.get('sha256') or '')[:10]} "
            f"model={row.get('model')} "
            f"roc_auc={'' if roc_before is None else f'{float(roc_before):.5f}'}"
            f"->{'' if roc_after is None else f'{float(roc_after):.5f}'} "
            f"d_roc={'' if d_roc is None else f'{float(d_roc):+.5f}'} "
            f"bal_acc={'' if bal_before is None else f'{float(bal_before):.5f}'}"
            f"->{'' if bal_after is None else f'{float(bal_after):.5f}'} "
            f"d_bal={'' if d_bal is None else f'{float(d_bal):+.5f}'} "
            f"cv_best={'' if cv_best is None else f'{float(cv_best):.5f}'} "
            f"status={row.get('status')}"
        )


def print_compact_submissions(console: Console, details: list[dict[str, Any]]) -> None:
    for detail in details:
        submission = detail.get("submission")
        if not isinstance(submission, dict):
            continue
        if submission.get("status") == "ok":
            console.print(
                "submission: "
                f"{submission.get('path')} "
                f"rows={submission.get('rows')} "
                f"fit_rows={submission.get('fit_rows')}"
            )
            if submission.get("model_dir"):
                console.print(f"models: {submission.get('model_dir')}")
            if submission.get("saved_stack_model"):
                console.print(f"stack model: {submission.get('saved_stack_model')}")
        else:
            console.print(
                "submission failed: "
                f"{submission.get('error') or 'unknown error'}"
            )


def print_compact_oof_predictions(console: Console, details: list[dict[str, Any]]) -> None:
    for detail in details:
        oof_predictions = detail.get("oof_predictions")
        if not isinstance(oof_predictions, dict):
            continue
        if oof_predictions.get("status") == "ok":
            ok_models = [
                model
                for model in oof_predictions.get("models", [])
                if model.get("status") == "ok"
            ]
            console.print(
                "oof predictions: "
                f"{oof_predictions.get('path')} "
                f"models={len(ok_models)} "
                f"manifest={oof_predictions.get('manifest_path')}"
            )
        else:
            console.print(
                "oof predictions failed: "
                f"{oof_predictions.get('error') or 'unknown error'}"
            )


def finite_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def target_sample_metadata(y_train: pd.Series, y_valid: pd.Series) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if pd.api.types.is_numeric_dtype(y_train):
        metadata["target_mean_train"] = float(pd.Series(y_train).mean())
        metadata["target_mean_valid"] = float(pd.Series(y_valid).mean())
    else:
        metadata["target_mean_train"] = None
        metadata["target_mean_valid"] = None
    metadata["target_counts_train"] = {
        str(key): int(value)
        for key, value in pd.Series(y_train).value_counts(dropna=False).items()
    }
    metadata["target_counts_valid"] = {
        str(key): int(value)
        for key, value in pd.Series(y_valid).value_counts(dropna=False).items()
    }
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    project_env = load_project_env()
    project_name = project_env["project_name"] or DEFAULT_PROJECT_NAME
    data_dir = Path(project_env["data_dir"]) if project_env["data_dir"] else None
    metric = (
        normalize_metric(project_env["project_metric"])
        if project_env["project_metric"]
        else DEFAULT_TUNE_METRIC
    )
    parser = argparse.ArgumentParser(
        description="Run a quick LazyPredict benchmark over top AIDE preprocessing artifacts.",
    )
    parser.add_argument("--index", type=Path, default=lab.DEFAULT_INDEX_PATH)
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--project", "--competition", dest="project", default=project_name)
    parser.add_argument("--run")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--sha256", "--sha", dest="sha256", action="append", default=[], metavar="PREFIX"
    )
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sample-train",
        type=int,
        default=None,
        help="Rows to keep for model training. Defaults to full data with --valid-fraction.",
    )
    parser.add_argument(
        "--sample-valid",
        type=int,
        default=None,
        help="Rows to keep for validation. Defaults to full data with --valid-fraction.",
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.2,
        help="Validation fraction when using full data or when one sample size is omitted.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["valid", "cv"],
        default="valid",
        help="Score models on a validation split or with cross-validation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds when --eval-mode cv is used. Defaults to 5.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--metric",
        type=normalize_metric,
        choices=sorted(SUPPORTED_METRICS),
        default=metric,
        help=(
            "Metric used to rank LazyPredict models. Defaults to "
            "AIDE_PROJECT_METRIC from .env."
        ),
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=0,
        help="Maximum number of LazyPredict models. 0 means all models.",
    )
    parser.add_argument(
        "--model-timeout",
        type=float,
        default=600.0,
        help="Per-model timeout in seconds.",
    )
    parser.add_argument(
        "--model-workers",
        type=int,
        default=16,
        help="Number of models to train in parallel.",
    )
    parser.add_argument(
        "--n-jobs-per-model",
        type=int,
        default=1,
        help="n_jobs/thread_count passed to each model when supported.",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=[],
        metavar="NAME",
        help="Additional LazyPredict model name to skip. Can be repeated.",
    )
    parser.add_argument(
        "--include-problem-models",
        action="store_true",
        help="Do not skip the default known-problem LazyPredict models.",
    )
    parser.add_argument(
        "--native-boosting-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For CatBoostClassifier, LGBMClassifier, and XGBClassifier, skip "
            "the shared sklearn imputer/encoder/scaler and pass native pandas "
            "data with categorical dtype handling. Enabled by default."
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            "After the LazyPredict sweep, tune top models with LazyPredict's "
            "HPO backend. Requires explicit --tune-timeout."
        ),
    )
    parser.add_argument("--tune-top-k", type=int, default=DEFAULT_TUNE_TOP_K)
    parser.add_argument("--tune-trials", type=int, default=DEFAULT_TUNE_TRIALS)
    parser.add_argument(
        "--tune-timeout",
        type=float,
        default=DEFAULT_TUNE_TIMEOUT,
        help=(
            "Optuna timeout per tuned model. Required with --tune. "
            "This is checked between trials; one CV trial can exceed it."
        ),
    )
    parser.add_argument("--tune-backend", default=DEFAULT_TUNE_BACKEND)
    parser.add_argument(
        "--tune-metric",
        type=normalize_metric,
        choices=sorted(SUPPORTED_METRICS),
        default=metric,
        help="Metric optimized during tuning. Defaults to AIDE_PROJECT_METRIC from .env.",
    )
    parser.add_argument(
        "--tune-n-jobs",
        type=int,
        default=DEFAULT_TUNE_N_JOBS,
        help="Parallel jobs used by tuning CV. Defaults to 16.",
    )
    parser.add_argument(
        "--stack-tuned",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After --tune, stack tuned models. Enabled by default.",
    )
    parser.add_argument(
        "--stack-tuned-top-k",
        type=int,
        default=0,
        help="How many tuned models to stack. 0 means all successful tuned models.",
    )
    parser.add_argument(
        "--stack-cv-folds",
        type=int,
        default=5,
        help="Internal StackingClassifier CV folds.",
    )
    parser.add_argument(
        "--stack-n-jobs",
        type=int,
        default=DEFAULT_TUNE_N_JOBS,
        help="n_jobs for StackingClassifier and stacked base models.",
    )
    parser.add_argument(
        "--stack-timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds for stacking.",
    )
    parser.add_argument(
        "--submission",
        action="store_true",
        help=(
            "After tuning, fit the tuned stack on sample_train+sample_valid "
            "and write a submission CSV for the full test set."
        ),
    )
    parser.add_argument(
        "--submission-output",
        type=Path,
        help="Optional explicit submission CSV path.",
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --submission, save fitted tuned base models and the fitted "
            "stacking model as joblib files. Enabled by default."
        ),
    )
    parser.add_argument(
        "--save-oof-preds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --tune, save leak-free OOF predictions and full-test "
            "predictions for tuned models. Enabled by default."
        ),
    )
    parser.add_argument(
        "--oof-dir",
        type=Path,
        help="Optional directory for saved OOF/test prediction CSV files.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="Optional directory for saved joblib models.",
    )
    parser.add_argument(
        "--lazy-verbose",
        type=int,
        default=1,
        help="LazyPredict verbosity. Use 0 to hide per-model progress.",
    )
    parser.add_argument("--preprocess-timeout", type=int, default=240)
    parser.add_argument("--categorical-encoder", default="onehot")
    parser.add_argument("--top-models", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Base artifact directory. A timestamped run directory is created inside it.",
    )
    parser.add_argument("--save-preprocessed", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    if args.tune and args.tune_timeout is None:
        console.print(
            "[red]--tune requires explicit --tune-timeout.[/red] "
            "This avoids silently using a bad timeout for CV tuning."
        )
        return 2
    if args.submission and not args.tune:
        console.print("[red]--submission requires --tune.[/red]")
        return 2
    if args.eval_mode == "cv" and args.cv_folds < 2:
        console.print("[red]--cv-folds must be at least 2.[/red]")
        return 2
    if args.stack_cv_folds < 2:
        console.print("[red]--stack-cv-folds must be at least 2.[/red]")
        return 2
    index = lab._load_json(args.index)
    if not index:
        console.print(f"[red]Missing submission index: {args.index}[/red]")
        return 2

    records = index.get("records", [])
    if args.sha256:
        try:
            sha_filters = lab.parse_sha256_filters(args.sha256)
            selected = lab.filter_records_by_sha256(records, sha_filters)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2
    else:
        selected = select_records(
            records,
            run=args.run,
            competition=args.project,
            limit=args.limit,
            dedupe=bool(args.dedupe),
        )
    if not selected:
        console.print("[red]No matching records found.[/red]")
        return 2

    max_models = None if args.max_models <= 0 else args.max_models
    exclude_models = set(args.exclude_model or [])
    if not args.include_problem_models:
        exclude_models |= DEFAULT_EXCLUDED_MODELS
    run_id = timestamp_now()
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.csv.gz"
    tuning_path = run_dir / "tuning.csv.gz"
    details_path = run_dir / "details.json"
    console.print(f"artifacts: {run_dir}")
    console.print(
        f"project: {args.project} data_dir: {args.data_dir or DEFAULT_DATA_DIR} "
        f"metric: {args.metric}"
    )
    render_selected(console, selected)
    if exclude_models:
        console.print(
            "Skipping models: "
            + ", ".join(sorted(exclude_models))
        )

    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for candidate_rank, record in enumerate(selected, start=1):
        label = (
            f"#{candidate_rank} step={record.get('step')} "
            f"sha={str(record.get('sha256') or '')[:10]}"
        )
        console.print(f"[bold]Running {label}[/bold]")
        detail: dict[str, Any] = {
            "candidate_rank": candidate_rank,
            "record": record,
            "status": "error",
        }
        detail_saved = False
        try:
            data_dir = resolve_data_dir(record, args.data_dir)
            train_fe, test_fe, y, sample_submission, prep_meta = run_artifact_preprocess(
                record,
                data_dir=data_dir,
                preprocess_time_limit=args.preprocess_timeout,
            )
            all_cleaned, clean_meta = clean_for_lazypredict(
                pd.concat([train_fe, test_fe], ignore_index=True)
            )
            cleaned = all_cleaned.iloc[: len(train_fe)].reset_index(drop=True)
            test_cleaned = all_cleaned.iloc[len(train_fe):].reset_index(drop=True)
            x_train, x_valid, y_train, y_valid, sample_meta = sample_train_valid(
                cleaned,
                y,
                sample_train=args.sample_train,
                sample_valid=args.sample_valid,
                valid_fraction=args.valid_fraction,
                random_state=args.random_state,
            )
            if args.save_preprocessed:
                preprocessed_path = (
                    run_dir
                    / f"rank{candidate_rank:02d}-step{record.get('step')}-features.parquet"
                )
                cleaned.to_parquet(preprocessed_path)
                detail["preprocessed_path"] = str(preprocessed_path)

            started = time.time()
            scores, errors, failures = run_lazypredict_parallel(
                x_train,
                x_valid,
                y_train,
                y_valid,
                random_state=args.random_state,
                max_models=max_models,
                timeout=args.model_timeout,
                categorical_encoder=args.categorical_encoder,
                model_workers=args.model_workers,
                n_jobs_per_model=args.n_jobs_per_model,
                eval_mode=args.eval_mode,
                cv_folds=args.cv_folds,
                native_boosting_data=args.native_boosting_data,
                exclude_models=exclude_models,
                score_metric=args.metric,
                verbose=args.lazy_verbose,
                console=console,
            )
            elapsed = time.time() - started
            scores_reset = scores.reset_index()
            detail.update(
                {
                    "status": "ok",
                    "preprocess": prep_meta,
                    "cleaning": clean_meta,
                    "lazy_elapsed": elapsed,
                    "score_metric": args.metric,
                    "sample": {
                        **sample_meta,
                        **target_sample_metadata(y_train, y_valid),
                    },
                    "model_errors": errors,
                    "model_failures": failures,
                    "scores": scores_reset.to_dict(orient="records"),
                }
            )
            candidate_failure_rows: list[dict[str, Any]] = []
            for failure in failures:
                candidate_failure_rows.append(
                    {
                        "candidate_rank": candidate_rank,
                        "run": record.get("run"),
                        "step": record.get("step"),
                        "timestamp": record.get("timestamp"),
                        "local_score": record.get("local_score"),
                        "sha256": record.get("sha256"),
                        "solution_path": record.get("solution_path"),
                        "model": failure.get("name"),
                        "time_taken": finite_or_none(failure.get("time")),
                        "error": failure.get("error"),
                    }
                )
            failure_rows.extend(candidate_failure_rows)

            candidate_summary_rows: list[dict[str, Any]] = []
            for model_rank, score_row in enumerate(
                scores_reset.to_dict(orient="records"),
                start=1,
            ):
                candidate_summary_rows.append(
                    {
                        "candidate_rank": candidate_rank,
                        "model_rank": model_rank,
                        "run": record.get("run"),
                        "step": record.get("step"),
                        "timestamp": record.get("timestamp"),
                        "local_score": record.get("local_score"),
                        "sha256": record.get("sha256"),
                        "solution_path": record.get("solution_path"),
                        "model": score_row.get("Model"),
                        "roc_auc": finite_or_none(score_row.get("ROC AUC")),
                        "balanced_accuracy": finite_or_none(score_row.get("Balanced Accuracy")),
                        "accuracy": finite_or_none(score_row.get("Accuracy")),
                        "f1": finite_or_none(score_row.get("F1 Score")),
                        "precision": finite_or_none(score_row.get("Precision")),
                        "recall": finite_or_none(score_row.get("Recall")),
                        "time_taken": finite_or_none(score_row.get("Time Taken")),
                        "preprocessed_columns": prep_meta["preprocessed_columns"],
                        "null_cells": clean_meta["null_cells"],
                        "inf_cells_replaced": clean_meta["inf_cells_replaced"],
                        "lazy_elapsed": elapsed,
                    }
                )
            summary_rows.extend(candidate_summary_rows)

            console.print(
                f"  ok: cols={prep_meta['preprocessed_columns']} "
                f"sample={len(x_train)}/{len(x_valid)} models={len(scores)}"
            )
            if candidate_summary_rows:
                render_result_preview(console, candidate_summary_rows, args.top_models)
                render_slowest_models(console, candidate_summary_rows)
            render_failed_models(console, candidate_failure_rows)
            print_compact_diagnostics(
                console,
                candidate_summary_rows,
                candidate_failure_rows,
            )
            details.append(detail)
            detail_saved = True
            write_outputs(
                summary_path=summary_path,
                tuning_path=tuning_path,
                details_path=details_path,
                summary_rows=summary_rows,
                tuning_rows=tuning_rows,
                details=details,
            )
            console.print(f"partial summary: {summary_path}")
            console.print(f"partial details: {details_path}")
            if args.tune:
                tuning_started = time.time()
                tuned_scores = run_lazy_tuning(
                    scores,
                    x_train,
                    x_valid,
                    y_train,
                    y_valid,
                    max_models=max_models,
                    exclude_models=exclude_models,
                    categorical_encoder=args.categorical_encoder,
                    random_state=args.random_state,
                    tune_top_k=args.tune_top_k,
                    tune_trials=args.tune_trials,
                    tune_timeout=args.tune_timeout,
                    tune_backend=args.tune_backend,
                    tune_n_jobs=args.tune_n_jobs,
                    tune_metric=args.tune_metric,
                    eval_mode=args.eval_mode,
                    cv_folds=args.cv_folds,
                    native_boosting_data=args.native_boosting_data,
                    verbose=args.lazy_verbose,
                    console=console,
                )
                tuning_elapsed = time.time() - tuning_started
                tuned_reset = tuned_scores.reset_index()
                detail["tuning"] = {
                    "elapsed": tuning_elapsed,
                    "top_k": args.tune_top_k,
                    "trials": args.tune_trials,
                    "timeout": args.tune_timeout,
                    "backend": args.tune_backend,
                    "metric": args.tune_metric,
                    "n_jobs": args.tune_n_jobs,
                    "scores": tuned_reset.to_dict(orient="records"),
                }
                for tune_rank, tune_row in enumerate(
                    tuned_reset.to_dict(orient="records"),
                    start=1,
                ):
                    tuning_rows.append(
                        {
                            "candidate_rank": candidate_rank,
                            "tune_rank": tune_rank,
                            "run": record.get("run"),
                            "step": record.get("step"),
                            "timestamp": record.get("timestamp"),
                            "local_score": record.get("local_score"),
                            "sha256": record.get("sha256"),
                            "solution_path": record.get("solution_path"),
                            "model": tune_row.get("Model"),
                            "baseline_roc_auc": finite_or_none(tune_row.get("Baseline ROC AUC")),
                            "tuned_roc_auc": finite_or_none(tune_row.get("Tuned ROC AUC")),
                            "delta_roc_auc": finite_or_none(tune_row.get("Delta ROC AUC")),
                            "baseline_balanced_accuracy": finite_or_none(
                                tune_row.get("Baseline Balanced Accuracy")
                            ),
                            "tuned_balanced_accuracy": finite_or_none(
                                tune_row.get("Tuned Balanced Accuracy")
                            ),
                            "delta_balanced_accuracy": finite_or_none(
                                tune_row.get("Delta Balanced Accuracy")
                            ),
                            "best_score": finite_or_none(tune_row.get("Best Score")),
                            "best_params": tune_row.get("Best Params"),
                            "tune_time": finite_or_none(tune_row.get("Tune Time")),
                            "search_time": finite_or_none(tune_row.get("Search Time")),
                            "refit_time": finite_or_none(tune_row.get("Refit Time")),
                            "status": tune_row.get("Status"),
                            "error": tune_row.get("Error"),
                            "tune_elapsed": tuning_elapsed,
                        }
                    )
                write_outputs(
                    summary_path=summary_path,
                    tuning_path=tuning_path,
                    details_path=details_path,
                    summary_rows=summary_rows,
                    tuning_rows=tuning_rows,
                    details=details,
                )
                render_tuning_results(console, tuning_rows)
                console.print(f"partial tuning: {tuning_path}")
                if args.save_oof_preds:
                    try:
                        if args.oof_dir is not None:
                            oof_dir = args.oof_dir
                            if len(selected) > 1:
                                oof_dir = (
                                    oof_dir
                                    / f"rank{candidate_rank:02d}-step{record.get('step')}"
                                )
                        else:
                            oof_dir = (
                                run_dir
                                / f"rank{candidate_rank:02d}-step{record.get('step')}-oof-preds"
                            )
                        oof_predictions = write_tuned_oof_predictions(
                            tuned_scores,
                            x_train,
                            x_valid,
                            y_train,
                            y_valid,
                            test_cleaned,
                            sample_submission,
                            oof_dir,
                            max_models=max_models,
                            exclude_models=exclude_models,
                            categorical_encoder=args.categorical_encoder,
                            random_state=args.random_state,
                            cv_folds=args.cv_folds,
                            n_jobs=args.stack_n_jobs,
                            timeout=args.stack_timeout,
                            native_boosting_data=args.native_boosting_data,
                            stack_top_k=(
                                None
                                if args.stack_tuned_top_k <= 0
                                else args.stack_tuned_top_k
                            ),
                            console=console,
                        )
                        detail["oof_predictions"] = oof_predictions
                        console.print(
                            "oof predictions: "
                            f"{oof_predictions['path']} "
                            f"manifest={oof_predictions['manifest_path']}"
                        )
                    except Exception as exc:  # noqa: BLE001
                        detail["oof_predictions"] = {
                            "status": "error",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        console.print(f"  [red]oof predictions failed:[/red] {exc!r}")
                    write_outputs(
                        summary_path=summary_path,
                        tuning_path=tuning_path,
                        details_path=details_path,
                        summary_rows=summary_rows,
                        tuning_rows=tuning_rows,
                        details=details,
                    )
                if args.stack_tuned:
                    stacking_started = time.time()
                    try:
                        stacking = run_tuned_stacking(
                            tuned_scores,
                            x_train,
                            x_valid,
                            y_train,
                            y_valid,
                            max_models=max_models,
                            exclude_models=exclude_models,
                            categorical_encoder=args.categorical_encoder,
                            random_state=args.random_state,
                            eval_mode=args.eval_mode,
                            cv_folds=args.cv_folds,
                            stack_cv_folds=args.stack_cv_folds,
                            stack_n_jobs=args.stack_n_jobs,
                            stack_timeout=args.stack_timeout,
                            native_boosting_data=args.native_boosting_data,
                            stack_top_k=(
                                None
                                if args.stack_tuned_top_k <= 0
                                else args.stack_tuned_top_k
                            ),
                            console=console,
                        )
                        stacking["elapsed"] = time.time() - stacking_started
                        detail["stacking"] = stacking
                        stack_summary_row = {
                            "candidate_rank": candidate_rank,
                            "model_rank": len(candidate_summary_rows) + 1,
                            "run": record.get("run"),
                            "step": record.get("step"),
                            "timestamp": record.get("timestamp"),
                            "local_score": record.get("local_score"),
                            "sha256": record.get("sha256"),
                            "solution_path": record.get("solution_path"),
                            "model": stacking["model"],
                            "roc_auc": stacking["roc_auc"],
                            "balanced_accuracy": stacking["balanced_accuracy"],
                            "accuracy": stacking["accuracy"],
                            "f1": stacking["f1"],
                            "precision": stacking["precision"],
                            "recall": stacking["recall"],
                            "time_taken": stacking["time_taken"],
                            "preprocessed_columns": prep_meta["preprocessed_columns"],
                            "null_cells": clean_meta["null_cells"],
                            "inf_cells_replaced": clean_meta["inf_cells_replaced"],
                            "lazy_elapsed": elapsed,
                        }
                        summary_rows.append(stack_summary_row)
                        candidate_summary_rows.append(stack_summary_row)
                        console.print(
                            "stacked: "
                            f"roc_auc={'' if stacking['roc_auc'] is None else f'{float(stacking['roc_auc']):.5f}'} "
                            f"bal_acc={'' if stacking['balanced_accuracy'] is None else f'{float(stacking['balanced_accuracy']):.5f}'} "
                            f"time={float(stacking['time_taken']):.2f}s"
                        )
                    except Exception as exc:  # noqa: BLE001
                        detail["stacking"] = {
                            "status": "error",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        console.print(f"  [red]stacking failed:[/red] {exc!r}")
                if args.submission:
                    try:
                        if args.submission_output is not None:
                            submission_path = args.submission_output
                        else:
                            submission_path = (
                                run_dir
                                / (
                                    f"rank{candidate_rank:02d}"
                                    f"-step{record.get('step')}-submission.csv"
                                )
                            )
                        models_dir = args.models_dir
                        if models_dir is None:
                            models_dir = (
                                run_dir
                                / f"rank{candidate_rank:02d}-step{record.get('step')}-models"
                            )
                        submission = write_tuned_stack_submission(
                            tuned_scores,
                            x_train,
                            x_valid,
                            y_train,
                            y_valid,
                            test_cleaned,
                            sample_submission,
                            submission_path,
                            models_dir=models_dir,
                            save_models=args.save_models,
                            max_models=max_models,
                            exclude_models=exclude_models,
                            categorical_encoder=args.categorical_encoder,
                            random_state=args.random_state,
                            stack_cv_folds=args.stack_cv_folds,
                            stack_n_jobs=args.stack_n_jobs,
                            stack_timeout=args.stack_timeout,
                            native_boosting_data=args.native_boosting_data,
                            stack_top_k=(
                                None
                                if args.stack_tuned_top_k <= 0
                                else args.stack_tuned_top_k
                            ),
                        )
                        detail["submission"] = submission
                        console.print(
                            "submission: "
                            f"{submission['path']} "
                            f"rows={submission['rows']} "
                            f"fit_rows={submission['fit_rows']} "
                            f"pred_min={submission['prediction_min']:.6f} "
                            f"pred_max={submission['prediction_max']:.6f} "
                            f"pred_mean={submission['prediction_mean']:.6f}"
                        )
                        if submission.get("model_dir"):
                            console.print(f"models: {submission['model_dir']}")
                        if submission.get("saved_stack_model"):
                            console.print(f"stack model: {submission['saved_stack_model']}")
                    except Exception as exc:  # noqa: BLE001
                        detail["submission"] = {
                            "status": "error",
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                        console.print(f"  [red]submission failed:[/red] {exc!r}")
                write_outputs(
                    summary_path=summary_path,
                    tuning_path=tuning_path,
                    details_path=details_path,
                    summary_rows=summary_rows,
                    tuning_rows=tuning_rows,
                    details=details,
                )
        except Exception as exc:  # noqa: BLE001 - preview should continue across artifacts
            detail.update(
                {
                    "status": "error",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            console.print(f"  [red]failed:[/red] {exc!r}")
            if not detail_saved:
                details.append(detail)

    write_outputs(
        summary_path=summary_path,
        tuning_path=tuning_path,
        details_path=details_path,
        summary_rows=summary_rows,
        tuning_rows=tuning_rows,
        details=details,
    )
    if summary_rows:
        render_result_preview(console, summary_rows, args.top_models)
        render_slowest_models(console, summary_rows)
    render_failed_models(console, failure_rows)
    render_tuning_results(console, tuning_rows)
    print_compact_diagnostics(console, summary_rows, failure_rows)
    print_compact_tuning(console, tuning_rows)
    print_compact_oof_predictions(console, details)
    print_compact_submissions(console, details)
    console.print(f"summary: {summary_path}")
    if tuning_rows:
        console.print(f"tuning: {tuning_path}")
    console.print(f"details: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
