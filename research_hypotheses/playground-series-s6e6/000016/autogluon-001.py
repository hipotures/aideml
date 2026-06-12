from __future__ import annotations

import json
import shutil
import warnings
import contextlib
import inspect
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

AIDE_AG_CONFIG = {'aux_file': 'star_classification.csv',
 'class_balance': 'balanced',
 'eval_metric': 'balanced_accuracy',
 'fit_args': {'auto_stack': False, 'fit_weighted_ensemble': True, 'save_space': True},
 'hyperparameters': {'CAT': [{'ag_args_fit': {'num_gpus': 1},
                              'devices': '0',
                              'gpu_ram_part': 0.8,
                              'task_type': 'GPU'}],
                     'GBM': [{'ag_args_fit': {'num_gpus': 1}, 'device': 'cuda'}],
                     'XGB': [{'ag_args': {'priority': 999},
                              'ag_args_fit': {'num_gpus': 1},
                              'device': 'cuda',
                              'tree_method': 'hist'}]},
 'included_model_types': ['XGB', 'GBM', 'CAT'],
 'preprocess_timeout': 180,
 'presets': 'medium_quality',
 'project_name': 'playground-series-s6e6',
 'time_limit': 600,
 'use_gpu': True,
 'validation_strategy': 'holdout'}
RESULT_MARKER = 'AIDE_RESULT_JSON:'
FORBIDDEN_SPLIT_MARKER = '__is_train__'
FORBIDDEN_ROW_ID = '__aide_row_id__'
CLASS_WEIGHT_COL = "__aide_class_weight__"


def _num(out, col):
    return pd.to_numeric(out[col], errors="coerce") if col in out.columns else pd.Series(0.0, index=out.index)


def _bands(out):
    return [c for c in ["u", "g", "r", "i", "z"] if c in out.columns]


def _clean_base(out):
    for c in ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            med = out[c].median()
            out[c] = out[c].fillna(0.0 if pd.isna(med) else med)
    for c in ["spectral_type", "galaxy_population"]:
        if c in out.columns:
            out[c] = out[c].astype("string").fillna("__missing__").str.upper().str.strip()
    return out


def _color_block(out):
    bands = _bands(out)
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r"), ("u", "i"), ("u", "z"), ("g", "i"), ("g", "z"), ("r", "z")]:
        if a in out.columns and b in out.columns:
            out[f"{a}_{b}"] = (_num(out, a) - _num(out, b)).astype("float32")
    if bands:
        m = out[bands].astype("float32")
        out["mag_mean"] = m.mean(axis=1).astype("float32")
        out["mag_std"] = m.std(axis=1).fillna(0).astype("float32")
        out["mag_range"] = (m.max(axis=1) - m.min(axis=1)).astype("float32")
        out["mag_min"] = m.min(axis=1).astype("float32")
        out["mag_max"] = m.max(axis=1).astype("float32")
        out["mag_argmin"] = np.argmin(m.to_numpy(), axis=1).astype("int8")
        out["mag_argmax"] = np.argmax(m.to_numpy(), axis=1).astype("int8")
    if "redshift" in out.columns:
        z = _num(out, "redshift")
        out["redshift_abs"] = z.abs().astype("float32")
        out["redshift_sq"] = np.square(z).astype("float32")
        out["redshift_log1p_abs"] = np.log1p(z.abs()).astype("float32")
        out["redshift_pos"] = np.maximum(z, 0).astype("float32")
        for edge in [0.0, 0.1, 0.5, 1.0, 2.0, 3.5]:
            out[f"redshift_gt_{str(edge).replace('.', 'p')}"] = (z > edge).astype("int8")
    return out


def _sky_block(out):
    if "alpha" in out.columns and "delta" in out.columns:
        ar = np.deg2rad(_num(out, "alpha"))
        dr = np.deg2rad(_num(out, "delta"))
        out["alpha_sin"] = np.sin(ar).astype("float32")
        out["alpha_cos"] = np.cos(ar).astype("float32")
        out["delta_sin"] = np.sin(dr).astype("float32")
        out["delta_cos"] = np.cos(dr).astype("float32")
        out["sky_x"] = (np.cos(dr) * np.cos(ar)).astype("float32")
        out["sky_y"] = (np.cos(dr) * np.sin(ar)).astype("float32")
        out["sky_z"] = np.sin(dr).astype("float32")
        alpha_bin = np.floor(np.clip(_num(out, "alpha"), 0, np.nextafter(360.0, 0.0)) / 15.0).astype("int16")
        delta_bin = np.floor((np.clip(_num(out, "delta"), -90, np.nextafter(90.0, -90.0)) + 90.0) / 10.0).astype("int16")
        out["sky_alpha_bin_24"] = alpha_bin
        out["sky_delta_bin_18"] = delta_bin
        out["sky_cell_bin"] = (alpha_bin * 18 + delta_bin).astype("int16")
        for k in [2, 3, 4]:
            out[f"alpha_sin{k}"] = np.sin(ar * k).astype("float32")
            out[f"alpha_cos{k}"] = np.cos(ar * k).astype("float32")
            out[f"delta_sin{k}"] = np.sin(dr * k).astype("float32")
            out[f"delta_cos{k}"] = np.cos(dr * k).astype("float32")
    return out


def _categorical_block(out):
    for c in ["spectral_type", "galaxy_population"]:
        if c in out.columns:
            s = out[c].astype("string").fillna("__missing__").str.upper().str.strip()
            counts = s.map(s.value_counts(dropna=False)).astype("float32")
            out[f"{c}_freq"] = counts
            out[f"{c}_freq_log1p"] = np.log1p(counts).astype("float32")
    if "spectral_type" in out.columns and "galaxy_population" in out.columns:
        out["spectral_type__galaxy_population"] = out["spectral_type"].astype(str) + "_" + out["galaxy_population"].astype(str)
    return out


def _rank_block(out):
    for c in ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]:
        if c in out.columns:
            out[f"{c}_qrank"] = _num(out, c).rank(method="average", pct=True).astype("float32")
    return out


def _source_feature_block(out):
    _color_block(out)
    _sky_block(out)
    _categorical_block(out)
    _rank_block(out)
    bands = _bands(out)
    if len(bands) == 5:
        x = np.arange(5, dtype="float32")
        m = out[bands].astype("float32").to_numpy()
        xm = float(x.mean())
        denom = float(((x - xm) ** 2).sum())
        slope = ((m - m.mean(axis=1, keepdims=True)) * (x - xm)).sum(axis=1) / denom
        intercept = m.mean(axis=1) - slope * xm
        fit = intercept[:, None] + slope[:, None] * x[None, :]
        resid = m - fit
        out["mag_sed_slope"] = slope.astype("float32")
        out["mag_sed_resid_mean_abs"] = np.abs(resid).mean(axis=1).astype("float32")
        out["mag_sed_resid_std"] = resid.std(axis=1).astype("float32")
        d2 = np.diff(m, n=2, axis=1)
        out["mag_sed_d2_mean"] = d2.mean(axis=1).astype("float32")
        out["mag_sed_d2_abs_mean"] = np.abs(d2).mean(axis=1).astype("float32")
        out["mag_sed_d2_std"] = d2.std(axis=1).astype("float32")
    return out

def preprocess(df: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    out = _clean_base(df.copy())
    _color_block(out)
    if isinstance(aux, pd.DataFrame) and not aux.empty and "class" in aux.columns:
        aux2 = _clean_base(aux.copy())
        _color_block(aux2)
        cols = [c for c in ["u", "g", "r", "i", "z", "redshift", "u_g", "g_r", "r_i", "i_z"] if c in out.columns and c in aux2.columns]
        for cls, grp in aux2.groupby(aux2["class"].astype(str).str.upper()):
            center = grp[cols].median(numeric_only=True)
            scale = (grp[cols].quantile(0.75) - grp[cols].quantile(0.25)).replace(0, 1.0)
            dist = (((out[cols] - center) / (scale + 1e-6)) ** 2).sum(axis=1) ** 0.5
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(cls))
            out[f"aux_dist_{safe}"] = dist.astype("float32")
            out[f"aux_sim_{safe}"] = (1.0 / (1.0 + dist)).astype("float32")
    return out


def _force_autogluon_cpu_resources() -> None:
    if AIDE_AG_CONFIG.get("use_gpu") is not False:
        return
    try:
        from autogluon.common.utils.resource_utils import ResourceManager
    except Exception:
        return
    ResourceManager.get_gpu_count = staticmethod(lambda: 0)
    ResourceManager.get_gpu_count_torch = staticmethod(lambda cuda_only=False: 0)


def _read_csv(data_dir: Path, stem: str) -> pd.DataFrame:
    gz_path = data_dir / f"{stem}.csv.gz"
    csv_path = data_dir / f"{stem}.csv"
    if gz_path.exists():
        return pd.read_csv(gz_path)
    return pd.read_csv(csv_path)


def _read_aux_csv(input_dir: Path) -> pd.DataFrame | None:
    aux_file = AIDE_AG_CONFIG.get("aux_file")
    if not aux_file:
        return None
    aux_path = input_dir / str(aux_file)
    if not aux_path.exists():
        raise FileNotFoundError(f"Configured aux file not found: {aux_path}")
    return pd.read_csv(aux_path)


def _preprocess_accepts_aux() -> bool:
    signature = inspect.signature(preprocess)
    return len(signature.parameters) == 2


def _run_preprocess(combined: pd.DataFrame, aux_df: pd.DataFrame | None) -> pd.DataFrame:
    if _preprocess_accepts_aux():
        if aux_df is None:
            aux_df = pd.DataFrame()
        return preprocess(combined.copy(), aux_df.copy())
    return preprocess(combined.copy())


def _balanced_sample_weight(labels: pd.Series) -> np.ndarray:
    labels = pd.Series(labels).reset_index(drop=True)
    counts = labels.value_counts(dropna=False)
    if counts.empty:
        raise ValueError("Cannot compute class weights for empty labels")
    weights_by_class = len(labels) / (len(counts) * counts.astype(float))
    return labels.map(weights_by_class).astype(float).to_numpy()


def _positive_probability(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    *,
    model: str | None = None,
) -> pd.Series:
    proba = predictor.predict_proba(data, model=model)
    return _positive_probability_from_proba(proba)


def _positive_probability_from_proba(proba) -> pd.Series:
    if isinstance(proba, pd.Series):
        return proba.reset_index(drop=True)
    for positive_class in (1, 1.0, "1", "1.0", True):
        if positive_class in proba.columns:
            return proba[positive_class].reset_index(drop=True)
    return proba.iloc[:, -1].reset_index(drop=True)


def _prediction_from_proba(proba) -> pd.Series:
    if isinstance(proba, pd.Series):
        return proba.reset_index(drop=True)
    return proba.idxmax(axis=1).reset_index(drop=True)


def _values_from_proba(proba, *, eval_metric: str) -> pd.Series:
    if eval_metric == "roc_auc":
        return _positive_probability_from_proba(proba)
    return _prediction_from_proba(proba)


def _predict_values(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    *,
    eval_metric: str,
    model: str | None = None,
) -> pd.Series:
    if eval_metric == "roc_auc":
        return _positive_probability(predictor, data, model=model)
    pred = predictor.predict(data, model=model)
    return pd.Series(pred).reset_index(drop=True)


def _make_combined_frame(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_part = train_df.copy()
    test_part = test_df.copy()
    return pd.concat([train_part, test_part], ignore_index=True, sort=False)


def _validate_preprocessed_frame(
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
        raise ValueError(
            f"preprocess changed row count: {len(after)} != {len(before)}. "
            "AutoGluon preprocess(df) must preserve one output row per input row. "
            "If row removal was intentional, such as outlier filtering, rewrite it "
            "without dropping rows: add an outlier flag, clipped/winsorized value, "
            "imputed clean value, anomaly score, or distance-from-normal feature instead."
        )
    if target_col in after.columns:
        raise ValueError(f"preprocess created forbidden target column: {target_col}")
    if FORBIDDEN_SPLIT_MARKER in after.columns:
        raise ValueError(f"preprocess created forbidden split marker column: {FORBIDDEN_SPLIT_MARKER}")
    if FORBIDDEN_ROW_ID in after.columns:
        raise ValueError(f"preprocess created forbidden row id column: {FORBIDDEN_ROW_ID}")
    if CLASS_WEIGHT_COL in after.columns:
        raise ValueError(f"preprocess created forbidden class weight column: {CLASS_WEIGHT_COL}")

    ordered = after.reset_index(drop=True)
    if len(ordered.iloc[:train_rows]) != train_rows:
        raise ValueError("preprocess changed number of train rows")
    if len(ordered.iloc[train_rows:]) != test_rows:
        raise ValueError("preprocess changed number of test rows")
    return ordered


def _configured_metric() -> str:
    return AIDE_AG_CONFIG["eval_metric"]


def _should_stratify_holdout(target: pd.Series) -> bool:
    unique_count = target.nunique(dropna=True)
    if unique_count == 2:
        return True
    if pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target):
        return True
    return False


def _json_safe_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _leaderboard_records(predictor: TabularPredictor) -> list[dict]:
    keep_columns = [
        "model",
        "score_val",
        "score_test",
        "eval_metric",
        "fit_time",
        "fit_time_marginal",
        "pred_time_val",
        "pred_time_val_marginal",
        "stack_level",
        "can_infer",
        "fit_order",
    ]
    try:
        leaderboard = predictor.leaderboard(silent=True)
    except Exception as exc:
        return [{"error": f"leaderboard unavailable: {exc}"}]
    records = []
    for row in leaderboard.to_dict(orient="records"):
        records.append(
            {
                column: _json_safe_scalar(row.get(column))
                for column in keep_columns
                if column in row
            }
        )
    return records


def _artifact_dir(working_dir: Path) -> Path:
    return Path(os.environ.get("AIDE_NODE_ARTIFACT_DIR", str(working_dir)))


def _save_submission(submission: pd.DataFrame, working_dir: Path) -> Path:
    submission_path = working_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    artifact_dir = _artifact_dir(working_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_submission_path = artifact_dir / "submission.csv"
    if artifact_submission_path.resolve() != submission_path.resolve():
        shutil.copy2(submission_path, artifact_submission_path)
    return submission_path


def _save_prediction_artifact(frame: pd.DataFrame, working_dir: Path, filename: str) -> Path:
    if not filename.endswith(".gz"):
        filename = f"{filename}.gz"
    working_path = working_dir / filename
    working_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(working_path, index=False, compression="gzip")
    artifact_dir = _artifact_dir(working_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.resolve() != working_path.resolve():
        shutil.copy2(working_path, artifact_path)
    return working_path


def _safe_prediction_name(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
    return safe or "model"


def _save_autogluon_prediction_artifacts(
    predictor: TabularPredictor,
    *,
    train_target: pd.Series,
    test_model: pd.DataFrame,
    test_ids: pd.Series,
    test_pred: pd.Series,
    eval_metric: str,
    working_dir: Path,
    id_col: str,
    target_col: str,
    valid_data: pd.DataFrame | None,
    valid_pred: pd.Series | None,
) -> dict:
    artifacts = {}
    test_predictions = pd.DataFrame({
        id_col: pd.Series(test_ids).reset_index(drop=True),
        target_col: pd.Series(test_pred).reset_index(drop=True),
    })
    test_path = _save_prediction_artifact(
        test_predictions,
        working_dir,
        "test_predictions.csv",
    )
    artifacts["test_predictions"] = str(test_path)

    try:
        per_model = []
        for model_name in predictor.model_names():
            try:
                model_oof_proba = predictor.predict_proba_oof(
                    model=model_name,
                    transformed=False,
                    as_multiclass=True,
                )
                model_oof_pred = _values_from_proba(
                    model_oof_proba,
                    eval_metric=eval_metric,
                )
                if len(model_oof_pred) != len(train_target):
                    raise ValueError(
                        f"OOF row count {len(model_oof_pred)} != train rows {len(train_target)}"
                    )
                safe_model_name = _safe_prediction_name(model_name)
                model_oof_frame = pd.DataFrame({
                    "row": np.arange(len(train_target)),
                    "target": pd.Series(train_target).reset_index(drop=True),
                    "prediction": model_oof_pred.reset_index(drop=True),
                    "model": model_name,
                })
                model_oof_path = _save_prediction_artifact(
                    model_oof_frame,
                    working_dir,
                    f"model_predictions/{safe_model_name}-oof.csv",
                )
                model_test_pred = _predict_values(
                    predictor,
                    test_model,
                    eval_metric=eval_metric,
                    model=model_name,
                )
                model_test_frame = pd.DataFrame({
                    id_col: pd.Series(test_ids).reset_index(drop=True),
                    target_col: pd.Series(model_test_pred).reset_index(drop=True),
                    "model": model_name,
                })
                model_test_path = _save_prediction_artifact(
                    model_test_frame,
                    working_dir,
                    f"model_predictions/{safe_model_name}-test.csv",
                )
                per_model.append({
                    "model": model_name,
                    "oof_predictions": str(model_oof_path),
                    "test_predictions": str(model_test_path),
                    "rows": int(len(model_oof_frame)),
                    "test_rows": int(len(model_test_frame)),
                })
            except Exception as exc:
                per_model.append({
                    "model": model_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        artifacts["model_predictions"] = per_model
        artifacts["model_predictions_ok"] = sum(1 for item in per_model if "error" not in item)

        oof_proba = predictor.predict_proba_oof(
            transformed=False,
            as_multiclass=True,
        )
        oof_pred = _values_from_proba(oof_proba, eval_metric=eval_metric)
        if len(oof_pred) != len(train_target):
            raise ValueError(f"OOF row count {len(oof_pred)} != train rows {len(train_target)}")
        oof_frame = pd.DataFrame({
            "row": np.arange(len(train_target)),
            "target": pd.Series(train_target).reset_index(drop=True),
            "prediction": oof_pred.reset_index(drop=True),
        })
        oof_path = _save_prediction_artifact(
            oof_frame,
            working_dir,
            "oof_predictions.csv",
        )
        artifacts["oof_predictions"] = str(oof_path)
        artifacts["oof_rows"] = int(len(oof_frame))
    except Exception as exc:
        artifacts["oof_error"] = f"{type(exc).__name__}: {exc}"

    if valid_data is not None and valid_pred is not None:
        validation_frame = pd.DataFrame({
            "row": np.arange(len(valid_data)),
            "target": valid_data[target_col].reset_index(drop=True),
            "prediction": pd.Series(valid_pred).reset_index(drop=True),
        })
        validation_path = _save_prediction_artifact(
            validation_frame,
            working_dir,
            "validation_predictions.csv",
        )
        artifacts["validation_predictions"] = str(validation_path)
        artifacts["validation_rows"] = int(len(validation_frame))
    return artifacts


def _make_submission(
    sample_submission: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    test_ids: pd.Series,
    test_pred: pd.Series,
) -> pd.DataFrame:
    prediction_frame = pd.DataFrame({
        id_col: pd.Series(test_ids).reset_index(drop=True),
        target_col: pd.Series(test_pred).reset_index(drop=True),
    })
    if prediction_frame[id_col].duplicated().any():
        raise ValueError(f"test data contains duplicate {id_col} values")

    submission = sample_submission.copy()
    mapped = submission[[id_col]].merge(
        prediction_frame,
        on=id_col,
        how="left",
        validate="one_to_one",
    )[target_col]
    if mapped.isna().any():
        missing = int(mapped.isna().sum())
        raise ValueError(f"missing predictions for {missing} sample_submission ids")

    submission[target_col] = mapped.to_numpy()
    return submission.sort_values(id_col, kind="mergesort").reset_index(drop=True)


@contextlib.contextmanager
def _preprocess_timeout(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    class PreprocessTimeoutError(TimeoutError):
        pass

    def _raise_preprocess_timeout(_signum, _frame):
        raise PreprocessTimeoutError(
            "AIDE AutoGluon preprocess exceeded the dedicated timeout of "
            f"{seconds} seconds. This timeout is separate from AutoGluon "
            "training time_limit. Analyze preprocess(df) and remove or replace "
            "time-consuming operations such as Python callbacks over groups or "
            "rolling windows, polynomial fitting, row-wise loops, or repeated "
            "full-frame copies."
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _raise_preprocess_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextlib.contextmanager
def _quiet_model_output(working_dir: Path):
    artifact_dir = _artifact_dir(working_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "autogluon_stdout.log"

    class _TeeWriter:
        def __init__(self, primary, log_file):
            self.primary = primary
            self.log_file = log_file

        def write(self, text):
            written = self.primary.write(text)
            self.log_file.write(text)
            if hasattr(self.primary, "flush"):
                self.primary.flush()
            self.log_file.flush()
            return written

        def flush(self):
            if hasattr(self.primary, "flush"):
                self.primary.flush()
            self.log_file.flush()

        def fileno(self):
            if hasattr(self.primary, "fileno"):
                return self.primary.fileno()
            fallback = getattr(sys, "__stderr__", None) or getattr(sys, "__stdout__", None)
            if fallback is not None and hasattr(fallback, "fileno"):
                return fallback.fileno()
            return self.log_file.fileno()

        def isatty(self):
            return bool(self.primary.isatty()) if hasattr(self.primary, "isatty") else False

        @property
        def encoding(self):
            return getattr(self.primary, "encoding", "utf-8")

    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        stdout_writer = _TeeWriter(sys.stdout, log_file)
        stderr_writer = _TeeWriter(sys.stderr, log_file)
        log_handler = logging.StreamHandler(stderr_writer)
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        logger_names = ["", "autogluon"]
        loggers = [logging.getLogger(name) for name in logger_names]
        previous_states = [
            (
                logger.level,
                logger.disabled,
                list(logger.handlers),
                logger.propagate,
            )
            for logger in loggers
        ]
        for logger in loggers:
            logger.disabled = False
            logger.setLevel(logging.INFO)
            logger.handlers = [log_handler]
            logger.propagate = False
        try:
            with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
                yield
        finally:
            for logger, (level, disabled, handlers, propagate) in zip(
                loggers,
                previous_states,
            ):
                logger.setLevel(level)
                logger.disabled = disabled
                logger.handlers = handlers
                logger.propagate = propagate


def main() -> None:
    _force_autogluon_cpu_resources()

    input_dir = Path("./input")
    working_dir = Path("./working")
    working_dir.mkdir(parents=True, exist_ok=True)

    train_df = _read_csv(input_dir, "train")
    test_df = _read_csv(input_dir, "test")
    sample_submission = _read_csv(input_dir, "sample_submission")
    aux_df = _read_aux_csv(input_dir)
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]
    if target_col not in train_df.columns:
        raise ValueError(f"Target column {target_col!r} not found in train data")

    y_train = train_df[target_col].reset_index(drop=True)
    train_features = train_df.drop(columns=[target_col, id_col], errors="ignore")
    test_features = test_df.drop(columns=[id_col], errors="ignore")
    combined = _make_combined_frame(train_features, test_features)
    if aux_df is not None:
        print(
            "AIDE AutoGluon: loaded aux file "
            f"{AIDE_AG_CONFIG.get('aux_file')} rows={len(aux_df)} "
            f"cols={len(aux_df.columns)} "
            f"passed_to_preprocess={_preprocess_accepts_aux()}",
            flush=True,
        )
    print("AIDE AutoGluon: starting preprocess", flush=True)
    preprocess_started_at = time.time()
    with _preprocess_timeout(int(AIDE_AG_CONFIG.get("preprocess_timeout", 180))):
        preprocessed = _run_preprocess(combined, aux_df)
    preprocess_time = time.time() - preprocess_started_at
    feature_count = int(len(preprocessed.columns))
    print(
        f"AIDE AutoGluon: finished preprocess rows={len(preprocessed)} cols={feature_count}",
        flush=True,
    )
    preprocessed = _validate_preprocessed_frame(
        combined,
        preprocessed,
        target_col=target_col,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )

    train_fe = preprocessed.iloc[:len(train_df)].copy()
    test_fe = preprocessed.iloc[len(train_df):].copy()
    train_model = train_fe.copy()
    train_model[target_col] = y_train.to_numpy()
    if AIDE_AG_CONFIG.get("class_balance") == "balanced":
        train_model[CLASS_WEIGHT_COL] = _balanced_sample_weight(y_train)
    test_model = test_fe.copy()

    eval_metric = _configured_metric()
    fit_args = dict(AIDE_AG_CONFIG.get("fit_args", {}) or {})
    bagged_mode = int(fit_args.get("num_bag_folds") or 0) > 0 or bool(fit_args.get("auto_stack"))
    defer_save_space = bool(bagged_mode and fit_args.pop("save_space", False))
    if bagged_mode:
        train_data = train_model
        valid_data = None
        print(
            "AIDE AutoGluon: bagged mode detected; using internal OOF validation without tuning_data",
            flush=True,
        )
    elif AIDE_AG_CONFIG.get("validation_strategy") == "holdout":
        stratify = train_model[target_col] if _should_stratify_holdout(train_model[target_col]) else None
        train_data, valid_data = train_test_split(
            train_model,
            test_size=AIDE_AG_CONFIG.get("validation_fraction", 0.2),
            random_state=AIDE_AG_CONFIG.get("seed", 42),
            stratify=stratify,
        )
    else:
        train_data = train_model
        valid_data = None

    model_dir = working_dir / "autogluon_model"
    shutil.rmtree(model_dir, ignore_errors=True)
    fit_kwargs = {
        "train_data": train_data,
        "presets": AIDE_AG_CONFIG["presets"],
        "time_limit": AIDE_AG_CONFIG["time_limit"],
    }
    if AIDE_AG_CONFIG.get("use_gpu") is not None:
        fit_kwargs["num_gpus"] = 1 if AIDE_AG_CONFIG["use_gpu"] else 0
    if valid_data is not None:
        fit_kwargs["tuning_data"] = valid_data
    if AIDE_AG_CONFIG.get("included_model_types"):
        fit_kwargs["included_model_types"] = AIDE_AG_CONFIG["included_model_types"]
    if AIDE_AG_CONFIG.get("hyperparameters"):
        fit_kwargs["hyperparameters"] = AIDE_AG_CONFIG["hyperparameters"]
    fit_kwargs.update(fit_args)
    training_started_at = time.time()
    with _quiet_model_output(working_dir):
        print("AIDE AutoGluon: starting fit", flush=True)
        predictor_kwargs = {
            "label": target_col,
            "eval_metric": eval_metric,
            "path": str(model_dir),
            "verbosity": 2,
        }
        if AIDE_AG_CONFIG.get("class_balance") == "balanced":
            predictor_kwargs["sample_weight"] = CLASS_WEIGHT_COL
            predictor_kwargs["weight_evaluation"] = False
        predictor = TabularPredictor(**predictor_kwargs)
        predictor.fit(**fit_kwargs)
        print("AIDE AutoGluon: finished fit", flush=True)
    actual_eval_metric = str(getattr(predictor.eval_metric, "name", predictor.eval_metric))
    training_time = time.time() - training_started_at
    model_records = _leaderboard_records(predictor)

    with _quiet_model_output(working_dir):
        print("AIDE AutoGluon: starting validation and prediction", flush=True)
        valid_pred = None
        if valid_data is None:
            leaderboard = predictor.leaderboard(silent=True)
            score_candidates = [
                col for col in ("score_val", "score_test") if col in leaderboard.columns
            ]
            metric_value = float(leaderboard[score_candidates[0]].max()) if score_candidates else float("nan")
            lower_is_better = False
        elif eval_metric == "roc_auc":
            valid_pred = _positive_probability(
                predictor,
                valid_data.drop(columns=[target_col, CLASS_WEIGHT_COL], errors="ignore"),
            )
            metric_value = float(roc_auc_score(valid_data[target_col], valid_pred))
            lower_is_better = False
        else:
            valid_pred = _predict_values(
                predictor,
                valid_data.drop(columns=[target_col, CLASS_WEIGHT_COL], errors="ignore"),
                eval_metric=eval_metric,
            )
            scores = predictor.evaluate(valid_data, silent=True)
            metric_value = float(scores.get(eval_metric))
            lower_is_better = False

        test_pred = _predict_values(
            predictor,
            test_model,
            eval_metric=eval_metric,
        )
        prediction_artifacts = _save_autogluon_prediction_artifacts(
            predictor,
            train_target=y_train,
            test_model=test_model,
            test_ids=test_df[id_col],
            test_pred=test_pred,
            eval_metric=eval_metric,
            working_dir=working_dir,
            id_col=id_col,
            target_col=target_col,
            valid_data=valid_data,
            valid_pred=valid_pred,
        )
        if defer_save_space:
            try:
                predictor.save_space(remove_data=True, remove_fit_stack=True)
                prediction_artifacts["save_space_after_artifacts"] = True
            except Exception as exc:
                prediction_artifacts["save_space_error"] = f"{type(exc).__name__}: {exc}"
    submission = _make_submission(
        sample_submission,
        id_col=id_col,
        target_col=target_col,
        test_ids=test_df[id_col],
        test_pred=test_pred,
    )
    submission_path = _save_submission(submission, working_dir)
    artifact_submission_path = _artifact_dir(working_dir) / "submission.csv"
    print("AIDE AutoGluon: finished validation and prediction", flush=True)
    print(f"AIDE AutoGluon: submission saved to {submission_path}", flush=True)
    if artifact_submission_path.resolve() != submission_path.resolve():
        print(f"AIDE AutoGluon: artifact submission saved to {artifact_submission_path}", flush=True)
    if prediction_artifacts.get("oof_predictions"):
        print(f"AIDE AutoGluon: OOF predictions saved to {prediction_artifacts['oof_predictions']}", flush=True)
    elif prediction_artifacts.get("oof_error"):
        print(f"AIDE AutoGluon: OOF predictions unavailable: {prediction_artifacts['oof_error']}", flush=True)
    if prediction_artifacts.get("validation_predictions"):
        print(f"AIDE AutoGluon: validation predictions saved to {prediction_artifacts['validation_predictions']}", flush=True)
    print(f"AIDE AutoGluon: test predictions saved to {prediction_artifacts['test_predictions']}", flush=True)

    summary = "AutoGluon preprocess wrapper completed."
    run_stats = {
        "feature_count": feature_count,
        "preprocess_time": float(preprocess_time),
        "training_time": float(training_time),
        "eval_metric": actual_eval_metric,
        "models": model_records,
        "prediction_artifacts": prediction_artifacts,
    }
    print(f"Validation {eval_metric}: {metric_value:.6f}")
    print("Submission saved successfully.")
    print(RESULT_MARKER + " " + json.dumps({
        "is_bug": False,
        "summary": summary,
        "metric": metric_value,
        "eval_metric": actual_eval_metric,
        "lower_is_better": lower_is_better,
        "run_stats": run_stats,
        
    }, sort_keys=True))


# AIDE executes generated code via exec with a custom globals dict, where __name__ is not
# guaranteed to be "__main__". Call main() directly so the wrapper actually runs
# inside the interpreter.
main()

