from __future__ import annotations

import json
import shutil
import warnings
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

AIDE_AG_CONFIG = {'fit_args': {'auto_stack': False, 'fit_weighted_ensemble': True, 'save_space': True},
 'included_model_types': ['XGB', 'GBM', 'CAT'],
 'preprocess_timeout': 180,
 'presets': 'medium_quality',
 'time_limit': 600,
 'use_gpu': False,
 'validation_strategy': 'holdout'}
RESULT_MARKER = 'AIDE_RESULT_JSON:'
FORBIDDEN_SPLIT_MARKER = '__is_train__'
FORBIDDEN_ROW_ID = '__aide_row_id__'


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    X = df.copy()

    lap_col = "LapTime (s)"
    delta_col = "LapTime_Delta"
    race_keys = ["Year", "Race"]
    snap_keys = ["Year", "Race", "LapNumber"]

    # Stable categorical cleanup for AutoGluon models.
    for col in ["Compound", "Driver", "Race"]:
        X[col] = X[col].astype("category")

    # Basic pace / tyre context features.
    X["TyreLife_sq"] = X["TyreLife"] * X["TyreLife"]
    X["RaceProgress_sq"] = X["RaceProgress"] * X["RaceProgress"]
    X["TyreLife_by_Stint"] = X["TyreLife"] / X["Stint"].clip(lower=1)
    X["TyreLife_by_RaceProgress"] = X["TyreLife"] / X["RaceProgress"].clip(lower=0.02)
    X["Deg_per_TyreLife"] = X["Cumulative_Degradation"] / X["TyreLife"].clip(lower=1)
    X["Deg_per_Progress"] = X["Cumulative_Degradation"] / X["RaceProgress"].clip(
        lower=0.02
    )

    # Snapshot-level robust aggregates for latent neutralization inference.
    g_snap = X.groupby(snap_keys, observed=True)

    X["snap_laptime_med"] = g_snap[lap_col].transform("median")
    X["snap_laptime_std"] = g_snap[lap_col].transform("std").fillna(0.0)
    X["snap_delta_med"] = g_snap[delta_col].transform("median")
    X["snap_delta_std"] = g_snap[delta_col].transform("std").fillna(0.0)

    X["snap_laptime_absdev"] = (X[lap_col] - X["snap_laptime_med"]).abs()
    X["snap_delta_absdev"] = (X[delta_col] - X["snap_delta_med"]).abs()
    X["snap_laptime_mad"] = (
        g_snap["snap_laptime_absdev"].transform("median").fillna(0.0)
    )
    X["snap_delta_mad"] = g_snap["snap_delta_absdev"].transform("median").fillna(0.0)

    lt_scale = X["snap_laptime_mad"] + 0.5 * X["snap_laptime_std"] + 1e-6
    d_scale = X["snap_delta_mad"] + 0.5 * X["snap_delta_std"] + 1e-6

    X["snap_laptime_robust_z"] = (X[lap_col] - X["snap_laptime_med"]) / lt_scale
    X["snap_delta_robust_z"] = (X[delta_col] - X["snap_delta_med"]) / d_scale

    extreme_slow = (
        (X["snap_laptime_robust_z"] > 1.5)
        | (X["snap_delta_robust_z"] > 1.5)
        | (X[delta_col] > 8.0)
    ).astype("float32")
    X["snap_extreme_slow_share"] = extreme_slow.groupby(
        [X[k] for k in snap_keys], observed=True
    ).transform("mean")
    X["snap_pit_share_now"] = g_snap["PitStop"].transform("mean").fillna(0.0)

    # Lag race-level snapshot summaries by one lap to avoid subtle same-lap leakage.
    snap = (
        X[
            snap_keys
            + [
                "snap_laptime_med",
                "snap_laptime_std",
                "snap_delta_med",
                "snap_delta_std",
                "snap_laptime_mad",
                "snap_delta_mad",
                "snap_extreme_slow_share",
                "snap_pit_share_now",
            ]
        ]
        .drop_duplicates(subset=snap_keys)
        .sort_values(race_keys + ["LapNumber"])
    )

    lag_cols = [
        "snap_laptime_med",
        "snap_laptime_std",
        "snap_delta_med",
        "snap_delta_std",
        "snap_laptime_mad",
        "snap_delta_mad",
        "snap_extreme_slow_share",
        "snap_pit_share_now",
    ]
    for col in lag_cols:
        snap[f"lag1_{col}"] = snap.groupby(race_keys, observed=True)[col].shift(1)

    keep_cols = snap_keys + [f"lag1_{c}" for c in lag_cols]
    X = X.merge(snap[keep_cols], on=snap_keys, how="left", sort=False)

    # Normalize current lap against lagged field state.
    lag_lt_scale = (
        X["lag1_snap_laptime_mad"].fillna(0.0)
        + 0.5 * X["lag1_snap_laptime_std"].fillna(0.0)
        + 1e-6
    )
    lag_d_scale = (
        X["lag1_snap_delta_mad"].fillna(0.0)
        + 0.5 * X["lag1_snap_delta_std"].fillna(0.0)
        + 1e-6
    )

    X["vs_prev_field_laptime_z"] = (
        X[lap_col] - X["lag1_snap_laptime_med"].fillna(X["snap_laptime_med"])
    ) / lag_lt_scale
    X["vs_prev_field_delta_z"] = (
        X[delta_col] - X["lag1_snap_delta_med"].fillna(X["snap_delta_med"])
    ) / lag_d_scale

    # Posterior-style neutralization score from lagged field shock and synchronized pitting.
    X["neutralization_score"] = (
        1.25 * X["lag1_snap_extreme_slow_share"].fillna(0.0)
        + 1.10 * X["lag1_snap_pit_share_now"].fillna(0.0)
        + 0.35 * np.tanh(X["lag1_snap_laptime_std"].fillna(0.0) / 12.0)
        + 0.35 * np.tanh(X["lag1_snap_delta_std"].fillna(0.0) / 6.0)
    )
    X["neutralization_posterior"] = 1.0 / (
        1.0 + np.exp(-4.0 * (X["neutralization_score"] - 0.35))
    )
    X["neutralized_flag"] = (X["neutralization_posterior"] > 0.5).astype("int8")

    # Interactions so the model can learn different pit windows under neutralized conditions.
    X["neut_x_tyrelife"] = X["neutralization_posterior"] * X["TyreLife"]
    X["neut_x_deg"] = X["neutralization_posterior"] * X["Cumulative_Degradation"]
    X["neut_x_progress"] = X["neutralization_posterior"] * X["RaceProgress"]
    X["neut_x_stint"] = X["neutralization_posterior"] * X["Stint"]
    X["neut_x_vs_prev_delta"] = (
        X["neutralization_posterior"] * X["vs_prev_field_delta_z"]
    )
    X["neut_x_vs_prev_laptime"] = (
        X["neutralization_posterior"] * X["vs_prev_field_laptime_z"]
    )

    # Relative race-position context around possible caution windows.
    X["Position_from_front"] = X["Position"] - 1
    X["Position_gain_pressure"] = (-X["Position_Change"]).clip(lower=0)
    X["neut_x_position"] = X["neutralization_posterior"] * X["Position"]
    X["neut_x_pos_change"] = X["neutralization_posterior"] * X["Position_Change"]

    # Keep helper columns as numeric signals; drop only temporary absdev columns.
    X = X.drop(columns=["snap_laptime_absdev", "snap_delta_absdev"])

    return X


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


def _positive_probability(predictor: TabularPredictor, data: pd.DataFrame) -> pd.Series:
    proba = predictor.predict_proba(data)
    for positive_class in (1, 1.0, "1", "1.0", True):
        if positive_class in proba.columns:
            return proba[positive_class]
    return proba.iloc[:, -1]


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

    ordered = after.reset_index(drop=True)
    if len(ordered.iloc[:train_rows]) != train_rows:
        raise ValueError("preprocess changed number of train rows")
    if len(ordered.iloc[train_rows:]) != test_rows:
        raise ValueError("preprocess changed number of test rows")
    return ordered


def _infer_metric(target: pd.Series) -> tuple[str, str | None]:
    configured = AIDE_AG_CONFIG.get("eval_metric", "auto")
    if configured != "auto":
        return configured, "binary" if target.nunique(dropna=True) == 2 else None
    if target.nunique(dropna=True) == 2:
        return "roc_auc", "binary"
    return "accuracy", None


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
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]
    if target_col not in train_df.columns:
        raise ValueError(f"Target column {target_col!r} not found in train data")

    y_train = train_df[target_col].reset_index(drop=True)
    train_features = train_df.drop(columns=[target_col, id_col], errors="ignore")
    test_features = test_df.drop(columns=[id_col], errors="ignore")
    combined = _make_combined_frame(train_features, test_features)
    print("AIDE AutoGluon: starting preprocess", flush=True)
    with _preprocess_timeout(int(AIDE_AG_CONFIG.get("preprocess_timeout", 180))):
        preprocessed = preprocess(combined.copy())
    print(
        f"AIDE AutoGluon: finished preprocess rows={len(preprocessed)} cols={len(preprocessed.columns)}",
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
    test_model = test_fe.copy()

    eval_metric, problem_type = _infer_metric(train_model[target_col])
    if AIDE_AG_CONFIG.get("validation_strategy") == "holdout":
        stratify = train_model[target_col] if problem_type == "binary" else None
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
    fit_kwargs.update(AIDE_AG_CONFIG.get("fit_args", {}))
    with _quiet_model_output(working_dir):
        print("AIDE AutoGluon: starting fit", flush=True)
        predictor = TabularPredictor(
            label=target_col,
            problem_type=problem_type,
            eval_metric=eval_metric,
            path=str(model_dir),
            verbosity=2,
        )
        predictor.fit(**fit_kwargs)
        print("AIDE AutoGluon: finished fit", flush=True)

    with _quiet_model_output(working_dir):
        print("AIDE AutoGluon: starting validation and prediction", flush=True)
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
                valid_data.drop(columns=[target_col]),
            )
            metric_value = float(roc_auc_score(valid_data[target_col], valid_pred))
            lower_is_better = False
        else:
            scores = predictor.evaluate(valid_data, silent=True)
            metric_value = float(scores.get(eval_metric))
            lower_is_better = False

        test_pred = _positive_probability(predictor, test_model)
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

    summary = "AutoGluon preprocess wrapper completed."
    print(f"Validation {eval_metric}: {metric_value:.6f}")
    print("Submission saved successfully.")
    print(RESULT_MARKER + " " + json.dumps({
        "is_bug": False,
        "summary": summary,
        "metric": metric_value,
        "lower_is_better": lower_is_better,
        
        "research_hypotheses_llm_claimed_used": ["000369"],
        "research_usage_note": "Verified assigned hypothesis 000369.",
    }, sort_keys=True))


# AIDE executes generated code via exec with a custom globals dict, where __name__ is not
# guaranteed to be "__main__". Call main() directly so the wrapper actually runs
# inside the interpreter.
main()
