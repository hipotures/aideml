from __future__ import annotations

import json
import shutil
import warnings
import contextlib
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
    from sklearn.cluster import KMeans

    X = df.copy()
    X["_row_id"] = np.arange(len(X), dtype=np.int64)

    race_driver_keys = ["Year", "Race", "Driver"]
    race_lap_keys = ["Year", "Race", "LapNumber"]

    X["_is_wet"] = X["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")
    X["_is_soft"] = X["Compound"].eq("SOFT").astype("int8")
    X["_is_hard"] = X["Compound"].eq("HARD").astype("int8")

    X = X.sort_values(race_driver_keys + ["LapNumber"], kind="mergesort")
    g_drv = X.groupby(race_driver_keys, sort=False)

    prev_compound = g_drv["Compound"].shift(1)
    prev_laptime = g_drv["LapTime (s)"].shift(1)
    prev_delta = g_drv["LapTime_Delta"].shift(1)
    prev_tyre = g_drv["TyreLife"].shift(1)

    X["_compound_switch"] = (
        prev_compound.notna() & X["Compound"].ne(prev_compound)
    ).astype("int8")
    X["_laptime_jump"] = X["LapTime (s)"] - prev_laptime
    X["_delta_jump"] = X["LapTime_Delta"] - prev_delta
    X["_tyrelife_jump"] = X["TyreLife"] - prev_tyre

    delta_thr = X["LapTime_Delta"].quantile(0.90)
    jump_thr = X["_laptime_jump"].quantile(0.90)
    X["_extreme_delta"] = (X["LapTime_Delta"] >= delta_thr).astype("int8")
    X["_extreme_jump"] = (X["_laptime_jump"] >= jump_thr).fillna(0).astype("int8")

    field = (
        X.groupby(race_lap_keys, sort=False)
        .agg(
            field_size=("Driver", "size"),
            lap_delta_mean=("LapTime_Delta", "mean"),
            lap_delta_std=("LapTime_Delta", "std"),
            lap_delta_med=("LapTime_Delta", "median"),
            laptime_mean=("LapTime (s)", "mean"),
            laptime_std=("LapTime (s)", "std"),
            poschg_mean=("Position_Change", "mean"),
            poschg_std=("Position_Change", "std"),
            tyrelife_mean=("TyreLife", "mean"),
            degr_mean=("Cumulative_Degradation", "mean"),
            wet_share=("_is_wet", "mean"),
            soft_share=("_is_soft", "mean"),
            hard_share=("_is_hard", "mean"),
            current_pit_share=("PitStop", "mean"),
            compound_switch_share=("_compound_switch", "mean"),
            extreme_delta_share=("_extreme_delta", "mean"),
            extreme_jump_share=("_extreme_jump", "mean"),
        )
        .reset_index()
    )

    field["lap_delta_std"] = field["lap_delta_std"].fillna(0.0)
    field["laptime_std"] = field["laptime_std"].fillna(0.0)
    field["poschg_std"] = field["poschg_std"].fillna(0.0)
    field["field_time_cv"] = field["laptime_std"] / field["laptime_mean"].replace(
        0, np.nan
    )
    field["delta_vol_per_car"] = field["lap_delta_std"] / np.sqrt(
        field["field_size"].clip(lower=1)
    )

    field = field.sort_values(race_lap_keys, kind="mergesort")
    race_groups = field.groupby(["Year", "Race"], sort=False)

    causal_cols = [
        "lap_delta_mean",
        "lap_delta_std",
        "wet_share",
        "compound_switch_share",
        "extreme_delta_share",
        "extreme_jump_share",
        "current_pit_share",
        "field_time_cv",
        "poschg_std",
    ]

    for col in causal_cols:
        prev1 = race_groups[col].shift(1)
        field[f"{col}_prev1"] = prev1
        field[f"{col}_chg1"] = field[col] - prev1
        field[f"{col}_roll3"] = race_groups[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )

    cluster_cols = [
        "lap_delta_mean",
        "lap_delta_std",
        "wet_share",
        "compound_switch_share",
        "extreme_delta_share",
        "extreme_jump_share",
        "current_pit_share",
        "field_time_cv",
        "poschg_std",
        "lap_delta_mean_prev1",
        "wet_share_prev1",
        "compound_switch_share_prev1",
        "current_pit_share_prev1",
        "lap_delta_mean_chg1",
        "wet_share_chg1",
        "compound_switch_share_chg1",
        "extreme_delta_share_chg1",
        "current_pit_share_chg1",
        "lap_delta_mean_roll3",
        "wet_share_roll3",
        "compound_switch_share_roll3",
        "current_pit_share_roll3",
    ]

    Z = field[cluster_cols].copy()
    Z = Z.replace([np.inf, -np.inf], np.nan)
    Z = Z.fillna(Z.median(numeric_only=True)).fillna(0.0)

    n_clusters = min(4, max(1, len(Z)))
    if n_clusters > 1:
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
        raw_state = km.fit_predict(Z)
    else:
        raw_state = np.zeros(len(Z), dtype=np.int32)

    field["_raw_regime_state"] = raw_state
    state_risk = (
        field.groupby("_raw_regime_state", sort=False)[
            [
                "wet_share",
                "extreme_delta_share",
                "extreme_jump_share",
                "compound_switch_share",
                "current_pit_share",
            ]
        ]
        .mean()
        .assign(
            risk=lambda t: (
                1.2 * t["wet_share"]
                + 1.1 * t["extreme_delta_share"]
                + 1.0 * t["extreme_jump_share"]
                + 0.9 * t["compound_switch_share"]
                + 0.8 * t["current_pit_share"]
            )
        )["risk"]
        .sort_values()
    )
    state_map = {state: i for i, state in enumerate(state_risk.index.tolist())}
    field["regime_state"] = field["_raw_regime_state"].map(state_map).astype("int8")

    field["prev_regime_state"] = (
        race_groups["regime_state"].shift(1).fillna(-1).astype("int8")
    )
    field["regime_transition"] = (
        field["regime_state"] != field["prev_regime_state"]
    ).astype("int8")
    field["regime_escalation"] = (
        field["regime_state"] > field["prev_regime_state"]
    ).astype("int8")
    field["regime_deescalation"] = (
        field["regime_state"] < field["prev_regime_state"]
    ).astype("int8")

    keep_cols = race_lap_keys + [
        "regime_state",
        "prev_regime_state",
        "regime_transition",
        "regime_escalation",
        "regime_deescalation",
        "wet_share",
        "compound_switch_share",
        "extreme_delta_share",
        "extreme_jump_share",
        "current_pit_share",
        "field_time_cv",
        "lap_delta_mean",
        "lap_delta_std",
        "lap_delta_mean_prev1",
        "wet_share_prev1",
        "compound_switch_share_prev1",
        "current_pit_share_prev1",
        "lap_delta_mean_chg1",
        "wet_share_chg1",
        "compound_switch_share_chg1",
        "extreme_delta_share_chg1",
        "current_pit_share_chg1",
        "lap_delta_mean_roll3",
        "wet_share_roll3",
        "compound_switch_share_roll3",
        "current_pit_share_roll3",
    ]

    X = X.merge(field[keep_cols], on=race_lap_keys, how="left", sort=False)

    X["regime_x_tyrelife"] = X["regime_state"].astype("float32") * X["TyreLife"].astype(
        "float32"
    )
    X["regime_x_degradation"] = X["regime_state"].astype("float32") * X[
        "Cumulative_Degradation"
    ].astype("float32")
    X["wet_regime_driver_match"] = X["wet_share"].astype("float32") * X[
        "_is_wet"
    ].astype("float32")
    X["switch_regime_driver_match"] = X["compound_switch_share"].astype("float32") * X[
        "_compound_switch"
    ].astype("float32")
    X["shock_vs_driver_delta"] = X["LapTime_Delta"].astype("float32") - X[
        "lap_delta_mean"
    ].astype("float32")
    X["shock_vs_prev_field"] = X["LapTime_Delta"].astype("float32") - X[
        "lap_delta_mean_prev1"
    ].astype("float32")
    X["pit_pressure_signal"] = (
        X["current_pit_share"].astype("float32")
        + X["compound_switch_share"].astype("float32")
        + X["extreme_delta_share"].astype("float32")
    )

    X = X.sort_values("_row_id", kind="mergesort").drop(columns=["_row_id"])
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
    preprocess_started_at = time.time()
    with _preprocess_timeout(int(AIDE_AG_CONFIG.get("preprocess_timeout", 180))):
        preprocessed = preprocess(combined.copy())
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
    training_started_at = time.time()
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
    training_time = time.time() - training_started_at
    model_records = _leaderboard_records(predictor)

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
    run_stats = {
        "feature_count": feature_count,
        "preprocess_time": float(preprocess_time),
        "training_time": float(training_time),
        "models": model_records,
    }
    print(f"Validation {eval_metric}: {metric_value:.6f}")
    print("Submission saved successfully.")
    print(RESULT_MARKER + " " + json.dumps({
        "is_bug": False,
        "summary": summary,
        "metric": metric_value,
        "lower_is_better": lower_is_better,
        "run_stats": run_stats,
        
        "research_hypotheses_llm_claimed_used": ["000326"],
        "research_usage_note": "Verified assigned hypothesis 000326.",
    }, sort_keys=True))


# AIDE executes generated code via exec with a custom globals dict, where __name__ is not
# guaranteed to be "__main__". Call main() directly so the wrapper actually runs
# inside the interpreter.
main()
