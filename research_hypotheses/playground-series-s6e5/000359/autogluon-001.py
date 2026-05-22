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

    out = df.copy()
    n = len(out)

    race_s = out["Race"].astype("string").fillna("missing")
    driver_s = out["Driver"].astype("string").fillna("missing")
    comp_s = out["Compound"].astype("string").fillna("missing")

    for c, s in [("Race", race_s), ("Driver", driver_s), ("Compound", comp_s)]:
        vc = s.value_counts(normalize=True)
        out[c + "_freq"] = s.map(vc).astype("float32")
        out[c] = s.astype("category")

    lap = pd.to_numeric(out["LapNumber"], errors="coerce").astype("float64")
    progress = (
        pd.to_numeric(out["RaceProgress"], errors="coerce")
        .clip(0.001, 1.0)
        .astype("float64")
    )
    tyre_life = pd.to_numeric(out["TyreLife"], errors="coerce").astype("float64")
    stint = pd.to_numeric(out["Stint"], errors="coerce").astype("float64")
    pos = pd.to_numeric(out["Position"], errors="coerce").astype("float64")
    lap_time = pd.to_numeric(out["LapTime (s)"], errors="coerce").astype("float64")
    lap_delta = pd.to_numeric(out["LapTime_Delta"], errors="coerce").astype("float64")
    pit = pd.to_numeric(out["PitStop"], errors="coerce").fillna(0).astype("int8")
    cum_deg = pd.to_numeric(out["Cumulative_Degradation"], errors="coerce").astype(
        "float64"
    )
    pos_change = pd.to_numeric(out["Position_Change"], errors="coerce").astype(
        "float64"
    )
    year = pd.to_numeric(out["Year"], errors="coerce").astype("int64")

    total_laps_est = (lap / progress).replace([np.inf, -np.inf], np.nan)
    laps_remaining = (total_laps_est - lap).clip(lower=0, upper=90).fillna(0)
    out["laps_remaining_est"] = laps_remaining.astype("float32")
    out["tyre_life_frac"] = (
        (tyre_life / total_laps_est.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    out["stint_lap_frac"] = (
        (tyre_life / (lap + 1.0))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    out["late_race_old_tyre"] = (progress * np.log1p(tyre_life)).astype("float32")
    out["position_norm"] = ((pos - 1.0) / 19.0).clip(0, 1).astype("float32")
    out["midpack_traffic_proxy"] = (
        (1.0 - ((pos - 10.5).abs() / 9.5)).clip(0, 1).astype("float32")
    )

    work = pd.DataFrame(
        {
            "_pos": np.arange(n, dtype=np.int64),
            "Year": year.to_numpy(),
            "Race": race_s.to_numpy(),
            "Driver": driver_s.to_numpy(),
            "Compound": comp_s.to_numpy(),
            "LapNumber": lap.to_numpy(),
            "Stint": stint.to_numpy(),
            "LapTime": lap_time.to_numpy(),
            "LapTime_Delta": lap_delta.to_numpy(),
            "PitStop": pit.to_numpy(),
            "Position": pos.to_numpy(),
            "TyreLife": tyre_life.to_numpy(),
            "RaceProgress": progress.to_numpy(),
        }
    ).sort_values(
        ["Year", "Race", "Driver", "Stint", "LapNumber", "_pos"], kind="mergesort"
    )

    def backfill(values):
        arr = np.empty(n, dtype="float64")
        arr[work["_pos"].to_numpy()] = np.asarray(values, dtype="float64")
        return arr

    drv_stint_keys = ["Year", "Race", "Driver", "Stint"]
    gds = work.groupby(drv_stint_keys, sort=False, observed=False)
    prev_laptime = gds["LapTime"].shift(1)
    prev_pit = gds["PitStop"].shift(1).fillna(0)
    stint_lap_diff = (work["LapTime"] - prev_laptime).clip(-15, 15)
    clean_delta_mask = (
        work["PitStop"].eq(0) & prev_pit.eq(0) & stint_lap_diff.between(-8, 8)
    )
    clean_delta = stint_lap_diff.where(clean_delta_mask, 0.0)
    clean_count = clean_delta_mask.astype("float64")
    cum_delta = clean_delta.groupby(
        [work[k] for k in drv_stint_keys], sort=False
    ).cumsum()
    cum_count = clean_count.groupby(
        [work[k] for k in drv_stint_keys], sort=False
    ).cumsum()
    stint_deg_slope = (cum_delta / cum_count.replace(0, np.nan)).fillna(0).clip(-3, 6)

    race_lap_keys = ["Year", "Race", "LapNumber"]
    clean_laptime = work["LapTime"].where(work["PitStop"].eq(0) & prev_pit.eq(0))
    race_lap_clean_med = clean_laptime.groupby(
        [work[k] for k in race_lap_keys], sort=False
    ).transform("median")
    race_clean_med = clean_laptime.groupby(
        [work["Year"], work["Race"]], sort=False
    ).transform("median")
    lap_baseline = race_lap_clean_med.fillna(race_clean_med).fillna(
        work["LapTime"].median()
    )
    lap_pace_delta_sorted = (work["LapTime"] - lap_baseline).clip(-20, 80)
    pace_rank_sorted = (
        work.groupby(race_lap_keys, sort=False, observed=False)["LapTime"]
        .rank(method="average", pct=True)
        .fillna(0.5)
    )

    outlap_penalty = lap_pace_delta_sorted.where(
        prev_pit.eq(1) & lap_pace_delta_sorted.between(5, 90)
    )
    event_tbl = pd.DataFrame(
        {
            "Year": work["Year"].to_numpy(),
            "Race": work["Race"].to_numpy(),
            "LapNumber": work["LapNumber"].to_numpy(),
            "sum": outlap_penalty.fillna(0).to_numpy(),
            "cnt": outlap_penalty.notna().astype("int16").to_numpy(),
        }
    )
    race_lap_events = event_tbl.groupby(
        ["Year", "Race", "LapNumber"], as_index=False, sort=False
    ).agg({"sum": "sum", "cnt": "sum"})
    race_lap_events = race_lap_events.sort_values(
        ["Year", "Race", "LapNumber"], kind="mergesort"
    )
    gr = race_lap_events.groupby(["Year", "Race"], sort=False, observed=False)
    prior_sum = gr["sum"].cumsum() - race_lap_events["sum"]
    prior_cnt = gr["cnt"].cumsum() - race_lap_events["cnt"]
    race_lap_events["pit_loss_prior_same_race"] = (
        prior_sum / prior_cnt.replace(0, np.nan)
    ).clip(8, 80)
    work = work.merge(
        race_lap_events[["Year", "Race", "LapNumber", "pit_loss_prior_same_race"]],
        on=["Year", "Race", "LapNumber"],
        how="left",
        sort=False,
    )

    yr_events = event_tbl.groupby(["Race", "Year"], as_index=False, sort=False).agg(
        {"sum": "sum", "cnt": "sum"}
    )
    yr_events = yr_events.sort_values(["Race", "Year"], kind="mergesort")
    gyr = yr_events.groupby("Race", sort=False, observed=False)
    hist_sum = gyr["sum"].cumsum() - yr_events["sum"]
    hist_cnt = gyr["cnt"].cumsum() - yr_events["cnt"]
    yr_events["pit_loss_hist_race"] = (hist_sum / hist_cnt.replace(0, np.nan)).clip(
        8, 80
    )

    global_yr = (
        event_tbl.groupby("Year", as_index=False, sort=False)
        .agg({"sum": "sum", "cnt": "sum"})
        .sort_values("Year", kind="mergesort")
    )
    global_yr["pit_loss_hist_global"] = (
        (global_yr["sum"].cumsum() - global_yr["sum"])
        / (global_yr["cnt"].cumsum() - global_yr["cnt"]).replace(0, np.nan)
    ).clip(8, 80)

    work = work.merge(
        yr_events[["Race", "Year", "pit_loss_hist_race"]],
        on=["Race", "Year"],
        how="left",
        sort=False,
    )
    work = work.merge(
        global_yr[["Year", "pit_loss_hist_global"]], on="Year", how="left", sort=False
    )
    pit_loss_est_sorted = (
        work["pit_loss_prior_same_race"]
        .fillna(work["pit_loss_hist_race"])
        .fillna(work["pit_loss_hist_global"])
        .fillna(22.0)
        .clip(8, 80)
    )

    slope = backfill(stint_deg_slope)
    lap_pace_delta_arr = backfill(lap_pace_delta_sorted)
    pace_rank_arr = backfill(pace_rank_sorted)
    pit_loss_est = backfill(pit_loss_est_sorted)

    slope_pos = np.maximum(slope, 0)
    lap_delta_pos = np.maximum(lap_delta.clip(-10, 20).fillna(0).to_numpy(), 0)
    pace_slow = np.maximum(lap_pace_delta_arr, 0)
    traffic = out["midpack_traffic_proxy"].to_numpy(dtype="float64")
    remaining = laps_remaining.to_numpy(dtype="float64")

    stay_out_cost = (
        slope_pos * (1.0 + np.log1p(tyre_life.fillna(0).to_numpy()) / 5.0)
        + 0.20 * lap_delta_pos
        + 0.015 * np.maximum(cum_deg.fillna(0).to_numpy(), 0)
    )
    traffic_cost = pace_slow * (0.25 + 0.75 * traffic)
    tyre_offset = slope_pos * np.log1p(tyre_life.fillna(0).to_numpy())
    warmup_penalty = 0.18 * pit_loss_est
    expected_undercut_gain = (
        stay_out_cost
        + 0.35 * traffic_cost
        + tyre_offset / np.maximum(remaining + 1.0, 1.0)
    )

    out["stint_deg_slope_prior"] = slope.astype("float32")
    out["lap_pace_delta_clean"] = lap_pace_delta_arr.astype("float32")
    out["lap_pace_rank_pct"] = pace_rank_arr.astype("float32")
    out["pit_loss_est_prior"] = pit_loss_est.astype("float32")
    out["pit_loss_per_lap_remaining"] = (
        pit_loss_est / np.maximum(remaining + 1.0, 1.0)
    ).astype("float32")
    out["traffic_cost_proxy"] = traffic_cost.astype("float32")
    out["stay_out_cost_1lap"] = stay_out_cost.astype("float32")
    out["tyre_offset_proxy"] = tyre_offset.astype("float32")
    out["expected_undercut_gain_now"] = expected_undercut_gain.astype("float32")
    out["pit_now_net_gain"] = (
        expected_undercut_gain * np.maximum(remaining, 1.0) - pit_loss_est
    ).astype("float32")
    out["pit_now_vs_next_lap_margin"] = (
        stay_out_cost
        + 0.25 * traffic_cost
        - warmup_penalty / np.maximum(remaining + 1.0, 1.0)
    ).astype("float32")
    out["clean_air_window_proxy"] = (
        traffic * pace_slow * (1.0 - out["position_norm"].to_numpy(dtype="float64"))
    ).astype("float32")
    out["undercut_pressure_index"] = (
        out["pit_now_vs_next_lap_margin"].to_numpy(dtype="float64")
        + 0.10 * np.maximum(pos_change.fillna(0).to_numpy(), 0)
        + 0.15 * pit.astype("float64")
    ).astype("float32")

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
        
        "research_hypotheses_llm_claimed_used": ["000359"],
        "research_usage_note": "Verified assigned hypothesis 000359.",
    }, sort_keys=True))


# AIDE executes generated code via exec with a custom globals dict, where __name__ is not
# guaranteed to be "__main__". Call main() directly so the wrapper actually runs
# inside the interpreter.
main()
