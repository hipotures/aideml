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

AIDE_AG_CONFIG = {'fit_args': {'auto_stack': False,
              'fit_weighted_ensemble': True,
              'num_bag_folds': 3,
              'num_bag_sets': None,
              'save_space': True},
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
    from sklearn.model_selection import GroupKFold

    out = df.copy()
    n_test = 188165
    n_train = max(len(out) - n_test, 0)

    comp = out["Compound"].astype(str).str.upper().str.strip()
    wet_mask = comp.isin(["WET", "INTERMEDIATE"]).to_numpy()

    for col in ["Compound", "Race", "Driver"]:
        s = out[col].astype(str).str.upper().str.strip()
        codes, _ = pd.factorize(s, sort=True)
        out[f"{col}_code"] = codes.astype("int32")
        out[f"{col}_freq"] = s.map(s.value_counts(normalize=True)).astype("float32")

    race_key = out["Year"].astype(str) + "|" + out["Race"].astype(str)
    max_lap = (
        out.groupby(race_key)["LapNumber"]
        .transform("max")
        .astype("float32")
        .clip(lower=1)
    )
    out["race_max_lap"] = max_lap
    out["laps_remaining_est"] = (max_lap - out["LapNumber"]).astype("float32")
    out["lap_frac_est"] = (out["LapNumber"] / max_lap).astype("float32")
    out["tyre_life_frac"] = (out["TyreLife"] / max_lap).astype("float32")
    out["degradation_per_tyre_lap"] = (
        (out["Cumulative_Degradation"] / out["TyreLife"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )
    out["is_wet_or_inter"] = wet_mask.astype("int8")

    tmp = out.iloc[:n_train][["Year", "Race", "Driver", "LapNumber", "PitStop"]].copy()
    tmp["_pos"] = np.arange(n_train)
    tmp["_pit"] = pd.to_numeric(tmp["PitStop"], errors="coerce").fillna(0).clip(0, 1)
    tmp = tmp.sort_values(["Year", "Race", "Driver", "LapNumber"], kind="mergesort")
    gcols = ["Year", "Race", "Driver"]
    tmp["_total"] = tmp.groupby(gcols)["_pit"].transform("sum")
    tmp["_cum"] = tmp.groupby(gcols)["_pit"].cumsum()
    y_aux = (tmp["_total"] - tmp["_cum"]).clip(0, 3).astype("int8")
    y_aux = y_aux.set_axis(tmp["_pos"]).sort_index().to_numpy()

    feature_cols = [
        "Year",
        "LapNumber",
        "LapTime (s)",
        "LapTime_Delta",
        "PitStop",
        "Position",
        "Position_Change",
        "RaceProgress",
        "Stint",
        "TyreLife",
        "Cumulative_Degradation",
        "Compound_code",
        "Race_code",
        "Driver_code",
        "Compound_freq",
        "Race_freq",
        "Driver_freq",
        "race_max_lap",
        "laps_remaining_est",
        "lap_frac_est",
        "tyre_life_frac",
        "degradation_per_tyre_lap",
        "is_wet_or_inter",
    ]
    X = (
        out[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-1)
        .astype("float32")
    )
    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]

    groups = (
        out.iloc[:n_train]["Year"].astype(str)
        + "|"
        + out.iloc[:n_train]["Race"].astype(str)
    ).to_numpy()
    n_splits = int(min(5, max(2, pd.Series(groups).nunique())))
    aux_prob = np.zeros((len(out), 4), dtype="float32")
    test_model_sum = np.zeros((len(X_test), 4), dtype="float32")
    test_fallback_sum = np.zeros((len(X_test), 4), dtype="float32")

    phase = (
        pd.cut(out["RaceProgress"], [-0.01, 0.25, 0.50, 0.75, 1.01], labels=False)
        .fillna(0)
        .astype("int8")
    )
    stint_bin = (
        pd.to_numeric(out["Stint"], errors="coerce").fillna(0).clip(0, 4).astype("int8")
    )
    fb_key = comp + "|" + phase.astype(str) + "|" + stint_bin.astype(str)
    global_prob = np.bincount(y_aux, minlength=4).astype("float32") + 1.0
    global_prob = global_prob / global_prob.sum()

    try:
        from lightgbm import LGBMClassifier

        model_factory = lambda seed: LGBMClassifier(
            objective="multiclass",
            num_class=4,
            n_estimators=80,
            learning_rate=0.07,
            num_leaves=31,
            min_child_samples=200,
            subsample=0.9,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model_factory = lambda seed: HistGradientBoostingClassifier(
            max_iter=80,
            learning_rate=0.07,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            random_state=seed,
        )

    splitter = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(splitter.split(X_train, y_aux, groups)):
        model = model_factory(903 + fold)
        model.fit(X_train.iloc[tr], y_aux[tr])

        va_prob = np.zeros((len(va), 4), dtype="float32")
        pred = model.predict_proba(X_train.iloc[va])
        for j, cls in enumerate(model.classes_):
            va_prob[:, int(cls)] = pred[:, j]

        te_prob = np.zeros((len(X_test), 4), dtype="float32")
        if len(X_test):
            pred = model.predict_proba(X_test)
            for j, cls in enumerate(model.classes_):
                te_prob[:, int(cls)] = pred[:, j]
            test_model_sum += te_prob / n_splits

        tr_wet = wet_mask[:n_train][tr]
        counts = pd.DataFrame(
            {"key": fb_key.iloc[:n_train].iloc[tr].to_numpy(), "y": y_aux[tr]}
        )
        counts = counts.loc[tr_wet]
        table = (
            pd.crosstab(counts["key"], counts["y"]) if len(counts) else pd.DataFrame()
        )
        for k in range(4):
            if k not in table.columns:
                table[k] = 0
        table = table[[0, 1, 2, 3]].astype("float32")
        denom = table.sum(axis=1).to_numpy()[:, None] + 20.0
        probs = (table.to_numpy() + 20.0 * global_prob) / denom
        prob_table = pd.DataFrame(probs, index=table.index, columns=[0, 1, 2, 3])

        va_fb = np.tile(global_prob, (len(va), 1)).astype("float32")
        mapped = fb_key.iloc[:n_train].iloc[va].map(prob_table.to_dict("index"))
        for i, item in enumerate(mapped):
            if isinstance(item, dict):
                va_fb[i] = [item[0], item[1], item[2], item[3]]
        va_wet = wet_mask[:n_train][va]
        va_prob[va_wet] = va_fb[va_wet]
        aux_prob[va] = va_prob

        if len(X_test):
            te_fb = np.tile(global_prob, (len(X_test), 1)).astype("float32")
            mapped = fb_key.iloc[n_train:].map(prob_table.to_dict("index"))
            for i, item in enumerate(mapped):
                if isinstance(item, dict):
                    te_fb[i] = [item[0], item[1], item[2], item[3]]
            test_fallback_sum += te_fb / n_splits

    if len(X_test):
        test_prob = test_model_sum
        test_prob[wet_mask[n_train:]] = test_fallback_sum[wet_mask[n_train:]]
        aux_prob[n_train:] = test_prob

    aux_prob = np.nan_to_num(aux_prob, nan=0.25, posinf=0.25, neginf=0.25)
    row_sums = aux_prob.sum(axis=1, keepdims=True)
    aux_prob = aux_prob / np.where(row_sums <= 0, 1.0, row_sums)

    for k in range(4):
        out[f"latent_remaining_stops_p{k}"] = aux_prob[:, k].astype("float32")
    out["latent_remaining_stops_argmax"] = aux_prob.argmax(axis=1).astype("int8")
    out["latent_remaining_stops_entropy"] = (
        -(aux_prob * np.log(np.clip(aux_prob, 1e-6, 1.0))).sum(axis=1)
    ).astype("float32")
    out["latent_remaining_stops_wet_fallback"] = wet_mask.astype("int8")

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
    return _positive_probability_from_proba(proba)


def _positive_probability_from_proba(proba) -> pd.Series:
    if isinstance(proba, pd.Series):
        return proba.reset_index(drop=True)
    for positive_class in (1, 1.0, "1", "1.0", True):
        if positive_class in proba.columns:
            return proba[positive_class].reset_index(drop=True)
    return proba.iloc[:, -1].reset_index(drop=True)


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


def _save_prediction_artifact(frame: pd.DataFrame, working_dir: Path, filename: str) -> Path:
    working_path = working_dir / filename
    frame.to_csv(working_path, index=False)
    artifact_dir = _artifact_dir(working_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / filename
    if artifact_path.resolve() != working_path.resolve():
        shutil.copy2(working_path, artifact_path)
    return working_path


def _save_autogluon_prediction_artifacts(
    predictor: TabularPredictor,
    *,
    train_target: pd.Series,
    test_ids: pd.Series,
    test_pred: pd.Series,
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
        oof_proba = predictor.predict_proba_oof(
            transformed=False,
            as_multiclass=True,
        )
        oof_pred = _positive_probability_from_proba(oof_proba)
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
                valid_data.drop(columns=[target_col]),
            )
            metric_value = float(roc_auc_score(valid_data[target_col], valid_pred))
            lower_is_better = False
        else:
            scores = predictor.evaluate(valid_data, silent=True)
            metric_value = float(scores.get(eval_metric))
            lower_is_better = False

        test_pred = _positive_probability(predictor, test_model)
        prediction_artifacts = _save_autogluon_prediction_artifacts(
            predictor,
            train_target=y_train,
            test_ids=test_df[id_col],
            test_pred=test_pred,
            working_dir=working_dir,
            id_col=id_col,
            target_col=target_col,
            valid_data=valid_data,
            valid_pred=valid_pred,
        )
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
        "models": model_records,
        "prediction_artifacts": prediction_artifacts,
    }
    print(f"Validation {eval_metric}: {metric_value:.6f}")
    print("Submission saved successfully.")
    print(RESULT_MARKER + " " + json.dumps({
        "is_bug": False,
        "summary": summary,
        "metric": metric_value,
        "lower_is_better": lower_is_better,
        "run_stats": run_stats,
        
        "research_hypotheses_llm_claimed_used": ["000903"],
        "research_usage_note": "Verified assigned hypothesis 000903.",
    }, sort_keys=True))


# AIDE executes generated code via exec with a custom globals dict, where __name__ is not
# guaranteed to be "__main__". Call main() directly so the wrapper actually runs
# inside the interpreter.
main()
