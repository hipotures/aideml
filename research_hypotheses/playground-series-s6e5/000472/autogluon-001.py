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
    year_num = pd.to_numeric(out["Year"], errors="coerce")
    year_i = year_num.fillna(-1).astype("int16")

    race_norm = (
        out["Race"]
        .astype(str)
        .str.upper()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.strip()
    ).replace(
        {
            "MEXICAN GRAND PRIX": "MEXICO CITY GRAND PRIX",
            "BRAZILIAN GRAND PRIX": "SAO PAULO GRAND PRIX",
            "SAO PAULO GP": "SAO PAULO GRAND PRIX",
            "UNITED STATES GP": "UNITED STATES GRAND PRIX",
        }
    )

    alloc_rows = []
    for y, triple, races in [
        (
            2022,
            (1, 2, 3),
            [
                "BAHRAIN GRAND PRIX",
                "SPANISH GRAND PRIX",
                "BRITISH GRAND PRIX",
                "DUTCH GRAND PRIX",
                "JAPANESE GRAND PRIX",
            ],
        ),
        (
            2022,
            (2, 3, 4),
            [
                "SAUDI ARABIAN GRAND PRIX",
                "EMILIA ROMAGNA GRAND PRIX",
                "MIAMI GRAND PRIX",
                "FRENCH GRAND PRIX",
                "HUNGARIAN GRAND PRIX",
                "BELGIAN GRAND PRIX",
                "ITALIAN GRAND PRIX",
                "UNITED STATES GRAND PRIX",
                "MEXICO CITY GRAND PRIX",
                "SAO PAULO GRAND PRIX",
            ],
        ),
        (2022, (2, 3, 5), ["AUSTRALIAN GRAND PRIX"]),
        (
            2022,
            (3, 4, 5),
            [
                "MONACO GRAND PRIX",
                "AZERBAIJAN GRAND PRIX",
                "CANADIAN GRAND PRIX",
                "AUSTRIAN GRAND PRIX",
                "SINGAPORE GRAND PRIX",
                "ABU DHABI GRAND PRIX",
            ],
        ),
        (
            2023,
            (1, 2, 3),
            [
                "BAHRAIN GRAND PRIX",
                "SPANISH GRAND PRIX",
                "BRITISH GRAND PRIX",
                "DUTCH GRAND PRIX",
                "JAPANESE GRAND PRIX",
                "QATAR GRAND PRIX",
            ],
        ),
        (
            2023,
            (2, 3, 4),
            [
                "SAUDI ARABIAN GRAND PRIX",
                "AUSTRALIAN GRAND PRIX",
                "MIAMI GRAND PRIX",
                "BELGIAN GRAND PRIX",
                "UNITED STATES GRAND PRIX",
                "SAO PAULO GRAND PRIX",
            ],
        ),
        (
            2023,
            (3, 4, 5),
            [
                "AZERBAIJAN GRAND PRIX",
                "EMILIA ROMAGNA GRAND PRIX",
                "MONACO GRAND PRIX",
                "CANADIAN GRAND PRIX",
                "AUSTRIAN GRAND PRIX",
                "HUNGARIAN GRAND PRIX",
                "ITALIAN GRAND PRIX",
                "SINGAPORE GRAND PRIX",
                "MEXICO CITY GRAND PRIX",
                "LAS VEGAS GRAND PRIX",
                "ABU DHABI GRAND PRIX",
            ],
        ),
        (
            2024,
            (1, 2, 3),
            [
                "BAHRAIN GRAND PRIX",
                "JAPANESE GRAND PRIX",
                "SPANISH GRAND PRIX",
                "BRITISH GRAND PRIX",
                "DUTCH GRAND PRIX",
                "QATAR GRAND PRIX",
            ],
        ),
        (
            2024,
            (2, 3, 4),
            [
                "SAUDI ARABIAN GRAND PRIX",
                "CHINESE GRAND PRIX",
                "MIAMI GRAND PRIX",
                "BELGIAN GRAND PRIX",
                "UNITED STATES GRAND PRIX",
            ],
        ),
        (
            2024,
            (3, 4, 5),
            [
                "AUSTRALIAN GRAND PRIX",
                "EMILIA ROMAGNA GRAND PRIX",
                "MONACO GRAND PRIX",
                "CANADIAN GRAND PRIX",
                "AUSTRIAN GRAND PRIX",
                "HUNGARIAN GRAND PRIX",
                "ITALIAN GRAND PRIX",
                "AZERBAIJAN GRAND PRIX",
                "SINGAPORE GRAND PRIX",
                "MEXICO CITY GRAND PRIX",
                "SAO PAULO GRAND PRIX",
                "LAS VEGAS GRAND PRIX",
                "ABU DHABI GRAND PRIX",
            ],
        ),
        (
            2025,
            (1, 2, 3),
            [
                "JAPANESE GRAND PRIX",
                "BAHRAIN GRAND PRIX",
                "SPANISH GRAND PRIX",
                "QATAR GRAND PRIX",
            ],
        ),
        (
            2025,
            (2, 3, 4),
            [
                "CHINESE GRAND PRIX",
                "BRITISH GRAND PRIX",
                "DUTCH GRAND PRIX",
                "SAO PAULO GRAND PRIX",
            ],
        ),
        (
            2025,
            (3, 4, 5),
            [
                "AUSTRALIAN GRAND PRIX",
                "SAUDI ARABIAN GRAND PRIX",
                "MIAMI GRAND PRIX",
                "AUSTRIAN GRAND PRIX",
                "HUNGARIAN GRAND PRIX",
                "ITALIAN GRAND PRIX",
                "SINGAPORE GRAND PRIX",
                "LAS VEGAS GRAND PRIX",
                "ABU DHABI GRAND PRIX",
            ],
        ),
        (
            2025,
            (4, 5, 6),
            [
                "EMILIA ROMAGNA GRAND PRIX",
                "MONACO GRAND PRIX",
                "CANADIAN GRAND PRIX",
                "AZERBAIJAN GRAND PRIX",
            ],
        ),
        (2025, (1, 3, 4), ["BELGIAN GRAND PRIX", "UNITED STATES GRAND PRIX"]),
        (2025, (2, 4, 5), ["MEXICO CITY GRAND PRIX"]),
    ]:
        for race in races:
            alloc_rows.append((y, race, triple[0], triple[1], triple[2]))

    lookup = pd.DataFrame(
        alloc_rows,
        columns=["Year", "Race_Norm", "Alloc_Hard_C", "Alloc_Medium_C", "Alloc_Soft_C"],
    )
    keys = pd.DataFrame(
        {
            "_row": np.arange(len(out)),
            "Year": year_i.to_numpy(),
            "Race_Norm": race_norm.to_numpy(),
        }
    )
    keys = keys.merge(
        lookup, how="left", on=["Year", "Race_Norm"], sort=False
    ).sort_values("_row", kind="stable")

    hard_c = keys["Alloc_Hard_C"].to_numpy(dtype=float)
    med_c = keys["Alloc_Medium_C"].to_numpy(dtype=float)
    soft_c = keys["Alloc_Soft_C"].to_numpy(dtype=float)
    alloc_known = np.isfinite(hard_c) & np.isfinite(med_c) & np.isfinite(soft_c)

    compound = out["Compound"].astype(str).str.upper().str.strip()
    is_hard = compound.eq("HARD").to_numpy()
    is_med = compound.eq("MEDIUM").to_numpy()
    is_soft = compound.eq("SOFT").to_numpy()
    is_int = compound.eq("INTERMEDIATE").to_numpy()
    is_wet = compound.eq("WET").to_numpy()

    c_num = np.select(
        [is_hard, is_med, is_soft], [hard_c, med_c, soft_c], default=np.nan
    ).astype(float)
    c_series = pd.Series(c_num, index=out.index)

    abs_code = pd.Series(np.full(len(out), "UNKNOWN", dtype=object), index=out.index)
    slick = is_hard | is_med | is_soft
    known_c = slick & c_series.notna().to_numpy()
    abs_code.loc[known_c] = "C" + c_series.loc[known_c].round().astype("int16").astype(
        str
    )
    abs_code.loc[slick & ~known_c] = "UNKNOWN_SLICK"
    abs_code.loc[is_int] = "INTERMEDIATE"
    abs_code.loc[is_wet] = "WET"

    adjacent = alloc_known & ((med_c - hard_c) == 1) & ((soft_c - med_c) == 1)
    alloc_mean_c = (hard_c + med_c + soft_c) / 3.0

    life_lookup = np.array([42.0, 38.0, 33.0, 27.0, 21.0, 16.0, 12.0])
    alloc_life = {}
    for name, arr in [
        ("Hard", hard_c),
        ("Medium", med_c),
        ("Soft", soft_c),
        ("Current", c_num),
    ]:
        idx = np.full(len(out), -1, dtype=np.int16)
        valid = np.isfinite(arr)
        idx[valid] = arr[valid].astype(np.int16)
        vals = np.full(len(out), np.nan, dtype=float)
        valid_idx = (idx >= 0) & (idx < len(life_lookup))
        vals[valid_idx] = life_lookup[idx[valid_idx]]
        alloc_life[name] = vals

    c_life = alloc_life["Current"]
    c_life[is_int] = 24.0
    c_life[is_wet] = 18.0
    alloc_mean_life = (
        alloc_life["Hard"] + alloc_life["Medium"] + alloc_life["Soft"]
    ) / 3.0

    out["Tyre_C_Code"] = abs_code
    out["Tyre_C_Number"] = c_num
    out["Tyre_Softness_Rank"] = c_num
    out["Tyre_Hardness_Rank"] = np.where(np.isfinite(c_num), 7.0 - c_num, np.nan)
    out["Tyre_Allocation_Hard_C"] = hard_c
    out["Tyre_Allocation_Medium_C"] = med_c
    out["Tyre_Allocation_Soft_C"] = soft_c
    out["Tyre_Allocation_Mean_C"] = alloc_mean_c
    out["Tyre_Allocation_Span_C"] = soft_c - hard_c
    out["Tyre_Allocation_Adjacent"] = adjacent.astype("int8")
    out["Tyre_Allocation_NonAdjacent"] = (alloc_known & ~adjacent).astype("int8")
    out["Tyre_Allocation_Unknown"] = (~alloc_known).astype("int8")
    out["Compound_Relative_Slot"] = np.select(
        [is_hard, is_med, is_soft], [0.0, 1.0, 2.0], default=np.nan
    )
    out["Compound_Is_Slick"] = slick.astype("int8")

    out["Tyre_Era_2022_First18"] = year_num.eq(2022).astype("int8")
    out["Tyre_Era_2023_C0_Reset"] = year_num.eq(2023).astype("int8")
    out["Tyre_Era_2024_No_C0"] = year_num.eq(2024).astype("int8")
    out["Tyre_Era_2025_C6_Range"] = year_num.eq(2025).astype("int8")
    out["Tyre_Range_Min_C"] = np.where(year_num.eq(2023), 0.0, 1.0)
    out["Tyre_Range_Max_C"] = np.where(year_num.eq(2025), 6.0, 5.0)

    known_series = pd.Series(alloc_known, index=out.index)
    h_txt = pd.Series(hard_c, index=out.index).round().astype("Int64").astype(str)
    m_txt = pd.Series(med_c, index=out.index).round().astype("Int64").astype(str)
    s_txt = pd.Series(soft_c, index=out.index).round().astype("Int64").astype(str)
    alloc_key = pd.Series("UNKNOWN_ALLOC", index=out.index, dtype=object)
    alloc_key.loc[known_series] = (
        "Y"
        + year_i.astype(str).loc[known_series]
        + "_C"
        + h_txt.loc[known_series]
        + "_C"
        + m_txt.loc[known_series]
        + "_C"
        + s_txt.loc[known_series]
    )
    out["Tyre_RaceYear_Allocation_Key"] = alloc_key

    tyre_life = pd.to_numeric(out["TyreLife"], errors="coerce").to_numpy(dtype=float)
    degradation = pd.to_numeric(
        out["Cumulative_Degradation"], errors="coerce"
    ).to_numpy(dtype=float)
    eps = 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        out["Tyre_AbsLife_Prior"] = c_life
        out["Tyre_AllocMeanLife_Prior"] = alloc_mean_life
        out["TyreLife_AbsLife_Ratio"] = tyre_life / (c_life + eps)
        out["TyreLife_AbsLife_Remaining"] = c_life - tyre_life
        out["TyreLife_AllocMeanLife_Ratio"] = tyre_life / (alloc_mean_life + eps)
        out["TyreLife_AllocMeanLife_Remaining"] = alloc_mean_life - tyre_life
        out["Degradation_Per_AbsLife"] = degradation / (c_life + eps)
        out["Degradation_Per_AllocMeanLife"] = degradation / (alloc_mean_life + eps)
        out["TyreLife_x_CSoftness"] = tyre_life * c_num
        out["Degradation_x_CSoftness"] = degradation * c_num

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
        
        "research_hypotheses_llm_claimed_used": ["000472"],
        "research_usage_note": "Verified assigned hypothesis 000472.",
    }, sort_keys=True))


# AIDE executes generated code via exec with a custom globals dict, where __name__ is not
# guaranteed to be "__main__". Call main() directly so the wrapper actually runs
# inside the interpreter.
main()
