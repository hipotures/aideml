from __future__ import annotations

import ast
import json
import pprint
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .utils.config import Config

AGENT_MODE = "autogluon_preprocess"
RESULT_MARKER = "AIDE_RESULT_JSON:"
FORBIDDEN_SPLIT_MARKER = "__is_train__"
FORBIDDEN_ROW_ID = "__aide_row_id__"
BASELINE_PLAN_PREFIX = "AutoGluon raw baseline"


def baseline_preprocess_source() -> str:
    return "def preprocess(df: pd.DataFrame) -> pd.DataFrame:\n    return df.copy()\n"


def sanitize_preprocess_prompt_text(
    text: Any,
    *,
    unavailable_columns: list[str | None] | tuple[str | None, ...] = (),
) -> str:
    """Remove references to columns that are not passed to preprocess(df)."""
    blocked = [str(col) for col in unavailable_columns if col]
    if not blocked:
        return str(text or "")

    patterns = [
        re.compile(rf"(?<![A-Za-z0-9_]){re.escape(col)}(?![A-Za-z0-9_])")
        for col in blocked
    ]
    kept: list[str] = []
    for line in str(text or "").splitlines():
        if any(pattern.search(line) for pattern in patterns):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


_PITSTOP_LEAKAGE_PATTERNS = (
    "next_pitstop",
    "next_pit_stop",
    "next_pit",
    "next pitstop",
    "next pit stop",
    "pitstop_known",
)


def is_autogluon_preprocess_mode(cfg: Config) -> bool:
    return getattr(cfg.agent, "mode", "legacy") == AGENT_MODE


def _source_for_first_preprocess_function(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "preprocess":
            return ast.get_source_segment(source, node)
    return None


def _python_code_block_candidates(text: str) -> list[str]:
    matches = re.findall(r"```(?:python)?\n*(.*?)\n*```", text, re.DOTALL)
    return [match for match in matches if match.strip()]


def extract_preprocess_source(text: str) -> str:
    source = _source_for_first_preprocess_function(text)
    if source:
        return source.strip()

    for candidate in _python_code_block_candidates(text):
        source = _source_for_first_preprocess_function(candidate)
        if source:
            return source.strip()
    raise ValueError("Expected Python code containing `def preprocess(df)`.")


def validate_preprocess_source(source: str, *, target_col: str | None = None) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"Invalid preprocess Python: {exc}") from exc

    funcs = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "preprocess"
    ]
    if len(funcs) != 1:
        raise ValueError("Code must contain exactly one top-level `def preprocess(df)`.")

    func = funcs[0]
    if not func.args.args or func.args.args[0].arg != "df":
        raise ValueError("`preprocess` must accept `df` as its first argument.")

    reasons: list[str] = []
    lowered = source.lower()
    for pattern in _PITSTOP_LEAKAGE_PATTERNS:
        if pattern in lowered:
            reasons.append(f"suspicious token '{pattern}'")

    lines = source.splitlines()
    for idx, line in enumerate(lines):
        normalized_line = re.sub(r"\s+", "", line.lower())
        if (
            "shift(-1" in normalized_line
            or "shift(periods=-1" in normalized_line
        ):
            window = "\n".join(lines[max(0, idx - 4) : idx + 5]).lower()
            if "pitstop" in window or "pit_stop" in window:
                reasons.append("future PitStop shift(-1)")

    forbidden_columns = [FORBIDDEN_ROW_ID, FORBIDDEN_SPLIT_MARKER]
    if target_col:
        forbidden_columns.append(target_col)
    for node in ast.walk(func):
        if not isinstance(node, ast.Constant):
            continue
        for col in forbidden_columns:
            if node.value == col:
                reasons.append(f"forbidden column '{col}' referenced in preprocess")

    if reasons:
        raise ValueError("Preprocess target leakage risk: " + "; ".join(dict.fromkeys(reasons)))


def _container(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def resolve_autogluon_settings(cfg: Config) -> dict[str, Any]:
    ag = cfg.agent.autogluon
    profiles = dict(_container(getattr(ag, "profiles", {})) or {})
    profile = getattr(ag, "profile", "full_boost")
    profile_settings = profiles.get(profile)
    if profile_settings is None:
        known = ", ".join(sorted(profiles))
        raise ValueError(
            f"Unknown AutoGluon profile {profile!r}. Expected one of: {known}"
        )
    if isinstance(profile_settings, list):
        profile_settings = {"included_model_types": profile_settings}
    if not isinstance(profile_settings, dict):
        raise ValueError(
            f"AutoGluon profile {profile!r} must be a mapping or a model list."
        )

    settings = {
        "presets": ag.presets,
        "time_limit": int(ag.time_limit),
        "validation_fraction": float(ag.validation_fraction),
        "seed": int(ag.seed),
        "eval_metric": ag.eval_metric,
        "included_model_types": None,
    }
    validation_strategy = getattr(ag, "validation_strategy", None)
    if validation_strategy is not None:
        settings["validation_strategy"] = validation_strategy
    use_gpu = getattr(ag, "use_gpu", None)
    if use_gpu is not None:
        settings["use_gpu"] = bool(use_gpu)
    hyperparameters = _container(getattr(ag, "hyperparameters", None))
    if hyperparameters is not None:
        settings["hyperparameters"] = dict(hyperparameters or {})
    fit_args = _container(getattr(ag, "fit_args", None))
    if fit_args is not None:
        settings["fit_args"] = dict(fit_args or {})
    settings.update(dict(profile_settings))

    explicit = _container(ag.included_model_types)
    if explicit is not None:
        settings["included_model_types"] = list(explicit)

    if settings["included_model_types"] is not None:
        settings["included_model_types"] = list(settings["included_model_types"])
    settings["time_limit"] = int(settings["time_limit"])
    settings["validation_fraction"] = float(settings["validation_fraction"])
    settings["seed"] = int(settings["seed"])
    if "use_gpu" in settings and settings["use_gpu"] is not None:
        settings["use_gpu"] = bool(settings["use_gpu"])
    if "validation_strategy" in settings and settings["validation_strategy"] is not None:
        settings["validation_strategy"] = str(settings["validation_strategy"])
    if "hyperparameters" in settings:
        settings["hyperparameters"] = dict(settings.get("hyperparameters") or {})
    if "fit_args" in settings:
        settings["fit_args"] = dict(settings.get("fit_args") or {})
    if settings.get("validation_strategy") not in {None, "holdout", "autogluon"}:
        raise ValueError(
            "AutoGluon validation_strategy must be one of: holdout, autogluon"
        )
    return settings


def resolve_autogluon_included_model_types(cfg: Config) -> list[str]:
    included = resolve_autogluon_settings(cfg).get("included_model_types")
    return list(included or [])


def _profile_settings_for_cfg(cfg: Config) -> dict[str, Any]:
    ag = cfg.agent.autogluon
    profiles = dict(_container(getattr(ag, "profiles", {})) or {})
    profile = getattr(ag, "profile", "full_boost")
    profile_settings = profiles.get(profile)
    if isinstance(profile_settings, list):
        return {"included_model_types": profile_settings}
    if profile_settings is None:
        return {}
    if not isinstance(profile_settings, dict):
        return {}
    return dict(profile_settings)


def build_visible_autogluon_config(cfg: Config, settings: dict[str, Any]) -> dict[str, Any]:
    visible: dict[str, Any] = {}
    profile_settings = _profile_settings_for_cfg(cfg)
    for key in (
        "included_model_types",
        "presets",
        "time_limit",
        "use_gpu",
        "hyperparameters",
        "validation_strategy",
        "validation_fraction",
        "seed",
        "fit_args",
    ):
        if key not in profile_settings:
            continue
        value = settings.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)) and not value:
            continue
        visible[key] = value

    explicit_included = _container(cfg.agent.autogluon.included_model_types)
    if explicit_included is not None:
        visible["included_model_types"] = list(explicit_included)

    eval_metric = settings.get("eval_metric")
    if eval_metric and eval_metric != "auto":
        visible["eval_metric"] = eval_metric
    return visible


def build_autogluon_wrapper(preprocess_source: str, cfg: Config) -> str:
    validate_preprocess_source(preprocess_source)
    settings = resolve_autogluon_settings(cfg)
    constants = build_visible_autogluon_config(cfg, settings)
    constants_literal = pprint.pformat(constants, sort_dicts=True, width=88)
    return (
        f'''from __future__ import annotations

import json
import shutil
import warnings
import contextlib
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

AIDE_AG_CONFIG = {constants_literal}
RESULT_MARKER = {RESULT_MARKER!r}
FORBIDDEN_SPLIT_MARKER = {FORBIDDEN_SPLIT_MARKER!r}
FORBIDDEN_ROW_ID = {FORBIDDEN_ROW_ID!r}


{preprocess_source.strip()}


def _read_csv(data_dir: Path, stem: str) -> pd.DataFrame:
    gz_path = data_dir / f"{{stem}}.csv.gz"
    csv_path = data_dir / f"{{stem}}.csv"
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
        raise ValueError(f"preprocess changed row count: {{len(after)}} != {{len(before)}}")
    if target_col in after.columns:
        raise ValueError(f"preprocess created forbidden target column: {{target_col}}")
    if FORBIDDEN_SPLIT_MARKER in after.columns:
        raise ValueError(f"preprocess created forbidden split marker column: {{FORBIDDEN_SPLIT_MARKER}}")
    if FORBIDDEN_ROW_ID in after.columns:
        raise ValueError(f"preprocess created forbidden row id column: {{FORBIDDEN_ROW_ID}}")

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
    prediction_frame = pd.DataFrame({{
        id_col: pd.Series(test_ids).reset_index(drop=True),
        target_col: pd.Series(test_pred).reset_index(drop=True),
    }})
    if prediction_frame[id_col].duplicated().any():
        raise ValueError(f"test data contains duplicate {{id_col}} values")

    submission = sample_submission.copy()
    mapped = submission[[id_col]].merge(
        prediction_frame,
        on=id_col,
        how="left",
        validate="one_to_one",
    )[target_col]
    if mapped.isna().any():
        missing = int(mapped.isna().sum())
        raise ValueError(f"missing predictions for {{missing}} sample_submission ids")

    submission[target_col] = mapped.to_numpy()
    return submission.sort_values(id_col, kind="mergesort").reset_index(drop=True)


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
    input_dir = Path("./input")
    working_dir = Path("./working")
    working_dir.mkdir(parents=True, exist_ok=True)

    train_df = _read_csv(input_dir, "train")
    test_df = _read_csv(input_dir, "test")
    sample_submission = _read_csv(input_dir, "sample_submission")
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]
    if target_col not in train_df.columns:
        raise ValueError(f"Target column {{target_col!r}} not found in train data")

    y_train = train_df[target_col].reset_index(drop=True)
    train_features = train_df.drop(columns=[target_col, id_col], errors="ignore")
    test_features = test_df.drop(columns=[id_col], errors="ignore")
    combined = _make_combined_frame(train_features, test_features)
    preprocessed = preprocess(combined.copy())
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
    fit_kwargs = {{
        "train_data": train_data,
        "presets": AIDE_AG_CONFIG["presets"],
        "time_limit": AIDE_AG_CONFIG["time_limit"],
    }}
    if AIDE_AG_CONFIG.get("use_gpu") is not None:
        fit_kwargs["num_gpus"] = 1 if AIDE_AG_CONFIG["use_gpu"] else 0
    if valid_data is not None:
        fit_kwargs["tuning_data"] = valid_data
    if AIDE_AG_CONFIG.get("included_model_types"):
        fit_kwargs["included_model_types"] = AIDE_AG_CONFIG["included_model_types"]
    if AIDE_AG_CONFIG.get("hyperparameters"):
        fit_kwargs["hyperparameters"] = AIDE_AG_CONFIG["hyperparameters"]
    fit_kwargs.update(AIDE_AG_CONFIG.get("fit_args", {{}}))
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
    print(f"AIDE AutoGluon: submission saved to {{submission_path}}", flush=True)
    if artifact_submission_path.resolve() != submission_path.resolve():
        print(f"AIDE AutoGluon: artifact submission saved to {{artifact_submission_path}}", flush=True)

    summary = "AutoGluon preprocess wrapper completed."
    print(f"Validation {{eval_metric}}: {{metric_value:.6f}}")
    print("Submission saved successfully.")
    print(RESULT_MARKER + " " + json.dumps({{
        "is_bug": False,
        "summary": summary,
        "metric": metric_value,
        "lower_is_better": lower_is_better,
    }}, sort_keys=True))


# AIDE executes generated code via exec with a custom globals dict, where __name__ is not
# guaranteed to be "__main__". Call main() directly so the wrapper actually runs
# inside the interpreter.
main()
'''
    ).strip() + "\n"


def parse_result_marker(text: str) -> dict[str, Any] | None:
    parsed: dict[str, Any] | None = None
    for line in text.splitlines():
        if RESULT_MARKER not in line:
            continue
        payload = line.split(RESULT_MARKER, 1)[1].strip()
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
    return parsed


def infer_sample_submission_columns(input_dir: Path) -> tuple[str, str] | None:
    import pandas as pd

    for name in ("sample_submission.csv.gz", "sample_submission.csv"):
        path = input_dir / name
        if not path.exists():
            continue
        sample = pd.read_csv(path, nrows=1)
        if len(sample.columns) >= 2:
            return sample.columns[0], sample.columns[1]
    return None
