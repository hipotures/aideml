from __future__ import annotations

import ast
import json
import pprint
import re
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .utils.config import Config
from .utils.response import extract_code

AGENT_MODE = "autogluon_preprocess"
RESULT_MARKER = "AIDE_RESULT_JSON:"
FORBIDDEN_SPLIT_MARKER = "__is_train__"
FORBIDDEN_ROW_ID = "__aide_row_id__"

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


def extract_preprocess_source(text: str) -> str:
    candidates = []
    code = extract_code(text)
    if code:
        candidates.append(code)
    candidates.append(text)

    for candidate in candidates:
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


def build_autogluon_wrapper(preprocess_source: str, cfg: Config) -> str:
    validate_preprocess_source(preprocess_source)
    ag = cfg.agent.autogluon
    included_model_types = list(_container(ag.included_model_types) or [])
    fit_args = dict(_container(ag.fit_args) or {})

    constants = {
        "presets": ag.presets,
        "time_limit": int(ag.time_limit),
        "validation_fraction": float(ag.validation_fraction),
        "seed": int(ag.seed),
        "use_gpu": bool(ag.use_gpu),
        "eval_metric": ag.eval_metric,
        "included_model_types": included_model_types,
        "fit_args": fit_args,
    }
    constants_literal = pprint.pformat(constants, sort_dicts=True, width=88)
    return (
        f'''from __future__ import annotations

import json
import shutil
import warnings
import contextlib
import os
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
    configured = AIDE_AG_CONFIG["eval_metric"]
    if configured != "auto":
        return configured, "binary" if target.nunique(dropna=True) == 2 else None
    if target.nunique(dropna=True) == 2:
        return "roc_auc", "binary"
    return "accuracy", None


@contextlib.contextmanager
def _quiet_model_output(working_dir: Path):
    artifact_dir = Path(os.environ.get("AIDE_NODE_ARTIFACT_DIR", str(working_dir)))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "autogluon_stdout.log"

    class _AutoFlushWriter:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def write(self, text):
            written = self.wrapped.write(text)
            self.wrapped.flush()
            return written

        def flush(self):
            self.wrapped.flush()

    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        writer = _AutoFlushWriter(log_file)
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            yield


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
    stratify = train_model[target_col] if problem_type == "binary" else None
    train_data, valid_data = train_test_split(
        train_model,
        test_size=AIDE_AG_CONFIG["validation_fraction"],
        random_state=AIDE_AG_CONFIG["seed"],
        stratify=stratify,
    )

    model_dir = working_dir / "autogluon_model"
    shutil.rmtree(model_dir, ignore_errors=True)
    fit_kwargs = {{
        "train_data": train_data,
        "tuning_data": valid_data,
        "presets": AIDE_AG_CONFIG["presets"],
        "time_limit": AIDE_AG_CONFIG["time_limit"],
        "num_gpus": 1 if AIDE_AG_CONFIG["use_gpu"] else 0,
    }}
    if AIDE_AG_CONFIG["included_model_types"]:
        fit_kwargs["included_model_types"] = AIDE_AG_CONFIG["included_model_types"]
    fit_kwargs.update(AIDE_AG_CONFIG["fit_args"])
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
        if eval_metric == "roc_auc":
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
        print("AIDE AutoGluon: finished validation and prediction", flush=True)
    submission = sample_submission.copy()
    submission[target_col] = test_pred.to_numpy()
    submission.to_csv(working_dir / "submission.csv", index=False)

    summary = (
        f"AutoGluon preprocess wrapper completed with {{eval_metric}}="
        f"{{metric_value:.6f}} using presets={{AIDE_AG_CONFIG['presets']}}."
    )
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
