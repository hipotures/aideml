from __future__ import annotations

import ast
import json
import os
import pprint
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from omegaconf import OmegaConf

from .utils.config import Config, aux_file_name

AGENT_MODE = "autogluon_preprocess"
RESULT_MARKER = "AIDE_RESULT_JSON:"
FORBIDDEN_SPLIT_MARKER = "__is_train__"
FORBIDDEN_ROW_ID = "__aide_row_id__"
BASELINE_PLAN_PREFIX = "AutoGluon raw baseline"


_LIGHTGBM_GPU_CATEGORICAL_FALLBACK_HELPER_SOURCE = r'''
def _lightgbm_gpu_categorical_fallback_config(ag_config):
    section = ag_config.get("lightgbm_gpu_categorical_fallback")
    if section is None:
        section = {}
        default_action = "none"
    elif isinstance(section, dict):
        default_action = "fallback_to_cpu"
    else:
        raise ValueError("AIDE_AG_CONFIG['lightgbm_gpu_categorical_fallback'] must be a mapping")
    return {
        "action": _lightgbm_gpu_categorical_fallback_action(
            section.get("action"),
            default=default_action,
        ),
        "max_categorical_cardinality": _lightgbm_gpu_categorical_positive_int(
            section.get("max_categorical_cardinality", 512),
            "lightgbm_gpu_categorical_fallback.max_categorical_cardinality",
        ),
    }


def _lightgbm_gpu_categorical_fallback_action(value, *, default):
    if value is None:
        return default
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "disabled": "none",
        "none": "none",
        "drop": "drop_columns",
        "drop_column": "drop_columns",
        "drop_columns": "drop_columns",
        "fallback2cpu": "fallback_to_cpu",
        "fallback_to_cpu": "fallback_to_cpu",
        "cpu": "fallback_to_cpu",
    }
    if normalized not in aliases:
        raise ValueError(
            "lightgbm_gpu_categorical_fallback.action must be one of: "
            "none, fallback_to_cpu, drop_columns"
        )
    return aliases[normalized]


def _lightgbm_gpu_categorical_positive_int(value, name):
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if numeric <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return numeric


def _apply_lightgbm_gpu_categorical_fallback(ag_config, train_frame, test_frame):
    updated_config = copy.deepcopy(ag_config)
    config = _lightgbm_gpu_categorical_fallback_config(ag_config)
    action = config["action"]
    max_cardinality = int(config["max_categorical_cardinality"])
    stats = {
        "action": action,
        "max_categorical_cardinality": max_cardinality,
        "triggered": False,
        "columns": {},
    }
    if action == "none":
        stats["reason"] = "disabled"
        return updated_config, train_frame, test_frame, stats
    if not _lightgbm_ag_config_uses_gpu(updated_config):
        stats["reason"] = "lightgbm_not_gpu"
        return updated_config, train_frame, test_frame, stats

    high_cardinality = _high_cardinality_categorical_columns(
        train_frame,
        max_cardinality=max_cardinality,
    )
    if not high_cardinality:
        stats["reason"] = "no_high_cardinality_categorical_columns"
        return updated_config, train_frame, test_frame, stats

    stats["triggered"] = True
    stats["columns"] = {
        str(column): int(cardinality)
        for column, cardinality in high_cardinality.items()
    }
    column_summary = ",".join(
        f"{column}:{cardinality}"
        for column, cardinality in sorted(stats["columns"].items())
    )
    print(
        "AIDE_RUNTIME|lightgbm_gpu_categorical_fallback"
        f"|action={action}"
        f"|threshold={max_cardinality}"
        f"|columns={column_summary}",
        flush=True,
    )
    if action == "drop_columns":
        drop_columns = list(high_cardinality)
        stats["dropped_columns"] = [str(column) for column in drop_columns]
        return (
            updated_config,
            train_frame.drop(columns=drop_columns),
            test_frame.drop(columns=drop_columns, errors="ignore"),
            stats,
        )
    if action == "fallback_to_cpu":
        updated_config, changed = _lightgbm_ag_config_forced_to_cpu(updated_config)
        stats["forced_cpu"] = changed
        return updated_config, train_frame, test_frame, stats
    raise ValueError(f"Unsupported LightGBM categorical fallback action: {action!r}")


def _high_cardinality_categorical_columns(frame, *, max_cardinality):
    high_cardinality = {}
    for column in frame.columns:
        series = frame[column]
        if not _is_categorical_series(series):
            continue
        cardinality = int(series.nunique(dropna=False))
        if cardinality > max_cardinality:
            high_cardinality[column] = cardinality
    return high_cardinality


def _is_categorical_series(series):
    dtype = series.dtype
    return bool(
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_categorical_dtype(dtype)
    )


def _lightgbm_ag_config_uses_gpu(ag_config):
    return any(_lightgbm_config_uses_gpu(config) for config in _lightgbm_configs(ag_config))


def _lightgbm_ag_config_forced_to_cpu(ag_config):
    ag_config = copy.deepcopy(ag_config)
    changed = False
    for config in _lightgbm_configs(ag_config):
        if not _lightgbm_config_uses_gpu(config):
            continue
        if "device" in config:
            config["device"] = "cpu"
        if "device_type" in config:
            config["device_type"] = "cpu"
        if "device" not in config and "device_type" not in config:
            config["device"] = "cpu"
        ag_args_fit = (
            dict(config.get("ag_args_fit") or {})
            if isinstance(config.get("ag_args_fit"), dict)
            else {}
        )
        ag_args_fit["num_gpus"] = 0
        config["ag_args_fit"] = ag_args_fit
        changed = True
    return ag_config, changed


def _lightgbm_configs(ag_config):
    included = ag_config.get("included_model_types") or []
    if included and "GBM" not in included:
        return []
    hyperparameters = ag_config.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        return []
    raw_configs = hyperparameters.get("GBM")
    if isinstance(raw_configs, dict):
        return [raw_configs]
    if isinstance(raw_configs, list):
        return [config for config in raw_configs if isinstance(config, dict)]
    return []


def _lightgbm_config_uses_gpu(config):
    for key in ("device", "device_type"):
        value = config.get(key)
        if isinstance(value, str) and value.strip().lower() in {"cuda", "gpu"}:
            return True
    ag_args_fit = config.get("ag_args_fit")
    if isinstance(ag_args_fit, dict):
        try:
            return int(ag_args_fit.get("num_gpus") or 0) > 0
        except (TypeError, ValueError):
            return False
    return False
'''


_CLASS_BALANCING_HELPER_SOURCE = r'''
_WEIGHT_BASED_CLASS_BALANCING_METHODS = {
    "inverse_frequency",
    "clipped_inverse_frequency",
    "effective_number",
}
_RESAMPLING_CLASS_BALANCING_METHODS = {
    "partial_random_oversample",
}


def _class_balance_config(value):
    if value is None:
        return {"method": "none"}
    if isinstance(value, str):
        method = value.strip().lower().replace("-", "_")
        if method in {"none", "off", "unweighted"}:
            return {"method": "none"}
        if method in {"balanced", "inverse_frequency"}:
            return {"method": "inverse_frequency", "alpha": 1.0}
        raise ValueError(
            "class_balance must be 'none', 'balanced', 'inverse_frequency', "
            "or a mapping"
        )
    if not isinstance(value, dict):
        raise ValueError("class_balance must be a string or mapping")
    unknown = set(value) - {"method", "alpha", "max_raw_weight", "beta", "target_minority_to_majority_ratio"}
    if unknown:
        raise ValueError(f"Unsupported class_balance options: {sorted(unknown)}")
    method = str(value.get("method", "none")).strip().lower().replace("-", "_")
    if method == "balanced":
        method = "inverse_frequency"
    if method == "none":
        if "alpha" in value or "max_raw_weight" in value or "beta" in value or "target_minority_to_majority_ratio" in value:
            raise ValueError(
                "class_balance alpha/max_raw_weight/beta require a balancing method"
            )
        return {"method": "none"}
    if method == "effective_number":
        if (
            "alpha" in value
            or "max_raw_weight" in value
            or "target_minority_to_majority_ratio" in value
            or "beta" not in value
        ):
            raise ValueError("effective_number requires beta and rejects alpha/max_raw_weight")
        try:
            beta = float(value["beta"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("class_balance.beta must be a finite number in [0, 1)") from exc
        if not np.isfinite(beta) or beta < 0 or beta >= 1:
            raise ValueError("class_balance.beta must be a finite number in [0, 1)")
        return {"method": method, "beta": beta}
    if method == "partial_random_oversample":
        if any(k in value for k in ("alpha", "max_raw_weight", "beta")) or "target_minority_to_majority_ratio" not in value:
            raise ValueError("partial_random_oversample requires ratio and rejects alpha/max_raw_weight/beta")
        try:
            ratio = float(value["target_minority_to_majority_ratio"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("target_minority_to_majority_ratio must be a finite number in (0, 1)") from exc
        if not np.isfinite(ratio) or ratio <= 0 or ratio >= 1:
            raise ValueError("target_minority_to_majority_ratio must be a finite number in (0, 1)")
        return {"method": method, "target_minority_to_majority_ratio": ratio}
    if method not in {"inverse_frequency", "clipped_inverse_frequency"}:
        raise ValueError(
            "class_balance.method must be 'none', 'inverse_frequency', "
            "'clipped_inverse_frequency', 'effective_number', or 'partial_random_oversample'"
        )
    if "beta" in value:
        raise ValueError("class_balance.beta is only valid for effective_number")
    if "target_minority_to_majority_ratio" in value:
        raise ValueError(
            "target_minority_to_majority_ratio is only valid for partial_random_oversample"
        )
    try:
        alpha = float(value.get("alpha", 1.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("class_balance.alpha must be a finite number >= 0") from exc
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("class_balance.alpha must be a finite number >= 0")
    if method == "inverse_frequency":
        if "max_raw_weight" in value:
            raise ValueError(
                "class_balance.max_raw_weight is only valid for clipped_inverse_frequency"
            )
        return {"method": "inverse_frequency", "alpha": alpha}
    if "max_raw_weight" not in value:
        raise ValueError(
            "class_balance.max_raw_weight is required for clipped_inverse_frequency"
        )
    try:
        max_raw_weight = float(value["max_raw_weight"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "class_balance.max_raw_weight must be a finite number > 0"
        ) from exc
    if not np.isfinite(max_raw_weight) or max_raw_weight <= 0:
        raise ValueError("class_balance.max_raw_weight must be a finite number > 0")
    return {
        "method": "clipped_inverse_frequency",
        "alpha": alpha,
        "max_raw_weight": max_raw_weight,
    }


def _partial_random_oversample(train_data, *, target_col, ratio, seed):
    try:
        ratio = float(ratio)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target_minority_to_majority_ratio must be a finite number in (0, 1)") from exc
    if not np.isfinite(ratio) or ratio <= 0 or ratio >= 1:
        raise ValueError("target_minority_to_majority_ratio must be a finite number in (0, 1)")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise ValueError("Oversampling seed must be an integer")
    if not isinstance(target_col, str) or not target_col:
        raise ValueError("Oversampling requires a target column name")
    if target_col not in train_data.columns:
        raise ValueError("Oversampling requires a present target column")
    labels = train_data[target_col]
    if labels.empty:
        raise ValueError("Oversampling requires nonempty training rows")
    if labels.isna().any():
        raise ValueError("Oversampling requires a nonmissing target")
    counts = labels.value_counts()
    if counts.empty or (counts <= 0).any():
        raise ValueError("Oversampling requires positive class counts")
    class_order = sorted(counts.index.tolist(), key=lambda label: (type(label).__name__, repr(label)))
    before_counts = {str(label): int(counts[label]) for label in class_order}
    majority = int(counts.max())
    rng = np.random.default_rng(int(seed))
    pieces = [train_data]
    for label in class_order:
        need = max(0, int(np.ceil(float(ratio) * majority)) - int(counts[label]))
        if need:
            pool = train_data[train_data[target_col] == label]
            pieces.append(pool.iloc[rng.integers(0, len(pool), size=need)])
    out = pd.concat(pieces, ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    after_raw_counts = out[target_col].value_counts()
    after_counts = {str(label): int(after_raw_counts[label]) for label in class_order}
    added = len(out) - len(train_data)
    return out, before_counts, after_counts, added


def _inverse_frequency_sample_weight(labels, *, alpha, max_raw_weight=None, beta=None):
    labels = pd.Series(labels).copy()
    if labels.empty:
        raise ValueError("Cannot compute class weights for empty labels")
    if labels.isna().any():
        raise ValueError("Cannot compute class weights for missing labels")
    try:
        alpha = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("class_balance.alpha must be a finite number >= 0") from exc
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("class_balance.alpha must be a finite number >= 0")
    if max_raw_weight is not None:
        try:
            max_raw_weight = float(max_raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_raw_weight must be a finite number > 0") from exc
        if not np.isfinite(max_raw_weight) or max_raw_weight <= 0:
            raise ValueError("max_raw_weight must be a finite number > 0")

    counts = labels.value_counts(dropna=False)
    if beta is not None:
        try:
            beta = float(beta)
        except (TypeError, ValueError) as exc:
            raise ValueError("class_balance.beta must be a finite number in [0, 1)") from exc
        if not np.isfinite(beta) or beta < 0 or beta >= 1:
            raise ValueError("class_balance.beta must be a finite number in [0, 1)")
        n = counts.astype(float)
        class_weights = (
            (1 - beta) / (-np.expm1(n * np.log(beta)))
            if beta > 0
            else pd.Series(1.0, index=counts.index)
        )
    else:
        base_weights = len(labels) / (len(counts) * counts.astype(float))
        class_weights = base_weights.pow(alpha)
    if max_raw_weight is not None:
        class_weights = class_weights.clip(upper=max_raw_weight)
    row_weights = labels.map(class_weights).astype(float)
    mean_weight = float(row_weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Calculated class weights must have a finite positive mean")
    row_weights = row_weights / mean_weight
    class_weights = class_weights / mean_weight
    if not np.isfinite(row_weights.to_numpy()).all() or not (row_weights > 0).all():
        raise ValueError("Calculated sample weights must be finite and positive")
    if not row_weights.index.equals(labels.index):
        raise AssertionError("Sample-weight indices do not align with training labels")
    return row_weights, {str(label): float(weight) for label, weight in class_weights.items()}


def _validate_custom_balancing_context(
    config,
    *,
    uses_bagging=None,
    bagged_mode=None,
    auto_stack=False,
    validation_strategy=None,
):
    if config["method"] == "none":
        return
    method = config["method"]
    if uses_bagging is None:
        uses_bagging = bool(bagged_mode)
    if method in _WEIGHT_BASED_CLASS_BALANCING_METHODS:
        return
    if method in _RESAMPLING_CLASS_BALANCING_METHODS and (uses_bagging or auto_stack):
        raise ValueError(
            f"Class resampling method {method!r} is not supported with "
            "bagging/auto_stack because duplicated rows could cross folds."
        )
    if validation_strategy != "holdout":
        raise ValueError(
            "Custom class balancing requires validation_strategy='holdout' so weights "
            "are derived from training labels only"
        )
'''


def baseline_preprocess_source() -> str:
    return "def preprocess(df: pd.DataFrame) -> pd.DataFrame:\n    return df.copy()\n"


def preprocess_task_prompt_text(text: Any) -> str:
    """Remove full-solution instructions that cannot be acted on by preprocess(df)."""
    cleaned = str(text or "")
    cleaned = cleaned.replace(
        "Submissions are evaluated using balanced accuracy. Higher is better.",
        "The fixed wrapper evaluates feature changes using balanced accuracy. Higher is better.",
    )
    cleaned = re.sub(
        (
            r"\n*Competition-specific modeling hint:.*?"
            r"`\.fit\(\)`\.\n*"
        ),
        "\n\n",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        (
            r"\n*The submission file must contain a header and exactly these "
            r"columns:\n\n```csv\n.*?```\n*"
        ),
        "\n",
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        (
            r"\n*Additional auxiliary data description for `[^`]+`:\n\n.*?"
            r"Generated code should decide whether and how to use this file\..*?"
            r"explicitly by the generated solution code\.\n*"
        ),
        "\n",
        cleaned,
        flags=re.DOTALL,
    )
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


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
    if (
        len(func.args.args) not in {1, 2}
        or func.args.vararg is not None
        or func.args.kwarg is not None
        or func.args.kwonlyargs
    ):
        raise ValueError("`preprocess` must accept `df`, optionally followed by `aux`.")
    if len(func.args.args) == 2 and func.args.args[1].arg != "aux":
        raise ValueError("`preprocess` second argument must be named `aux`.")


def _container(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _prioritize_xgboost_hyperparameters(settings: dict[str, Any]) -> None:
    if settings.get("fair_model_scheduling"):
        return
    hyperparameters = settings.get("hyperparameters")
    if not isinstance(hyperparameters, dict) or "XGB" not in hyperparameters:
        return

    xgb_configs = hyperparameters["XGB"]
    if not isinstance(xgb_configs, list):
        xgb_configs = [xgb_configs]
        hyperparameters["XGB"] = xgb_configs

    for model_cfg in xgb_configs:
        if not isinstance(model_cfg, dict):
            continue
        ag_args = model_cfg.setdefault("ag_args", {})
        if isinstance(ag_args, dict):
            ag_args["priority"] = 999


def _force_cpu_boost_hyperparameters(settings: dict[str, Any]) -> None:
    if settings.get("use_gpu") is True:
        return

    included = settings.get("included_model_types") or []
    if "XGB" not in included:
        return

    fair_model_scheduling = bool(settings.get("fair_model_scheduling"))
    if not fair_model_scheduling:
        settings["included_model_types"] = ["XGB"] + [
            model_type for model_type in included if model_type != "XGB"
        ]
    settings["use_gpu"] = False
    hyperparameters = settings.setdefault("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        return

    xgb_configs = hyperparameters.get("XGB") or [{}]
    if not isinstance(xgb_configs, list):
        xgb_configs = [xgb_configs]
    hyperparameters["XGB"] = xgb_configs

    for model_cfg in xgb_configs:
        if not isinstance(model_cfg, dict):
            continue
        model_cfg["device"] = "cpu"
        model_cfg["tree_method"] = "hist"
        ag_args = model_cfg.setdefault("ag_args", {})
        if isinstance(ag_args, dict) and not fair_model_scheduling:
            ag_args["priority"] = 999
        ag_args_fit = model_cfg.setdefault("ag_args_fit", {})
        if isinstance(ag_args_fit, dict):
            ag_args_fit["num_gpus"] = 0

    for model_type in ("GBM", "CAT"):
        if model_type not in included:
            continue
        model_configs = hyperparameters.get(model_type) or [{}]
        if not isinstance(model_configs, list):
            model_configs = [model_configs]
        hyperparameters[model_type] = model_configs
        for model_cfg in model_configs:
            if not isinstance(model_cfg, dict):
                continue
            ag_args_fit = model_cfg.setdefault("ag_args_fit", {})
            if isinstance(ag_args_fit, dict):
                ag_args_fit["num_gpus"] = 0


def lightgbm_gpu_categorical_fallback_options(value: object) -> dict[str, object]:
    section = _container(value)
    if section is None:
        section = {}
        default_action = "none"
    elif isinstance(section, dict):
        default_action = "fallback_to_cpu"
    else:
        raise ValueError("agent.autogluon.lightgbm_gpu_categorical_fallback must be a mapping")
    return {
        "action": _categorical_fallback_action_option(
            section.get("action"),
            default=default_action,
        ),
        "max_categorical_cardinality": _positive_int_option(
            section.get("max_categorical_cardinality", 512),
            "agent.autogluon.lightgbm_gpu_categorical_fallback.max_categorical_cardinality",
        ),
    }


def _categorical_fallback_action_option(value: object, *, default: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "false": "none",
        "disabled": "none",
        "none": "none",
        "drop": "drop_columns",
        "drop_column": "drop_columns",
        "drop_columns": "drop_columns",
        "fallback2cpu": "fallback_to_cpu",
        "fallback_to_cpu": "fallback_to_cpu",
        "cpu": "fallback_to_cpu",
    }
    if normalized not in aliases:
        raise ValueError(
            "agent.autogluon.lightgbm_gpu_categorical_fallback.action must be one of: "
            "none, fallback_to_cpu, drop_columns"
        )
    return aliases[normalized]


def _positive_int_option(value: object, name: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if numeric <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return numeric


def resolve_autogluon_settings(
    cfg: Config,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    ag = cfg.agent.autogluon
    profiles = dict(_container(getattr(ag, "profiles", {})) or {})
    profile = profile or getattr(ag, "profile", "full_boost")
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
        "preprocess_timeout": int(getattr(ag, "preprocess_timeout", 600)),
        "save_prediction_artifacts": bool(
            getattr(ag, "save_prediction_artifacts", True)
        ),
        "validation_fraction": float(ag.validation_fraction),
        "seed": int(ag.seed),
        "included_model_types": None,
        "lightgbm_gpu_categorical_fallback": _container(
            getattr(ag, "lightgbm_gpu_categorical_fallback", None)
        ),
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
    _force_cpu_boost_hyperparameters(settings)
    settings["time_limit"] = int(settings["time_limit"])
    settings["preprocess_timeout"] = int(settings.get("preprocess_timeout", 600))
    settings["validation_fraction"] = float(settings["validation_fraction"])
    settings["seed"] = int(settings["seed"])
    if "use_gpu" in settings and settings["use_gpu"] is not None:
        settings["use_gpu"] = bool(settings["use_gpu"])
    if "validation_strategy" in settings and settings["validation_strategy"] is not None:
        settings["validation_strategy"] = str(settings["validation_strategy"])
    if "hyperparameters" in settings:
        settings["hyperparameters"] = dict(settings.get("hyperparameters") or {})
        _prioritize_xgboost_hyperparameters(settings)
    if "fit_args" in settings:
        settings["fit_args"] = dict(settings.get("fit_args") or {})
    settings["lightgbm_gpu_categorical_fallback"] = (
        lightgbm_gpu_categorical_fallback_options(
            settings.get("lightgbm_gpu_categorical_fallback")
        )
    )
    if settings.get("validation_strategy") not in {None, "holdout", "autogluon"}:
        raise ValueError(
            "AutoGluon validation_strategy must be one of: holdout, autogluon"
        )
    return settings


def legacy_starter_design(cfg: Config, *, profile: str) -> str:
    """Describe the deterministic legacy baseline represented by ``profile``."""
    settings = resolve_autogluon_settings(cfg, profile=profile)
    model_names = {
        "XGB": "XGBoost",
        "GBM": "LightGBM",
        "CAT": "CatBoost",
    }
    included = settings.get("included_model_types") or []
    models = [model_names.get(str(model), str(model)) for model in included]
    model_text = ", ".join(models) if models else "AutoGluon's default model set"
    fit_args = dict(settings.get("fit_args") or {})
    folds = int(fit_args.get("num_bag_folds") or 0)
    device = "GPU" if settings.get("use_gpu") else "CPU"
    ensemble_text = (
        "and fits AutoGluon's weighted ensemble"
        if fit_args.get("fit_weighted_ensemble", True)
        else "without a weighted ensemble"
    )
    balance = settings.get("class_balance")
    normalized_balance = balance.strip().lower() if isinstance(balance, str) else balance
    if normalized_balance is None or (
        isinstance(normalized_balance, str)
        and normalized_balance in {"none", "off", "unweighted"}
    ):
        balance_text = "Training uses the original class frequencies without reweighting."
    else:
        balance_text = (
            "Inverse-frequency sample weights normalized to unit mean address class "
            "imbalance during fitting."
        )
    validation_text = (
        f"AutoGluon's internal {folds}-fold bagging supplies OOF validation predictions"
        if folds > 0
        else "The configured AutoGluon validation strategy supplies validation predictions"
    )
    return (
        f"{BASELINE_PLAN_PREFIX}: The fixed {profile!r} starter trains {model_text} "
        "on the raw competition covariates, ignores the identifier, and adds no "
        f"engineered or auxiliary features. It runs on {device}; {validation_text} "
        f"for the configured project metric {ensemble_text}. {balance_text} The common "
        "runner writes the submission plus OOF and test prediction artifacts for direct "
        "comparison with later legacy branches."
    )


def _load_project_env() -> dict[str, str]:
    load_dotenv(dotenv_path=Path(".env"), override=True)
    project_name = os.getenv("AIDE_PROJECT_NAME", "").strip()
    eval_metric = os.getenv("AIDE_PROJECT_METRIC", "").strip()
    if not project_name:
        raise ValueError("AIDE_PROJECT_NAME must be set in .env for AutoGluon runs.")
    if not eval_metric:
        raise ValueError("AIDE_PROJECT_METRIC must be set in .env for AutoGluon runs.")
    return {
        "project_name": project_name,
        "eval_metric": eval_metric,
    }


def resolve_autogluon_included_model_types(cfg: Config) -> list[str]:
    included = resolve_autogluon_settings(cfg).get("included_model_types")
    return list(included or [])


def _profile_settings_for_cfg(
    cfg: Config,
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    ag = cfg.agent.autogluon
    profiles = dict(_container(getattr(ag, "profiles", {})) or {})
    profile = profile or getattr(ag, "profile", "full_boost")
    profile_settings = profiles.get(profile)
    if isinstance(profile_settings, list):
        return {"included_model_types": profile_settings}
    if profile_settings is None:
        return {}
    if not isinstance(profile_settings, dict):
        return {}
    return dict(profile_settings)


def build_visible_autogluon_config(
    cfg: Config,
    settings: dict[str, Any],
    *,
    profile: str | None = None,
) -> dict[str, Any]:
    profile = profile or str(cfg.agent.autogluon.profile)
    visible: dict[str, Any] = {"profile": profile}
    profile_settings = _profile_settings_for_cfg(cfg, profile=profile)
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
        "class_balance",
        "fair_model_scheduling",
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

    included = settings.get("included_model_types") or []
    if (
        settings.get("use_gpu") is False
        and "XGB" in included
        and isinstance(settings.get("hyperparameters"), dict)
    ):
        visible["use_gpu"] = False
        visible["hyperparameters"] = settings["hyperparameters"]

    visible.update(_load_project_env())
    aux_name = aux_file_name(cfg)
    if aux_name is not None:
        visible["aux_file"] = aux_name
    visible["preprocess_timeout"] = int(settings.get("preprocess_timeout", 600))
    visible["save_prediction_artifacts"] = bool(
        settings.get("save_prediction_artifacts", True)
    )
    fallback = settings.get("lightgbm_gpu_categorical_fallback")
    if isinstance(fallback, dict):
        visible["lightgbm_gpu_categorical_fallback"] = dict(fallback)
    return visible


def build_autogluon_wrapper(
    preprocess_source: str,
    cfg: Config,
    *,
    research_hypothesis_id: str | None = None,
    profile: str | None = None,
) -> str:
    validate_preprocess_source(preprocess_source)
    settings = resolve_autogluon_settings(cfg, profile=profile)
    constants = build_visible_autogluon_config(cfg, settings, profile=profile)
    constants_literal = pprint.pformat(constants, sort_dicts=True, width=88)
    _ = research_hypothesis_id
    research_marker_fields = ""
    return (
        f'''from __future__ import annotations

import hashlib
import json
import shutil
import warnings
import contextlib
import copy
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

AIDE_AG_CONFIG = {constants_literal}
RESULT_MARKER = {RESULT_MARKER!r}
FORBIDDEN_SPLIT_MARKER = {FORBIDDEN_SPLIT_MARKER!r}
FORBIDDEN_ROW_ID = {FORBIDDEN_ROW_ID!r}
CLASS_WEIGHT_COL = "__aide_class_weight__"


{_LIGHTGBM_GPU_CATEGORICAL_FALLBACK_HELPER_SOURCE.strip()}


{_CLASS_BALANCING_HELPER_SOURCE.strip()}


{preprocess_source.strip()}


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
    gz_path = data_dir / f"{{stem}}.csv.gz"
    csv_path = data_dir / f"{{stem}}.csv"
    if gz_path.exists():
        return pd.read_csv(gz_path)
    return pd.read_csv(csv_path)


def _read_aux_csv(input_dir: Path) -> pd.DataFrame | None:
    aux_file = AIDE_AG_CONFIG.get("aux_file")
    if not aux_file:
        return None
    aux_path = input_dir / str(aux_file)
    if not aux_path.exists():
        raise FileNotFoundError(f"Configured aux file not found: {{aux_path}}")
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
    model_label = model if model is not None else "autogluon_best"
    started_at = time.time()
    print(
        f"AIDE AutoGluon: predict start model={{model_label}} rows={{len(data)}} metric={{eval_metric}}",
        flush=True,
    )
    try:
        if eval_metric == "roc_auc":
            return _positive_probability(predictor, data, model=model)
        pred = predictor.predict(data, model=model)
        return pd.Series(pred).reset_index(drop=True)
    finally:
        print(
            f"AIDE AutoGluon: predict finished model={{model_label}} "
            f"elapsed={{time.time() - started_at:.1f}}s",
            flush=True,
        )


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
            f"preprocess changed row count: {{len(after)}} != {{len(before)}}. "
            "AutoGluon preprocess(df) must preserve one output row per input row. "
            "If row removal was intentional, such as outlier filtering, rewrite it "
            "without dropping rows: add an outlier flag, clipped/winsorized value, "
            "imputed clean value, anomaly score, or distance-from-normal feature instead."
        )
    if target_col in after.columns:
        raise ValueError(f"preprocess created forbidden target column: {{target_col}}")
    if FORBIDDEN_SPLIT_MARKER in after.columns:
        raise ValueError(f"preprocess created forbidden split marker column: {{FORBIDDEN_SPLIT_MARKER}}")
    if FORBIDDEN_ROW_ID in after.columns:
        raise ValueError(f"preprocess created forbidden row id column: {{FORBIDDEN_ROW_ID}}")
    if CLASS_WEIGHT_COL in after.columns:
        raise ValueError(f"preprocess created forbidden class weight column: {{CLASS_WEIGHT_COL}}")

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
        return [{{"error": f"leaderboard unavailable: {{exc}}"}}]
    records = []
    for row in leaderboard.to_dict(orient="records"):
        records.append(
            {{
                column: _json_safe_scalar(row.get(column))
                for column in keep_columns
                if column in row
            }}
        )
    return records


def _selected_model_metadata(predictor: TabularPredictor) -> dict:
    try:
        selected_model = predictor.model_best
    except Exception as exc:
        return {{"selected_model_error": f"{{type(exc).__name__}}: {{exc}}"}}

    composition = None
    if str(selected_model).lower().startswith("weightedensemble"):
        try:
            ensemble = predictor._trainer.load_model(selected_model)
            raw_weights = getattr(ensemble, "model_weights", None)
            if raw_weights is None:
                raw_weights = getattr(ensemble, "weights_", None)
            if raw_weights is None:
                raw_weights = getattr(ensemble, "weights", None)
            if raw_weights is None:
                get_model_weights = getattr(ensemble, "_get_model_weights", None)
                if callable(get_model_weights):
                    raw_weights = get_model_weights()
            if isinstance(raw_weights, dict):
                composition = {{
                    str(name): _json_safe_scalar(weight)
                    for name, weight in raw_weights.items()
                }}
            else:
                base_models = list(getattr(ensemble, "base_model_names", []) or [])
                if base_models and isinstance(raw_weights, (list, tuple)):
                    composition = {{
                        str(name): _json_safe_scalar(weight)
                        for name, weight in zip(base_models, raw_weights)
                    }}
        except Exception as exc:
            composition = {{"error": f"{{type(exc).__name__}}: {{exc}}"}}
    return {{
        "selected_model": _json_safe_scalar(selected_model),
        "ensemble_composition": composition,
    }}


def _artifact_dir(working_dir: Path) -> Path:
    return Path(os.environ.get("AIDE_NODE_ARTIFACT_DIR", str(working_dir)))


def _clear_prediction_artifacts(working_dir: Path) -> None:
    for base_dir in {{working_dir, _artifact_dir(working_dir)}}:
        for filename in (
            "oof_predictions.csv",
            "oof_predictions.csv.gz",
            "test_predictions.csv",
            "test_predictions.csv.gz",
            "validation_predictions.csv",
            "validation_predictions.csv.gz",
        ):
            (base_dir / filename).unlink(missing_ok=True)
        shutil.rmtree(base_dir / "model_predictions", ignore_errors=True)


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
        filename = f"{{filename}}.gz"
    working_path = working_dir / filename
    working_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    print(
        f"AIDE AutoGluon: writing prediction artifact {{filename}} "
        f"rows={{len(frame)}} cols={{len(frame.columns)}}",
        flush=True,
    )
    frame.to_csv(working_path, index=False, compression="gzip")
    artifact_dir = _artifact_dir(working_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.resolve() != working_path.resolve():
        shutil.copy2(working_path, artifact_path)
    print(
        f"AIDE AutoGluon: wrote prediction artifact {{filename}} "
        f"elapsed={{time.time() - started_at:.1f}}s",
        flush=True,
    )
    return working_path


def _safe_prediction_name(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
    return safe or "model"


def _model_family(name: str) -> str | None:
    normalized = str(name).lower()
    if "xgboost" in normalized or normalized.startswith("xgb"):
        return "XGB"
    if "lightgbm" in normalized or normalized.startswith("gbm"):
        return "GBM"
    if "catboost" in normalized or normalized.startswith("cat"):
        return "CAT"
    return None


def _heldout_export_models(predictor: TabularPredictor) -> list[dict]:
    leaderboard = predictor.leaderboard(silent=True)
    selected_model = str(predictor.model_best)
    selected = []
    selected_names = set()
    for family in ("XGB", "GBM", "CAT"):
        for row in leaderboard.to_dict(orient="records"):
            name = str(row.get("model"))
            if (
                _model_family(name) != family
                or not bool(row.get("can_infer", False))
                or int(row.get("stack_level", 1)) != 1
                or "weightedensemble" in name.lower()
            ):
                continue
            selected.append(
                {{"model": name, "model_family": family, "selected": name == selected_model}}
            )
            selected_names.add(name)
            break
        else:
            raise ValueError(f"No inferable stack-1 {{family}} model for held-out export")
    if selected_model not in selected_names:
        selected_row = next(
            (
                row
                for row in leaderboard.to_dict(orient="records")
                if str(row.get("model")) == selected_model and bool(row.get("can_infer", False))
            ),
            None,
        )
        if selected_row is None:
            raise ValueError("Selected model is not inferable for held-out export")
        selected.append(
            {{
                "model": selected_model,
                "model_family": _model_family(selected_model) or "selected",
                "selected": True,
            }}
        )
    return selected


def _ordered_value_sha256(values) -> str:
    payload = json.dumps(
        [_json_safe_scalar(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    artifacts = {{}}
    test_predictions = pd.DataFrame({{
        id_col: pd.Series(test_ids).reset_index(drop=True),
        target_col: pd.Series(test_pred).reset_index(drop=True),
    }})
    test_path = _save_prediction_artifact(
        test_predictions,
        working_dir,
        "test_predictions.csv",
    )
    artifacts["test_predictions"] = str(test_path)

    # Fixed held-out fold: export explicit-validation probabilities only.
    # This branch must never call any OOF API.
    if valid_data is not None:
        valid_features = valid_data.drop(
            columns=[target_col, CLASS_WEIGHT_COL], errors="ignore"
        )
        valid_target = valid_data[target_col].copy()
        if not valid_features.index.equals(valid_data.index):
            raise AssertionError("Held-out feature indices do not align with validation rows")
        if valid_target.isna().any() or not valid_target.index.equals(valid_data.index):
            raise AssertionError("Held-out targets do not align with validation rows")
        class_order = list(getattr(predictor, "class_labels", []) or [])
        if not class_order or not set(valid_target.unique()).issubset(set(class_order)):
            raise ValueError("Held-out target classes do not match predictor class schema")
        validation_row_sha256 = _ordered_value_sha256(valid_data.index)
        validation_target_sha256 = _ordered_value_sha256(valid_target)
        for model_info in _heldout_export_models(predictor):
            name = model_info["model"]
            proba = predictor.predict_proba(valid_features, model=name)
            if not isinstance(proba, pd.DataFrame):
                raise TypeError("Held-out probabilities must be a DataFrame")
            if len(proba) != len(valid_data) or not proba.index.equals(valid_data.index):
                raise AssertionError("Held-out probability rows do not align with validation rows")
            if list(proba.columns) != class_order:
                raise ValueError("Held-out probability classes do not match predictor class order")
            values = proba.to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Held-out probabilities must be finite and nonnegative")
            if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("Held-out probability rows must sum to one")
            frame = pd.concat(
                [
                    pd.DataFrame(
                        {{
                            "row": valid_data.index.to_numpy(),
                            "target": valid_target.to_numpy(),
                        }}
                    ),
                    proba.reset_index(drop=True),
                ],
                axis=1,
            )
            if not frame["target"].equals(valid_target.reset_index(drop=True)):
                raise AssertionError("Held-out target export does not preserve validation targets")
            safe_name = _safe_prediction_name(name)
            relative_path = f"model_predictions/{{safe_name}}-heldout-probabilities.csv.gz"
            path = _save_prediction_artifact(frame, working_dir, relative_path[:-3])
            artifacts.setdefault("heldout_probability_files", []).append(
                {{
                    **model_info,
                    "path": str(path),
                    "relative_path": relative_path,
                    "class_order": [str(label) for label in class_order],
                    "rows": int(len(frame)),
                    "validation_row_sha256": validation_row_sha256,
                    "validation_target_sha256": validation_target_sha256,
                }}
            )
        artifacts["heldout_probability_kind"] = "fixed_heldout_fold_probabilities"
        artifacts["heldout_probability_note"] = "single_fixed_holdout_not_oof"
        return artifacts

    try:
        per_model = []
        model_names = list(predictor.model_names())
        for model_index, model_name in enumerate(model_names, start=1):
            print(
                f"AIDE AutoGluon: model artifact start {{model_index}}/{{len(model_names)}} "
                f"model={{model_name}}",
                flush=True,
            )
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
                        f"OOF row count {{len(model_oof_pred)}} != train rows {{len(train_target)}}"
                    )
                safe_model_name = _safe_prediction_name(model_name)
                model_oof_frame = pd.DataFrame({{
                    "row": np.arange(len(train_target)),
                    "target": pd.Series(train_target).reset_index(drop=True),
                    "prediction": model_oof_pred.reset_index(drop=True),
                    "model": model_name,
                }})
                model_oof_path = _save_prediction_artifact(
                    model_oof_frame,
                    working_dir,
                    f"model_predictions/{{safe_model_name}}-oof.csv",
                )
                model_test_pred = _predict_values(
                    predictor,
                    test_model,
                    eval_metric=eval_metric,
                    model=model_name,
                )
                model_test_frame = pd.DataFrame({{
                    id_col: pd.Series(test_ids).reset_index(drop=True),
                    target_col: pd.Series(model_test_pred).reset_index(drop=True),
                    "model": model_name,
                }})
                model_test_path = _save_prediction_artifact(
                    model_test_frame,
                    working_dir,
                    f"model_predictions/{{safe_model_name}}-test.csv",
                )
                per_model.append({{
                    "model": model_name,
                    "oof_predictions": str(model_oof_path),
                    "test_predictions": str(model_test_path),
                    "rows": int(len(model_oof_frame)),
                    "test_rows": int(len(model_test_frame)),
                }})
            except Exception as exc:
                per_model.append({{
                    "model": model_name,
                    "error": f"{{type(exc).__name__}}: {{exc}}",
                }})
            print(
                f"AIDE AutoGluon: model artifact finished {{model_index}}/{{len(model_names)}} "
                f"model={{model_name}}",
                flush=True,
            )
        artifacts["model_predictions"] = per_model
        artifacts["model_predictions_ok"] = sum(1 for item in per_model if "error" not in item)

        print("AIDE AutoGluon: OOF artifact start model=autogluon_best", flush=True)
        oof_proba = predictor.predict_proba_oof(
            transformed=False,
            as_multiclass=True,
        )
        oof_pred = _values_from_proba(oof_proba, eval_metric=eval_metric)
        if len(oof_pred) != len(train_target):
            raise ValueError(f"OOF row count {{len(oof_pred)}} != train rows {{len(train_target)}}")
        oof_frame = pd.DataFrame({{
            "row": np.arange(len(train_target)),
            "target": pd.Series(train_target).reset_index(drop=True),
            "prediction": oof_pred.reset_index(drop=True),
        }})
        oof_path = _save_prediction_artifact(
            oof_frame,
            working_dir,
            "oof_predictions.csv",
        )
        artifacts["oof_predictions"] = str(oof_path)
        artifacts["oof_rows"] = int(len(oof_frame))
        print("AIDE AutoGluon: OOF artifact finished model=autogluon_best", flush=True)
    except Exception as exc:
        artifacts["oof_error"] = f"{{type(exc).__name__}}: {{exc}}"

    if valid_data is not None and valid_pred is not None:
        validation_frame = pd.DataFrame({{
            "row": np.arange(len(valid_data)),
            "target": valid_data[target_col].reset_index(drop=True),
            "prediction": pd.Series(valid_pred).reset_index(drop=True),
        }})
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
def _preprocess_timeout(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    class PreprocessTimeoutError(TimeoutError):
        pass

    def _raise_preprocess_timeout(_signum, _frame):
        raise PreprocessTimeoutError(
            "AIDE AutoGluon preprocess exceeded the dedicated timeout of "
            f"{{seconds}} seconds. This timeout is separate from AutoGluon "
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
    if not AIDE_AG_CONFIG.get("save_prediction_artifacts", True):
        _clear_prediction_artifacts(working_dir)

    train_df = _read_csv(input_dir, "train")
    test_df = _read_csv(input_dir, "test")
    sample_submission = _read_csv(input_dir, "sample_submission")
    aux_df = _read_aux_csv(input_dir)
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]
    if target_col not in train_df.columns:
        raise ValueError(f"Target column {{target_col!r}} not found in train data")

    y_train = train_df[target_col].reset_index(drop=True)
    train_features = train_df.drop(columns=[target_col], errors="ignore")
    test_features = test_df.copy()
    combined = _make_combined_frame(train_features, test_features)
    if aux_df is not None:
        print(
            "AIDE AutoGluon: loaded aux file "
            f"{{AIDE_AG_CONFIG.get('aux_file')}} rows={{len(aux_df)}} "
            f"cols={{len(aux_df.columns)}} "
            f"passed_to_preprocess={{_preprocess_accepts_aux()}}",
            flush=True,
        )
    print("AIDE AutoGluon: starting preprocess", flush=True)
    preprocess_started_at = time.time()
    with _preprocess_timeout(int(AIDE_AG_CONFIG.get("preprocess_timeout", 600))):
        preprocessed = _run_preprocess(combined, aux_df)
    preprocess_time = time.time() - preprocess_started_at
    feature_count = int(len(preprocessed.columns))
    print(
        f"AIDE AutoGluon: finished preprocess rows={{len(preprocessed)}} cols={{feature_count}}",
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
    AIDE_AG_CONFIG_UPDATE, train_fe, test_fe, lightgbm_categorical_fallback_stats = (
        _apply_lightgbm_gpu_categorical_fallback(
            AIDE_AG_CONFIG,
            train_fe,
            test_fe,
        )
    )
    AIDE_AG_CONFIG.clear()
    AIDE_AG_CONFIG.update(AIDE_AG_CONFIG_UPDATE)
    feature_count = int(len(train_fe.columns))
    train_model = train_fe.copy()
    train_model[target_col] = y_train.to_numpy()
    test_model = test_fe.copy()

    eval_metric = _configured_metric()
    class_balance = _class_balance_config(AIDE_AG_CONFIG.get("class_balance"))
    fit_args = dict(AIDE_AG_CONFIG.get("fit_args", {{}}) or {{}})
    uses_bagging = int(fit_args.get("num_bag_folds") or 0) > 0
    auto_stack = bool(fit_args.get("auto_stack"))
    bagged_mode = uses_bagging or auto_stack
    _validate_custom_balancing_context(
        class_balance,
        bagged_mode=bagged_mode,
        uses_bagging=uses_bagging,
        auto_stack=auto_stack,
        validation_strategy=AIDE_AG_CONFIG.get("validation_strategy"),
    )
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

    explicit_training_partition = (not bagged_mode) and (
        AIDE_AG_CONFIG.get("validation_strategy") == "holdout"
    )
    class_weight_scope = (
        "explicit_training_partition"
        if explicit_training_partition
        else "global_training"
    )

    if class_balance["method"] == "partial_random_oversample":
        train_data, before_counts, after_counts, added_rows = _partial_random_oversample(
            train_data,
            target_col=target_col,
            ratio=class_balance["target_minority_to_majority_ratio"],
            seed=AIDE_AG_CONFIG.get("seed", 42),
        )
        print(
            "AIDE_RUNTIME|class_resampling"
            + f"|method=partial_random_oversample|ratio={{class_balance['target_minority_to_majority_ratio']}}"
            + f"|seed={{AIDE_AG_CONFIG.get('seed', 42)}}"
            + f"|before={{json.dumps(before_counts, sort_keys=True)}}"
            + f"|after={{json.dumps(after_counts, sort_keys=True)}}|added={{added_rows}}",
            flush=True,
        )

    if class_balance["method"] in {{
        "inverse_frequency",
        "clipped_inverse_frequency",
        "effective_number",
    }}:
        train_weights, class_weight_mapping = _inverse_frequency_sample_weight(
            (
                train_data[target_col]
                if explicit_training_partition
                else train_model[target_col]
            ),
            alpha=class_balance.get("alpha", 1.0),
            max_raw_weight=class_balance.get("max_raw_weight"),
            beta=class_balance.get("beta"),
        )
        train_data = train_data.copy()
        train_data[CLASS_WEIGHT_COL] = train_weights
        if not train_data[CLASS_WEIGHT_COL].index.equals(train_data[target_col].index):
            raise AssertionError("Sample weights are not aligned with training rows")
        class_weight_log_params = [f"|method={{class_balance['method']}}"]
        if class_balance["method"] == "effective_number":
            class_weight_log_params.append(f"|beta={{class_balance['beta']}}")
        else:
            class_weight_log_params.append(f"|alpha={{class_balance['alpha']}}")
        if class_balance["method"] == "clipped_inverse_frequency":
            class_weight_log_params.append(
                f"|max_raw_weight={{class_balance['max_raw_weight']}}"
            )
        print(
            "AIDE_RUNTIME|class_weights"
            + f"|scope={{class_weight_scope}}"
            + "".join(class_weight_log_params)
            + f"|mapping={{json.dumps(class_weight_mapping, sort_keys=True)}}",
            flush=True,
        )
        if class_weight_scope == "global_training":
            warnings.simplefilter("always", RuntimeWarning)
            warnings.warn(
                "Sample-weight scope is global_training (not fold_local); "
                "weights are derived from full training labels and are not fold-safe.",
                RuntimeWarning,
                stacklevel=1,
            )

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
    fit_kwargs.update(fit_args)
    training_started_at = time.time()
    with _quiet_model_output(working_dir):
        print("AIDE AutoGluon: starting fit", flush=True)
        predictor_kwargs = {{
            "label": target_col,
            "eval_metric": eval_metric,
            "path": str(model_dir),
            "verbosity": 2,
            "learner_kwargs": {{"ignored_columns": [id_col]}},
        }}
        if class_balance["method"] in {{
            "inverse_frequency",
            "clipped_inverse_frequency",
            "effective_number",
        }}:
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
        if AIDE_AG_CONFIG.get("save_prediction_artifacts", True):
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
        else:
            prediction_artifacts = {{"disabled": True}}
            print("AIDE AutoGluon: prediction artifact export disabled", flush=True)
        if defer_save_space:
            try:
                predictor.save_space(remove_data=True, remove_fit_stack=True)
                prediction_artifacts["save_space_after_artifacts"] = True
            except Exception as exc:
                prediction_artifacts["save_space_error"] = f"{{type(exc).__name__}}: {{exc}}"
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
    if prediction_artifacts.get("oof_predictions"):
        print(f"AIDE AutoGluon: OOF predictions saved to {{prediction_artifacts['oof_predictions']}}", flush=True)
    elif prediction_artifacts.get("oof_error"):
        print(f"AIDE AutoGluon: OOF predictions unavailable: {{prediction_artifacts['oof_error']}}", flush=True)
    if prediction_artifacts.get("validation_predictions"):
        print(f"AIDE AutoGluon: validation predictions saved to {{prediction_artifacts['validation_predictions']}}", flush=True)
    if prediction_artifacts.get("test_predictions"):
        print(f"AIDE AutoGluon: test predictions saved to {{prediction_artifacts['test_predictions']}}", flush=True)

    summary = "AutoGluon preprocess wrapper completed."
    run_stats = {{
        "feature_count": feature_count,
        "preprocess_time": float(preprocess_time),
        "training_time": float(training_time),
        "eval_metric": actual_eval_metric,
        "lightgbm_gpu_categorical_fallback": lightgbm_categorical_fallback_stats,
        "models": model_records,
        **_selected_model_metadata(predictor),
        "prediction_artifacts": prediction_artifacts,
    }}
    print(f"Validation {{eval_metric}}: {{metric_value:.6f}}")
    print("Submission saved successfully.")
    print(RESULT_MARKER + " " + json.dumps({{
        "is_bug": False,
        "summary": summary,
        "metric": metric_value,
        "eval_metric": actual_eval_metric,
        "lower_is_better": lower_is_better,
        "run_stats": run_stats,
        {research_marker_fields}
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
