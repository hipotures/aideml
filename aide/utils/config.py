"""configuration and setup utils"""

import gzip
import json
import os
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Callable, Hashable, Literal, cast

import coolname
import rich
from dotenv import load_dotenv
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from . import tree_export
from . import copytree, preproc_data, serialize
from .artifact_manifest import write_node_artifact_manifest
from .node_artifacts import (
    new_artifact_dir_name,
    node_artifact_dir as artifact_dir_for_node,
)
from .path_portability import sanitize_persisted_payload, sanitize_text

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("aide")
logger.setLevel(logging.WARNING)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_relative_path(value):
    if value is None:
        return value
    if isinstance(value, Path):
        path = value
    else:
        return value
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except ValueError:
        return value


def _portable_config_value(value):
    value = _repo_relative_path(value)
    if isinstance(value, dict):
        return {key: _portable_config_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_config_value(item) for item in value]
    return value


def _portable_config(cfg):
    if OmegaConf.is_config(cfg):
        container = OmegaConf.to_container(cfg, resolve=False, enum_to_str=True)
    elif is_dataclass(cfg):
        container = asdict(cfg)
    else:
        container = {
            key: value
            for key, value in vars(cfg).items()
            if not key.startswith("_")
        }
    return OmegaConf.create(_portable_config_value(container))


def _copy_prediction_artifact_gz(source: Path, destination: Path) -> None:
    if source.suffix == ".gz":
        shutil.copy2(source, destination)
        return
    with source.open("rb") as src, gzip.open(destination, "wb") as dst:
        shutil.copyfileobj(src, dst)


def _copy_prediction_dir(source_dir: Path, destination_dir: Path) -> None:
    if not source_dir.exists():
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("*.csv.gz"):
        shutil.copy2(source, destination_dir / source.name)


def _solution_helper_source_path() -> Path:
    return _repo_root() / "aide" / "solution_helpers.py"


def copy_solution_helper(destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        _solution_helper_source_path(),
        destination_dir / "aide_solution_helpers.py",
    )


def link_artifact_input_dir(workspace_dir: Path | str, artifact_dir: Path | str) -> None:
    source = Path(workspace_dir) / "input"
    destination = Path(artifact_dir) / "input"
    if not source.exists() or destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(os.path.relpath(source, start=destination.parent))


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float | None
    reasoning_effort: str | None = None
    timeout: int | None = None


@dataclass(frozen=True)
class ResolvedModelConfig:
    model: str
    reasoning_effort: str | None


VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
AGENT_MODE_ALIASES = {"autogluon": "autogluon_preprocess"}
DEPRECATED_CONFIG_KEYS = (
    "agent.search.seeded_base_max_children",
    "agent.autogluon.eval_metric",
)


def _split_model_effort(model: str) -> tuple[str, str | None]:
    if ":" not in model:
        return model, None
    base, suffix = model.rsplit(":", 1)
    if suffix not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"Invalid reasoning effort suffix in model {model!r}. "
            f"Expected one of: {', '.join(sorted(VALID_REASONING_EFFORTS))}."
        )
    if not base:
        raise ValueError(f"Invalid empty model name in {model!r}.")
    return base, suffix


def resolve_model_config(
    model: str,
    reasoning_effort: str | None,
    *,
    allow_suffix_override: bool = False,
) -> ResolvedModelConfig:
    base_model, suffix_effort = _split_model_effort(model)
    if suffix_effort is not None:
        if (
            reasoning_effort is not None
            and reasoning_effort != suffix_effort
            and not allow_suffix_override
        ):
            raise ValueError(
                "Model reasoning effort was provided twice with different values: "
                f"{model!r} and reasoning_effort={reasoning_effort!r}."
            )
        reasoning_effort = suffix_effort
    if reasoning_effort is not None and reasoning_effort not in VALID_REASONING_EFFORTS:
        raise ValueError(
            f"Invalid reasoning_effort {reasoning_effort!r}. "
            f"Expected one of: {', '.join(sorted(VALID_REASONING_EFFORTS))}."
        )
    return ResolvedModelConfig(base_model, reasoning_effort)


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    exploration_weight: float = 0.0
    best_score_min_children_before_exploration: int = 5
    disable_oom_saturated_parents: bool = False
    disable_timeout_debugging: bool = True
    hypothesis_child_order: str = "root_score"
    forced_root: str | None = None
    forced_hypothesis: str | None = None
    hypothesis_max_non_improving_children_per_parent: int = 10
    hypothesis_min_improvement_epsilon: float = 0.00006
    plateau_block_epsilon: float = 0.00001
    public_score_bonus_weight: float = 0.0
    public_score_bonus_cap: float = 0.0005


@dataclass
class AutoGluonConfig:
    profile: str = "full_boost"
    profiles: dict = field(default_factory=dict)
    presets: str = "medium_quality"
    time_limit: int = 600
    preprocess_timeout: int = 180
    validation_fraction: float = 0.2
    seed: int = 42
    included_model_types: list[str] | None = None
    validation_strategy: str | None = None
    use_gpu: bool | None = None
    hyperparameters: dict | None = None
    fit_args: dict | None = None


@dataclass
class AgentConfig:
    steps: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool

    code: StageConfig
    feedback: StageConfig

    search: SearchConfig
    memory_recent_steps: int = 50
    memory_full_recent_steps: int = 10
    include_parent_process_stdout: bool = True
    parent_process_stdout_max_bytes: int = 5000
    hypotheses: int = 0
    gpu: bool = False
    aux: bool | str | None = False
    mode: str = "legacy"
    autogluon: AutoGluonConfig = field(default_factory=AutoGluonConfig)


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool
    memory_limit_gb: float | None = 80.0


@dataclass
class ResearchConfig:
    enabled: bool = False
    mode: str = "llm"
    materialize: bool = True
    execute: bool = True
    every_steps: int = 10
    top_k_best: int = 5
    top_k_worst: int = 5
    previous_summary_count: int = 5
    timeout: int = 900
    model: str = "gpt-5.4-mini"
    reasoning_effort: str | None = "low"
    manual_sample_size: int = 3
    manual_seed: int = 42
    hypothesis_root_limit: int = 100
    hypothesis_root_order: str = "default"
    hypothesis_root_score_mode: str = "autogluon"
    hypothesis_root_generate_workers: int = 1
    seed_scored_roots: bool = True
    ignore_hypothesis_agent_modes: bool = False


@dataclass
class SynthesisConfig:
    enabled: bool = False
    every_scored_steps: int = 15
    top_k: int = 5
    source_scope: str = "current"
    source_runs: list[str] = field(default_factory=list)
    score_round_decimals: int = 5
    prediction_round_decimals: int = 5
    prediction_similarity_sample_size: int = 200
    prediction_similarity_min_common_sample_size: int = 100
    prediction_similarity_rmse_threshold: float = 0.015
    timeout: int = 900
    model: str = "gpt-5.4-mini"
    reasoning_effort: str | None = "low"


@dataclass
class RefactorConfig:
    enabled: bool = False
    timeout: int = 300
    max_input_chars: int = 240_000
    model: str = "gpt-5.4-mini"
    reasoning_effort: str | None = "low"


@dataclass
class WebDashboardConfig:
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 8766
    refresh_seconds: float = 2.0


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig
    research: ResearchConfig = field(default_factory=ResearchConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    refactor: RefactorConfig = field(default_factory=RefactorConfig)
    web: WebDashboardConfig = field(default_factory=WebDashboardConfig)


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml",
    use_cli_args=True,
    cli_args: Sequence[str] | None = None,
) -> Config:
    cfg = OmegaConf.load(path)
    load_dotenv(dotenv_path=Path(".env"), override=False)
    _apply_env_aliases(cfg)
    if use_cli_args:
        raw_cli_args = list(sys.argv[1:] if cli_args is None else cli_args)
        _validate_cli_model_effort_conflicts(raw_cli_args)
        raw_cli_args = _normalize_model_effort_cli_overrides(raw_cli_args)
        raw_cli_args = _normalize_forced_root_cli_overrides(raw_cli_args)
        cli_cfg = OmegaConf.from_dotlist(raw_cli_args)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def _cli_value(arg: str) -> tuple[str, str] | None:
    if "=" not in arg:
        return None
    key, value = arg.split("=", 1)
    return key, value


def _is_nullish(value: str) -> bool:
    return value.strip().lower() in {"", "null", "none", "~"}


def _is_missing_config_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return _is_nullish(value)
    return False


def _env_bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return None


def _env_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def _env_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _apply_env_aliases(cfg: Config) -> None:
    aliases = {
        "AIDE_GENERATE_REPORT": ("generate_report", _env_bool),
        "AIDE_PREPROCESS_DATA": ("preprocess_data", _env_bool),
        "AIDE_COPY_DATA": ("copy_data", _env_bool),
        "AIDE_AGENT_STEPS": ("agent.steps", _env_int),
        "AIDE_AGENT_MODE": ("agent.mode", str.strip),
        "AIDE_AGENT_AUX": ("agent.aux", str.strip),
        "AIDE_AGENT_GPU": ("agent.gpu", _env_bool),
        "AIDE_AGENT_HYPOTHESES": ("agent.hypotheses", _env_int),
        "AIDE_AGENT_K_FOLD_VALIDATION": ("agent.k_fold_validation", _env_int),
        "AIDE_AGENT_DATA_PREVIEW": ("agent.data_preview", _env_bool),
        "AIDE_AGENT_INCLUDE_PARENT_PROCESS_STDOUT": (
            "agent.include_parent_process_stdout",
            _env_bool,
        ),
        "AIDE_AGENT_PARENT_PROCESS_STDOUT_MAX_BYTES": (
            "agent.parent_process_stdout_max_bytes",
            _env_int,
        ),
        "AIDE_AGENT_CODE_MODEL": ("agent.code.model", str.strip),
        "AIDE_AGENT_CODE_REASONING_EFFORT": (
            "agent.code.reasoning_effort",
            str.strip,
        ),
        "AIDE_AGENT_CODE_TIMEOUT": ("agent.code.timeout", _env_int),
        "AIDE_AGENT_FEEDBACK_MODEL": ("agent.feedback.model", str.strip),
        "AIDE_AGENT_FEEDBACK_REASONING_EFFORT": (
            "agent.feedback.reasoning_effort",
            str.strip,
        ),
        "AIDE_AGENT_SEARCH_NUM_DRAFTS": ("agent.search.num_drafts", _env_int),
        "AIDE_AGENT_SEARCH_MAX_DEBUG_DEPTH": (
            "agent.search.max_debug_depth",
            _env_int,
        ),
        "AIDE_AGENT_SEARCH_DEBUG_PROB": ("agent.search.debug_prob", _env_float),
        "AIDE_AGENT_SEARCH_EXPLORATION_WEIGHT": (
            "agent.search.exploration_weight",
            _env_float,
        ),
        "AIDE_AGENT_SEARCH_FORCED_ROOT": ("agent.search.forced_root", str.strip),
        "AIDE_AGENT_SEARCH_FORCED_HYPOTHESIS": (
            "agent.search.forced_hypothesis",
            str.strip,
        ),
        "AIDE_AGENT_AUTOGLUON_PROFILE": ("agent.autogluon.profile", str.strip),
        "AIDE_AGENT_AUTOGLUON_TIME_LIMIT": (
            "agent.autogluon.time_limit",
            _env_int,
        ),
        "AIDE_AGENT_AUTOGLUON_PREPROCESS_TIMEOUT": (
            "agent.autogluon.preprocess_timeout",
            _env_int,
        ),
        "AIDE_AGENT_AUTOGLUON_USE_GPU": ("agent.autogluon.use_gpu", _env_bool),
        "AIDE_EXEC_TIMEOUT": ("exec.timeout", _env_int),
        "AIDE_EXEC_MEMORY_LIMIT_GB": ("exec.memory_limit_gb", _env_float),
        "AIDE_RESEARCH_ENABLED": ("research.enabled", _env_bool),
        "AIDE_RESEARCH_MODE": ("research.mode", str.strip),
        "AIDE_RESEARCH_MATERIALIZE": ("research.materialize", _env_bool),
        "AIDE_RESEARCH_EXECUTE": ("research.execute", _env_bool),
        "AIDE_RESEARCH_TIMEOUT": ("research.timeout", _env_int),
        "AIDE_RESEARCH_MODEL": ("research.model", str.strip),
        "AIDE_RESEARCH_REASONING_EFFORT": (
            "research.reasoning_effort",
            str.strip,
        ),
        "AIDE_RESEARCH_HYPOTHESIS_ROOT_LIMIT": (
            "research.hypothesis_root_limit",
            _env_int,
        ),
        "AIDE_RESEARCH_HYPOTHESIS_ROOT_ORDER": (
            "research.hypothesis_root_order",
            str.strip,
        ),
        "AIDE_RESEARCH_HYPOTHESIS_ROOT_SCORE_MODE": (
            "research.hypothesis_root_score_mode",
            str.strip,
        ),
        "AIDE_RESEARCH_HYPOTHESIS_ROOT_GENERATE_WORKERS": (
            "research.hypothesis_root_generate_workers",
            _env_int,
        ),
        "AIDE_RESEARCH_SEED_SCORED_ROOTS": (
            "research.seed_scored_roots",
            _env_bool,
        ),
        "AIDE_RESEARCH_IGNORE_HYPOTHESIS_AGENT_MODES": (
            "research.ignore_hypothesis_agent_modes",
            _env_bool,
        ),
        "AIDE_SYNTHESIS_ENABLED": ("synthesis.enabled", _env_bool),
        "AIDE_SYNTHESIS_TIMEOUT": ("synthesis.timeout", _env_int),
        "AIDE_SYNTHESIS_MODEL": ("synthesis.model", str.strip),
        "AIDE_SYNTHESIS_REASONING_EFFORT": (
            "synthesis.reasoning_effort",
            str.strip,
        ),
        "AIDE_REFACTOR_REASONING_EFFORT": ("refactor.reasoning_effort", str.strip),
        "AIDE_WEB_ENABLED": ("web.enabled", _env_bool),
        "AIDE_WEB_HOST": ("web.host", str.strip),
        "AIDE_WEB_PORT": ("web.port", _env_int),
    }
    for env_name, (path, parser) in aliases.items():
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        value = parser(raw)
        if value is None:
            continue
        OmegaConf.update(cfg, path, value, merge=False)


def _apply_project_env_defaults(cfg: Config) -> None:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    if _is_missing_config_value(cfg.data_dir):
        data_dir = os.getenv("AIDE_PROJECT_DATA_DIR", "").strip()
        if data_dir:
            cfg.data_dir = data_dir
    if _is_missing_config_value(cfg.desc_file) and _is_missing_config_value(cfg.goal):
        desc_file = os.getenv("AIDE_PROJECT_DESC_FILE", "").strip()
        if desc_file:
            cfg.desc_file = desc_file
    refactor_enabled = os.getenv("AIDE_REFACTOR_ENABLED", "").strip().lower()
    if refactor_enabled in {"1", "true", "yes", "on"}:
        cfg.refactor.enabled = True
    elif refactor_enabled in {"0", "false", "no", "off"}:
        cfg.refactor.enabled = False
    refactor_model = os.getenv("AIDE_REFACTOR_MODEL", "").strip()
    if refactor_model:
        cfg.refactor.model = refactor_model
    refactor_timeout = os.getenv("AIDE_REFACTOR_TIMEOUT_S", "").strip()
    if refactor_timeout:
        try:
            cfg.refactor.timeout = int(refactor_timeout)
        except ValueError:
            pass
    refactor_max_input = os.getenv("AIDE_REFACTOR_MAX_INPUT_CHARS", "").strip()
    if refactor_max_input:
        try:
            cfg.refactor.max_input_chars = int(refactor_max_input)
        except ValueError:
            pass


def _validate_cli_model_effort_conflicts(cli_args: Sequence[str]) -> None:
    values = dict(item for arg in cli_args if (item := _cli_value(arg)) is not None)
    for model_key in _model_config_keys():
        model = values.get(model_key)
        if model is None:
            continue
        _base, suffix_effort = _split_model_effort(model)
        if suffix_effort is None:
            continue
        effort_key = model_key.removesuffix(".model") + ".reasoning_effort"
        explicit_effort = values.get(effort_key)
        if explicit_effort is not None and not _is_nullish(explicit_effort):
            raise ValueError(
                f"{model_key} uses a model:reasoning_effort suffix, but "
                f"{effort_key} is also set. Use only one form."
            )


def _model_config_keys() -> list[str]:
    return [
        "agent.code.model",
        "agent.feedback.model",
        "report.model",
        "research.model",
        "synthesis.model",
        "refactor.model",
    ]


def _normalize_model_effort_cli_overrides(cli_args: Sequence[str]) -> list[str]:
    normalized = list(cli_args)
    values = dict(item for arg in cli_args if (item := _cli_value(arg)) is not None)
    for model_key in _model_config_keys():
        model = values.get(model_key)
        if model is None:
            continue
        _base, suffix_effort = _split_model_effort(model)
        effort_key = model_key.removesuffix(".model") + ".reasoning_effort"
        if suffix_effort is None and effort_key not in values:
            normalized.append(f"{effort_key}=null")
    return normalized


def _normalize_forced_root_cli_overrides(cli_args: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for arg in cli_args:
        item = _cli_value(arg)
        if item is None:
            normalized.append(arg)
            continue
        key, value = item
        aliases = {
            "forced_root": "agent.search.forced_root",
            "agent.search.forced_root": "agent.search.forced_root",
            "forced_hypothesis": "agent.search.forced_hypothesis",
            "agent.search.forced_hypothesis": "agent.search.forced_hypothesis",
        }
        target_key = aliases.get(key)
        if target_key is None:
            normalized.append(arg)
            continue
        if _is_nullish(value):
            normalized.append(f"{target_key}=null")
            continue
        normalized.append(f"{target_key}={json.dumps(value)}")
    return normalized


def load_cfg(
    path: Path = Path(__file__).parent / "config.yaml",
    cli_args: Sequence[str] | None = None,
) -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path, cli_args=cli_args))


def prep_cfg(cfg: Config):
    _apply_project_env_defaults(cfg)

    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    data_dir_value = str(cfg.data_dir)
    if data_dir_value.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / data_dir_value
    elif data_dir_value.startswith("aide/example_tasks/"):
        cfg.data_dir = _repo_root() / data_dir_value
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        desc_file_value = str(cfg.desc_file)
        if desc_file_value.startswith("aide/example_tasks/"):
            cfg.desc_file = _repo_root() / desc_file_value
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    ind = max(_get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir))
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
    cfg.exp_name = f"{ind}-{cfg.exp_name}"

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    _drop_deprecated_config_keys(cfg)
    cfg = OmegaConf.merge(cfg_schema, cfg)
    _normalize_agent_mode_aliases(cfg)
    _normalize_hypothesis_pipeline_config(cfg)
    _resolve_all_model_configs(cfg)

    return cast(Config, cfg)


def _drop_deprecated_config_keys(cfg: Config) -> None:
    for key_path in DEPRECATED_CONFIG_KEYS:
        node = cfg
        parts = key_path.split(".")
        for part in parts[:-1]:
            if part not in node:
                node = None
                break
            node = node[part]
        if node is not None and parts[-1] in node:
            del node[parts[-1]]


def _normalize_agent_mode_aliases(cfg: Config) -> None:
    mode = getattr(cfg.agent, "mode", None)
    if isinstance(mode, str) and mode in AGENT_MODE_ALIASES:
        cfg.agent.mode = AGENT_MODE_ALIASES[mode]


def _normalize_hypothesis_pipeline_config(cfg: Config) -> None:
    try:
        root_quota = int(getattr(cfg.agent, "hypotheses", 0) or 0)
    except (TypeError, ValueError):
        root_quota = 0
    cfg.agent.hypotheses = max(root_quota, 0)
    if cfg.agent.hypotheses <= 0:
        return
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"


def _resolve_stage_config(stage: StageConfig) -> None:
    resolved = resolve_model_config(
        stage.model,
        stage.reasoning_effort,
        allow_suffix_override=True,
    )
    stage.model = resolved.model
    stage.reasoning_effort = resolved.reasoning_effort


def _resolve_model_attrs(section) -> None:
    resolved = resolve_model_config(
        section.model,
        section.reasoning_effort,
        allow_suffix_override=True,
    )
    section.model = resolved.model
    section.reasoning_effort = resolved.reasoning_effort


def _resolve_all_model_configs(cfg: Config) -> None:
    _resolve_stage_config(cfg.agent.code)
    _resolve_stage_config(cfg.agent.feedback)
    _resolve_stage_config(cfg.report)
    _resolve_model_attrs(cfg.research)
    _resolve_model_attrs(cfg.synthesis)
    _resolve_model_attrs(cfg.refactor)


def _best_solution_node(journal):
    scored_nodes = [
        node
        for node in journal.nodes
        if node.metric is not None and node.metric.value is not None
    ]
    if scored_nodes:
        return max(scored_nodes, key=lambda node: node.metric)
    return journal.nodes[-1] if journal.nodes else None


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return _with_aux_task_desc_note(f.read(), cfg)

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return _with_aux_task_desc_note(task_desc, cfg)


AuxMode = Literal["off", "merged", "file"]
RESERVED_AUX_INPUT_NAMES = {
    "train.csv",
    "train.csv.gz",
    "test.csv",
    "test.csv.gz",
    "sample_submission.csv",
    "sample_submission.csv.gz",
}


def aux_mode(cfg: Config) -> AuxMode:
    raw = getattr(cfg.agent, "aux", False)
    if raw is None or raw is False:
        return "off"
    if raw is True:
        return "merged"
    if not isinstance(raw, str):
        raise ValueError(
            "agent.aux must be false, true, 'merged', 'merge', or a single "
            ".csv/.csv.gz filename."
        )

    value = raw.strip()
    lowered = value.lower()
    if lowered in {"", "false", "off", "none", "null", "~"}:
        return "off"
    if lowered in {"true", "merged", "merge"}:
        return "merged"
    _validate_aux_filename(value)
    return "file"


def aux_file_name(cfg: Config) -> str | None:
    if aux_mode(cfg) != "file":
        return None
    return str(getattr(cfg.agent, "aux")).strip()


def _validate_aux_filename(value: str) -> None:
    if "," in value or value.startswith("[") or value.endswith("]"):
        raise ValueError("agent.aux accepts only one auxiliary CSV file per run.")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.name != value or ".." in candidate.parts:
        raise ValueError(
            "agent.aux file mode accepts a filename only, not a path: "
            f"{value!r}"
        )
    if not (value.endswith(".csv") or value.endswith(".csv.gz")):
        raise ValueError(
            "agent.aux file mode requires a .csv or .csv.gz filename: "
            f"{value!r}"
        )
    if value in RESERVED_AUX_INPUT_NAMES:
        raise ValueError(f"agent.aux cannot overwrite competition input file {value!r}.")


def resolve_aux_source_file(cfg: Config) -> Path | None:
    name = aux_file_name(cfg)
    if name is None:
        return None

    direct = cfg.data_dir / name
    if direct.exists() and direct.is_file():
        return direct

    matches = sorted(path for path in cfg.data_dir.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(
            f"agent.aux={name!r} did not match any file under {cfg.data_dir}"
        )
    if len(matches) > 1:
        formatted = ", ".join(str(path.relative_to(cfg.data_dir)) for path in matches)
        raise ValueError(
            f"agent.aux={name!r} matched multiple files under {cfg.data_dir}: "
            f"{formatted}"
        )
    return matches[0]


def _aux_description_stem(source: Path) -> str:
    name = source.name
    if name.endswith(".csv.gz"):
        return name[: -len(".csv.gz")]
    return source.stem


def resolve_aux_description_file(cfg: Config) -> Path | None:
    source = resolve_aux_source_file(cfg)
    if source is None:
        return None
    stem = _aux_description_stem(source)
    for suffix in (".txt", ".md"):
        candidate = source.with_name(f"{stem}{suffix}")
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _append_task_desc_note(task_desc, title: str, body: str):
    if isinstance(task_desc, str):
        return f"{task_desc.rstrip()}\n\n{title}\n\n{body.rstrip()}\n"

    task_desc = dict(task_desc)
    task_desc[title.rstrip(":")] = body.rstrip()
    return task_desc


def _with_aux_task_desc_note(task_desc, cfg: Config):
    mode = aux_mode(cfg)
    if mode == "off":
        return task_desc

    if mode == "file":
        description_path = resolve_aux_description_file(cfg)
        if description_path is None:
            return task_desc
        name = aux_file_name(cfg)
        body = description_path.read_text(encoding="utf-8")
        return _append_task_desc_note(
            task_desc,
            f"Additional auxiliary data description for `{name}`:",
            body,
        )

    note = (
        "Additional data note: In this run, `train.csv.gz` has been prebuilt by "
        "merging the competition training rows with the original/external F1 "
        "strategy dataset. The competition train/test files are synthetic Kaggle "
        "Playground tabular data; the auxiliary rows come from the original F1 "
        "strategy data. The merged train keeps the same feature columns as the "
        "competition train file; source/provenance-only columns from the auxiliary "
        "dataset are not exposed to the model. `test.csv.gz` remains the "
        "competition test set only with the same feature columns as the original "
        "competition test file."
    )

    return _append_task_desc_note(task_desc, "Additional data note:", note)


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    copy_solution_helper(cfg.workspace_dir)

    mode = aux_mode(cfg)
    if mode == "merged":
        copy_aux_input(cfg)
    else:
        copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
        hide_aux_input(cfg.workspace_dir / "input")
        if mode == "file":
            copy_aux_file_input(cfg)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def _copy_or_symlink_file(source: Path, destination: Path, *, copy_data: bool) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if copy_data:
        shutil.copyfile(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, start=destination.parent))


def copy_aux_input(cfg: Config) -> None:
    source_train = cfg.data_dir / "train-aux.csv.gz"
    source_test = cfg.data_dir / "test-aux.csv.gz"
    source_sample = cfg.data_dir / "sample_submission.csv.gz"
    if not source_train.exists():
        raise FileNotFoundError(
            f"agent.aux=true requires prebuilt merged train file: {source_train}"
        )
    if not source_test.exists():
        raise FileNotFoundError(
            f"agent.aux=true requires prebuilt auxiliary test file: {source_test}"
        )
    if not source_sample.exists():
        raise FileNotFoundError(
            f"agent.aux=true requires sample submission file: {source_sample}"
        )
    input_dir = cfg.workspace_dir / "input"
    _copy_or_symlink_file(
        source_train,
        input_dir / "train.csv.gz",
        copy_data=bool(cfg.copy_data),
    )
    _copy_or_symlink_file(
        source_test,
        input_dir / "test.csv.gz",
        copy_data=bool(cfg.copy_data),
    )
    _copy_or_symlink_file(
        source_sample,
        input_dir / "sample_submission.csv.gz",
        copy_data=bool(cfg.copy_data),
    )
    validate_aux_workspace_input(input_dir)


def copy_aux_file_input(cfg: Config) -> None:
    source = resolve_aux_source_file(cfg)
    if source is None:
        return
    input_dir = cfg.workspace_dir / "input"
    destination = input_dir / source.name

    _copy_or_symlink_file(source, destination, copy_data=True)

    source_rel = source.relative_to(cfg.data_dir)
    if len(source_rel.parts) > 1:
        copied_top_level = input_dir / source_rel.parts[0]
        if copied_top_level.exists() or copied_top_level.is_symlink():
            if copied_top_level.is_symlink() or copied_top_level.is_file():
                copied_top_level.unlink()
            else:
                shutil.rmtree(copied_top_level)


def validate_aux_workspace_input(input_dir: Path) -> None:
    expected = {"train.csv.gz", "test.csv.gz", "sample_submission.csv.gz"}
    present = {path.name for path in input_dir.iterdir() if path.is_file() or path.is_symlink()}
    if present != expected:
        raise ValueError(
            "agent.aux=true workspace input must contain exactly "
            f"{sorted(expected)}, got {sorted(present)}"
        )


def hide_aux_input(input_dir: Path) -> None:
    for name in ("train-aux.csv.gz", "test-aux.csv.gz"):
        path = input_dir / name
        if path.exists() or path.is_symlink():
            path.unlink()


def _node_error_text(node) -> str | None:
    if not getattr(node, "is_buggy", False):
        return None

    sections: list[str] = []
    if getattr(node, "exc_type", None):
        sections.append(f"Exception type:\n{node.exc_type}")
    if getattr(node, "exc_info", None):
        sections.append(
            "Exception info:\n"
            + json.dumps(
                sanitize_persisted_payload(node.exc_info),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    if getattr(node, "exc_stack", None):
        sections.append(
            "Exception stack:\n"
            + json.dumps(
                sanitize_persisted_payload(node.exc_stack),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    if getattr(node, "_term_out", None):
        sections.append("Terminal output:\n" + sanitize_text("".join(node._term_out)).rstrip())
    if getattr(node, "analysis", None):
        sections.append("Analysis:\n" + sanitize_text(str(node.analysis)).rstrip())
    if getattr(node, "submission_validation", None):
        sections.append(
            "Submission validation:\n"
            + json.dumps(
                sanitize_persisted_payload(node.submission_validation),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    return "\n\n".join(section for section in sections if section).strip() or "Unknown error"


def _save_node_artifacts(cfg: Config, node) -> None:
    artifact_dir = artifact_dir_for_node(cfg.log_dir, node)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    link_artifact_input_dir(cfg.workspace_dir, artifact_dir)

    with open(artifact_dir / "solution.py", "w") as f:
        f.write(node.code)

    workspace_helper = cfg.workspace_dir / "aide_solution_helpers.py"
    if workspace_helper.exists():
        shutil.copy2(workspace_helper, artifact_dir / "aide_solution_helpers.py")

    submission_path = cfg.workspace_dir / "working" / "submission.csv"
    if submission_path.exists() and submission_path.stat().st_mtime >= node.ctime:
        shutil.copy2(submission_path, artifact_dir / "submission.csv")

    for name in (
        "oof_predictions.csv",
        "test_predictions.csv",
        "validation_predictions.csv",
    ):
        gzip_name = f"{name}.gz"
        for prediction_path in (
            cfg.workspace_dir / "working" / gzip_name,
            cfg.workspace_dir / "working" / name,
        ):
            if prediction_path.exists() and prediction_path.stat().st_mtime >= node.ctime:
                _copy_prediction_artifact_gz(prediction_path, artifact_dir / gzip_name)
                break
    _copy_prediction_dir(
        cfg.workspace_dir / "working" / "model_predictions",
        artifact_dir / "model_predictions",
    )

    error_text = _node_error_text(node)
    if error_text is not None:
        (artifact_dir / "error.txt").write_text(error_text + "\n")

    write_node_artifact_manifest(cfg=cfg, node=node, artifact_dir=artifact_dir)


def _ensure_node_artifact_slot(cfg: Config, node) -> Path:
    if isinstance(node.artifact_dir_name, str) and node.artifact_dir_name.strip():
        node.artifact_dir_name = node.artifact_dir_name.strip()
        artifact_dir = artifact_dir_for_node(cfg.log_dir, node)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        link_artifact_input_dir(cfg.workspace_dir, artifact_dir)
        return artifact_dir

    artifacts_dir = Path(cfg.log_dir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    while True:
        dir_name = new_artifact_dir_name(ctime=node.ctime, step=node.step)
        artifact_dir = artifacts_dir / dir_name
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        node.artifact_dir_name = dir_name
        link_artifact_input_dir(cfg.workspace_dir, artifact_dir)
        return artifact_dir


def _write_missing_solution_artifacts(cfg: Config, journal, current_node=None) -> None:
    for node in journal.nodes:
        artifact_dir = _ensure_node_artifact_slot(cfg, node)
        solution_path = artifact_dir / "solution.py"
        if node is current_node or not solution_path.exists():
            solution_path.write_text(node.code, encoding="utf-8")


def save_run(
    cfg: Config,
    journal,
    current_node=None,
    progress_callback: Callable[[str], None] | None = None,
):
    def notify(message: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(message)
            except Exception:
                logger.exception("Progress callback failed while saving run")

    notify("Preparing log directory")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    notify("Saving node artifacts")
    _write_missing_solution_artifacts(cfg, journal, current_node=current_node)
    if current_node is not None:
        _save_node_artifacts(cfg, current_node)

    # save journal
    notify("Saving journal")
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # save config
    notify("Saving config")
    OmegaConf.save(config=_portable_config(cfg), f=cfg.log_dir / "config.yaml")
    # create the tree + code visualization
    notify("Rendering tree HTML")
    tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # save the best found solution
    notify("Saving best solution")
    best_node = _best_solution_node(journal)
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
