"""configuration and setup utils"""

import datetime as dt
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable, cast

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from . import tree_export
from . import copytree, preproc_data, serialize
from .artifact_manifest import write_node_artifact_manifest

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("aide")
logger.setLevel(logging.WARNING)


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float | None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class ResolvedModelConfig:
    model: str
    reasoning_effort: str | None


VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


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
    exploration_weight: float = 0.05


@dataclass
class AutoGluonConfig:
    profile: str = "full_boost"
    profiles: dict = field(default_factory=dict)
    presets: str = "medium_quality"
    time_limit: int = 600
    validation_fraction: float = 0.2
    seed: int = 42
    eval_metric: str = "auto"
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
    every_steps: int = 10
    top_k_best: int = 5
    top_k_worst: int = 5
    previous_summary_count: int = 5
    timeout: int = 900
    model: str = "gpt-5.4-mini"
    reasoning_effort: str | None = "low"


@dataclass
class SynthesisConfig:
    enabled: bool = False
    every_scored_steps: int = 15
    top_k: int = 5
    source_runs: list[str] = field(default_factory=list)
    score_round_decimals: int = 5
    prediction_round_decimals: int = 5
    prediction_similarity_rmse_threshold: float = 0.015
    timeout: int = 900
    model: str = "gpt-5.4-mini"
    reasoning_effort: str | None = "low"


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
    if use_cli_args:
        raw_cli_args = list(sys.argv[1:] if cli_args is None else cli_args)
        _validate_cli_model_effort_conflicts(raw_cli_args)
        raw_cli_args = _normalize_model_effort_cli_overrides(raw_cli_args)
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


def load_cfg(
    path: Path = Path(__file__).parent / "config.yaml",
    cli_args: Sequence[str] | None = None,
) -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path, cli_args=cli_args))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
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
    cfg = OmegaConf.merge(cfg_schema, cfg)
    _resolve_all_model_configs(cfg)

    return cast(Config, cfg)


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
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def _node_artifact_timestamp(node) -> str:
    return dt.datetime.fromtimestamp(node.ctime).strftime("%Y%m%dT%H%M%S")


def _node_error_text(node) -> str | None:
    if not getattr(node, "is_buggy", False):
        return None

    sections: list[str] = []
    if getattr(node, "exc_type", None):
        sections.append(f"Exception type:\n{node.exc_type}")
    if getattr(node, "exc_info", None):
        sections.append(
            "Exception info:\n"
            + json.dumps(node.exc_info, indent=2, ensure_ascii=False, default=str)
        )
    if getattr(node, "exc_stack", None):
        sections.append(
            "Exception stack:\n"
            + json.dumps(node.exc_stack, indent=2, ensure_ascii=False, default=str)
        )
    if getattr(node, "_term_out", None):
        sections.append("Terminal output:\n" + "".join(node._term_out).rstrip())
    if getattr(node, "analysis", None):
        sections.append("Analysis:\n" + str(node.analysis).rstrip())
    if getattr(node, "submission_validation", None):
        sections.append(
            "Submission validation:\n"
            + json.dumps(
                node.submission_validation,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    return "\n\n".join(section for section in sections if section).strip() or "Unknown error"


def _save_node_artifacts(cfg: Config, node) -> None:
    timestamp = _node_artifact_timestamp(node)
    artifact_dir = cfg.log_dir / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "solution.py", "w") as f:
        f.write(node.code)

    submission_path = cfg.workspace_dir / "working" / "submission.csv"
    if submission_path.exists() and submission_path.stat().st_mtime >= node.ctime:
        shutil.copy2(submission_path, artifact_dir / "submission.csv")

    error_text = _node_error_text(node)
    if error_text is not None:
        (artifact_dir / "error.txt").write_text(error_text + "\n")

    write_node_artifact_manifest(cfg=cfg, node=node, artifact_dir=artifact_dir)


def save_run(
    cfg: Config,
    journal,
    current_node=None,
    progress_callback: Callable[[str], None] | None = None,
):
    def notify(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    notify("Preparing log directory")
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    notify("Saving journal")
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # save config
    notify("Saving config")
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    # create the tree + code visualization
    notify("Rendering tree HTML")
    tree_export.generate(cfg, journal, cfg.log_dir / "tree_plot.html")
    # save the best found solution
    notify("Saving best solution")
    best_node = journal.get_best_node(only_good=False)
    with open(cfg.log_dir / "best_solution.py", "w") as f:
        f.write(best_node.code)

    if current_node is not None:
        notify("Saving node artifacts")
        _save_node_artifacts(cfg, current_node)
