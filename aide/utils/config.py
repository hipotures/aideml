"""configuration and setup utils"""

import datetime as dt
import shutil
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


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class AutoGluonConfig:
    profile: str = "full_boost"
    presets: str = "medium_quality"
    time_limit: int = 600
    validation_fraction: float = 0.2
    seed: int = 42
    use_gpu: bool = False
    eval_metric: str = "auto"
    included_model_types: list[str] | None = None
    fit_args: dict = field(
        default_factory=lambda: {
            "save_space": True,
            "fit_weighted_ensemble": False,
            "auto_stack": False,
        }
    )


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
    model: str = "gpt-5.5"
    reasoning_effort: str = "medium"


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
    model: str = "gpt-5.5"
    reasoning_effort: str = "medium"


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
        cli_cfg = (
            OmegaConf.from_dotlist(list(cli_args))
            if cli_args is not None
            else OmegaConf.from_cli()
        )
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


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

    return cast(Config, cfg)


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


def _save_node_artifacts(cfg: Config, node) -> None:
    timestamp = _node_artifact_timestamp(node)
    artifact_dir = cfg.log_dir / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "solution.py", "w") as f:
        f.write(node.code)

    submission_path = cfg.workspace_dir / "working" / "submission.csv"
    if submission_path.exists() and submission_path.stat().st_mtime >= node.ctime:
        shutil.copy2(submission_path, artifact_dir / "submission.csv")


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
