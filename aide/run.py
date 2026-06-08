import atexit
import base64
import datetime as dt
import io
import json
import logging
import os
import queue
import re
import select
import shutil
import sys
import termios
import threading
import time
import textwrap
import tty
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, cast

from .agent import Agent
from .interpreter import ExecutionInterrupted, Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from .research import (
    HypothesisRootReservation,
    ResearchAdvisor,
    _compatible_manual_hypotheses,
    clear_hypothesis_root_generation_failure,
    count_scored_working_nodes,
    effective_hypothesis_root_limit,
    hypothesis_id_for_node,
    load_manual_hypothesis_library,
    record_manual_prompt_node,
    record_hypothesis_root_generation_failure,
    reserve_hypothesis_roots,
    scored_hypothesis_root_nodes,
)
from .synthesis import SYNTHESIS_PLAN_PREFIX, SynthesisAdvisor, SynthesisNode
from .telegram_notifications import (
    append_node_with_best_score_notification,
    send_telegram_test_message,
)
from .autogluon_preprocess import AGENT_MODE, BASELINE_PLAN_PREFIX
from .utils.artifact_manifest import SEEDED_BASE_PLAN_PREFIX
from .utils.seed_artifact import (
    SeedArtifactSource,
    find_seed_artifact,
    seed_journal_from_artifact,
    source_is_autogluon,
)
from omegaconf import OmegaConf
from rich.console import Console, Group
from rich._loop import loop_last
from rich.layout import Layout
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils import serialize
from .utils.config import (
    Config,
    _drop_deprecated_config_keys,
    _load_cfg,
    _normalize_agent_mode_aliases,
    _normalize_forced_root_cli_overrides,
    _resolve_all_model_configs,
    _normalize_model_effort_cli_overrides,
    _validate_cli_model_effort_conflicts,
    aux_file_name,
    aux_mode,
    copy_aux_file_input,
    load_task_desc,
    prep_agent_workspace,
    save_run,
    load_cfg,
)
from .utils.memory_debug import MemoryDebugLogger
from .utils.metric import MetricValue, WorstMetricValue
from .utils.node_artifacts import (
    new_artifact_dir_name,
    node_artifact_dir as artifact_dir_for_node,
    node_artifact_submission_path as artifact_submission_path_for_node,
)
from .utils.plateau import (
    DEFAULT_PLATEAU_BLOCK_EPSILON,
    is_plateau_blocked_descendant,
)
from .utils.public_scores import (
    load_public_scores_by_node_id,
    public_adjusted_oriented_score,
)
from .utils.resource_monitor import (
    DEFAULT_RESOURCE_HISTORY_WINDOW_SECONDS,
    ResourceHistory,
    ResourceSnapshot,
)
from .utils.submission_validation import (
    file_signature,
    find_sample_submission,
    validate_submission_file,
    validate_workspace_submission,
)
from .utils.response import extract_text_up_to_code
from .web_dashboard.server import AideWebServer
from .web_dashboard.state import (
    WebDashboardSnapshot,
    WebDashboardState,
    WebRunDatum,
    WebRunSection,
)
from .web_dashboard.tree import build_web_tree_lines

logger = logging.getLogger("aide")


DEFAULT_PUBLIC_TREE_SCORE_BONUS_WEIGHT = 0.5


@dataclass(frozen=True)
class ResumeRequest:
    requested: bool = False
    run_id: str | None = None

    @property
    def use_latest(self) -> bool:
        return self.requested and self.run_id is None


@dataclass(frozen=True)
class RuntimeOptions:
    show_invalid_submission_branches: bool = False
    force_check_submissions: bool = False
    telegram_test_message: bool = False
    debug: bool = False
    skip_execution: bool = False
    generate_only_hypothesis_ids: tuple[str, ...] = ()
    seed_sha_prefix: str | None = None
    seed_source_run: str | None = None
    web_enabled: bool = False
    web_host: str | None = None
    web_port: int | None = None
    public_tree_scores: bool = False


@dataclass(frozen=True)
class TreeViewItem:
    item_id: str
    parent_id: str | None
    line: Text
    node: Node | None = None
    focus_start: int = 0


@dataclass(frozen=True)
class TreeView:
    items: list[TreeViewItem]
    index_by_id: dict[str, int]
    parent_by_id: dict[str, str | None]
    children_by_id: dict[str, list[str]]
    active_item_id: str | None = None


@dataclass(frozen=True)
class ActiveRootGeneration:
    hypothesis_id: str
    launched_index: int


@dataclass
class ParallelRootFailureState:
    attempts_by_hypothesis: dict[str, int] = field(default_factory=dict)
    stop_refill: bool = False

    def record_failure(self, hypothesis_id: str, exc: BaseException) -> bool:
        _ = exc
        attempts = self.attempts_by_hypothesis.get(hypothesis_id, 0) + 1
        self.attempts_by_hypothesis[hypothesis_id] = attempts
        if attempts >= 3:
            self.stop_refill = True
            return True
        return False


@dataclass(frozen=True)
class ParallelRootJob:
    reservation: HypothesisRootReservation
    node_ctime: float
    artifact_dir_name: str
    artifact_dir: Path
    launched_index: int


@dataclass(frozen=True)
class ParallelRootResult:
    job: ParallelRootJob
    node: Node


@dataclass(frozen=True)
class LastErrorRecord:
    node: Node
    lines: list[str]


@dataclass(frozen=True)
class CheckpointStatusRecord:
    label: str
    status: str | None
    timestamp: str | None


def parse_runtime_args(
    argv: list[str],
) -> tuple[ResumeRequest, RuntimeOptions, list[str]]:
    def require_option_value(name: str, index: int) -> tuple[str, int]:
        next_index = index + 1
        if next_index >= len(argv) or argv[next_index].startswith("-"):
            raise ValueError(f"`{name}` requires a value.")
        return argv[next_index], next_index

    def parse_port(name: str, value: str) -> int:
        try:
            port = int(value)
        except ValueError as exc:
            raise ValueError(f"`{name}` must be an integer port.") from exc
        if port < 1 or port > 65535:
            raise ValueError(f"`{name}` must be in range 1..65535.")
        return port

    remaining: list[str] = []
    resume = ResumeRequest()
    runtime = RuntimeOptions()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--resume":
            if resume.requested:
                raise ValueError("`--resume` can only be provided once.")
            run_id = None
            next_i = i + 1
            if (
                next_i < len(argv)
                and "=" not in argv[next_i]
                and not argv[next_i].startswith("-")
            ):
                run_id = argv[next_i]
                i += 1
            resume = ResumeRequest(requested=True, run_id=run_id)
        elif arg.startswith("--resume="):
            if resume.requested:
                raise ValueError("`--resume` can only be provided once.")
            run_id = arg.split("=", 1)[1] or None
            resume = ResumeRequest(requested=True, run_id=run_id)
        elif arg == "--show-invalid-submission-branches":
            runtime = replace(runtime, show_invalid_submission_branches=True)
        elif arg == "--public-tree-scores":
            runtime = replace(runtime, public_tree_scores=True)
        elif arg == "--force-check-submissions":
            runtime = replace(runtime, force_check_submissions=True)
        elif arg == "--telegram-test-message":
            runtime = replace(runtime, telegram_test_message=True)
        elif arg == "--debug":
            runtime = replace(runtime, debug=True)
        elif arg == "--web":
            runtime = replace(runtime, web_enabled=True)
        elif arg == "--web-host":
            value, i = require_option_value("--web-host", i)
            runtime = replace(runtime, web_host=value)
        elif arg.startswith("--web-host="):
            runtime = replace(runtime, web_host=arg.split("=", 1)[1])
        elif arg == "--web-port":
            value, i = require_option_value("--web-port", i)
            runtime = replace(runtime, web_port=parse_port("--web-port", value))
        elif arg.startswith("--web-port="):
            runtime = replace(
                runtime,
                web_port=parse_port("--web-port", arg.split("=", 1)[1]),
            )
        elif arg in {"--skip-execution", "--generate-only"}:
            hypothesis_ids: list[str] = []
            if arg == "--generate-only":
                next_i = i + 1
                while next_i < len(argv):
                    next_arg = argv[next_i]
                    if next_arg.startswith("-") or "=" in next_arg:
                        break
                    hypothesis_ids.extend(
                        part.strip()
                        for part in next_arg.split(",")
                        if part.strip()
                    )
                    next_i += 1
                if hypothesis_ids:
                    i = next_i - 1
            runtime = replace(
                runtime,
                skip_execution=True,
                generate_only_hypothesis_ids=tuple(
                    dict.fromkeys(
                        [
                            *runtime.generate_only_hypothesis_ids,
                            *hypothesis_ids,
                        ]
                    )
                ),
            )
        elif arg == "--seed-from-sha":
            if runtime.seed_sha_prefix is not None:
                raise ValueError("`--seed-from-sha` can only be provided once.")
            value, i = require_option_value("--seed-from-sha", i)
            runtime = replace(runtime, seed_sha_prefix=value)
        elif arg.startswith("--seed-from-sha="):
            if runtime.seed_sha_prefix is not None:
                raise ValueError("`--seed-from-sha` can only be provided once.")
            runtime = replace(runtime, seed_sha_prefix=arg.split("=", 1)[1])
        elif arg == "--seed-source-run":
            if runtime.seed_source_run is not None:
                raise ValueError("`--seed-source-run` can only be provided once.")
            value, i = require_option_value("--seed-source-run", i)
            runtime = replace(runtime, seed_source_run=value)
        elif arg.startswith("--seed-source-run="):
            if runtime.seed_source_run is not None:
                raise ValueError("`--seed-source-run` can only be provided once.")
            runtime = replace(runtime, seed_source_run=arg.split("=", 1)[1] or None)
        else:
            remaining.append(arg)
        i += 1
    if resume.requested and runtime.seed_sha_prefix is not None:
        raise ValueError("`--seed-from-sha` cannot be combined with `--resume`.")
    if runtime.seed_source_run is not None and runtime.seed_sha_prefix is None:
        raise ValueError("`--seed-source-run` requires `--seed-from-sha`.")
    return resume, runtime, remaining


def validate_hypothesis_root_generate_workers(cfg: Config) -> int:
    raw = getattr(cfg.research, "hypothesis_root_generate_workers", 1)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(
            "research.hypothesis_root_generate_workers must be an integer from 1 to 8."
        )
    if raw < 1 or raw > 8:
        raise ValueError(
            "research.hypothesis_root_generate_workers must be an integer from 1 to 8."
        )
    return raw


def parse_resume_args(argv: list[str]) -> tuple[ResumeRequest, list[str]]:
    resume, _runtime, remaining = parse_runtime_args(argv)
    return resume, remaining


def find_latest_run_id(top_log_dir: Path) -> str:
    journals = list(top_log_dir.glob("*/journal.json"))
    if not journals:
        raise FileNotFoundError(f"No resumable runs found in {top_log_dir}.")
    return max(journals, key=lambda path: path.stat().st_mtime).parent.name


def _cli_sets_key(cli_overrides: list[str], key: str) -> bool:
    prefix = f"{key}="
    return any(arg == key or arg.startswith(prefix) for arg in cli_overrides)


def _migrate_resume_memory_prompt_defaults(cfg: Any) -> None:
    if "agent" not in cfg:
        return
    if cfg.agent.get("memory_recent_steps") == 100:
        cfg.agent.memory_recent_steps = 50
    if cfg.agent.get("memory_full_recent_steps") == 20:
        cfg.agent.memory_full_recent_steps = 10


def apply_runtime_web_options(cfg: Config, runtime: RuntimeOptions) -> None:
    if runtime.web_enabled:
        cfg.web.enabled = True
    if runtime.web_host is not None:
        cfg.web.host = runtime.web_host
    if runtime.web_port is not None:
        cfg.web.port = runtime.web_port


def _is_base_root(node: Node) -> bool:
    if node.parent is not None:
        return False
    plan = str(node.plan or "")
    return plan.startswith(BASELINE_PLAN_PREFIX) or plan.startswith(
        SEEDED_BASE_PLAN_PREFIX
    )


def _node_public_score_bonus_active(
    node: Node,
    *,
    public_scores_by_node_id: dict[str, float],
    weight: float,
    cap: float,
) -> bool:
    if node.metric is None or node.metric.value is None:
        return False
    local_score = float(node.metric.value)
    adjusted = public_adjusted_oriented_score(
        local_score=local_score,
        public_score=public_scores_by_node_id.get(node.id),
        maximize=node.metric.maximize is not False,
        weight=weight,
        cap=cap,
    )
    oriented_local = local_score if node.metric.maximize is not False else -local_score
    return adjusted > oriented_local


def _node_has_public_score(
    node: Node,
    *,
    public_scores_by_node_id: dict[str, float],
) -> bool:
    return node.id in public_scores_by_node_id


def _public_adjusted_tree_score(
    node: Node,
    *,
    public_scores_by_node_id: dict[str, float],
    weight: float,
    cap: float,
) -> float:
    assert node.metric is not None and node.metric.value is not None
    return public_adjusted_oriented_score(
        local_score=float(node.metric.value),
        public_score=public_scores_by_node_id.get(node.id),
        maximize=node.metric.maximize is not False,
        weight=weight,
        cap=cap,
    )


def _node_hypothesis_suffix(node: Node) -> str:
    hypothesis_id = hypothesis_id_for_node(node)
    if hypothesis_id is not None:
        return f"·{hypothesis_id}"
    if node.step is not None:
        return f"·{node.step}"
    return ""


def _node_runtime_suffix(node: Node) -> str:
    try:
        seconds = float(node.exec_time)
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    minutes = int(round(seconds / 60.0))
    if minutes <= 0:
        return ""
    return f"·{minutes}m"


def _is_timeout_node(node: Node) -> bool:
    if node.exc_type in {"TimeoutError", "PreprocessTimeoutError"}:
        return True
    raw_term_out = getattr(node, "_term_out", None)
    term_out = "".join(raw_term_out) if isinstance(raw_term_out, list) else ""
    text = f"{node.analysis or ''}\n{term_out}"
    return (
        "AIDE AutoGluon preprocess exceeded the dedicated timeout" in text
        or "TimeoutError: Execution exceeded the time limit" in text
    )


def _show_hypothesis_failure_in_tree(node: Node) -> bool:
    return node.status == "failed" and hypothesis_id_for_node(node) is not None


def mark_node_generated_only(node: Node) -> None:
    node.status = "generated"
    node.is_buggy = False
    node.metric = None  # type: ignore[assignment]
    node.exec_time = None  # type: ignore[assignment]
    node.exc_type = None
    node.exc_info = None
    node.exc_stack = None
    node._term_out = []
    node.analysis = "Generated only; execution skipped by --skip-execution."


def _is_seeded_scored_root_node(node: Node) -> bool:
    if node.parent is not None:
        return False
    if node.status != "ok":
        return False
    if node.exec_time != 0.0:
        return False
    if node.artifact_dir_name is not None:
        return False
    if isinstance(node.run_stats, dict) and node.run_stats.get("seeded_from_manifest"):
        return True
    plan = str(node.plan or "")
    if not plan.startswith("Seeded scored ROOT hypothesis "):
        return False
    term_out = "".join(getattr(node, "_term_out", []) or [])
    return "Seeded from code_manifest.json;" in term_out


def record_generated_only_node(
    *,
    agent: Agent,
    journal: Journal,
    node: Node,
    experiment_id: str,
) -> None:
    mark_node_generated_only(node)
    agent.save_hypothesis_root_code_for_node(node, activate=False)
    append_node_with_best_score_notification(
        journal=journal,
        node=node,
        experiment_id=experiment_id,
    )


GENERATE_ONLY_FINISHED_MESSAGE = (
    "Skip-execution mode finished generating root candidates; no code was executed."
)


def save_parallel_generate_only_run(
    *,
    cfg: Config,
    journal: Journal,
    current_node: Node | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    save_run(
        cfg,
        journal,
        current_node=current_node,
        progress_callback=progress_callback,
    )
    return GENERATE_ONLY_FINISHED_MESSAGE


def should_cleanup_workspace_on_exit(*, is_resume: bool, journal: Journal) -> bool:
    return not is_resume and len(journal) == 0


def maybe_seed_scored_hypothesis_roots(
    cfg: Config,
    journal: Journal,
    *,
    is_resume: bool,
) -> int:
    if is_resume:
        return 0
    if not getattr(cfg.research, "seed_scored_roots", False):
        return 0
    if not getattr(cfg.research, "enabled", False):
        return 0
    if getattr(cfg.research, "mode", None) != "hypothesis":
        return 0

    nodes = scored_hypothesis_root_nodes(cfg)
    for node in nodes:
        journal.append(node)
    return len(nodes)


def _artifact_context(path: Path) -> dict[str, Any]:
    context_path = path / "context.json"
    if not context_path.exists():
        return {}
    try:
        payload = json.loads(context_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_status(path: Path) -> str | None:
    status_path = path / "status.json"
    if not status_path.exists():
        return None
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def _artifact_hypothesis_id(path: Path) -> str | None:
    request_path = path / "request.md"
    if not request_path.exists():
        return None
    text = request_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"# Hypothesis under verification\b.*?Hypothesis ID:\s*(\d{6})\b",
        text,
        re.DOTALL,
    )
    if match:
        return match.group(1)
    match = re.search(r"\bHypothesis ID:\s*(\d{6})\b", text)
    return match.group(1) if match else None


def _artifact_node_ctime(path: Path, context: dict[str, Any]) -> float:
    raw = context.get("node_ctime")
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return float(raw)
    return path.stat().st_mtime


def _source_hash_by_offered_hypothesis(cfg: Config) -> dict[str, str]:
    path = Path(cfg.log_dir) / "research_hypotheses" / "offers.jsonl"
    if not path.exists():
        return {}
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        source_hash = payload.get("source_hash")
        if not isinstance(source_hash, str):
            continue
        offered = payload.get("offered")
        if not isinstance(offered, list):
            continue
        for hypothesis_id in offered:
            if isinstance(hypothesis_id, str):
                hashes.setdefault(hypothesis_id, source_hash)
    return hashes


def _root_hypothesis_ids_in_journal(journal: Journal) -> set[str]:
    ids: set[str] = set()
    for node in journal.nodes:
        if node.parent is not None:
            continue
        hypothesis_id = hypothesis_id_for_node(node)
        if hypothesis_id is not None:
            ids.add(hypothesis_id)
    return ids


def _recoverable_generated_root_artifacts(
    cfg: Config,
    journal: Journal,
) -> list[tuple[Path, str, float]]:
    artifacts_dir = Path(cfg.log_dir) / "artifacts"
    if not artifacts_dir.exists():
        return []
    existing_ids = _root_hypothesis_ids_in_journal(journal)
    seen_ids = set(existing_ids)
    candidates: list[tuple[float, str, Path, str]] = []
    for path in sorted(item for item in artifacts_dir.iterdir() if item.is_dir()):
        context = _artifact_context(path)
        if context.get("phase") != "generate":
            continue
        if context.get("parent_node_id") is not None:
            continue
        if context.get("agent_mode") != cfg.agent.mode:
            continue
        if _artifact_status(path) != "completed":
            continue
        if not (path / "response.py").exists():
            continue
        hypothesis_id = _artifact_hypothesis_id(path)
        if hypothesis_id is None or hypothesis_id in seen_ids:
            continue
        node_ctime = _artifact_node_ctime(path, context)
        candidates.append((node_ctime, path.name, path, hypothesis_id))
        seen_ids.add(hypothesis_id)
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [(path, hypothesis_id, node_ctime) for node_ctime, _name, path, hypothesis_id in candidates]


def recover_generated_only_root_artifacts(
    *,
    cfg: Config,
    journal: Journal,
    agent: Agent,
) -> int:
    if not cfg.research.enabled or getattr(cfg.research, "mode", "llm") != "hypothesis":
        return 0
    source_hashes = _source_hash_by_offered_hypothesis(cfg)
    recovered = 0
    for artifact_dir, hypothesis_id, node_ctime in _recoverable_generated_root_artifacts(
        cfg,
        journal,
    ):
        code = (artifact_dir / "response.py").read_text(
            encoding="utf-8",
            errors="replace",
        )
        raw_path = artifact_dir / "response_raw.txt"
        raw_response = (
            raw_path.read_text(encoding="utf-8", errors="replace")
            if raw_path.exists()
            else ""
        )
        plan = extract_text_up_to_code(raw_response).strip()
        if not plan:
            plan = (
                "Recovered generated-only root code for hypothesis "
                f"{hypothesis_id} from {artifact_dir.name}."
            )
        node = Node(
            code=code,
            plan=plan,
            ctime=node_ctime,
            artifact_dir_name=artifact_dir.name,
        )
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        node.research_source_hash = source_hashes.get(hypothesis_id)
        record_manual_prompt_node(cfg, node)
        record_generated_only_node(
            agent=agent,
            journal=journal,
            node=node,
            experiment_id=cfg.exp_name,
        )
        recovered += 1
    return recovered


def _hypothesis_root_for_node(node: Node) -> Node:
    current = node
    while current.parent is not None:
        current = current.parent
    return current


def _matches_forced_hypothesis_root(node: Node, forced_root: str | None) -> bool:
    if forced_root is None:
        return True
    return hypothesis_id_for_node(_hypothesis_root_for_node(node)) == forced_root


def _matches_forced_hypothesis(node: Node, forced_hypothesis: str | None) -> bool:
    if forced_hypothesis is None:
        return True
    return hypothesis_id_for_node(node) == forced_hypothesis


def next_generated_only_node(
    journal: Journal,
    *,
    forced_root: str | None = None,
    forced_hypothesis: str | None = None,
) -> Node | None:
    for node in journal.nodes:
        if node.status == "generated" and _matches_forced_hypothesis_root(
            node,
            forced_root,
        ) and _matches_forced_hypothesis(node, forced_hypothesis):
            return node
    return None


def _rebase_resume_repo_path(value: Any, *, repo_root: Path) -> Path | Any:
    if value is None:
        return value
    path = Path(value)
    if not path.is_absolute():
        return path

    repo_markers = (
        ("aide", "example_tasks"),
        ("logs",),
        ("workspaces",),
        ("research_hypotheses",),
        ("assets",),
        ("scripts",),
        ("reports",),
    )
    parts = path.parts
    for marker in repo_markers:
        marker_len = len(marker)
        for index in range(0, len(parts) - marker_len + 1):
            if parts[index : index + marker_len] != marker:
                continue
            rebased = repo_root / Path(*parts[index:])
            if rebased.exists():
                return rebased
    return path


def _rebase_resume_config_paths(cfg: Any, *, repo_root: Path) -> None:
    if "data_dir" in cfg:
        cfg.data_dir = _rebase_resume_repo_path(cfg.data_dir, repo_root=repo_root)
    if "desc_file" in cfg and cfg.desc_file is not None:
        cfg.desc_file = _rebase_resume_repo_path(cfg.desc_file, repo_root=repo_root)


def load_resume_state(
    *,
    run_id: str,
    top_log_dir: Path,
    top_workspace_dir: Path,
    cli_overrides: list[str],
    force_check_submissions: bool = False,
) -> tuple[Config, Journal]:
    log_dir = (top_log_dir / run_id).resolve()
    workspace_dir = (top_workspace_dir / run_id).resolve()
    config_path = log_dir / "config.yaml"
    journal_path = log_dir / "journal.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing resume config: {config_path}")
    if not journal_path.exists():
        raise FileNotFoundError(f"Missing resume journal: {journal_path}")
    if not workspace_dir.exists():
        raise FileNotFoundError(f"Missing resume workspace: {workspace_dir}")
    if (
        not (workspace_dir / "input").exists()
        or not (workspace_dir / "working").exists()
    ):
        raise FileNotFoundError(
            f"Resume workspace must contain input/ and working/: {workspace_dir}"
        )

    _validate_cli_model_effort_conflicts(cli_overrides)
    cli_overrides = _normalize_model_effort_cli_overrides(cli_overrides)
    cli_overrides = _normalize_forced_root_cli_overrides(cli_overrides)
    forced_root_overridden = _cli_sets_key(cli_overrides, "agent.search.forced_root")
    cfg = OmegaConf.load(config_path)
    _migrate_resume_memory_prompt_defaults(cfg)
    if (
        not forced_root_overridden
        and "agent" in cfg
        and "search" in cfg.agent
        and "forced_root" in cfg.agent.search
    ):
        cfg.agent.search.forced_root = None
    if (
        _cli_sets_key(cli_overrides, "agent.autogluon.profile")
        and not _cli_sets_key(cli_overrides, "agent.autogluon.included_model_types")
        and "agent" in cfg
        and "autogluon" in cfg.agent
    ):
        cfg.agent.autogluon.included_model_types = None
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    _rebase_resume_config_paths(cfg, repo_root=top_log_dir.resolve().parent)
    cfg.exp_name = run_id
    cfg.log_dir = log_dir
    cfg.workspace_dir = workspace_dir
    cfg_schema: Config = OmegaConf.structured(Config)
    _drop_deprecated_config_keys(cfg)
    cfg = OmegaConf.merge(cfg_schema, cfg)
    _normalize_agent_mode_aliases(cfg)
    _resolve_all_model_configs(cfg)
    if aux_mode(cfg) == "file":
        copy_aux_file_input(cfg)
    journal = serialize.load_json(journal_path, Journal)
    if enforce_journal_submission_contract(
        cfg,
        journal,
        force_check_submissions=force_check_submissions,
    ):
        serialize.dump_json(journal, journal_path)
    return cast(Config, cfg), journal


def confirm_resume_latest(run_id: str, completed_steps: int, total_steps: int) -> bool:
    print(f"Resume latest run: {run_id}")
    print(f"Completed steps: {completed_steps}/{total_steps}")
    if not sys.stdin.isatty():
        print("Continue? [y/N] N")
        return False
    answer = input("Continue? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def journal_to_rich_tree(
    journal: Journal,
    *,
    active_parent_node: Node | None = None,
    active_stage: str | None = None,
    active_hypothesis_id: str | None = None,
    blink_on: bool = True,
    show_invalid_submission_branches: bool = False,
    disable_oom_saturated_parents: bool = False,
    plateau_block_epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
    synthesis_node_ids: set[str] | None = None,
):
    if show_invalid_submission_branches:
        best_node = journal.get_best_node()
    else:
        visible_good_nodes = [
            node
            for node in journal.good_nodes
            if not node.is_in_submission_contract_error_branch
            and not is_plateau_blocked_descendant(
                node,
                epsilon=plateau_block_epsilon,
            )
            and (
                not disable_oom_saturated_parents
                or not node.is_oom_blocked_parent
            )
        ]
        best_node = max(visible_good_nodes, key=lambda node: node.metric, default=None)
    journal_nodes = set(journal.nodes)

    def active_placeholder_style() -> str:
        if active_stage == "generating":
            return "bold white"
        if active_stage == "executing":
            return "bold yellow"
        if active_stage == "reviewing":
            return "bold blue"
        return "bold yellow"

    def append_active_placeholder(tree):
        if active_stage is None:
            return
        indicator = "[*]" if blink_on else "[ ]"
        placeholder = Text(indicator, style=active_placeholder_style())
        if active_hypothesis_id:
            placeholder.append(f"·{active_hypothesis_id}", style=active_placeholder_style())
        tree.add(placeholder)

    def node_order_key(node: Node):
        return (
            node.step is None,
            node.step if node.step is not None else len(journal.nodes),
            node.ctime,
            node.id,
        )

    def is_synthesis_root(node: Node) -> bool:
        if synthesis_node_ids and node.id in synthesis_node_ids:
            return True
        return node.parent is None and str(node.plan or "").startswith(
            SYNTHESIS_PLAN_PREFIX
        )

    def append_rec(node: Node, tree):
        if node.is_terminal_failure and not _show_hypothesis_failure_in_tree(node):
            return
        if (
            node.is_submission_contract_error
            and not show_invalid_submission_branches
        ):
            return

        synthesis_root = is_synthesis_root(node)
        suffix = _node_hypothesis_suffix(node)
        runtime_suffix = _node_runtime_suffix(node)
        if node.status == "generated":
            s = f"[cyan]● generated{suffix}[/cyan]"
        elif node.status == "failed" and suffix:
            s = f"[red]failed{suffix}{runtime_suffix}[/red]"
        elif synthesis_root and (
            node.is_buggy or node.metric is None or node.metric.value is None
        ):
            s = f"[bold blue]◆[/bold blue] [red]bug{suffix}{runtime_suffix}[/red]"
        elif _is_timeout_node(node):
            s = f"[red]● timeout{suffix}{runtime_suffix}"
        elif node.is_buggy or node.metric is None or node.metric.value is None:
            s = f"[red]● bug{suffix}{runtime_suffix}"
        else:
            metric_text = f"{node.metric.value:.5f}"

            if is_plateau_blocked_descendant(
                node,
                epsilon=plateau_block_epsilon,
            ):
                s = f"[bright_black]◉ {metric_text}{suffix}{runtime_suffix}"
            elif disable_oom_saturated_parents and node.is_oom_blocked_parent:
                s = f"[bright_black]✕ {metric_text}{suffix}{runtime_suffix}"
            else:
                metric_style = "bold yellow" if node is best_node else "green"
                s = f"[green]●[/green] [{metric_style}]{metric_text}{suffix}{runtime_suffix}"

        subtree = tree.add(s)
        for child in sorted(
            (child for child in node.children if child in journal_nodes),
            key=node_order_key,
        ):
            append_rec(child, subtree)
        if node is active_parent_node:
            append_active_placeholder(subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in list(journal.draft_nodes):
        append_rec(n, tree)
    if active_parent_node is None:
        append_active_placeholder(tree)
    return tree


def _visible_best_node(
    journal: Journal,
    *,
    show_invalid_submission_branches: bool,
    disable_oom_saturated_parents: bool = False,
    plateau_block_epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
    score_fn: Callable[[Node], float] | None = None,
) -> Node | None:
    if show_invalid_submission_branches:
        return journal.get_best_node()

    visible_good_nodes = [
        node
        for node in journal.good_nodes
        if node.metric is not None
        and node.metric.value is not None
        and not node.is_in_submission_contract_error_branch
        and not is_plateau_blocked_descendant(
            node,
            epsilon=plateau_block_epsilon,
        )
        and (
            not disable_oom_saturated_parents
            or not node.is_oom_blocked_parent
        )
    ]
    if score_fn is not None:
        return max(visible_good_nodes, key=score_fn, default=None)
    return max(visible_good_nodes, key=lambda node: node.metric, default=None)


def _tree_node_label(
    node: Node,
    *,
    best_node: Node | None,
    public_best_node: Node | None = None,
    disable_oom_saturated_parents: bool = False,
    plateau_block_epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
    synthesis_node_ids: set[str] | None = None,
    public_node_ids: set[str] | None = None,
    public_bonus_node_ids: set[str] | None = None,
) -> Text:
    suffix = _node_hypothesis_suffix(node)
    runtime_suffix = _node_runtime_suffix(node)
    if node.status == "generated":
        return Text(f"● generated{suffix}", style="cyan")
    if node.is_terminal_failure:
        return Text(f"● failed{suffix}{runtime_suffix}", style="red")

    synthesis_root = bool(synthesis_node_ids and node.id in synthesis_node_ids) or (
        node.parent is None and str(node.plan or "").startswith(SYNTHESIS_PLAN_PREFIX)
    )
    if synthesis_root and (
        node.is_buggy or node.metric is None or node.metric.value is None
    ):
        label = Text()
        label.append("◆", style="bold blue")
        label.append(f" bug{suffix}{runtime_suffix}", style="red")
        return label
    if _is_timeout_node(node):
        return Text(f"● timeout{suffix}{runtime_suffix}", style="red")
    if node.is_buggy or node.metric is None or node.metric.value is None:
        return Text(f"● bug{suffix}{runtime_suffix}", style="red")

    if is_plateau_blocked_descendant(node, epsilon=plateau_block_epsilon):
        return Text(
            f"◉ {node.metric.value:.5f}{suffix}{runtime_suffix}",
            style="bright_black",
        )

    if disable_oom_saturated_parents and node.is_oom_blocked_parent:
        return Text(
            f"✕ {node.metric.value:.5f}{suffix}{runtime_suffix}",
            style="bright_black",
        )

    label = Text()
    is_public = bool(public_node_ids and node.id in public_node_ids)
    is_public_bonus = bool(public_bonus_node_ids and node.id in public_bonus_node_ids)
    if is_public:
        if node is public_best_node:
            diamond_style = "bold yellow"
        elif is_public_bonus:
            diamond_style = "bright_blue"
        else:
            diamond_style = "bright_black"
        label.append("◆ ", style=diamond_style)
    else:
        label.append("● ", style="green")

    metric_style = "bold yellow" if node is best_node else "green"
    metric_text = f"{node.metric.value:.5f}{suffix}{runtime_suffix}"
    label.append(metric_text, style=metric_style)
    return label


def _tree_active_node_label(
    node: Node,
    *,
    active_stage: str | None,
    blink_on: bool,
    best_node: Node | None,
    public_best_node: Node | None = None,
    disable_oom_saturated_parents: bool = False,
    plateau_block_epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
    synthesis_node_ids: set[str] | None = None,
    public_node_ids: set[str] | None = None,
    public_bonus_node_ids: set[str] | None = None,
) -> Text:
    line = _tree_active_placeholder_line(
        active_stage=active_stage,
        active_hypothesis_id=None,
        blink_on=blink_on,
    )
    line.append(" ")
    label = _tree_node_label(
        node,
        best_node=best_node,
        public_best_node=public_best_node,
        disable_oom_saturated_parents=disable_oom_saturated_parents,
        plateau_block_epsilon=plateau_block_epsilon,
        synthesis_node_ids=synthesis_node_ids,
        public_node_ids=public_node_ids,
        public_bonus_node_ids=public_bonus_node_ids,
    )
    if node.status == "generated":
        suffix = _node_hypothesis_suffix(node)
        line.append(f"generated{suffix}", style="cyan")
    else:
        line.append_text(label)
    return line


def _tree_active_placeholder_line(
    *,
    active_stage: str | None,
    active_hypothesis_id: str | None = None,
    blink_on: bool,
) -> Text:
    indicator = "[*]" if blink_on else "[ ]"
    style = "bold yellow"
    if active_stage == "generating":
        style = "bold white"
    elif active_stage == "executing":
        style = "bold yellow"
    elif active_stage == "reviewing":
        style = "bold blue"
    line = Text(indicator, style=style)
    if active_hypothesis_id:
        line.append(f"·{active_hypothesis_id}", style=style)
    return line


def synthesis_injected_node_ids(log_dir: Path | str) -> set[str]:
    synthesis_dir = Path(log_dir) / "synthesis"
    if not synthesis_dir.exists():
        return set()

    node_ids: set[str] = set()
    for status_path in synthesis_dir.glob("checkpoint-*/status.json"):
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        node_id = status.get("injected_node_id")
        if isinstance(node_id, str) and node_id:
            node_ids.add(node_id)
    return node_ids


def build_tree_view(
    journal: Journal,
    *,
    active_node: Node | None = None,
    active_parent_node: Node | None = None,
    active_stage: str | None = None,
    active_hypothesis_id: str | None = None,
    active_root_generations: list[ActiveRootGeneration] | None = None,
    blink_on: bool = True,
    show_invalid_submission_branches: bool = False,
    disable_oom_saturated_parents: bool = False,
    plateau_block_epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
    synthesis_node_ids: set[str] | None = None,
    public_scores_by_node_id: dict[str, float] | None = None,
    public_score_bonus_weight: float = 0.0,
    public_score_bonus_cap: float = 0.0,
) -> TreeView:
    items: list[TreeViewItem] = [
        TreeViewItem(
            "header",
            None,
            Text("Solution tree", style="bold blue"),
            focus_start=0,
        )
    ]
    children_by_id: dict[str, list[str]] = {"header": []}
    parent_by_id: dict[str, str | None] = {"header": None}
    journal_nodes = set(journal.nodes)
    active_existing_node = active_node if active_node in journal_nodes else None
    active_root_generations = active_root_generations or []
    latest_active_root = (
        max(active_root_generations, key=lambda item: item.launched_index)
        if active_root_generations
        else None
    )
    active_item_id: str | None = None
    public_scores_by_node_id = public_scores_by_node_id or {}
    public_node_ids = {
        node.id
        for node in journal.good_nodes
        if _node_has_public_score(
            node,
            public_scores_by_node_id=public_scores_by_node_id,
        )
    }
    public_bonus_node_ids = {
        node.id
        for node in journal.good_nodes
        if _node_public_score_bonus_active(
            node,
            public_scores_by_node_id=public_scores_by_node_id,
            weight=public_score_bonus_weight,
            cap=public_score_bonus_cap,
        )
    }
    score_fn: Callable[[Node], float] | None = None
    if public_scores_by_node_id and public_score_bonus_weight > 0.0:
        def public_adjusted_score(node: Node) -> float:
            return _public_adjusted_tree_score(
                node,
                public_scores_by_node_id=public_scores_by_node_id,
                weight=public_score_bonus_weight,
                cap=public_score_bonus_cap,
            )

        score_fn = public_adjusted_score
    best_node = _visible_best_node(
        journal,
        show_invalid_submission_branches=show_invalid_submission_branches,
        disable_oom_saturated_parents=disable_oom_saturated_parents,
        plateau_block_epsilon=plateau_block_epsilon,
    )
    public_best_node = _visible_best_node(
        journal,
        show_invalid_submission_branches=show_invalid_submission_branches,
        disable_oom_saturated_parents=disable_oom_saturated_parents,
        plateau_block_epsilon=plateau_block_epsilon,
        score_fn=score_fn,
    ) if score_fn is not None else None
    if public_best_node is not None and public_best_node.id not in public_bonus_node_ids:
        public_best_node = None

    def node_order_key(node: Node):
        return (
            node.step is None,
            node.step if node.step is not None else len(journal.nodes),
            node.ctime,
            node.id,
        )

    def append_item(item: TreeViewItem) -> None:
        items.append(item)
        parent_by_id[item.item_id] = item.parent_id
        children_by_id.setdefault(item.item_id, [])
        if item.parent_id is not None:
            children_by_id.setdefault(item.parent_id, []).append(item.item_id)

    def append_active(parent_id: str, ancestor_has_next: list[bool]) -> None:
        nonlocal active_item_id
        if active_stage is None:
            return
        if active_root_generations and parent_id == "header":
            return
        if active_existing_node is not None:
            return
        prefix = "".join(
            "│   " if has_next else "    "
            for has_next in ancestor_has_next
        )
        prefix += "└── "
        line = Text(prefix)
        line.append_text(
            _tree_active_placeholder_line(
                active_stage=active_stage,
                active_hypothesis_id=active_hypothesis_id,
                blink_on=blink_on,
            )
        )
        append_item(TreeViewItem("active", parent_id, line, focus_start=len(prefix)))
        active_item_id = "active"

    def append_active_root_generations(parent_id: str) -> None:
        nonlocal active_item_id
        for index, generation in enumerate(active_root_generations):
            is_latest = latest_active_root is not None and (
                generation.hypothesis_id == latest_active_root.hypothesis_id
            )
            item_id = f"active:{generation.hypothesis_id}"
            prefix = (
                "└── "
                if index == len(active_root_generations) - 1
                else "├── "
            )
            line = Text(prefix)
            line.append_text(
                _tree_active_placeholder_line(
                    active_stage=active_stage,
                    active_hypothesis_id=generation.hypothesis_id,
                    blink_on=blink_on,
                )
            )
            append_item(TreeViewItem(item_id, parent_id, line, focus_start=len(prefix)))
            if is_latest:
                active_item_id = item_id

    def append_rec(
        node: Node,
        parent_id: str,
        ancestor_has_next: list[bool],
        is_last: bool,
    ) -> None:
        if node.is_terminal_failure and not _show_hypothesis_failure_in_tree(node):
            return
        if node.is_submission_contract_error and not show_invalid_submission_branches:
            return

        prefix = "".join(
            "│   " if has_next else "    "
            for has_next in ancestor_has_next
        )
        prefix += "└── " if is_last else "├── "
        line = Text(prefix)
        if node is active_existing_node and active_stage is not None:
            nonlocal active_item_id
            active_item_id = node.id
            line.append_text(
                _tree_active_node_label(
                    node,
                    active_stage=active_stage,
                    blink_on=blink_on,
                    best_node=best_node,
                    public_best_node=public_best_node,
                    disable_oom_saturated_parents=disable_oom_saturated_parents,
                    plateau_block_epsilon=plateau_block_epsilon,
                    synthesis_node_ids=synthesis_node_ids,
                    public_node_ids=public_node_ids,
                    public_bonus_node_ids=public_bonus_node_ids,
                )
            )
        else:
            line.append_text(
                _tree_node_label(
                    node,
                    best_node=best_node,
                    public_best_node=public_best_node,
                    disable_oom_saturated_parents=disable_oom_saturated_parents,
                    plateau_block_epsilon=plateau_block_epsilon,
                    synthesis_node_ids=synthesis_node_ids,
                    public_node_ids=public_node_ids,
                    public_bonus_node_ids=public_bonus_node_ids,
                )
            )
        append_item(
            TreeViewItem(
                node.id,
                parent_id,
                line,
                node=node,
                focus_start=len(prefix),
            )
        )

        children = sorted(
            (child for child in node.children if child in journal_nodes),
            key=node_order_key,
        )
        visible_children = [
            child
            for child in children
            if (
                not child.is_terminal_failure
                or _show_hypothesis_failure_in_tree(child)
            )
            and (
                show_invalid_submission_branches
                or not child.is_submission_contract_error
            )
        ]
        next_ancestors = [*ancestor_has_next, not is_last]
        has_active_child = node is active_parent_node and active_existing_node is None
        for index, child in enumerate(visible_children):
            append_rec(
                child,
                node.id,
                next_ancestors,
                index == len(visible_children) - 1 and not has_active_child,
            )
        if node is active_parent_node:
            append_active(node.id, next_ancestors)

    roots = [
        node
        for node in journal.draft_nodes
        if (
            not node.is_terminal_failure
            or _show_hypothesis_failure_in_tree(node)
        )
        and (
            show_invalid_submission_branches
            or not node.is_submission_contract_error
        )
    ]
    has_root_active = (
        active_parent_node is None
        and active_stage is not None
        and active_existing_node is None
        and not active_root_generations
    )
    for index, node in enumerate(roots):
        has_active_after_roots = has_root_active or bool(active_root_generations)
        append_rec(
            node,
            "header",
            [],
            index == len(roots) - 1 and not has_active_after_roots,
        )
    if active_parent_node is None and active_root_generations:
        append_active_root_generations("header")
    elif active_parent_node is None:
        append_active("header", [])

    return TreeView(
        items=items,
        index_by_id={item.item_id: index for index, item in enumerate(items)},
        parent_by_id=parent_by_id,
        children_by_id=children_by_id,
        active_item_id=active_item_id,
    )


def move_tree_focus(view: TreeView, focused_item_id: str, direction: str) -> str:
    if focused_item_id not in view.index_by_id:
        focused_item_id = "header"
    if direction == "left":
        return view.parent_by_id.get(focused_item_id) or focused_item_id
    if direction == "right":
        children = view.children_by_id.get(focused_item_id) or []
        return children[0] if children else focused_item_id
    if direction in {"up", "down"}:
        if focused_item_id == "header":
            children = view.children_by_id.get("header") or []
            return children[0] if direction == "down" and children else "header"
        parent_id = view.parent_by_id.get(focused_item_id)
        siblings = view.children_by_id.get(parent_id or "header") or [focused_item_id]
        sibling_index = siblings.index(focused_item_id)
        if direction == "up":
            if sibling_index == 0:
                return parent_id or "header"
            return siblings[sibling_index - 1]
        return siblings[min(len(siblings) - 1, sibling_index + 1)]
    return focused_item_id


def clamp_tree_viewport(
    *,
    total_lines: int,
    viewport_height: int,
    focus_index: int,
    current_scroll: int,
) -> int:
    viewport_height = max(1, viewport_height)
    max_scroll = max(0, total_lines - viewport_height)
    scroll = min(max(0, current_scroll), max_scroll)
    if focus_index < scroll:
        scroll = focus_index
    elif focus_index >= scroll + viewport_height:
        scroll = focus_index - viewport_height + 1
    return min(max(0, scroll), max_scroll)


def center_tree_viewport(
    *,
    total_lines: int,
    viewport_height: int,
    focus_index: int,
) -> int:
    viewport_height = max(1, viewport_height)
    max_scroll = max(0, total_lines - viewport_height)
    scroll = focus_index - viewport_height // 2
    return min(max(0, scroll), max_scroll)


def best_tree_item_id(
    view: TreeView,
    journal: Journal,
    *,
    show_invalid_submission_branches: bool,
    disable_oom_saturated_parents: bool = False,
) -> str | None:
    best_node = _visible_best_node(
        journal,
        show_invalid_submission_branches=show_invalid_submission_branches,
        disable_oom_saturated_parents=disable_oom_saturated_parents,
    )
    if best_node is None or best_node.id not in view.index_by_id:
        return None
    return best_node.id


def active_tree_item_id(view: TreeView) -> str | None:
    if view.active_item_id is not None and view.active_item_id in view.index_by_id:
        return view.active_item_id
    return "active" if "active" in view.index_by_id else None


def recover_tree_focus_by_index(
    view: TreeView,
    *,
    fallback_index: int,
) -> str:
    if not view.items:
        return "header"
    index = min(max(0, fallback_index), len(view.items) - 1)
    return view.items[index].item_id


def _is_scored_hypothesis_node(node: Node) -> bool:
    return (
        not node.is_buggy
        and not node.is_terminal_failure
        and node.metric is not None
        and node.metric.value is not None
        and hypothesis_id_for_node(node) is not None
    )


def _metric_sort_key(node: Node) -> float:
    assert node.metric is not None
    assert node.metric.value is not None
    value = float(node.metric.value)
    return value if node.metric.maximize is not False else -value


def _score_text(node: Node) -> str:
    if node.metric is None or node.metric.value is None:
        return "n/a"
    return f"{node.metric.value:.5f}"


def _plain_line_view(title: str, lines: list[Text]) -> TreeView:
    items = [TreeViewItem("header", None, Text(title, style="bold blue"), focus_start=0)]
    children_by_id: dict[str, list[str]] = {"header": []}
    parent_by_id: dict[str, str | None] = {"header": None}
    for index, line in enumerate(lines):
        item_id = f"row:{index}"
        items.append(TreeViewItem(item_id, "header", line, focus_start=0))
        children_by_id["header"].append(item_id)
        children_by_id[item_id] = []
        parent_by_id[item_id] = "header"
    return TreeView(
        items=items,
        index_by_id={item.item_id: index for index, item in enumerate(items)},
        parent_by_id=parent_by_id,
        children_by_id=children_by_id,
    )


def _hypothesis_mode_label(agent_modes: list[str]) -> str:
    labels = []
    if "autogluon" in agent_modes:
        labels.append("ag")
    if "legacy" in agent_modes:
        labels.append("leg")
    return ",".join(labels) if labels else "n/a"


def _hypothesis_mode_labels(cfg: Config) -> dict[str, str]:
    try:
        library = load_manual_hypothesis_library(cfg)
    except (OSError, ValueError):
        return {}
    return {
        hypothesis.id: _hypothesis_mode_label(hypothesis.agent_modes)
        for hypothesis in library.hypotheses
    }


def build_root_hypotheses_view(
    journal: Journal,
    *,
    hypothesis_mode_labels: dict[str, str] | None = None,
) -> TreeView:
    roots = [
        node
        for node in journal.nodes
        if node.parent is None
        and not _is_base_root(node)
        and _is_scored_hypothesis_node(node)
    ]
    roots.sort(key=_metric_sort_key, reverse=True)
    lines = [Text("#    score    hypothesis  time         mode", style=TUI_ROW_LABEL_STYLE)]
    for node in roots:
        hypothesis_id = hypothesis_id_for_node(node) or "n/a"
        step = node.step if node.step is not None else "?"
        step_text = f"{step:03d}" if isinstance(step, int) else str(step).rjust(3)
        time_text = dt.datetime.fromtimestamp(node.ctime).strftime("%m-%d %H:%M")
        mode_text = (
            hypothesis_mode_labels.get(hypothesis_id, "n/a")
            if hypothesis_mode_labels is not None
            else "n/a"
        )
        line = Text()
        line.append(f"{step_text}  ", style=TUI_INACTIVE_VALUE_STYLE)
        line.append(f"{_score_text(node):<9}", style=TUI_METRIC_VALUE_STYLE)
        line.append(f"{hypothesis_id:<12}", style=TUI_NEUTRAL_VALUE_STYLE)
        line.append(f"{time_text}  ", style=TUI_INACTIVE_VALUE_STYLE)
        line.append(mode_text, style=TUI_INACTIVE_VALUE_STYLE)
        lines.append(line)
    if not roots:
        lines.append(Text("n/a", style=TUI_INACTIVE_VALUE_STYLE))
    return _plain_line_view("Root hypotheses", lines)


def build_all_hypotheses_view(journal: Journal) -> TreeView:
    rows: dict[str, dict[str, object]] = {}
    for node in journal.nodes:
        hypothesis_id = hypothesis_id_for_node(node)
        if hypothesis_id is None or not _is_scored_hypothesis_node(node):
            continue
        row = rows.setdefault(
            hypothesis_id,
            {
                "hypothesis_id": hypothesis_id,
                "best_node": node,
                "uses_total": 0,
                "root_uses": 0,
                "branch_uses": 0,
            },
        )
        row["uses_total"] = int(row["uses_total"]) + 1
        if node.parent is None:
            row["root_uses"] = int(row["root_uses"]) + 1
        else:
            row["branch_uses"] = int(row["branch_uses"]) + 1
        best_node = row["best_node"]
        assert isinstance(best_node, Node)
        if _metric_sort_key(node) > _metric_sort_key(best_node):
            row["best_node"] = node

    sorted_rows = sorted(
        rows.values(),
        key=lambda row: _metric_sort_key(cast(Node, row["best_node"])),
        reverse=True,
    )
    lines = [Text("best      hypothesis  uses", style=TUI_ROW_LABEL_STYLE)]
    for row in sorted_rows:
        best_node = cast(Node, row["best_node"])
        line = Text()
        line.append(f"{_score_text(best_node):<10}", style=TUI_METRIC_VALUE_STYLE)
        line.append(f"{row['hypothesis_id']:<12}", style=TUI_NEUTRAL_VALUE_STYLE)
        line.append(
            f"{row['uses_total']} (root {row['root_uses']}, branch {row['branch_uses']})",
            style=TUI_INACTIVE_VALUE_STYLE,
        )
        lines.append(line)
    if not sorted_rows:
        lines.append(Text("n/a", style=TUI_INACTIVE_VALUE_STYLE))
    return _plain_line_view("All hypotheses", lines)


def build_best_branch_view(journal: Journal) -> TreeView:
    best_node = _best_scored_node(journal)
    if best_node is None:
        return _plain_line_view(
            "Best branch",
            [Text("n/a", style=TUI_INACTIVE_VALUE_STYLE)],
        )

    path: list[Node] = []
    node: Node | None = best_node
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    path = [path_node for path_node in path if _is_scored_hypothesis_node(path_node)]

    line = Text()
    for index, path_node in enumerate(path):
        if index:
            line.append(" -> ", style=TUI_SEPARATOR_STYLE)
        line.append(
            hypothesis_id_for_node(path_node) or "n/a",
            style=TUI_NEUTRAL_VALUE_STYLE,
        )
        line.append(f" {_score_text(path_node)}", style=TUI_METRIC_VALUE_STYLE)
    return _plain_line_view("Best branch", [line])


def _decision_metric_text(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "n/a"
    value = payload.get("metric")
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value):.5f}"


def _decision_node_label(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "n/a"
    hypothesis_id = payload.get("hypothesis_id") or "n/a"
    return f"{_decision_metric_text(payload)}*{hypothesis_id}"


def _append_decision_payload_line(
    lines: list[Text],
    label: str,
    payload: dict[str, Any] | None,
) -> None:
    line = Text(f"{label:<16}", style=TUI_ROW_LABEL_STYLE)
    line.append(_decision_node_label(payload), style=TUI_METRIC_VALUE_STYLE)
    if payload:
        step = payload.get("step")
        if step is not None:
            line.append(f" step={step}", style=TUI_INACTIVE_VALUE_STYLE)
        child_count = payload.get("child_count")
        if child_count is not None:
            line.append(f" children={child_count}", style=TUI_INACTIVE_VALUE_STYLE)
    lines.append(line)


def _decision_number(value: Any, digits: int = 5) -> str:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _append_policy_debug_lines(
    lines: list[Text],
    diagnostics: dict[str, Any],
) -> None:
    lines.append(Text(""))
    lines.append(Text("POLICY SCORE", style="bold blue"))

    candidate_count = diagnostics.get("candidate_count", "n/a")
    exploration_weight = _decision_number(diagnostics.get("exploration_weight"), 3)
    line = Text("policy config   ", style=TUI_ROW_LABEL_STYLE)
    line.append(f"candidates={candidate_count}", style=TUI_INACTIVE_VALUE_STYLE)
    line.append(f" exploration_weight={exploration_weight}", style=TUI_INACTIVE_VALUE_STYLE)
    lines.append(line)

    line = Text("metric range    ", style=TUI_ROW_LABEL_STYLE)
    line.append(
        (
            f"{_decision_number(diagnostics.get('metric_min'))}.."
            f"{_decision_number(diagnostics.get('metric_max'))}"
        ),
        style=TUI_INACTIVE_VALUE_STYLE,
    )
    line.append(
        f" span={_decision_number(diagnostics.get('metric_span'))}",
        style=TUI_INACTIVE_VALUE_STYLE,
    )
    lines.append(line)

    for label, key in [("selected", "selected"), ("best filtered", "best")]:
        payload = cast(dict[str, Any] | None, diagnostics.get(key))
        if not payload:
            continue
        line = Text(f"{label:<16}", style=TUI_ROW_LABEL_STYLE)
        line.append(
            f"policy={_decision_number(payload.get('policy_score'))}",
            style=TUI_NEUTRAL_VALUE_STYLE,
        )
        line.append(
            f" norm={_decision_number(payload.get('normalized_metric'))}",
            style=TUI_INACTIVE_VALUE_STYLE,
        )
        line.append(
            f" bonus={_decision_number(payload.get('exploration_bonus'))}",
            style=TUI_INACTIVE_VALUE_STYLE,
        )
        lines.append(line)

    override = cast(dict[str, Any] | None, diagnostics.get("selection_override"))
    if override:
        line = Text("override        ", style=TUI_ROW_LABEL_STYLE)
        line.append(
            (
                "best children "
                f"{override.get('best_child_count', '?')}/"
                f"{override.get('min_children', '?')} before exploration"
            ),
            style=TUI_OPERATOR_NOTICE_STYLE,
        )
        lines.append(line)

    line = Text("selected-best   ", style=TUI_ROW_LABEL_STYLE)
    line.append(
        f"policy={_decision_number(diagnostics.get('selected_minus_best_policy_score'))}",
        style=TUI_NEUTRAL_VALUE_STYLE,
    )
    line.append(
        f" metric={_decision_number(diagnostics.get('selected_minus_best_metric'))}",
        style=TUI_INACTIVE_VALUE_STYLE,
    )
    lines.append(line)

    threshold = cast(
        dict[str, Any] | None,
        diagnostics.get("fresh_child_metric_threshold"),
    )
    if threshold:
        child_count = threshold.get("child_count", 0)
        direction = threshold.get("direction", ">=")
        metric = _decision_number(threshold.get("metric"))
        line = Text("threshold       ", style=TUI_ROW_LABEL_STYLE)
        line.append(
            f"children={child_count} beats best if score {direction} {metric}",
            style=TUI_OPERATOR_NOTICE_STYLE,
        )
        lines.append(line)


def build_search_decision_debug_view(decision: dict[str, Any] | None) -> Group:
    if not decision:
        return Group(
            Text("SEARCH DECISION", style="bold blue"),
            Text("No search decision recorded yet.", style=TUI_INACTIVE_VALUE_STYLE),
        )

    lines: list[Text] = [
        Text(
            (
                f"SEARCH DECISION step={decision.get('step', 'n/a')} "
                f"mode={decision.get('mode', 'n/a')}"
            ),
            style="bold blue",
        )
    ]
    reason = decision.get("reason")
    if reason:
        line = Text("policy reason  ", style=TUI_ROW_LABEL_STYLE)
        line.append(str(reason), style=TUI_NEUTRAL_VALUE_STYLE)
        lines.append(line)
    forced_root = decision.get("forced_hypothesis_root")
    if forced_root:
        line = Text("forced_root    ", style=TUI_ROW_LABEL_STYLE)
        line.append(str(forced_root), style=TUI_NEUTRAL_VALUE_STYLE)
        lines.append(line)

    _append_decision_payload_line(
        lines,
        "SELECTED",
        cast(dict[str, Any] | None, decision.get("selected")),
    )
    best = cast(dict[str, Any] | None, decision.get("best_node"))
    _append_decision_payload_line(lines, "BEST SCORE NODE", best)
    if best and not best.get("selected"):
        rejected_at = best.get("rejected_at", "unknown")
        rejection_reason = best.get("reason", "unknown")
        line = Text("not selected:   ", style=TUI_ROW_LABEL_STYLE)
        line.append(
            f"{rejected_at} / {rejection_reason}",
            style=TUI_OPERATOR_NOTICE_STYLE,
        )
        lines.append(line)
        parent_metric = best.get("parent_metric")
        parent_text = "n/a" if parent_metric is None else f"{float(parent_metric):.5f}"
        line = Text("parent          ", style=TUI_ROW_LABEL_STYLE)
        line.append(f"metric={parent_text}", style=TUI_INACTIVE_VALUE_STYLE)
        if best.get("parent_is_buggy") is not None:
            line.append(
                f" buggy={best.get('parent_is_buggy')}",
                style=TUI_INACTIVE_VALUE_STYLE,
            )
        lines.append(line)

    diagnostics = cast(dict[str, Any] | None, decision.get("policy_diagnostics"))
    if diagnostics:
        _append_policy_debug_lines(lines, diagnostics)

    counts = cast(dict[str, Any], decision.get("counts") or {})
    if counts:
        lines.append(Text(""))
        lines.append(Text("FILTER COUNTS", style="bold blue"))
        for key, value in counts.items():
            line = Text(f"{key:<32}", style=TUI_ROW_LABEL_STYLE)
            line.append(str(value), style=TUI_NEUTRAL_VALUE_STYLE)
            lines.append(line)

    candidates = cast(list[dict[str, Any]], decision.get("top_candidates") or [])
    if candidates:
        lines.append(Text(""))
        lines.append(Text("TOP CANDIDATES", style="bold blue"))
        for candidate in candidates[:8]:
            line = Text(
                f"{candidate.get('rank', '?')}. ",
                style=TUI_ROW_LABEL_STYLE,
            )
            line.append(_decision_node_label(candidate), style=TUI_METRIC_VALUE_STYLE)
            policy_score = candidate.get("policy_score")
            if isinstance(policy_score, (float, int)) and not isinstance(
                policy_score,
                bool,
            ):
                line.append(f" policy={float(policy_score):.5f}", style=TUI_INACTIVE_VALUE_STYLE)
            lines.append(line)

    return Group(*lines)


def _runtime_generated_decision(node: Node, *, journal_size: int) -> dict[str, Any]:
    parent = node.parent
    parent_metric = (
        parent.metric.value
        if parent is not None
        and parent.metric is not None
        and parent.metric.value is not None
        else None
    )
    return {
        "step": journal_size,
        "mode": "runtime",
        "reason": "execute_generated_only_node",
        "selected": {
            "node_id": node.id,
            "step": node.step,
            "hypothesis_id": hypothesis_id_for_node(node),
            "metric": None,
            "is_buggy": node.is_buggy,
            "status": node.status,
            "parent_id": parent.id if parent is not None else None,
            "parent_step": parent.step if parent is not None else None,
            "parent_metric": parent_metric,
            "parent_is_buggy": parent.is_buggy if parent is not None else None,
            "child_count": len(node.children),
        },
        "best_node": None,
        "counts": {},
        "rejections": {},
        "top_candidates": [],
    }


class _Overlay:
    """Render a centered overlay panel over an existing Rich renderable."""

    def __init__(
        self,
        background,
        overlay,
        *,
        overlay_width: int,
        top: int | None = None,
        edge_margin: int = 3,
        dim: bool = True,
    ) -> None:
        self.background = background
        self.overlay = overlay
        self.overlay_width = overlay_width
        self.top = top
        self.edge_margin = edge_margin
        self.dim = dim

    def _slice_line(self, line, start: int, end: int):
        if start >= end:
            return []
        result = []
        pos = 0
        for segment in line:
            seg_len = segment.cell_length
            if seg_len == 0:
                if result:
                    result.append(segment)
                continue
            seg_end = pos + seg_len
            if seg_end <= start:
                pos = seg_end
                continue
            if pos >= end:
                break
            cut_start = max(start - pos, 0)
            cut_end = min(end - pos, seg_len)
            if cut_start == 0 and cut_end == seg_len:
                result.append(segment)
            else:
                _, right = segment.split_cells(cut_start)
                mid, _ = right.split_cells(cut_end - cut_start)
                result.append(mid)
            pos = seg_end
        return result

    def __rich_console__(self, console, options):
        width, height = options.size
        bg_lines = console.render_lines(self.background, options, pad=True)
        bg_lines = Segment.set_shape(bg_lines, width, height)
        if self.dim:
            bg_lines = [
                list(
                    Segment.apply_style(
                        line,
                        Style(dim=True),
                        post_style=Style(color="#5a5a5a"),
                    )
                )
                for line in bg_lines
            ]

        overlay_width = min(max(1, self.overlay_width), width)
        overlay_lines = console.render_lines(
            self.overlay,
            options.update(width=overlay_width),
            pad=True,
        )
        overlay_lines = [
            Segment.adjust_line_length(line, overlay_width) for line in overlay_lines
        ]

        left = max((width - overlay_width) // 2, 0)
        top = (
            self.top
            if self.top is not None
            else _overlay_top(
                console_height=height,
                overlay_height=len(overlay_lines),
                edge_margin=self.edge_margin,
            )
        )
        for index, overlay_line in enumerate(overlay_lines):
            target_row = top + index
            if target_row < 0 or target_row >= height:
                continue
            bg_line = bg_lines[target_row]
            left_segment = self._slice_line(bg_line, 0, left)
            right_segment = self._slice_line(bg_line, left + overlay_width, width)
            bg_lines[target_row] = left_segment + overlay_line + right_segment

        for last, line in loop_last(bg_lines):
            yield from line
            if not last:
                yield Segment.line()


def _overlay_top(
    *,
    console_height: int,
    overlay_height: int,
    edge_margin: int = 3,
) -> int:
    if console_height <= 0:
        return 0
    edge_margin = max(0, edge_margin)
    centered = max(0, (console_height - overlay_height) // 2)
    lowest_top = max(0, console_height - overlay_height - edge_margin)
    if lowest_top < edge_margin:
        return min(edge_margin, max(0, console_height - 1))
    return min(max(centered, edge_margin), lowest_top)


def render_tree_view(
    view: TreeView,
    *,
    focused_item_id: str,
    scroll_top: int,
    viewport_height: int,
) -> Group:
    viewport_height = max(1, viewport_height)
    visible_items = view.items[scroll_top : scroll_top + viewport_height]
    lines: list[Text] = []
    for item in visible_items:
        line = item.line.copy()
        if item.item_id == focused_item_id:
            line.stylize("reverse", item.focus_start, len(line.plain))
        lines.append(line)
    return Group(*lines)


def build_final_tree_renderable(
    journal: Journal,
    *,
    show_invalid_submission_branches: bool = False,
    disable_oom_saturated_parents: bool = False,
    plateau_block_epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
    synthesis_node_ids: set[str] | None = None,
    public_scores_by_node_id: dict[str, float] | None = None,
    public_score_bonus_weight: float = 0.0,
    public_score_bonus_cap: float = 0.0,
) -> Group:
    view = build_tree_view(
        journal,
        show_invalid_submission_branches=show_invalid_submission_branches,
        disable_oom_saturated_parents=disable_oom_saturated_parents,
        plateau_block_epsilon=plateau_block_epsilon,
        synthesis_node_ids=synthesis_node_ids,
        public_scores_by_node_id=public_scores_by_node_id,
        public_score_bonus_weight=public_score_bonus_weight,
        public_score_bonus_cap=public_score_bonus_cap,
    )
    return Group(*(item.line for item in view.items))


def print_final_tree(
    journal: Journal,
    *,
    log_dir: Path | str | None = None,
    show_invalid_submission_branches: bool = False,
    disable_oom_saturated_parents: bool = False,
    plateau_block_epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
    public_scores_by_node_id: dict[str, float] | None = None,
    public_score_bonus_weight: float = 0.0,
    public_score_bonus_cap: float = 0.0,
) -> None:
    synthesis_node_ids = synthesis_injected_node_ids(log_dir) if log_dir else set()
    Console().print(
        build_final_tree_renderable(
            journal,
            show_invalid_submission_branches=show_invalid_submission_branches,
            disable_oom_saturated_parents=disable_oom_saturated_parents,
            plateau_block_epsilon=plateau_block_epsilon,
            synthesis_node_ids=synthesis_node_ids,
            public_scores_by_node_id=public_scores_by_node_id,
            public_score_bonus_weight=public_score_bonus_weight,
            public_score_bonus_cap=public_score_bonus_cap,
        )
    )


def emit_completion_bell(
    *,
    tty_path: Path | str = "/dev/tty",
    sleep_seconds: float = 0.3,
) -> None:
    def ring() -> None:
        try:
            with open(tty_path, "w") as tty:
                tty.write("\x07")
                tty.flush()
        except OSError:
            sys.stderr.write("\x07")
            sys.stderr.flush()

    ring()
    time.sleep(sleep_seconds)
    ring()


class ArrowKeyReader:
    KEY_MAP = {
        b"\x1b[A": "up",
        b"\x1b[B": "down",
        b"\x1b[C": "right",
        b"\x1b[D": "left",
        b"\x1bOA": "up",
        b"\x1bOB": "down",
        b"\x1bOC": "right",
        b"\x1bOD": "left",
    }
    CHAR_KEY_MAP = {
        b"a": "active",
        b"A": "active",
        b"b": "best",
        b"B": "best",
        b"d": "debug",
        b"D": "debug",
        b"f": "follow",
        b"F": "follow",
        b"p": "public",
        b"P": "public",
        b"v": "view",
        b"V": "view",
        b"1": "copy_aide_panel",
        b"2": "copy_run_data_panel",
        b"3": "copy_logs_panel",
    }

    def __init__(self):
        self.enabled = False
        self.fd: int | None = None
        self._old_termios = None

    def __enter__(self):
        if not sys.stdin.isatty():
            return self
        try:
            self.fd = sys.stdin.fileno()
            self._old_termios = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
        except (OSError, termios.error):
            self.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self.fd is not None and self._old_termios is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._old_termios)
        self.enabled = False

    def read_key(self) -> str | None:
        if not self.enabled or self.fd is None:
            return None
        try:
            ready, _, _ = select.select([self.fd], [], [], 0)
            if not ready:
                return None
            first = os.read(self.fd, 1)
            if first != b"\x1b":
                return self.CHAR_KEY_MAP.get(first)

            data = bytearray(first)
            deadline = time.monotonic() + 0.05
            while time.monotonic() < deadline and len(data) < 8:
                ready, _, _ = select.select([self.fd], [], [], 0)
                if not ready:
                    time.sleep(0.001)
                    continue
                data.extend(os.read(self.fd, 1))
                key = self.KEY_MAP.get(bytes(data))
                if key is not None:
                    return key
            if bytes(data) == b"\x1b":
                return "dismiss_overlay"
            return self.KEY_MAP.get(bytes(data[:3]))
        except (OSError, termios.error):
            return None


def _display_base_path(base_path: Path) -> str:
    path = str(base_path)
    if not path.endswith(os.sep):
        path += os.sep
    return path


def _relative_display_path(path: Path, base_path: Path) -> str:
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(base_path)
    except ValueError:
        return str(resolved_path)
    return str(relative_path)


def build_path_summary(
    log_dir: Path,
    workspace_dir: Path,
    *,
    active_artifact_dir: Path | None = None,
) -> Group:
    path_entries = [
        ("Agent workspace directory", workspace_dir),
        ("Experiment log directory", log_dir),
    ]
    if active_artifact_dir is not None:
        path_entries.append(("Current artifact directory", active_artifact_dir))
    resolved_paths = [path.resolve() for _, path in path_entries]
    base_path = Path(os.path.commonpath([str(path) for path in resolved_paths]))

    lines = [
        Text("Base path", style="bold cyan"),
        Text(f"▶ {_display_base_path(base_path)}", style="yellow"),
        "",
    ]
    for label, path in path_entries:
        lines.extend(
            [
                Text(label, style="bold cyan"),
                Text(f"▶ {_relative_display_path(path, base_path)}", style="yellow"),
                "",
            ]
        )
    return Group(*lines)


def _clip_error_line(line: str, max_chars: int = 88) -> str:
    line = " ".join(line.split())
    if len(line) <= max_chars:
        return line
    return line[: max_chars - 1].rstrip() + "…"


def _clean_error_lines(lines: list[str], *, max_lines: int) -> list[str]:
    return [
        _clip_error_line(line)
        for line in lines
        if line.strip() and not line.strip().startswith("Execution time:")
    ][-max_lines:]


def _terminal_output_has_error(lines: list[str]) -> bool:
    markers = (
        "Traceback",
        "Error:",
        "Exception:",
        "KeyboardInterrupt",
        "MemoryError",
        "cannot ",
        "failed",
    )
    return any(any(marker in line for marker in markers) for line in lines)


def _exception_lines(node: Node) -> list[str]:
    if not node.exc_type:
        return []
    args = []
    if isinstance(node.exc_info, dict):
        raw_args = node.exc_info.get("args")
        if isinstance(raw_args, list):
            args = [str(arg) for arg in raw_args if str(arg)]
    if args:
        return [f"{node.exc_type}: {args[0]}"]
    return [str(node.exc_type)]


def last_error_record(
    journal: Journal,
    *,
    max_lines: int = 2,
) -> LastErrorRecord | None:
    for node in reversed(journal.buggy_nodes):
        term_lines = []
        if node._term_out:
            term_lines.extend("".join(node._term_out).splitlines())

        if term_lines and _terminal_output_has_error(term_lines):
            lines = _clean_error_lines(term_lines, max_lines=max_lines)
            if lines:
                return LastErrorRecord(node=node, lines=lines)

        lines = _clean_error_lines(_exception_lines(node), max_lines=max_lines)
        if lines:
            return LastErrorRecord(node=node, lines=lines)

        if node.analysis:
            lines = _clean_error_lines(
                str(node.analysis).splitlines(),
                max_lines=max_lines,
            )
            if lines:
                return LastErrorRecord(node=node, lines=lines)
    return None


def last_error_lines(journal: Journal, *, max_lines: int = 2) -> list[str]:
    record = last_error_record(journal, max_lines=max_lines)
    return record.lines if record is not None else []


def _last_error_title(record: LastErrorRecord | None) -> str:
    if record is None:
        return "Last Error"
    step = record.node.step if record.node.step is not None else "?"
    timestamp = dt.datetime.fromtimestamp(record.node.ctime).strftime("%m-%d %H:%M")
    hypothesis_id = hypothesis_id_for_node(record.node)
    hypothesis_text = f" · {hypothesis_id}" if hypothesis_id is not None else ""
    return (
        f"Last Error {_format_run_status_step(step)} @ {timestamp}"
        f"{hypothesis_text}"
    )


def build_last_error_summary(journal: Journal, *, log_dir: Path | None = None) -> Group:
    record = last_error_record(journal)
    lines: list[Text] = [Text(_last_error_title(record), style="bold red")]
    error_lines = record.lines if record is not None else []
    if not error_lines:
        lines.append(Text("-", style="dim"))
    else:
        if log_dir is not None and record is not None:
            artifact_dir = artifact_dir_for_node(log_dir, record.node)
            base_path = log_dir.resolve().parents[1]
            lines.append(
                Text(
                    "artifact "
                    + _relative_display_path(artifact_dir, base_path),
                    style="yellow",
                )
            )
        lines.extend(Text(line, style="dim") for line in error_lines)
    return Group(*lines)


STATUS_TIMESTAMP_KEYS: dict[str, tuple[str, ...]] = {
    "completed": ("completed_at", "started_at", "queued_at"),
    "injected": ("injected_at", "recorded_at", "completed_at", "started_at"),
    "ready": ("completed_at", "started_at", "queued_at"),
    "failed": ("failed_at", "completed_at", "started_at", "queued_at"),
    "running": ("started_at", "queued_at"),
    "queued": ("queued_at", "started_at"),
}

STATUS_SYMBOLS = {
    "completed": "✓",
    "injected": "✓",
    "ready": "…",
    "running": "▶",
    "queued": "…",
    "failed": "✗",
}
RUN_STATUS_LABEL_WIDTH = len("★ Best Score")
RUN_STATUS_STEP_WIDTH = 3
TUI_RESEARCH_ICON = "◆"
TUI_PHASE_ICON = "⬢"
TUI_SYNTHESIS_ICON = "◆"
TUI_BEST_SCORE_ICON = "★"
TUI_ROW_LABEL_STYLE = "bold cyan"
TUI_SEPARATOR_STYLE = "dim"
TUI_NEUTRAL_VALUE_STYLE = "yellow"
TUI_METRIC_VALUE_STYLE = "green"
TUI_INACTIVE_VALUE_STYLE = "dim"
TUI_OPERATOR_NOTICE_STYLE = "yellow"

LeftPanelView = Literal["tree", "root", "all", "branch"]
LEFT_PANEL_VIEW_ORDER: tuple[LeftPanelView, ...] = ("tree", "root", "all", "branch")


def next_left_panel_view(current: str) -> LeftPanelView:
    if current not in LEFT_PANEL_VIEW_ORDER:
        return "tree"
    index = LEFT_PANEL_VIEW_ORDER.index(cast(LeftPanelView, current))
    return LEFT_PANEL_VIEW_ORDER[(index + 1) % len(LEFT_PANEL_VIEW_ORDER)]


def _parse_status_time(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.strftime("%H:%M:%S")


def _status_timestamp(status_data: dict) -> str | None:
    status = str(status_data.get("status") or "")
    keys = STATUS_TIMESTAMP_KEYS.get(
        status,
        (
            "completed_at",
            "injected_at",
            "recorded_at",
            "failed_at",
            "started_at",
            "queued_at",
        ),
    )
    for key in keys:
        parsed = _parse_status_time(status_data.get(key))
        if parsed is not None:
            return parsed
    return None


def _latest_checkpoint_status(
    *,
    log_dir: Path | str,
    kind: Literal["research", "synthesis"],
) -> CheckpointStatusRecord | None:
    checkpoint_dir = Path(log_dir) / kind
    if not checkpoint_dir.exists():
        return None

    records: list[CheckpointStatusRecord] = []
    for status_path in sorted(checkpoint_dir.glob("checkpoint-*/status.json")):
        try:
            status_data = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(status_data, dict):
            continue
        status = status_data.get("status")
        label = status_path.parent.name.removeprefix("checkpoint-")
        records.append(
            CheckpointStatusRecord(
                label=label,
                status=str(status) if status is not None else None,
                timestamp=_status_timestamp(status_data),
            )
        )
    return records[-1] if records else None


def _status_style(status: str | None, fallback_status: str | None) -> str:
    if status in {"completed", "injected"}:
        return "green"
    if status in {"running", "queued"}:
        return "cyan"
    if status == "failed":
        return "red"
    if status in {"ready"}:
        return "yellow"

    fallback = fallback_status or ""
    for markup, style in (
        ("[green]", "green"),
        ("[cyan]", "cyan"),
        ("[red]", "red"),
        ("[yellow]", "yellow"),
        ("[dim]", "dim"),
    ):
        if fallback.startswith(markup):
            return style
    return "yellow"


def _fallback_status_symbol(status_text: str | None) -> str:
    if not status_text:
        return "?"
    for symbol in ("✓", "▶", "…", "✗", "?", "○"):
        if symbol in status_text:
            return symbol
    return "?"


def _fallback_checkpoint_label(status_text: str | None) -> str | None:
    if not status_text:
        return None
    match = re.search(r"\b(\d{6})\b", status_text)
    return match.group(1) if match else None


def _fallback_hypothesis_label(status_text: str | None) -> str | None:
    if not status_text:
        return None
    match = re.search(r"@\s*(\d{6})\b", status_text)
    return match.group(1) if match else None


def _compact_step_label(label: str | None) -> str | None:
    if label is None:
        return None
    stripped = label.strip()
    if stripped.isdigit():
        return str(int(stripped))
    return stripped


def _format_run_status_step(step: object) -> str:
    text = str(step).strip()
    if text.isdigit():
        value = int(text)
        if value <= 999:
            return f"{value:03d}"
        return str(value)
    return text.rjust(RUN_STATUS_STEP_WIDTH)


def _run_status_label(icon: str, title: str) -> str:
    return f"{icon} {title}".ljust(RUN_STATUS_LABEL_WIDTH)


def _run_status_line_prefix(icon: str, title: str) -> Text:
    line = Text()
    line.append(_run_status_label(icon, title), style=TUI_ROW_LABEL_STYLE)
    return line


def build_checkpoint_status_line(
    *,
    title: str,
    icon: str,
    status_text: str | None,
    record: CheckpointStatusRecord | None,
) -> Text | None:
    if status_text is None:
        return None

    label = _compact_step_label(
        record.label if record is not None else _fallback_checkpoint_label(status_text)
    )
    symbol = (
        STATUS_SYMBOLS.get(record.status or "", "?")
        if record is not None
        else _fallback_status_symbol(status_text)
    )
    style = _status_style(record.status if record is not None else None, status_text)

    line = _run_status_line_prefix(icon, title)
    if label is None:
        line.append(f" {symbol}", style=style)
        return line

    line.append(f" {_format_run_status_step(label)}", style=style)
    if record is not None and record.timestamp is not None:
        line.append(f" @ {record.timestamp}", style=style)
    hypothesis_label = (
        None if record is not None else _fallback_hypothesis_label(status_text)
    )
    if hypothesis_label is not None:
        line.append(f" @ {hypothesis_label}", style=style)
    line.append(f" {symbol}", style=style)
    return line


def _best_scored_node(journal: Journal) -> Node | None:
    candidates = [
        node
        for node in journal.good_nodes
        if not node.is_in_submission_contract_error_branch
        and node.metric is not None
        and node.metric.value is not None
    ]
    return max(candidates, key=lambda node: node.metric, default=None)


def build_best_score_status(journal: Journal) -> Text | None:
    node = _best_scored_node(journal)
    if node is None:
        return None
    step = node.step if node.step is not None else "?"
    timestamp = dt.datetime.fromtimestamp(node.ctime).strftime("%m-%d %H:%M")
    hypothesis_id = hypothesis_id_for_node(node)
    suffix = f" · {hypothesis_id}" if hypothesis_id is not None else ""
    line = _run_status_line_prefix(TUI_BEST_SCORE_ICON, "Best Score")
    line.append(
        f" {_format_run_status_step(step)} @ {timestamp} {node.metric.value:.5f}"
        f"{suffix}",
        style=TUI_METRIC_VALUE_STYLE,
    )
    return line


def _count_hypothesis_root_nodes(
    journal: Journal,
    *,
    include_generated: bool = True,
    compatible_hypothesis_ids: set[str] | None = None,
) -> int:
    return sum(
        1
        for node in journal.nodes
        if node.parent is None
        and hypothesis_id_for_node(node) is not None
        and (
            compatible_hypothesis_ids is None
            or hypothesis_id_for_node(node) in compatible_hypothesis_ids
        )
        and (include_generated or node.status != "generated")
    )


def _count_hypothesis_completed_nodes(
    journal: Journal,
    *,
    include_generated: bool,
) -> int:
    if include_generated:
        return len(journal.nodes)
    return sum(1 for node in journal.nodes if node.status != "generated")


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _source_ref_compatible_hypothesis_count(cfg: Config) -> int | None:
    source_ref_path = Path(cfg.log_dir) / "research_hypotheses" / "source_ref.json"
    try:
        payload = json.loads(source_ref_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    try:
        compatible_count = int(payload.get("compatible_hypothesis_count"))
    except (TypeError, ValueError):
        return None
    return compatible_count if compatible_count >= 0 else None


def _source_ref_compatible_hypothesis_ids(cfg: Config) -> set[str] | None:
    source_ref_path = Path(cfg.log_dir) / "research_hypotheses" / "source_ref.json"
    try:
        payload = json.loads(source_ref_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    raw_ids = payload.get("compatible_hypothesis_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
        return None
    return set(raw_ids)


def _current_compatible_hypothesis_ids(cfg: Config) -> set[str] | None:
    try:
        library = load_manual_hypothesis_library(cfg)
    except (FileNotFoundError, OSError, ValueError):
        return None
    return {
        hypothesis.id
        for hypothesis in _compatible_manual_hypotheses(cfg, library)
    }


def _compatible_hypothesis_ids_for_phase_status(cfg: Config) -> set[str] | None:
    return _source_ref_compatible_hypothesis_ids(cfg) or _current_compatible_hypothesis_ids(
        cfg
    )


def build_hypothesis_phase_status(
    cfg: Config,
    journal: Journal,
    *,
    include_generated_roots: bool = True,
    selected_hypothesis_ids: tuple[str, ...] | None = None,
) -> Text | None:
    if not cfg.research.enabled or getattr(cfg.research, "mode", "llm") != "hypothesis":
        return None

    total_budget = _positive_int(getattr(cfg.agent, "steps", 0), 0)
    if total_budget <= 0:
        return None

    selected_ids = set(selected_hypothesis_ids or ())
    compatible_hypothesis_ids = (
        selected_ids if selected_ids else _compatible_hypothesis_ids_for_phase_status(cfg)
    )
    root_count = _count_hypothesis_root_nodes(
        journal,
        include_generated=include_generated_roots,
        compatible_hypothesis_ids=compatible_hypothesis_ids,
    )
    all_root_count = _count_hypothesis_root_nodes(
        journal,
        compatible_hypothesis_ids=compatible_hypothesis_ids,
    )
    configured_root_limit = _positive_int(
        getattr(cfg.research, "hypothesis_root_limit", 100),
        100,
    )
    if selected_ids:
        root_limit = len(selected_ids)
    else:
        compatible_count = (
            len(compatible_hypothesis_ids)
            if compatible_hypothesis_ids is not None
            else _source_ref_compatible_hypothesis_count(cfg)
        )
        root_limit = (
            effective_hypothesis_root_limit(cfg, compatible_count=compatible_count)
            if compatible_count is not None
            else configured_root_limit
        )
    exploration_budget = min(
        max(root_limit, all_root_count),
        total_budget,
    )
    exploitation_budget = max(total_budget - exploration_budget, 0)
    completed_count = (
        root_count
        if selected_ids
        else min(
            _count_hypothesis_completed_nodes(
                journal,
                include_generated=include_generated_roots,
            ),
            total_budget,
        )
    )
    exploitation_count = max(completed_count - root_count, 0)
    if exploitation_budget > 0:
        exploitation_count = min(exploitation_count, exploitation_budget)

    exploration_active = root_count < exploration_budget
    exploration_style = (
        TUI_METRIC_VALUE_STYLE if exploration_active else TUI_INACTIVE_VALUE_STYLE
    )
    exploitation_style = (
        TUI_INACTIVE_VALUE_STYLE if exploration_active else TUI_METRIC_VALUE_STYLE
    )

    line = _run_status_line_prefix(TUI_PHASE_ICON, "Phase")
    line.append(" ")
    line.append(
        f"exploration {root_count}/{exploration_budget}",
        style=exploration_style,
    )
    line.append(" · ", style=TUI_SEPARATOR_STYLE)
    line.append(
        f"exploitation {exploitation_count}/{exploitation_budget}",
        style=exploitation_style,
    )
    return line


def _format_percent(value: float) -> str:
    if value == 0 or value >= 10:
        return f"{value:.0f}%"
    return f"{value:.1f}%"


def _format_gib(value: int) -> str:
    return f"{value / 1024**3:.1f}G"


def _format_gib_float(value: float) -> str:
    return f"{value:.1f}G"


def _format_watts(value: float) -> str:
    if value == 0 or value >= 10:
        return f"{value:.0f}W"
    return f"{value:.1f}W"


def _format_celsius(value: float) -> str:
    if value == 0 or value >= 10:
        return f"{value:.0f}C"
    return f"{value:.1f}C"


RESOURCE_SPARKLINE_LEVELS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], *, width: int, ceiling: float) -> str:
    if width <= 0:
        return ""
    sampled = values[-width:]
    if not sampled:
        return " " * width

    top = max(ceiling, max(sampled), 1.0)
    chars: list[str] = []
    for value in sampled:
        ratio = min(max(value / top, 0.0), 1.0)
        chars.append(RESOURCE_SPARKLINE_LEVELS[round(ratio * (len(RESOURCE_SPARKLINE_LEVELS) - 1))])
    return "".join(chars)


def _hbar(value: float, *, ceiling: float, width: int = 10) -> str:
    top = max(ceiling, 1.0)
    filled = round(min(max(value / top, 0.0), 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def _resource_row(
    label: str,
    value: float | None,
    *,
    formatted_value: str,
    history_values: list[float],
    ceiling: float,
    graph_width: int,
) -> tuple[str, str, str, str]:
    if value is None:
        return (label, "n/a", _hbar(0.0, ceiling=ceiling), " " * graph_width)
    return (
        label,
        formatted_value,
        _hbar(value, ceiling=ceiling),
        _sparkline(history_values, width=graph_width, ceiling=ceiling),
    )


def _resource_text(label: str, value: str, bar: str, spark: str) -> Text:
    prefix = f"▶ {label:<4} "
    padded_value = f"{value:>7}"
    if label == "GPU" and value.endswith("%"):
        try:
            gpu_percent = float(value.rstrip("%"))
        except ValueError:
            gpu_percent = 0.0
        if gpu_percent > 10.0:
            line = Text(prefix, style="yellow")
            line.append(padded_value, style="red")
            line.append(f" {bar} {spark}", style="yellow")
            return line
    return Text(f"{prefix}{padded_value} {bar} {spark}", style="yellow")


def build_resource_summary(
    resource_history: ResourceHistory | None,
    *,
    graph_width: int = 24,
) -> Group:
    lines: list[Text] = [Text("Resources", style="bold cyan")]
    snapshot = resource_history.latest if resource_history is not None else None
    if snapshot is None or resource_history is None:
        lines.append(Text("▶ waiting for code execution sample", style="yellow"))
        return Group(*lines)
    else:
        cpu_ceiling = float((os.cpu_count() or 1) * 100)
        memory_ceiling = max(
            1.0,
            max(resource_history.ram_gib_values + resource_history.peak_ram_gib_values) * 1.2,
        )
        gpu_percent_values = resource_history.gpu_percent_values
        gpu_memory_used_gib_values = resource_history.gpu_memory_used_gib_values
        gpu_power_draw_watts_values = resource_history.gpu_power_draw_watts_values
        gpu_temperature_celsius_values = resource_history.gpu_temperature_celsius_values
        gpu_memory_used_gib = (
            snapshot.gpu_memory_used_bytes / 1024**3
            if snapshot.gpu_memory_used_bytes is not None
            else None
        )
        gpu_memory_total_gib = (
            snapshot.gpu_memory_total_bytes / 1024**3
            if snapshot.gpu_memory_total_bytes is not None
            else None
        )
        gpu_memory_ceiling = max(
            1.0,
            gpu_memory_total_gib or 0.0,
            max(gpu_memory_used_gib_values, default=0.0) * 1.2,
        )
        gpu_power_ceiling = max(
            1.0,
            snapshot.gpu_power_limit_watts or 0.0,
            max(gpu_power_draw_watts_values, default=0.0) * 1.2,
        )
        gpu_temperature_ceiling = max(
            100.0,
            max(gpu_temperature_celsius_values, default=0.0) * 1.2,
        )
        values = [
            (
                "CPU",
                _format_percent(snapshot.cpu_percent),
                _hbar(snapshot.cpu_percent, ceiling=cpu_ceiling),
                _sparkline(
                    resource_history.cpu_percent_values,
                    width=graph_width,
                    ceiling=cpu_ceiling,
                ),
            ),
            (
                "RAM",
                _format_gib(snapshot.ram_bytes),
                _hbar(snapshot.ram_bytes / 1024**3, ceiling=memory_ceiling),
                _sparkline(
                    resource_history.ram_gib_values,
                    width=graph_width,
                    ceiling=memory_ceiling,
                ),
            ),
            (
                "peak",
                _format_gib(snapshot.peak_ram_bytes),
                _hbar(snapshot.peak_ram_bytes / 1024**3, ceiling=memory_ceiling),
                _sparkline(
                    resource_history.peak_ram_gib_values,
                    width=graph_width,
                    ceiling=memory_ceiling,
                ),
            ),
            _resource_row(
                "GPU",
                snapshot.gpu_percent,
                formatted_value=(
                    _format_percent(snapshot.gpu_percent)
                    if snapshot.gpu_percent is not None
                    else "n/a"
                ),
                history_values=gpu_percent_values,
                ceiling=100.0,
                graph_width=graph_width,
            ),
            _resource_row(
                "VRAM",
                gpu_memory_used_gib,
                formatted_value=(
                    _format_gib_float(gpu_memory_used_gib)
                    if gpu_memory_used_gib is not None
                    else "n/a"
                ),
                history_values=gpu_memory_used_gib_values,
                ceiling=gpu_memory_ceiling,
                graph_width=graph_width,
            ),
            _resource_row(
                "PWR",
                snapshot.gpu_power_draw_watts,
                formatted_value=(
                    _format_watts(snapshot.gpu_power_draw_watts)
                    if snapshot.gpu_power_draw_watts is not None
                    else "n/a"
                ),
                history_values=gpu_power_draw_watts_values,
                ceiling=gpu_power_ceiling,
                graph_width=graph_width,
            ),
            _resource_row(
                "TEMP",
                snapshot.gpu_temperature_celsius,
                formatted_value=(
                    _format_celsius(snapshot.gpu_temperature_celsius)
                    if snapshot.gpu_temperature_celsius is not None
                    else "n/a"
                ),
                history_values=gpu_temperature_celsius_values,
                ceiling=gpu_temperature_ceiling,
                graph_width=graph_width,
            ),
        ]
    lines.extend(_resource_text(label, value, bar, spark) for label, value, bar, spark in values)
    return Group(*lines)


RUN_LOG_FILE_NAMES = ("process_stdout.log", "autogluon_stdout.log", "autogluon.log")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MISSING_LOG_HINT_KEYS = (
    "Title",
    "Summary",
    "Try",
    "Rationale",
    "Implementation",
    "Expected effect",
    "Risk",
    "Sources",
)


def active_run_log_path(active_artifact_dir: Path | None) -> Path | None:
    if active_artifact_dir is None:
        return None
    for name in RUN_LOG_FILE_NAMES:
        path = active_artifact_dir / name
        if path.exists():
            return path
    return None


def _tail_log_lines(path: Path, *, max_lines: int, max_bytes: int = 65536) -> list[str]:
    if max_lines <= 0:
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return lines


def _clip_log_line(line: str, *, max_width: int) -> str:
    clean = ANSI_ESCAPE_RE.sub("", line).replace("\t", "    ")
    if max_width <= 1:
        return clean[:max_width]
    if len(clean) <= max_width:
        return clean
    return clean[: max_width - 1].rstrip() + "…"


def _format_missing_log_hint_line(line: str, *, max_width: int) -> Text:
    clean = (
        ANSI_ESCAPE_RE.sub("", line)
        .replace("\\n", " ")
        .replace("\\r", " ")
        .replace("\\t", " ")
        .replace("\t", "    ")
    )
    for key in MISSING_LOG_HINT_KEYS:
        prefix = f"{key}: "
        if clean.startswith(prefix):
            text = Text()
            text.append(prefix, style="bold cyan")
            text.append(clean[len(prefix) :], style="dim")
            return text
    if clean.startswith("Hypothesis "):
        key, value = clean.split(" ", 1)
        text = Text()
        text.append(f"{key} ", style="bold cyan")
        text.append(value, style="dim")
        return text
    return Text(clean, style="dim")


def _wrap_missing_log_hint_lines(lines: list[str], *, max_width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        clean = (
            ANSI_ESCAPE_RE.sub("", line)
            .replace("\\n", " ")
            .replace("\\r", " ")
            .replace("\\t", " ")
            .replace("\t", "    ")
        )
        if max_width <= 1 or len(clean) <= max_width:
            wrapped.append(clean)
            continue
        segments = textwrap.wrap(
            clean,
            width=max_width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        wrapped.extend(segments or [""])
    return wrapped


def build_run_log_summary(
    active_artifact_dir: Path | None,
    *,
    max_lines: int,
    max_width: int,
    missing_log_hint: str | None = None,
) -> Group:
    log_path = active_run_log_path(active_artifact_dir)
    if log_path is None:
        if missing_log_hint:
            hint_lines = _wrap_missing_log_hint_lines(
                missing_log_hint.splitlines(),
                max_width=max_width,
            )
            if len(hint_lines) > max_lines:
                hint_lines = hint_lines[:max_lines]
            return Group(
                *(
                    _format_missing_log_hint_line(line, max_width=max_width)
                    for line in hint_lines
                )
            )
        return Group(Text("waiting for process log", style="dim"))

    log_lines = _tail_log_lines(log_path, max_lines=max_lines)
    if not log_lines:
        return Group(Text(f"{log_path.name} is empty", style="dim"))

    return Group(
        *(
            Text(_clip_log_line(line, max_width=max_width), style="dim")
            for line in log_lines
        )
    )


ModelSetting = tuple[str, str, str | None]


def build_model_summary(model_settings: list[ModelSetting] | None) -> Group | None:
    if not model_settings:
        return None
    lines: list[Text] = [Text("Models", style=TUI_ROW_LABEL_STYLE)]
    for label, model, effort in model_settings:
        line = Text()
        line.append(f"▶ {label:<9} ", style=TUI_ROW_LABEL_STYLE)
        line.append(f"{model} - {effort or '-'}", style=TUI_NEUTRAL_VALUE_STYLE)
        lines.append(line)
    return Group(*lines)


def build_agent_mode_summary(
    cfg: Config | None,
    *,
    skip_execution: bool = False,
    hypothesis_root_generate_workers: int = 1,
) -> Group | None:
    if cfg is None:
        return None
    mode = str(getattr(cfg.agent, "mode", "legacy"))
    mode_line = Text()
    mode_line.append("▶ mode      ", style=TUI_ROW_LABEL_STYLE)
    mode_line.append(mode, style=TUI_NEUTRAL_VALUE_STYLE)
    ag_profile_line = None
    if mode == "autogluon_preprocess":
        ag_profile_line = Text()
        ag_profile_line.append("▶ ag.profile ", style=TUI_ROW_LABEL_STYLE)
        ag_profile_line.append(
            str(getattr(cfg.agent.autogluon, "profile", "") or "-"),
            style=TUI_NEUTRAL_VALUE_STYLE,
        )
    aux_line = Text()
    aux_line.append("▶ aux       ", style=TUI_ROW_LABEL_STYLE)
    mode_value = aux_mode(cfg)
    aux_value = aux_file_name(cfg) if mode_value == "file" else mode_value
    aux_line.append(aux_value or "false", style=TUI_NEUTRAL_VALUE_STYLE)
    gpu_line = Text()
    gpu_line.append("▶ gpu       ", style=TUI_ROW_LABEL_STYLE)
    gpu_line.append(
        "true" if bool(getattr(cfg.agent, "gpu", False)) else "false",
        style=TUI_NEUTRAL_VALUE_STYLE,
    )
    run_line = Text()
    run_line.append("▶ run       ", style=TUI_ROW_LABEL_STYLE)
    run_line.append(
        "generate-only" if skip_execution else "execute",
        style=TUI_NEUTRAL_VALUE_STYLE,
    )
    lines = [
        Text("Agent", style=TUI_ROW_LABEL_STYLE),
        mode_line,
    ]
    if ag_profile_line is not None:
        lines.append(ag_profile_line)
    lines.extend([aux_line, gpu_line, run_line])
    if skip_execution:
        workers_line = Text()
        workers_line.append("▶ workers   ", style=TUI_ROW_LABEL_STYLE)
        workers_line.append(
            str(hypothesis_root_generate_workers),
            style=TUI_NEUTRAL_VALUE_STYLE,
        )
        lines.append(workers_line)
    return Group(*lines)


def build_operator_notice_summary(notice: str | None) -> Group | None:
    if notice is None or not notice.strip():
        return None
    return Group(
        Text("Operator Notice", style=TUI_ROW_LABEL_STYLE),
        Text(notice.strip(), style=TUI_OPERATOR_NOTICE_STYLE),
    )


def _sanitize_panel_copy_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return cleaned or "panel"


def panel_copy_path(panel_name: str, run_id: str, *, tmp_dir: Path = Path("/tmp")) -> Path:
    name = _sanitize_panel_copy_name(panel_name).lower()
    run = _sanitize_panel_copy_name(run_id)
    return tmp_dir / f"panel-{name}-{run}.txt"


def render_panel_copy_text(title: str, renderable, *, width: int = 100) -> str:
    buffer = io.StringIO()
    console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=max(20, int(width)),
        record=True,
    )
    console.print(renderable)
    body = console.export_text(styles=False).rstrip()
    title = title.strip()
    if body:
        return f"# {title}\n\n{body}\n"
    return f"# {title}\n"


def osc52_clipboard_sequence(text: str, *, tmux: bool = False) -> str:
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    sequence = f"\x1b]52;c;{encoded}\x07"
    if tmux:
        return f"\x1bPtmux;\x1b{sequence}\x1b\\"
    return sequence


def copy_text_to_clipboard_osc52(
    text: str,
    *,
    tty_path: Path | str = "/dev/tty",
    tmux: bool | None = None,
) -> bool:
    sequence = osc52_clipboard_sequence(
        text,
        tmux=("TMUX" in os.environ if tmux is None else tmux),
    )
    try:
        with open(tty_path, "w", encoding="utf-8") as tty:
            tty.write(sequence)
            tty.flush()
        return True
    except OSError:
        try:
            sys.stdout.write(sequence)
            sys.stdout.flush()
            return True
        except OSError:
            return False


def save_panel_copy(
    panel_name: str,
    run_id: str,
    text: str,
    *,
    tmp_dir: Path = Path("/tmp"),
) -> Path:
    path = panel_copy_path(panel_name, run_id, tmp_dir=tmp_dir)
    path.write_text(text, encoding="utf-8")
    return path


def build_panel_copy_notice(
    *,
    panel_title: str,
    path: Path,
    osc52_sent: bool,
) -> Panel:
    status = "Sent to clipboard using OSC 52." if osc52_sent else "OSC 52 unavailable."
    return Panel(
        Group(
            Text(status, style=TUI_NEUTRAL_VALUE_STYLE),
            Text("Fallback file:", style=TUI_ROW_LABEL_STYLE),
            Text(str(path), style="yellow"),
        ),
        title=f"[b]Copied {panel_title}",
        border_style="green" if osc52_sent else "yellow",
    )


def model_settings_for_run(cfg: Config) -> list[ModelSetting]:
    settings: list[ModelSetting] = [
        ("code", cfg.agent.code.model, cfg.agent.code.reasoning_effort),
        ("feedback", cfg.agent.feedback.model, cfg.agent.feedback.reasoning_effort),
        ("report", cfg.report.model, cfg.report.reasoning_effort),
    ]
    if cfg.research.enabled and getattr(cfg.research, "mode", "llm") != "hypothesis":
        settings.append(("research", cfg.research.model, cfg.research.reasoning_effort))
    if cfg.synthesis.enabled:
        settings.append(
            ("synthesis", cfg.synthesis.model, cfg.synthesis.reasoning_effort)
        )
    return settings


def build_run_data(
    *,
    progress,
    status,
    research_status: str | None,
    synthesis_status: str | None,
    journal: Journal,
    log_dir: Path,
    workspace_dir: Path,
    resource_snapshot: ResourceSnapshot | None = None,
    resource_history: ResourceHistory | None = None,
    resource_active: bool = False,
    resource_graph_width: int = 24,
    model_settings: list[ModelSetting] | None = None,
    active_artifact_dir: Path | None = None,
    cfg: Config | None = None,
    operator_notice: str | None = None,
    include_generated_hypothesis_roots: bool = True,
    skip_execution: bool = False,
    hypothesis_root_generate_workers: int = 1,
    selected_hypothesis_ids: tuple[str, ...] = (),
) -> Group:
    if resource_history is None and resource_snapshot is not None:
        resource_history = ResourceHistory()
        resource_history.add(resource_snapshot)
    lines = [progress, status]
    research_line = build_checkpoint_status_line(
        title="Research",
        icon=TUI_RESEARCH_ICON,
        status_text=research_status,
        record=_latest_checkpoint_status(log_dir=log_dir, kind="research"),
    )
    synthesis_line = build_checkpoint_status_line(
        title="Synthesis",
        icon=TUI_SYNTHESIS_ICON,
        status_text=synthesis_status,
        record=_latest_checkpoint_status(log_dir=log_dir, kind="synthesis"),
    )
    best_score_status = build_best_score_status(journal)
    if research_line is not None:
        lines.append("")
        lines.append(research_line)
    phase_line = (
        build_hypothesis_phase_status(
            cfg,
            journal,
            include_generated_roots=include_generated_hypothesis_roots,
            selected_hypothesis_ids=selected_hypothesis_ids,
        )
        if cfg is not None
        else None
    )
    if phase_line is not None:
        if research_line is None:
            lines.append("")
        lines.append(phase_line)
    if synthesis_line is not None:
        if research_line is None and phase_line is None:
            lines.append("")
        lines.append(synthesis_line)
    if best_score_status is not None:
        if research_line is None and phase_line is None and synthesis_line is None:
            lines.append("")
        lines.append(best_score_status)
    model_summary = build_model_summary(model_settings)
    if model_summary is not None:
        lines.extend(["", model_summary])
    agent_mode_summary = build_agent_mode_summary(
        cfg,
        skip_execution=skip_execution,
        hypothesis_root_generate_workers=hypothesis_root_generate_workers,
    )
    if agent_mode_summary is not None:
        lines.extend(["", agent_mode_summary])
    lines.extend(
        [
            "",
            build_path_summary(
                log_dir,
                workspace_dir,
                active_artifact_dir=active_artifact_dir,
            ),
        ]
    )
    operator_notice_summary = build_operator_notice_summary(operator_notice)
    if operator_notice_summary is not None:
        lines.extend([Rule(style="dim"), operator_notice_summary])
    lines.extend([Rule(style="dim"), build_last_error_summary(journal, log_dir=log_dir)])
    if resource_active:
        lines.extend(
            [
                Rule(style="dim"),
                build_resource_summary(
                    resource_history,
                    graph_width=resource_graph_width,
                ),
            ]
        )
    return Group(*lines)


def _submission_validation_record(
    *,
    status: str,
    sample_path: Path,
    submission_path: Path,
    error: str | None = None,
    previous_metric: MetricValue | None = None,
) -> dict:
    record = {
        "status": status,
        "sample_signature": file_signature(sample_path),
        "submission_signature": file_signature(submission_path),
    }
    if error is not None:
        record["error"] = error
    if previous_metric is not None and previous_metric.value is not None:
        record["previous_metric"] = {
            "value": previous_metric.value,
            "maximize": previous_metric.maximize,
        }
    return record


def _submission_validation_cache_matches(
    node: Node,
    *,
    sample_path: Path,
    submission_path: Path,
) -> bool:
    record = node.submission_validation
    return (
        isinstance(record, dict)
        and record.get("status") == "ok"
        and record.get("sample_signature") == file_signature(sample_path)
        and record.get("submission_signature") == file_signature(submission_path)
    )


def _mark_node_submission_ok(
    node: Node,
    *,
    sample_path: Path,
    submission_path: Path,
) -> bool:
    record = _submission_validation_record(
        status="ok",
        sample_path=sample_path,
        submission_path=submission_path,
    )
    previous_metric = None
    if isinstance(node.submission_validation, dict):
        previous_metric = node.submission_validation.get("previous_metric")
    if node.is_submission_contract_error and isinstance(previous_metric, dict):
        value = previous_metric.get("value")
        if value is not None:
            node.metric = MetricValue(
                float(value),
                maximize=previous_metric.get("maximize"),
            )
            node.is_buggy = False
            node.exc_type = None
            node.exc_info = None
            node.exc_stack = None

    if node.submission_validation == record:
        return False
    node.submission_validation = record
    return True


def _mark_node_submission_bug(
    node: Node,
    error: str,
    *,
    sample_path: Path | None = None,
    submission_path: Path | None = None,
) -> bool:
    if node.is_buggy and node.exc_type == "SubmissionValidationError":
        return False

    previous_metric = node.metric if isinstance(node.metric, MetricValue) else None
    node.is_buggy = True
    node.metric = WorstMetricValue()
    node.exc_type = node.exc_type or "SubmissionValidationError"
    node.exc_info = node.exc_info or {"args": [error]}
    if sample_path is not None and submission_path is not None:
        node.submission_validation = _submission_validation_record(
            status="error",
            sample_path=sample_path,
            submission_path=submission_path,
            error=error,
            previous_metric=previous_metric,
        )
    validation_message = f"SubmissionValidationError: {error}\n"
    if node._term_out is None:
        node._term_out = []
    node._term_out.append(validation_message)
    previous_analysis = node.analysis or ""
    node.analysis = (
        f"Submission validation failed: {error}"
        if not previous_analysis
        else f"{previous_analysis}\n\nSubmission validation failed: {error}"
    )
    return True


def _lightgbm_cuda_code_evidence(code: str | None) -> bool:
    if not code:
        return False
    lower_code = code.lower()
    has_lightgbm = "lightgbm" in lower_code or "lgbmclassifier" in lower_code
    has_cuda = (
        '"cuda"' in lower_code
        or "'cuda'" in lower_code
        or "device_type=\"cuda\"" in lower_code
        or "device_type='cuda'" in lower_code
        or "device=\"cuda\"" in lower_code
        or "device='cuda'" in lower_code
    )
    return has_lightgbm and has_cuda


def _lightgbm_code_evidence(code: str | None) -> bool:
    if not code:
        return False
    lower_code = code.lower()
    return "lightgbm" in lower_code or "lgbmclassifier" in lower_code


def _execution_crash_log_diagnostic(
    artifact_dir: Path | None, *, code: str | None = None
) -> str | None:
    if artifact_dir is None:
        return None

    lightgbm_code = _lightgbm_code_evidence(code)
    lightgbm_cuda_code = _lightgbm_cuda_code_evidence(code)

    for log_name in ("autogluon_stdout.log", "process_stdout.log"):
        log_path = artifact_dir / log_name
        if not log_path.exists():
            continue
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lower_log = log_text.lower()
        is_nvidia_mmu_fault = (
            "nvrm: xid" in lower_log
            or "mmu fault" in lower_log
            or "fault_pde" in lower_log
        )
        is_lightgbm_cuda_log = (
            ("lightgbm" in lower_log or lightgbm_code or lightgbm_cuda_code)
            and ("cuda" in lower_log or is_nvidia_mmu_fault)
            and (
                "mmu fault" in lower_log
                or "fault_pde" in lower_log
                or "nvrm: xid" in lower_log
                or "process died" in lower_log
                or "segmentation fault" in lower_log
            )
        )
        if is_lightgbm_cuda_log:
            evidence = ""
            for line in log_text.splitlines():
                lower_line = line.lower()
                if (
                    "nvrm: xid" in lower_line
                    or "mmu fault" in lower_line
                    or "fault_pde" in lower_line
                    or "segmentation fault" in lower_line
                ):
                    evidence = line.strip()
                    break

            diagnostic = (
                "LightGBM CUDA native crash while the REPL child process was "
                f"executing. Evidence from {log_name}"
            )
            if evidence:
                diagnostic += f": {evidence}"
            return diagnostic

        is_cuda_oom = "cuda error 2: out of memory" in lower_log
        is_catboost_cuda_oom = (
            "catboost" in lower_log
            and "cuda" in lower_log
            and "out of memory" in lower_log
        )
        if not is_cuda_oom and not is_catboost_cuda_oom:
            continue

        evidence = ""
        for line in log_text.splitlines():
            lower_line = line.lower()
            if "cuda error" in lower_line or "out of memory" in lower_line:
                evidence = line.strip()
                break

        diagnostic = (
            "CatBoost GPU ran out of memory while the REPL child process was "
            f"executing. Evidence from {log_name}"
        )
        if evidence:
            diagnostic += f": {evidence}"
        return diagnostic

    if lightgbm_cuda_code:
        return (
            "LightGBM CUDA native crash likely terminated the REPL child process "
            "before Python could raise an exception. Evidence: generated code "
            "uses LightGBM with CUDA and no Python traceback was captured."
        )

    return None


def _mark_node_execution_crash(
    node: Node,
    exc: RuntimeError,
    *,
    artifact_dir: Path | None = None,
) -> None:
    message = str(exc) or exc.__class__.__name__
    diagnostic = _execution_crash_log_diagnostic(artifact_dir, code=node.code)
    if diagnostic is not None:
        message = f"{message}\n\n{diagnostic}"
    node._term_out = [f"{exc.__class__.__name__}: {message}\n"]
    node.exec_time = 0.0
    node.exc_type = exc.__class__.__name__
    node.exc_info = {"args": [message]}
    node.exc_stack = None
    node.analysis = message
    node.metric = WorstMetricValue()
    node.is_buggy = True
    node.status = "failed" if diagnostic is not None else "bug"


def enforce_submission_contract(cfg, node: Node) -> bool:
    workspace_dir = Path(cfg.workspace_dir)
    sample_path = find_sample_submission(workspace_dir / "input")
    if sample_path is None:
        return False

    submission_path = workspace_dir / "working" / "submission.csv"
    if not submission_path.exists():
        return _mark_node_submission_bug(
            node,
            "missing working/submission.csv while sample_submission exists",
        )
    if submission_path.stat().st_mtime < node.ctime:
        return _mark_node_submission_bug(
            node,
            "stale working/submission.csv from before this node execution",
        )

    error = validate_workspace_submission(workspace_dir)
    if error is None:
        return _mark_node_submission_ok(
            node,
            sample_path=sample_path,
            submission_path=submission_path,
        )
    return _mark_node_submission_bug(
        node,
        error,
        sample_path=sample_path,
        submission_path=submission_path,
    )


def _node_artifact_submission_path(cfg, node: Node) -> Path:
    return artifact_submission_path_for_node(cfg.log_dir, node)


def _node_artifact_dir(cfg, node: Node) -> Path:
    return artifact_dir_for_node(cfg.log_dir, node)


def _node_artifact_dir_from_ctime(cfg, ctime: float) -> Path:
    timestamp = dt.datetime.fromtimestamp(ctime).strftime("%Y%m%dT%H%M%S")
    return Path(cfg.log_dir) / "artifacts" / timestamp


def allocate_node_artifact_slot(
    log_dir: Path | str,
    *,
    step: int | str | None = None,
) -> tuple[float, str, Path]:
    ctime = time.time()
    artifacts_dir = Path(log_dir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    while True:
        dir_name = new_artifact_dir_name(ctime=ctime, step=step)
        artifact_dir = artifacts_dir / dir_name
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return ctime, dir_name, artifact_dir


def ensure_node_artifact_slot(cfg: Config, node: Node) -> Path:
    if node.artifact_dir_name is not None:
        artifact_dir = _node_artifact_dir(cfg, node)
        artifact_dir.mkdir(parents=True, exist_ok=True)
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
        return artifact_dir


def make_parallel_root_job(
    *,
    cfg: Config,
    reservation: HypothesisRootReservation,
    launched_index: int,
) -> ParallelRootJob:
    node_ctime, artifact_dir_name, artifact_dir = allocate_node_artifact_slot(
        cfg.log_dir,
        step=reservation.completed_steps,
    )
    return ParallelRootJob(
        reservation=reservation,
        node_ctime=node_ctime,
        artifact_dir_name=artifact_dir_name,
        artifact_dir=artifact_dir,
        launched_index=launched_index,
    )


def generate_reserved_hypothesis_root(
    *,
    base_agent: Agent,
    journal: Journal,
    job: ParallelRootJob,
) -> ParallelRootResult:
    worker_agent = Agent(
        task_desc=base_agent.task_desc,
        cfg=base_agent.cfg,
        journal=Journal(nodes=list(journal.nodes)),
    )
    worker_agent.data_preview = base_agent.data_preview
    if worker_agent.data_preview is None:
        worker_agent.update_data_preview()
    node = worker_agent.generate_preselected_hypothesis_root(
        job.reservation.selection,
        node_ctime=job.node_ctime,
        llm_log_dir=job.artifact_dir,
        artifact_dir_name=job.artifact_dir_name,
    )
    return ParallelRootResult(job=job, node=node)


def enforce_journal_submission_contract(
    cfg,
    journal: Journal,
    *,
    force_check_submissions: bool = False,
) -> int:
    sample_path = find_sample_submission(Path(cfg.workspace_dir) / "input")
    if sample_path is None:
        return 0

    changed = 0
    for node in journal.nodes:
        if node.status != "generated":
            continue
        if node.is_buggy or node.exc_type is not None or node._term_out:
            mark_node_generated_only(node)
            changed += 1

    nodes_to_check = [
        node
        for node in journal.nodes
        if node.status != "generated"
        if not _is_seeded_scored_root_node(node)
        if not node.is_buggy or node.is_submission_contract_error
    ]
    for node in nodes_to_check:
        submission_path = _node_artifact_submission_path(cfg, node)
        if not submission_path.exists():
            error = "missing artifact submission.csv while sample_submission exists"
            if _mark_node_submission_bug(node, error):
                changed += 1
            continue

        if (
            not force_check_submissions
            and _submission_validation_cache_matches(
                node,
                sample_path=sample_path,
                submission_path=submission_path,
            )
        ):
            continue

        error = validate_submission_file(submission_path, sample_path)
        if error is None:
            if _mark_node_submission_ok(
                node,
                sample_path=sample_path,
                submission_path=submission_path,
            ):
                changed += 1
        else:
            if _mark_node_submission_bug(
                node,
                error,
                sample_path=sample_path,
                submission_path=submission_path,
            ):
                changed += 1
    return changed


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return ""
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    if minutes:
        return f" ({minutes}m {seconds:02d}s)"
    return f" ({seconds}s)"


def stage_status_message(
    active_stage: str | None,
    elapsed: float | None = None,
    *,
    agent_mode: str | None = None,
    active_artifact_dir: Path | None = None,
    active_hypothesis_id: str | None = None,
    active_hypothesis_ids: list[str] | None = None,
) -> str:
    elapsed_text = _format_elapsed(elapsed)
    if active_hypothesis_ids:
        hypothesis_text = " @ " + ", ".join(active_hypothesis_ids)
    else:
        hypothesis_text = (
            f" @ {active_hypothesis_id}" if active_hypothesis_id is not None else ""
        )
    if active_stage == "generating":
        return f"[green]Generating code{hypothesis_text}...{elapsed_text}"
    if active_stage == "executing":
        if agent_mode == AGENT_MODE:
            autogluon_log = (
                active_artifact_dir / "autogluon_stdout.log"
                if active_artifact_dir is not None
                else None
            )
            if autogluon_log is not None and autogluon_log.exists():
                return f"[magenta]Training AutoGluon...{elapsed_text}"
            return f"[magenta]Preprocessing features...{elapsed_text}"
        return f"[magenta]Executing code...{elapsed_text}"
    if active_stage == "reviewing":
        return f"[cyan]Reviewing result...{elapsed_text}"
    return f"[green]Generating code{hypothesis_text}...{elapsed_text}"


KeyboardInterruptAction = Literal["continue", "abort"]
LIVE_REFRESH_INTERVAL_SECONDS = 1.0


def run_with_live_refresh(
    live: Live,
    render,
    func,
    tick=None,
    on_keyboard_interrupt: Callable[[], KeyboardInterruptAction] | None = None,
):
    result_queue = queue.Queue(maxsize=1)

    def worker():
        try:
            result_queue.put((True, func()))
        except BaseException as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        try:
            if tick is not None:
                tick()
            live.update(render(), refresh=True)
            thread.join(timeout=LIVE_REFRESH_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            if on_keyboard_interrupt is None:
                raise
            if on_keyboard_interrupt() == "abort":
                raise ExecutionInterrupted("Execution interrupted by user.") from None
    try:
        if tick is not None:
            tick()
        live.update(render(), refresh=True)
    except KeyboardInterrupt:
        if on_keyboard_interrupt is None or on_keyboard_interrupt() == "abort":
            raise

    ok, result = result_queue.get()
    if ok:
        return result
    raise result


def run(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    resume_request, runtime_options, cli_args = parse_runtime_args(raw_argv)
    if runtime_options.telegram_test_message:
        send_telegram_test_message()

    seed_source: SeedArtifactSource | None = None
    if resume_request.requested:
        base_cfg = _load_cfg(cli_args=cli_args)
        top_log_dir = Path(base_cfg.log_dir).resolve()
        top_workspace_dir = Path(base_cfg.workspace_dir).resolve()
        run_id = resume_request.run_id or find_latest_run_id(top_log_dir)
        cfg, journal = load_resume_state(
            run_id=run_id,
            top_log_dir=top_log_dir,
            top_workspace_dir=top_workspace_dir,
            cli_overrides=cli_args,
            force_check_submissions=runtime_options.force_check_submissions,
        )
        if resume_request.use_latest and not confirm_resume_latest(
            run_id,
            completed_steps=len(journal),
            total_steps=cfg.agent.steps,
        ):
            print("Resume cancelled.")
            return
        is_resume = True
    else:
        cfg = load_cfg(cli_args=cli_args)
        if runtime_options.seed_sha_prefix is not None:
            seed_source = find_seed_artifact(
                Path(cfg.log_dir).parent,
                runtime_options.seed_sha_prefix,
                source_run=runtime_options.seed_source_run,
            )
            if not _cli_sets_key(cli_args, "agent.search.num_drafts"):
                cfg.agent.search.num_drafts = 1
            if source_is_autogluon(seed_source) and not _cli_sets_key(
                cli_args,
                "agent.mode",
            ):
                cfg.agent.mode = "autogluon_preprocess"
        journal = Journal()
        is_resume = False

    if (
        runtime_options.public_tree_scores
        and not _cli_sets_key(cli_args, "agent.search.public_score_bonus_weight")
        and float(getattr(cfg.agent.search, "public_score_bonus_weight", 0.0)) <= 0.0
    ):
        cfg.agent.search.public_score_bonus_weight = (
            DEFAULT_PUBLIC_TREE_SCORE_BONUS_WEIGHT
        )

    apply_runtime_web_options(cfg, runtime_options)
    hypothesis_root_generate_workers = validate_hypothesis_root_generate_workers(cfg)

    logger.info(f'Starting run "{cfg.exp_name}"')
    os.environ["AIDE_RUN_ID"] = cfg.exp_name
    os.environ["AIDE_LOG_DIR"] = str(cfg.log_dir)
    debug_logger = MemoryDebugLogger(
        enabled=runtime_options.debug,
        run_id=cfg.exp_name,
    )
    debug_logger.log(
        "run_start",
        phase="startup",
        extra={
            "resume": is_resume,
            "run_id": cfg.exp_name,
            "log_dir": str(cfg.log_dir),
            "workspace_dir": str(cfg.workspace_dir),
            "cli_overrides": cli_args,
        },
    )

    task_desc = load_task_desc(cfg)

    if not is_resume:
        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(cfg)
        if seed_source is not None:
            with Status("Seeding run from existing artifact ..."):
                journal, _seed_node, _seed_artifact_dir = seed_journal_from_artifact(
                    cfg,
                    seed_source,
                )
                save_run(cfg, journal)
        else:
            with Status("Seeding run from scored hypothesis roots ..."):
                seeded_count = maybe_seed_scored_hypothesis_roots(
                    cfg,
                    journal,
                    is_resume=is_resume,
                )
                if seeded_count:
                    save_run(cfg, journal)

    def cleanup():
        if should_cleanup_workspace_on_exit(is_resume=is_resume, journal=journal):
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    recovered_generated_roots = recover_generated_only_root_artifacts(
        cfg=cfg,
        journal=journal,
        agent=agent,
    )
    if recovered_generated_roots:
        with Status(
            f"Recovered {recovered_generated_roots} generated root artifacts ..."
        ):
            save_run(cfg, journal)
    research_advisor = ResearchAdvisor(cfg=cfg, task_desc=task_desc)
    synthesis_advisor = SynthesisAdvisor(cfg=cfg, task_desc=task_desc)
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),  # type: ignore
    )

    def _node_debug_payload(node: Node | None) -> dict | None:
        if node is None:
            return None
        metric = getattr(node, "metric", None)
        return {
            "id": node.id,
            "step": node.step,
            "status": node.status,
            "is_buggy": node.is_buggy,
            "metric": getattr(metric, "value", None),
            "metric_maximize": getattr(metric, "maximize", None),
            "parent_id": node.parent.id if node.parent is not None else None,
            "parent_step": node.parent.step if node.parent is not None else None,
            "ctime": node.ctime,
        }

    def _resource_snapshot_payload(snapshot: ResourceSnapshot) -> dict:
        return {
            "cpu_percent": snapshot.cpu_percent,
            "ram_bytes": snapshot.ram_bytes,
            "peak_ram_bytes": snapshot.peak_ram_bytes,
            "process_count": snapshot.process_count,
            "gpu_percent": snapshot.gpu_percent,
            "gpu_memory_used_bytes": snapshot.gpu_memory_used_bytes,
            "gpu_memory_total_bytes": snapshot.gpu_memory_total_bytes,
            "gpu_power_draw_watts": snapshot.gpu_power_draw_watts,
            "gpu_power_limit_watts": snapshot.gpu_power_limit_watts,
            "gpu_temperature_celsius": snapshot.gpu_temperature_celsius,
        }

    def debug_log(
        event: str,
        *,
        phase: str | None = None,
        node: Node | None = None,
        extra: dict | None = None,
    ) -> None:
        payload = dict(extra or {})
        node_payload = _node_debug_payload(node)
        if node_payload is not None:
            payload["node"] = node_payload
        process = getattr(interpreter, "process", None)
        if process is not None:
            try:
                is_alive = process.is_alive() if getattr(process, "pid", None) else None
            except Exception:  # noqa: BLE001 - debug logging must not stop the run
                is_alive = None
            payload["interpreter_process"] = {
                "pid": getattr(process, "pid", None),
                "exitcode": getattr(process, "exitcode", None),
                "is_alive": is_alive,
            }
        debug_logger.log(event, phase=phase, extra=payload)

    debug_log("interpreter_ready", phase="startup")

    global_step = len(journal)
    generated_only_evaluations = 0

    def completed_work_units() -> int:
        return len(journal) + generated_only_evaluations

    prog = Progress(
        TextColumn(f"[{TUI_ROW_LABEL_STYLE}]" + "{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=completed_work_units())
    status_override: str | None = None
    operator_notice: str | None = None
    public_scores_notice: str | None = None
    public_scores_notice_until: float | None = None
    stop_after_current_node = False
    execution_interrupt_count = 0
    resource_history = ResourceHistory(
        window_seconds=DEFAULT_RESOURCE_HISTORY_WINDOW_SECONDS,
        interval_seconds=1,
    )
    resource_active = False
    focused_tree_item_id = "header"
    focused_tree_item_index = 0
    tree_scroll_top = 0
    tree_follow_mode = "off"
    search_debug_visible = False
    left_panel_view: LeftPanelView = "tree"
    focused_table_item_id = "header"
    focused_table_item_index = 0
    table_scroll_top = 0
    key_reader: ArrowKeyReader | None = None
    pending_copy_action: str | None = None
    copy_notice: Panel | None = None
    copy_notice_until: float | None = None
    pending_artifact_dir: Path | None = None
    display_node: Node | None = None
    active_root_generations: list[ActiveRootGeneration] = []
    web_state = WebDashboardState()
    web_server: AideWebServer | None = None
    if cfg.web.enabled:
        try:
            web_server = AideWebServer(
                web_state,
                host=str(cfg.web.host),
                port=int(cfg.web.port),
                refresh_seconds=float(cfg.web.refresh_seconds),
            )
            web_server.start()
            print(f"AIDE web dashboard: http://{cfg.web.host}:{web_server.port}/")
        except OSError as exc:
            logger.warning(f"Web dashboard disabled: {exc}")
            web_server = None

    def request_execution_interrupt() -> KeyboardInterruptAction:
        nonlocal execution_interrupt_count, operator_notice, status_override
        nonlocal stop_after_current_node
        execution_interrupt_count += 1
        if execution_interrupt_count == 1:
            stop_after_current_node = True
            operator_notice = (
                "Ctrl+C received. Waiting for current code to finish. "
                "The node will be reviewed and saved, then the run will stop. "
                "Press Ctrl+C again to stop now."
            )
            return "continue"
        status_override = "[red]Stopping current code execution..."
        interpreter.interrupt_execution()
        return "abort"

    def exec_callback(*args, **kwargs):
        nonlocal status_override, stop_after_current_node, resource_active

        def on_resource(snapshot: ResourceSnapshot):
            resource_history.add(snapshot)
            debug_log(
                "execution_resource_sample",
                phase="execute",
                node=agent.active_node,
                extra={"resource_snapshot": _resource_snapshot_payload(snapshot)},
            )

        debug_log("before_interpreter_run", phase="execute", node=agent.active_node)
        try:
            resource_active = True
            result = interpreter.run(
                *args,
                interrupt_callback=lambda _count: request_execution_interrupt(),
                resource_callback=on_resource,
                **kwargs,
            )
            debug_log(
                "after_interpreter_run",
                phase="execute",
                node=agent.active_node,
                extra={
                    "exec_time": result.exec_time,
                    "exc_type": result.exc_type,
                    "term_out_parts": len(result.term_out or []),
                },
            )
            return result
        except BaseException as exc:
            debug_log(
                "interpreter_run_exception",
                phase="execute",
                node=agent.active_node,
                extra={
                    "exception_type": exc.__class__.__name__,
                    "exception": str(exc),
                },
            )
            raise
        finally:
            resource_active = False
            if not stop_after_current_node:
                status_override = None

    def prepare_node_artifact_env(node: Node) -> str | None:
        previous = os.environ.get("AIDE_NODE_ARTIFACT_DIR")
        artifact_dir = ensure_node_artifact_slot(cfg, node)
        os.environ["AIDE_NODE_ARTIFACT_DIR"] = str(artifact_dir)
        return previous

    def restore_node_artifact_env(previous: str | None) -> None:
        if previous is None:
            os.environ.pop("AIDE_NODE_ARTIFACT_DIR", None)
        else:
            os.environ["AIDE_NODE_ARTIFACT_DIR"] = previous

    def update_save_status(message: str, live: Live) -> None:
        nonlocal status_override
        status_override = f"[blue]Saving run: {message}..."
        live.update(generate_live(), refresh=True)

    def tree_viewport_height() -> int:
        return max(1, shutil.get_terminal_size((120, 40)).lines - 4)

    def run_log_dimensions() -> tuple[int, int]:
        terminal_size = shutil.get_terminal_size((120, 40))
        right_column_width = max(20, int(terminal_size.columns * 2 / 5))
        right_column_height = max(6, terminal_size.lines - 2)
        log_panel_height = max(3, right_column_height // 3)
        content_height = max(1, log_panel_height - 2)
        content_width = max(10, right_column_width - 4)
        return content_height, content_width

    def resource_graph_width() -> int:
        terminal_size = shutil.get_terminal_size((120, 40))
        right_column_width = max(20, int(terminal_size.columns * 2 / 5))
        content_width = max(10, right_column_width - 4)
        fixed_width_before_spark = 26
        right_margin = 2
        return max(0, content_width - fixed_width_before_spark - right_margin)

    def is_panel_copy_key(key: str | None) -> bool:
        return key in {
            "copy_aide_panel",
            "copy_run_data_panel",
            "copy_logs_panel",
        }

    def public_score_bonus_enabled() -> bool:
        return (
            runtime_options.public_tree_scores
            and float(getattr(cfg.agent.search, "public_score_bonus_weight", 0.0)) > 0.0
            and float(getattr(cfg.agent.search, "public_score_bonus_cap", 0.0)) > 0.0
        )

    def refresh_public_scores() -> None:
        nonlocal public_scores_notice, public_scores_notice_until
        if not public_score_bonus_enabled():
            public_scores_notice = "Public scores disabled"
            public_scores_notice_until = time.monotonic() + 4.0
            return
        scores = load_public_scores_by_node_id(cfg.log_dir)
        agent.public_scores_by_node_id = scores
        public_scores_notice = f"Public scores refreshed: {len(scores)} nodes"
        public_scores_notice_until = time.monotonic() + 4.0

    if runtime_options.public_tree_scores:
        refresh_public_scores()

    def dismiss_overlays() -> None:
        nonlocal copy_notice, copy_notice_until, search_debug_visible
        nonlocal public_scores_notice, public_scores_notice_until
        copy_notice = None
        copy_notice_until = None
        public_scores_notice = None
        public_scores_notice_until = None
        search_debug_visible = False

    def drain_left_panel_navigation(view: TreeView) -> None:
        nonlocal focused_tree_item_id, focused_tree_item_index, tree_scroll_top
        nonlocal focused_table_item_id, focused_table_item_index, table_scroll_top
        nonlocal tree_follow_mode, left_panel_view, search_debug_visible
        nonlocal pending_copy_action
        if left_panel_view != "tree":
            if focused_table_item_id not in view.index_by_id:
                focused_table_item_id = recover_tree_focus_by_index(
                    view,
                    fallback_index=focused_table_item_index,
                )
            while key_reader is not None:
                key = key_reader.read_key()
                if key is None:
                    break
                if is_panel_copy_key(key):
                    pending_copy_action = key
                    return
                if key == "dismiss_overlay":
                    dismiss_overlays()
                    return
                if key == "view":
                    left_panel_view = next_left_panel_view(left_panel_view)
                    focused_table_item_id = "header"
                    focused_table_item_index = 0
                    table_scroll_top = 0
                    return
                if key == "debug":
                    search_debug_visible = not search_debug_visible
                    return
                if key == "public":
                    refresh_public_scores()
                    return
                if key in {"up", "down"}:
                    focused_table_item_id = move_tree_focus(
                        view,
                        focused_table_item_id,
                        key,
                    )
            focus_index = view.index_by_id.get(focused_table_item_id, 0)
            table_scroll_top = clamp_tree_viewport(
                total_lines=len(view.items),
                viewport_height=tree_viewport_height(),
                focus_index=focus_index,
                current_scroll=table_scroll_top,
            )
            focused_table_item_index = view.index_by_id.get(focused_table_item_id, 0)
            return

        if tree_follow_mode == "active":
            active_id = active_tree_item_id(view)
            if active_id is not None:
                focused_tree_item_id = active_id
                tree_scroll_top = center_tree_viewport(
                    total_lines=len(view.items),
                    viewport_height=tree_viewport_height(),
                    focus_index=view.index_by_id[active_id],
                )
        if focused_tree_item_id not in view.index_by_id:
            focused_tree_item_id = recover_tree_focus_by_index(
                view,
                fallback_index=focused_tree_item_index,
            )
        while key_reader is not None:
            key = key_reader.read_key()
            if key is None:
                break
            if is_panel_copy_key(key):
                pending_copy_action = key
                return
            if key == "dismiss_overlay":
                dismiss_overlays()
                return
            if key == "view":
                left_panel_view = next_left_panel_view(left_panel_view)
                focused_table_item_id = "header"
                focused_table_item_index = 0
                table_scroll_top = 0
                return
            if key == "debug":
                search_debug_visible = not search_debug_visible
                return
            if key == "public":
                refresh_public_scores()
                return
            if key == "follow":
                tree_follow_mode = (
                    "active" if tree_follow_mode == "off" else "off"
                )
                if tree_follow_mode == "active":
                    target_id = active_tree_item_id(view)
                    if target_id is not None:
                        focused_tree_item_id = target_id
                        tree_scroll_top = center_tree_viewport(
                            total_lines=len(view.items),
                            viewport_height=tree_viewport_height(),
                            focus_index=view.index_by_id[target_id],
                        )
                continue
            if key == "best":
                target_id = best_tree_item_id(
                    view,
                    journal,
                    show_invalid_submission_branches=(
                        runtime_options.show_invalid_submission_branches
                    ),
                    disable_oom_saturated_parents=(
                        cfg.agent.search.disable_oom_saturated_parents
                    ),
                )
                if target_id is not None:
                    focused_tree_item_id = target_id
                    tree_scroll_top = center_tree_viewport(
                        total_lines=len(view.items),
                        viewport_height=tree_viewport_height(),
                        focus_index=view.index_by_id[target_id],
                    )
                continue
            if key == "active":
                target_id = active_tree_item_id(view)
                if target_id is not None:
                    focused_tree_item_id = target_id
                    tree_scroll_top = center_tree_viewport(
                        total_lines=len(view.items),
                        viewport_height=tree_viewport_height(),
                        focus_index=view.index_by_id[target_id],
                    )
                continue
            focused_tree_item_id = move_tree_focus(view, focused_tree_item_id, key)
        focus_index = view.index_by_id.get(focused_tree_item_id, 0)
        tree_scroll_top = clamp_tree_viewport(
            total_lines=len(view.items),
            viewport_height=tree_viewport_height(),
            focus_index=focus_index,
            current_scroll=tree_scroll_top,
        )
        focused_tree_item_index = view.index_by_id.get(focused_tree_item_id, 0)

    def active_hypothesis_id_for_display() -> str | None:
        if agent.active_research_hypothesis_id is not None:
            return agent.active_research_hypothesis_id
        for node in (agent.active_node, display_node, agent.active_parent_node):
            if node is None:
                continue
            hypothesis_id = hypothesis_id_for_node(node)
            if hypothesis_id is not None:
                return hypothesis_id
        return None

    def active_root_hypothesis_ids_for_display() -> list[str]:
        return [
            generation.hypothesis_id
            for generation in sorted(
                active_root_generations,
                key=lambda item: item.launched_index,
            )
        ]

    def _plain_web_text(value: object) -> str:
        text = str(value or "")
        text = re.sub(r"\[[^\]]+\]", "", text)
        return text.replace("[/]", "").strip()

    def _web_run_sections(
        *,
        status_text: str,
        active_artifact_dir: Path | None,
    ) -> list[WebRunSection]:
        overview = [
            WebRunDatum("progress", f"{completed_work_units()}/{cfg.agent.steps}"),
            WebRunDatum("status", _plain_web_text(status_text)),
        ]
        best = _best_scored_node(journal)
        if best is not None and best.metric is not None and best.metric.value is not None:
            step = best.step if best.step is not None else "?"
            timestamp = dt.datetime.fromtimestamp(best.ctime).strftime("%m-%d %H:%M")
            overview.append(
                WebRunDatum("best score", f"{step} @ {timestamp} {best.metric.value:.5f}")
            )

        sections = [
            WebRunSection("Run", overview),
            WebRunSection(
                "Models",
                [
                    WebRunDatum(label, f"{model} - {effort or '-'}")
                    for label, model, effort in model_settings_for_run(cfg)
                ],
            ),
            WebRunSection(
                "Agent",
                [
                    WebRunDatum("mode", str(cfg.agent.mode)),
                    WebRunDatum("aux", "on" if aux_mode(cfg) else "off"),
                    WebRunDatum("gpu", str(cfg.agent.gpu).lower()),
                    WebRunDatum(
                        "run",
                        "generate only" if runtime_options.skip_execution else "execute",
                    ),
                ],
            ),
            WebRunSection(
                "Paths",
                [
                    WebRunDatum("log", str(cfg.log_dir)),
                    WebRunDatum("workspace", str(cfg.workspace_dir)),
                ],
            ),
        ]
        if active_artifact_dir is not None:
            sections[-1].items.append(WebRunDatum("artifact", str(active_artifact_dir)))

        error = last_error_record(journal)
        if error is not None:
            step = error.node.step if error.node.step is not None else "?"
            sections.append(
                WebRunSection(
                    "Last Error",
                    [WebRunDatum(str(step), " / ".join(error.lines))],
                )
            )
        if operator_notice:
            sections.append(
                WebRunSection("Notice", [WebRunDatum("message", operator_notice)])
            )
        return [
            section
            for section in sections
            if section.items
        ]

    def _web_log_lines(active_artifact_dir: Path | None) -> list[str]:
        log_path = active_run_log_path(active_artifact_dir)
        if log_path is None:
            hint = (
                agent.active_research_hypothesis_log_hint
                if agent.active_stage == "generating"
                else None
            )
            if hint:
                return hint.splitlines()[:80]
            return ["waiting for process log"]
        lines = _tail_log_lines(log_path, max_lines=120)
        if not lines:
            return [f"{log_path.name} is empty"]
        return [_clip_log_line(line, max_width=180) for line in lines]

    def publish_web_snapshot(
        *,
        status_text: str,
        active_artifact_dir: Path | None,
    ) -> None:
        if web_server is None:
            return
        try:
            web_state.update(
                WebDashboardSnapshot(
                    run_id=cfg.exp_name,
                    refresh_seconds=float(cfg.web.refresh_seconds),
                    tree_title="Solution tree",
                    tree_lines=build_web_tree_lines(
                        journal,
                        active_parent_node=agent.active_parent_node,
                        active_stage=agent.active_stage,
                        active_hypothesis_id=active_hypothesis_id_for_display(),
                        plateau_block_epsilon=(
                            cfg.agent.search.plateau_block_epsilon
                        ),
                        public_scores_by_node_id=agent.public_scores_by_node_id,
                        public_score_bonus_weight=(
                            cfg.agent.search.public_score_bonus_weight
                        ),
                        public_score_bonus_cap=(
                            cfg.agent.search.public_score_bonus_cap
                        ),
                    ),
                    run_sections=_web_run_sections(
                        status_text=status_text,
                        active_artifact_dir=active_artifact_dir,
                    ),
                    log_lines=_web_log_lines(active_artifact_dir),
                    status=_plain_web_text(status_text),
                )
            )
        except Exception as exc:  # noqa: BLE001 - dashboard must not stop AIDE
            logger.debug("Failed to publish web dashboard snapshot: %s", exc)

    def current_tree_view(*, blink_on: bool) -> TreeView:
        return build_tree_view(
            journal,
            active_node=agent.active_node,
            active_parent_node=agent.active_parent_node,
            active_stage=agent.active_stage,
            active_hypothesis_id=active_hypothesis_id_for_display(),
            active_root_generations=active_root_generations,
            blink_on=blink_on,
            show_invalid_submission_branches=(
                runtime_options.show_invalid_submission_branches
            ),
            disable_oom_saturated_parents=(
                cfg.agent.search.disable_oom_saturated_parents
            ),
            plateau_block_epsilon=cfg.agent.search.plateau_block_epsilon,
            synthesis_node_ids=synthesis_injected_node_ids(cfg.log_dir),
            public_scores_by_node_id=agent.public_scores_by_node_id,
            public_score_bonus_weight=cfg.agent.search.public_score_bonus_weight,
            public_score_bonus_cap=cfg.agent.search.public_score_bonus_cap,
        )

    def current_left_panel_view(*, blink_on: bool) -> TreeView:
        if left_panel_view == "tree":
            return current_tree_view(blink_on=blink_on)
        if left_panel_view == "root":
            return build_root_hypotheses_view(
                journal,
                hypothesis_mode_labels=_hypothesis_mode_labels(cfg),
            )
        if left_panel_view == "all":
            return build_all_hypotheses_view(journal)
        return build_best_branch_view(journal)

    def handle_pending_panel_copy(
        *,
        left_panel_content,
        data_panel_content,
        log_panel_content,
        left_width: int,
        right_width: int,
    ) -> None:
        nonlocal pending_copy_action, copy_notice, copy_notice_until
        if pending_copy_action is None:
            return

        action = pending_copy_action
        pending_copy_action = None
        if action == "copy_aide_panel":
            panel_name = "aide"
            panel_title = f'AIDE: "{cfg.exp_name}"'
            renderable = left_panel_content
            width = left_width
        elif action == "copy_run_data_panel":
            panel_name = "run-data"
            panel_title = "Run data"
            renderable = data_panel_content
            width = right_width
        elif action == "copy_logs_panel":
            panel_name = "logs"
            panel_title = "Logs"
            renderable = log_panel_content
            width = right_width
        else:
            return

        text = render_panel_copy_text(panel_title, renderable, width=width)
        path = save_panel_copy(panel_name, cfg.exp_name, text)
        osc52_sent = copy_text_to_clipboard_osc52(text)
        copy_notice = build_panel_copy_notice(
            panel_title=panel_title,
            path=path,
            osc52_sent=osc52_sent,
        )
        copy_notice_until = time.monotonic() + 6.0

    def generate_live():
        nonlocal copy_notice, copy_notice_until
        nonlocal public_scores_notice, public_scores_notice_until
        blink_on = int(time.monotonic() * 2) % 2 == 0
        if (
            public_scores_notice is not None
            and public_scores_notice_until is not None
            and time.monotonic() >= public_scores_notice_until
        ):
            public_scores_notice = None
            public_scores_notice_until = None
        rendered_panel_view = left_panel_view
        left_view = current_left_panel_view(blink_on=blink_on)
        drain_left_panel_navigation(left_view)
        if left_panel_view != rendered_panel_view:
            left_view = current_left_panel_view(blink_on=blink_on)
        if left_panel_view == "tree":
            focused_item_id = focused_tree_item_id
            scroll_top = tree_scroll_top
        else:
            focused_item_id = focused_table_item_id
            scroll_top = table_scroll_top
        left_panel_content = render_tree_view(
            left_view,
            focused_item_id=focused_item_id,
            scroll_top=scroll_top,
            viewport_height=tree_viewport_height(),
        )
        prog.update(prog.task_ids[0], completed=completed_work_units())
        elapsed = (
            time.monotonic() - agent.active_stage_started_at
            if agent.active_stage_started_at is not None
            else None
        )
        active_artifact_dir = (
            _node_artifact_dir(cfg, agent.active_node)
            if agent.active_node is not None
            else pending_artifact_dir
        )
        status_text = status_override or stage_status_message(
                agent.active_stage,
                elapsed,
                agent_mode=cfg.agent.mode,
                active_artifact_dir=active_artifact_dir,
                active_hypothesis_id=active_hypothesis_id_for_display(),
                active_hypothesis_ids=active_root_hypothesis_ids_for_display(),
            )
        status.update(status_text)
        publish_web_snapshot(
            status_text=status_text,
            active_artifact_dir=active_artifact_dir,
        )
        terminal_size = shutil.get_terminal_size((120, 40))
        left_copy_width = max(20, int(terminal_size.columns * 3 / 5) - 4)
        right_copy_width = max(20, int(terminal_size.columns * 2 / 5) - 4)
        data_panel_content = Padding(
            build_run_data(
                progress=prog,
                status=status,
                research_status=(
                    research_advisor.status_text() if cfg.research.enabled else None
                ),
                synthesis_status=(
                    synthesis_advisor.status_text()
                    if cfg.synthesis.enabled
                    else None
                ),
                journal=journal,
                log_dir=cfg.log_dir,
                workspace_dir=cfg.workspace_dir,
                resource_history=resource_history,
                resource_active=resource_active,
                resource_graph_width=resource_graph_width(),
                model_settings=model_settings_for_run(cfg),
                active_artifact_dir=active_artifact_dir,
                cfg=cfg,
                operator_notice=operator_notice or public_scores_notice,
                include_generated_hypothesis_roots=runtime_options.skip_execution,
                skip_execution=runtime_options.skip_execution,
                hypothesis_root_generate_workers=hypothesis_root_generate_workers,
                selected_hypothesis_ids=runtime_options.generate_only_hypothesis_ids,
            ),
            (0, 1, 0, 1),
        )
        log_line_count, log_width = run_log_dimensions()
        log_panel_content = Padding(
            build_run_log_summary(
                active_artifact_dir,
                max_lines=log_line_count,
                max_width=log_width,
                missing_log_hint=(
                    agent.active_research_hypothesis_log_hint
                    if agent.active_stage == "generating"
                    else None
                ),
            ),
            (0, 1, 0, 1),
        )
        handle_pending_panel_copy(
            left_panel_content=left_panel_content,
            data_panel_content=data_panel_content,
            log_panel_content=log_panel_content,
            left_width=left_copy_width,
            right_width=right_copy_width,
        )

        tree_panel = Panel(
            Padding(left_panel_content, (0, 1, 0, 1)),
            title=f'[b]AIDE (1): [bold green]"{cfg.exp_name}[/b]"',
            subtitle=(
                "↑/↓ move  ← parent  → child  b best  a active  "
                f"f follow:{tree_follow_mode}  v view:{left_panel_view}  "
                f"d debug:{'on' if search_debug_visible else 'off'}  "
                f"{'p public  ' if public_score_bonus_enabled() else ''}"
                "1/2/3 copy  Ctrl+C stop"
            ),
        )
        data_panel = Panel(
            data_panel_content,
            title="[b]Run data (2)",
        )
        log_panel = Panel(
            log_panel_content,
            title="[b]Logs (3)",
        )

        layout = Layout()
        data_layout = Layout(name="data", ratio=2)
        data_layout.split_column(
            Layout(data_panel, name="run_data", ratio=2),
            Layout(log_panel, name="logs", ratio=1),
        )
        layout.split_row(
            Layout(tree_panel, name="tree", ratio=3),
            data_layout,
        )
        renderable = layout

        if search_debug_visible:
            overlay_width = min(max(80, terminal_size.columns - 8), terminal_size.columns)
            debug_panel = Panel(
                Padding(build_search_decision_debug_view(agent.last_search_decision), (0, 1)),
                title="[b]Search decision debug",
                subtitle=f"{Path(cfg.log_dir) / 'search_decisions.jsonl'}",
                border_style="yellow",
            )
            renderable = _Overlay(
                renderable,
                debug_panel,
                overlay_width=overlay_width,
                edge_margin=3,
                dim=True,
            )

        if copy_notice is not None and copy_notice_until is not None:
            if time.monotonic() >= copy_notice_until:
                copy_notice = None
                copy_notice_until = None
            else:
                notice_width = min(max(72, terminal_size.columns - 16), terminal_size.columns)
                renderable = _Overlay(
                    renderable,
                    copy_notice,
                    overlay_width=notice_width,
                    edge_margin=3,
                    dim=True,
                )

        return renderable

    def parallel_root_workers_enabled() -> bool:
        return (
            runtime_options.skip_execution
            and cfg.research.mode == "hypothesis"
            and (
                hypothesis_root_generate_workers > 1
                or bool(runtime_options.generate_only_hypothesis_ids)
            )
        )

    def refresh_active_root_generations(
        futures: dict[Future[ParallelRootResult], ParallelRootJob],
    ) -> None:
        nonlocal active_root_generations, pending_artifact_dir
        active_root_generations = [
            ActiveRootGeneration(
                hypothesis_id=job.reservation.hypothesis_id,
                launched_index=job.launched_index,
            )
            for job in futures.values()
        ]
        latest_job = max(
            futures.values(),
            key=lambda job: job.launched_index,
            default=None,
        )
        pending_artifact_dir = latest_job.artifact_dir if latest_job else None

    def run_parallel_generate_only_roots(live: Live) -> None:
        nonlocal display_node, operator_notice, pending_artifact_dir
        agent.set_active_stage("generating")
        failure_state = ParallelRootFailureState()
        launched_counter = 0
        exhausted = False

        def sleep_with_live_refresh(seconds: int) -> None:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                live.update(generate_live(), refresh=True)
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

        def launch_job(
            executor: ThreadPoolExecutor,
            futures: dict[Future[ParallelRootResult], ParallelRootJob],
            reservation: HypothesisRootReservation,
        ) -> None:
            nonlocal launched_counter
            launched_counter += 1
            job = make_parallel_root_job(
                cfg=cfg,
                reservation=reservation,
                launched_index=launched_counter,
            )
            future = executor.submit(
                generate_reserved_hypothesis_root,
                base_agent=agent,
                journal=journal,
                job=job,
            )
            futures[future] = job
            refresh_active_root_generations(futures)

        def launch_until_full(
            executor: ThreadPoolExecutor,
            futures: dict[Future[ParallelRootResult], ParallelRootJob],
        ) -> int:
            nonlocal exhausted
            if failure_state.stop_refill or exhausted:
                return 0
            remaining_steps = cfg.agent.steps - len(journal) - len(futures)
            slots = min(hypothesis_root_generate_workers - len(futures), remaining_steps)
            if slots <= 0:
                return 0
            reservations = reserve_hypothesis_roots(
                cfg,
                journal=journal,
                count=slots,
                completed_steps=len(journal) + len(futures),
                reserved_hypothesis_ids={
                    job.reservation.hypothesis_id for job in futures.values()
                },
                forced_hypothesis_ids=runtime_options.generate_only_hypothesis_ids,
            )
            if not reservations:
                exhausted = True
                return 0
            for reservation in reservations:
                launch_job(executor, futures, reservation)
            return len(reservations)

        with ThreadPoolExecutor(max_workers=hypothesis_root_generate_workers) as executor:
            futures: dict[Future[ParallelRootResult], ParallelRootJob] = {}
            launch_until_full(executor, futures)
            while futures:
                done, _pending = wait(
                    futures,
                    timeout=1.0,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    live.update(generate_live(), refresh=True)
                    continue
                for future in done:
                    job = futures.pop(future)
                    refresh_active_root_generations(futures)
                    try:
                        result = future.result()
                    except BaseException as exc:  # noqa: BLE001
                        is_final = failure_state.record_failure(
                            job.reservation.hypothesis_id,
                            exc,
                        )
                        attempts = failure_state.attempts_by_hypothesis[
                            job.reservation.hypothesis_id
                        ]
                        message = f"{exc.__class__.__name__}: {exc}"
                        record_hypothesis_root_generation_failure(
                            cfg,
                            hypothesis_id=job.reservation.hypothesis_id,
                            attempts=attempts,
                            message=message,
                        )
                        operator_notice = (
                            "Hypothesis root generation failed for "
                            f"{job.reservation.hypothesis_id} "
                            f"(attempt {attempts}/3): {message}"
                        )
                        if not is_final:
                            sleep_with_live_refresh(5)
                            launch_job(executor, futures, job.reservation)
                        continue

                    clear_hypothesis_root_generation_failure(
                        cfg,
                        hypothesis_id=result.job.reservation.hypothesis_id,
                    )
                    record_generated_only_node(
                        agent=agent,
                        journal=journal,
                        node=result.node,
                        experiment_id=cfg.exp_name,
                    )
                    display_node = result.node
                    launch_until_full(executor, futures)
                    refresh_active_root_generations(futures)
                live.update(generate_live(), refresh=True)
        refresh_active_root_generations({})
        agent.clear_active_step()

    interrupted = False
    interrupt_message = ""
    try:
        with ArrowKeyReader() as reader:
            key_reader = reader
            with Live(
                get_renderable=generate_live,
                refresh_per_second=1,
                screen=True,
            ) as live:
                while completed_work_units() < cfg.agent.steps:
                    synthesized: SynthesisNode | None = None
                    result_node: Node | None = None
                    node_already_in_journal = False
                    display_node = None
                    try:
                        if parallel_root_workers_enabled():
                            run_parallel_generate_only_roots(live)
                            interrupted = True
                            interrupt_message = save_parallel_generate_only_run(
                                cfg=cfg,
                                journal=journal,
                                current_node=display_node,
                                progress_callback=lambda message: update_save_status(
                                    message,
                                    live,
                                ),
                            )
                            break
                        pending_generated_node = (
                            None
                            if runtime_options.skip_execution
                            else next_generated_only_node(
                                journal,
                                forced_root=getattr(
                                    cfg.agent.search,
                                    "forced_root",
                                    None,
                                ),
                                forced_hypothesis=getattr(
                                    cfg.agent.search,
                                    "forced_hypothesis",
                                    None,
                                ),
                            )
                        )
                        if pending_generated_node is not None:
                            result_node = pending_generated_node
                            display_node = result_node
                            node_already_in_journal = True
                            agent.active_parent_node = result_node.parent
                            agent.last_search_decision = _runtime_generated_decision(
                                result_node,
                                journal_size=len(journal),
                            )
                            debug_log(
                                "resume_generated_only_node",
                                phase="execute",
                                node=result_node,
                                extra={"global_step": global_step},
                            )
                        elif cfg.synthesis.enabled and not runtime_options.skip_execution:
                            agent.active_parent_node = None
                            agent.set_active_stage("generating")

                            def maybe_generate_synthesis() -> SynthesisNode | None:
                                debug_log(
                                    "before_synthesis_generate",
                                    phase="generate",
                                    extra={
                                        "completed_steps": count_scored_working_nodes(
                                            journal
                                        )
                                    },
                                )
                                try:
                                    result = synthesis_advisor.generate_node_if_due(
                                        journal=journal,
                                        completed_steps=count_scored_working_nodes(
                                            journal
                                        ),
                                    )
                                except BaseException as exc:
                                    debug_log(
                                        "synthesis_generate_exception",
                                        phase="generate",
                                        extra={
                                            "exception_type": exc.__class__.__name__,
                                            "exception": str(exc),
                                        },
                                    )
                                    raise
                                debug_log(
                                    "after_synthesis_generate",
                                    phase="generate",
                                    node=result.node if result is not None else None,
                                    extra={
                                        "completed_steps": (
                                            result.completed_steps
                                            if result is not None
                                            else None
                                        ),
                                        "checkpoint_dir": (
                                            str(result.checkpoint_dir)
                                            if result is not None
                                            else None
                                        ),
                                        "ready_for_execution": (
                                            result.ready_for_execution
                                            if result is not None
                                            else None
                                        ),
                                    },
                                )
                                return result

                            synthesized = run_with_live_refresh(
                                live,
                                generate_live,
                                maybe_generate_synthesis,
                                tick=lambda: drain_left_panel_navigation(
                                    current_left_panel_view(blink_on=True)
                                ),
                            )
                            if synthesized is None:
                                agent.clear_active_step()

                        if result_node is None and synthesized is None:
                            debug_log(
                                "before_prepare_step",
                                phase="generate",
                                extra={"global_step": global_step},
                            )
                            parent_node = agent.prepare_step()
                            if runtime_options.skip_execution and parent_node is not None:
                                interrupted = True
                                interrupt_message = (
                                    "Skip-execution mode finished generating root "
                                    "candidates; no code was executed."
                                )
                                break
                            debug_log(
                                "after_prepare_step",
                                phase="generate",
                                node=parent_node,
                                extra={"global_step": global_step},
                            )
                            (
                                node_ctime,
                                artifact_dir_name,
                                pending_artifact_dir,
                            ) = allocate_node_artifact_slot(
                                cfg.log_dir,
                                step=global_step,
                            )

                            def generate_current_node() -> Node:
                                debug_log(
                                    "before_generate_node",
                                    phase="generate",
                                    node=parent_node,
                                    extra={
                                        "node_ctime": node_ctime,
                                        "llm_log_dir": str(pending_artifact_dir),
                                    },
                                )
                                try:
                                    node = agent.generate_node(
                                        parent_node,
                                        node_ctime=node_ctime,
                                        llm_log_dir=pending_artifact_dir,
                                    )
                                except BaseException as exc:
                                    debug_log(
                                        "generate_node_exception",
                                        phase="generate",
                                        node=parent_node,
                                        extra={
                                            "exception_type": exc.__class__.__name__,
                                            "exception": str(exc),
                                        },
                                    )
                                    raise
                                debug_log(
                                    "after_generate_node",
                                    phase="generate",
                                    node=node,
                                    extra={"llm_log_dir": str(pending_artifact_dir)},
                                )
                                return node

                            result_node = run_with_live_refresh(
                                live,
                                generate_live,
                                generate_current_node,
                                tick=lambda: drain_left_panel_navigation(
                                    current_left_panel_view(blink_on=True)
                                ),
                            )
                            if result_node.artifact_dir_name is None:
                                result_node.artifact_dir_name = artifact_dir_name
                            agent.save_hypothesis_root_code_for_node(
                                result_node,
                                activate=False,
                            )
                            display_node = result_node
                        elif result_node is None:
                            result_node = synthesized.node
                            display_node = result_node
                        pending_artifact_dir = None
                        assert result_node is not None

                        if runtime_options.skip_execution and not node_already_in_journal:
                            debug_log(
                                "before_append_generated_only_node",
                                phase="journal",
                                node=result_node,
                            )
                            record_generated_only_node(
                                agent=agent,
                                journal=journal,
                                node=result_node,
                                experiment_id=cfg.exp_name,
                            )
                            debug_log(
                                "after_append_generated_only_node",
                                phase="journal",
                                node=result_node,
                            )
                        elif (
                            synthesized is not None
                            and not synthesized.ready_for_execution
                        ):
                            debug_log(
                                "before_append_nonexecuted_synthesis_node",
                                phase="journal",
                                node=result_node,
                            )
                            append_node_with_best_score_notification(
                                journal=journal,
                                node=result_node,
                                experiment_id=cfg.exp_name,
                            )
                            debug_log(
                                "after_append_nonexecuted_synthesis_node",
                                phase="journal",
                                node=result_node,
                            )
                            synthesis_advisor.mark_recorded(
                                synthesized,
                                node=result_node,
                            )
                        else:
                            previous_artifact_env = prepare_node_artifact_env(
                                result_node
                            )
                            try:
                                try:
                                    debug_log(
                                        "before_execute_node",
                                        phase="execute",
                                        node=result_node,
                                    )
                                    exec_result = agent.execute_node(
                                        result_node,
                                        lambda *args, **kwargs: run_with_live_refresh(
                                            live,
                                            generate_live,
                                            lambda: exec_callback(*args, **kwargs),
                                            tick=lambda: drain_left_panel_navigation(
                                                current_left_panel_view(blink_on=True)
                                            ),
                                            on_keyboard_interrupt=request_execution_interrupt,
                                        ),
                                    )
                                    debug_log(
                                        "after_execute_node",
                                        phase="execute",
                                        node=result_node,
                                        extra={
                                            "exec_time": exec_result.exec_time,
                                            "exc_type": exec_result.exc_type,
                                        },
                                    )
                                except RuntimeError as exc:
                                    debug_log(
                                        "execute_node_runtime_error",
                                        phase="execute",
                                        node=result_node,
                                        extra={
                                            "exception_type": exc.__class__.__name__,
                                            "exception": str(exc),
                                        },
                                    )
                                    _mark_node_execution_crash(
                                        result_node,
                                        exc,
                                        artifact_dir=_node_artifact_dir(
                                            cfg,
                                            result_node,
                                        ),
                                    )
                                    debug_log(
                                        "after_mark_execution_crash",
                                        phase="execute",
                                        node=result_node,
                                    )
                                    debug_log(
                                        "before_append_crashed_node",
                                        phase="journal",
                                        node=result_node,
                                    )
                                    append_node_with_best_score_notification(
                                        journal=journal,
                                        node=result_node,
                                        experiment_id=cfg.exp_name,
                                    )
                                    debug_log(
                                        "after_append_crashed_node",
                                        phase="journal",
                                        node=result_node,
                                    )
                                    if synthesized is not None:
                                        synthesis_advisor.mark_injected(
                                            synthesized,
                                            node=result_node,
                                        )
                                    continue
                            finally:
                                restore_node_artifact_env(previous_artifact_env)

                            def review_current_node() -> None:
                                debug_log(
                                    "before_review_node",
                                    phase="review",
                                    node=result_node,
                                )
                                try:
                                    agent.review_node(result_node, exec_result)
                                except BaseException as exc:
                                    debug_log(
                                        "review_node_exception",
                                        phase="review",
                                        node=result_node,
                                        extra={
                                            "exception_type": exc.__class__.__name__,
                                            "exception": str(exc),
                                        },
                                    )
                                    raise
                                debug_log(
                                    "after_review_node",
                                    phase="review",
                                    node=result_node,
                                    extra={
                                        "exc_type": result_node.exc_type,
                                        "is_buggy": result_node.is_buggy,
                                    },
                                )

                            run_with_live_refresh(
                                live,
                                generate_live,
                                review_current_node,
                                tick=lambda: drain_left_panel_navigation(
                                    current_left_panel_view(blink_on=True)
                                ),
                            )
                            debug_log(
                                "before_submission_validation",
                                phase="submission_validation",
                                node=result_node,
                            )
                            submission_changed = enforce_submission_contract(
                                cfg,
                                result_node,
                            )
                            agent.save_hypothesis_root_code_for_node(result_node)
                            debug_log(
                                "after_submission_validation",
                                phase="submission_validation",
                                node=result_node,
                                extra={"changed": submission_changed},
                            )
                            debug_log(
                                "before_append_node",
                                phase="journal",
                                node=result_node,
                            )
                            if not node_already_in_journal:
                                append_node_with_best_score_notification(
                                    journal=journal,
                                    node=result_node,
                                    experiment_id=cfg.exp_name,
                                )
                            debug_log(
                                "after_append_node",
                                phase="journal",
                                node=result_node,
                                extra={"already_in_journal": node_already_in_journal},
                            )
                            if synthesized is not None:
                                synthesis_advisor.mark_injected(
                                    synthesized,
                                    node=result_node,
                                )
                    except ExecutionInterrupted:
                        interrupted = True
                        interrupt_message = (
                            "Execution stopped immediately by user. Current node was not saved; "
                            "previous journal state is preserved."
                        )
                        break
                    except KeyboardInterrupt:
                        interrupted = True
                        interrupt_message = (
                            "Run interrupted by user. Current node was not saved; "
                            "previous journal state is preserved."
                        )
                        break
                    finally:
                        agent.clear_active_step()
                        pending_artifact_dir = None
                    debug_log(
                        "before_save_run",
                        phase="save",
                        node=result_node,
                        extra={"global_step": global_step, "journal_size": len(journal)},
                    )
                    save_run(
                        cfg,
                        journal,
                        current_node=result_node,
                        progress_callback=lambda message: update_save_status(
                            message,
                            live,
                        ),
                    )
                    debug_log(
                        "after_save_run",
                        phase="save",
                        node=result_node,
                        extra={"global_step": global_step, "journal_size": len(journal)},
                    )
                    status_override = None
                    if node_already_in_journal:
                        generated_only_evaluations += 1
                    else:
                        global_step = len(journal)
                    if stop_after_current_node:
                        interrupted = True
                        interrupt_message = (
                            "Execution stopped by user after saving current node."
                        )
                        break
                    debug_log(
                        "before_research_maybe_start",
                        phase="research",
                        extra={"completed_steps": count_scored_working_nodes(journal)},
                    )
                    research_started = research_advisor.maybe_start(
                        journal=journal,
                        completed_steps=count_scored_working_nodes(journal),
                    )
                    debug_log(
                        "after_research_maybe_start",
                        phase="research",
                        extra={
                            "completed_steps": count_scored_working_nodes(journal),
                            "started": research_started,
                        },
                    )
                live.update(generate_live(), refresh=True)
    finally:
        debug_log("before_cleanup_session", phase="cleanup")
        interpreter.cleanup_session()
        if web_server is not None:
            web_server.stop()
        debug_log("after_cleanup_session", phase="cleanup")

    print_final_tree(
        journal,
        log_dir=cfg.log_dir,
        show_invalid_submission_branches=runtime_options.show_invalid_submission_branches,
        disable_oom_saturated_parents=cfg.agent.search.disable_oom_saturated_parents,
        plateau_block_epsilon=cfg.agent.search.plateau_block_epsilon,
        public_scores_by_node_id=agent.public_scores_by_node_id,
        public_score_bonus_weight=cfg.agent.search.public_score_bonus_weight,
        public_score_bonus_cap=cfg.agent.search.public_score_bonus_cap,
    )
    emit_completion_bell()

    if interrupted:
        print(interrupt_message)
        return

    if cfg.generate_report:
        print("Generating final report from journal...")
        report = journal2report(journal, task_desc, cfg.report, log_dir=cfg.log_dir)
        print(report)
        report_file_path = cfg.log_dir / "report.md"
        with open(report_file_path, "w") as f:
            f.write(report)
        print("Report written to file:", report_file_path)


if __name__ == "__main__":
    run()
