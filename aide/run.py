import atexit
import datetime as dt
import json
import logging
import os
import queue
import select
import shutil
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, cast

from .agent import Agent
from .interpreter import ExecutionInterrupted, Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from .research import ResearchAdvisor, count_scored_working_nodes
from .synthesis import SYNTHESIS_PLAN_PREFIX, SynthesisAdvisor, SynthesisNode
from .autogluon_preprocess import BASELINE_PLAN_PREFIX
from omegaconf import OmegaConf
from rich.console import Group
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
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils import serialize
from .utils.config import (
    Config,
    _load_cfg,
    _resolve_all_model_configs,
    _normalize_model_effort_cli_overrides,
    _validate_cli_model_effort_conflicts,
    load_task_desc,
    prep_agent_workspace,
    save_run,
    load_cfg,
)
from .utils.metric import MetricValue, WorstMetricValue
from .utils.resource_monitor import ResourceHistory, ResourceSnapshot, downsample_max
from .utils.submission_validation import (
    file_signature,
    find_sample_submission,
    validate_submission_file,
    validate_workspace_submission,
)

logger = logging.getLogger("aide")


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


@dataclass(frozen=True)
class LastErrorRecord:
    node: Node
    lines: list[str]


def parse_runtime_args(
    argv: list[str],
) -> tuple[ResumeRequest, RuntimeOptions, list[str]]:
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
            runtime = RuntimeOptions(
                show_invalid_submission_branches=True,
                force_check_submissions=runtime.force_check_submissions,
            )
        elif arg == "--force-check-submissions":
            runtime = RuntimeOptions(
                show_invalid_submission_branches=runtime.show_invalid_submission_branches,
                force_check_submissions=True,
            )
        else:
            remaining.append(arg)
        i += 1
    return resume, runtime, remaining


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
    cfg = OmegaConf.load(config_path)
    if (
        _cli_sets_key(cli_overrides, "agent.autogluon.profile")
        and not _cli_sets_key(cli_overrides, "agent.autogluon.included_model_types")
        and "agent" in cfg
        and "autogluon" in cfg.agent
    ):
        cfg.agent.autogluon.included_model_types = None
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    cfg.exp_name = run_id
    cfg.log_dir = log_dir
    cfg.workspace_dir = workspace_dir
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)
    _resolve_all_model_configs(cfg)
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
    blink_on: bool = True,
    show_invalid_submission_branches: bool = False,
    disable_oom_saturated_parents: bool = False,
    synthesis_node_ids: set[str] | None = None,
):
    if show_invalid_submission_branches:
        best_node = journal.get_best_node()
    else:
        visible_good_nodes = [
            node
            for node in journal.good_nodes
            if not node.is_in_submission_contract_error_branch
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
        tree.add(Text(indicator, style=active_placeholder_style()))

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

    def is_baseline_root(node: Node) -> bool:
        return node.parent is None and str(node.plan or "").startswith(
            BASELINE_PLAN_PREFIX
        )

    def append_rec(node: Node, tree):
        if node.is_terminal_failure:
            return
        if (
            node.is_submission_contract_error
            and not show_invalid_submission_branches
        ):
            return

        synthesis_root = is_synthesis_root(node)
        if synthesis_root and (
            node.is_buggy or node.metric is None or node.metric.value is None
        ):
            s = "[bold blue]◆[/bold blue] [red]bug[/red]"
        elif node.is_buggy or node.metric is None or node.metric.value is None:
            s = "[red]● bug"
        else:
            metric_text = f"{node.metric.value:.5f}"

            if disable_oom_saturated_parents and node.is_oom_blocked_parent:
                s = f"[bright_black]✕ {metric_text}"
            elif node is best_node:
                style = "bold "
                s = f"[bold yellow]*[/bold yellow] [{style}green]{metric_text}"
            elif is_baseline_root(node):
                style = "bold " if node is best_node else ""
                s = f"[bright_magenta]◎[/bright_magenta] [{style}green]{metric_text}"
            elif synthesis_root:
                style = "bold " if node is best_node else ""
                metric_style = f"{style}green"
                s = f"[blue]◆[/blue] [{metric_style}]{metric_text}"
            else:
                style = "bold " if node is best_node else ""
                s = f"[{style}green]● {metric_text}"

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
) -> Node | None:
    if show_invalid_submission_branches:
        return journal.get_best_node()

    visible_good_nodes = [
        node
        for node in journal.good_nodes
        if not node.is_in_submission_contract_error_branch
        and (
            not disable_oom_saturated_parents
            or not node.is_oom_blocked_parent
        )
    ]
    return max(visible_good_nodes, key=lambda node: node.metric, default=None)


def _tree_node_label(
    node: Node,
    *,
    best_node: Node | None,
    disable_oom_saturated_parents: bool = False,
    synthesis_node_ids: set[str] | None = None,
) -> Text:
    if node.is_terminal_failure:
        return Text("failed", style="red")

    synthesis_root = bool(synthesis_node_ids and node.id in synthesis_node_ids) or (
        node.parent is None and str(node.plan or "").startswith(SYNTHESIS_PLAN_PREFIX)
    )
    baseline_root = node.parent is None and str(node.plan or "").startswith(
        BASELINE_PLAN_PREFIX
    )
    if synthesis_root and (
        node.is_buggy or node.metric is None or node.metric.value is None
    ):
        label = Text()
        label.append("◆", style="bold blue")
        label.append(" bug", style="red")
        return label
    if node.is_buggy or node.metric is None or node.metric.value is None:
        return Text("● bug", style="red")

    if disable_oom_saturated_parents and node.is_oom_blocked_parent:
        return Text(f"✕ {node.metric.value:.5f}", style="bright_black")

    label = Text()
    if node is best_node:
        label.append("* ", style="bold yellow")
    elif baseline_root:
        label.append("◎ ", style="bright_magenta")
    elif synthesis_root:
        label.append("◆", style="blue")
        label.append(" ")
    else:
        label.append("● ", style="green")

    metric_style = "bold yellow" if node is best_node else "green"
    metric_text = f"{node.metric.value:.5f}"
    label.append(metric_text, style=metric_style)
    return label


def _tree_active_placeholder_line(
    *,
    active_stage: str | None,
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
    return Text(indicator, style=style)


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
    active_parent_node: Node | None = None,
    active_stage: str | None = None,
    blink_on: bool = True,
    show_invalid_submission_branches: bool = False,
    disable_oom_saturated_parents: bool = False,
    synthesis_node_ids: set[str] | None = None,
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
    best_node = _visible_best_node(
        journal,
        show_invalid_submission_branches=show_invalid_submission_branches,
        disable_oom_saturated_parents=disable_oom_saturated_parents,
    )

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
        if active_stage is None:
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
                blink_on=blink_on,
            )
        )
        append_item(TreeViewItem("active", parent_id, line, focus_start=len(prefix)))

    def append_rec(
        node: Node,
        parent_id: str,
        ancestor_has_next: list[bool],
        is_last: bool,
    ) -> None:
        if node.is_terminal_failure:
            return
        if node.is_submission_contract_error and not show_invalid_submission_branches:
            return

        prefix = "".join(
            "│   " if has_next else "    "
            for has_next in ancestor_has_next
        )
        prefix += "└── " if is_last else "├── "
        line = Text(prefix)
        line.append_text(
            _tree_node_label(
                node,
                best_node=best_node,
                disable_oom_saturated_parents=disable_oom_saturated_parents,
                synthesis_node_ids=synthesis_node_ids,
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
            if not child.is_terminal_failure
            and (
                show_invalid_submission_branches
                or not child.is_submission_contract_error
            )
        ]
        next_ancestors = [*ancestor_has_next, not is_last]
        has_active_child = node is active_parent_node
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
        if not node.is_terminal_failure
        and (
            show_invalid_submission_branches
            or not node.is_submission_contract_error
        )
    ]
    has_root_active = active_parent_node is None and active_stage is not None
    for index, node in enumerate(roots):
        append_rec(node, "header", [], index == len(roots) - 1 and not has_root_active)
    if active_parent_node is None:
        append_active("header", [])

    return TreeView(
        items=items,
        index_by_id={item.item_id: index for index, item in enumerate(items)},
        parent_by_id=parent_by_id,
        children_by_id=children_by_id,
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
                return None

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
    timestamp = dt.datetime.fromtimestamp(record.node.ctime).strftime("%H:%M:%S")
    return f"Last Error · {step}@{timestamp}"


def build_last_error_summary(journal: Journal) -> Group:
    record = last_error_record(journal)
    lines: list[Text] = [Text(_last_error_title(record), style="bold red")]
    error_lines = record.lines if record is not None else []
    if not error_lines:
        lines.append(Text("-", style="dim"))
    else:
        lines.extend(Text(line, style="dim") for line in error_lines)
    return Group(*lines)


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
    sampled = downsample_max(values, width)
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
    lines.extend(
        Text(f"▶ {label:<4} {value:>7} {bar} {spark}", style="yellow")
        for label, value, bar, spark in values
    )
    return Group(*lines)


ModelSetting = tuple[str, str, str | None]


def build_model_summary(model_settings: list[ModelSetting] | None) -> Group | None:
    if not model_settings:
        return None
    lines: list[Text] = [Text("Models", style="bold #c8c4ff")]
    for label, model, effort in model_settings:
        lines.append(Text(f"▶ {label:<9} {model} - {effort or '-'}", style="yellow"))
    return Group(*lines)


def model_settings_for_run(cfg: Config) -> list[ModelSetting]:
    settings: list[ModelSetting] = [
        ("code", cfg.agent.code.model, cfg.agent.code.reasoning_effort),
        ("feedback", cfg.agent.feedback.model, cfg.agent.feedback.reasoning_effort),
        ("report", cfg.report.model, cfg.report.reasoning_effort),
    ]
    if cfg.research.enabled:
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
    model_settings: list[ModelSetting] | None = None,
    active_artifact_dir: Path | None = None,
) -> Group:
    if resource_history is None and resource_snapshot is not None:
        resource_history = ResourceHistory()
        resource_history.add(resource_snapshot)
    lines = [progress, status]
    if research_status is not None:
        lines.append("")
        lines.append(research_status)
    if synthesis_status is not None:
        if research_status is None:
            lines.append("")
        lines.append(synthesis_status)
    model_summary = build_model_summary(model_settings)
    if model_summary is not None:
        lines.extend(["", model_summary])
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
    lines.extend([Rule(style="dim"), build_last_error_summary(journal)])
    if resource_active:
        lines.extend([Rule(style="dim"), build_resource_summary(resource_history)])
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


def _execution_crash_log_diagnostic(artifact_dir: Path | None) -> str | None:
    if artifact_dir is None:
        return None

    for log_name in ("autogluon_stdout.log", "process_stdout.log"):
        log_path = artifact_dir / log_name
        if not log_path.exists():
            continue
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        lower_log = log_text.lower()
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

    return None


def _mark_node_execution_crash(
    node: Node,
    exc: RuntimeError,
    *,
    artifact_dir: Path | None = None,
) -> None:
    message = str(exc) or exc.__class__.__name__
    diagnostic = _execution_crash_log_diagnostic(artifact_dir)
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
    timestamp = dt.datetime.fromtimestamp(node.ctime).strftime("%Y%m%dT%H%M%S")
    return Path(cfg.log_dir) / "artifacts" / timestamp / "submission.csv"


def _node_artifact_dir(cfg, node: Node) -> Path:
    timestamp = dt.datetime.fromtimestamp(node.ctime).strftime("%Y%m%dT%H%M%S")
    return Path(cfg.log_dir) / "artifacts" / timestamp


def _node_artifact_dir_from_ctime(cfg, ctime: float) -> Path:
    timestamp = dt.datetime.fromtimestamp(ctime).strftime("%Y%m%dT%H%M%S")
    return Path(cfg.log_dir) / "artifacts" / timestamp


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
    nodes_to_check = [
        node
        for node in journal.nodes
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


def stage_status_message(active_stage: str | None, elapsed: float | None = None) -> str:
    elapsed_text = _format_elapsed(elapsed)
    if active_stage == "generating":
        return f"[green]Generating code...{elapsed_text}"
    if active_stage == "executing":
        return f"[magenta]Executing code...{elapsed_text}"
    if active_stage == "reviewing":
        return f"[cyan]Reviewing result...{elapsed_text}"
    return f"[green]Generating code...{elapsed_text}"


KeyboardInterruptAction = Literal["continue", "abort"]


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
            thread.join(timeout=0.25)
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
        journal = Journal()
        is_resume = False

    logger.info(f'Starting run "{cfg.exp_name}"')
    os.environ["AIDE_RUN_ID"] = cfg.exp_name
    os.environ["AIDE_LOG_DIR"] = str(cfg.log_dir)

    task_desc = load_task_desc(cfg)

    if not is_resume:
        with Status("Preparing agent workspace (copying and extracting files) ..."):
            prep_agent_workspace(cfg)

    def cleanup():
        if not is_resume and global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    research_advisor = ResearchAdvisor(cfg=cfg, task_desc=task_desc)
    synthesis_advisor = SynthesisAdvisor(cfg=cfg, task_desc=task_desc)
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),  # type: ignore
    )

    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)
    status_override: str | None = None
    stop_after_current_node = False
    execution_interrupt_count = 0
    resource_history = ResourceHistory(window_seconds=30 * 60, interval_seconds=1)
    resource_active = False
    focused_tree_item_id = "header"
    tree_scroll_top = 0
    key_reader: ArrowKeyReader | None = None
    pending_artifact_dir: Path | None = None

    def request_execution_interrupt() -> KeyboardInterruptAction:
        nonlocal execution_interrupt_count, status_override, stop_after_current_node
        execution_interrupt_count += 1
        if execution_interrupt_count == 1:
            stop_after_current_node = True
            status_override = (
                "[yellow]Ctrl+C received. Waiting for current code to finish. "
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

        try:
            resource_active = True
            result = interpreter.run(
                *args,
                interrupt_callback=lambda _count: request_execution_interrupt(),
                resource_callback=on_resource,
                **kwargs,
            )
            return result
        finally:
            resource_active = False
            if not stop_after_current_node:
                status_override = None

    def prepare_node_artifact_env(node: Node) -> str | None:
        previous = os.environ.get("AIDE_NODE_ARTIFACT_DIR")
        artifact_dir = _node_artifact_dir(cfg, node)
        artifact_dir.mkdir(parents=True, exist_ok=True)
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

    def drain_tree_navigation(view: TreeView) -> None:
        nonlocal focused_tree_item_id, tree_scroll_top
        if focused_tree_item_id not in view.index_by_id:
            focused_tree_item_id = "header"
        while key_reader is not None:
            direction = key_reader.read_key()
            if direction is None:
                break
            focused_tree_item_id = move_tree_focus(
                view,
                focused_tree_item_id,
                direction,
            )
        focus_index = view.index_by_id.get(focused_tree_item_id, 0)
        tree_scroll_top = clamp_tree_viewport(
            total_lines=len(view.items),
            viewport_height=tree_viewport_height(),
            focus_index=focus_index,
            current_scroll=tree_scroll_top,
        )

    def current_tree_view(*, blink_on: bool) -> TreeView:
        return build_tree_view(
            journal,
            active_parent_node=agent.active_parent_node,
            active_stage=agent.active_stage,
            blink_on=blink_on,
            show_invalid_submission_branches=(
                runtime_options.show_invalid_submission_branches
            ),
            disable_oom_saturated_parents=(
                cfg.agent.search.disable_oom_saturated_parents
            ),
            synthesis_node_ids=synthesis_injected_node_ids(cfg.log_dir),
        )

    def generate_live():
        nonlocal focused_tree_item_id, tree_scroll_top
        blink_on = int(time.monotonic() * 2) % 2 == 0
        tree_view = current_tree_view(blink_on=blink_on)
        drain_tree_navigation(tree_view)
        tree = render_tree_view(
            tree_view,
            focused_item_id=focused_tree_item_id,
            scroll_top=tree_scroll_top,
            viewport_height=tree_viewport_height(),
        )
        prog.update(prog.task_ids[0], completed=global_step)
        elapsed = (
            time.monotonic() - agent.active_stage_started_at
            if agent.active_stage_started_at is not None
            else None
        )
        status.update(
            status_override or stage_status_message(agent.active_stage, elapsed)
        )

        tree_panel = Panel(
            Padding(tree, (0, 1, 0, 1)),
            title=f'[b]AIDE: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="↑/↓ move  ← parent  → child  Ctrl+C stop",
        )
        data_panel = Panel(
            Padding(
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
                    model_settings=model_settings_for_run(cfg),
                    active_artifact_dir=(
                        _node_artifact_dir(cfg, agent.active_node)
                        if agent.active_node is not None
                        else pending_artifact_dir
                    ),
                ),
                (0, 1, 0, 1),
            ),
            title="[b]Run data",
        )

        layout = Layout()
        layout.split_row(
            Layout(tree_panel, name="tree", ratio=3),
            Layout(data_panel, name="data", ratio=2),
        )
        return layout

    interrupted = False
    interrupt_message = ""
    try:
        with ArrowKeyReader() as reader:
            key_reader = reader
            with Live(
                get_renderable=generate_live,
                refresh_per_second=16,
                screen=True,
            ) as live:
                while global_step < cfg.agent.steps:
                    synthesized: SynthesisNode | None = None
                    try:
                        if cfg.synthesis.enabled:
                            agent.active_parent_node = None
                            agent.set_active_stage("generating")
                            synthesized = run_with_live_refresh(
                                live,
                                generate_live,
                                lambda: synthesis_advisor.generate_node_if_due(
                                    journal=journal,
                                    completed_steps=count_scored_working_nodes(journal),
                                ),
                                tick=lambda: drain_tree_navigation(
                                    current_tree_view(blink_on=True)
                                ),
                            )
                            if synthesized is None:
                                agent.clear_active_step()

                        if synthesized is None:
                            parent_node = agent.prepare_step()
                            node_ctime = time.time()
                            pending_artifact_dir = _node_artifact_dir_from_ctime(
                                cfg,
                                node_ctime,
                            )
                            pending_artifact_dir.mkdir(parents=True, exist_ok=True)
                            result_node = run_with_live_refresh(
                                live,
                                generate_live,
                                lambda: agent.generate_node(
                                    parent_node,
                                    node_ctime=node_ctime,
                                    llm_log_dir=pending_artifact_dir,
                                ),
                                tick=lambda: drain_tree_navigation(
                                    current_tree_view(blink_on=True)
                                ),
                            )
                        else:
                            result_node = synthesized.node
                        pending_artifact_dir = None

                        if (
                            synthesized is not None
                            and not synthesized.ready_for_execution
                        ):
                            journal.append(result_node)
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
                                    exec_result = agent.execute_node(
                                        result_node,
                                        lambda *args, **kwargs: run_with_live_refresh(
                                            live,
                                            generate_live,
                                            lambda: exec_callback(*args, **kwargs),
                                            tick=lambda: drain_tree_navigation(
                                                current_tree_view(blink_on=True)
                                            ),
                                            on_keyboard_interrupt=request_execution_interrupt,
                                        ),
                                    )
                                except RuntimeError as exc:
                                    _mark_node_execution_crash(
                                        result_node,
                                        exc,
                                        artifact_dir=_node_artifact_dir(
                                            cfg,
                                            result_node,
                                        ),
                                    )
                                    journal.append(result_node)
                                    if synthesized is not None:
                                        synthesis_advisor.mark_injected(
                                            synthesized,
                                            node=result_node,
                                        )
                                    continue
                            finally:
                                restore_node_artifact_env(previous_artifact_env)
                            run_with_live_refresh(
                                live,
                                generate_live,
                                lambda: agent.review_node(result_node, exec_result),
                                tick=lambda: drain_tree_navigation(
                                    current_tree_view(blink_on=True)
                                ),
                            )
                            enforce_submission_contract(cfg, result_node)
                            journal.append(result_node)
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
                    save_run(
                        cfg,
                        journal,
                        current_node=journal[-1],
                        progress_callback=lambda message: update_save_status(
                            message,
                            live,
                        ),
                    )
                    status_override = None
                    global_step = len(journal)
                    if stop_after_current_node:
                        interrupted = True
                        interrupt_message = (
                            "Execution stopped by user after saving current node."
                        )
                        break
                    research_advisor.maybe_start(
                        journal=journal,
                        completed_steps=count_scored_working_nodes(journal),
                    )
                live.update(generate_live(), refresh=True)
    finally:
        interpreter.cleanup_session()

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
