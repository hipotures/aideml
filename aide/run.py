import atexit
import datetime as dt
import logging
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .agent import Agent
from .interpreter import ExecutionInterrupted, Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from .research import ResearchAdvisor, count_scored_working_nodes
from .synthesis import SYNTHESIS_PLAN_PREFIX, SynthesisAdvisor, SynthesisNode
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
    load_task_desc,
    prep_agent_workspace,
    save_run,
    load_cfg,
)
from .utils.metric import MetricValue, WorstMetricValue
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

    cfg = OmegaConf.load(config_path)
    if cli_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
    cfg.exp_name = run_id
    cfg.log_dir = log_dir
    cfg.workspace_dir = workspace_dir
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)
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
):
    if show_invalid_submission_branches:
        best_node = journal.get_best_node()
    else:
        visible_good_nodes = [
            node
            for node in journal.good_nodes
            if not node.is_in_submission_contract_error_branch
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
        return node.parent is None and str(node.plan or "").startswith(
            SYNTHESIS_PLAN_PREFIX
        )

    def append_rec(node: Node, tree):
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
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""
            metric_text = f"{node.metric.value:.5f}"

            if synthesis_root:
                suffix = " (best)" if node is best_node else ""
                s = f"[{style}blue]◆ {metric_text}{suffix}"
            elif node is best_node:
                s = f"[{style}green]● {metric_text} (best)"
            else:
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


def build_path_summary(log_dir: Path, workspace_dir: Path) -> Group:
    path_entries = [
        ("Agent workspace directory", workspace_dir),
        ("Experiment log directory", log_dir),
    ]
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


def last_error_lines(journal: Journal, *, max_lines: int = 2) -> list[str]:
    for node in reversed(journal.buggy_nodes):
        term_lines = []
        if node._term_out:
            term_lines.extend("".join(node._term_out).splitlines())

        if term_lines and _terminal_output_has_error(term_lines):
            lines = _clean_error_lines(term_lines, max_lines=max_lines)
            if lines:
                return lines

        lines = _clean_error_lines(_exception_lines(node), max_lines=max_lines)
        if lines:
            return lines

        if node.analysis:
            lines = _clean_error_lines(
                str(node.analysis).splitlines(),
                max_lines=max_lines,
            )
            if lines:
                return lines
    return []


def build_last_error_summary(journal: Journal) -> Group:
    lines: list[Text] = [Text("Last Error", style="bold red")]
    error_lines = last_error_lines(journal)
    if not error_lines:
        lines.append(Text("-", style="dim"))
    else:
        lines.extend(Text(line, style="dim") for line in error_lines)
    return Group(*lines)


def build_run_data(
    *,
    progress,
    status,
    research_status: str | None,
    synthesis_status: str | None,
    journal: Journal,
    log_dir: Path,
    workspace_dir: Path,
) -> Group:
    lines = [progress, status]
    if research_status is not None:
        lines.append("")
        lines.append(research_status)
    if synthesis_status is not None:
        if research_status is None:
            lines.append("")
        lines.append(synthesis_status)
    lines.extend(["", build_path_summary(log_dir, workspace_dir)])
    lines.extend([Rule(style="dim"), build_last_error_summary(journal)])
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


def run_with_live_refresh(live: Live, render, func):
    result_queue = queue.Queue(maxsize=1)

    def worker():
        try:
            result_queue.put((True, func()))
        except BaseException as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        live.update(render(), refresh=True)
        thread.join(timeout=0.25)
    live.update(render(), refresh=True)

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
    stop_after_current_execution = False

    def exec_callback(*args, **kwargs):
        nonlocal status_override, stop_after_current_execution

        def on_interrupt(interrupt_count: int):
            nonlocal status_override, stop_after_current_execution
            if interrupt_count == 1:
                stop_after_current_execution = True
                status_override = (
                    "[yellow]Ctrl+C received. Waiting for current code to finish. "
                    "The run will stop before review. Press Ctrl+C again to stop now."
                )
            else:
                status_override = "[red]Stopping current code execution..."

        try:
            result = interpreter.run(
                *args,
                interrupt_callback=on_interrupt,
                **kwargs,
            )
            if stop_after_current_execution:
                raise ExecutionInterrupted(
                    "Execution finished after interrupt request."
                )
            return result
        finally:
            if not stop_after_current_execution:
                status_override = None

    def update_save_status(message: str, live: Live) -> None:
        nonlocal status_override
        status_override = f"[blue]Saving run: {message}..."
        live.update(generate_live(), refresh=True)

    def generate_live():
        blink_on = int(time.monotonic() * 2) % 2 == 0
        tree = journal_to_rich_tree(
            journal,
            active_parent_node=agent.active_parent_node,
            active_stage=agent.active_stage,
            blink_on=blink_on,
            show_invalid_submission_branches=(
                runtime_options.show_invalid_submission_branches
            ),
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
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
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
                        )
                        if synthesized is None:
                            agent.clear_active_step()

                    if synthesized is None:
                        parent_node = agent.prepare_step()
                        result_node = run_with_live_refresh(
                            live,
                            generate_live,
                            lambda: agent.generate_node(parent_node),
                        )
                    else:
                        result_node = synthesized.node

                    exec_result = agent.execute_node(result_node, exec_callback)
                    run_with_live_refresh(
                        live,
                        generate_live,
                        lambda: agent.review_node(result_node, exec_result),
                    )
                    enforce_submission_contract(cfg, result_node)
                    journal.append(result_node)
                    if synthesized is not None:
                        synthesis_advisor.mark_injected(synthesized, node=result_node)
                except ExecutionInterrupted:
                    interrupted = True
                    interrupt_message = (
                        "Execution stopped by user. Current node was not saved; "
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
                save_run(
                    cfg,
                    journal,
                    current_node=journal[-1],
                    progress_callback=lambda message: update_save_status(message, live),
                )
                status_override = None
                global_step = len(journal)
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
        report = journal2report(journal, task_desc, cfg.report)
        print(report)
        report_file_path = cfg.log_dir / "report.md"
        with open(report_file_path, "w") as f:
            f.write(report)
        print("Report written to file:", report_file_path)


if __name__ == "__main__":
    run()
