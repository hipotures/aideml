import atexit
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
    TimeRemainingColumn,
)
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

logger = logging.getLogger("aide")


@dataclass(frozen=True)
class ResumeRequest:
    requested: bool = False
    run_id: str | None = None

    @property
    def use_latest(self) -> bool:
        return self.requested and self.run_id is None


def parse_resume_args(argv: list[str]) -> tuple[ResumeRequest, list[str]]:
    remaining: list[str] = []
    resume = ResumeRequest()
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
        else:
            remaining.append(arg)
        i += 1
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
):
    best_node = journal.get_best_node()
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

    def append_rec(node: Node, tree):
        if node.is_buggy or node.metric is None or node.metric.value is None:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""
            metric_text = f"{node.metric.value:.3f}"

            if node is best_node:
                s = f"[{style}green]● {metric_text} (best)"
            else:
                s = f"[{style}green]● {metric_text}"

        subtree = tree.add(s)
        for child in list(node.children):
            if child in journal_nodes:
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
    resume_request, cli_args = parse_resume_args(raw_argv)

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
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),  # type: ignore
    )

    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
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
                Group(
                    prog,
                    status,
                    "",
                    build_path_summary(cfg.log_dir, cfg.workspace_dir),
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
                try:
                    parent_node = agent.prepare_step()
                    result_node = run_with_live_refresh(
                        live,
                        generate_live,
                        lambda: agent.generate_node(parent_node),
                    )
                    exec_result = agent.execute_node(result_node, exec_callback)
                    run_with_live_refresh(
                        live,
                        generate_live,
                        lambda: agent.review_node(result_node, exec_result),
                    )
                    journal.append(result_node)
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
