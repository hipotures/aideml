import atexit
import sys
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .agent import Agent
from .interpreter import Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
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

    def append_active_placeholder(tree):
        if active_stage is None:
            return
        indicator = "[*]" if blink_on else "[ ]"
        tree.add(Text(indicator, style="bold yellow"))

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)
        if node is active_parent_node:
            append_active_placeholder(subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    if active_parent_node is None:
        append_active_placeholder(tree)
    return tree


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

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def generate_live():
        blink_on = int(time.monotonic() * 2) % 2 == 0
        tree = journal_to_rich_tree(
            journal,
            active_parent_node=agent.active_parent_node,
            active_stage=agent.active_stage,
            blink_on=blink_on,
        )
        prog.update(prog.task_ids[0], completed=global_step)

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]
        left = Group(prog, status)
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (0, 1, 0, 1)),
                Columns(
                    [Padding(left, (0, 1, 0, 1)), Padding(right, (0, 1, 0, 1))],
                    equal=False,
                    expand=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    with Live(
        get_renderable=generate_live,
        refresh_per_second=16,
        screen=True,
    ) as live:
        while global_step < cfg.agent.steps:
            agent.step(exec_callback=exec_callback)
            save_run(cfg, journal, current_node=journal[-1])
            global_step = len(journal)
            live.update(generate_live())
    interpreter.cleanup_session()

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
