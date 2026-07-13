"""Show per-step Codex token usage for an AIDE run."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from aide.web_dashboard.token_usage import TokenUsageRow, collect_token_usage


def render_table(run_dir: Path, rows: list[TokenUsageRow]) -> Table:
    table = Table(title=f"Codex token usage · {run_dir.name}")
    table.add_column("Step", justify="right")
    table.add_column("Agent")
    table.add_column("Session / thread ID", no_wrap=True, min_width=36)
    table.add_column("Turn ID", no_wrap=True, min_width=36)
    table.add_column("Action")
    table.add_column("Input", justify="right")
    table.add_column("Cached", justify="right")
    table.add_column("Uncached", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Turn total", justify="right")
    table.add_column("Thread total", justify="right")
    step_styles = {
        step: "on grey30" if rank % 2 == 1 else None
        for rank, step in enumerate(dict.fromkeys(row.step for row in rows))
    }
    for row in rows:
        table.add_row(
            str(row.step),
            row.agent,
            row.thread_id,
            row.turn_id,
            row.action,
            f"{row.input_tokens:,}",
            f"{row.cached_input_tokens:,}",
            f"{row.input_tokens - row.cached_input_tokens:,}",
            f"{row.output_tokens:,}",
            f"{row.turn_total_tokens:,}",
            f"{row.thread_total_tokens:,}",
            style=step_styles[row.step],
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Run name under logs/ or a path to a run directory")
    parser.add_argument(
        "--full-view",
        action="store_true",
        help="Include pre-session app-server steps without a thread action",
    )
    args = parser.parse_args()
    run_dir = Path(args.run).expanduser()
    if not run_dir.is_dir():
        run_dir = Path("logs") / args.run
    run_dir = run_dir.resolve()
    rows = collect_token_usage(run_dir, full_view=args.full_view)
    Console().print(render_table(run_dir, rows))


if __name__ == "__main__":
    main()
