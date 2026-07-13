"""Show per-step Codex token usage for an AIDE run."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class TokenUsageRow:
    step: int
    thread_id: str
    turn_id: str
    action: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    turn_total_tokens: int
    thread_total_tokens: int


def _integer(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _row_from_response(step: int, response_path: Path) -> TokenUsageRow | None:
    try:
        response = json.loads(response_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    info = response.get("info")
    if not isinstance(info, dict):
        return None
    thread_id = info.get("thread_id")
    turn_id = info.get("turn_id")
    usage = info.get("usage")
    token_usage = usage.get("tokenUsage") if isinstance(usage, dict) else None
    last = token_usage.get("last") if isinstance(token_usage, dict) else None
    total = token_usage.get("total") if isinstance(token_usage, dict) else None
    if not all(
        isinstance(value, dict) for value in (last, total)
    ) or not all(isinstance(value, str) and value for value in (thread_id, turn_id)):
        return None

    input_tokens = _integer(last.get("inputTokens"))
    cached_input_tokens = _integer(last.get("cachedInputTokens"))
    output_tokens = _integer(last.get("outputTokens"))
    turn_total_tokens = _integer(last.get("totalTokens"))
    thread_total_tokens = _integer(total.get("totalTokens"))
    values = (
        input_tokens,
        cached_input_tokens,
        output_tokens,
        turn_total_tokens,
        thread_total_tokens,
    )
    if any(value is None for value in values):
        return None
    return TokenUsageRow(
        step=step,
        thread_id=thread_id,
        turn_id=turn_id,
        action=str(info.get("thread_action") or ""),
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        turn_total_tokens=turn_total_tokens,
        thread_total_tokens=thread_total_tokens,
    )


def collect_token_usage(run_dir: Path) -> list[TokenUsageRow]:
    rows_by_step: dict[int, TokenUsageRow] = {}
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"AIDE artifacts directory not found: {artifacts_dir}")
    for artifact_dir in sorted(artifacts_dir.iterdir()):
        if not artifact_dir.is_dir():
            continue
        try:
            step = int(artifact_dir.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        response_path = artifact_dir / "response.json"
        if not response_path.exists():
            continue
        row = _row_from_response(step, response_path)
        if row is not None:
            rows_by_step[step] = row
    return [rows_by_step[step] for step in sorted(rows_by_step)]


def render_table(run_dir: Path, rows: list[TokenUsageRow]) -> Table:
    table = Table(title=f"Codex token usage · {run_dir.name}")
    table.add_column("Step", justify="right")
    table.add_column("Session / thread ID", no_wrap=True, min_width=36)
    table.add_column("Action")
    table.add_column("Input", justify="right")
    table.add_column("Cached", justify="right")
    table.add_column("Uncached", justify="right")
    table.add_column("Output", justify="right")
    table.add_column("Turn total", justify="right")
    table.add_column("Thread total", justify="right")
    for row in rows:
        table.add_row(
            str(row.step),
            row.thread_id,
            row.action,
            f"{row.input_tokens:,}",
            f"{row.cached_input_tokens:,}",
            f"{row.input_tokens - row.cached_input_tokens:,}",
            f"{row.output_tokens:,}",
            f"{row.turn_total_tokens:,}",
            f"{row.thread_total_tokens:,}",
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Run name under logs/ or a path to a run directory")
    args = parser.parse_args()
    run_dir = Path(args.run).expanduser()
    if not run_dir.is_dir():
        run_dir = Path("logs") / args.run
    run_dir = run_dir.resolve()
    rows = collect_token_usage(run_dir)
    terminal = Console()
    Console(width=max(terminal.width, 150)).print(render_table(run_dir, rows))


if __name__ == "__main__":
    main()
