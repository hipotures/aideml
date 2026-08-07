#!/usr/bin/env python3
"""Queue timeout nodes for another execution with the current AIDE config."""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path

from aide.journal import Journal, Node
from aide.run import mark_node_generated_only
from aide.utils import serialize


def is_timeout_node(node: Node) -> bool:
    return node.status != "generated" and node.is_timeout_failure


def select_timeout_nodes(
    journal: Journal,
    *,
    steps: set[int] | None = None,
) -> list[Node]:
    return [
        node
        for node in journal.nodes
        if (steps is None or node.step in steps) and is_timeout_node(node)
    ]


def backup_journal(journal_path: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = journal_path.with_name(f"journal.json.bak-timeout-{timestamp}")
    suffix = 1
    while backup.exists():
        backup = journal_path.with_name(
            f"journal.json.bak-timeout-{timestamp}-{suffix}"
        )
        suffix += 1
    shutil.copy2(journal_path, backup)
    return backup


def reset_timeout_nodes(
    log_dir: Path,
    *,
    steps: set[int] | None = None,
    dry_run: bool = False,
) -> tuple[list[int], Path | None]:
    log_dir = Path(log_dir).expanduser().resolve()
    journal_path = log_dir / "journal.json"
    if not journal_path.exists():
        raise FileNotFoundError(f"Missing journal: {journal_path}")

    journal = serialize.load_json(journal_path, Journal)
    selected = select_timeout_nodes(journal, steps=steps)
    selected_steps = [node.step for node in selected if node.step is not None]
    if dry_run or not selected:
        return selected_steps, None

    backup = backup_journal(journal_path)
    for node in selected:
        mark_node_generated_only(node)
    serialize.dump_json(journal, journal_path)
    return selected_steps, backup


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Queue timeout nodes from an AIDE run for re-execution with the "
            "current timeout settings."
        )
    )
    parser.add_argument("log_dir", type=Path, help="Run log directory, e.g. logs/manual")
    parser.add_argument(
        "--step",
        type=int,
        action="append",
        dest="steps",
        help="Only reset this step; may be repeated. By default, reset all timeout nodes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching timeout nodes without changing the journal.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_steps, backup = reset_timeout_nodes(
        args.log_dir,
        steps=set(args.steps) if args.steps else None,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"Selected timeout step(s): {selected_steps or '-'}")
        return 0
    if backup is None:
        print("No timeout nodes selected.")
        return 0
    print(f"Queued timeout step(s): {selected_steps}")
    print(f"Journal backup: {backup}")
    print("Resume with the increased timeout using: uv run aide --resume <RUN_ID>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
