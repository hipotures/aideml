from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class PlannedReset:
    root_node_id: str
    step: int | None
    hypothesis_id: str
    status: str | None
    removed_node_count: int


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _backup_journal(path: Path) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = path.with_name(f"{path.name}.{timestamp}.bak")
    counter = 1
    while backup_path.exists():
        backup_path = path.with_name(f"{path.name}.{timestamp}.{counter}.bak")
        counter += 1
    backup_path.write_bytes(path.read_bytes())
    return backup_path


def _node_id(node: dict[str, Any]) -> str | None:
    node_id = node.get("id")
    return node_id if isinstance(node_id, str) and node_id else None


def _parent_id(node: dict[str, Any], node2parent: dict[str, Any]) -> str | None:
    node_id = _node_id(node)
    if node_id is not None:
        parent_id = node2parent.get(node_id)
        if isinstance(parent_id, str) and parent_id:
            return parent_id
    parent = node.get("parent")
    if isinstance(parent, str) and parent:
        return parent
    if isinstance(parent, dict):
        return _node_id(parent)
    return None


def _hypothesis_id_for_node(node: dict[str, Any]) -> str | None:
    offered = node.get("research_hypotheses_offered")
    if isinstance(offered, list) and len(offered) == 1 and isinstance(offered[0], str):
        return offered[0]
    return None


def _nodes_and_parent_map(journal: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes = journal.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("journal.json must contain a nodes list.")
    if not all(isinstance(node, dict) for node in nodes):
        raise ValueError("journal.json nodes must be JSON objects.")
    node2parent = journal.get("node2parent")
    if not isinstance(node2parent, dict):
        node2parent = {}
    return nodes, node2parent


def _children_by_parent(
    nodes: list[dict[str, Any]],
    node2parent: dict[str, Any],
) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {}
    for node in nodes:
        node_id = _node_id(node)
        parent_id = _parent_id(node, node2parent)
        if node_id is None or parent_id is None:
            continue
        children.setdefault(parent_id, []).append(node_id)
    return children


def _subtree_ids(root_id: str, children: dict[str, list[str]]) -> set[str]:
    result: set[str] = set()
    stack = [root_id]
    while stack:
        node_id = stack.pop()
        if node_id in result:
            continue
        result.add(node_id)
        stack.extend(children.get(node_id, []))
    return result


def _is_buggy_root_node(node: dict[str, Any], node2parent: dict[str, Any]) -> bool:
    if _parent_id(node, node2parent) is not None:
        return False
    if _hypothesis_id_for_node(node) is None:
        return False
    status = node.get("status")
    return status in {"bug", "failed"} or node.get("is_buggy") is True


def plan_resets(journal: dict[str, Any]) -> list[PlannedReset]:
    nodes, node2parent = _nodes_and_parent_map(journal)
    children = _children_by_parent(nodes, node2parent)
    resets: list[PlannedReset] = []
    for node in nodes:
        if not _is_buggy_root_node(node, node2parent):
            continue
        node_id = _node_id(node)
        hypothesis_id = _hypothesis_id_for_node(node)
        if node_id is None or hypothesis_id is None:
            continue
        status = node.get("status")
        resets.append(
            PlannedReset(
                root_node_id=node_id,
                step=node.get("step") if isinstance(node.get("step"), int) else None,
                hypothesis_id=hypothesis_id,
                status=status if isinstance(status, str) else None,
                removed_node_count=len(_subtree_ids(node_id, children)),
            )
        )
    return resets


def apply_resets(journal: dict[str, Any], resets: list[PlannedReset]) -> None:
    if not resets:
        return
    nodes, node2parent = _nodes_and_parent_map(journal)
    children = _children_by_parent(nodes, node2parent)
    root_ids = {reset.root_node_id for reset in resets}
    remove_ids: set[str] = set()
    for root_id in root_ids:
        remove_ids.update(_subtree_ids(root_id, children))

    journal["nodes"] = [
        node for node in nodes if (node_id := _node_id(node)) not in remove_ids
    ]
    if isinstance(journal.get("node2parent"), dict):
        journal["node2parent"] = {
            child_id: parent_id
            for child_id, parent_id in journal["node2parent"].items()
            if child_id not in remove_ids and parent_id not in remove_ids
        }


def _print_resets(resets: list[PlannedReset]) -> None:
    if not resets:
        print("No buggy root hypotheses found.")
        return

    console = Console()
    console.print(
        f"Planned buggy root hypothesis resets: {len(resets)} root node(s)"
    )
    table = Table(
        box=box.SIMPLE,
        show_lines=False,
        pad_edge=False,
        expand=False,
    )
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("node", no_wrap=True)
    table.add_column("hypothesis", no_wrap=True)
    table.add_column("change", no_wrap=True)
    table.add_column("removed", justify="right", no_wrap=True)
    for index, reset in enumerate(resets):
        table.add_row(
            "?" if reset.step is None else str(reset.step),
            reset.root_node_id[:8],
            reset.hypothesis_id,
            f"{reset.status or 'bug'} -> reset",
            str(reset.removed_node_count),
            style=["on grey11", "on grey23"][index % 2],
        )
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reset buggy root hypothesis nodes by removing their journal subtrees. "
            "Hypothesis files, manifests, and artifacts are left untouched."
        )
    )
    parser.add_argument("run_id", help="Run id under logs/, for example 2-name")
    parser.add_argument("--logs-dir", default="logs", help="Top-level logs directory")
    parser.add_argument("--dry-run", action="store_true", help="Print resets only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.logs_dir) / args.run_id
    journal_path = run_dir / "journal.json"
    if not journal_path.exists():
        raise FileNotFoundError(f"Missing journal: {journal_path}")

    journal = _read_json(journal_path)
    resets = plan_resets(journal)
    print(f"Run: {args.run_id}")
    print(f"Journal: {journal_path}")
    _print_resets(resets)

    if args.dry_run or not resets:
        return 0

    answer = input("Reset these root hypotheses in journal.json? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Aborted; journal unchanged.")
        return 1

    backup_path = _backup_journal(journal_path)
    print(f"Backup written: {backup_path}")
    apply_resets(journal, resets)
    _write_json_atomic(journal_path, journal)
    print(f"Updated {journal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
