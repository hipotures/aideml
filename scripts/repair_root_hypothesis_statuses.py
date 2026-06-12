from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class PlannedChange:
    node_index: int
    node_id: str
    step: int | None
    hypothesis_id: str
    manifest_file: str | None
    field: str
    before: Any
    after: Any


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


def _task_slug_from_config(run_dir: Path) -> str | None:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^data_dir:\s*(.+?)\s*$", line)
        if match:
            return Path(match.group(1).strip().strip("'\"")).name
    return None


def _agent_mode_from_config(run_dir: Path) -> str | None:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*mode:\s*(.+?)\s*$", line)
        if not match:
            continue
        raw = match.group(1).strip().strip("'\"")
        if raw.startswith("autogluon"):
            return "autogluon"
        if raw:
            return raw
    return None


def _manifest_active_file(manifest: dict[str, Any], agent_mode: str) -> str | None:
    for key in ("active", "active_versions"):
        active = manifest.get(key)
        if isinstance(active, dict) and isinstance(active.get(agent_mode), str):
            return active[agent_mode]
    return None


def _manifest_entries(manifest: dict[str, Any], agent_mode: str) -> list[dict[str, Any]]:
    versions = manifest.get("versions")
    if not isinstance(versions, dict):
        return []
    entries = versions.get(agent_mode)
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _entry_for_active_file(
    manifest: dict[str, Any],
    *,
    agent_mode: str,
) -> dict[str, Any] | None:
    entries = _manifest_entries(manifest, agent_mode)
    active_file = _manifest_active_file(manifest, agent_mode)
    if active_file is not None:
        for entry in entries:
            if entry.get("file") == active_file:
                return entry
    return entries[-1] if entries else None


def _numeric_score(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _expected_status(entry: dict[str, Any]) -> str:
    status = entry.get("status")
    if isinstance(status, str) and status:
        return status
    if entry.get("buggy") is True:
        return "bug"
    if _numeric_score(entry.get("score")) is not None:
        return "ok"
    return "generated"


def _expected_is_buggy(entry: dict[str, Any], status: str) -> bool:
    buggy = entry.get("buggy")
    if isinstance(buggy, bool):
        return buggy
    return status in {"bug", "failed"}


def _expected_metric(entry: dict[str, Any], status: str) -> dict[str, Any] | None:
    score = _numeric_score(entry.get("score"))
    if status == "ok" and score is not None:
        return {"value": score, "maximize": True}
    return None


def _values_equal(field: str, before: Any, after: Any) -> bool:
    if field == "metric":
        if after is None and isinstance(before, dict):
            value = before.get("value")
            maximize = before.get("maximize")
            if value is None and maximize is None:
                return True
    return before == after


def _root_node_indices(journal: dict[str, Any]) -> list[int]:
    nodes = journal.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("journal.json must contain a nodes list.")
    node2parent = journal.get("node2parent")
    child_ids = set(node2parent.keys()) if isinstance(node2parent, dict) else set()
    result: list[int] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        is_child = isinstance(node_id, str) and node_id in child_ids
        if node.get("parent") is None and not is_child:
            result.append(index)
    return result


def _hypothesis_id_for_root(node: dict[str, Any]) -> str | None:
    offered = node.get("research_hypotheses_offered")
    if isinstance(offered, list) and len(offered) == 1 and isinstance(offered[0], str):
        return offered[0]
    return None


def _plan_changes(
    *,
    journal: dict[str, Any],
    hypothesis_root: Path,
    agent_mode: str,
) -> list[PlannedChange]:
    nodes = journal["nodes"]
    changes: list[PlannedChange] = []
    for index in _root_node_indices(journal):
        node = nodes[index]
        hypothesis_id = _hypothesis_id_for_root(node)
        if hypothesis_id is None:
            continue
        manifest_path = hypothesis_root / hypothesis_id / "code_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json(manifest_path)
        entry = _entry_for_active_file(manifest, agent_mode=agent_mode)
        if entry is None:
            continue

        expected_status = _expected_status(entry)
        expected_is_buggy = _expected_is_buggy(entry, expected_status)
        expected_metric = _expected_metric(entry, expected_status)
        expected = {
            "status": expected_status,
            "is_buggy": expected_is_buggy,
            "metric": expected_metric,
        }
        for field, after in expected.items():
            before = node.get(field)
            if not _values_equal(field, before, after):
                changes.append(
                    PlannedChange(
                        node_index=index,
                        node_id=str(node.get("id") or ""),
                        step=node.get("step") if isinstance(node.get("step"), int) else None,
                        hypothesis_id=hypothesis_id,
                        manifest_file=(
                            entry.get("file") if isinstance(entry.get("file"), str) else None
                        ),
                        field=field,
                        before=before,
                        after=after,
                    )
                )
    return changes


def _apply_changes(journal: dict[str, Any], changes: list[PlannedChange]) -> None:
    nodes = journal["nodes"]
    for change in changes:
        node = nodes[change.node_index]
        if change.after is None:
            node[change.field] = None
        else:
            node[change.field] = change.after


def _format_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _node_group_key(change: PlannedChange) -> tuple[int, str]:
    return change.node_index, change.node_id


def _print_changes(changes: list[PlannedChange]) -> None:
    if not changes:
        print("No root hypothesis status mismatches found.")
        return

    console = Console()
    node_count = len({_node_group_key(change) for change in changes})
    console.print(
        f"Planned root hypothesis journal updates: "
        f"{len(changes)} field change(s) across {node_count} node(s)"
    )

    table = Table(
        box=box.SIMPLE,
        show_lines=False,
        pad_edge=False,
        expand=False,
    )
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("node", no_wrap=True)
    table.add_column("hyp", no_wrap=True)
    table.add_column("source", no_wrap=True)
    table.add_column("field", no_wrap=True)
    table.add_column("journal")
    table.add_column("manifest")

    group_styles = ["on grey11", "on grey23"]
    previous_group: tuple[int, str] | None = None
    group_index = -1
    for change in changes:
        group = _node_group_key(change)
        step = "?" if change.step is None else str(change.step)
        if group != previous_group:
            group_index += 1
            previous_group = group
            step_label = step
            node_label = change.node_id[:8]
            hypothesis_label = change.hypothesis_id
            source_label = change.manifest_file or "manifest"
        else:
            step_label = ""
            node_label = ""
            hypothesis_label = ""
            source_label = ""
        style = group_styles[group_index % len(group_styles)]
        table.add_row(
            step_label,
            node_label,
            hypothesis_label,
            source_label,
            change.field,
            _format_value(change.before),
            _format_value(change.after),
            style=style,
        )
    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair root hypothesis node status fields in a run journal using "
            "research_hypotheses/<task>/<id>/code_manifest.json as the source of truth."
        )
    )
    parser.add_argument("run_id", help="Run id under logs/, for example 2-name")
    parser.add_argument("--logs-dir", default="logs", help="Top-level logs directory")
    parser.add_argument(
        "--hypotheses-dir",
        default="research_hypotheses",
        help="Top-level research_hypotheses directory",
    )
    parser.add_argument(
        "--task-slug",
        default=None,
        help="Task slug under research_hypotheses/. Defaults to config.yaml data_dir name.",
    )
    parser.add_argument(
        "--agent-mode",
        default=None,
        help="Manifest mode to use, e.g. autogluon or legacy. Defaults from config.yaml agent.mode.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.logs_dir) / args.run_id
    journal_path = run_dir / "journal.json"
    if not journal_path.exists():
        raise FileNotFoundError(f"Missing journal: {journal_path}")

    task_slug = args.task_slug or _task_slug_from_config(run_dir)
    if not task_slug:
        raise ValueError("Could not infer task slug; pass --task-slug.")
    agent_mode = args.agent_mode or _agent_mode_from_config(run_dir)
    if not agent_mode:
        raise ValueError("Could not infer agent mode; pass --agent-mode.")

    hypothesis_root = Path(args.hypotheses_dir) / task_slug
    if not hypothesis_root.exists():
        raise FileNotFoundError(f"Missing hypothesis directory: {hypothesis_root}")

    journal = _read_json(journal_path)
    changes = _plan_changes(
        journal=journal,
        hypothesis_root=hypothesis_root,
        agent_mode=agent_mode,
    )
    print(f"Run: {args.run_id}")
    print(f"Journal: {journal_path}")
    print(f"Hypotheses: {hypothesis_root}")
    print(f"Agent mode: {agent_mode}")
    _print_changes(changes)

    if args.dry_run or not changes:
        return 0

    answer = input("Update journal.json with these changes? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        print("Aborted; journal unchanged.")
        return 1

    backup_path = _backup_journal(journal_path)
    print(f"Backup written: {backup_path}")
    _apply_changes(journal, changes)
    _write_json_atomic(journal_path, journal)
    print(f"Updated {journal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
