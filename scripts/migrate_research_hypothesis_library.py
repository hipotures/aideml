from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - command line environment diagnostic
    yaml = None


HYPOTHESIS_RE = re.compile(r"^hypothesis-(\d{6})\.json$")
ROOT_CODE_RE = re.compile(r"^(autogluon|legacy)-(\d{3})\.py$")


@dataclass(frozen=True)
class HypothesisMove:
    source: Path
    destination: Path
    hypothesis_id: str


@dataclass(frozen=True)
class MigrationPlan:
    task_dir: Path
    moves: tuple[HypothesisMove, ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class ExportEntry:
    node_id: str
    hypothesis_id: str
    mode: str
    destination: Path
    code: str
    buggy: bool
    score: float | None
    timestamp: str | None


@dataclass(frozen=True)
class ExportPlan:
    journal_path: Path
    entries: tuple[ExportEntry, ...]
    conflicts: tuple[str, ...]
    skipped: tuple[str, ...]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _task_dir(root: Path, task: str) -> Path:
    return root / "research_hypotheses" / task


def plan_structure_migration(*, root: Path, task: str) -> MigrationPlan:
    task_dir = _task_dir(root, task)
    old_dir = task_dir / "hypotheses"
    conflicts: list[str] = []
    moves: list[HypothesisMove] = []
    if not old_dir.exists():
        conflicts.append(f"Missing legacy hypotheses directory: {old_dir}")
        return MigrationPlan(task_dir=task_dir, moves=(), conflicts=tuple(conflicts))

    for source in sorted(old_dir.glob("hypothesis-*.json")):
        match = HYPOTHESIS_RE.match(source.name)
        if match is None:
            conflicts.append(f"Invalid hypothesis filename: {source}")
            continue
        hypothesis_id = match.group(1)
        destination = task_dir / hypothesis_id / source.name
        if destination.exists():
            conflicts.append(f"Destination already exists: {destination}")
            continue
        moves.append(
            HypothesisMove(
                source=source,
                destination=destination,
                hypothesis_id=hypothesis_id,
            )
        )

    if not moves and not conflicts:
        conflicts.append(f"No hypothesis JSON files found in {old_dir}")
    return MigrationPlan(
        task_dir=task_dir,
        moves=tuple(moves),
        conflicts=tuple(conflicts),
    )


def apply_structure_migration(plan: MigrationPlan, *, dry_run: bool) -> None:
    if dry_run:
        return
    if plan.conflicts:
        raise ValueError("Refusing to migrate while conflicts are present.")
    for move in plan.moves:
        move.destination.parent.mkdir(parents=True, exist_ok=True)
        move.source.replace(move.destination)
    legacy_dir = plan.task_dir / "hypotheses"
    try:
        legacy_dir.rmdir()
    except OSError:
        pass


def _pathlib_tolerant_yaml_loader():
    if yaml is None:
        return None

    class Loader(yaml.SafeLoader):
        pass

    def construct_path(loader, node):
        value = loader.construct_sequence(node)
        return str(Path(*value))

    Loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
        construct_path,
    )
    return Loader


def _read_run_mode(run_dir: Path) -> str | None:
    if yaml is None:
        return None
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    loader = _pathlib_tolerant_yaml_loader()
    if loader is None:
        return None
    payload = yaml.load(config_path.read_text(encoding="utf-8"), Loader=loader)
    if not isinstance(payload, dict):
        return None
    agent = payload.get("agent")
    if not isinstance(agent, dict):
        return None
    mode = agent.get("mode")
    return mode if isinstance(mode, str) else None


def _normalize_agent_mode(mode: str | None) -> str:
    if mode in {"autogluon", "autogluon_preprocess"}:
        return "autogluon"
    return "legacy"


def _journal_root_nodes(journal_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, list):
        raise ValueError(f"Invalid journal format: {journal_path}")
    node2parent = payload.get("node2parent")
    if isinstance(node2parent, dict):
        child_ids = {str(node_id) for node_id in node2parent}
        return [
            node
            for node in nodes
            if isinstance(node, dict)
            and str(node.get("id") or "") not in child_ids
            and node.get("research_mode") == "hypothesis"
        ]
    return [
        node
        for node in nodes
        if isinstance(node, dict)
        and node.get("parent") is None
        and node.get("research_mode") == "hypothesis"
    ]


def _node_timestamp(node: dict[str, Any]) -> str | None:
    ctime = node.get("ctime")
    if isinstance(ctime, (float, int)) and not isinstance(ctime, bool):
        return dt.datetime.fromtimestamp(float(ctime)).isoformat(timespec="seconds")
    return None


def _node_score(node: dict[str, Any]) -> float | None:
    metric = node.get("metric")
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value)


def _legacy_artifact_dir_name(node: dict[str, Any]) -> str | None:
    ctime = node.get("ctime")
    if isinstance(ctime, (float, int)) and not isinstance(ctime, bool):
        return dt.datetime.fromtimestamp(float(ctime)).strftime("%Y%m%dT%H%M%S")
    return None


def _node_code_path(journal_path: Path, node: dict[str, Any]) -> Path:
    code_path = node.get("code_path")
    if isinstance(code_path, str) and code_path.strip():
        return journal_path.parent / code_path.strip()
    artifact_dir_name = node.get("artifact_dir_name")
    if isinstance(artifact_dir_name, str) and artifact_dir_name.strip():
        return journal_path.parent / "artifacts" / artifact_dir_name.strip() / "solution.py"
    legacy_name = _legacy_artifact_dir_name(node)
    if legacy_name is not None:
        return journal_path.parent / "artifacts" / legacy_name / "solution.py"
    raise ValueError(f"Node {node.get('id')} has no code_path or artifact path.")


def _node_code(journal_path: Path, node: dict[str, Any]) -> str:
    return _node_code_path(journal_path, node).read_text(encoding="utf-8")


def plan_root_code_export(
    *,
    root: Path,
    task: str,
    journal_path: Path,
    agent_mode: str | None = None,
) -> ExportPlan:
    mode = _normalize_agent_mode(agent_mode or _read_run_mode(journal_path.parent))
    task_dir = _task_dir(root, task)
    conflicts: list[str] = []
    skipped: list[str] = []
    root_nodes_by_hypothesis: dict[str, dict[str, Any]] = {}

    for node in _journal_root_nodes(journal_path):
        offered = node.get("research_hypotheses_offered")
        if (
            not isinstance(offered, list)
            or len(offered) != 1
            or not isinstance(offered[0], str)
        ):
            skipped.append(f"Root node {node.get('id')} has no single hypothesis id.")
            continue
        hypothesis_id = offered[0]
        try:
            code = _node_code(journal_path, node)
        except (OSError, ValueError) as exc:
            skipped.append(f"Root node {node.get('id')} has no artifact code: {exc}")
            continue
        if not code.strip():
            skipped.append(f"Root node {node.get('id')} has empty artifact code.")
            continue
        hypothesis_dir = task_dir / hypothesis_id
        hypothesis_json = hypothesis_dir / f"hypothesis-{hypothesis_id}.json"
        if not hypothesis_json.exists():
            conflicts.append(
                f"Missing flat hypothesis JSON for {hypothesis_id}: "
                f"{hypothesis_json}"
            )
            continue

        existing = root_nodes_by_hypothesis.get(hypothesis_id)
        if existing is not None:
            conflicts.append(
                "Multiple journal ROOT nodes for hypothesis "
                f"{hypothesis_id}: {existing.get('id')} and {node.get('id')}"
            )
            continue
        root_nodes_by_hypothesis[hypothesis_id] = node

    entries: list[ExportEntry] = []
    conflicted_ids = {
        match.group(1)
        for conflict in conflicts
        if (match := re.search(r"hypothesis (\d{6})", conflict))
    }
    for hypothesis_id, node in sorted(root_nodes_by_hypothesis.items()):
        if hypothesis_id in conflicted_ids:
            continue
        hypothesis_dir = task_dir / hypothesis_id
        entries.append(
            ExportEntry(
                node_id=str(node.get("id") or ""),
                hypothesis_id=hypothesis_id,
                mode=mode,
                destination=hypothesis_dir / f"{mode}-001.py",
                code=_node_code(journal_path, node),
                buggy=bool(node.get("is_buggy")),
                score=_node_score(node),
                timestamp=_node_timestamp(node),
            )
        )

    return ExportPlan(
        journal_path=journal_path,
        entries=tuple(entries),
        conflicts=tuple(conflicts),
        skipped=tuple(skipped),
    )


def _load_manifest(hypothesis_dir: Path) -> dict[str, Any]:
    path = hypothesis_dir / "code_manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def apply_root_code_export(plan: ExportPlan, *, dry_run: bool) -> None:
    if dry_run:
        return
    if plan.conflicts:
        raise ValueError("Refusing to export while conflicts are present.")
    for entry in plan.entries:
        entry.destination.parent.mkdir(parents=True, exist_ok=True)
        entry.destination.write_text(entry.code, encoding="utf-8")

        manifest_path = entry.destination.parent / "code_manifest.json"
        manifest = _load_manifest(entry.destination.parent)
        active = manifest.setdefault("active", {})
        if not isinstance(active, dict):
            manifest["active"] = active = {}
        versions = manifest.setdefault("versions", {})
        if not isinstance(versions, dict):
            manifest["versions"] = versions = {}
        mode_versions = versions.setdefault(entry.mode, [])
        if not isinstance(mode_versions, list):
            versions[entry.mode] = mode_versions = []
        exported_entry = {
            "file": entry.destination.name,
            "node_id": entry.node_id,
            "created_at": entry.timestamp,
            "buggy": entry.buggy,
            "score": entry.score,
        }
        versions[entry.mode] = [
            item
            for item in mode_versions
            if not (isinstance(item, dict) and item.get("file") == entry.destination.name)
        ]
        versions[entry.mode].append(exported_entry)
        if not entry.buggy:
            active[entry.mode] = entry.destination.name
        else:
            active.pop(entry.mode, None)
        _write_json(manifest_path, manifest)


def _print_migration_report(plan: MigrationPlan, *, dry_run: bool) -> None:
    action = "Would move" if dry_run else "Moved"
    ids = {move.hypothesis_id for move in plan.moves}
    print(f"{action} {len(plan.moves)} hypothesis files into {len(ids)} directories.")
    if plan.conflicts:
        print(f"Conflicts: {len(plan.conflicts)}")
        for conflict in plan.conflicts:
            print(f"! {conflict}")
    for move in plan.moves:
        print(f"- {move.source} -> {move.destination}")


def _print_export_report(plan: ExportPlan, *, dry_run: bool) -> None:
    action = "Would export" if dry_run else "Exported"
    by_mode: dict[str, int] = {}
    buggy = 0
    for entry in plan.entries:
        by_mode[entry.mode] = by_mode.get(entry.mode, 0) + 1
        buggy += int(entry.buggy)
    print(f"{action} {len(plan.entries)} root code files from {plan.journal_path}.")
    for mode, count in sorted(by_mode.items()):
        print(f"- {mode}: {count}")
    print(f"Buggy versions: {buggy}")
    if plan.conflicts:
        print(f"Conflicts: {len(plan.conflicts)}")
        for conflict in plan.conflicts:
            print(f"! {conflict}")
    if plan.skipped:
        print(f"Skipped: {len(plan.skipped)}")
        for skipped in plan.skipped:
            print(f"! {skipped}")
    for entry in plan.entries:
        status = "buggy" if entry.buggy else "ok"
        print(f"- {entry.node_id} {entry.hypothesis_id} {status} -> {entry.destination}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate research hypotheses to flat per-id directories and export root code.",
    )
    parser.add_argument("--task", default="playground-series-s6e6")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--dry-run", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("migrate-structure")

    export_parser = subparsers.add_parser("export-root-code")
    export_parser.add_argument("journal", type=Path)
    export_parser.add_argument(
        "--agent-mode",
        choices=["legacy", "autogluon", "autogluon_preprocess"],
        default=None,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.repo_root.resolve()
    try:
        if args.command == "migrate-structure":
            plan = plan_structure_migration(root=root, task=args.task)
            apply_structure_migration(plan, dry_run=args.dry_run)
            _print_migration_report(plan, dry_run=args.dry_run)
            return 1 if plan.conflicts else 0
        if args.command == "export-root-code":
            plan = plan_root_code_export(
                root=root,
                task=args.task,
                journal_path=args.journal,
                agent_mode=args.agent_mode,
            )
            apply_root_code_export(plan, dry_run=args.dry_run)
            _print_export_report(plan, dry_run=args.dry_run)
            return 1 if plan.conflicts else 0
    except Exception as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
