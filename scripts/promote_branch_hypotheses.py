from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HYPOTHESIS_RE = re.compile(r"^hypothesis-(\d{6})\.json$")


@dataclass(frozen=True)
class PromotionEntry:
    hypothesis_id: str
    source_run_id: str
    source_node_id: str
    source_agent_mode: str
    source_score: float
    source_aux: bool
    source_branch_path: tuple[str, ...]
    source_artifact_dir: str | None
    code: str
    created_at: str
    destination_dir: Path


@dataclass(frozen=True)
class PromotionPlan:
    root: Path
    task: str
    journal_path: Path
    created: tuple[PromotionEntry, ...]
    existing: tuple[PromotionEntry, ...]
    conflicts: tuple[str, ...]
    skipped: tuple[str, ...]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_agent_mode(mode: str | None) -> str:
    if mode in {"autogluon", "autogluon_preprocess"}:
        return "autogluon"
    return "legacy"


def _read_run_config_text(run_dir: Path) -> str:
    config_path = run_dir / "config.yaml"
    try:
        return config_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_run_agent_mode(run_dir: Path) -> str:
    text = _read_run_config_text(run_dir)
    match = re.search(r"(?m)^\s*mode:\s*([A-Za-z0-9_]+)\s*$", text)
    return _normalize_agent_mode(match.group(1) if match else None)


def _read_run_aux(run_dir: Path) -> bool:
    text = _read_run_config_text(run_dir)
    match = re.search(r"(?m)^\s*aux:\s*(true|false|True|False)\s*$", text)
    return bool(match and match.group(1).lower() == "true")


def _load_journal(journal_path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes") if isinstance(payload, dict) else None
    if not isinstance(nodes, list):
        raise ValueError(f"Invalid journal format: {journal_path}")
    node_dicts = [node for node in nodes if isinstance(node, dict)]
    node2parent = payload.get("node2parent") if isinstance(payload, dict) else None
    parent_by_id: dict[str, str] = {}
    if isinstance(node2parent, dict):
        parent_by_id.update(
            {
                str(node_id): str(parent_id)
                for node_id, parent_id in node2parent.items()
                if parent_id is not None
            }
        )
    else:
        for node in node_dicts:
            node_id = str(node.get("id") or "")
            parent = node.get("parent")
            if isinstance(parent, dict):
                parent_id = parent.get("id")
            else:
                parent_id = parent
            if node_id and parent_id:
                parent_by_id[node_id] = str(parent_id)
    return node_dicts, parent_by_id


def _node_score(node: dict[str, Any]) -> tuple[float, float] | None:
    metric = node.get("metric")
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    rank_score = score if metric.get("maximize") is not False else -score
    return score, rank_score


def _hypothesis_id_for_node(node: dict[str, Any]) -> str | None:
    offered = node.get("research_hypotheses_offered")
    if (
        isinstance(offered, list)
        and len(offered) == 1
        and isinstance(offered[0], str)
    ):
        return offered[0]
    return None


def _branch_path(
    node: dict[str, Any],
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    parent_by_id: dict[str, str],
) -> tuple[str, ...]:
    path: list[str] = []
    current: dict[str, Any] | None = node
    seen: set[str] = set()
    while current is not None:
        node_id = str(current.get("id") or "")
        if not node_id or node_id in seen:
            break
        seen.add(node_id)
        hypothesis_id = _hypothesis_id_for_node(current)
        if hypothesis_id is not None:
            path.append(hypothesis_id)
        parent_id = parent_by_id.get(node_id)
        current = nodes_by_id.get(parent_id) if parent_id is not None else None
    return tuple(reversed(path))


def _next_hypothesis_number(task_dir: Path) -> int:
    max_id = 0
    if not task_dir.exists():
        return 1
    for path in task_dir.glob("*/hypothesis-*.json"):
        if not path.parent.name.isdigit():
            continue
        match = HYPOTHESIS_RE.match(path.name)
        if match is not None and match.group(1) == path.parent.name:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def _existing_promotions(task_dir: Path) -> dict[tuple[str, str], str]:
    existing: dict[tuple[str, str], str] = {}
    for path in sorted(task_dir.glob("*/hypothesis-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        origin = payload.get("origin")
        if not isinstance(origin, dict):
            continue
        source_run_id = origin.get("source_run_id")
        source_node_id = origin.get("source_node_id")
        if isinstance(source_run_id, str) and isinstance(source_node_id, str):
            existing[(source_run_id, source_node_id)] = path.parent.name
    return existing


def _source_artifact_dir(journal_path: Path, node: dict[str, Any]) -> str | None:
    artifact_dir_name = node.get("artifact_dir_name")
    if not isinstance(artifact_dir_name, str) or not artifact_dir_name:
        return None
    run_dir = journal_path.parent
    try:
        relative_run_dir = run_dir.relative_to(journal_path.parents[2])
    except ValueError:
        relative_run_dir = Path(run_dir.name)
    return (relative_run_dir / "artifacts" / artifact_dir_name).as_posix()


def _node_timestamp(node: dict[str, Any]) -> str:
    ctime = node.get("ctime")
    if isinstance(ctime, int | float) and not isinstance(ctime, bool):
        return dt.datetime.fromtimestamp(float(ctime)).isoformat(timespec="seconds")
    return dt.datetime.now().isoformat(timespec="seconds")


def _promotion_entry(
    *,
    hypothesis_id: str,
    source_run_id: str,
    source_agent_mode: str,
    source_aux: bool,
    node: dict[str, Any],
    score: float,
    branch_path: tuple[str, ...],
    artifact_dir: str | None,
    destination_dir: Path,
) -> PromotionEntry:
    return PromotionEntry(
        hypothesis_id=hypothesis_id,
        source_run_id=source_run_id,
        source_node_id=str(node.get("id") or ""),
        source_agent_mode=source_agent_mode,
        source_score=score,
        source_aux=source_aux,
        source_branch_path=branch_path,
        source_artifact_dir=artifact_dir,
        code=str(node["code"]),
        created_at=_node_timestamp(node),
        destination_dir=destination_dir,
    )


def plan_branch_hypothesis_promotion(
    *,
    root: Path,
    task: str,
    journal_path: Path,
    top_n: int,
    agent_mode: str | None = None,
) -> PromotionPlan:
    if top_n <= 0:
        raise ValueError("--top-n must be greater than zero")

    root = Path(root)
    task_dir = root / "research_hypotheses" / task
    source_run_id = journal_path.parent.name
    source_agent_mode = _normalize_agent_mode(agent_mode or _read_run_agent_mode(journal_path.parent))
    source_aux = _read_run_aux(journal_path.parent)
    nodes, parent_by_id = _load_journal(journal_path)
    nodes_by_id = {str(node.get("id") or ""): node for node in nodes}
    existing_by_source = _existing_promotions(task_dir)
    conflicts: list[str] = []
    skipped: list[str] = []

    candidates: list[tuple[float, float, dict[str, Any]]] = []
    for node in nodes:
        node_id = str(node.get("id") or "")
        if node_id not in parent_by_id:
            continue
        if node.get("research_mode") != "hypothesis":
            continue
        if node.get("is_buggy") is True:
            continue
        code = node.get("code")
        if not isinstance(code, str) or not code.strip():
            skipped.append(f"Branch node {node_id} has empty code.")
            continue
        score = _node_score(node)
        if score is None:
            skipped.append(f"Branch node {node_id} has no numeric score.")
            continue
        candidates.append((score[1], score[0], node))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:top_n]
    next_id = _next_hypothesis_number(task_dir)
    created: list[PromotionEntry] = []
    existing: list[PromotionEntry] = []

    for _rank_score, score, node in selected:
        source_node_id = str(node.get("id") or "")
        existing_id = existing_by_source.get((source_run_id, source_node_id))
        branch_path = _branch_path(
            node,
            nodes_by_id=nodes_by_id,
            parent_by_id=parent_by_id,
        )
        artifact_dir = _source_artifact_dir(journal_path, node)
        if existing_id is not None:
            existing.append(
                _promotion_entry(
                    hypothesis_id=existing_id,
                    source_run_id=source_run_id,
                    source_agent_mode=source_agent_mode,
                    source_aux=source_aux,
                    node=node,
                    score=score,
                    branch_path=branch_path,
                    artifact_dir=artifact_dir,
                    destination_dir=task_dir / existing_id,
                )
            )
            continue
        while (task_dir / f"{next_id:06d}").exists():
            next_id += 1
        hypothesis_id = f"{next_id:06d}"
        next_id += 1
        created.append(
            _promotion_entry(
                hypothesis_id=hypothesis_id,
                source_run_id=source_run_id,
                source_agent_mode=source_agent_mode,
                source_aux=source_aux,
                node=node,
                score=score,
                branch_path=branch_path,
                artifact_dir=artifact_dir,
                destination_dir=task_dir / hypothesis_id,
            )
        )

    return PromotionPlan(
        root=root,
        task=task,
        journal_path=journal_path,
        created=tuple(created),
        existing=tuple(existing),
        conflicts=tuple(conflicts),
        skipped=tuple(skipped),
    )


def _hypothesis_payload(entry: PromotionEntry) -> dict[str, Any]:
    mode = entry.source_agent_mode
    return {
        "enabled": True,
        "agent_modes": [mode],
        "title": f"Promoted branch {entry.source_node_id[:8]}",
        "summary": (
            "Promoted high-scoring branch solution so it can be evaluated as a "
            "root hypothesis."
        ),
        "rationale": (
            "The source branch produced a strong validation score and should be "
            "re-tested independently from a root position."
        ),
        "implementation_hint": (
            "Use the promoted root code exactly as the starting implementation."
        ),
        "expected_effect": (
            "Preserve the source branch behavior while allowing normal root "
            "selection and aux-data evaluation."
        ),
        "risk": "The promoted branch may overfit the original run context.",
        "sources": [
            f"aide://{entry.source_run_id}/journal.json#{entry.source_node_id}"
        ],
        "origin": {
            "kind": "promoted_branch",
            "source_run_id": entry.source_run_id,
            "source_node_id": entry.source_node_id,
            "source_agent_mode": entry.source_agent_mode,
            "source_score": entry.source_score,
            "source_aux": entry.source_aux,
            "source_branch_path": list(entry.source_branch_path),
            "source_artifact_dir": entry.source_artifact_dir,
        },
    }


def _manifest_payload(entry: PromotionEntry) -> dict[str, Any]:
    mode = entry.source_agent_mode
    file_name = f"{mode}-001.py"
    manifest_entry = {
        "file": file_name,
        "buggy": False,
        "status": "ok",
        "node_id": entry.source_node_id,
        "score": entry.source_score,
        "created_at": entry.created_at,
        "aux": entry.source_aux,
        "source_run_id": entry.source_run_id,
        "source_node_id": entry.source_node_id,
        "source_agent_mode": entry.source_agent_mode,
        "source_score": entry.source_score,
        "source_aux": entry.source_aux,
        "source_branch_path": list(entry.source_branch_path),
        "source_artifact_dir": entry.source_artifact_dir,
    }
    return {
        "active": {mode: file_name},
        "versions": {mode: [manifest_entry]},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def apply_promotion_plan(plan: PromotionPlan, *, dry_run: bool) -> None:
    if dry_run:
        return
    if plan.conflicts:
        raise ValueError("Refusing to promote while conflicts are present.")
    for entry in plan.created:
        entry.destination_dir.mkdir(parents=True, exist_ok=False)
        hypothesis_path = (
            entry.destination_dir / f"hypothesis-{entry.hypothesis_id}.json"
        )
        mode = entry.source_agent_mode
        code_path = entry.destination_dir / f"{mode}-001.py"
        manifest_path = entry.destination_dir / "code_manifest.json"
        _write_json(hypothesis_path, _hypothesis_payload(entry))
        code_path.write_text(entry.code, encoding="utf-8")
        _write_json(manifest_path, _manifest_payload(entry))


def _print_report(plan: PromotionPlan, *, dry_run: bool) -> None:
    action = "Would create" if dry_run else "Created"
    print(
        f"{action} {len(plan.created)} promoted root hypotheses "
        f"from {plan.journal_path}."
    )
    print(f"Existing: {len(plan.existing)}")
    if plan.conflicts:
        print(f"Conflicts: {len(plan.conflicts)}")
        for conflict in plan.conflicts:
            print(f"! {conflict}")
    if plan.skipped:
        print(f"Skipped: {len(plan.skipped)}")
        for skipped in plan.skipped:
            print(f"! {skipped}")
    for entry in plan.existing:
        print(
            f"= {entry.source_node_id} -> {entry.hypothesis_id} "
            f"score={entry.source_score:.5f}"
        )
    for entry in plan.created:
        print(
            f"+ {entry.source_node_id} -> {entry.hypothesis_id} "
            f"score={entry.source_score:.5f}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote top scored branch hypothesis nodes to new ROOT hypotheses.",
    )
    parser.add_argument("journal", type=Path)
    parser.add_argument("--task", default="playground-series-s6e5")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument("--top-n", type=int, required=True)
    parser.add_argument(
        "--agent-mode",
        choices=["legacy", "autogluon", "autogluon_preprocess"],
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = plan_branch_hypothesis_promotion(
            root=args.repo_root.resolve(),
            task=args.task,
            journal_path=args.journal,
            top_n=args.top_n,
            agent_mode=args.agent_mode,
        )
        apply_promotion_plan(plan, dry_run=args.dry_run)
        _print_report(plan, dry_run=args.dry_run)
        return 1 if plan.conflicts else 0
    except Exception as exc:
        print(f"Promotion failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
