from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.progress import Progress


HYPOTHESIS_RE = re.compile(r"^hypothesis-(\d{6})\.json$")


@dataclass(frozen=True)
class PromotionEntry:
    hypothesis_id: str
    source_run_id: str
    source_node_id: str
    source_step: int | None
    source_kind: str
    source_agent_mode: str
    source_score: float
    source_aux: bool
    source_branch_path: tuple[str, ...]
    source_artifact_dir: str | None
    source_plan: str | None
    source_analysis: str | None
    source_validity_warning: str | None
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


@dataclass(frozen=True)
class PromotionCandidate:
    rank_score: float
    source_score: float
    code: str
    journal_path: Path
    source_run_id: str
    source_agent_mode: str
    source_aux: bool
    node: dict[str, Any]
    branch_path: tuple[str, ...]
    artifact_dir: str | None


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


def _legacy_artifact_dir_name(node: dict[str, Any]) -> str | None:
    ctime = node.get("ctime")
    if isinstance(ctime, int | float) and not isinstance(ctime, bool):
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


def _node_timestamp(node: dict[str, Any]) -> str:
    ctime = node.get("ctime")
    if isinstance(ctime, int | float) and not isinstance(ctime, bool):
        return dt.datetime.fromtimestamp(float(ctime)).isoformat(timespec="seconds")
    return dt.datetime.now().isoformat(timespec="seconds")


def _node_step(node: dict[str, Any]) -> int | None:
    step = node.get("step")
    if isinstance(step, bool):
        return None
    if isinstance(step, int):
        return step
    if isinstance(step, float) and step.is_integer():
        return int(step)
    return None


def _node_text(node: dict[str, Any], key: str) -> str | None:
    value = node.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


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
    code: str,
    source_kind: str = "promoted_branch",
) -> PromotionEntry:
    return PromotionEntry(
        hypothesis_id=hypothesis_id,
        source_run_id=source_run_id,
        source_node_id=str(node.get("id") or ""),
        source_step=_node_step(node),
        source_kind=source_kind,
        source_agent_mode=source_agent_mode,
        source_score=score,
        source_aux=source_aux,
        source_branch_path=branch_path,
        source_artifact_dir=artifact_dir,
        source_plan=_node_text(node, "plan"),
        source_analysis=_node_text(node, "analysis"),
        source_validity_warning=_node_text(node, "validity_warning"),
        code=code,
        created_at=_node_timestamp(node),
        destination_dir=destination_dir,
    )


def _branch_candidates_from_journal(
    *,
    journal_path: Path,
    agent_mode: str | None,
    skipped: list[str],
) -> list[PromotionCandidate]:
    source_run_id = journal_path.parent.name
    source_agent_mode = _normalize_agent_mode(
        agent_mode or _read_run_agent_mode(journal_path.parent)
    )
    source_aux = _read_run_aux(journal_path.parent)
    nodes, parent_by_id = _load_journal(journal_path)
    nodes_by_id = {str(node.get("id") or ""): node for node in nodes}
    candidates: list[PromotionCandidate] = []
    for node in nodes:
        node_id = str(node.get("id") or "")
        if node_id not in parent_by_id:
            continue
        if node.get("research_mode") != "hypothesis":
            continue
        if node.get("is_buggy") is True:
            continue
        try:
            code = _node_code(journal_path, node)
        except (OSError, ValueError) as exc:
            skipped.append(f"{source_run_id}:{node_id} branch node has no artifact code: {exc}")
            continue
        if not code.strip():
            skipped.append(f"{source_run_id}:{node_id} branch node has empty artifact code.")
            continue
        score = _node_score(node)
        if score is None:
            skipped.append(
                f"{source_run_id}:{node_id} branch node has no numeric score."
            )
            continue
        candidates.append(
            PromotionCandidate(
                rank_score=score[1],
                source_score=score[0],
                code=code,
                journal_path=journal_path,
                source_run_id=source_run_id,
                source_agent_mode=source_agent_mode,
                source_aux=source_aux,
                node=node,
                branch_path=_branch_path(
                    node,
                    nodes_by_id=nodes_by_id,
                    parent_by_id=parent_by_id,
                ),
                artifact_dir=_source_artifact_dir(journal_path, node),
            )
        )
    return candidates


def _promotion_kind_for_node(node: dict[str, Any]) -> str:
    return (
        "promoted_branch"
        if _hypothesis_id_for_node(node) is not None
        else "promoted_classic_node"
    )


def _plan_from_candidates(
    *,
    root: Path,
    task: str,
    journal_path: Path,
    candidates: list[PromotionCandidate],
    top_n: int,
    skipped: list[str],
) -> PromotionPlan:
    if top_n <= 0:
        raise ValueError("--top-n must be greater than zero")

    task_dir = root / "research_hypotheses" / task
    existing_by_source = _existing_promotions(task_dir)
    conflicts: list[str] = []

    candidates.sort(key=lambda item: item.rank_score, reverse=True)
    selected = candidates[:top_n]
    next_id = _next_hypothesis_number(task_dir)
    created: list[PromotionEntry] = []
    existing: list[PromotionEntry] = []

    for candidate in selected:
        source_node_id = str(candidate.node.get("id") or "")
        existing_id = existing_by_source.get(
            (candidate.source_run_id, source_node_id)
        )
        if existing_id is not None:
            existing.append(
                _promotion_entry(
                    hypothesis_id=existing_id,
                    source_run_id=candidate.source_run_id,
                    source_agent_mode=candidate.source_agent_mode,
                    source_aux=candidate.source_aux,
                    node=candidate.node,
                    score=candidate.source_score,
                    branch_path=candidate.branch_path,
                    artifact_dir=candidate.artifact_dir,
                    destination_dir=task_dir / existing_id,
                    code=candidate.code,
                    source_kind=_promotion_kind_for_node(candidate.node),
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
                source_run_id=candidate.source_run_id,
                source_agent_mode=candidate.source_agent_mode,
                source_aux=candidate.source_aux,
                node=candidate.node,
                score=candidate.source_score,
                branch_path=candidate.branch_path,
                artifact_dir=candidate.artifact_dir,
                destination_dir=task_dir / hypothesis_id,
                code=candidate.code,
                source_kind=_promotion_kind_for_node(candidate.node),
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
    skipped: list[str] = []
    candidates = _branch_candidates_from_journal(
        journal_path=journal_path,
        agent_mode=agent_mode,
        skipped=skipped,
    )
    return _plan_from_candidates(
        root=root,
        task=task,
        journal_path=journal_path,
        candidates=candidates,
        top_n=top_n,
        skipped=skipped,
    )


def plan_node_promotion(
    *,
    root: Path,
    task: str,
    journal_path: Path,
    step: int | None,
    node_id: str | None,
    agent_mode: str | None = None,
) -> PromotionPlan:
    if step is None and not node_id:
        raise ValueError("Either step or node_id must be provided.")

    root = Path(root)
    source_run_id = journal_path.parent.name
    source_agent_mode = _normalize_agent_mode(
        agent_mode or _read_run_agent_mode(journal_path.parent)
    )
    source_aux = _read_run_aux(journal_path.parent)
    nodes, parent_by_id = _load_journal(journal_path)
    nodes_by_id = {str(node.get("id") or ""): node for node in nodes}

    selected: dict[str, Any] | None = None
    for node in nodes:
        current_node_id = str(node.get("id") or "")
        if node_id and current_node_id != node_id:
            continue
        if step is not None and _node_step(node) != step:
            continue
        selected = node
        break

    if selected is None:
        label = f"node_id={node_id!r}" if node_id else f"step={step!r}"
        raise ValueError(f"No node matching {label} found in {journal_path}.")
    if selected.get("is_buggy") is True:
        raise ValueError("Refusing to promote a buggy node.")
    try:
        code = _node_code(journal_path, selected)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Refusing to promote a node without artifact code: {exc}") from exc
    if not code.strip():
        raise ValueError("Refusing to promote a node with empty artifact code.")
    score = _node_score(selected)
    if score is None:
        raise ValueError("Refusing to promote a node without a numeric score.")

    task_dir = root / "research_hypotheses" / task
    source_node_id = str(selected.get("id") or "")
    existing_id = _existing_promotions(task_dir).get((source_run_id, source_node_id))
    branch_path = _branch_path(
        selected,
        nodes_by_id=nodes_by_id,
        parent_by_id=parent_by_id,
    )
    artifact_dir = _source_artifact_dir(journal_path, selected)
    source_kind = _promotion_kind_for_node(selected)
    existing: tuple[PromotionEntry, ...] = ()
    created: tuple[PromotionEntry, ...] = ()

    if existing_id is not None:
        existing = (
            _promotion_entry(
                hypothesis_id=existing_id,
                source_run_id=source_run_id,
                source_agent_mode=source_agent_mode,
                source_aux=source_aux,
                node=selected,
                score=score[0],
                branch_path=branch_path,
                artifact_dir=artifact_dir,
                destination_dir=task_dir / existing_id,
                code=code,
                source_kind=source_kind,
            ),
        )
    else:
        next_id = _next_hypothesis_number(task_dir)
        while (task_dir / f"{next_id:06d}").exists():
            next_id += 1
        hypothesis_id = f"{next_id:06d}"
        created = (
            _promotion_entry(
                hypothesis_id=hypothesis_id,
                source_run_id=source_run_id,
                source_agent_mode=source_agent_mode,
                source_aux=source_aux,
                node=selected,
                score=score[0],
                branch_path=branch_path,
                artifact_dir=artifact_dir,
                destination_dir=task_dir / hypothesis_id,
                code=code,
                source_kind=source_kind,
            ),
        )

    return PromotionPlan(
        root=root,
        task=task,
        journal_path=journal_path,
        created=created,
        existing=existing,
        conflicts=(),
        skipped=(),
    )


def _run_journal_paths(logs_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in logs_dir.glob("*/journal.json")
        if path.is_file()
    )


def plan_branch_hypothesis_promotion_from_logs(
    *,
    root: Path,
    task: str,
    logs_dir: Path,
    top_n: int,
    agent_mode: str | None = None,
    show_progress: bool = False,
) -> PromotionPlan:
    if top_n <= 0:
        raise ValueError("--top-n must be greater than zero")

    root = Path(root)
    journals = _run_journal_paths(logs_dir)
    skipped: list[str] = []
    candidates: list[PromotionCandidate] = []

    if show_progress and journals:
        with Progress() as progress:
            task_id = progress.add_task("Scanning journals", total=len(journals))
            for journal_path in journals:
                candidates.extend(
                    _branch_candidates_from_journal(
                        journal_path=journal_path,
                        agent_mode=agent_mode,
                        skipped=skipped,
                    )
                )
                progress.advance(task_id)
    else:
        for journal_path in journals:
            candidates.extend(
                _branch_candidates_from_journal(
                    journal_path=journal_path,
                    agent_mode=agent_mode,
                    skipped=skipped,
                )
            )

    return _plan_from_candidates(
        root=root,
        task=task,
        journal_path=logs_dir,
        candidates=candidates,
        top_n=top_n,
        skipped=skipped,
    )


def _first_sentence(text: str | None) -> str | None:
    if not text:
        return None
    cleaned = " ".join(text.split())
    match = re.match(r"(.+?[.!?])(?:\s|$)", cleaned)
    return match.group(1) if match else cleaned


def _truncate_text(text: str, *, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _promotion_title(entry: PromotionEntry, source_label: str) -> str:
    if not entry.source_analysis:
        raise ValueError(
            f"Cannot promote {entry.source_run_id}:{entry.source_node_id}: "
            "source node has no analysis."
        )
    return _truncate_text(_first_sentence(entry.source_analysis) or entry.source_analysis)


def _promotion_summary(entry: PromotionEntry) -> str:
    if not entry.source_analysis:
        raise ValueError(
            f"Cannot promote {entry.source_run_id}:{entry.source_node_id}: "
            "source node has no analysis."
        )
    return entry.source_analysis


def _promotion_rationale(entry: PromotionEntry) -> str:
    if not entry.source_plan:
        raise ValueError(
            f"Cannot promote {entry.source_run_id}:{entry.source_node_id}: "
            "source node has no plan."
        )
    return entry.source_plan


def _promotion_implementation_hint(entry: PromotionEntry) -> str:
    if not entry.source_plan:
        raise ValueError(
            f"Cannot promote {entry.source_run_id}:{entry.source_node_id}: "
            "source node has no plan."
        )
    return entry.source_plan


def _promotion_sources(entry: PromotionEntry) -> list[str]:
    sources = [f"aide://{entry.source_run_id}/journal.json#{entry.source_node_id}"]
    if entry.source_artifact_dir:
        sources.append(f"{entry.source_artifact_dir}/aide_result.json")
    return sources


def _hypothesis_payload(entry: PromotionEntry) -> dict[str, Any]:
    mode = entry.source_agent_mode
    is_classic = entry.source_kind == "promoted_classic_node"
    source_label = "classic node" if is_classic else "branch"
    return {
        "enabled": True,
        "agent_modes": [mode],
        "title": _promotion_title(entry, source_label),
        "summary": _promotion_summary(entry),
        "rationale": _promotion_rationale(entry),
        "implementation_hint": _promotion_implementation_hint(entry),
        "expected_effect": _promotion_summary(entry),
        "risk": entry.source_validity_warning or "",
        "sources": _promotion_sources(entry),
        "origin": {
            "kind": entry.source_kind,
            "source_run_id": entry.source_run_id,
            "source_node_id": entry.source_node_id,
            "source_step": entry.source_step,
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
        "source_step": entry.source_step,
        "source_kind": entry.source_kind,
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
    parser.add_argument("journal", type=Path, nargs="?")
    parser.add_argument(
        "--run",
        default=None,
        help="Run id under --logs-dir. Useful with --step or --node-id.",
    )
    parser.add_argument("--task", default="playground-series-s6e6")
    parser.add_argument("--repo-root", type=Path, default=repo_root())
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Directory containing AIDE run logs. Used when no journal is provided.",
    )
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--node-id", default=None)
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
        root = args.repo_root.resolve()
        logs_dir = args.logs_dir or root / "logs"
        journal = args.journal
        if journal is None and args.run:
            journal = logs_dir / args.run / "journal.json"
        if args.step is not None or args.node_id is not None:
            if journal is None:
                raise ValueError("--step/--node-id requires a journal path or --run.")
            plan = plan_node_promotion(
                root=root,
                task=args.task,
                journal_path=journal,
                step=args.step,
                node_id=args.node_id,
                agent_mode=args.agent_mode,
            )
        elif journal is None:
            if args.top_n is None:
                raise ValueError("--top-n is required unless --step or --node-id is set.")
            plan = plan_branch_hypothesis_promotion_from_logs(
                root=root,
                task=args.task,
                logs_dir=logs_dir,
                top_n=args.top_n,
                agent_mode=args.agent_mode,
                show_progress=True,
            )
        else:
            if args.top_n is None:
                raise ValueError("--top-n is required unless --step or --node-id is set.")
            plan = plan_branch_hypothesis_promotion(
                root=root,
                task=args.task,
                journal_path=journal,
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
