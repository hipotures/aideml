from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aide.journal import Journal, Node
from aide.utils import serialize
from aide.utils.artifact_manifest import artifact_timestamp_from_ctime


@dataclass(frozen=True)
class ExportResult:
    export_dir: Path
    meta_path: Path
    nodes_path: Path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metric_value(node: Node) -> float | None:
    if node.metric is None or node.metric.value is None:
        return None
    return float(node.metric.value)


def _metric_maximize(node: Node) -> bool | None:
    return None if node.metric is None else bool(node.metric.maximize)


def _created_at(node: Node) -> str:
    return dt.datetime.fromtimestamp(node.ctime).astimezone().isoformat()


def _node_depth(node: Node) -> int:
    depth = 0
    parent = node.parent
    seen = {node.id}
    while parent is not None and parent.id not in seen:
        depth += 1
        seen.add(parent.id)
        parent = parent.parent
    return depth


def _artifact_dir(log_dir: Path, node: Node) -> Path:
    return log_dir / "artifacts" / artifact_timestamp_from_ctime(node.ctime)


def _node_record(log_dir: Path, node: Node) -> dict[str, Any]:
    artifact_dir = _artifact_dir(log_dir, node)
    children = sorted(node.children, key=lambda child: child.step)
    return {
        "step": node.step,
        "node_id": node.id,
        "parent_id": node.parent.id if node.parent is not None else None,
        "children_ids": [child.id for child in children],
        "depth": _node_depth(node),
        "status": node.status,
        "is_buggy": bool(node.is_buggy),
        "is_terminal_failure": bool(node.is_terminal_failure),
        "origin": "source_node",
        "local_cv_score": _metric_value(node),
        "kaggle_public_score": None,
        "metric_maximize": _metric_maximize(node),
        "created_at": _created_at(node),
        "exec_time": node.exec_time,
        "artifact_dir": str(artifact_dir) if artifact_dir.exists() else None,
        "code_sha256": _sha256_text(node.code or ""),
        "submission_sha256": None,
        "duplicate": {},
        "plan": node.plan,
        "analysis": node.analysis,
        "validity_warning": node.validity_warning,
        "error": {
            "exc_type": node.exc_type,
            "summary": node.exc_type,
        },
        "code": node.code,
    }


def _meta_record(log_dir: Path, journal: Journal, export_dir: Path) -> dict[str, Any]:
    scored = [node for node in journal.nodes if _metric_value(node) is not None]
    best = max(scored, key=lambda node: node.metric, default=None)
    return {
        "schema_version": 1,
        "run": log_dir.name,
        "exported_at": dt.datetime.now().astimezone().isoformat(),
        "node_count": len(journal.nodes),
        "scored_node_count": len(scored),
        "best_local": None
        if best is None
        else {
            "step": best.step,
            "node_id": best.id,
            "local_cv_score": _metric_value(best),
        },
        "best_public": None,
        "config": {},
        "notes_for_ai": (
            "This is a complete AIDE tree export. Nodes are ordered by step and "
            "connected by parent_id/children_ids. Duplicate hints are advisory; "
            "no node was pruned."
        ),
    }


def export_run_for_ai(
    log_dir: Path,
    *,
    output_dir: Path = Path("exports"),
    near_duplicates: bool = True,
    near_submission_rmse_threshold: float = 1e-6,
    prediction_similarity_sample_size: int = 200,
    prediction_similarity_min_common_sample_size: int = 100,
) -> ExportResult:
    if not (log_dir / "journal.json").exists():
        raise FileNotFoundError(f"Missing journal.json in {log_dir}")

    journal = serialize.load_json(log_dir / "journal.json", Journal)
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    export_dir = output_dir / f"{log_dir.name}-{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=False)
    meta_path = export_dir / "run_export.meta.json"
    nodes_path = export_dir / "run_export.nodes.jsonl"

    nodes = [_node_record(log_dir, node) for node in sorted(journal.nodes, key=lambda n: n.step)]
    with nodes_path.open("w", encoding="utf-8") as f:
        for record in nodes:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    meta_path.write_text(
        json.dumps(_meta_record(log_dir, journal, export_dir), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return ExportResult(export_dir=export_dir, meta_path=meta_path, nodes_path=nodes_path)
