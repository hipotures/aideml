from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..journal import Journal, Node
from .metric import MetricValue

RESULT_MANIFEST_NAME = "aide_result.json"
RESULT_SCHEMA_VERSION = 1
BASELINE_PLAN_PREFIX = "AutoGluon raw baseline"
SYNTHESIS_PLAN_PREFIX = "External Codex synthesis checkpoint"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_autogluon_config(code: str) -> dict[str, Any] | None:
    match = re.search(r"AIDE_AG_CONFIG\s*=\s*(\{.*?\})\nRESULT_MARKER", code, re.S)
    if match is None:
        match = re.search(r"AIDE_AG_CONFIG\s*=\s*(\{.*?\})(?:\n|$)", code, re.S)
    if match is None:
        return None
    try:
        parsed = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def artifact_timestamp_from_ctime(ctime: float) -> str:
    return dt.datetime.fromtimestamp(ctime).strftime("%Y%m%dT%H%M%S")


def file_entry(path: Path, *, base_dir: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return {
        "path": path.relative_to(base_dir).as_posix(),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": sha256_file(path),
    }


def metric_payload(node: Node) -> dict[str, Any]:
    metric = node.metric
    if metric is None:
        return {"value": None, "maximize": None}
    return {
        "value": metric.value,
        "maximize": metric.maximize,
    }


def node_status(node: Node) -> str:
    if node.status == "failed":
        return "failed"
    if node.is_buggy:
        return "bug"
    return "ok"


def node_origin(node: Node) -> str:
    plan = str(node.plan or "")
    if plan.startswith(BASELINE_PLAN_PREFIX):
        return "baseline"
    if plan.startswith(SYNTHESIS_PLAN_PREFIX):
        return "synthesis"
    if node.status == "failed":
        return "failed"
    return "normal"


def autogluon_payload(code: str) -> dict[str, Any]:
    ag_config = parse_autogluon_config(code) or {}
    return {
        "profile": ag_config.get("profile"),
        "presets": ag_config.get("presets"),
        "included_model_types": ag_config.get("included_model_types"),
        "time_limit": ag_config.get("time_limit"),
        "process_timeout": ag_config.get("process_timeout"),
        "use_gpu": ag_config.get("use_gpu"),
        "resolved_settings": ag_config,
    }


def build_node_artifact_manifest(
    *,
    cfg: Any,
    node: Node,
    artifact_dir: Path,
) -> dict[str, Any]:
    solution_path = artifact_dir / "solution.py"
    submission_path = artifact_dir / "submission.csv"
    error_path = artifact_dir / "error.txt"
    code = solution_path.read_text(encoding="utf-8") if solution_path.exists() else node.code
    metric = metric_payload(node)
    status = node_status(node)
    run = Path(cfg.log_dir).name
    timestamp = artifact_dir.name
    submission = file_entry(submission_path, base_dir=artifact_dir)
    error = file_entry(error_path, base_dir=artifact_dir)
    autogluon = autogluon_payload(code)

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "source_node",
        "run": run,
        "timestamp": timestamp,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "status": status,
        "local_score": metric["value"],
        "metric_maximize": metric["maximize"],
        "is_buggy": bool(node.is_buggy),
        "sha256": submission["sha256"] if submission else None,
        "profile": autogluon.get("profile"),
        "autogluon_presets": autogluon.get("presets"),
        "included_model_types": autogluon.get("included_model_types"),
        "time_limit": autogluon.get("time_limit"),
        "files": {
            "solution": file_entry(solution_path, base_dir=artifact_dir),
            "submission": submission,
            "error": error,
        },
        "node": {
            "id": node.id,
            "step": node.step,
            "ctime": node.ctime,
            "parent_id": node.parent.id if node.parent is not None else None,
            "status": status,
            "origin": node_origin(node),
            "plan": node.plan,
            "analysis": node.analysis,
            "is_buggy": bool(node.is_buggy),
            "metric": metric,
            "submission_validation": node.submission_validation,
        },
        "execution": {
            "exec_time": node.exec_time,
            "exc_type": node.exc_type,
            "exc_info": node.exc_info,
            "exc_stack": node.exc_stack,
        },
        "submission_validation": node.submission_validation,
        "autogluon": autogluon,
        "source": {
            "source_run": None,
            "source_node_id": None,
            "source_step": None,
            "source_timestamp": None,
            "source_sha256": None,
        },
    }


def write_node_artifact_manifest(*, cfg: Any, node: Node, artifact_dir: Path) -> dict[str, Any]:
    manifest = build_node_artifact_manifest(cfg=cfg, node=node, artifact_dir=artifact_dir)
    write_json(artifact_dir / RESULT_MANIFEST_NAME, manifest)
    return manifest


def _node_sort_key(payload: tuple[dict[str, Any], Path]) -> tuple[bool, int, float, str]:
    manifest, _artifact_dir = payload
    node = manifest.get("node") or {}
    step = node.get("step")
    ctime = node.get("ctime")
    return (
        step is None,
        int(step) if step is not None else 0,
        float(ctime) if ctime is not None else 0.0,
        str(node.get("id") or ""),
    )


def _node_from_manifest(manifest: dict[str, Any], artifact_dir: Path) -> Node:
    node_payload = manifest.get("node") or {}
    execution = manifest.get("execution") or {}
    metric_payload_ = node_payload.get("metric") or {}
    solution_path = artifact_dir / "solution.py"
    code = solution_path.read_text(encoding="utf-8") if solution_path.exists() else ""
    node = Node(
        code=code,
        plan=node_payload.get("plan"),
        id=str(node_payload.get("id")),
        ctime=float(node_payload.get("ctime") or 0.0),
    )
    node.step = node_payload.get("step")
    node.status = node_payload.get("status")
    node.analysis = node_payload.get("analysis")
    node.is_buggy = bool(node_payload.get("is_buggy"))
    node.metric = MetricValue(
        metric_payload_.get("value"),
        maximize=metric_payload_.get("maximize", True),
    )
    node.exec_time = execution.get("exec_time")
    node.exc_type = execution.get("exc_type")
    node.exc_info = execution.get("exc_info")
    node.exc_stack = execution.get("exc_stack")
    node.submission_validation = node_payload.get("submission_validation") or manifest.get(
        "submission_validation"
    )
    return node


def reconstruct_journal_from_artifacts(log_dir: Path) -> Journal:
    manifests: list[tuple[dict[str, Any], Path]] = []
    artifacts_dir = log_dir / "artifacts"
    if not artifacts_dir.exists():
        return Journal()
    for manifest_path in sorted(artifacts_dir.glob(f"*/{RESULT_MANIFEST_NAME}")):
        manifest = load_json(manifest_path)
        if manifest.get("kind") != "source_node":
            continue
        node_payload = manifest.get("node") or {}
        if not node_payload.get("id"):
            continue
        manifests.append((manifest, manifest_path.parent))

    node_by_id: dict[str, Node] = {}
    parent_by_id: dict[str, str | None] = {}
    journal = Journal()
    for manifest, artifact_dir in sorted(manifests, key=_node_sort_key):
        node = _node_from_manifest(manifest, artifact_dir)
        node_by_id[node.id] = node
        parent_by_id[node.id] = (manifest.get("node") or {}).get("parent_id")
        journal.nodes.append(node)

    for node in journal.nodes:
        parent_id = parent_by_id.get(node.id)
        if parent_id and parent_id in node_by_id:
            parent = node_by_id[parent_id]
            node.parent = parent
            parent.children.add(node)

    return journal
