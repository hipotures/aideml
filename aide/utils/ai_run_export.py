from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aide.journal import Journal, Node
from aide.utils import serialize
from aide.utils.artifact_manifest import artifact_timestamp_from_ctime
from aide.utils.prediction_similarity import submission_prediction_rmse
from scripts.smart_kaggle_submit import _parse_public_score, _sha256_matches


@dataclass(frozen=True)
class ExportResult:
    export_dir: Path
    meta_path: Path
    nodes_path: Path
    data_paths: tuple[Path, ...] = ()


ProgressCallback = Callable[[str, int, int | None], None]
RAW_DATA_STEMS = ("train", "test", "sample_submission")
RAW_DATA_SUFFIXES = (".csv.gz", ".csv")


def _report_progress(
    progress_callback: ProgressCallback | None,
    stage: str,
    completed: int,
    total: int | None = None,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, completed, total)


def _sha256_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_raw_data_file(data_dir: Path, stem: str) -> Path:
    for suffix in RAW_DATA_SUFFIXES:
        candidate = data_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    expected = " or ".join(f"{stem}{suffix}" for suffix in RAW_DATA_SUFFIXES)
    raise FileNotFoundError(f"Missing raw data file in {data_dir}: expected {expected}")


def _copy_raw_data_files(
    data_dir: Path,
    export_dir: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[Path]:
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {data_dir}")
    _report_progress(progress_callback, "Copying raw data files", 0, len(RAW_DATA_STEMS))
    copied_paths: list[Path] = []
    for index, stem in enumerate(RAW_DATA_STEMS, start=1):
        source_path = _find_raw_data_file(data_dir, stem)
        target_path = export_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied_paths.append(target_path)
        _report_progress(
            progress_callback,
            "Copying raw data files",
            index,
            len(RAW_DATA_STEMS),
        )
    return copied_paths


def _raw_data_records(export_dir: Path, data_paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in data_paths:
        role = path.name.removesuffix(".csv.gz").removesuffix(".csv")
        records.append(
            {
                "role": role,
                "path": path.relative_to(export_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _load_registry(log_dir: Path) -> list[dict[str, Any]]:
    registry_path = log_dir.parent / "submission_registry.json"
    if not registry_path.exists():
        return []

    data = json.loads(registry_path.read_text())
    if isinstance(data, dict):
        entries = data.get("submissions", [])
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Malformed submission registry: {registry_path}")

    if not isinstance(entries, list):
        raise ValueError(f"Malformed submission registry entries: {registry_path}")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"Malformed submission registry entry: {registry_path}")
    return entries


def _metric_value(node: Node) -> float | None:
    if node.metric is None or node.metric.value is None:
        return None
    return float(node.metric.value)


def _metric_maximize(node: Node) -> bool | None:
    return None if node.metric is None else node.metric.maximize


def _public_score_rank(score: float, maximize: bool | None) -> float:
    return score if maximize is not False else -score


def _score_sort_value(record: dict[str, Any]) -> tuple[float, int]:
    score = record.get("local_cv_score")
    if score is None:
        normalized = float("-inf")
    else:
        normalized = _public_score_rank(float(score), record.get("metric_maximize"))
    step = int(record["step"]) if record.get("step") is not None else 10**12
    return normalized, -step


def _node_score_rank(node: Node) -> float:
    score = _metric_value(node)
    if score is None:
        return float("-inf")
    return _public_score_rank(float(score), _metric_maximize(node))


def _canonical_by_best_score(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(records, key=_score_sort_value)


def _group_records(
    node_records: list[dict[str, Any]],
    *,
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in node_records:
        value = record.get(key)
        if value:
            groups.setdefault(str(value), []).append(record)
    return groups


def annotate_exact_duplicates(node_records: list[dict[str, Any]]) -> None:
    for record in node_records:
        record["duplicate"] = {
            "exact_code_group": f"code:{record['code_sha256']}",
            "exact_code_role": "canonical",
            "exact_code_canonical_node_id": record["node_id"],
            "exact_submission_group": None,
            "exact_submission_role": None,
            "exact_submission_canonical_node_id": None,
            "near_submission_canonical_node_id": None,
            "near_submission_rmse": None,
        }

    for code_hash, group in _group_records(node_records, key="code_sha256").items():
        canonical = _canonical_by_best_score(group)
        for record in group:
            record["duplicate"]["exact_code_group"] = f"code:{code_hash}"
            record["duplicate"]["exact_code_role"] = (
                "canonical" if record is canonical else "duplicate"
            )
            record["duplicate"]["exact_code_canonical_node_id"] = canonical["node_id"]

    for submission_hash, group in _group_records(
        node_records,
        key="submission_sha256",
    ).items():
        canonical = _canonical_by_best_score(group)
        for record in group:
            record["duplicate"]["exact_submission_group"] = (
                f"submission:{submission_hash}"
            )
            record["duplicate"]["exact_submission_role"] = (
                "canonical" if record is canonical else "duplicate"
            )
            record["duplicate"]["exact_submission_canonical_node_id"] = canonical[
                "node_id"
            ]


def annotate_near_submission_duplicates(
    node_records: list[dict[str, Any]],
    *,
    threshold: float,
    sample_size: int,
    min_common_sample_size: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    candidates = [
        record for record in node_records if record.get("submission_path") is not None
    ]
    canonicals: list[dict[str, Any]] = []
    sorted_candidates = sorted(candidates, key=_score_sort_value, reverse=True)
    _report_progress(
        progress_callback,
        "Checking near duplicates",
        0,
        len(sorted_candidates),
    )
    for index, record in enumerate(sorted_candidates, start=1):
        if record["duplicate"].get("exact_submission_role") == "duplicate":
            _report_progress(
                progress_callback,
                "Checking near duplicates",
                index,
                len(sorted_candidates),
            )
            continue
        matched = None
        matched_rmse = None
        for canonical in canonicals:
            rmse = submission_prediction_rmse(
                Path(record["submission_path"]),
                Path(canonical["submission_path"]),
                sample_size=sample_size,
                min_common_sample_size=min_common_sample_size,
            )
            if rmse is not None and rmse <= threshold:
                matched = canonical
                matched_rmse = rmse
                break
        if matched is None:
            canonicals.append(record)
            _report_progress(
                progress_callback,
                "Checking near duplicates",
                index,
                len(sorted_candidates),
            )
            continue
        record["duplicate"]["near_submission_canonical_node_id"] = matched["node_id"]
        record["duplicate"]["near_submission_rmse"] = matched_rmse
        _report_progress(
            progress_callback,
            "Checking near duplicates",
            index,
            len(sorted_candidates),
        )


def _created_at(node: Node) -> str:
    return dt.datetime.fromtimestamp(node.ctime).astimezone().isoformat()


def _timestamp_from_node(node: Node) -> str:
    return artifact_timestamp_from_ctime(node.ctime)


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
    return log_dir / "artifacts" / _timestamp_from_node(node)


def _public_score_for_node(
    registry_entries: list[dict[str, Any]],
    run_id: str,
    node: Node,
    submission_sha256: str | None,
) -> float | None:
    timestamp = _timestamp_from_node(node)
    scores = []
    for entry in registry_entries:
        if str(entry.get("remote_status") or "").upper() != "COMPLETE":
            continue

        score = _parse_public_score(entry.get("public_score"))
        if score is None:
            continue

        node_id_matches = entry.get("node_id") == node.id
        run_step_timestamp_matches = (
            entry.get("run") == run_id
            and str(entry.get("step")) == str(node.step)
            and entry.get("timestamp") == timestamp
        )
        sha_matches = (
            submission_sha256 is not None
            and _sha256_matches(entry.get("sha256"), submission_sha256)
        )
        if node_id_matches or run_step_timestamp_matches or sha_matches:
            scores.append(score)

    return max(
        scores,
        key=lambda score: _public_score_rank(score, _metric_maximize(node)),
        default=None,
    )


def _node_record(
    log_dir: Path,
    node: Node,
    registry_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    artifact_dir = _artifact_dir(log_dir, node)
    submission_path = artifact_dir / "submission.csv"
    submission_sha256 = _sha256_file(submission_path) if submission_path.exists() else None
    public_score = _public_score_for_node(
        registry_entries,
        log_dir.name,
        node,
        submission_sha256,
    )
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
        "kaggle_public_score": public_score,
        "metric_maximize": _metric_maximize(node),
        "created_at": _created_at(node),
        "exec_time": node.exec_time,
        "artifact_dir": str(artifact_dir) if artifact_dir.exists() else None,
        "code_sha256": _sha256_text(node.code or ""),
        "submission_path": str(submission_path) if submission_path.exists() else None,
        "submission_sha256": submission_sha256,
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


def _meta_record(
    log_dir: Path,
    journal: Journal,
    node_records: list[dict[str, Any]],
    raw_data_files: list[dict[str, Any]],
) -> dict[str, Any]:
    scored = [node for node in journal.nodes if _metric_value(node) is not None]
    best = max(scored, key=_node_score_rank, default=None)
    public_scored = [
        record
        for record in node_records
        if record.get("kaggle_public_score") is not None
    ]
    best_public = max(
        public_scored,
        key=lambda record: _public_score_rank(
            record["kaggle_public_score"],
            record["metric_maximize"],
        ),
        default=None,
    )
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
        "best_public": None
        if best_public is None
        else {
            "step": best_public["step"],
            "node_id": best_public["node_id"],
            "kaggle_public_score": best_public["kaggle_public_score"],
            "submission_sha256": best_public["submission_sha256"],
        },
        "config": {},
        "raw_data_files": raw_data_files,
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
    data_dir: Path | None = None,
    near_duplicates: bool = True,
    near_submission_rmse_threshold: float = 1e-6,
    prediction_similarity_sample_size: int = 200,
    prediction_similarity_min_common_sample_size: int = 100,
    progress_callback: ProgressCallback | None = None,
) -> ExportResult:
    if not (log_dir / "journal.json").exists():
        raise FileNotFoundError(f"Missing journal.json in {log_dir}")

    _report_progress(progress_callback, "Loading journal", 0)
    journal = serialize.load_json(log_dir / "journal.json", Journal)
    _report_progress(progress_callback, "Loading journal", 1, 1)
    timestamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    export_dir = output_dir / f"{log_dir.name}-{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=False)
    meta_path = export_dir / "run_export.meta.json"
    nodes_path = export_dir / "run_export.nodes.jsonl"
    data_paths: list[Path] = []
    if data_dir is not None:
        data_paths = _copy_raw_data_files(
            Path(data_dir),
            export_dir,
            progress_callback=progress_callback,
        )

    registry_entries = _load_registry(log_dir)
    sorted_nodes = sorted(journal.nodes, key=lambda n: n.step)
    _report_progress(
        progress_callback,
        "Building node records",
        0,
        len(sorted_nodes),
    )
    nodes = []
    for index, node in enumerate(sorted_nodes, start=1):
        nodes.append(_node_record(log_dir, node, registry_entries))
        _report_progress(
            progress_callback,
            "Building node records",
            index,
            len(sorted_nodes),
        )
    _report_progress(progress_callback, "Annotating exact duplicates", 0)
    annotate_exact_duplicates(nodes)
    _report_progress(progress_callback, "Annotating exact duplicates", 1, 1)
    if near_duplicates:
        annotate_near_submission_duplicates(
            nodes,
            threshold=near_submission_rmse_threshold,
            sample_size=prediction_similarity_sample_size,
            min_common_sample_size=prediction_similarity_min_common_sample_size,
            progress_callback=progress_callback,
        )
    _report_progress(progress_callback, "Writing export", 0)
    with nodes_path.open("w", encoding="utf-8") as f:
        for record in nodes:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    meta_path.write_text(
        json.dumps(
            _meta_record(log_dir, journal, nodes, _raw_data_records(export_dir, data_paths)),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _report_progress(progress_callback, "Writing export", 1, 1)
    return ExportResult(
        export_dir=export_dir,
        meta_path=meta_path,
        nodes_path=nodes_path,
        data_paths=tuple(data_paths),
    )
