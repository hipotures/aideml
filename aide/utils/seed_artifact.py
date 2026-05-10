from __future__ import annotations

import datetime as dt
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..journal import Journal, Node
from .artifact_manifest import (
    RESULT_MANIFEST_NAME,
    SEEDED_BASE_PLAN_PREFIX,
    artifact_timestamp_from_ctime,
    autogluon_payload,
    file_entry,
    load_json,
    metric_payload,
    node_origin,
    node_status,
    parse_autogluon_config,
    write_json,
)
from .metric import MetricValue

SHA_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{6,64}$")
SEEDABLE_MANIFEST_KINDS = {"source_node", "profile_eval"}


@dataclass(frozen=True)
class SeedArtifactSource:
    manifest: dict[str, Any]
    manifest_path: Path
    artifact_dir: Path
    matched_sha256: str
    matched_kind: str

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run") or self.artifact_dir.parents[1].name)

    @property
    def timestamp(self) -> str:
        return str(self.manifest.get("timestamp") or self.artifact_dir.name)

    @property
    def node_payload(self) -> dict[str, Any]:
        payload = self.manifest.get("node")
        return payload if isinstance(payload, dict) else {}

    @property
    def source_step(self) -> int | None:
        step = self.node_payload.get("step")
        return int(step) if isinstance(step, int | str) and str(step).isdigit() else None

    @property
    def local_score(self) -> float | None:
        value = self.manifest.get("local_score")
        if value is None:
            value = (self.node_payload.get("metric") or {}).get("value")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None


def _manifest_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    submission_hash = manifest.get("sha256")
    if isinstance(submission_hash, str) and submission_hash:
        hashes["submission"] = submission_hash.lower()

    files = manifest.get("files") or {}
    if isinstance(files, dict):
        solution = files.get("solution") or {}
        if isinstance(solution, dict):
            solution_hash = solution.get("sha256")
            if isinstance(solution_hash, str) and solution_hash:
                hashes["solution"] = solution_hash.lower()
    return hashes


def _candidate_sort_key(source: SeedArtifactSource) -> tuple[float, str, float]:
    score = source.local_score
    timestamp = source.timestamp
    try:
        mtime = source.manifest_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (
        float("-inf") if score is None else score,
        timestamp,
        mtime,
    )


def find_seed_artifact(
    top_log_dir: Path,
    sha_prefix: str,
    *,
    source_run: str | None = None,
) -> SeedArtifactSource:
    normalized_prefix = sha_prefix.strip().lower()
    if not SHA_PREFIX_RE.match(normalized_prefix):
        raise ValueError(
            "`--seed-from-sha` must be a 6-64 character hexadecimal SHA prefix."
        )

    candidates: list[SeedArtifactSource] = []
    for manifest_path in sorted(top_log_dir.glob(f"*/artifacts/*/{RESULT_MANIFEST_NAME}")):
        manifest = load_json(manifest_path)
        if manifest.get("kind") not in SEEDABLE_MANIFEST_KINDS:
            continue
        artifact_dir = manifest_path.parent
        run_id = str(manifest.get("run") or artifact_dir.parents[1].name)
        if source_run is not None and run_id != source_run:
            continue
        for kind, full_hash in _manifest_hashes(manifest).items():
            if full_hash.startswith(normalized_prefix):
                candidates.append(
                    SeedArtifactSource(
                        manifest=manifest,
                        manifest_path=manifest_path,
                        artifact_dir=artifact_dir,
                        matched_sha256=full_hash,
                        matched_kind=kind,
                    )
                )

    if not candidates:
        scope = f" in run {source_run!r}" if source_run else ""
        raise FileNotFoundError(f"No artifact found for SHA prefix {sha_prefix!r}{scope}.")

    matched_hashes = {candidate.matched_sha256 for candidate in candidates}
    if len(matched_hashes) > 1:
        matches = ", ".join(sorted(matched_hashes))
        raise ValueError(
            f"SHA prefix {sha_prefix!r} is ambiguous across artifact hashes: {matches}"
        )

    return max(candidates, key=_candidate_sort_key)


def source_is_autogluon(source: SeedArtifactSource) -> bool:
    if source.manifest.get("profile") is not None:
        return True
    if isinstance(source.manifest.get("autogluon"), dict):
        autogluon = source.manifest["autogluon"]
        if autogluon.get("profile") is not None or bool(
            autogluon.get("resolved_settings")
        ):
            return True
    solution_path = source.artifact_dir / "solution.py"
    if not solution_path.exists():
        return False
    try:
        return parse_autogluon_config(solution_path.read_text(encoding="utf-8")) is not None
    except OSError:
        return False


def _target_artifact_dir(log_dir: Path, ctime: float) -> tuple[Path, float]:
    next_ctime = ctime
    while True:
        timestamp = artifact_timestamp_from_ctime(next_ctime)
        artifact_dir = log_dir / "artifacts" / timestamp
        if not artifact_dir.exists():
            return artifact_dir, next_ctime
        next_ctime += 1.0


def _seed_plan(source: SeedArtifactSource) -> str:
    parts = [
        f"{SEEDED_BASE_PLAN_PREFIX}:",
        f"source_run={source.run_id}",
        f"source_step={source.source_step if source.source_step is not None else '?'}",
        f"source_timestamp={source.timestamp}",
        f"{source.matched_kind}_sha256={source.matched_sha256}",
    ]
    source_plan = source.node_payload.get("plan")
    if source_plan:
        parts.append(f"Original plan: {source_plan}")
    return " ".join(parts)


def _metric_from_source(source: SeedArtifactSource) -> MetricValue:
    metric = source.node_payload.get("metric") or {}
    return MetricValue(
        metric.get("value", source.manifest.get("local_score")),
        maximize=metric.get("maximize", source.manifest.get("metric_maximize", True)),
    )


def _term_out_from_source(source: SeedArtifactSource) -> list[str]:
    for log_name in ("process_stdout.log", "autogluon_stdout.log", "autogluon.log"):
        log_path = source.artifact_dir / log_name
        if log_path.exists():
            try:
                return [log_path.read_text(encoding="utf-8", errors="replace")]
            except OSError:
                return []
    return []


def _seed_node_from_source(source: SeedArtifactSource, *, ctime: float) -> Node:
    solution_path = source.artifact_dir / "solution.py"
    if not solution_path.exists():
        raise FileNotFoundError(f"Source artifact has no solution.py: {source.artifact_dir}")
    node_payload = source.node_payload
    execution = source.manifest.get("execution") or {}
    node = Node(
        code=solution_path.read_text(encoding="utf-8"),
        plan=_seed_plan(source),
        ctime=ctime,
        _term_out=_term_out_from_source(source),
    )
    node.status = node_payload.get("status") or source.manifest.get("status")
    node.analysis = node_payload.get("analysis")
    node.validity_warning = node_payload.get("validity_warning")
    node.is_buggy = bool(node_payload.get("is_buggy", source.manifest.get("is_buggy", False)))
    node.metric = _metric_from_source(source)
    node.exec_time = execution.get("exec_time")
    node.exc_type = execution.get("exc_type")
    node.exc_info = execution.get("exc_info")
    node.exc_stack = execution.get("exc_stack")
    node.submission_validation = node_payload.get(
        "submission_validation"
    ) or source.manifest.get("submission_validation")
    return node


def _rewrite_manifest(
    *,
    cfg: Any,
    node: Node,
    source: SeedArtifactSource,
    artifact_dir: Path,
) -> None:
    manifest = dict(source.manifest)
    solution_path = artifact_dir / "solution.py"
    submission_path = artifact_dir / "submission.csv"
    error_path = artifact_dir / "error.txt"
    code = solution_path.read_text(encoding="utf-8") if solution_path.exists() else node.code
    metric = metric_payload(node)
    submission = file_entry(submission_path, base_dir=artifact_dir)
    error = file_entry(error_path, base_dir=artifact_dir)
    autogluon = autogluon_payload(code)

    manifest.update(
        {
            "run": Path(cfg.log_dir).name,
            "timestamp": artifact_dir.name,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "artifact_dir": str(artifact_dir),
            "status": node_status(node),
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
                "parent_id": None,
                "status": node_status(node),
                "origin": node_origin(node),
                "plan": node.plan,
                "analysis": node.analysis,
                "validity_warning": node.validity_warning,
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
                "source_run": source.run_id,
                "source_node_id": source.node_payload.get("id"),
                "source_step": source.source_step,
                "source_timestamp": source.timestamp,
                "source_sha256": source.matched_sha256,
                "source_match_kind": source.matched_kind,
            },
        }
    )
    write_json(artifact_dir / RESULT_MANIFEST_NAME, manifest)


def seed_journal_from_artifact(
    cfg: Any,
    source: SeedArtifactSource,
    *,
    ctime: float | None = None,
) -> tuple[Journal, Node, Path]:
    log_dir = Path(cfg.log_dir)
    artifact_dir, node_ctime = _target_artifact_dir(log_dir, time.time() if ctime is None else ctime)
    shutil.copytree(source.artifact_dir, artifact_dir)

    node = _seed_node_from_source(source, ctime=node_ctime)
    journal = Journal()
    journal.append(node)
    _rewrite_manifest(cfg=cfg, node=node, source=source, artifact_dir=artifact_dir)

    submission_path = artifact_dir / "submission.csv"
    if submission_path.exists():
        working_dir = Path(cfg.workspace_dir) / "working"
        working_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(submission_path, working_dir / "submission.csv")

    return journal, node, artifact_dir
