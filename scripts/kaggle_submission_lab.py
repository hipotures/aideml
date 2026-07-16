from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from dotenv import dotenv_values
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    track,
)
from rich.table import Table
from rich.text import Text

from scripts import smart_kaggle_submit as smart
from aide.journal import Journal
from aide.utils import serialize
from aide.utils.artifact_manifest import (
    RESULT_MANIFEST_NAME,
    artifact_timestamp_from_ctime,
    write_node_artifact_manifest,
)
from aide.utils.path_portability import (
    resolve_portable_path,
    sanitize_persisted_payload,
    to_portable_path,
)


DEFAULT_COMPETITION = smart.DEFAULT_COMPETITION
DEFAULT_LOGS_DIR = smart.DEFAULT_LOGS_DIR
DEFAULT_INDEX_PATH = Path("logs/submission_index.json")
DEFAULT_REGISTRY = smart.DEFAULT_REGISTRY
DEFAULT_TABLE_LIMIT = 20
INDEX_VERSION = 4
SOURCE_RERUN_SHA_STYLE = "bold black on bright_yellow"
ARTIFACT_VISIBILITY_OVERRIDES_NAME = "artifact_visibility_overrides.json"
PROFILE_CALIBRATION_RERUN_ROLE = "profile_calibration_rerun"
CANONICAL_FAST_MODEL_FAMILY = ["XGB", "GBM", "CAT"]
FAST_AUTOGLOON_PRESETS = {"medium_quality"}


def default_competition() -> str:
    env_value = os.getenv("AIDE_PROJECT_NAME", "").strip()
    if env_value:
        return env_value
    dotenv_value = dotenv_values(Path(".env")).get("AIDE_PROJECT_NAME")
    if dotenv_value:
        return str(dotenv_value).strip() or DEFAULT_COMPETITION
    return DEFAULT_COMPETITION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def stat_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def run_scan_signature(run_dir: Path) -> dict[str, Any]:
    artifact_files: dict[str, dict[str, Any]] = {}
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        for artifact_path in sorted(artifacts_dir.glob("*")):
            if not artifact_path.is_dir():
                continue
            for name in ("solution.py", "submission.csv", RESULT_MANIFEST_NAME):
                path = artifact_path / name
                if path.exists():
                    artifact_files[f"{artifact_path.name}/{name}"] = stat_signature(path)
    return {
        "artifacts": artifact_files,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_persisted_payload(payload), indent=2, sort_keys=True) + "\n"
    )


def _artifact_visibility_overrides_path(logs_dir: Path) -> Path:
    return Path(logs_dir) / ARTIFACT_VISIBILITY_OVERRIDES_NAME


def _load_artifact_visibility_overrides(logs_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(_artifact_visibility_overrides_path(logs_dir))
    if not isinstance(payload, dict):
        return {}
    overrides = payload.get("artifacts", {})
    if not isinstance(overrides, dict):
        return {}
    return {
        str(artifact_dir): dict(metadata)
        for artifact_dir, metadata in overrides.items()
        if isinstance(metadata, dict)
    }


def _model_family(record: dict[str, Any]) -> list[str]:
    models = record.get("included_model_types")
    if not isinstance(models, list):
        return []
    return [
        str(model).strip().upper() for model in models if str(model).strip()
    ]


def _profile_calibration_family_invalid_reason(record: dict[str, Any]) -> str | None:
    family = _model_family(record)
    if family == CANONICAL_FAST_MODEL_FAMILY:
        return None
    if "CAT" not in family:
        return "catboost_omitted_from_required_model_family"
    if set(family) == set(CANONICAL_FAST_MODEL_FAMILY):
        return "required_model_family_incomplete"
    return "required_model_family_changed"


def _profile_calibration_metadata(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("kind") != "profile_eval":
        return {}

    metadata: dict[str, Any] = {
        "artifact_role": PROFILE_CALIBRATION_RERUN_ROLE,
        "hide_from_submission_lab_default": True,
        "not_a_submission_candidate": True,
    }
    if record.get("profile_calibration_session_id"):
        family_invalid_reason = _profile_calibration_family_invalid_reason(record)
        preset = str(record.get("autogluon_presets") or "").strip().lower()
        try:
            time_limit = int(record.get("time_limit"))
        except (TypeError, ValueError):
            time_limit = 0
        invalid_reason = record.get("invalid_reason")
        if family_invalid_reason is not None:
            invalid_reason = family_invalid_reason
        elif preset not in FAST_AUTOGLOON_PRESETS or not 0 < time_limit <= 600:
            invalid_reason = "full_best_or_long_profile"
        elif not record.get("all_required_model_types_trained"):
            invalid_reason = "required_model_family_not_fully_trained"
        valid = bool(record.get("valid_for_final_profile_selection")) and not invalid_reason
        return {
            **metadata,
            "historical_only": False,
            "profile_calibration_session_id": record.get(
                "profile_calibration_session_id"
            ),
            "valid_for_final_profile_selection": valid,
            "valid_for_current_final_selection": valid,
            "invalid_reason": invalid_reason,
        }

    family_invalid_reason = _profile_calibration_family_invalid_reason(record)
    if family_invalid_reason is not None:
        return {
            **metadata,
            "historical_only": True,
            "valid_for_final_profile_selection": False,
            "valid_for_current_final_selection": False,
            "invalid_reason": family_invalid_reason,
        }

    preset = str(record.get("autogluon_presets") or "").strip().lower()
    try:
        time_limit = int(record.get("time_limit"))
    except (TypeError, ValueError):
        time_limit = 0
    if preset not in FAST_AUTOGLOON_PRESETS or not 0 < time_limit <= 600:
        return {
            **metadata,
            "historical_only": True,
            "valid_for_final_profile_selection": False,
            "valid_for_current_final_selection": False,
            "invalid_reason": "full_best_or_long_profile",
        }
    return {
        **metadata,
        "historical_only": True,
        "valid_for_final_profile_selection": False,
        "valid_for_current_final_selection": False,
        "invalid_reason": "historical_profile_calibration_rerun",
    }


def _merge_profile_calibration_metadata(
    records: Iterable[dict[str, Any]],
    *,
    overrides: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for record in records:
        artifact_dir = str(record.get("artifact_dir") or "")
        merged_record = {**record, **overrides.get(artifact_dir, {})}
        merged.append(
            {
                **merged_record,
                **_profile_calibration_metadata(merged_record),
            }
        )
    return merged


def _record_path(value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        return Path("/__aideml_missing_path__")
    return resolve_portable_path(text)


def backfill_legacy_source_manifests(run_dir: Path) -> int:
    journal_path = run_dir / "journal.json"
    artifacts_dir = run_dir / "artifacts"
    if not journal_path.exists() or not artifacts_dir.exists():
        return 0

    try:
        journal = serialize.load_json(journal_path, Journal)
    except FileNotFoundError:
        return 0
    cfg = SimpleNamespace(log_dir=run_dir)
    written = 0
    for node in journal.nodes:
        artifact_dir = artifacts_dir / artifact_timestamp_from_ctime(node.ctime)
        manifest_path = artifact_dir / RESULT_MANIFEST_NAME
        if manifest_path.exists():
            continue
        if not (artifact_dir / "solution.py").exists():
            continue
        write_node_artifact_manifest(cfg=cfg, node=node, artifact_dir=artifact_dir)
        written += 1
    return written


def journal_hypothesis_lookup(run_dir: Path) -> dict[tuple[str, Any], str]:
    journal_path = run_dir / "journal.json"
    if not journal_path.exists():
        return {}
    try:
        journal = serialize.load_json(journal_path, Journal)
    except Exception:
        return {}

    lookup: dict[tuple[str, Any], str] = {}
    for node in journal.nodes:
        offered = getattr(node, "research_hypotheses_offered", []) or []
        if len(offered) != 1 or not isinstance(offered[0], str):
            continue
        hypothesis_id = offered[0]
        lookup[("node_id", getattr(node, "id", None))] = hypothesis_id
        lookup[("step", getattr(node, "step", None))] = hypothesis_id
    return lookup


def parse_autogluon_config(code: str) -> dict[str, Any] | None:
    match = re.search(r"AIDE_AG_CONFIG\s*=\s*(\{.*?\})\nRESULT_MARKER", code, re.S)
    if match is None:
        match = re.search(r"AIDE_AG_CONFIG\s*=\s*(\{.*?\})(?:\n|$)", code, re.S)
    if match is None:
        return None
    import ast

    try:
        parsed = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _record_sort_key(record: dict[str, Any]) -> tuple[float, str]:
    score = record.get("local_score")
    metric_maximize = record.get("metric_maximize")
    if score is None:
        normalized = float("-inf")
    elif metric_maximize is False:
        normalized = -float(score)
    else:
        normalized = float(score)
    return normalized, str(record.get("timestamp") or "")


def _autogluon_payload_has_settings(payload: dict[str, Any]) -> bool:
    return (
        payload.get("profile") is not None
        or payload.get("presets") is not None
        or payload.get("included_model_types") is not None
        or payload.get("time_limit") is not None
        or bool(payload.get("resolved_settings"))
    )


def _autogluon_eval_metric(payload: dict[str, Any]) -> Any:
    resolved_settings = payload.get("resolved_settings")
    if not isinstance(resolved_settings, dict):
        resolved_settings = {}
    return payload.get("eval_metric") or resolved_settings.get("eval_metric")


def _record_looks_autogluon(record: dict[str, Any]) -> bool:
    explicit = record.get("algo")
    if isinstance(explicit, str):
        normalized = explicit.strip().lower()
        if normalized in {"ag", "autogluon"}:
            return True
        if normalized in {"leg", "legacy"}:
            return False
    return (
        record.get("kind") == "profile_eval"
        or record.get("profile") is not None
        or record.get("autogluon_presets") is not None
        or record.get("included_model_types") is not None
        or record.get("time_limit") is not None
    )


def _record_is_seed_copy(record: dict[str, Any]) -> bool:
    return (
        record.get("kind") != "profile_eval"
        and record.get("origin") not in {"manual_blend", "auto_blend"}
        and bool(record.get("source_sha256"))
    )


def _format_algo(record: dict[str, Any], *, unknown_if_missing: bool = False) -> str:
    explicit = record.get("algo")
    if isinstance(explicit, str):
        normalized = explicit.strip().lower()
        if normalized in {"ag", "autogluon"}:
            return "AG"
        if normalized in {"leg", "legacy"}:
            return "Leg"
    if _record_looks_autogluon(record):
        return "AG"
    if unknown_if_missing and "kind" not in record:
        return "?"
    return "Leg"


def deduplicate_records_by_sha256(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_sha: dict[str, dict[str, Any]] = {}
    for record in records:
        sha = record.get("sha256")
        if not sha:
            continue
        current = best_by_sha.get(str(sha))
        if current is not None and _record_is_seed_copy(record) and not _record_is_seed_copy(current):
            continue
        if (
            current is not None
            and record.get("not_a_submission_candidate")
            and not current.get("not_a_submission_candidate")
        ):
            continue
        if (
            current is None
            or (
                not _record_is_seed_copy(record)
                and _record_is_seed_copy(current)
            )
            or (
                not record.get("not_a_submission_candidate")
                and current.get("not_a_submission_candidate")
            )
            or _record_sort_key(record) > _record_sort_key(current)
        ):
            best_by_sha[str(sha)] = record
    return sorted(best_by_sha.values(), key=_record_sort_key, reverse=True)


def _artifact_record_base(
    *,
    competition: str,
    run: str,
    timestamp: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    submission_path = artifact_dir / "submission.csv"
    solution_path = artifact_dir / "solution.py"
    code = solution_path.read_text() if solution_path.exists() else ""
    ag_config = parse_autogluon_config(code) or {}
    included_model_types = ag_config.get("included_model_types")
    profile = ag_config.get("profile")
    if profile is None:
        models = ",".join(included_model_types or [])
        if models == "XGB,GBM":
            profile = "fast_boost"
        elif models == "XGB,GBM,CAT":
            profile = "full_boost"

    return {
        "competition": competition,
        "run": run,
        "timestamp": timestamp,
        "artifact_dir": to_portable_path(artifact_dir),
        "solution_path": to_portable_path(solution_path),
        "submission_path": to_portable_path(submission_path),
        "sha256": sha256_file(submission_path) if submission_path.exists() else None,
        "profile": profile,
        "autogluon_presets": ag_config.get("presets"),
        "included_model_types": included_model_types,
        "time_limit": ag_config.get("time_limit"),
    }


def build_manifest_records(
    *,
    logs_dir: Path,
    run: str,
    competition: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    run_dir = logs_dir / run
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        return records
    hypothesis_lookup = journal_hypothesis_lookup(run_dir)
    for manifest_path in sorted(artifacts_dir.glob(f"*/{RESULT_MANIFEST_NAME}")):
        artifact_dir = manifest_path.parent
        timestamp = artifact_dir.name
        manifest = _load_json(manifest_path)
        node = manifest.get("node") or {}
        source = manifest.get("source") or {}
        autogluon = manifest.get("autogluon") or {}
        base = _artifact_record_base(
            competition=manifest.get("competition", competition),
            run=str(manifest.get("run") or run),
            timestamp=str(manifest.get("timestamp") or timestamp),
            artifact_dir=artifact_dir,
        )
        metric = node.get("metric") or {}
        status = str(manifest.get("status") or node.get("status") or "unknown")
        run_stats = manifest.get("run_stats") or {}
        hypothesis_id = (
            node.get("hypothesis_id")
            or manifest.get("hypothesis_id")
            or hypothesis_lookup.get(("node_id", node.get("id")))
            or hypothesis_lookup.get(("step", node.get("step")))
        )
        records.append(
            {
                **base,
                "kind": manifest.get("kind") or "source_node",
                "hypothesis_id": hypothesis_id,
                "step": node.get("step"),
                "node_id": node.get("id"),
                "parent_node_id": node.get("parent_id"),
                "origin": node.get("origin") or manifest.get("origin"),
                "local_score": manifest.get("local_score", metric.get("value")),
                "metric_maximize": manifest.get(
                    "metric_maximize",
                    metric.get("maximize", True),
                ),
                "eval_metric": manifest.get("eval_metric")
                or metric.get("name")
                or _autogluon_eval_metric(autogluon),
                "is_buggy": bool(manifest.get("is_buggy") or status != "ok"),
                "exec_time": (manifest.get("execution") or {}).get("exec_time"),
                "status": status,
                "algo": (
                    "AG"
                    if _autogluon_payload_has_settings(autogluon)
                    or _record_looks_autogluon(base)
                    or manifest.get("kind") == "profile_eval"
                    else "Leg"
                ),
                "profile": manifest.get("profile")
                or autogluon.get("profile")
                or base.get("profile"),
                "autogluon_presets": manifest.get("autogluon_presets")
                or autogluon.get("presets")
                or base.get("autogluon_presets"),
                "included_model_types": manifest.get("included_model_types")
                or autogluon.get("included_model_types")
                or base.get("included_model_types"),
                "time_limit": manifest.get("time_limit")
                or autogluon.get("time_limit")
                or base.get("time_limit"),
                "source_run": source.get("source_run"),
                "source_node_id": source.get("source_node_id"),
                "source_step": source.get("source_step"),
                "source_timestamp": source.get("source_timestamp"),
                "source_sha256": source.get("source_sha256"),
                "source_solution_path": source.get("source_solution_path")
                or manifest.get("source_solution_path"),
                "source_solution_sha256": source.get("source_solution_sha256")
                or manifest.get("source_solution_sha256"),
                "artifact_role": manifest.get("artifact_role"),
                "hide_from_submission_lab_default": manifest.get(
                    "hide_from_submission_lab_default"
                ),
                "not_a_submission_candidate": manifest.get(
                    "not_a_submission_candidate"
                ),
                "historical_only": manifest.get("historical_only"),
                "profile_calibration_session_id": manifest.get(
                    "profile_calibration_session_id"
                ),
                "valid_for_final_profile_selection": manifest.get(
                    "valid_for_final_profile_selection"
                ),
                "valid_for_current_final_selection": manifest.get(
                    "valid_for_current_final_selection"
                ),
                "invalid_reason": manifest.get("invalid_reason"),
                "configured_model_types": manifest.get("configured_model_types"),
                "trained_model_types": manifest.get("trained_model_types"),
                "failed_or_skipped_model_types": manifest.get(
                    "failed_or_skipped_model_types"
                ),
                "all_required_model_types_trained": manifest.get(
                    "all_required_model_types_trained"
                ),
                "submission_only": bool(run_stats.get("submission_only")),
                "blend_kind": run_stats.get("blend_kind"),
                "blend_mode": run_stats.get("blend_mode"),
                "blend_weighting": run_stats.get("blend_weighting"),
                "blend_recipe_hash": run_stats.get("blend_recipe_hash"),
                "blend_component_count": run_stats.get("blend_component_count"),
                "blend_component_sha256": run_stats.get("component_sha256")
                or run_stats.get("blend_component_sha256"),
            }
        )
    return records


def build_run_records(
    *,
    logs_dir: Path,
    run_dir: Path,
    competition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run = run_dir.name
    records = build_manifest_records(
        logs_dir=logs_dir,
        run=run,
        competition=competition,
    )
    artifact_signatures = {}
    for record in records:
        submission_path = _record_path(record["submission_path"])
        if submission_path.exists():
            artifact_signatures[record["timestamp"]] = {
                "submission_sha256": record["sha256"],
                "submission_signature": file_signature(submission_path),
            }
    return records, {
        "scan_signature": run_scan_signature(run_dir),
        "artifact_signatures": artifact_signatures,
    }


def _run_is_unchanged(
    *,
    run: str,
    run_dir: Path,
    cached_runs: dict[str, Any],
) -> bool:
    cached = cached_runs.get(run)
    if not isinstance(cached, dict):
        return False
    return cached.get("scan_signature") == run_scan_signature(run_dir)


def refresh_index(
    *,
    logs_dir: Path = DEFAULT_LOGS_DIR,
    index_path: Path = DEFAULT_INDEX_PATH,
    competition: str = DEFAULT_COMPETITION,
    runs: list[str] | None = None,
    reindex: bool = False,
    progress: Any | None = None,
) -> dict[str, Any]:
    logs_dir = Path(logs_dir)
    index = _load_json(index_path) or {"records": [], "runs": {}}
    needs_index_upgrade = index.get("version") != INDEX_VERSION
    existing_records = list(index.get("records", []))
    cached_runs = dict(index.get("runs", {}))
    records_by_run: dict[str, list[dict[str, Any]]] = {}
    for record in existing_records:
        records_by_run.setdefault(record.get("run"), []).append(record)

    run_signatures: dict[str, Any] = dict(cached_runs)
    all_run_dirs = [
        run_dir
        for run_dir in sorted(logs_dir.iterdir())
        if run_dir.is_dir()
        and (run_dir / "artifacts").exists()
    ]
    if runs:
        run_lookup = {run_dir.name: run_dir for run_dir in all_run_dirs}
        missing_runs = [run for run in runs if run not in run_lookup]
        if missing_runs:
            raise ValueError(
                "No AIDE log run matches --run: " + ", ".join(missing_runs)
            )
        run_dirs = [run_lookup[run] for run in runs]
    else:
        run_dirs = all_run_dirs
    task_id = None
    if progress is not None:
        task_id = progress.add_task("Indexing AIDE runs", total=len(run_dirs))
    for run_dir in run_dirs:
        run = run_dir.name
        try:
            backfilled = backfill_legacy_source_manifests(run_dir)
            needs_hypothesis_backfill = any(
                "hypothesis_id" not in record
                for record in records_by_run.get(run, [])
            )
            if not reindex and _run_is_unchanged(
                run=run,
                run_dir=run_dir,
                cached_runs=cached_runs,
            ) and backfilled == 0 and not needs_hypothesis_backfill and not needs_index_upgrade:
                continue
            records, run_meta = build_run_records(
                logs_dir=logs_dir,
                run_dir=run_dir,
                competition=competition,
            )
            records_by_run[run] = records
            run_signatures[run] = run_meta
        finally:
            if progress is not None and task_id is not None:
                progress.advance(task_id)

    all_records = [
        record
        for run in sorted(records_by_run)
        for record in records_by_run.get(run, [])
    ]
    all_records = _merge_profile_calibration_metadata(
        all_records,
        overrides=_load_artifact_visibility_overrides(logs_dir),
    )
    refreshed = {
        "version": INDEX_VERSION,
        "competition": competition,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": all_records,
        "runs": run_signatures,
    }
    _write_json(index_path, refreshed)
    return refreshed


def _record_is_submit_ready(record: dict[str, Any]) -> bool:
    score_or_submission_only = record.get("local_score") is not None or (
        bool(record.get("submission_only"))
        and record.get("origin") in {"manual_blend", "auto_blend"}
    )
    return (
        record.get("status") == "ok"
        and not _record_is_seed_copy(record)
        and not record.get("not_a_submission_candidate")
        and not record.get("is_buggy")
        and score_or_submission_only
        and record.get("sha256") is not None
        and _record_path(record.get("submission_path", "")).exists()
    )


def _is_profile_calibration_rerun(record: dict[str, Any]) -> bool:
    return record.get("artifact_role") == PROFILE_CALIBRATION_RERUN_ROLE


def _has_submitted_known_public_score(
    record: dict[str, Any],
    *,
    registry: smart.SubmissionRegistry | None = None,
    competition: str | None = None,
) -> bool:
    if (
        str(record.get("remote_status") or "").upper() == "COMPLETE"
        and smart._parse_public_score(record.get("public_score")) is not None
    ):
        return True
    if (
        str(record.get("submit") or "").lower() == "submitted"
        and smart._parse_public_score(record.get("public_score")) is not None
    ):
        return True
    sha = record.get("sha256")
    if registry is None or not sha:
        return False
    for entry in registry.entries:
        if competition is not None and entry.get("competition") != competition:
            continue
        if (
            smart._sha256_matches(entry.get("sha256"), sha)
            and str(entry.get("remote_status") or "").upper() == "COMPLETE"
            and smart._parse_public_score(entry.get("public_score")) is not None
        ):
            return True
    return False


def filter_submission_lab_visibility(
    records: Iterable[dict[str, Any]],
    *,
    registry: smart.SubmissionRegistry | None = None,
    competition: str | None = None,
    include_profile_reruns: bool = False,
    profile_reruns_only: bool = False,
) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for record in records:
        is_profile_rerun = _is_profile_calibration_rerun(record)
        if profile_reruns_only:
            if is_profile_rerun:
                visible.append(record)
            continue
        if include_profile_reruns:
            visible.append(record)
            continue
        if (
            is_profile_rerun
            and record.get("hide_from_submission_lab_default")
            and not _has_submitted_known_public_score(
                record,
                registry=registry,
                competition=competition,
            )
        ):
            continue
        visible.append(record)
    return visible


def parse_sha256_filters(values: list[str] | None) -> list[str]:
    return smart.parse_sha256_filters(values)


def parse_run_filters(values: list[str] | None) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        for part in str(value).split(","):
            run = part.strip()
            if not run:
                continue
            if "/" in run or "\\" in run:
                raise ValueError(f"Invalid run id: {run!r}")
            if run not in seen:
                selected.append(run)
                seen.add(run)
    return selected


def filter_records_by_run(
    records: list[dict[str, Any]],
    run_filters: list[str],
) -> list[dict[str, Any]]:
    if not run_filters:
        return records
    selected_runs = set(run_filters)
    return [
        record
        for record in records
        if str(record.get("run") or "") in selected_runs
    ]


def filter_records_by_sha256(
    records: list[dict[str, Any]],
    sha256_filters: list[str],
) -> list[dict[str, Any]]:
    if not sha256_filters:
        return records
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    for sha_filter in sha256_filters:
        matches = [
            record
            for record in records
            if str(record.get("sha256") or "").lower().startswith(sha_filter)
        ]
        if not matches:
            raise ValueError(f"No submission index record matches sha256 prefix: {sha_filter}")
        matched_hashes = {record.get("sha256") for record in matches}
        if len(matched_hashes) > 1:
            preview = ", ".join(sorted(str(value)[:10] for value in matched_hashes))
            raise ValueError(f"Ambiguous sha256 prefix {sha_filter}; matches: {preview}")
        record = deduplicate_records_by_sha256(matches)[0]
        if record.get("sha256") not in selected_hashes:
            selected.append(record)
            selected_hashes.add(str(record.get("sha256")))
    return selected


def select_top_records(
    records: Iterable[dict[str, Any]],
    *,
    registry: smart.SubmissionRegistry,
    competition: str,
    limit: int,
) -> list[dict[str, Any]]:
    ready = [
        record
        for record in records
        if _record_is_submit_ready(record)
        and not registry.is_submitted(
            competition=competition,
            sha256=record.get("sha256"),
            run=str(record.get("run")),
            step=int(record.get("step") or -1),
            timestamp=str(record.get("timestamp")),
        )
    ]
    return deduplicate_records_by_sha256(ready)[:limit]


def _format_score(value: Any) -> str:
    return "-" if value is None else f"{float(value):.5f}"


def _format_step(value: Any) -> str:
    if value is None:
        return "-"
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return str(value)
    return "-" if parsed < 0 else str(parsed)


def _format_public_score(value: Any) -> str:
    parsed = smart._parse_public_score(value)
    return "" if parsed is None else f"{parsed:.5f}"


def _timestamp_date(value: Any) -> str:
    text = str(value or "")
    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if iso_match is not None:
        return "".join(iso_match.groups())
    return text[:8] if len(text) >= 8 else text


def _record_timestamp_ctime(value: Any) -> float:
    text = str(value or "")
    match = re.match(r"^\d{8}T\d{6}", text)
    if match is None:
        raise ValueError(f"Invalid artifact timestamp: {text!r}")
    return dt.datetime.strptime(match.group(0), "%Y%m%dT%H%M%S").timestamp()


def _display_date(value: Any) -> str:
    text = smart._date_to_string(value) or ""
    if re.match(r"\d{4}-\d{2}-\d{2}", text):
        return text[:10].replace("-", "")
    return text[:8] if len(text) >= 8 else text


def _short_profile(value: Any) -> str:
    text = str(value or "")
    return text.replace("_boost", "")


def _short_run(value: Any) -> str:
    return str(value or "")


def _short_models(models: list[Any] | None) -> str:
    labels = {"XGB": "x", "GBM": "g", "CAT": "c"}
    return "".join(labels.get(str(model), str(model)[:1].lower()) for model in models or [])


def _format_duration(value: Any) -> str:
    if value is None or value == "":
        return "-"
    text = str(value).strip()
    if text.endswith("m"):
        try:
            return f"{float(text[:-1]):.1f}m"
        except (TypeError, ValueError):
            return "-"
    if text.endswith("s"):
        try:
            seconds = float(text[:-1])
        except (TypeError, ValueError):
            return "-"
        return f"{seconds / 60.0:.1f}m"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{seconds / 60.0:.1f}m"


def adaptive_table_limit(
    *,
    default: int = DEFAULT_TABLE_LIMIT,
    terminal_rows: int | None = None,
) -> int:
    if terminal_rows is None:
        terminal_rows = shutil.get_terminal_size(fallback=(80, default)).lines
    if terminal_rows <= 0:
        return default
    return max(default, int(terminal_rows * 0.65))


def render_table(
    console: Console,
    records: list[dict[str, Any]],
    *,
    full_view: bool = False,
) -> None:
    columns, rows = candidate_display_table(records, full_view=full_view)
    show_source = "src_sha" in columns
    show_hypothesis = "hyp" in columns
    table = Table(title="Top unsent submit-ready candidates", padding=(0, 1))
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("cv", justify="right", no_wrap=True)
    table.add_column("time", justify="right", no_wrap=True)
    table.add_column("metric", no_wrap=True)
    table.add_column("k", no_wrap=True)
    if full_view:
        table.add_column("prof", no_wrap=True)
    table.add_column("m", no_wrap=True)
    table.add_column("run", no_wrap=True, overflow="ellipsis", max_width=42)
    if show_hypothesis:
        table.add_column("hyp", no_wrap=True)
    table.add_column("Algo", no_wrap=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("date", no_wrap=True)
    table.add_column("sha", no_wrap=True)
    if show_source:
        table.add_column("src_sha", no_wrap=True)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def candidate_display_table(
    records: list[dict[str, Any]],
    *,
    full_view: bool = False,
) -> tuple[list[str], list[list[str]]]:
    show_source = any(record.get("kind") == "profile_eval" for record in records)
    show_hypothesis = any(record.get("hypothesis_id") for record in records)
    columns = ["#", "cv", "time", "metric", "k"]
    if full_view:
        columns.append("prof")
    columns.extend(["m", "run"])
    if show_hypothesis:
        columns.append("hyp")
    columns.extend(["Algo", "step", "date", "sha"])
    if show_source:
        columns.append("src_sha")

    rows: list[list[str]] = []
    for rank, record in enumerate(records, start=1):
        models = _short_models(record.get("included_model_types"))
        source_sha = ""
        if record.get("kind") == "profile_eval":
            source_sha = str(record.get("source_sha256") or "")[:10]
        row = [
            str(rank),
            _format_score(record.get("local_score")),
            _format_duration(record.get("exec_time")),
            str(record.get("eval_metric") or "-"),
            "e" if record.get("kind") == "profile_eval" else "n",
        ]
        if full_view:
            row.append(_short_profile(record.get("profile")))
        row.extend(
            [
                models,
                _short_run(record.get("run")),
            ]
        )
        if show_hypothesis:
            row.append(str(record.get("hypothesis_id") or "-"))
        row.extend(
            [
                _format_algo(record),
                _format_step(record.get("step")),
                _timestamp_date(record.get("timestamp")),
                str(record.get("sha256") or "")[:10],
            ]
        )
        if show_source:
            row.append(source_sha)
        rows.append(row)
    return columns, rows


def render_text_table(
    console: Console,
    title: str,
    columns: list[str],
    rows: list[list[str]],
) -> None:
    console.print(title)
    console.print("\t".join(columns))
    for row in rows:
        console.print("\t".join(row))
    console.print()


def _remote_identity(remote: Any) -> tuple[Any, str, str]:
    description = smart._remote_attr(remote, "description")
    parsed = smart.parse_submission_description(description)
    return (
        smart._remote_ref(remote),
        parsed.get("timestamp") or "",
        str(smart._remote_attr(remote, "file_name") or ""),
    )


def _registry_display_sort_key(row: dict[str, Any]) -> tuple[bool, float, str]:
    public_score = (
        None if _row_is_local_invalid(row) else smart._parse_public_score(row.get("public_score"))
    )
    return (
        public_score is not None,
        public_score if public_score is not None else float("-inf"),
        str(row.get("date") or ""),
    )


def _row_is_local_invalid(row: dict[str, Any]) -> bool:
    remote_status = str(row.get("remote_status") or "").upper()
    return (
        str(row.get("manual_status") or "").lower() == "failed"
        or remote_status == "FAILED_LOCAL_INVALID"
        or bool(row.get("manual_invalid_reason"))
    )


def _row_display_status(row: dict[str, Any]) -> str:
    if _row_is_local_invalid(row):
        return "FAILED_LOCAL_INVALID"
    return str(row.get("remote_status") or "")


def _remote_display_rows(
    registry: smart.SubmissionRegistry,
    remote_submissions: list[Any] | None,
    record_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
    competition: str | None = None,
) -> list[dict[str, Any]]:
    if remote_submissions is None:
        return []

    registry_entries = [
        entry
        for entry in registry.entries
        if competition is None or entry.get("competition") == competition
    ]
    known_refs = {smart._entry_ref(entry) for entry in registry_entries}
    known_timestamps = {str(entry.get("timestamp") or "") for entry in registry_entries}
    known_hashes = [
        str(entry.get("sha256") or "")
        for entry in registry_entries
        if entry.get("sha256")
    ]
    known_files = {
        str(entry.get("remote_filename") or "")
        for entry in registry_entries
        if entry.get("remote_filename")
    }
    rows = []
    for remote in remote_submissions:
        description = smart._remote_attr(remote, "description")
        parsed = smart.parse_submission_description(description)
        remote_ref, remote_timestamp, remote_file = _remote_identity(remote)
        remote_sha = parsed.get("sha")
        if remote_ref is not None and remote_ref in known_refs:
            continue
        if remote_timestamp and remote_timestamp in known_timestamps:
            continue
        if remote_sha and any(
            smart._sha256_matches(known_sha, remote_sha) for known_sha in known_hashes
        ):
            continue
        if remote_file and remote_file in known_files:
            continue

        row = {
            "local_score": smart._parse_public_score(parsed.get("cv")),
            "public_score": smart._remote_attr(remote, "public_score"),
            "remote_status": smart._status_to_string(
                smart._remote_attr(remote, "status")
            ),
            "eval_metric": parsed.get("metric") or parsed.get("eval_metric"),
            "exec_time": parsed.get("exec_time") or parsed.get("time"),
            "algo": parsed.get("algo"),
            "run": parsed.get("run")
            or str(smart._remote_attr(remote, "file_name") or "-"),
            "step": parsed.get("step"),
            "date": parsed.get("timestamp") or smart._remote_attr(remote, "date"),
            "sha256": remote_sha,
        }
        if smart._description_marks_local_invalid(parsed):
            row.update(
                {
                    "local_score": None,
                    "public_score": "",
                    "remote_status": "FAILED_LOCAL_INVALID",
                    "manual_status": "failed",
                    "manual_invalid_reason": (
                        parsed.get("reason")
                        or parsed.get("invalid_reason")
                        or "marked ignored in Kaggle description"
                    ),
                }
            )
        if record_lookup is not None:
            row["algo"] = row.get("algo") or _registry_entry_algo(row, record_lookup)
            row["eval_metric"] = row.get("eval_metric") or _registry_entry_eval_metric(
                row,
                record_lookup,
            )
            row["artifact_dir"] = _registry_entry_artifact_dir(row, record_lookup)
            row["hypothesis_id"] = _registry_entry_hypothesis_id(row, record_lookup)
            row.update(_registry_entry_visibility_metadata(row, record_lookup))
        rows.append(row)
    return rows


def _put_lookup_record(
    lookup: dict[tuple[str, str], dict[str, Any]],
    key: tuple[str, str],
    data: dict[str, Any],
) -> None:
    existing = lookup.get(key)
    if existing is None:
        lookup[key] = data
    elif existing != data:
        lookup[key] = {}


def _registry_record_lookup(
    records: list[dict[str, Any]] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records or []:
        data = {
            "algo": _format_algo(record),
            "artifact_dir": record.get("artifact_dir"),
            "submission_path": record.get("submission_path"),
            "hypothesis_id": record.get("hypothesis_id"),
            "eval_metric": record.get("eval_metric"),
            "exec_time": record.get("exec_time"),
            "source_sha256": record.get("source_sha256"),
            "source_solution_path": record.get("source_solution_path"),
            "source_solution_sha256": record.get("source_solution_sha256"),
            "artifact_role": record.get("artifact_role"),
            "hide_from_submission_lab_default": record.get(
                "hide_from_submission_lab_default"
            ),
            "not_a_submission_candidate": record.get("not_a_submission_candidate"),
            "valid_for_final_profile_selection": record.get(
                "valid_for_final_profile_selection"
            ),
            "invalid_reason": record.get("invalid_reason"),
        }
        sha = str(record.get("sha256") or "")
        if sha:
            _put_lookup_record(lookup, ("sha", sha), data)
            if len(sha) >= 10:
                _put_lookup_record(lookup, ("sha", sha[:10]), data)
        run = str(record.get("run") or "")
        step = str(record.get("step") if record.get("step") is not None else "")
        timestamp = str(record.get("timestamp") or "")
        if run and step and timestamp:
            _put_lookup_record(lookup, ("node", f"{run}|{step}|{timestamp}"), data)
    return lookup


def _registry_lookup_record(
    entry: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    sha = str(entry.get("sha256") or "")
    if sha and ("sha", sha) in lookup:
        record = lookup[("sha", sha)]
        if record:
            return record

    run = str(entry.get("run") or "")
    step = str(entry.get("step") if entry.get("step") is not None else "")
    timestamp = str(entry.get("timestamp") or "")
    if run and step and timestamp:
        return lookup.get(("node", f"{run}|{step}|{timestamp}"))
    return None


def _registry_entry_visibility_metadata(
    entry: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    record = _registry_lookup_record(entry, lookup) or {}
    return {
        key: record.get(key, entry.get(key))
        for key in (
            "artifact_role",
            "hide_from_submission_lab_default",
            "not_a_submission_candidate",
            "valid_for_final_profile_selection",
            "invalid_reason",
        )
    }


def _registry_entry_algo(
    entry: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> str | None:
    explicit = entry.get("algo")
    if explicit:
        return _format_algo({"algo": explicit})

    record = _registry_lookup_record(entry, lookup)
    if record is not None:
        return str(record.get("algo") or "") or None

    artifact_dir = _registry_entry_artifact_dir(entry, lookup)
    if artifact_dir != "-":
        solution_path = _record_path(artifact_dir) / "solution.py"
        if solution_path.exists():
            try:
                code = solution_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                code = ""
            if parse_autogluon_config(code) is not None:
                return "AG"
            return "Leg"

    return None


def _registry_entry_artifact_dir(
    entry: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> str:
    record = _registry_lookup_record(entry, lookup) or {}
    for key in ("artifact_dir", "submission_path", "upload_path"):
        value = record.get(key) if key in record else entry.get(key)
        if not value:
            continue
        path = _record_path(value)
        if key in {"submission_path", "upload_path"}:
            path = path.parent
        return to_portable_path(path)
    return "-"


def _registry_entry_hypothesis_id(
    entry: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> str | None:
    explicit = entry.get("hypothesis_id")
    if explicit:
        return str(explicit)
    record = _registry_lookup_record(entry, lookup)
    if record is not None and record.get("hypothesis_id"):
        return str(record.get("hypothesis_id"))
    return None


def _registry_entry_eval_metric(
    entry: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> str | None:
    explicit = entry.get("eval_metric") or entry.get("metric")
    if explicit:
        return str(explicit)
    record = _registry_lookup_record(entry, lookup)
    if record is not None and record.get("eval_metric"):
        return str(record.get("eval_metric"))
    return None


def _registry_entry_exec_time(
    entry: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> Any:
    explicit = entry.get("exec_time")
    if explicit is not None:
        return explicit
    record = _registry_lookup_record(entry, lookup)
    if record is not None:
        return record.get("exec_time")
    return None


def _registry_entry_source_sha256(
    entry: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> str | None:
    explicit = entry.get("source_sha256")
    if explicit:
        return str(explicit)
    record = _registry_lookup_record(entry, lookup)
    if record is not None and record.get("source_sha256"):
        return str(record.get("source_sha256"))
    return None


def _registry_entry_source_solution_path(
    entry: dict[str, Any],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> str | None:
    explicit = entry.get("source_solution_path")
    if explicit:
        return str(explicit)
    record = _registry_lookup_record(entry, lookup)
    if record is not None and record.get("source_solution_path"):
        return str(record.get("source_solution_path"))
    return None


def _mark_rows_with_source_reruns(rows: list[dict[str, Any]]) -> None:
    source_shas = [
        str(row.get("source_sha256") or "")
        for row in rows
        if row.get("source_sha256")
    ]
    for row in rows:
        sha = str(row.get("sha256") or "")
        row["has_source_rerun"] = any(
            smart._sha256_matches(sha, source_sha) for source_sha in source_shas
        )


def _format_registry_sha(entry: dict[str, Any]) -> str | Text:
    value = str(entry.get("sha256") or "")[:10] or "-"
    if entry.get("has_source_rerun"):
        return Text(value, style=SOURCE_RERUN_SHA_STYLE)
    return value


def render_registry_table(
    console: Console,
    registry: smart.SubmissionRegistry,
    remote_submissions: list[Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    full_view: bool = False,
    limit: int | None = 20,
    run_filters: list[str] | None = None,
    competition: str | None = None,
    include_profile_reruns: bool = False,
    profile_reruns_only: bool = False,
) -> None:
    sorted_rows = registry_display_rows(
        registry,
        remote_submissions,
        records=records,
        limit=limit,
        run_filters=run_filters,
        competition=competition,
        include_profile_reruns=include_profile_reruns,
        profile_reruns_only=profile_reruns_only,
    )
    table = Table(title="Submission registry", padding=(0, 1))
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("cv", justify="right", no_wrap=True)
    table.add_column("public", justify="right", no_wrap=True)
    table.add_column("time", justify="right", no_wrap=True)
    table.add_column("metric", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("run", no_wrap=True, overflow="ellipsis", max_width=42)
    show_hypothesis = any(entry.get("hypothesis_id") for entry in sorted_rows)
    if show_hypothesis:
        table.add_column("hyp", no_wrap=True)
    table.add_column("Algo", no_wrap=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("date", no_wrap=True)
    table.add_column("sha", no_wrap=True)
    show_source = any(entry.get("source_sha256") for entry in sorted_rows)
    if show_source:
        table.add_column("src_sha", no_wrap=True)
    if full_view:
        table.add_column("artifact", no_wrap=True, overflow="fold")

    complete_rank = 0
    for entry in sorted_rows:
        remote_status = str(entry.get("remote_status") or "")
        display_status = _row_display_status(entry)
        if remote_status.upper() == "COMPLETE" and not _row_is_local_invalid(entry):
            complete_rank += 1
            display_rank = str(complete_rank)
        else:
            display_rank = "-"
        row = [
            display_rank,
            _format_score(entry.get("local_score")),
            _format_public_score(entry.get("public_score")),
            _format_duration(entry.get("exec_time")),
            str(entry.get("eval_metric") or "-"),
            display_status or "-",
            str(entry.get("run") or "-"),
        ]
        if show_hypothesis:
            row.append(str(entry.get("hypothesis_id") or "-"))
        row.extend(
            [
                _format_algo(entry, unknown_if_missing=True),
                _format_step(entry.get("step")),
                _display_date(entry.get("date")),
                _format_registry_sha(entry),
            ]
        )
        if show_source:
            row.append(str(entry.get("source_sha256") or "")[:10] or "-")
        if full_view:
            row.append(str(entry.get("artifact_dir") or "-"))
        table.add_row(*row)
    console.print(table)


def registry_display_rows(
    registry: smart.SubmissionRegistry,
    remote_submissions: list[Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    limit: int | None = 20,
    run_filters: list[str] | None = None,
    competition: str | None = None,
    include_profile_reruns: bool = False,
    profile_reruns_only: bool = False,
) -> list[dict[str, Any]]:
    record_lookup = _registry_record_lookup(records)
    rows = [
        {
            "local_score": entry.get("local_score"),
            "exec_time": _registry_entry_exec_time(entry, record_lookup),
            "public_score": entry.get("public_score"),
            "remote_status": entry.get("remote_status"),
            "manual_status": entry.get("manual_status"),
            "manual_invalid_reason": entry.get("manual_invalid_reason"),
            "eval_metric": _registry_entry_eval_metric(entry, record_lookup),
            "algo": _registry_entry_algo(entry, record_lookup),
            "hypothesis_id": _registry_entry_hypothesis_id(entry, record_lookup),
            "artifact_dir": _registry_entry_artifact_dir(entry, record_lookup),
            "run": entry.get("run"),
            "step": entry.get("step"),
            "date": entry.get("timestamp"),
            "sha256": entry.get("sha256"),
            "source_sha256": _registry_entry_source_sha256(entry, record_lookup),
            "source_solution_path": _registry_entry_source_solution_path(
                entry,
                record_lookup,
            ),
            **_registry_entry_visibility_metadata(entry, record_lookup),
        }
        for entry in registry.entries
        if competition is None or entry.get("competition") == competition
    ]
    rows.extend(
        _remote_display_rows(
            registry,
            remote_submissions,
            record_lookup,
            competition=competition,
        )
    )
    if run_filters:
        selected_runs = set(run_filters)
        rows = [
            row
            for row in rows
            if str(row.get("run") or "") in selected_runs
        ]
    rows = filter_submission_lab_visibility(
        rows,
        registry=registry,
        competition=competition,
        include_profile_reruns=include_profile_reruns,
        profile_reruns_only=profile_reruns_only,
    )
    _mark_rows_with_source_reruns(rows)

    sorted_rows = sorted(rows, key=_registry_display_sort_key, reverse=True)
    if limit is not None and limit > 0:
        sorted_rows = sorted_rows[:limit]
    return sorted_rows


def registry_display_table(
    registry: smart.SubmissionRegistry,
    remote_submissions: list[Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    full_view: bool = False,
    limit: int | None = 20,
    run_filters: list[str] | None = None,
    competition: str | None = None,
    include_profile_reruns: bool = False,
    profile_reruns_only: bool = False,
) -> tuple[list[str], list[list[str]]]:
    display_rows = registry_display_rows(
        registry,
        remote_submissions,
        records=records,
        limit=limit,
        run_filters=run_filters,
        competition=competition,
        include_profile_reruns=include_profile_reruns,
        profile_reruns_only=profile_reruns_only,
    )
    columns = [
        "#",
        "cv",
        "public",
        "time",
        "metric",
        "status",
        "run",
        "hyp",
        "Algo",
        "step",
        "date",
        "sha",
    ]
    show_source = any(row.get("source_sha256") for row in display_rows)
    show_hypothesis = any(row.get("hypothesis_id") for row in display_rows)
    if not show_hypothesis:
        columns.remove("hyp")
    if show_source:
        columns.append("src_sha")
    if full_view:
        columns.append("artifact")

    rows = []
    complete_rank = 0
    for entry in display_rows:
        remote_status = str(entry.get("remote_status") or "")
        display_status = _row_display_status(entry)
        if remote_status.upper() == "COMPLETE" and not _row_is_local_invalid(entry):
            complete_rank += 1
            display_rank = str(complete_rank)
        else:
            display_rank = "-"
        row = [
            display_rank,
            _format_score(entry.get("local_score")),
            _format_public_score(entry.get("public_score")),
            _format_duration(entry.get("exec_time")),
            str(entry.get("eval_metric") or "-"),
            display_status or "-",
            str(entry.get("run") or "-"),
        ]
        if show_hypothesis:
            row.append(str(entry.get("hypothesis_id") or "-"))
        row.extend(
            [
                _format_algo(entry, unknown_if_missing=True),
                _format_step(entry.get("step")),
                _display_date(entry.get("date")),
                str(entry.get("sha256") or "")[:10] or "-",
            ]
        )
        if show_source:
            row.append(str(entry.get("source_sha256") or "")[:10] or "-")
        if full_view:
            row.append(str(entry.get("artifact_dir") or "-"))
        rows.append(row)
    return columns, rows


def _merge_tree_artifact(
    artifacts: dict[str, dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    sha = str(payload.get("sha256") or "")
    if not sha:
        return
    existing = artifacts.setdefault("sha:" + sha, {"sha256": sha})
    for key, value in payload.items():
        if value is None or value == "":
            continue
        existing[key] = value


def _tree_row_is_profile_eval(row: dict[str, Any]) -> bool:
    if row.get("kind") == "profile_eval":
        return True
    try:
        step = int(row.get("step"))
    except (TypeError, ValueError):
        step = None
    return bool(row.get("source_sha256")) and step is not None and step < 0


def _tree_parent_sha(row: dict[str, Any]) -> str:
    if not _tree_row_is_profile_eval(row):
        return ""
    return str(row.get("source_sha256") or "")


def _tree_artifact_from_record(
    record: dict[str, Any],
    *,
    selected: bool = False,
) -> dict[str, Any]:
    return {
        "sha256": record.get("sha256"),
        "source_sha256": record.get("source_sha256"),
        "source_solution_path": record.get("source_solution_path"),
        "source_solution_sha256": record.get("source_solution_sha256"),
        "kind": record.get("kind") or "source_node",
        "profile": record.get("profile"),
        "local_score": record.get("local_score"),
        "exec_time": record.get("exec_time"),
        "public_score": record.get("public_score"),
        "remote_status": record.get("remote_status"),
        "manual_status": record.get("manual_status"),
        "manual_invalid_reason": record.get("manual_invalid_reason"),
        "status": record.get("status"),
        "eval_metric": record.get("eval_metric"),
        "algo": record.get("algo"),
        "run": record.get("run"),
        "step": record.get("step"),
        "timestamp": record.get("timestamp") or record.get("date"),
        "artifact_dir": record.get("artifact_dir"),
        "selected": selected,
    }


def _tree_artifact_from_registry_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "sha256": row.get("sha256"),
        "source_sha256": row.get("source_sha256"),
        "source_solution_path": row.get("source_solution_path"),
        "source_solution_sha256": row.get("source_solution_sha256"),
        "kind": "profile_eval" if _tree_row_is_profile_eval(row) else "source_node",
        "local_score": row.get("local_score"),
        "exec_time": row.get("exec_time"),
        "public_score": row.get("public_score"),
        "remote_status": row.get("remote_status"),
        "manual_status": row.get("manual_status"),
        "manual_invalid_reason": row.get("manual_invalid_reason"),
        "eval_metric": row.get("eval_metric"),
        "algo": row.get("algo"),
        "run": row.get("run"),
        "step": row.get("step"),
        "timestamp": row.get("date"),
        "artifact_dir": row.get("artifact_dir"),
    }


def _tree_public_score(row: dict[str, Any]) -> float | None:
    if _row_is_local_invalid(row):
        return None
    return smart._parse_public_score(row.get("public_score"))


def _tree_cv_score(row: dict[str, Any]) -> float | None:
    if _row_is_local_invalid(row):
        return None
    value = row.get("local_score")
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rank_tree_artifacts(
    artifacts: list[dict[str, Any]],
    score_fn,
) -> dict[str, int]:
    scored = [
        (score, str(row.get("sha256") or ""))
        for row in artifacts
        if (score := score_fn(row)) is not None and row.get("sha256")
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return {sha: rank for rank, (_score, sha) in enumerate(scored, start=1)}


def _tree_state(row: dict[str, Any]) -> str:
    if _row_is_local_invalid(row):
        return "invalid"
    remote_status = str(row.get("remote_status") or "")
    if remote_status.upper() == "COMPLETE":
        return "submitted"
    if remote_status:
        return remote_status.lower()
    if row.get("selected") or str(row.get("status") or "").lower() == "ok":
        return "ready"
    return str(row.get("status") or "-")


def _tree_origin_label(row: dict[str, Any]) -> str:
    if _tree_row_is_profile_eval(row):
        return "rerun"
    if row.get("parent_node_id"):
        return "branch"
    return "source"


def _tree_algo_label(row: dict[str, Any]) -> str:
    algo = _format_algo(row, unknown_if_missing=True)
    if algo == "AG":
        return "ag"
    if algo == "Leg":
        return "legacy"
    return str(algo or "unknown").lower()


def _tree_profile_label(row: dict[str, Any]) -> str:
    text = str(row.get("profile") or "").strip()
    if not text:
        return ""
    return re.sub(r"-+", "-", text.replace("_", "-")).strip("-").lower()


def _tree_kind_profile(row: dict[str, Any]) -> str:
    origin = _tree_origin_label(row)
    algo = _tree_algo_label(row)
    profile = _tree_profile_label(row)
    suffix = f"-{profile}" if profile else ""
    return f"{origin}/{algo}{suffix}"


def _tree_family_sort_key(
    family_rows: list[dict[str, Any]],
    *,
    sort_by: str,
) -> tuple[float, float, str]:
    public_scores = [
        score for row in family_rows if (score := _tree_public_score(row)) is not None
    ]
    cv_scores = [
        score for row in family_rows if (score := _tree_cv_score(row)) is not None
    ]
    best_public = max(public_scores) if public_scores else float("-inf")
    best_cv = max(cv_scores) if cv_scores else float("-inf")
    root_sha = str(family_rows[0].get("sha256") or "")
    if sort_by == "cv":
        return (best_cv, best_public, root_sha)
    return (best_public, best_cv, root_sha)


def _tree_child_sort_key(row: dict[str, Any], *, sort_by: str) -> tuple[float, float, str]:
    public_score = _tree_public_score(row)
    cv_score = _tree_cv_score(row)
    public_value = public_score if public_score is not None else float("-inf")
    cv_value = cv_score if cv_score is not None else float("-inf")
    sha = str(row.get("sha256") or "")
    if sort_by == "cv":
        return (cv_value, public_value, sha)
    return (public_value, cv_value, sha)


def candidate_tree_display_table(
    *,
    selected: list[dict[str, Any]],
    registry: smart.SubmissionRegistry,
    remote_submissions: list[Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    sort_by: str = "cv",
    limit: int | None = 20,
    run_filters: list[str] | None = None,
    competition: str | None = None,
    show_seeds: bool = False,
    include_profile_reruns: bool = False,
    profile_reruns_only: bool = False,
) -> tuple[list[str], list[list[str]]]:
    if sort_by not in {"public", "cv"}:
        raise ValueError("sort_by must be 'public' or 'cv'")

    artifacts: dict[str, dict[str, Any]] = {}
    visible_root_shas: set[str] = set()
    tree_records = [
        record
        for record in filter_submission_lab_visibility(
            records or [],
            registry=registry,
            competition=competition,
            include_profile_reruns=include_profile_reruns,
            profile_reruns_only=profile_reruns_only,
        )
        if show_seeds or not _record_is_seed_copy(record)
    ]
    for record in tree_records:
        _merge_tree_artifact(artifacts, _tree_artifact_from_record(record))
        if (
            (include_profile_reruns or profile_reruns_only)
            and _is_profile_calibration_rerun(record)
        ):
            root_sha = _tree_parent_sha(record) or str(record.get("sha256") or "")
            if root_sha:
                visible_root_shas.add(root_sha)

    for record in selected:
        if not show_seeds and _record_is_seed_copy(record):
            continue
        artifact = _tree_artifact_from_record(record, selected=True)
        _merge_tree_artifact(artifacts, artifact)
        root_sha = _tree_parent_sha(artifact) or str(record.get("sha256") or "")
        if root_sha:
            visible_root_shas.add(root_sha)

    registry_rows = registry_display_rows(
        registry,
        remote_submissions,
        records=tree_records,
        limit=None,
        run_filters=run_filters,
        competition=competition,
        include_profile_reruns=include_profile_reruns,
        profile_reruns_only=profile_reruns_only,
    )
    for row in registry_rows:
        artifact = _tree_artifact_from_registry_row(row)
        _merge_tree_artifact(artifacts, artifact)
        root_sha = _tree_parent_sha(artifact) or str(row.get("sha256") or "")
        if root_sha:
            visible_root_shas.add(root_sha)

    all_artifacts = list(artifacts.values())
    for source_sha in visible_root_shas:
        if not any(str(row.get("sha256") or "") == source_sha for row in all_artifacts):
            synthesized = {"sha256": source_sha, "kind": "source_node"}
            _merge_tree_artifact(artifacts, synthesized)
            all_artifacts = list(artifacts.values())

    by_sha = {str(row.get("sha256") or ""): row for row in all_artifacts}
    families: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for source_sha in visible_root_shas:
        root = by_sha.get(source_sha, {"sha256": source_sha, "kind": "source_node"})
        children = [
            row
            for row in all_artifacts
            if _tree_parent_sha(row) == source_sha
            and str(row.get("sha256") or "") != source_sha
        ]
        children.sort(key=lambda row: _tree_child_sort_key(row, sort_by=sort_by), reverse=True)
        families.append((root, children))

    families.sort(
        key=lambda family: _tree_family_sort_key(
            [family[0], *family[1]],
            sort_by=sort_by,
        ),
        reverse=True,
    )
    if limit is not None and limit > 0:
        families = families[:limit]

    displayed_artifacts = [row for root, children in families for row in [root, *children]]
    cv_ranks = _rank_tree_artifacts(displayed_artifacts, _tree_cv_score)
    public_ranks = _rank_tree_artifacts(displayed_artifacts, _tree_public_score)

    columns = [
        "#",
        "CV#",
        "PUB#",
        "cv",
        "public",
        "submit",
        "kind/profile",
        "time",
        "metric",
        "run",
        "step",
        "date",
        "sha",
    ]
    rows: list[list[str]] = []
    for family_rank, (root, children) in enumerate(families, start=1):
        for child_index, row in enumerate([root, *children]):
            sha = str(row.get("sha256") or "")
            marker = str(family_rank)
            if child_index > 0:
                marker = "└─" if child_index == len(children) else "├─"
            rows.append(
                [
                    marker,
                    str(cv_ranks.get(sha) or "-"),
                    str(public_ranks.get(sha) or "-"),
                    _format_score(row.get("local_score")),
                    _format_public_score(row.get("public_score")),
                    _tree_state(row),
                    _tree_kind_profile(row),
                    _format_duration(row.get("exec_time")),
                    str(row.get("eval_metric") or "-"),
                    str(row.get("run") or "-"),
                    _format_step(row.get("step")),
                    _timestamp_date(row.get("timestamp")),
                    sha[:10] or "-",
                ]
            )
    return columns, rows


def ready_submit_commands(rows: list[list[str]], *, max_count: int = 5) -> list[str]:
    commands: list[str] = []
    for row in rows:
        if len(row) < 13 or row[5] != "ready":
            continue
        sha = str(row[12] or "").strip()
        if not sha or sha == "-":
            continue
        commands.append(
            f"uv run python scripts/kaggle_submission_lab.py --sha {sha}"
        )
        if len(commands) >= max_count:
            break
    return commands


def _tree_row_is_root(row: list[str]) -> bool:
    return bool(row) and row[0].strip().isdigit()


def rerun_profile_commands(
    rows: list[list[str]],
    *,
    records: list[dict[str, Any]] | None = None,
    max_count: int = 5,
    profile: str = "best_boost_gpu_1h",
) -> list[str]:
    record_by_sha_prefix: dict[str, dict[str, Any]] = {}
    if records is not None:
        records_by_sha: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            sha = str(record.get("sha256") or "")
            if not sha:
                continue
            records_by_sha.setdefault(sha, []).append(record)
        for sha, sha_records in records_by_sha.items():
            record_by_sha_prefix[sha[:10]] = deduplicate_records_by_sha256(sha_records)[0]

    commands: list[str] = []
    index = 0
    while index < len(rows):
        root = rows[index]
        if not _tree_row_is_root(root):
            index += 1
            continue
        children: list[list[str]] = []
        index += 1
        while index < len(rows) and not _tree_row_is_root(rows[index]):
            children.append(rows[index])
            index += 1

        if len(root) < 13:
            continue
        is_source = root[6] == "source"
        has_public = bool(str(root[4] or "").strip())
        has_rerun_child = any(len(child) > 6 and child[6] != "source" for child in children)
        if not is_source or not has_public or has_rerun_child:
            continue
        sha = str(root[12] or "").strip()
        if not sha or sha == "-":
            continue
        record = record_by_sha_prefix.get(sha)
        if record is None or _format_algo(record) != "AG":
            continue
        commands.append(
            "uv run python scripts/rerun_autogluon_profile.py "
            f"--sha {sha} --profile {profile} --execute"
        )
        if len(commands) >= max_count:
            break
    return commands


def render_candidate_tree_table(
    console: Console,
    *,
    selected: list[dict[str, Any]],
    registry: smart.SubmissionRegistry,
    remote_submissions: list[Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    sort_by: str = "public",
    limit: int | None = 20,
    run_filters: list[str] | None = None,
    competition: str | None = None,
    show_seeds: bool = False,
    include_profile_reruns: bool = False,
    profile_reruns_only: bool = False,
) -> None:
    columns, rows = candidate_tree_display_table(
        selected=selected,
        registry=registry,
        remote_submissions=remote_submissions,
        records=records,
        sort_by=sort_by,
        limit=limit,
        run_filters=run_filters,
        competition=competition,
        show_seeds=show_seeds,
        include_profile_reruns=include_profile_reruns,
        profile_reruns_only=profile_reruns_only,
    )
    table = Table(
        title=f"Submission table (sorted by {sort_by})",
        box=box.SIMPLE,
        show_edge=False,
        padding=(0, 1),
        expand=False,
    )
    for column in columns:
        justify = (
            "right"
            if column in {"CV#", "PUB#", "cv", "public", "time", "step"}
            else "left"
        )
        if column == "#":
            table.add_column(column, justify=justify, no_wrap=True, min_width=3)
        elif column == "run":
            table.add_column(column, justify=justify, no_wrap=True, overflow="ellipsis", max_width=42)
        elif column == "kind/profile":
            table.add_column(
                column,
                justify=justify,
                no_wrap=True,
                overflow="ellipsis",
                max_width=22,
            )
        else:
            table.add_column(column, justify=justify, no_wrap=True)
    current_family_style: str | None = None
    for row in rows:
        is_root_row = row[0].strip().isdigit()
        if is_root_row:
            family_rank = int(row[0].strip())
            current_family_style = "on grey30" if family_rank % 2 == 1 else None
        styled_row: list[str | Text] = list(row)
        if is_root_row:
            styled_row[0] = Text(row[0], style="bold")
        if row[1] == "1":
            styled_row[3] = Text(row[3], style="bold black on bright_green")
        if row[2] == "1":
            styled_row[4] = Text(row[4], style="bold black on bright_cyan")
        if row[5] == "submitted":
            styled_row[5] = Text(row[5], style="green")
        elif row[5] == "ready":
            styled_row[5] = Text(row[5], style="bold bright_yellow")
        table.add_row(
            *styled_row,
            style=current_family_style,
        )
    console.print(table)
    commands = ready_submit_commands(rows)
    if commands:
        console.print()
        console.print("Ready submit commands:")
        for command in commands:
            console.print(command)
    rerun_commands = rerun_profile_commands(rows, records=records)
    if rerun_commands:
        console.print()
        console.print("Ready rerun commands:")
        for command in rerun_commands:
            console.print(command)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_output_payload(
    *,
    selected: list[dict[str, Any]],
    registry: smart.SubmissionRegistry,
    remote_submissions: list[Any] | None,
    records: list[dict[str, Any]],
    full_view: bool,
    registry_limit: int | None,
    run_filters: list[str] | None,
    competition: str | None = None,
    include_profile_reruns: bool = False,
    profile_reruns_only: bool = False,
) -> dict[str, Any]:
    visible_records = filter_submission_lab_visibility(
        records,
        registry=registry,
        competition=competition,
        include_profile_reruns=include_profile_reruns,
        profile_reruns_only=profile_reruns_only,
    )
    return {
        "selected": _json_safe(selected),
        "profile_reruns": _json_safe(
            [
                record
                for record in visible_records
                if _is_profile_calibration_rerun(record)
            ]
        )
        if include_profile_reruns or profile_reruns_only
        else [],
        "registry": _json_safe(
            registry_display_rows(
                registry,
                remote_submissions,
                records=records,
                limit=registry_limit,
                run_filters=run_filters,
                competition=competition,
                include_profile_reruns=include_profile_reruns,
                profile_reruns_only=profile_reruns_only,
            )
        ),
        "remote_visible": None if remote_submissions is None else len(remote_submissions),
        "full_view": full_view,
        "run_filters": run_filters or [],
    }


def sync_registry_from_kaggle(
    *,
    console: Console,
    registry: smart.SubmissionRegistry,
    competition: str,
) -> tuple[Any | None, list[Any] | None]:
    try:
        client = smart._build_kaggle_client()
        remote_submissions = smart.fetch_remote_submissions(client, competition)
        synced = smart.sync_registry_from_remote(
            registry=registry,
            competition=competition,
            remote_submissions=remote_submissions,
        )
        if synced:
            console.print(f"Synchronized {synced} Kaggle submission(s).")
        return client, remote_submissions
    except Exception as exc:
        console.print(f"Remote Kaggle submissions unavailable: {exc}")
        return None, None


def _record_to_candidate(record: dict[str, Any]) -> smart.Candidate:
    step = int(record.get("step") if record.get("step") is not None else -1)
    return smart.Candidate(
        competition=str(record.get("competition") or DEFAULT_COMPETITION),
        run=str(record.get("run")),
        step=step,
        node_id=str(record.get("node_id") or f"profile-eval-{record.get('timestamp')}"),
        parent_node_id=None,
        ancestor_node_ids=(),
        timestamp=str(record.get("timestamp")),
        ctime=_record_timestamp_ctime(record.get("timestamp")),
        local_score=record.get("local_score"),
        metric_maximize=record.get("metric_maximize", True),
        is_buggy=bool(record.get("is_buggy")),
        submission_path=_record_path(record.get("submission_path")),
        sha256=record.get("sha256"),
        validation_error=None,
        algo=_format_algo(record),
        eval_metric=record.get("eval_metric"),
        hypothesis_id=record.get("hypothesis_id"),
        source_sha256=record.get("source_sha256"),
        exec_time=record.get("exec_time"),
        origin=record.get("origin"),
        submission_only=bool(record.get("submission_only")),
        blend_kind=record.get("blend_kind"),
        blend_mode=record.get("blend_mode"),
        blend_weighting=record.get("blend_weighting"),
        blend_recipe_hash=record.get("blend_recipe_hash"),
        blend_component_count=record.get("blend_component_count"),
        blend_component_sha256=record.get("blend_component_sha256"),
    )


def submit_records(
    records: Iterable[dict[str, Any]],
    *,
    registry: smart.SubmissionRegistry,
    client: Any,
    competition: str,
) -> list[dict[str, Any]]:
    return smart.submit_candidates(
        [_record_to_candidate(record) for record in records],
        registry=registry,
        client=client,
        competition=competition,
    )


def _build_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submission index viewer and Kaggle submitter."
    )
    parser.add_argument("--competition", default=default_competition())
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum candidate families shown and selected. "
            "Default adapts to terminal height."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=["rich", "json", "txt"],
        default="rich",
        help="Output format for the candidate and registry views.",
    )
    parser.add_argument(
        "--registry-limit",
        type=int,
        default=None,
        help=(
            "Maximum rows shown in Submission registry. "
            "Default adapts to terminal height; use 0 for no limit."
        ),
    )
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument(
        "--sha256",
        "--sha",
        dest="sha256",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Submit the matching sha prefix. Can be repeated. Without --sha, "
            "the command only renders the candidate tree."
        ),
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="RUN_ID",
        help="Restrict indexing and candidate selection to one run. Can be repeated or comma-separated.",
    )
    parser.add_argument("--full-view", action="store_true")
    parser.add_argument(
        "--tree-sort",
        choices=["public", "cv"],
        default="cv",
        help="Sort submission table families by best CV or public score.",
    )
    parser.add_argument(
        "--show-seeds",
        action="store_true",
        help="Show seeded clone artifacts in the candidate tree.",
    )
    profile_rerun_view = parser.add_mutually_exclusive_group()
    profile_rerun_view.add_argument(
        "--include-profile-reruns",
        action="store_true",
        help="Show internal profile-calibration reruns alongside submission rows.",
    )
    profile_rerun_view.add_argument(
        "--profile-reruns-only",
        action="store_true",
        help="Show only internal profile-calibration reruns for diagnostic review.",
    )
    parser.add_argument(
        "--no-remote",
        action="store_true",
        help="Do not fetch Kaggle submissions or synchronize public scores.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console(stderr=args.output_format in {"json", "txt"})
    output_console = Console(soft_wrap=True)
    try:
        sha_filters = parse_sha256_filters(args.sha256)
        run_filters = parse_run_filters(args.run)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    if args.limit is not None and args.limit < 0:
        console.print("[red]--limit must be greater than or equal to 0.[/red]")
        return 2
    if args.registry_limit is not None and args.registry_limit < 0:
        console.print("[red]--registry-limit must be greater than or equal to 0.[/red]")
        return 2
    candidate_limit = adaptive_table_limit() if args.limit is None else args.limit
    registry_limit = (
        adaptive_table_limit()
        if args.registry_limit is None
        else None if args.registry_limit == 0 else args.registry_limit
    )

    registry = smart.SubmissionRegistry.load(args.registry)
    client = None
    remote_submissions = None
    if not args.no_remote:
        client, remote_submissions = sync_registry_from_kaggle(
            console=console,
            registry=registry,
            competition=args.competition,
        )
    with _build_progress(console) as progress:
        try:
            index = refresh_index(
                logs_dir=args.logs_dir,
                index_path=args.index,
                competition=args.competition,
                runs=run_filters,
                reindex=args.reindex,
                progress=progress,
            )
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            return 2

    records = filter_records_by_run(list(index.get("records", [])), run_filters)
    if remote_submissions is not None:
        recovered = smart.recover_registry_from_remote(
            registry=registry,
            competition=args.competition,
            remote_submissions=remote_submissions,
            candidates=[_record_to_candidate(record) for record in records],
        )
        if recovered:
            console.print(f"Recovered {recovered} interrupted submission(s).")
    visible_records = filter_submission_lab_visibility(
        records,
        registry=registry,
        competition=args.competition,
        include_profile_reruns=args.include_profile_reruns,
        profile_reruns_only=args.profile_reruns_only,
    )
    try:
        if sha_filters:
            selected = filter_records_by_sha256(records, sha_filters)
            for record in selected:
                if not _record_is_submit_ready(record):
                    raise ValueError(
                        f"Record {(record.get('sha256') or '')[:10]} is not submit-ready."
                    )
                if registry.is_submitted(
                    competition=args.competition,
                    sha256=record.get("sha256"),
                    run=str(record.get("run")),
                    step=int(record.get("step") or -1),
                    timestamp=str(record.get("timestamp")),
                ):
                    raise ValueError(
                        f"Record {(record.get('sha256') or '')[:10]} is already submitted."
                    )
        else:
            selected = select_top_records(
                visible_records,
                registry=registry,
                competition=args.competition,
                limit=candidate_limit,
            )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    if sha_filters:
        if client is None:
            client = smart._build_kaggle_client()
        try:
            submitted = submit_records(
                track(selected, description="Submitting to Kaggle"),
                registry=registry,
                client=client,
                competition=args.competition,
            )
        except smart.KaggleSubmitError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        console.print(f"Submitted {len(submitted)} candidate(s).")
    else:
        if args.output_format == "rich":
            render_candidate_tree_table(
                console,
                selected=selected,
                registry=registry,
                remote_submissions=remote_submissions,
                records=records,
                sort_by=args.tree_sort,
                limit=candidate_limit,
                run_filters=run_filters,
                competition=args.competition,
                show_seeds=args.show_seeds,
                include_profile_reruns=args.include_profile_reruns,
                profile_reruns_only=args.profile_reruns_only,
            )
            if remote_submissions is not None:
                console.print(f"Remote Kaggle submissions visible: {len(remote_submissions)}")
        elif args.output_format == "txt":
            render_text_table(
                output_console,
                "Top unsent submit-ready candidates",
                *candidate_display_table(selected, full_view=args.full_view),
            )
            if args.include_profile_reruns or args.profile_reruns_only:
                profile_reruns = [
                    record
                    for record in visible_records
                    if _is_profile_calibration_rerun(record)
                ]
                render_text_table(
                    output_console,
                    "Profile calibration reruns",
                    *candidate_display_table(profile_reruns, full_view=args.full_view),
                )
            render_text_table(
                output_console,
                "Submission registry",
                *registry_display_table(
                    registry,
                    remote_submissions,
                    records=records,
                    full_view=args.full_view,
                    limit=registry_limit,
                    run_filters=run_filters,
                    competition=args.competition,
                    include_profile_reruns=args.include_profile_reruns,
                    profile_reruns_only=args.profile_reruns_only,
                ),
            )
            if remote_submissions is not None:
                output_console.print(
                    f"Remote Kaggle submissions visible: {len(remote_submissions)}"
                )
        else:
            print(
                json.dumps(
                    build_output_payload(
                        selected=selected,
                        registry=registry,
                        remote_submissions=remote_submissions,
                        records=records,
                        full_view=args.full_view,
                        registry_limit=registry_limit,
                        run_filters=run_filters,
                        competition=args.competition,
                        include_profile_reruns=args.include_profile_reruns,
                        profile_reruns_only=args.profile_reruns_only,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
