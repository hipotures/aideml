from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

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

from scripts import smart_kaggle_submit as smart
from aide.journal import Journal
from aide.utils import serialize
from aide.utils.artifact_manifest import (
    RESULT_MANIFEST_NAME,
    artifact_timestamp_from_ctime,
    write_node_artifact_manifest,
)


DEFAULT_COMPETITION = smart.DEFAULT_COMPETITION
DEFAULT_LOGS_DIR = smart.DEFAULT_LOGS_DIR
DEFAULT_INDEX_PATH = Path("logs/submission_index.json")
DEFAULT_REGISTRY = smart.DEFAULT_REGISTRY


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def backfill_legacy_source_manifests(run_dir: Path) -> int:
    journal_path = run_dir / "journal.json"
    artifacts_dir = run_dir / "artifacts"
    if not journal_path.exists() or not artifacts_dir.exists():
        return 0

    journal = serialize.load_json(journal_path, Journal)
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
        if current is None or _record_sort_key(record) > _record_sort_key(current):
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
        "artifact_dir": str(artifact_dir),
        "solution_path": str(solution_path),
        "submission_path": str(submission_path),
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
    artifacts_dir = logs_dir / run / "artifacts"
    if not artifacts_dir.exists():
        return records
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
        records.append(
            {
                **base,
                "kind": manifest.get("kind") or "source_node",
                "step": node.get("step"),
                "node_id": node.get("id"),
                "parent_node_id": node.get("parent_id"),
                "local_score": manifest.get("local_score", metric.get("value")),
                "metric_maximize": manifest.get(
                    "metric_maximize",
                    metric.get("maximize", True),
                ),
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
        submission_path = Path(record["submission_path"])
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
    reindex: bool = False,
    progress: Any | None = None,
) -> dict[str, Any]:
    logs_dir = Path(logs_dir)
    index = _load_json(index_path) or {"records": [], "runs": {}}
    existing_records = list(index.get("records", []))
    cached_runs = dict(index.get("runs", {}))
    records_by_run: dict[str, list[dict[str, Any]]] = {}
    for record in existing_records:
        records_by_run.setdefault(record.get("run"), []).append(record)

    run_signatures: dict[str, Any] = dict(cached_runs)
    run_dirs = [
        run_dir
        for run_dir in sorted(logs_dir.iterdir())
        if run_dir.is_dir()
        and (run_dir / "artifacts").exists()
    ]
    task_id = None
    if progress is not None:
        task_id = progress.add_task("Indexing AIDE runs", total=len(run_dirs))
    for run_dir in run_dirs:
        run = run_dir.name
        try:
            backfilled = backfill_legacy_source_manifests(run_dir)
            if not reindex and _run_is_unchanged(
                run=run,
                run_dir=run_dir,
                cached_runs=cached_runs,
            ) and backfilled == 0:
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
    refreshed = {
        "version": 1,
        "competition": competition,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": all_records,
        "runs": run_signatures,
    }
    _write_json(index_path, refreshed)
    return refreshed


def _record_is_submit_ready(record: dict[str, Any]) -> bool:
    return (
        record.get("status") == "ok"
        and not record.get("is_buggy")
        and record.get("local_score") is not None
        and record.get("sha256") is not None
        and Path(record.get("submission_path", "")).exists()
    )


def parse_sha256_filters(values: list[str] | None) -> list[str]:
    return smart.parse_sha256_filters(values)


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
        record = matches[0]
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


def render_table(
    console: Console,
    records: list[dict[str, Any]],
    *,
    full_view: bool = False,
) -> None:
    show_source = any(record.get("kind") == "profile_eval" for record in records)
    table = Table(title="Top unsent submit-ready candidates", padding=(0, 1))
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("cv", justify="right", no_wrap=True)
    table.add_column("k", no_wrap=True)
    if full_view:
        table.add_column("prof", no_wrap=True)
    table.add_column("m", no_wrap=True)
    table.add_column("run", no_wrap=True, overflow="ellipsis", max_width=42)
    table.add_column("Algo", no_wrap=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("date", no_wrap=True)
    table.add_column("sha", no_wrap=True)
    if show_source:
        table.add_column("src_sha", no_wrap=True)
    for rank, record in enumerate(records, start=1):
        models = _short_models(record.get("included_model_types"))
        source_sha = ""
        if record.get("kind") == "profile_eval":
            source_sha = str(record.get("source_sha256") or "")[:10]
        row = [
            str(rank),
            _format_score(record.get("local_score")),
            "e" if record.get("kind") == "profile_eval" else "n",
        ]
        if full_view:
            row.append(_short_profile(record.get("profile")))
        row.extend(
            [
                models,
                _short_run(record.get("run")),
                _format_algo(record),
                _format_step(record.get("step")),
                _timestamp_date(record.get("timestamp")),
                str(record.get("sha256") or "")[:10],
            ]
        )
        if show_source:
            row.append(source_sha)
        table.add_row(*row)
    console.print(table)


def _remote_identity(remote: Any) -> tuple[Any, str, str]:
    description = smart._remote_attr(remote, "description")
    parsed = smart.parse_submission_description(description)
    return (
        smart._remote_ref(remote),
        parsed.get("timestamp") or "",
        str(smart._remote_attr(remote, "file_name") or ""),
    )


def _registry_display_sort_key(row: dict[str, Any]) -> tuple[bool, float, str]:
    public_score = smart._parse_public_score(row.get("public_score"))
    return (
        public_score is not None,
        public_score if public_score is not None else float("-inf"),
        str(row.get("date") or ""),
    )


def _remote_display_rows(
    registry: smart.SubmissionRegistry,
    remote_submissions: list[Any] | None,
    record_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if remote_submissions is None:
        return []

    known_refs = {smart._entry_ref(entry) for entry in registry.entries}
    known_timestamps = {str(entry.get("timestamp") or "") for entry in registry.entries}
    known_hashes = [
        str(entry.get("sha256") or "")
        for entry in registry.entries
        if entry.get("sha256")
    ]
    known_files = {
        str(entry.get("remote_filename") or "")
        for entry in registry.entries
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
            "algo": parsed.get("algo"),
            "run": parsed.get("run")
            or str(smart._remote_attr(remote, "file_name") or "-"),
            "step": parsed.get("step"),
            "date": parsed.get("timestamp") or smart._remote_attr(remote, "date"),
            "sha256": remote_sha,
        }
        if record_lookup is not None:
            row["algo"] = row.get("algo") or _registry_entry_algo(row, record_lookup)
            row["artifact_dir"] = _registry_entry_artifact_dir(row, record_lookup)
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
        return lookup[("sha", sha)]

    run = str(entry.get("run") or "")
    step = str(entry.get("step") if entry.get("step") is not None else "")
    timestamp = str(entry.get("timestamp") or "")
    if run and step and timestamp:
        return lookup.get(("node", f"{run}|{step}|{timestamp}"))
    return None


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
        solution_path = Path(artifact_dir) / "solution.py"
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
        path = Path(str(value))
        if key in {"submission_path", "upload_path"}:
            path = path.parent
        return str(path)
    return "-"


def render_registry_table(
    console: Console,
    registry: smart.SubmissionRegistry,
    remote_submissions: list[Any] | None = None,
    records: list[dict[str, Any]] | None = None,
    full_view: bool = False,
) -> None:
    table = Table(title="Submission registry", padding=(0, 1))
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("cv", justify="right", no_wrap=True)
    table.add_column("public", justify="right", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("run", no_wrap=True, overflow="ellipsis", max_width=42)
    table.add_column("Algo", no_wrap=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("date", no_wrap=True)
    table.add_column("sha", no_wrap=True)
    if full_view:
        table.add_column("artifact", no_wrap=True, overflow="fold")

    record_lookup = _registry_record_lookup(records)
    rows = [
        {
            "local_score": entry.get("local_score"),
            "public_score": entry.get("public_score"),
            "remote_status": entry.get("remote_status"),
            "algo": _registry_entry_algo(entry, record_lookup),
            "artifact_dir": _registry_entry_artifact_dir(entry, record_lookup),
            "run": entry.get("run"),
            "step": entry.get("step"),
            "date": entry.get("timestamp"),
            "sha256": entry.get("sha256"),
        }
        for entry in registry.entries
    ]
    rows.extend(_remote_display_rows(registry, remote_submissions, record_lookup))

    complete_rank = 0
    for entry in sorted(rows, key=_registry_display_sort_key, reverse=True):
        remote_status = str(entry.get("remote_status") or "")
        if remote_status.upper() == "COMPLETE":
            complete_rank += 1
            display_rank = str(complete_rank)
        else:
            display_rank = "-"
        row = [
            display_rank,
            _format_score(entry.get("local_score")),
            _format_public_score(entry.get("public_score")),
            remote_status or "-",
            str(entry.get("run") or "-"),
        ]
        row.extend(
            [
                _format_algo(entry, unknown_if_missing=True),
                _format_step(entry.get("step")),
                _display_date(entry.get("date")),
                str(entry.get("sha256") or "")[:10] or "-",
            ]
        )
        if full_view:
            row.append(str(entry.get("artifact_dir") or "-"))
        table.add_row(*row)
    console.print(table)


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
        submission_path=Path(record.get("submission_path")),
        sha256=record.get("sha256"),
        validation_error=None,
        algo=_format_algo(record),
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
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--sha256", action="append", default=[], metavar="PREFIX")
    parser.add_argument("--full-view", action="store_true")
    parser.add_argument(
        "--no-remote",
        action="store_true",
        help="Do not fetch Kaggle submissions or synchronize public scores.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    try:
        sha_filters = parse_sha256_filters(args.sha256)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

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
        index = refresh_index(
            logs_dir=args.logs_dir,
            index_path=args.index,
            competition=args.competition,
            reindex=args.reindex,
            progress=progress,
        )

    records = list(index.get("records", []))
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
                records,
                registry=registry,
                competition=args.competition,
                limit=args.limit,
            )
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    if args.submit:
        if client is None:
            client = smart._build_kaggle_client()
        submitted = submit_records(
            track(selected, description="Submitting to Kaggle"),
            registry=registry,
            client=client,
            competition=args.competition,
        )
        console.print(f"Submitted {len(submitted)} candidate(s).")
    else:
        render_table(console, selected, full_view=args.full_view)
        render_registry_table(
            console,
            registry,
            remote_submissions,
            records=records,
            full_view=args.full_view,
        )
        if remote_submissions is not None:
            console.print(f"Remote Kaggle submissions visible: {len(remote_submissions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
