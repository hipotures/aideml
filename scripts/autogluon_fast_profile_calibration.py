from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import dotenv_values

from aide.autogluon_preprocess import extract_preprocess_source, resolve_autogluon_settings
from scripts import kaggle_submission_lab as lab
from scripts import rerun_autogluon_profile as rerun


REQUIRED_MODEL_FAMILY = ["XGB", "GBM", "CAT"]
SESSION_STATE_NAME = "state.json"
EXPERIMENTS_NAME = "experiments.jsonl"
DECISIONS_NAME = "decisions.jsonl"
SOURCE_INVENTORY_NAME = "source_inventory.csv"
SOURCE_SETS_NAME = "source_sets.json"
PROFILE_METRICS_NAME = "profile_metrics.json"
PROFILE_METRICS_CSV_NAME = "profile_metrics.csv"
HUMAN_SUMMARY_NAME = "human_summary.md"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_timestamp() -> str:
    return utc_now().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    x_sum = sum((value - mean_x) ** 2 for value in xs)
    y_sum = sum((value - mean_y) ** 2 for value in ys)
    if x_sum == 0 or y_sum == 0:
        return None
    return sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys)
    ) / (x_sum * y_sum) ** 0.5


def _spearman(rows: list[dict[str, Any]]) -> float | None:
    local = [float(row["local_cv"]) for row in rows]
    public = [float(row["source_public_score"]) for row in rows]
    return _pearson(_rankdata(local), _rankdata(public))


def _top_k_overlap(rows: list[dict[str, Any]], k: int) -> float | None:
    if len(rows) < k:
        return None
    local = {
        row["source_sha256"]
        for row in sorted(rows, key=lambda row: row["local_cv"], reverse=True)[:k]
    }
    public = {
        row["source_sha256"]
        for row in sorted(
            rows, key=lambda row: row["source_public_score"], reverse=True
        )[:k]
    }
    return len(local & public) / k


def _source_subset_sensitivity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        value
        for index in range(len(rows))
        if (value := _spearman(rows[:index] + rows[index + 1 :])) is not None
    ]
    if not values:
        return {"method": "leave_one_source_out", "n": 0}
    return {
        "method": "leave_one_source_out",
        "n": len(values),
        "min_spearman": min(values),
        "median_spearman": statistics.median(values),
        "max_spearman": max(values),
        "range_spearman": max(values) - min(values),
    }


def _bootstrap_spearman(rows: list[dict[str, Any]], *, draws: int = 400) -> dict[str, Any]:
    if len(rows) < 3:
        return {"method": "paired_bootstrap", "draws": 0}
    generator = random.Random(1729)
    values = []
    for _ in range(draws):
        sample = [rows[generator.randrange(len(rows))] for _ in rows]
        if (value := _spearman(sample)) is not None:
            values.append(value)
    if not values:
        return {"method": "paired_bootstrap", "draws": draws, "usable_draws": 0}
    values.sort()
    lower = values[max(0, int(0.025 * (len(values) - 1)))]
    upper = values[min(len(values) - 1, int(0.975 * (len(values) - 1)))]
    return {
        "method": "paired_bootstrap",
        "draws": draws,
        "usable_draws": len(values),
        "spearman_95_ci": [lower, upper],
    }


def _event_panel(event: dict[str, Any]) -> str | None:
    signature = event.get("source_feature_signature")
    if not isinstance(signature, dict):
        return None
    panel = signature.get("panel")
    return str(panel) if panel else None


def _require_session_id(state: dict[str, Any]) -> str:
    session_id = str(state.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("Session state is missing a non-empty session_id.")
    return session_id


def _verification_updates(
    events: list[dict[str, Any]], *, session_id: str
) -> dict[str, dict[str, Any]]:
    """Return the most recent eligibility evidence for each completed rerun.

    Experiment JSONL is append-only.  A later verification event is therefore
    the only permitted way to correct family-eligibility evidence for a prior
    completion without rewriting the original observation.
    """
    updates: dict[str, dict[str, Any]] = {}
    for event in events:
        if (
            event.get("event_type") == "eligibility_verification"
            and event.get("session_id") == session_id
            and event.get("experiment_id")
        ):
            updates[str(event["experiment_id"])] = event
    return updates


def _resolved_completed_events(
    events: list[dict[str, Any]], *, session_id: str
) -> list[dict[str, Any]]:
    updates = _verification_updates(events, session_id=session_id)
    completed: list[dict[str, Any]] = []
    for event in events:
        if (
            event.get("event_type") != "completed"
            or event.get("session_id") != session_id
        ):
            continue
        resolved = dict(event)
        verification = updates.get(str(event.get("experiment_id") or ""))
        if verification is not None:
            for field in (
                "successfully_trained_model_types",
                "eligible_for_selection_model_types",
                "failed_or_skipped_model_types",
                "all_required_model_types_trained",
                "source_code_unchanged",
                "valid",
                "invalid_reason",
            ):
                if field in verification:
                    resolved[field] = verification[field]
            resolved["eligibility_evidence_verified"] = bool(
                verification.get("verification_status") == "ok"
            )
        completed.append(resolved)
    return completed


def _event_is_valid_for_metrics(event: dict[str, Any]) -> bool:
    configured = event.get("configured_model_types")
    try:
        time_limit = int(event.get("time_limit"))
    except (TypeError, ValueError):
        return False
    return (
        event.get("status") == "ok"
        and bool(event.get("valid"))
        and bool(event.get("all_required_model_types_trained"))
        and configured == REQUIRED_MODEL_FAMILY
        and event.get("successfully_trained_model_types") == REQUIRED_MODEL_FAMILY
        and event.get("eligible_for_selection_model_types") == REQUIRED_MODEL_FAMILY
        and event.get("preset") == "medium_quality"
        and 0 < time_limit <= 600
        and event.get("source_code_unchanged") is True
    )


def _profile_metrics(
    events: list[dict[str, Any]],
    *,
    session_id: str,
    panel: str | None = None,
    expected_source_sha256s: set[str] | None = None,
) -> dict[str, Any]:
    if not str(session_id or "").strip():
        raise ValueError("Current-session metrics require a non-empty session_id.")
    completed = [
        event
        for event in _resolved_completed_events(events, session_id=session_id)
        if panel is None or _event_panel(event) == panel
    ]
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in completed:
        by_profile[str(event.get("profile") or "")].append(event)

    source_sets = {
        profile: {
            str(event["source_sha256"])
            for event in events_
            if _event_is_valid_for_metrics(event) and event.get("source_sha256")
        }
        for profile, events_ in by_profile.items()
    }
    common_sources = (
        sorted(set.intersection(*source_sets.values())) if source_sets else []
    )
    summaries = []
    for profile, profile_events in sorted(by_profile.items()):
        valid = []
        for event in profile_events:
            local_cv = _float(event.get("local_cv"))
            public_score = _float(event.get("source_public_score"))
            if (
                _event_is_valid_for_metrics(event)
                and (
                    expected_source_sha256s is None
                    or str(event.get("source_sha256") or "")
                    in expected_source_sha256s
                )
                and local_cv is not None
                and public_score is not None
            ):
                valid.append(
                    {
                        **event,
                        "local_cv": local_cv,
                        "source_public_score": public_score,
                    }
                )
        errors = [
            {
                "source_sha256": row["source_sha256"],
                "local_cv": row["local_cv"],
                "source_public_score": row["source_public_score"],
                "signed_error": row["local_cv"] - row["source_public_score"],
                "absolute_error": abs(row["local_cv"] - row["source_public_score"]),
            }
            for row in valid
        ]
        runtimes = [
            runtime
            for event in profile_events
            if (runtime := _float(event.get("wall_clock_seconds"))) is not None
        ]
        public_median = _median(row["source_public_score"] for row in valid)
        top_local = sorted(valid, key=lambda row: row["local_cv"], reverse=True)[:3]
        under_median_top = [
            row["source_sha256"]
            for row in top_local
            if public_median is not None and row["source_public_score"] < public_median
        ]
        trained_success = sum(
            event.get("successfully_trained_model_types") == REQUIRED_MODEL_FAMILY
            and event.get("eligible_for_selection_model_types")
            == REQUIRED_MODEL_FAMILY
            and bool(event.get("all_required_model_types_trained"))
            for event in profile_events
        )
        failures = [event for event in profile_events if event.get("status") != "ok"]
        timed_out = [
            event
            for event in profile_events
            if event.get("execution_exc_type") == "TimeoutError"
        ]
        signed_errors = [error["signed_error"] for error in errors]
        absolute_errors = [error["absolute_error"] for error in errors]
        summaries.append(
            {
                "profile": profile,
                "panel": panel,
                "attempted": len(profile_events),
                "valid": len(valid),
                "invalid": len(profile_events) - len(valid),
                "failed": len(failures),
                "timed_out": len(timed_out),
                "failure_rate": len(failures) / len(profile_events)
                if profile_events
                else None,
                "required_model_training_success_count": trained_success,
                "required_model_training_success_rate": trained_success
                / len(profile_events)
                if profile_events
                else None,
                "source_sha256s": sorted(
                    row["source_sha256"] for row in valid
                ),
                "common_source_sha256s_with_all_profiles": common_sources,
                "source_set_matches_all_profiles": bool(source_sets)
                and set(row["source_sha256"] for row in valid) == set(common_sources),
                "expected_source_sha256s": sorted(expected_source_sha256s or []),
                "complete_frozen_panel": (
                    expected_source_sha256s is not None
                    and len(valid) == len(expected_source_sha256s)
                    and set(row["source_sha256"] for row in valid)
                    == expected_source_sha256s
                ),
                "pearson": _pearson(
                    [row["local_cv"] for row in valid],
                    [row["source_public_score"] for row in valid],
                ),
                "spearman": _spearman(valid),
                "top_3_overlap": _top_k_overlap(valid, 3),
                "top_5_overlap": _top_k_overlap(valid, 5),
                "top_local_below_public_median": under_median_top,
                "mae": _mean(absolute_errors),
                "median_absolute_error": _median(absolute_errors),
                "signed_bias": _mean(signed_errors),
                "worst_over_optimistic": sorted(
                    errors, key=lambda row: row["signed_error"], reverse=True
                )[:3],
                "worst_under_optimistic": sorted(
                    errors, key=lambda row: row["signed_error"]
                )[:3],
                "mean_runtime_seconds": _mean(runtimes),
                "median_runtime_seconds": _median(runtimes),
                "max_runtime_seconds": max(runtimes) if runtimes else None,
                "source_subset_sensitivity": _source_subset_sensitivity(valid),
                "spearman_uncertainty": _bootstrap_spearman(valid),
            }
        )
    return {
        "required_model_family": REQUIRED_MODEL_FAMILY,
        "panel": panel,
        "expected_source_sha256s": sorted(expected_source_sha256s or []),
        "completed_experiments": len(completed),
        "valid_experiments": sum(summary["valid"] for summary in summaries),
        "profiles": summaries,
    }


def _profile_metrics_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for summary in payload.get("profiles", []):
        rows.append(
            {
                key: summary.get(key)
                for key in (
                    "profile",
                    "attempted",
                    "valid",
                    "invalid",
                    "failed",
                    "timed_out",
                    "required_model_training_success_rate",
                    "pearson",
                    "spearman",
                    "top_3_overlap",
                    "top_5_overlap",
                    "mae",
                    "median_absolute_error",
                    "signed_bias",
                    "mean_runtime_seconds",
                    "median_runtime_seconds",
                    "max_runtime_seconds",
                )
            }
        )
    return rows


def _write_profile_metrics_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = _profile_metrics_csv_rows(payload)
    fields = list(rows[0]) if rows else ["profile"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _deadline_state(state: dict[str, Any]) -> tuple[int, int]:
    start = dt.datetime.fromisoformat(str(state["task_start_time"]).replace("Z", "+00:00"))
    deadline = dt.datetime.fromisoformat(str(state["task_deadline"]).replace("Z", "+00:00"))
    now = utc_now()
    return (
        max(0, int((now - start).total_seconds())),
        max(0, int((deadline - now).total_seconds())),
    )


def _state_counts(events: list[dict[str, Any]], *, session_id: str) -> dict[str, int]:
    if not str(session_id or "").strip():
        raise ValueError("Current-session counts require a non-empty session_id.")
    completed = _resolved_completed_events(events, session_id=session_id)
    return {
        "attempted": len(completed),
        "successful": sum(event.get("status") == "ok" for event in completed),
        "failed": sum(event.get("status") != "ok" for event in completed),
        "timed_out": sum(
            event.get("execution_exc_type") == "TimeoutError" for event in completed
        ),
        "invalid": sum(not _event_is_valid_for_metrics(event) for event in completed),
        "valid": sum(_event_is_valid_for_metrics(event) for event in completed),
    }


def _comparison_eligible_profiles(
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        summary
        for summary in summaries
        if summary.get("complete_frozen_panel") is True
        and summary.get("required_model_training_success_rate") == 1.0
        and summary.get("spearman") is not None
    ]


def refresh_session_outputs(session_dir: Path) -> dict[str, Any]:
    state_path = session_dir / SESSION_STATE_NAME
    state = _read_json(state_path)
    session_id = _require_session_id(state)
    panel_sources = _frozen_panel_source_sets(session_dir / SOURCE_SETS_NAME)
    if "development" not in panel_sources or "confirmation" not in panel_sources:
        raise ValueError("Frozen source sets must define development and confirmation panels.")
    events = _load_jsonl(session_dir / EXPERIMENTS_NAME)
    all_current = _profile_metrics(events, session_id=session_id)
    development = _profile_metrics(
        events,
        session_id=session_id,
        panel="development",
        expected_source_sha256s=panel_sources["development"],
    )
    confirmation = _profile_metrics(
        events,
        session_id=session_id,
        panel="confirmation",
        expected_source_sha256s=panel_sources["confirmation"],
    )
    metrics = {
        "session_id": session_id,
        "generated_at": utc_timestamp(),
        "required_model_family": REQUIRED_MODEL_FAMILY,
        "all_current_session": all_current,
        "development": development,
        "confirmation": confirmation,
        # Compatibility aliases are deliberately development-only, so a
        # caller cannot mistake confirmation evidence for selection evidence.
        "completed_experiments": all_current["completed_experiments"],
        "valid_experiments": all_current["valid_experiments"],
        "profiles": development["profiles"],
    }
    _write_json(session_dir / PROFILE_METRICS_NAME, metrics)
    _write_profile_metrics_csv(session_dir / PROFILE_METRICS_CSV_NAME, metrics)

    elapsed, remaining = _deadline_state(state)
    state["elapsed_seconds"] = elapsed
    state["remaining_seconds"] = remaining
    state["experiments"] = _state_counts(events, session_id=session_id)
    completed = _resolved_completed_events(events, session_id=session_id)
    if completed:
        state["last_completed_experiment"] = completed[-1].get("experiment_id")

    eligible = _comparison_eligible_profiles(development["profiles"])
    state["development_comparison_ready"] = len(eligible) >= 2
    if len(eligible) >= 2:
        best = max(
            eligible,
            key=lambda summary: (
                float(summary["spearman"]),
                -(summary.get("mae") or float("inf")),
            ),
        )
        state["current_best_development_profile"] = best["profile"]
    else:
        state["current_best_development_profile"] = None

    finalists = [
        str(profile)
        for profile in state.get("frozen_finalists", [])
        if str(profile).strip()
    ]
    confirmation_events = [
        event for event in completed if _event_panel(event) == "confirmation"
    ]
    confirmation_by_profile = {
        str(summary.get("profile")): summary for summary in confirmation["profiles"]
    }
    if not finalists:
        state["confirmation_status"] = (
            "protocol_breach_unfrozen_profile"
            if confirmation_events
            else "not_started"
        )
    elif all(
        confirmation_by_profile.get(profile, {}).get("complete_frozen_panel") is True
        and confirmation_by_profile.get(profile, {}).get(
            "required_model_training_success_rate"
        )
        == 1.0
        for profile in finalists
    ):
        state["confirmation_status"] = "complete"
    elif confirmation_events:
        state["confirmation_status"] = "in_progress"
    else:
        state["confirmation_status"] = "awaiting_confirmation"
    state.setdefault("finalization_status", "active")
    _write_json(state_path, state)

    lines = [
        f"# AutoGluon fast-profile session {session_id}",
        "",
        f"Fresh completed reruns: {all_current['completed_experiments']}; valid: {all_current['valid_experiments']}.",
        "",
        "## Development panel",
        "",
        "| Profile | Valid | Spearman | MAE | Median runtime (s) | Three-family rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in development["profiles"]:
        lines.append(
            "| {profile} | {valid} | {spearman} | {mae} | {runtime} | {training} |".format(
                profile=summary["profile"],
                valid=summary["valid"],
                spearman=_format_metric(summary.get("spearman")),
                mae=_format_metric(summary.get("mae")),
                runtime=_format_metric(summary.get("median_runtime_seconds")),
                training=_format_metric(
                    summary.get("required_model_training_success_rate")
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Confirmation panel",
            "",
            "| Profile | Valid | Spearman | MAE | Median runtime (s) | Three-family rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in confirmation["profiles"]:
        lines.append(
            "| {profile} | {valid} | {spearman} | {mae} | {runtime} | {training} |".format(
                profile=summary["profile"],
                valid=summary["valid"],
                spearman=_format_metric(summary.get("spearman")),
                mae=_format_metric(summary.get("mae")),
                runtime=_format_metric(summary.get("median_runtime_seconds")),
                training=_format_metric(
                    summary.get("required_model_training_success_rate")
                ),
            )
        )
    (session_dir / HUMAN_SUMMARY_NAME).write_text("\n".join(lines) + "\n")
    return metrics


def _format_metric(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _memory_bytes() -> dict[str, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw_value = line.split(":", 1)
            values[key] = int(raw_value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return {"total_bytes": None, "available_bytes": None}
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
    }


def _profile_config(profile: str, competition: str) -> dict[str, Any]:
    cfg = rerun.build_profile_config(
        source_record={},
        profile=profile,
        competition=competition,
        presets=None,
        time_limit=None,
        fit_args=None,
    )
    settings = resolve_autogluon_settings(cfg)
    rerun.validate_profile_calibration_settings(
        included_model_types=rerun.resolve_autogluon_included_model_types(cfg),
        presets=str(settings["presets"]),
        time_limit=int(settings["time_limit"]),
    )
    return settings


def _profile_config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _active_rerun_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError("Unable to inspect running processes before a rerun.")
    active = []
    current_processes = {os.getpid(), os.getppid()}
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or int(fields[0]) in current_processes:
            continue
        command = fields[1]
        if (
            "rerun_autogluon_profile.py" in command
            or "autogluon_fast_profile_calibration.py" in command
        ):
            active.append(line.strip())
    return active


def _read_inventory(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["source_sha256"]): row for row in rows}


def _source_panels(path: Path) -> dict[str, str]:
    payload = _read_json(path)
    panels = payload.get("panels") or {}
    source_panels: dict[str, str] = {}
    for panel_name, panel in panels.items():
        for source in panel.get("sources", []):
            source_sha = str(source.get("source_sha256") or "")
            if not source_sha:
                raise ValueError(f"Panel {panel_name!r} has a source without a SHA.")
            if source_sha in source_panels:
                raise ValueError(f"Source {source_sha} appears in more than one panel.")
            source_panels[source_sha] = str(panel_name)
    return source_panels


def _frozen_panel_source_sets(path: Path) -> dict[str, set[str]]:
    payload = _read_json(path)
    panels = payload.get("panels")
    if not isinstance(panels, dict):
        raise ValueError(f"Frozen panel file is missing panels: {path}")
    source_sets: dict[str, set[str]] = {}
    for panel_name, panel in panels.items():
        sources = panel.get("sources", []) if isinstance(panel, dict) else []
        shas = {
            str(source.get("source_sha256") or "")
            for source in sources
            if isinstance(source, dict)
        }
        if not shas or "" in shas:
            raise ValueError(f"Frozen panel {panel_name!r} has an invalid source SHA.")
        source_sets[str(panel_name)] = shas
    return source_sets


def _source_record(
    *,
    index_path: Path,
    source_sha256: str,
    source_solution_path: str,
    competition: str,
) -> dict[str, Any]:
    index = _read_json(index_path)
    matches = [
        record
        for record in index.get("records", [])
        if record.get("competition") == competition
        and record.get("kind") == "source_node"
        and record.get("sha256") == source_sha256
    ]
    expected_solution_path = lab._record_path(source_solution_path).resolve()
    path_matches = [
        record
        for record in matches
        if lab._record_path(record.get("solution_path")).resolve()
        == expected_solution_path
    ]
    if len(path_matches) == 1:
        return path_matches[0]
    if len(matches) != 1:
        raise ValueError(
            "Expected one original source_node matching the frozen solution path for "
            f"{source_sha256}, found {len(path_matches)} path matches among {len(matches)}."
        )
    return matches[0]


def _source_code_hash(path: str | None) -> str | None:
    source_path = lab._record_path(path)
    return lab.sha256_file(source_path) if source_path.exists() else None


def build_source_inventory(
    *,
    registry_path: Path,
    output_path: Path,
    competition: str,
) -> list[dict[str, Any]]:
    registry_payload = _read_json(registry_path)
    entries = registry_payload.get("submissions") or registry_payload.get("entries") or []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.get("competition") != competition:
            continue
        public_score = lab.smart._parse_public_score(entry.get("public_score"))
        if str(entry.get("remote_status") or "").upper() != "COMPLETE" or public_score is None:
            continue
        submission_path = lab._record_path(entry.get("submission_path"))
        artifact_dir = submission_path.parent
        manifest = _read_json(artifact_dir / "aide_result.json")
        if manifest.get("kind") != "source_node" or manifest.get("is_buggy"):
            continue
        source_sha256 = str(entry.get("sha256") or manifest.get("sha256") or "")
        if not source_sha256 or source_sha256 in seen:
            continue
        solution_path = artifact_dir / "solution.py"
        if not solution_path.exists():
            continue
        solution_code_sha256 = lab.sha256_file(solution_path)
        try:
            extract_preprocess_source(solution_path.read_text())
            rerunnable = True
        except ValueError:
            rerunnable = False
        run_stats = manifest.get("run_stats") or {}
        node = manifest.get("node") or {}
        rows.append(
            {
                "source_sha256": source_sha256,
                "source_code_sha256": solution_code_sha256,
                "source_solution_path": str(solution_path),
                "source_artifact_dir": str(artifact_dir),
                "source_run": manifest.get("run"),
                "source_step": node.get("step"),
                "source_public_score": public_score,
                "source_local_score": manifest.get("local_score"),
                "eval_metric": manifest.get("eval_metric"),
                "feature_count": run_stats.get("feature_count"),
                "rerunnable": rerunnable,
                "remote_status": entry.get("remote_status"),
            }
        )
        seen.add(source_sha256)
    rows.sort(key=lambda row: (row["source_public_score"], row["source_sha256"]))
    fields = [
        "source_sha256",
        "source_code_sha256",
        "source_solution_path",
        "source_artifact_dir",
        "source_run",
        "source_step",
        "source_public_score",
        "source_local_score",
        "eval_metric",
        "feature_count",
        "rerunnable",
        "remote_status",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_data_audit(*, output_path: Path, data_dir: Path) -> dict[str, Any]:
    train = pd.read_csv(data_dir / "train.csv.gz")
    test = pd.read_csv(data_dir / "test.csv.gz")
    sample = pd.read_csv(data_dir / "sample_submission.csv.gz")
    target = sample.columns[1]
    id_column = sample.columns[0]
    feature_columns = [column for column in test.columns if column != id_column]
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(test[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    package_versions = {}
    for package in ("autogluon.tabular", "xgboost", "lightgbm", "catboost"):
        try:
            from importlib.metadata import version

            package_versions[package] = version(package)
        except Exception as exc:
            package_versions[package] = f"unavailable: {type(exc).__name__}"
    gpu_probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    audit = {
        "generated_at": utc_timestamp(),
        "data_dir": str(data_dir),
        "train": {"shape": list(train.shape), "dtypes": {k: str(v) for k, v in train.dtypes.items()}},
        "test": {"shape": list(test.shape), "dtypes": {k: str(v) for k, v in test.dtypes.items()}},
        "sample_submission_shape": list(sample.shape),
        "target_column": target,
        "identifier_column": id_column,
        "evaluation_metric": "balanced_accuracy",
        "class_counts": {str(k): int(v) for k, v in train[target].value_counts().items()},
        "class_imbalance_max_min_ratio": float(
            train[target].value_counts().max() / train[target].value_counts().min()
        ),
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_counts": {
            "train": {k: int(v) for k, v in train[feature_columns].isna().sum().items()},
            "test": {k: int(v) for k, v in test[feature_columns].isna().sum().items()},
        },
        "category_levels": {
            column: sorted(str(value) for value in train[column].dropna().unique())
            for column in categorical_columns
        },
        "resources": {
            "cpu_count": os.cpu_count(),
            "memory": _memory_bytes(),
            "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
            "package_versions": package_versions,
            "gpu_probe_returncode": gpu_probe.returncode,
            "gpu_probe_stdout": gpu_probe.stdout.strip(),
            "gpu_probe_stderr": gpu_probe.stderr.strip(),
        },
    }
    _write_json(output_path, audit)
    return audit


def record_decision(session_dir: Path, *, decision: dict[str, Any]) -> None:
    state = _read_json(session_dir / SESSION_STATE_NAME)
    session_id = _require_session_id(state)
    _append_jsonl(
        session_dir / DECISIONS_NAME,
        {
            "session_id": session_id,
            "timestamp": utc_timestamp(),
            **decision,
        },
    )
    next_experiment = decision.get("next_planned_experiment")
    if next_experiment:
        state["next_planned_experiment"] = next_experiment
        _write_json(session_dir / SESSION_STATE_NAME, state)


def _eligibility_verification_event(
    event: dict[str, Any], *, session_id: str
) -> dict[str, Any]:
    artifact_dir = lab._record_path(event.get("artifact_dir"))
    metadata_path = artifact_dir / "submission_eval.json"
    metadata = _read_json(metadata_path)
    metadata_is_current = (
        metadata.get("artifact_role") == "profile_calibration_rerun"
        and metadata.get("profile_calibration_session_id") == session_id
    )
    run_stats = metadata.get("run_stats") if metadata_is_current else {}
    trained = rerun.trained_model_types_from_run_stats(run_stats)
    eligible = rerun.eligible_model_types_from_run_stats(run_stats)
    all_required = (
        trained == REQUIRED_MODEL_FAMILY and eligible == REQUIRED_MODEL_FAMILY
    )
    source_unchanged = (
        event.get("source_code_unchanged") is True
        and metadata.get("source_code_unchanged") is True
    )
    verification_status = "ok" if metadata_is_current else "error"
    invalid_reason = None
    if not metadata_is_current:
        invalid_reason = "missing_or_mismatched_eligibility_evidence"
    elif not source_unchanged:
        invalid_reason = "candidate_source_code_changed"
    elif not all_required:
        invalid_reason = "required_model_family_not_fully_trained"
    elif metadata.get("status") != "ok":
        invalid_reason = "rerun_execution_failed"
    valid = (
        verification_status == "ok"
        and event.get("status") == "ok"
        and metadata.get("status") == "ok"
        and source_unchanged
        and all_required
    )
    evidence_payload = {
        "artifact_metadata_sha256": (
            lab.sha256_file(metadata_path) if metadata_path.exists() else None
        ),
        "metadata_is_current": metadata_is_current,
        "trained": trained,
        "eligible": eligible,
        "source_unchanged": source_unchanged,
    }
    return {
        "session_id": session_id,
        "event_type": "eligibility_verification",
        "experiment_id": event.get("experiment_id"),
        "timestamp": utc_timestamp(),
        "verification_status": verification_status,
        "verification_source": str(metadata_path),
        "evidence_fingerprint": _profile_config_hash(evidence_payload),
        "successfully_trained_model_types": trained,
        "eligible_for_selection_model_types": eligible,
        "failed_or_skipped_model_types": rerun.missing_required_model_types(eligible),
        "all_required_model_types_trained": all_required,
        "source_code_unchanged": source_unchanged,
        "valid": valid,
        "invalid_reason": invalid_reason,
    }


def reconcile_eligibility_evidence(session_dir: Path) -> int:
    """Append verified training-family evidence for prior completed reruns."""
    state = _read_json(session_dir / SESSION_STATE_NAME)
    session_id = _require_session_id(state)
    events_path = session_dir / EXPERIMENTS_NAME
    events = _load_jsonl(events_path)
    existing = _verification_updates(events, session_id=session_id)
    appended = 0
    for event in _resolved_completed_events(events, session_id=session_id):
        if not event.get("experiment_id"):
            continue
        verification = _eligibility_verification_event(event, session_id=session_id)
        previous = existing.get(str(event.get("experiment_id") or ""))
        if (
            previous is not None
            and previous.get("evidence_fingerprint")
            == verification["evidence_fingerprint"]
            and previous.get("verification_status") == verification["verification_status"]
        ):
            continue
        _append_jsonl(events_path, verification)
        appended += 1
    return appended


def freeze_finalists(session_dir: Path, *, profiles: list[str]) -> None:
    state = _read_json(session_dir / SESSION_STATE_NAME)
    session_id = _require_session_id(state)
    normalized = list(dict.fromkeys(str(profile).strip() for profile in profiles if str(profile).strip()))
    if not normalized:
        raise ValueError("At least one finalist profile is required.")
    events = _load_jsonl(session_dir / EXPERIMENTS_NAME)
    completed = _resolved_completed_events(events, session_id=session_id)
    if any(_event_panel(event) == "confirmation" for event in completed):
        raise ValueError("Cannot freeze finalists after confirmation evidence exists.")
    metrics = refresh_session_outputs(session_dir)
    development = {
        str(summary.get("profile")): summary
        for summary in metrics["development"]["profiles"]
    }
    incomplete = [
        profile
        for profile in normalized
        if development.get(profile, {}).get("complete_frozen_panel") is not True
        or development.get(profile, {}).get("required_model_training_success_rate")
        != 1.0
    ]
    if incomplete:
        raise ValueError(
            "Finalists need a complete valid development panel: " + ", ".join(incomplete)
        )
    config_hashes = {
        profile: _profile_config_hash(_profile_config(profile, "playground-series-s6e7"))
        for profile in normalized
    }
    _append_jsonl(
        session_dir / DECISIONS_NAME,
        {
            "session_id": session_id,
            "timestamp": utc_timestamp(),
            "decision_type": "freeze_finalists",
            "frozen_finalists": normalized,
            "frozen_finalist_profile_config_hashes": config_hashes,
            "confirmation_rule": "confirmation runs may use only these frozen profile hashes",
        },
    )
    state = _read_json(session_dir / SESSION_STATE_NAME)
    state["frozen_finalists"] = normalized
    state["frozen_finalist_profile_config_hashes"] = config_hashes
    state["confirmation_status"] = "awaiting_confirmation"
    state["next_planned_experiment"] = "record confirmation-panel experiment"
    _write_json(session_dir / SESSION_STATE_NAME, state)
    refresh_session_outputs(session_dir)


def set_finalization_status(session_dir: Path, *, status: str) -> None:
    state = _read_json(session_dir / SESSION_STATE_NAME)
    session_id = _require_session_id(state)
    _append_jsonl(
        session_dir / DECISIONS_NAME,
        {
            "session_id": session_id,
            "timestamp": utc_timestamp(),
            "decision_type": "finalization_status",
            "finalization_status": status,
        },
    )
    state["finalization_status"] = status
    _write_json(session_dir / SESSION_STATE_NAME, state)


def set_reference_profile(
    session_dir: Path, *, profile: str, competition: str
) -> None:
    _profile_config(profile, competition)
    state = _read_json(session_dir / SESSION_STATE_NAME)
    session_id = _require_session_id(state)
    _append_jsonl(
        session_dir / DECISIONS_NAME,
        {
            "session_id": session_id,
            "timestamp": utc_timestamp(),
            "decision_type": "set_reference_profile",
            "reference_profile": profile,
        },
    )
    state["reference_profile"] = profile
    _write_json(session_dir / SESSION_STATE_NAME, state)


def run_one(
    *,
    session_dir: Path,
    source_sha256: str,
    profile: str,
    profile_intent: str,
    competition: str,
    logs_dir: Path,
    index_path: Path,
    timeout: int,
    memory_limit_gb: float,
) -> dict[str, Any]:
    state = _read_json(session_dir / SESSION_STATE_NAME)
    session_id = _require_session_id(state)
    elapsed, remaining = _deadline_state(state)
    if remaining <= 0:
        raise RuntimeError("The calibration session deadline has passed.")
    if remaining < timeout:
        raise RuntimeError(
            f"Only {remaining}s remain, less than the requested {timeout}s rerun timeout."
        )
    active = _active_rerun_processes()
    if active:
        raise RuntimeError("Another rerun is active: " + "; ".join(active))

    inventory = _read_inventory(session_dir / SOURCE_INVENTORY_NAME)
    source = inventory.get(source_sha256)
    if source is None:
        raise ValueError(f"Source {source_sha256} is not in the frozen inventory.")
    if str(source.get("rerunnable")).lower() != "true":
        raise ValueError(f"Source {source_sha256} is not faithfully rerunnable.")
    panel = _source_panels(session_dir / SOURCE_SETS_NAME).get(source_sha256)
    if panel is None:
        raise ValueError(f"Source {source_sha256} is not in a declared frozen panel.")
    source_record = _source_record(
        index_path=index_path,
        source_sha256=source_sha256,
        source_solution_path=str(source["source_solution_path"]),
        competition=competition,
    )
    source_hash_before = _source_code_hash(source_record.get("solution_path"))
    if source_hash_before != source.get("source_code_sha256"):
        raise ValueError("Source code hash does not match the frozen inventory.")
    profile_config = _profile_config(profile, competition)
    profile_config_hash = _profile_config_hash(profile_config)
    if panel == "confirmation":
        frozen = {
            str(value)
            for value in state.get("frozen_finalists", [])
            if str(value).strip()
        }
        frozen_hashes = state.get("frozen_finalist_profile_config_hashes") or {}
        if profile not in frozen:
            raise ValueError(
                "Confirmation reruns require a previously frozen finalist profile."
            )
        if frozen_hashes.get(profile) != profile_config_hash:
            raise ValueError(
                "Confirmation rerun profile settings differ from the frozen finalist."
            )
    configured_model_types = rerun.resolve_autogluon_included_model_types(
        rerun.build_profile_config(
            source_record=source_record,
            profile=profile,
            competition=competition,
            presets=None,
            time_limit=None,
            fit_args=None,
        )
    )
    experiment_number = len(
        [
            event
            for event in _load_jsonl(session_dir / EXPERIMENTS_NAME)
            if event.get("event_type") == "completed"
        ]
    ) + 1
    experiment_id = f"{session_id}-exp-{experiment_number:03d}"
    plan = {
        "session_id": session_id,
        "event_type": "planned",
        "experiment_id": experiment_id,
        "timestamp": utc_timestamp(),
        "source_sha256": source_sha256,
        "source_public_score": _float(source.get("source_public_score")),
        "source_feature_signature": {
            "feature_count": _float(source.get("feature_count")),
            "source_run": source.get("source_run"),
            "panel": panel,
        },
        "profile": profile,
        "profile_intent": profile_intent,
        "resolved_profile_config": profile_config,
        "profile_config_hash": profile_config_hash,
        "preset": profile_config.get("presets"),
        "time_limit": profile_config.get("time_limit"),
        "configured_model_types": configured_model_types,
        "process_timeout": timeout,
        "submission_guard": "local rerun_autogluon_profile.run_profile_eval only; no Kaggle submission path",
        "expected_deadline_impact_seconds": timeout,
        "elapsed_seconds_before": elapsed,
        "remaining_seconds_before": remaining,
    }
    _append_jsonl(session_dir / EXPERIMENTS_NAME, plan)
    state["next_planned_experiment"] = experiment_id
    _write_json(session_dir / SESSION_STATE_NAME, state)

    try:
        record = rerun.run_profile_eval(
            source_record,
            logs_dir=logs_dir,
            profile=profile,
            competition=competition,
            timeout=timeout,
            memory_limit_gb=memory_limit_gb,
            profile_calibration=True,
            profile_calibration_session_id=session_id,
        )
        source_hash_after = _source_code_hash(source_record.get("solution_path"))
        trained_model_types = list(record.get("trained_model_types") or [])
        eligible_model_types = list(
            record.get("eligible_for_selection_model_types") or []
        )
        all_required_model_types_trained = (
            bool(record.get("all_required_model_types_trained"))
            and trained_model_types == REQUIRED_MODEL_FAMILY
            and eligible_model_types == REQUIRED_MODEL_FAMILY
        )
        source_code_unchanged = source_hash_before == source_hash_after
        valid = (
            bool(record.get("valid_for_current_final_selection"))
            and all_required_model_types_trained
            and source_code_unchanged
        )
        invalid_reason = record.get("invalid_reason")
        if not invalid_reason and not all_required_model_types_trained:
            invalid_reason = "required_model_family_not_fully_trained"
        if not invalid_reason and not source_code_unchanged:
            invalid_reason = "candidate_source_code_changed"
        event = {
            **plan,
            "event_type": "completed",
            "completed_at": utc_timestamp(),
            "status": record.get("status"),
            "valid": valid,
            "invalid_reason": invalid_reason,
            "local_cv": _float(record.get("local_score")),
            "signed_error": (
                _float(record.get("local_score")) - _float(source.get("source_public_score"))
                if _float(record.get("local_score")) is not None
                and _float(source.get("source_public_score")) is not None
                else None
            ),
            "absolute_error": (
                abs(_float(record.get("local_score")) - _float(source.get("source_public_score")))
                if _float(record.get("local_score")) is not None
                and _float(source.get("source_public_score")) is not None
                else None
            ),
            "successfully_trained_model_types": trained_model_types,
            "eligible_for_selection_model_types": eligible_model_types,
            "failed_or_skipped_model_types": record.get(
                "failed_or_skipped_model_types", []
            ),
            "all_required_model_types_trained": all_required_model_types_trained,
            "selected_model": record.get("selected_model"),
            "ensemble_composition": record.get("ensemble_composition"),
            "wall_clock_seconds": _float(record.get("exec_time")),
            "execution_exc_type": record.get("execution_exc_type"),
            "artifact_dir": record.get("artifact_dir"),
            "source_code_sha256_before": source_hash_before,
            "source_code_sha256_after": source_hash_after,
            "source_code_unchanged": source_code_unchanged,
            "eligibility_evidence_verified": True,
            "compact_error_summary": record.get("recovery_reason"),
        }
    except Exception as exc:
        event = {
            **plan,
            "event_type": "completed",
            "completed_at": utc_timestamp(),
            "status": "error",
            "valid": False,
            "invalid_reason": "rerun_execution_failed",
            "execution_exc_type": type(exc).__name__,
            "compact_error_summary": str(exc)[:500],
            "successfully_trained_model_types": [],
            "eligible_for_selection_model_types": [],
            "failed_or_skipped_model_types": REQUIRED_MODEL_FAMILY,
            "all_required_model_types_trained": False,
        }
    _append_jsonl(session_dir / EXPERIMENTS_NAME, event)
    refresh_session_outputs(session_dir)
    state = _read_json(session_dir / SESSION_STATE_NAME)
    state["next_planned_experiment"] = "awaiting recorded next experimental decision"
    _write_json(session_dir / SESSION_STATE_NAME, state)
    return event


def _default_data_dir() -> Path:
    value = dotenv_values(Path(".env")).get("AIDE_PROJECT_DATA_DIR")
    if not value:
        raise ValueError("AIDE_PROJECT_DATA_DIR is missing from .env")
    return Path(str(value))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Operate a fresh AutoGluon CV-to-public-score calibration session."
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--competition", default="playground-series-s6e7")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--index", type=Path, default=Path("logs/submission_index.json"))
    parser.add_argument("--registry", type=Path, default=Path("logs/submission_registry.json"))
    parser.add_argument("--build-source-inventory", action="store_true")
    parser.add_argument("--write-data-audit", action="store_true")
    parser.add_argument("--record-decision-json")
    parser.add_argument("--reconcile-eligibility-evidence", action="store_true")
    parser.add_argument("--set-reference-profile")
    parser.add_argument("--freeze-finalist", action="append", default=[])
    parser.add_argument(
        "--set-finalization-status",
        choices=["active", "finalization_window", "completed"],
    )
    parser.add_argument("--run-one", action="store_true")
    parser.add_argument("--source-sha256")
    parser.add_argument("--profile")
    parser.add_argument("--profile-intent", default="profile calibration")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--memory-limit-gb", type=float, default=80.0)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.build_source_inventory:
        rows = build_source_inventory(
            registry_path=args.registry,
            output_path=args.session_dir / SOURCE_INVENTORY_NAME,
            competition=args.competition,
        )
        print(f"Wrote {len(rows)} source inventory rows.")
    if args.write_data_audit:
        write_data_audit(
            output_path=args.session_dir / "data_audit.json",
            data_dir=_default_data_dir(),
        )
        print("Wrote data audit.")
    if args.record_decision_json:
        record_decision(
            args.session_dir,
            decision=json.loads(args.record_decision_json),
        )
        print("Recorded decision.")
    if args.reconcile_eligibility_evidence:
        appended = reconcile_eligibility_evidence(args.session_dir)
        print(f"Appended {appended} eligibility verification event(s).")
    if args.set_reference_profile:
        set_reference_profile(
            args.session_dir,
            profile=args.set_reference_profile,
            competition=args.competition,
        )
        print(f"Set reference profile to {args.set_reference_profile}.")
    if args.freeze_finalist:
        freeze_finalists(args.session_dir, profiles=args.freeze_finalist)
        print("Froze finalist profile settings.")
    if args.set_finalization_status:
        set_finalization_status(args.session_dir, status=args.set_finalization_status)
        print(f"Set finalization status to {args.set_finalization_status}.")
    if args.run_one:
        if not args.source_sha256 or not args.profile:
            raise ValueError("--run-one requires --source-sha256 and --profile.")
        event = run_one(
            session_dir=args.session_dir,
            source_sha256=args.source_sha256,
            profile=args.profile,
            profile_intent=args.profile_intent,
            competition=args.competition,
            logs_dir=args.logs_dir,
            index_path=args.index,
            timeout=args.timeout,
            memory_limit_gb=args.memory_limit_gb,
        )
        print(json.dumps(event, indent=2, sort_keys=True))
        if event.get("status") != "ok":
            return 1
    if args.aggregate:
        refresh_session_outputs(args.session_dir)
        print("Refreshed session metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
