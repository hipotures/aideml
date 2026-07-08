from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


FAST_PRESETS = {"medium", "medium_quality"}
FULL_REFERENCE_PRESETS = {"best", "best_quality", "high_quality"}


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_exec_time_seconds(value: Any) -> float | None:
    numeric = parse_float(value)
    if numeric is not None:
        return numeric
    text = str(value or "").strip().lower()
    if not text:
        return None
    multiplier = 1.0
    if text.endswith("ms"):
        multiplier = 0.001
        text = text[:-2]
    elif text.endswith("s"):
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 60.0
        text = text[:-1]
    elif text.endswith("h"):
        multiplier = 3600.0
        text = text[:-1]
    numeric = parse_float(text)
    return numeric * multiplier if numeric is not None else None


def _index_records_by_sha(index_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in index_payload.get("records", []):
        sha = record.get("sha256")
        if sha:
            records[str(sha)] = record
    return records


def source_table(
    lab_payload: dict[str, Any],
    *,
    index_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    index_by_sha = _index_records_by_sha(index_payload)
    rows: list[dict[str, Any]] = []
    for row in lab_payload.get("registry", []):
        local_score = parse_float(row.get("local_score"))
        public_score = parse_float(row.get("public_score"))
        if (
            row.get("algo") != "AG"
            or str(row.get("remote_status") or "").upper() != "COMPLETE"
            or row.get("eval_metric") != "balanced_accuracy"
            or local_score is None
            or public_score is None
            or not row.get("sha256")
        ):
            continue
        index_record = index_by_sha.get(str(row["sha256"]), {})
        exec_time_seconds = parse_exec_time_seconds(row.get("exec_time"))
        if exec_time_seconds is None:
            exec_time_seconds = parse_exec_time_seconds(index_record.get("exec_time"))
        rows.append(
            {
                "run": row.get("run"),
                "step": row.get("step"),
                "sha256": row.get("sha256"),
                "source_sha256": row.get("source_sha256"),
                "source_solution_path": row.get("source_solution_path"),
                "algo": row.get("algo"),
                "local_score": local_score,
                "public_score": public_score,
                "signed_gap": local_score - public_score,
                "absolute_gap": abs(local_score - public_score),
                "eval_metric": row.get("eval_metric"),
                "remote_status": row.get("remote_status"),
                "exec_time": row.get("exec_time"),
                "exec_time_seconds": exec_time_seconds,
                "artifact_dir": row.get("artifact_dir"),
                "has_source_rerun": row.get("has_source_rerun"),
                "manual_status": row.get("manual_status"),
                "manual_invalid_reason": row.get("manual_invalid_reason"),
                "profile": index_record.get("profile"),
                "autogluon_presets": index_record.get("autogluon_presets"),
                "time_limit": index_record.get("time_limit"),
            }
        )
    return rows


def _lookup_by_sha_or_prefix(
    lookup: dict[str, dict[str, Any]], sha256: Any
) -> dict[str, Any] | None:
    text = str(sha256 or "")
    if not text:
        return None
    if text in lookup:
        return lookup[text]
    matches = [
        row
        for known_sha, row in lookup.items()
        if known_sha.startswith(text) or text.startswith(known_sha)
    ]
    return matches[0] if len(matches) == 1 else None


def _profile_category(record: dict[str, Any], *, competition: str) -> str:
    if record.get("competition") != competition:
        return "excluded"
    presets = str(record.get("autogluon_presets") or "").strip()
    time_limit = int(parse_float(record.get("time_limit")) or 0)
    if presets in FAST_PRESETS and 0 < time_limit <= 600:
        return "fast_candidate"
    if presets in FULL_REFERENCE_PRESETS or time_limit >= 1800:
        return "full_reference"
    return "other_profile"


def profile_rows(
    *,
    source_rows: list[dict[str, Any]],
    index_payload: dict[str, Any],
    competition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_lookup = {str(row["sha256"]): row for row in source_rows}
    fast_rows: list[dict[str, Any]] = []
    full_reference_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for record in index_payload.get("records", []):
        if record.get("kind") != "profile_eval":
            continue
        source = _lookup_by_sha_or_prefix(source_lookup, record.get("source_sha256"))
        category = _profile_category(record, competition=competition)
        local_score = parse_float(record.get("local_score"))
        base = {
            "profile": record.get("profile"),
            "status": record.get("status"),
            "competition": record.get("competition"),
            "autogluon_presets": record.get("autogluon_presets"),
            "time_limit": record.get("time_limit"),
            "sha256": record.get("sha256"),
            "source_sha256": record.get("source_sha256"),
            "source_run": record.get("source_run"),
            "source_step": record.get("source_step"),
            "source_solution_path": record.get("source_solution_path"),
            "local_score": local_score,
            "public_score": source.get("public_score") if source else None,
            "source_original_local_score": source.get("local_score") if source else None,
            "eval_metric": record.get("eval_metric"),
            "exec_time": parse_exec_time_seconds(record.get("exec_time")),
            "artifact_dir": record.get("artifact_dir"),
            "timestamp": record.get("timestamp"),
        }
        if source is None or record.get("eval_metric") != "balanced_accuracy":
            excluded_rows.append({**base, "exclusion_reason": "missing_source_or_metric"})
            continue
        if category == "fast_candidate":
            fast_rows.append(base)
        elif category == "full_reference":
            full_reference_rows.append(base)
        else:
            reason = (
                "wrong_competition"
                if record.get("competition") != competition
                else "unsupported_profile"
            )
            excluded_rows.append({**base, "exclusion_reason": reason})
    return fast_rows, full_reference_rows, excluded_rows


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for order_index in range(index, end):
            ranks[order[order_index]] = rank
        index = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / math.sqrt(
        x_var * y_var
    )


def _top_k_hit_rate(rows: list[dict[str, Any]], k: int) -> float | None:
    if len(rows) < k:
        return None
    top_local = {
        row["source_sha256"]
        for row in sorted(rows, key=lambda row: row["local_score"], reverse=True)[:k]
    }
    top_public = {
        row["source_sha256"]
        for row in sorted(rows, key=lambda row: row["public_score"], reverse=True)[:k]
    }
    return len(top_local & top_public) / k


def summarize_profiles(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("profile") or "")].append(row)
    summaries = []
    for profile, profile_rows_ in groups.items():
        usable = [
            {
                **row,
                "local_score": float(row["local_score"]),
                "public_score": float(row["public_score"]),
            }
            for row in profile_rows_
            if row.get("status", "ok") == "ok"
            and parse_float(row.get("local_score")) is not None
            and parse_float(row.get("public_score")) is not None
        ]
        failures = [row for row in profile_rows_ if row.get("status", "ok") != "ok"]
        runtimes = [
            runtime
            for row in usable
            if (runtime := parse_exec_time_seconds(row.get("exec_time"))) is not None
        ]
        errors = [
            {
                **row,
                "signed_error": row["local_score"] - row["public_score"],
                "absolute_error": abs(row["local_score"] - row["public_score"]),
            }
            for row in usable
        ]
        local_scores = [row["local_score"] for row in usable]
        public_scores = [row["public_score"] for row in usable]
        absolute_errors = [row["absolute_error"] for row in errors]
        signed_errors = [row["signed_error"] for row in errors]
        bias = mean(signed_errors) if signed_errors else None
        bias_corrected_errors = (
            [abs(error - bias) for error in signed_errors] if bias is not None else []
        )
        loo_bias_corrected_errors = []
        if len(signed_errors) > 1:
            error_sum = sum(signed_errors)
            for error in signed_errors:
                loo_bias = (error_sum - error) / (len(signed_errors) - 1)
                loo_bias_corrected_errors.append(abs(error - loo_bias))
        summaries.append(
            {
                "profile": profile,
                "attempted": len(profile_rows_),
                "n": len(usable),
                "source_sha256s": [str(row["source_sha256"]) for row in usable],
                "pearson": pearson(local_scores, public_scores),
                "spearman": pearson(rankdata(local_scores), rankdata(public_scores)),
                "top_2_hit_rate": _top_k_hit_rate(usable, 2),
                "top_3_hit_rate": _top_k_hit_rate(usable, 3),
                "mae": mean(absolute_errors) if absolute_errors else None,
                "bias": bias,
                "bias_corrected_mae": mean(bias_corrected_errors)
                if bias_corrected_errors
                else None,
                "loo_bias_corrected_mae": mean(loo_bias_corrected_errors)
                if loo_bias_corrected_errors
                else None,
                "median_absolute_error": median(absolute_errors)
                if absolute_errors
                else None,
                "worst_over_optimistic": sorted(
                    errors, key=lambda row: row["signed_error"], reverse=True
                )[:3],
                "worst_under_optimistic": sorted(
                    errors, key=lambda row: row["signed_error"]
                )[:3],
                "avg_runtime_seconds": mean(runtimes) if runtimes else None,
                "max_runtime_seconds": max(runtimes) if runtimes else None,
                "failure_rate": len(failures) / len(profile_rows_)
                if profile_rows_
                else None,
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            row["n"],
            row["spearman"] if row["spearman"] is not None else float("-inf"),
            -(row["mae"] if row["mae"] is not None else float("inf")),
        ),
        reverse=True,
    )


def select_representative_sources(
    source_rows: list[dict[str, Any]], limit: int = 12
) -> list[dict[str, Any]]:
    buckets = [
        sorted(source_rows, key=lambda row: (-row["public_score"], -row["local_score"])),
        sorted(source_rows, key=lambda row: (-row["absolute_gap"], -row["local_score"])),
        sorted(source_rows, key=lambda row: (-row["local_score"], row["public_score"])),
    ]
    by_public = sorted(source_rows, key=lambda row: row["public_score"])
    mid = len(by_public) // 2
    buckets.append(by_public[max(0, mid - limit // 2) : mid + limit // 2])

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(selected) < limit and any(buckets):
        for bucket in buckets:
            while bucket:
                row = bucket.pop(0)
                sha = str(row["sha256"])
                if sha in seen:
                    continue
                selected.append(row)
                seen.add(sha)
                break
            if len(selected) >= limit:
                break
    return selected


def _parse_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y%m%dT%H%M%S"):
        try:
            parsed = dt.datetime.strptime(value, fmt)
            return parsed.replace(tzinfo=None)
        except ValueError:
            pass
    return None


def build_analysis_payload(
    *,
    lab_payload: dict[str, Any],
    index_payload: dict[str, Any],
    competition: str,
    task_start_time: str | None = None,
) -> dict[str, Any]:
    sources = source_table(lab_payload, index_payload=index_payload)
    fast_rows, full_reference_rows, excluded_rows = profile_rows(
        source_rows=sources,
        index_payload=index_payload,
        competition=competition,
    )
    task_start = _parse_timestamp(task_start_time)
    current_task_rows = []
    historical_fast_rows = []
    for row in fast_rows:
        row_time = _parse_timestamp(row.get("timestamp"))
        if task_start is not None and row_time is not None and row_time >= task_start:
            current_task_rows.append(row)
        else:
            historical_fast_rows.append(row)
    return {
        "competition": competition,
        "source_table": sources,
        "representative_source_set": select_representative_sources(sources),
        "historical_submitted_fast_screening_artifacts": sources,
        "new_fast_candidate_rows": fast_rows,
        "current_task_fast_candidate_rows": current_task_rows,
        "historical_fast_candidate_rows": historical_fast_rows,
        "historical_full_reference_rows": full_reference_rows,
        "excluded_profile_rows": excluded_rows,
        "new_fast_candidate_profiles": summarize_profiles(fast_rows),
        "current_task_fast_candidate_profiles": summarize_profiles(current_task_rows),
        "historical_full_reference_profiles": summarize_profiles(full_reference_rows),
    }


SUMMARY_COLUMNS = [
    "profile",
    "n",
    "pearson",
    "spearman",
    "attempted",
    "top_2_hit_rate",
    "top_3_hit_rate",
    "mae",
    "bias",
    "bias_corrected_mae",
    "loo_bias_corrected_mae",
    "median_absolute_error",
    "avg_runtime_seconds",
    "max_runtime_seconds",
    "failure_rate",
    "source_sha256s",
]


SOURCE_COLUMNS = [
    "run",
    "step",
    "sha256",
    "source_sha256",
    "source_solution_path",
    "algo",
    "local_score",
    "public_score",
    "signed_gap",
    "absolute_gap",
    "eval_metric",
    "remote_status",
    "exec_time",
    "exec_time_seconds",
    "artifact_dir",
    "has_source_rerun",
    "manual_status",
    "manual_invalid_reason",
    "profile",
    "autogluon_presets",
    "time_limit",
]


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summaries:
            writer.writerow(
                {
                    key: (
                        ";".join(row[key])
                        if key == "source_sha256s" and isinstance(row.get(key), list)
                        else row.get(key)
                    )
                    for key in SUMMARY_COLUMNS
                }
            )


def write_sources_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in SOURCE_COLUMNS})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze fast medium AutoGluon profile alignment with public scores."
    )
    parser.add_argument("--lab-json", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("logs/submission_index.json"))
    parser.add_argument("--competition", default="playground-series-s6e7")
    parser.add_argument("--task-start-time")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--sources-csv", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_analysis_payload(
        lab_payload=json.loads(args.lab_json.read_text()),
        index_payload=json.loads(args.index.read_text()),
        competition=args.competition,
        task_start_time=args.task_start_time,
    )
    if args.output_json is None:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_csv is not None:
        write_summary_csv(args.output_csv, payload["new_fast_candidate_profiles"])
    if args.sources_csv is not None:
        write_sources_csv(args.sources_csv, payload["source_table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
