from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for order_index in range(index, end):
            ranks[order[order_index]] = average_rank
        index = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((value - x_mean) ** 2 for value in xs)
    y_var = sum((value - y_mean) ** 2 for value in ys)
    if x_var == 0 or y_var == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / math.sqrt(
        x_var * y_var
    )


def summarize_agreement(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    usable = [
        row
        for row in rows
        if parse_float(row.get("local_score")) is not None
        and parse_float(row.get("public_score")) is not None
    ]
    local_scores = [float(row["local_score"]) for row in usable]
    public_scores = [float(row["public_score"]) for row in usable]
    runtimes = [
        runtime
        for row in usable
        if (runtime := parse_float(row.get("exec_time"))) is not None
    ]
    if not usable:
        return {
            "n": 0,
            "pearson": None,
            "spearman": None,
            "mae": None,
            "bias": None,
            "avg_runtime_seconds": None,
        }
    return {
        "n": len(usable),
        "pearson": pearson(local_scores, public_scores),
        "spearman": pearson(rankdata(local_scores), rankdata(public_scores)),
        "mae": mean(
            abs(local_score - public_score)
            for local_score, public_score in zip(local_scores, public_scores)
        ),
        "bias": mean(
            local_score - public_score
            for local_score, public_score in zip(local_scores, public_scores)
        ),
        "avg_runtime_seconds": mean(runtimes) if runtimes else None,
    }


def usable_registry_rows(lab_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in lab_payload.get("registry", []):
        local_score = parse_float(row.get("local_score"))
        public_score = parse_float(row.get("public_score"))
        if (
            row.get("algo") != "AG"
            or str(row.get("remote_status") or "").upper() != "COMPLETE"
            or row.get("eval_metric") != "balanced_accuracy"
            or local_score is None
            or public_score is None
        ):
            continue
        rows.append(
            {
                **row,
                "local_score": local_score,
                "public_score": public_score,
            }
        )
    return rows


def _source_public_ranks(registry_rows: list[dict[str, Any]]) -> dict[str, int]:
    ranked = sorted(
        registry_rows,
        key=lambda row: (
            -float(row["public_score"]),
            -float(row["local_score"]),
            str(row.get("sha256") or ""),
        ),
    )
    return {
        str(row.get("sha256")): rank
        for rank, row in enumerate(ranked, start=1)
        if row.get("sha256")
    }


def _public_lookup(registry_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sha256"]): row for row in registry_rows if row.get("sha256")}


def _lookup_by_sha_or_prefix(
    lookup: dict[str, dict[str, Any]],
    sha256: str | None,
) -> dict[str, Any] | None:
    if not sha256:
        return None
    if sha256 in lookup:
        return lookup[sha256]
    matches = [
        row
        for known_sha, row in lookup.items()
        if known_sha.startswith(sha256) or sha256.startswith(known_sha)
    ]
    return matches[0] if len(matches) == 1 else None


def profile_eval_rows(
    *,
    registry_rows: list[dict[str, Any]],
    index_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    public_lookup = _public_lookup(registry_rows)
    public_ranks = _source_public_ranks(registry_rows)
    rows = []
    for record in index_records:
        local_score = parse_float(record.get("local_score"))
        source_sha = record.get("source_sha256")
        source_row = _lookup_by_sha_or_prefix(public_lookup, source_sha)
        if (
            record.get("kind") != "profile_eval"
            or record.get("status") != "ok"
            or record.get("eval_metric") != "balanced_accuracy"
            or local_score is None
            or source_row is None
        ):
            continue
        public_score = parse_float(source_row.get("public_score"))
        if public_score is None:
            continue
        source_sha_text = str(source_sha)
        rows.append(
            {
                "profile": record.get("profile"),
                "time_limit": record.get("time_limit"),
                "sha256": record.get("sha256"),
                "source_sha256": source_sha,
                "source_run": record.get("source_run"),
                "source_step": record.get("source_step"),
                "local_score": local_score,
                "public_score": public_score,
                "source_public_rank": public_ranks.get(source_sha_text),
                "eval_metric": record.get("eval_metric"),
                "status": record.get("status"),
                "exec_time": parse_float(record.get("exec_time")),
                "artifact_dir": record.get("artifact_dir"),
                "source_artifact_dir": source_row.get("artifact_dir"),
            }
        )
    return rows


def summarize_profiles(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("profile") or "")].append(row)
    summaries = []
    for profile, profile_rows_ in groups.items():
        summary = summarize_agreement(profile_rows_)
        summaries.append({"profile": profile, **summary})
    return sorted(
        summaries,
        key=lambda row: (
            row["n"],
            row["pearson"] if row["pearson"] is not None else float("-inf"),
            -(row["mae"] if row["mae"] is not None else float("inf")),
        ),
        reverse=True,
    )


def candidate_source_rows(registry_rows: list[dict[str, Any]], limit: int = 10) -> dict[str, Any]:
    top_public = sorted(
        registry_rows,
        key=lambda row: (row["public_score"], row["local_score"]),
        reverse=True,
    )[:limit]
    largest_gap = sorted(
        registry_rows,
        key=lambda row: abs(row["local_score"] - row["public_score"]),
        reverse=True,
    )[:limit]
    return {
        "top_public": top_public,
        "largest_abs_gap": largest_gap,
    }


def build_analysis_payload(
    *,
    lab_payload: dict[str, Any],
    index_payload: dict[str, Any],
) -> dict[str, Any]:
    registry_rows = usable_registry_rows(lab_payload)
    profile_rows_ = profile_eval_rows(
        registry_rows=registry_rows,
        index_records=list(index_payload.get("records", [])),
    )
    return {
        "baseline": summarize_agreement(registry_rows),
        "profiles": summarize_profiles(profile_rows_),
        "profile_rows": profile_rows_,
        "candidate_sources": candidate_source_rows(registry_rows),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze local CV/public-score agreement for AutoGluon artifacts."
    )
    parser.add_argument("--lab-json", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("logs/submission_index.json"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lab_payload = json.loads(args.lab_json.read_text())
    index_payload = json.loads(args.index.read_text())
    payload = build_analysis_payload(
        lab_payload=lab_payload,
        index_payload=index_payload,
    )
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
