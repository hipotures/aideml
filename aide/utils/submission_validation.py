from __future__ import annotations

from pathlib import Path

import pandas as pd


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def find_sample_submission(input_dir: Path) -> Path | None:
    for name in ("sample_submission.csv.gz", "sample_submission.csv"):
        path = input_dir / name
        if path.exists():
            return path
    return None


def validate_submission_file(
    submission_path: Path,
    sample_submission_path: Path,
) -> str | None:
    try:
        submission = pd.read_csv(submission_path)
        sample = pd.read_csv(sample_submission_path)
    except Exception as exc:
        return f"cannot read submission/sample: {type(exc).__name__}: {exc}"

    if list(submission.columns) != list(sample.columns):
        return f"columns {list(submission.columns)} != expected {list(sample.columns)}"

    if len(sample.columns) < 2:
        return "sample_submission must contain id and target columns"

    id_col = sample.columns[0]
    target_col = sample.columns[1]

    if len(submission) != len(sample):
        return f"row count {len(submission)} != expected {len(sample)}"

    duplicate_count = int(submission[id_col].duplicated().sum())
    if duplicate_count:
        return f"duplicate {id_col} rows: {duplicate_count}"

    missing_ids = len(set(sample[id_col]) - set(submission[id_col]))
    extra_ids = len(set(submission[id_col]) - set(sample[id_col]))
    if missing_ids or extra_ids:
        return f"id mismatch: missing={missing_ids}, extra={extra_ids}"

    predictions = pd.to_numeric(submission[target_col], errors="coerce")
    if predictions.isna().any():
        return f"{target_col} contains non-numeric or null values"

    return None


def validate_workspace_submission(
    workspace_dir: Path,
    *,
    require_submission: bool = True,
) -> str | None:
    sample_path = find_sample_submission(workspace_dir / "input")
    if sample_path is None:
        return None

    submission_path = workspace_dir / "working" / "submission.csv"
    if not submission_path.exists():
        if require_submission:
            return "missing working/submission.csv while sample_submission exists"
        return None

    return validate_submission_file(submission_path, sample_path)
