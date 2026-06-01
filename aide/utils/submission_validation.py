from __future__ import annotations

import csv
import gzip
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


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


@dataclass(frozen=True)
class SampleSubmissionContract:
    columns: tuple[str, ...]
    id_col: str
    target_col: str
    target_is_numeric: bool
    row_count: int
    ids: frozenset[str]


def _open_csv_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _normalize_id(value: str) -> str:
    return str(value)


def _is_numeric_submission_value(value: str) -> bool:
    text = str(value).strip()
    if text == "":
        return False
    try:
        number = float(text)
    except ValueError:
        return False
    return not math.isnan(number)


def _is_categorical_submission_value(value: str) -> bool:
    return str(value).strip() != ""


def _row_value(row: list[str], index: int) -> str:
    return row[index] if index < len(row) else ""


@lru_cache(maxsize=8)
def _sample_contract_cached(
    path: str,
    size: int,
    mtime_ns: int,
) -> SampleSubmissionContract | str:
    del size, mtime_ns
    sample_path = Path(path)
    with _open_csv_text(sample_path) as f:
        reader = csv.reader(f)
        try:
            columns = tuple(next(reader))
        except StopIteration:
            columns = ()

        if len(columns) < 2:
            return "sample_submission must contain id and target columns"

        id_col = columns[0]
        target_col = columns[1]
        id_index = 0
        target_index = 1
        ids: set[str] = set()
        row_count = 0
        target_is_numeric = True
        for row in reader:
            row_count += 1
            ids.add(_normalize_id(_row_value(row, id_index)))
            if not _is_numeric_submission_value(_row_value(row, target_index)):
                target_is_numeric = False

    return SampleSubmissionContract(
        columns=columns,
        id_col=id_col,
        target_col=target_col,
        target_is_numeric=target_is_numeric,
        row_count=row_count,
        ids=frozenset(ids),
    )


def _sample_contract(path: Path) -> SampleSubmissionContract | str:
    signature = file_signature(path)
    return _sample_contract_cached(
        str(path.resolve()),
        signature["size"],
        signature["mtime_ns"],
    )


def validate_submission_file(
    submission_path: Path,
    sample_submission_path: Path,
) -> str | None:
    try:
        sample_contract = _sample_contract(sample_submission_path)
        if isinstance(sample_contract, str):
            return sample_contract

        with _open_csv_text(submission_path) as f:
            reader = csv.reader(f)
            try:
                submission_columns = tuple(next(reader))
            except StopIteration:
                submission_columns = ()

            if list(submission_columns) != list(sample_contract.columns):
                return (
                    f"columns {list(submission_columns)} != expected "
                    f"{list(sample_contract.columns)}"
                )

            id_index = submission_columns.index(sample_contract.id_col)
            target_index = submission_columns.index(sample_contract.target_col)
            seen_ids: set[str] = set()
            duplicate_count = 0
            extra_ids = 0
            invalid_prediction = False
            row_count = 0
            for row in reader:
                row_count += 1
                row_id = _normalize_id(_row_value(row, id_index))
                if row_id in seen_ids:
                    duplicate_count += 1
                else:
                    seen_ids.add(row_id)
                if row_id not in sample_contract.ids:
                    extra_ids += 1
                target_value = _row_value(row, target_index)
                if sample_contract.target_is_numeric:
                    valid_prediction = _is_numeric_submission_value(target_value)
                else:
                    valid_prediction = _is_categorical_submission_value(target_value)
                if not valid_prediction:
                    invalid_prediction = True
    except Exception as exc:
        return f"cannot read submission/sample: {type(exc).__name__}: {exc}"

    if row_count != sample_contract.row_count:
        return f"row count {row_count} != expected {sample_contract.row_count}"

    if duplicate_count:
        return f"duplicate {sample_contract.id_col} rows: {duplicate_count}"

    missing_ids = len(sample_contract.ids - seen_ids)
    if missing_ids or extra_ids:
        return f"id mismatch: missing={missing_ids}, extra={extra_ids}"

    if invalid_prediction:
        if sample_contract.target_is_numeric:
            return f"{sample_contract.target_col} contains non-numeric or null values"
        return f"{sample_contract.target_col} contains empty or null class labels"

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
