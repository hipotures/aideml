from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich import box
from sklearn.metrics import roc_auc_score

from aide.utils.submission_validation import validate_submission_file
from scripts import kaggle_submission_lab as lab
from scripts import smart_kaggle_submit as smart


DEFAULT_TASK = "playground-series-s6e5"
DEFAULT_DATA_DIR = Path("aide/example_tasks") / DEFAULT_TASK
DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_INDEX_PATH = Path("logs/submission_index.json")
DEFAULT_REGISTRY_PATH = smart.DEFAULT_REGISTRY
DEFAULT_OUTPUT_RUN = "blended"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataframe_csv(frame: pd.DataFrame) -> str:
    with tempfile.NamedTemporaryFile("w+b", suffix=".csv") as handle:
        frame.to_csv(handle.name, index=False)
        return sha256_file(Path(handle.name))


def existing_submission_sha256(index: dict[str, Any]) -> set[str]:
    return {
        str(record.get("sha256") or "")
        for record in index.get("records", [])
        if record.get("sha256")
    }


def file_payload(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    stat = path.stat()
    payload = {
        "path": str(path.relative_to(relative_to)) if relative_to else str(path),
        "sha256": sha256_file(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return payload


def stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def parse_csv_floats(value: str | None) -> list[float] | None:
    if value is None:
        return None
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not weights:
        raise ValueError("--weights must contain at least one number")
    return weights


def normalize_weights(weights: Iterable[float]) -> list[float]:
    raw = [float(weight) for weight in weights]
    if any(weight < 0 for weight in raw):
        raise ValueError("Blend weights must be non-negative")
    total = sum(raw)
    if total <= 0:
        raise ValueError("Blend weights must sum to a positive value")
    return [weight / total for weight in raw]


def compute_weights(
    scores: list[float],
    *,
    mode: str,
    manual: list[float] | None,
    power: float,
    temp: float,
    eps: float,
    lower_is_better: bool,
) -> list[float]:
    if mode == "manual":
        if manual is None or len(manual) != len(scores):
            raise ValueError("--weights must match the number of selected components")
        return normalize_weights(manual)

    if not scores:
        raise ValueError("No scores available for weight computation")

    if mode == "uniform":
        raw = [1.0] * len(scores)
    elif mode in {"local", "public"}:
        if lower_is_better:
            raw = [1.0 / max(float(score), eps) for score in scores]
        else:
            raw = [float(score) for score in scores]
    elif mode == "rank":
        raw = [float(len(scores) - idx) for idx in range(len(scores))]
    elif mode == "power":
        if lower_is_better:
            raw = [(1.0 / max(float(score), eps)) ** power for score in scores]
        else:
            raw = [float(score) ** power for score in scores]
    elif mode == "softmax":
        if temp <= 0:
            raise ValueError("--temp must be positive")
        values = [-float(score) if lower_is_better else float(score) for score in scores]
        max_value = max(values)
        raw = [math.exp((value - max_value) / temp) for value in values]
    elif mode == "offset":
        if lower_is_better:
            max_score = max(scores)
            raw = [max(max_score - float(score) + eps, 0.0) for score in scores]
        else:
            min_score = min(scores)
            raw = [max(float(score) - min_score + eps, 0.0) for score in scores]
    else:
        raise ValueError(f"Unknown weighting mode: {mode}")

    return normalize_weights(raw)


def transform_predictions(frame: pd.DataFrame, column: str, mode: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="raise").astype(float)
    if mode == "raw":
        return values
    if mode == "rank":
        return values.rank(method="average", pct=True)
    if mode == "logit":
        clipped = values.clip(1e-9, 1.0 - 1e-9)
        return np.log(clipped / (1.0 - clipped))
    raise ValueError(f"Unknown blend mode: {mode}")


def finish_blend(values: pd.Series | np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if mode == "logit":
        return 1.0 / (1.0 + np.exp(-arr))
    return arr


def _read_prediction_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer")


def blend_prediction_files(
    paths: list[Path],
    *,
    weights: list[float],
    id_column: str,
    prediction_column: str,
    mode: str,
) -> pd.DataFrame:
    base = _read_prediction_file(paths[0])
    if id_column not in base.columns or prediction_column not in base.columns:
        raise ValueError(f"{paths[0]} must contain {id_column!r} and {prediction_column!r}")
    base_ids = base[id_column].copy()
    blended = np.zeros(len(base), dtype=float)

    for path, weight in zip(paths, weights, strict=True):
        frame = _read_prediction_file(path)
        if id_column not in frame.columns or prediction_column not in frame.columns:
            raise ValueError(f"{path} must contain {id_column!r} and {prediction_column!r}")
        if frame[id_column].equals(base_ids):
            aligned = frame
        else:
            aligned = frame.set_index(id_column).reindex(base_ids).reset_index()
            if aligned[prediction_column].isna().any():
                raise ValueError(f"Cannot align {path} by {id_column!r}")
        blended += weight * transform_predictions(aligned, prediction_column, mode)

    out = pd.DataFrame(
        {
            id_column: base_ids,
            prediction_column: finish_blend(blended, mode),
        }
    )
    return out


def blend_oof_files(
    paths: list[Path],
    *,
    weights: list[float],
    mode: str,
) -> tuple[pd.DataFrame, float]:
    base = _read_prediction_file(paths[0])
    required = {"row", "target", "prediction"}
    if not required.issubset(base.columns):
        raise ValueError(f"{paths[0]} must contain {sorted(required)}")
    base_rows = base["row"].copy()
    base_target = base["target"].copy()
    base_target_values = pd.to_numeric(base_target, errors="raise").to_numpy()
    blended = np.zeros(len(base), dtype=float)

    for path, weight in zip(paths, weights, strict=True):
        frame = _read_prediction_file(path)
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} must contain {sorted(required)}")
        if frame["row"].equals(base_rows):
            aligned = frame
        else:
            aligned = frame.set_index("row").reindex(base_rows).reset_index()
            if aligned["prediction"].isna().any() or aligned["target"].isna().any():
                raise ValueError(f"Cannot align OOF file {path} by 'row'")
        aligned_target_values = pd.to_numeric(aligned["target"], errors="raise").to_numpy()
        if not np.array_equal(aligned_target_values, base_target_values):
            raise ValueError(f"OOF target mismatch in {path}")
        blended += weight * transform_predictions(aligned, "prediction", mode)

    prediction = finish_blend(blended, mode)
    auc = float(roc_auc_score(base_target, prediction))
    out = pd.DataFrame(
        {
            "row": base_rows,
            "target": base_target,
            "prediction": prediction,
        }
    )
    return out, auc


def is_blend_record(record: dict[str, Any]) -> bool:
    text = " ".join(
        str(record.get(key) or "")
        for key in ("hypothesis_id", "timestamp", "artifact_dir")
    ).lower()
    return any(token in text for token in ("blend", "stack", "ensemble", "manual"))


def public_score_for_record(
    record: dict[str, Any],
    registry: smart.SubmissionRegistry,
) -> float | None:
    sha = str(record.get("sha256") or "")
    for entry in registry.entries:
        if entry.get("competition") != record.get("competition"):
            continue
        if smart._sha256_matches(entry.get("sha256"), sha):
            parsed = smart._parse_public_score(entry.get("public_score"))
            if parsed is not None:
                return float(parsed)
    return None


def record_label(record: dict[str, Any]) -> str:
    hypothesis = record.get("hypothesis_id")
    if hypothesis:
        return str(hypothesis)
    step = record.get("step")
    if step is not None:
        return f"step{step}"
    return str(record.get("sha256") or "component")[:10]


def make_unique_labels(records: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, int] = {}
    labels = []
    for record in records:
        base = re.sub(r"[^A-Za-z0-9_.-]+", "-", record_label(record)).strip("-")
        if not base:
            base = str(record.get("sha256") or "component")[:10]
        count = seen.get(base, 0)
        seen[base] = count + 1
        labels.append(base if count == 0 else f"{base}-{count + 1}")
    return labels


def make_indexed_labels(
    records: list[dict[str, Any]],
    selected_indices: set[int],
) -> list[str]:
    labels = make_unique_labels(records)
    if len(labels) != len(selected_indices):
        return labels
    return [
        f"#{idx}:{label}"
        for idx, label in zip(sorted(selected_indices), labels, strict=True)
    ]


def _record_node_key(record: dict[str, Any]) -> tuple[str, str] | None:
    run = str(record.get("run") or "")
    node_id = str(record.get("node_id") or "")
    if not run or not node_id:
        return None
    return run, node_id


def _record_score(record: dict[str, Any]) -> float | None:
    score = record.get("local_score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def _score_within_epsilon(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    epsilon: float,
) -> bool:
    left_score = _record_score(left)
    right_score = _record_score(right)
    if left_score is None or right_score is None:
        return False
    return abs(left_score - right_score) <= epsilon


def record_oof_path(record: dict[str, Any]) -> Path:
    return Path(str(record.get("artifact_dir") or "")) / "oof_predictions.csv.gz"


def record_test_or_submission_path(record: dict[str, Any]) -> Path:
    artifact_dir = Path(str(record.get("artifact_dir") or ""))
    test_path = artifact_dir / "test_predictions.csv.gz"
    if test_path.exists():
        return test_path
    return artifact_dir / "submission.csv"


def record_has_prediction_inputs(record: dict[str, Any]) -> bool:
    return record_test_or_submission_path(record).exists()


def hide_near_duplicate_family_descendants(
    records: list[dict[str, Any]],
    *,
    epsilon: float,
    max_depth: int,
) -> list[dict[str, Any]]:
    if epsilon < 0 or max_depth <= 0:
        return records

    by_node: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = _record_node_key(record)
        if key is not None:
            by_node[key] = record

    visible = []
    for record in records:
        run = str(record.get("run") or "")
        parent_id = str(record.get("parent_node_id") or "")
        hide = False
        for _depth in range(max_depth):
            if not parent_id:
                break
            ancestor = by_node.get((run, parent_id))
            if ancestor is not None and _score_within_epsilon(
                record,
                ancestor,
                epsilon=epsilon,
            ):
                hide = True
                break
            parent_id = str((ancestor or {}).get("parent_node_id") or "")
        if not hide:
            visible.append(record)
    return visible


def select_records(
    *,
    index: dict[str, Any],
    registry: smart.SubmissionRegistry,
    run_filters: list[str],
    sha_filters: list[str],
    top_n: int | None,
    include_blends: bool,
    weighting: str,
    competition: str,
    family_dedupe: bool,
    family_dedupe_epsilon: float,
    family_dedupe_depth: int,
) -> list[dict[str, Any]]:
    records = lab.filter_records_by_run(list(index.get("records", [])), run_filters)
    ready = [
        record
        for record in records
        if lab._record_is_submit_ready(record)
        and record_has_prediction_inputs(record)
    ]
    if sha_filters:
        selected = lab.filter_records_by_sha256(ready, sha_filters)
    else:
        if top_n is None:
            raise ValueError("Provide --sha256 or --top-n")
        selected_pool = ready if include_blends else [
            record for record in ready if not is_blend_record(record)
        ]
        if family_dedupe:
            selected_pool = hide_near_duplicate_family_descendants(
                selected_pool,
                epsilon=family_dedupe_epsilon,
                max_depth=family_dedupe_depth,
            )
        if weighting == "public":
            selected_pool = [
                record for record in selected_pool
                if public_score_for_record(record, registry) is not None
            ]
            selected = sorted(
                selected_pool,
                key=lambda record: (
                    public_score_for_record(record, registry) or float("-inf"),
                    str(record.get("timestamp") or ""),
                ),
                reverse=True,
            )[:top_n]
        else:
            selected = lab.deduplicate_records_by_sha256(selected_pool)[:top_n]

    selected = [
        record
        for record in selected
        if str(record.get("competition") or competition) == competition
    ]
    if len(selected) < 2:
        raise ValueError("Need at least two components to blend")
    return selected


def next_step_for_run(index: dict[str, Any], output_run: str) -> int:
    steps = []
    for record in index.get("records", []):
        if str(record.get("run") or "") != output_run:
            continue
        try:
            step = int(record.get("step"))
        except (TypeError, ValueError):
            continue
        steps.append(step)
    return max(steps, default=0) + 1


def sample_submission_path(data_dir: Path) -> Path:
    for name in ("sample_submission.csv.gz", "sample_submission.csv"):
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing sample_submission.csv[.gz] in {data_dir}")


def write_solution_py(
    artifact_dir: Path,
    *,
    labels: list[str],
    weights: list[float],
    score: float | None,
    mode: str,
    submission_only: bool,
) -> None:
    payload = {
        "mode": mode,
        "weights": dict(zip(labels, weights, strict=True)),
        "cv_score": score,
    }
    lines = [
        "# Manual prediction blend artifact generated by scripts/blend_submissions.py.",
        f"BLEND_MODE = {mode!r}",
        f"BLEND_WEIGHTS = {payload['weights']!r}",
        f"SUBMISSION_ONLY = {submission_only!r}",
        f"CV_SCORE = {score!r}",
        "",
    ]
    artifact_dir.joinpath("solution.py").write_text("\n".join(lines))


def write_manifest(
    artifact_dir: Path,
    *,
    competition: str,
    output_run: str,
    timestamp: str,
    created_at: dt.datetime,
    step: int,
    node_id: str,
    labels: list[str],
    selected: list[dict[str, Any]],
    weights: list[float],
    score: float | None,
    mode: str,
    label: str,
    sample_path: Path,
    validation_error: str | None,
    submission_only: bool,
) -> dict[str, Any]:
    first = selected[0]
    component_artifacts = {
        name: str(Path(str(record.get("artifact_dir"))))
        for name, record in zip(labels, selected, strict=True)
    }
    component_sha256 = {
        name: str(record.get("sha256") or "")
        for name, record in zip(labels, selected, strict=True)
    }
    source_payload = {
        "source_run": first.get("run"),
        "source_node_id": first.get("node_id"),
        "source_step": first.get("step"),
        "source_timestamp": first.get("timestamp"),
        "source_sha256": first.get("sha256"),
    }
    submission_signature = stat_signature(artifact_dir / "submission.csv")
    sample_signature = stat_signature(sample_path)
    validation_payload = {
        "status": "ok" if validation_error is None else "error",
        "submission_signature": submission_signature,
        "sample_signature": sample_signature,
    }
    if validation_error:
        validation_payload["error"] = validation_error

    manifest = {
        "schema_version": 1,
        "kind": "source_node",
        "competition": competition,
        "run": output_run,
        "timestamp": timestamp,
        "artifact_dir": str(artifact_dir),
        "created_at": created_at.isoformat(),
        "status": "ok" if validation_error is None else "error",
        "is_buggy": validation_error is not None,
        "local_score": score,
        "metric_maximize": None if submission_only else True,
        "hypothesis_id": label,
        "sha256": sha256_file(artifact_dir / "submission.csv"),
        "node": {
            "id": node_id,
            "parent_id": first.get("node_id"),
            "step": step,
            "ctime": created_at.timestamp(),
            "status": "ok" if validation_error is None else "error",
            "is_buggy": validation_error is not None,
            "metric": {"value": score, "maximize": None if submission_only else True},
            "analysis": (
                f"Manual {mode} blend computed from existing predictions."
                if submission_only
                else (
                    f"Manual {mode} blend computed from existing OOF/test predictions. "
                    f"CV ROC AUC={score:.10f}."
                )
            ),
            "plan": (
                f"Manual fixed-weight submission blend {label}: "
                f"{dict(zip(labels, weights, strict=True))}"
            ),
            "origin": "manual_blend",
            "hypothesis_id": label,
            "validity_warning": validation_error,
            "submission_validation": validation_payload,
        },
        "execution": {"exec_time": 0.0, "exc_type": None, "exc_info": None, "exc_stack": None},
        "run_stats": {
            "metric_name": "roc_auc",
            "cv_score": score,
            "submission_only": submission_only,
            "blend_mode": mode,
            "blend_weights": dict(zip(labels, weights, strict=True)),
            "components": labels,
            "component_artifacts": component_artifacts,
            "component_sha256": component_sha256,
            "oof_path": None if submission_only else "working/oof_predictions.csv.gz",
            "test_predictions_path": "working/test_predictions.csv.gz",
            "submission_path": "working/submission.csv",
        },
        "files": {
            "submission": file_payload(artifact_dir / "submission.csv", relative_to=artifact_dir),
            "solution": file_payload(artifact_dir / "solution.py", relative_to=artifact_dir),
            "oof_predictions": (
                None
                if submission_only
                else file_payload(
                    artifact_dir / "oof_predictions.csv.gz",
                    relative_to=artifact_dir,
                )
            ),
            "test_predictions": file_payload(
                artifact_dir / "test_predictions.csv.gz",
                relative_to=artifact_dir,
            ),
            "validation_predictions": None,
            "model_predictions": [],
            "error": None if validation_error is None else validation_error,
        },
        "submission_validation": validation_payload,
        "source": source_payload,
        "autogluon": {
            "profile": None,
            "presets": None,
            "included_model_types": None,
            "time_limit": None,
            "process_timeout": None,
            "use_gpu": None,
            "resolved_settings": {},
        },
        "profile": None,
        "autogluon_presets": None,
        "included_model_types": None,
        "time_limit": None,
    }
    artifact_dir.joinpath(lab.RESULT_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def write_upload_copy(
    artifact_dir: Path,
    *,
    timestamp: str,
    step: int,
    node_id: str,
    sha256: str,
    score: float | None,
) -> Path:
    score_part = "cv-na" if score is None else f"cv-{score:.5f}"
    upload_name = (
        f"sub_{timestamp}_step-{step}_node-{node_id[:8]}_"
        f"sha-{sha256[:10]}_{score_part}.csv"
    )
    upload_path = artifact_dir / upload_name
    shutil.copy2(artifact_dir / "submission.csv", upload_path)
    return upload_path


def render_selection_table(
    console: Console,
    records: list[dict[str, Any]],
    *,
    labels: list[str],
    weights: list[float],
    public_scores: list[float | None],
) -> None:
    table = Table(title="Blend components", padding=(0, 1))
    table.add_column("#", justify="right")
    table.add_column("label")
    table.add_column("cv", justify="right")
    table.add_column("public", justify="right")
    table.add_column("weight", justify="right")
    table.add_column("run")
    table.add_column("hyp")
    table.add_column("step", justify="right")
    table.add_column("sha")
    for idx, (record, label, weight, public) in enumerate(
        zip(records, labels, weights, public_scores, strict=True),
        start=1,
    ):
        table.add_row(
            str(idx),
            label,
            "-" if record.get("local_score") is None else f"{float(record['local_score']):.5f}",
            "-" if public is None else f"{public:.5f}",
            f"{weight:.6f}",
            str(record.get("run") or "-"),
            str(record.get("hypothesis_id") or "-"),
            str(record.get("step") if record.get("step") is not None else "-"),
            str(record.get("sha256") or "")[:10],
        )
    console.print(table)


def create_blend_artifact(
    *,
    args: argparse.Namespace,
    console: Console,
    selected: list[dict[str, Any]],
    labels: list[str],
    weights: list[float],
    output_run: str,
    label: str | None,
    mode: str,
    dry_run: bool,
    index: dict[str, Any],
    allow_submission_only: bool | None = None,
    known_submission_sha256: set[str] | None = None,
    step_override: int | None = None,
    refresh_index_after_write: bool = True,
) -> tuple[float, str | None, bool]:
    public_scores = [
        public_score_for_record(record, smart.SubmissionRegistry.load(args.registry))
        for record in selected
    ]
    render_selection_table(
        console,
        selected,
        labels=labels,
        weights=weights,
        public_scores=public_scores,
    )

    artifact_dirs = [Path(str(record["artifact_dir"])) for record in selected]
    oof_paths = [path / "oof_predictions.csv.gz" for path in artifact_dirs]
    missing_oof = [str(path) for path in oof_paths if not path.exists()]
    submission_only = False
    if missing_oof:
        console.print("Missing OOF predictions:")
        for path in missing_oof:
            console.print(f"  {path}")
        if allow_submission_only is None:
            allow_submission_only = Confirm.ask(
                "Create submission-only blend without CV?",
                default=False,
            )
        if not allow_submission_only:
            raise FileNotFoundError("Missing OOF predictions: " + ", ".join(missing_oof))
        submission_only = True

    test_paths = [
        path / "test_predictions.csv.gz"
        if (path / "test_predictions.csv.gz").exists()
        else path / "submission.csv"
        for path in artifact_dirs
    ]

    oof = None
    cv_score: float | None = None
    if not submission_only:
        oof, cv_score = blend_oof_files(oof_paths, weights=weights, mode=mode)
    test_predictions = blend_prediction_files(
        test_paths,
        weights=weights,
        id_column="id",
        prediction_column="PitNextLap",
        mode=mode,
    )
    submission = test_predictions[["id", "PitNextLap"]].copy()
    submission_sha256 = sha256_dataframe_csv(submission)

    if cv_score is None:
        console.print("Blend CV ROC AUC: n/a (submission-only)")
    else:
        console.print(f"Blend CV ROC AUC: {cv_score:.10f}")
    duplicate_pool = known_submission_sha256 or existing_submission_sha256(index)
    if submission_sha256 in duplicate_pool:
        console.print(f"SKIP duplicate submission sha: {submission_sha256[:10]}")
        return cv_score or float("nan"), None, True
    if dry_run:
        console.print("Dry run: artifact not written.")
        return cv_score or float("nan"), None, False

    created_at = dt.datetime.now(dt.timezone.utc)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label or "-".join(labels)).strip("-")
    if not safe_label:
        safe_label = "blend"
    timestamp = created_at.strftime("%Y%m%dT%H%M%S") + f"-manual-{safe_label}"
    artifact_dir = args.logs_dir / output_run / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=False)

    if oof is not None:
        oof.to_csv(artifact_dir / "oof_predictions.csv.gz", index=False)
    test_predictions.to_csv(artifact_dir / "test_predictions.csv.gz", index=False)
    submission.to_csv(artifact_dir / "submission.csv", index=False)
    write_solution_py(
        artifact_dir,
        labels=labels,
        weights=weights,
        score=cv_score,
        mode=mode,
        submission_only=submission_only,
    )

    if step_override is not None:
        step = step_override
    else:
        step = args.step if args.step is not None else next_step_for_run(index, output_run)
    node_id = hashlib.md5(f"{timestamp}:{safe_label}:{cv_score}".encode()).hexdigest()
    sample_path = sample_submission_path(args.data_dir)
    validation_error = validate_submission_file(
        artifact_dir / "submission.csv",
        sample_path,
    )
    manifest = write_manifest(
        artifact_dir,
        competition=args.competition,
        output_run=output_run,
        timestamp=timestamp,
        created_at=created_at,
        step=step,
        node_id=node_id,
        labels=labels,
        selected=selected,
        weights=weights,
        score=cv_score,
        mode=mode,
        label=safe_label,
        sample_path=sample_path,
        validation_error=validation_error,
        submission_only=submission_only,
    )
    upload_path = write_upload_copy(
        artifact_dir,
        timestamp=timestamp,
        step=step,
        node_id=node_id,
        sha256=manifest["sha256"],
        score=cv_score,
    )
    if refresh_index_after_write:
        lab.refresh_index(
            logs_dir=args.logs_dir,
            index_path=args.index,
            competition=args.competition,
            runs=[output_run],
            reindex=True,
        )

    console.print(f"Wrote artifact: {artifact_dir}")
    console.print(f"Upload copy: {upload_path.name}")
    console.print(f"sha256: {manifest['sha256'][:10]}")
    if validation_error:
        console.print(f"Validation error: {validation_error}")
    if known_submission_sha256 is not None:
        known_submission_sha256.add(str(manifest["sha256"]))
    return cv_score or float("nan"), str(artifact_dir), False


def _interactive_candidate_sort_key(
    record: dict[str, Any],
    registry: smart.SubmissionRegistry,
) -> tuple[bool, float, float, str]:
    public = public_score_for_record(record, registry)
    local = record.get("local_score")
    local_score = float(local) if local is not None else float("-inf")
    if public is not None:
        return True, public, local_score, str(record.get("timestamp") or "")
    return False, float("-inf"), local_score, str(record.get("timestamp") or "")


def interactive_candidate_records(
    *,
    index: dict[str, Any],
    registry: smart.SubmissionRegistry,
    run_filters: list[str],
    competition: str,
    include_blends: bool,
    family_dedupe: bool,
    family_dedupe_epsilon: float,
    family_dedupe_depth: int,
) -> list[dict[str, Any]]:
    records = lab.filter_records_by_run(list(index.get("records", [])), run_filters)
    ready = [
        record
        for record in records
        if lab._record_is_submit_ready(record)
        and str(record.get("competition") or competition) == competition
        and record_has_prediction_inputs(record)
    ]
    if not include_blends:
        ready = [record for record in ready if not is_blend_record(record)]
    if family_dedupe:
        ready = hide_near_duplicate_family_descendants(
            ready,
            epsilon=family_dedupe_epsilon,
            max_depth=family_dedupe_depth,
        )
    return sorted(
        lab.deduplicate_records_by_sha256(ready),
        key=lambda record: _interactive_candidate_sort_key(record, registry),
        reverse=True,
    )


def render_interactive_candidate_table(
    console: Console,
    records: list[dict[str, Any]],
    *,
    registry: smart.SubmissionRegistry,
    limit: int,
    selected_indices: set[int] | None = None,
) -> int:
    table = Table(title="Blend source candidates", padding=(0, 1))
    headers = ["#", "cv", "public", "oof", "run", "hyp", "step", "sha", "artifact"]
    rows: list[list[str]] = []
    for idx, record in enumerate(records[:limit], start=1):
        public = public_score_for_record(record, registry)
        has_oof = record_oof_path(record).exists()
        rows.append(
            [
                str(idx),
                "-" if record.get("local_score") is None else f"{float(record['local_score']):.5f}",
                "-" if public is None else f"{public:.5f}",
                "✓" if has_oof else "x",
                str(record.get("run") or "-"),
                str(record.get("hypothesis_id") or "-"),
                str(record.get("step") if record.get("step") is not None else "-"),
                str(record.get("sha256") or "")[:10],
                Path(str(record.get("artifact_dir") or "-")).name,
            ]
        )
    widths = [
        max([len(header)] + [len(row[index]) for row in rows])
        for index, header in enumerate(headers)
    ]
    table_width = min(console.width, sum(widths) + (2 * len(headers)) + len(headers) + 1)
    for header in headers:
        table.add_column(
            header,
            justify="right" if header in {"#", "cv", "public", "step"} else "center" if header == "oof" else "left",
        )
    selected_indices = selected_indices or set()
    for row in rows:
        row_index = int(row[0])
        style = "reverse bold" if row_index in selected_indices else None
        table.add_row(*row, style=style)
    console.print(table)
    return table_width


def parse_interactive_selection(
    value: str,
    *,
    displayed_records: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tokens = [token for token in re.split(r"[\s,]+", value.strip()) if token]
    if len(tokens) < 2:
        raise ValueError("Select at least two candidates")

    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for token in tokens:
        record: dict[str, Any]
        if token.isdigit():
            idx = int(token)
            if idx < 1 or idx > len(displayed_records):
                raise ValueError(f"Candidate index out of range: {idx}")
            record = displayed_records[idx - 1]
        else:
            matches = [
                candidate
                for candidate in all_records
                if str(candidate.get("sha256") or "").lower().startswith(token.lower())
            ]
            matched_hashes = {str(match.get("sha256") or "") for match in matches}
            if not matches:
                raise ValueError(f"No candidate matches sha prefix: {token}")
            if len(matched_hashes) > 1:
                preview = ", ".join(sorted(value[:10] for value in matched_hashes))
                raise ValueError(f"Ambiguous sha prefix {token}; matches: {preview}")
            record = matches[0]
        sha = str(record.get("sha256") or "")
        if sha and sha not in seen_hashes:
            selected.append(record)
            seen_hashes.add(sha)

    if len(selected) < 2:
        raise ValueError("Select at least two distinct candidates")
    return selected


def configure_interactive(args: argparse.Namespace, console: Console) -> None:
    run_filters = lab.parse_run_filters(args.run)
    index = lab.refresh_index(
        logs_dir=args.logs_dir,
        index_path=args.index,
        competition=args.competition,
        runs=run_filters or None,
        reindex=args.reindex,
    )
    registry = smart.SubmissionRegistry.load(args.registry)
    candidates = interactive_candidate_records(
        index=index,
        registry=registry,
        run_filters=run_filters,
        competition=args.competition,
        include_blends=args.include_blends,
        family_dedupe=not args.no_family_dedupe,
        family_dedupe_epsilon=args.family_dedupe_epsilon,
        family_dedupe_depth=args.family_dedupe_depth,
    )
    if not candidates:
        raise ValueError("No submit-ready candidates found")

    display_limit = min(40, len(candidates))
    render_interactive_candidate_table(
        console,
        candidates,
        registry=registry,
        limit=display_limit,
    )

    while True:
        selected_text = Prompt.ask(
            "Select components by # or sha prefix, separated with commas"
        )
        try:
            selected = parse_interactive_selection(
                selected_text,
                displayed_records=candidates[:display_limit],
                all_records=candidates,
            )
            break
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")

    labels = make_unique_labels(selected)
    if args.output_run is None:
        args.output_run = DEFAULT_OUTPUT_RUN

    weights_text = Prompt.ask(
        "Weights, comma-separated; empty means uniform",
        default="",
        show_default=False,
    ).strip()
    if weights_text:
        args.weighting = "manual"
        args.weights = weights_text
    else:
        args.weighting = "uniform"
        args.weights = None

    args.mode = Prompt.ask(
        "Blend mode",
        choices=["raw", "rank", "logit"],
        default=args.mode,
    )
    default_label = "blend-" + "-".join(label[:8] for label in labels)
    args.label = Prompt.ask("Artifact label", default=args.label or default_label)
    args.dry_run = Confirm.ask("Dry run only", default=args.dry_run)
    args.sha256 = [str(record.get("sha256") or "")[:12] for record in selected]


class InteractiveBlender:
    def __init__(self, args: argparse.Namespace, console: Console):
        self.args = args
        self.console = console
        self.run_filters = lab.parse_run_filters(args.run)
        self.registry = smart.SubmissionRegistry.load(args.registry)
        self.index = self._refresh_index()
        self.include_blends = bool(args.include_blends)
        self.display_count = 20
        self.strategy = args.weighting if args.weighting != "manual" else "uniform"
        self.strategy_params = {
            "power": args.power,
            "temp": args.temp,
            "eps": args.eps,
            "manual_weights": None,
        }
        self.mode = args.mode
        self.selected_indices: set[int] = set()
        self.excluded_indices: set[int] = set()
        self.history: list[dict[str, Any]] = []
        self.candidates = self._load_candidates()

    def _refresh_index(self) -> dict[str, Any]:
        with lab._build_progress(self.console) as progress:
            return lab.refresh_index(
                logs_dir=self.args.logs_dir,
                index_path=self.args.index,
                competition=self.args.competition,
                runs=self.run_filters or None,
                reindex=self.args.reindex,
                progress=progress,
            )

    def _load_candidates(self) -> list[dict[str, Any]]:
        return interactive_candidate_records(
            index=self.index,
            registry=self.registry,
            run_filters=self.run_filters,
            competition=self.args.competition,
            include_blends=self.include_blends,
            family_dedupe=not self.args.no_family_dedupe,
            family_dedupe_epsilon=self.args.family_dedupe_epsilon,
            family_dedupe_depth=self.args.family_dedupe_depth,
        )

    def _selected(self) -> list[dict[str, Any]]:
        return [
            self.candidates[idx - 1]
            for idx in sorted(self.selected_indices)
            if 1 <= idx <= len(self.candidates) and idx not in self.excluded_indices
        ]

    def _scores_for_weighting(self, selected: list[dict[str, Any]]) -> list[float]:
        if self.strategy == "public":
            scores = []
            for record in selected:
                public = public_score_for_record(record, self.registry)
                if public is None:
                    raise ValueError(
                        f"Missing public score for {str(record.get('sha256') or '')[:10]}"
                    )
                scores.append(public)
            return scores
        return [float(record.get("local_score")) for record in selected]

    def _weights(self, selected: list[dict[str, Any]]) -> list[float]:
        if self.strategy == "manual":
            manual = self.strategy_params.get("manual_weights")
        else:
            manual = None
        return compute_weights(
            self._scores_for_weighting(selected),
            mode=self.strategy,
            manual=manual,
            power=float(self.strategy_params["power"]),
            temp=float(self.strategy_params["temp"]),
            eps=float(self.strategy_params["eps"]),
            lower_is_better=bool(self.args.lower_is_better),
        )

    def render(self) -> None:
        self.console.clear()
        table_width = render_interactive_candidate_table(
            self.console,
            self.candidates,
            registry=self.registry,
            limit=self.display_count,
            selected_indices=self.selected_indices,
        )
        command_text = """[1-9] select top N    [0] top 10    [1,3,5] select IDs/sha
[A] include/exclude blend artifacts  [W] weighting strategy  [B] blend mode
[D] display count                    [E] exclude selected    [X] clear
[C] create queued AIDE artifact      [M] multi-strategy batch
[F] fetch public scores from Kaggle  [Q] quit"""
        self.console.print(
            Panel(command_text, title="Commands", box=box.ROUNDED, width=table_width)
        )
        selected = self._selected()
        selected_text = ", ".join(str(idx) for idx in sorted(self.selected_indices)) or "-"
        self.console.print(
            Panel(
                f"Selected: {selected_text}\n"
                f"Components: {len(selected)}\n"
                f"Strategy: {self.strategy}\n"
                f"Blend mode: {self.mode}\n"
                f"Include prior blends: {self.include_blends}\n"
                f"Family dedupe: {not self.args.no_family_dedupe} "
                f"(eps={self.args.family_dedupe_epsilon:g}, "
                f"depth={self.args.family_dedupe_depth})",
                title="Current config",
                box=box.ROUNDED,
                width=table_width,
            )
        )
        if self.history:
            table = Table(title="Created artifacts", box=box.SIMPLE)
            table.add_column("cv", justify="right")
            table.add_column("mode")
            table.add_column("strategy")
            table.add_column("artifact")
            for item in self.history[-8:]:
                table.add_row(
                    f"{item['cv']:.5f}",
                    item["mode"],
                    item["strategy"],
                    item["artifact"] or "dry-run",
                )
            self.console.print(table)

    def select_top(self, count: int) -> None:
        self.selected_indices = set(range(1, min(count, len(self.candidates)) + 1))

    def select_specific(self, value: str) -> None:
        selected = parse_interactive_selection(
            value,
            displayed_records=self.candidates[: self.display_count],
            all_records=self.candidates,
        )
        by_sha = {str(record.get("sha256") or ""): idx for idx, record in enumerate(self.candidates, start=1)}
        self.selected_indices = {
            by_sha[str(record.get("sha256") or "")]
            for record in selected
            if str(record.get("sha256") or "") in by_sha
        }

    def choose_strategy(self) -> None:
        text = """1. public   - public leaderboard score from registry
2. local    - local CV score
3. power    - score ** power
4. rank     - rank weights
5. softmax  - score softmax
6. offset   - score offset from minimum
7. uniform  - equal weights
8. manual   - custom weights"""
        self.console.print(Panel(text, title="Weighting strategy", box=box.ROUNDED))
        choice = Prompt.ask("Choice", choices=[str(i) for i in range(1, 9)], default="7")
        mapping = {
            "1": "public",
            "2": "local",
            "3": "power",
            "4": "rank",
            "5": "softmax",
            "6": "offset",
            "7": "uniform",
            "8": "manual",
        }
        self.strategy = mapping[choice]
        if self.strategy == "power":
            self.strategy_params["power"] = float(Prompt.ask("Power", default=str(self.strategy_params["power"])))
        elif self.strategy == "softmax":
            self.strategy_params["temp"] = float(Prompt.ask("Temperature", default=str(self.strategy_params["temp"])))
        elif self.strategy == "offset":
            self.strategy_params["eps"] = float(Prompt.ask("Epsilon", default=str(self.strategy_params["eps"])))
        elif self.strategy == "manual":
            count = len(self._selected())
            if count < 2:
                self.console.print("[red]Select components first.[/red]")
                self.strategy = "uniform"
                Prompt.ask("Press Enter")
                return
            raw = Prompt.ask(f"Enter {count} comma/space separated weights")
            weights = [float(part) for part in re.split(r"[\s,]+", raw.strip()) if part]
            if len(weights) != count:
                self.console.print(f"[red]Expected {count} weights, got {len(weights)}[/red]")
                self.strategy = "uniform"
                Prompt.ask("Press Enter")
                return
            self.strategy_params["manual_weights"] = weights

    def create_current(self, *, dry_run: bool | None = None, label_suffix: str = "") -> None:
        selected = self._selected()
        if len(selected) < 2:
            self.console.print("[red]Select at least two components.[/red]")
            Prompt.ask("Press Enter")
            return
        labels = make_indexed_labels(selected, self.selected_indices)
        weights = self._weights(selected)
        output_run = self.args.output_run
        if output_run is None:
            output_run = DEFAULT_OUTPUT_RUN
        default_label = f"blend-{self.strategy}-{self.mode}-" + "-".join(label[:8] for label in labels)
        if label_suffix:
            default_label = f"{default_label}-{label_suffix}"
        label = Prompt.ask("Artifact label", default=default_label)
        write_dry_run = self.args.dry_run if dry_run is None else dry_run
        if dry_run is None:
            write_dry_run = Confirm.ask("Dry run only", default=False)
        try:
            cv, artifact, skipped_duplicate = create_blend_artifact(
                args=self.args,
                console=self.console,
                selected=selected,
                labels=labels,
                weights=weights,
                output_run=output_run,
                label=label,
                mode=self.mode,
                dry_run=write_dry_run,
                index=self.index,
                allow_submission_only=None,
                refresh_index_after_write=False,
            )
        except FileNotFoundError as exc:
            self.console.print(f"[red]{exc}[/red]")
            Prompt.ask("Press Enter")
            return
        self.history.append(
            {
                "cv": cv,
                "mode": self.mode,
                "strategy": self.strategy,
                "artifact": Path(artifact).name if artifact else None,
            }
        )
        if artifact:
            self.index = self._refresh_index()
        self.console.print(
            f"Queued: {1 if artifact else 0}; duplicate skipped: {1 if skipped_duplicate else 0}"
        )
        Prompt.ask("Press Enter")

    def multi_strategy_batch(self) -> None:
        selected = self._selected()
        if len(selected) < 2:
            self.console.print("[red]Select at least two components.[/red]")
            Prompt.ask("Press Enter")
            return
        strategy_specs = [
            ("uniform", {}, "uniform"),
            ("rank", {}, "rank"),
            ("local", {}, "local"),
            ("power", {"power": 4.0}, "power4"),
            ("power", {"power": 8.0}, "power8"),
            ("offset", {"eps": 1e-6}, "offset"),
        ]
        if all(public_score_for_record(record, self.registry) is not None for record in selected):
            strategy_specs.insert(0, ("public", {}, "public"))
        modes = ["raw", "rank", "logit"]
        output_run = self.args.output_run
        if output_run is None:
            output_run = DEFAULT_OUTPUT_RUN
        write = Confirm.ask(
            f"Create {len(strategy_specs) * len(modes)} queued blend artifacts?",
            default=False,
        )
        if not write:
            return
        original_strategy = self.strategy
        original_mode = self.mode
        original_params = dict(self.strategy_params)
        labels = make_indexed_labels(selected, self.selected_indices)
        known_sha256 = existing_submission_sha256(self.index)
        next_step = next_step_for_run(self.index, output_run)
        written_count = 0
        duplicate_count = 0
        for strategy, params, suffix in strategy_specs:
            self.strategy = strategy
            self.strategy_params.update(params)
            for mode in modes:
                self.mode = mode
                weights = self._weights(selected)
                try:
                    cv, artifact, skipped_duplicate = create_blend_artifact(
                        args=self.args,
                        console=self.console,
                        selected=selected,
                        labels=labels,
                        weights=weights,
                        output_run=output_run,
                        label=f"blend-{suffix}-{mode}-" + "-".join(label[:8] for label in labels),
                        mode=mode,
                        dry_run=False,
                        index=self.index,
                        allow_submission_only=None,
                        known_submission_sha256=known_sha256,
                        step_override=next_step,
                        refresh_index_after_write=False,
                    )
                except FileNotFoundError as exc:
                    self.console.print(f"[red]{exc}[/red]")
                    Prompt.ask("Press Enter")
                    return
                if skipped_duplicate:
                    duplicate_count += 1
                    continue
                if artifact:
                    written_count += 1
                    next_step += 1
                self.history.append(
                    {
                        "cv": cv,
                        "mode": mode,
                        "strategy": strategy,
                        "artifact": Path(artifact).name if artifact else None,
                    }
                )
        if written_count:
            self.index = self._refresh_index()
        self.strategy = original_strategy
        self.mode = original_mode
        self.strategy_params = original_params
        self.console.print(f"Queued: {written_count}; duplicate skipped: {duplicate_count}")
        Prompt.ask("Press Enter")

    def fetch_scores(self) -> None:
        self.registry = smart.SubmissionRegistry.load(self.args.registry)
        lab.sync_registry_from_kaggle(
            console=self.console,
            registry=self.registry,
            competition=self.args.competition,
        )
        self.registry = smart.SubmissionRegistry.load(self.args.registry)
        self.candidates = self._load_candidates()
        Prompt.ask("Press Enter")

    def loop(self) -> None:
        while True:
            self.render()
            cmd = Prompt.ask("Command", default="q").strip().lower()
            if cmd == "q":
                return
            if cmd in {str(i) for i in range(1, 10)}:
                self.select_top(int(cmd))
            elif cmd == "0":
                self.select_top(10)
            elif "," in cmd or re.fullmatch(r"[0-9a-f]{6,}", cmd):
                try:
                    self.select_specific(cmd)
                except ValueError as exc:
                    self.console.print(f"[red]{exc}[/red]")
                    Prompt.ask("Press Enter")
            elif cmd == "a":
                self.include_blends = not self.include_blends
                self.candidates = self._load_candidates()
                self.selected_indices.clear()
            elif cmd == "w":
                self.choose_strategy()
            elif cmd == "b":
                self.mode = Prompt.ask("Blend mode", choices=["raw", "rank", "logit"], default=self.mode)
            elif cmd == "d":
                self.display_count = int(Prompt.ask("Display count", default=str(self.display_count)))
            elif cmd == "e":
                raw = Prompt.ask("Exclude indices, comma-separated", default="")
                self.excluded_indices.update(
                    int(part) for part in re.split(r"[\s,]+", raw.strip()) if part
                )
            elif cmd == "x":
                self.selected_indices.clear()
                self.excluded_indices.clear()
            elif cmd == "c":
                self.create_current()
            elif cmd == "m":
                self.multi_strategy_batch()
            elif cmd == "f":
                self.fetch_scores()
            else:
                self.console.print("[red]Unknown command.[/red]")
                Prompt.ask("Press Enter")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create AIDE submit-ready blend artifacts from existing OOF/test predictions."
    )
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--competition", default=smart.DEFAULT_COMPETITION)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--run", action="append", help="Source run filter; may be repeated.")
    parser.add_argument(
        "--output-run",
        default=DEFAULT_OUTPUT_RUN,
        help="Run directory where the blend artifact is written.",
    )
    parser.add_argument("--sha256", action="append", help="Component sha256 prefix; may be repeated.")
    parser.add_argument("--top-n", type=int, help="Use top-N indexed records when --sha256 is omitted.")
    parser.add_argument("--include-blends", action="store_true", help="Allow existing blend artifacts in top-N selection.")
    parser.add_argument("--no-family-dedupe", action="store_true", help="Do not hide near-duplicate descendants from the same run family.")
    parser.add_argument("--family-dedupe-epsilon", type=float, default=0.00003, help="Hide descendants within this local-score delta.")
    parser.add_argument("--family-dedupe-depth", type=int, default=3, help="Maximum parent-chain distance for near-duplicate hiding.")
    parser.add_argument("--weighting", choices=["manual", "uniform", "local", "public", "rank", "power", "softmax", "offset"], default="manual")
    parser.add_argument("--weights", help="Comma-separated manual weights, e.g. 0.7,0.3")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--temp", type=float, default=0.001)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--lower-is-better", action="store_true")
    parser.add_argument("--mode", choices=["raw", "rank", "logit"], default="raw")
    parser.add_argument("--label", help="Artifact label/hypothesis id; default is generated from components.")
    parser.add_argument("--step", type=int, help="Manual step number; default is max(output-run step)+1.")
    parser.add_argument("--reindex", action="store_true", help="Refresh selected AIDE run index before blending.")
    parser.add_argument("--dry-run", action="store_true", help="Show selected components and CV without writing an artifact.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    console = Console()
    if args.sha256 is None and args.top_n is None:
        InteractiveBlender(args, console).loop()
        return

    run_filters = lab.parse_run_filters(args.run)
    sha_filters = lab.parse_sha256_filters(args.sha256)
    output_run = args.output_run

    index = lab.refresh_index(
        logs_dir=args.logs_dir,
        index_path=args.index,
        competition=args.competition,
        runs=run_filters or None,
        reindex=args.reindex,
    )
    registry = smart.SubmissionRegistry.load(args.registry)
    selected = select_records(
        index=index,
        registry=registry,
        run_filters=run_filters,
        sha_filters=sha_filters,
        top_n=args.top_n,
        include_blends=args.include_blends,
        weighting=args.weighting,
        competition=args.competition,
        family_dedupe=not args.no_family_dedupe,
        family_dedupe_epsilon=args.family_dedupe_epsilon,
        family_dedupe_depth=args.family_dedupe_depth,
    )

    labels = make_unique_labels(selected)
    public_scores = [public_score_for_record(record, registry) for record in selected]
    if args.weighting == "public":
        score_values = []
        for record, public in zip(selected, public_scores, strict=True):
            if public is None:
                raise ValueError(
                    f"Missing public score for {record.get('sha256', '')[:10]}"
                )
            score_values.append(public)
    else:
        score_values = [float(record.get("local_score")) for record in selected]

    weights = compute_weights(
        score_values,
        mode=args.weighting,
        manual=parse_csv_floats(args.weights),
        power=args.power,
        temp=args.temp,
        eps=args.eps,
        lower_is_better=args.lower_is_better,
    )

    create_blend_artifact(
        args=args,
        console=console,
        selected=selected,
        labels=labels,
        weights=weights,
        output_run=output_run,
        label=args.label,
        mode=args.mode,
        dry_run=args.dry_run,
        index=index,
        allow_submission_only=None,
    )


if __name__ == "__main__":
    main()
