from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from aide.utils.submission_validation import validate_submission_file
from scripts import kaggle_submission_lab as lab
from scripts import smart_kaggle_submit as smart


DEFAULT_OUTPUT_RUN = "blended"
DEFAULT_TOP_K = 12
DEFAULT_MAX_COMBOS_PER_SIZE = 6
DEFAULT_BAD_BLEND_PUBLIC_MARGIN = 0.00005
DEFAULT_BAD_BLEND_OVERLAP = 0.75


@dataclass(frozen=True)
class BlendSpec:
    blend_kind: str
    mode: str
    weighting: str
    component_shas: tuple[str, ...]
    weights: tuple[float, ...]
    params: dict[str, Any]

    @property
    def recipe_payload(self) -> dict[str, Any]:
        return {
            "blend_kind": self.blend_kind,
            "mode": self.mode,
            "weighting": self.weighting,
            "component_shas": list(self.component_shas),
            "weights": [round(float(weight), 12) for weight in self.weights],
            "params": self.params,
        }

    @property
    def recipe_hash(self) -> str:
        payload = json.dumps(self.recipe_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class BlendResult:
    spec: BlendSpec
    records: tuple[dict[str, Any], ...]
    submission: pd.DataFrame
    test_predictions: pd.DataFrame
    oof_predictions: pd.DataFrame | None
    score: float | None
    component_score: float
    submission_sha256: str


class PredictionCache:
    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        self._lock = Lock()

    def read(self, path: Path) -> pd.DataFrame:
        key = str(path)
        with self._lock:
            if key not in self._frames:
                self._frames[key] = read_prediction_file(path)
            return self._frames[key]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataframe_csv(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fast_dataframe_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    for column in frame.columns:
        series = frame[column]
        if is_numeric_series(series):
            values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
            digest.update(values.tobytes())
            continue
        values = series.astype(str).to_numpy()
        labels = sorted(pd.unique(values))
        digest.update(json.dumps(labels, separators=(",", ":")).encode("utf-8"))
        dtype = np.uint16 if len(labels) <= np.iinfo(np.uint16).max else np.uint32
        codes = np.empty(len(values), dtype=dtype)
        for code, label in enumerate(labels):
            codes[values == label] = code
        digest.update(codes.tobytes())
    return digest.hexdigest()


def normalize_weights(weights: Iterable[float]) -> tuple[float, ...]:
    raw = [float(weight) for weight in weights]
    if any(weight < 0 for weight in raw):
        raise ValueError("Blend weights must be non-negative")
    total = sum(raw)
    if total <= 0:
        raise ValueError("Blend weights must sum to a positive value")
    return tuple(weight / total for weight in raw)


def score_for_record(record: dict[str, Any]) -> float | None:
    public_score = record.get("public_score")
    if public_score not in {None, ""}:
        try:
            return float(public_score)
        except (TypeError, ValueError):
            pass
    local_score = record.get("local_score")
    if local_score is None:
        return None
    try:
        return float(local_score)
    except (TypeError, ValueError):
        return None


def enrich_records_with_registry(
    records: list[dict[str, Any]],
    registry: smart.SubmissionRegistry,
    *,
    competition: str,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in registry.entries
        if str(entry.get("competition") or competition) == competition
        and entry.get("sha256")
    ]
    enriched = []
    for record in records:
        item = dict(record)
        record_sha = item.get("sha256")
        for entry in entries:
            if not smart._sha256_matches(record_sha, entry.get("sha256")):
                continue
            for key in (
                "public_score",
                "private_score",
                "remote_status",
                "blend_component_sha256",
            ):
                if entry.get(key) not in {None, ""}:
                    item[key] = entry.get(key)
            break
        enriched.append(item)
    return enriched


def blend_component_set(record: dict[str, Any]) -> frozenset[str]:
    raw = record.get("blend_component_sha256")
    if isinstance(raw, dict):
        return frozenset(str(value) for value in raw.values() if value)
    if isinstance(raw, (list, tuple, set)):
        return frozenset(str(value) for value in raw if value)
    return frozenset()


def bad_blend_component_sets(
    records: list[dict[str, Any]],
    *,
    public_margin: float = DEFAULT_BAD_BLEND_PUBLIC_MARGIN,
) -> list[frozenset[str]]:
    scored = [
        (score, components)
        for record in records
        if is_blend_record(record)
        and (score := smart._parse_public_score(record.get("public_score"))) is not None
        and (components := blend_component_set(record))
    ]
    if len(scored) < 2:
        return []
    best_public = max(score for score, _components in scored)
    return [
        components
        for score, components in scored
        if score < best_public - public_margin
    ]


def too_similar_to_bad_blend(
    spec: BlendSpec,
    bad_component_sets: Iterable[frozenset[str]],
    *,
    overlap_threshold: float = DEFAULT_BAD_BLEND_OVERLAP,
) -> bool:
    candidate = frozenset(spec.component_shas)
    if not candidate:
        return False
    for bad_components in bad_component_sets:
        if not bad_components:
            continue
        overlap = len(candidate & bad_components) / min(len(candidate), len(bad_components))
        if overlap >= overlap_threshold:
            return True
    return False


def record_test_or_submission_path(record: dict[str, Any]) -> Path:
    artifact_dir = Path(str(record.get("artifact_dir") or ""))
    test_path = artifact_dir / "test_predictions.csv.gz"
    if test_path.exists():
        return test_path
    return artifact_dir / "submission.csv"


def record_oof_path(record: dict[str, Any]) -> Path:
    return Path(str(record.get("artifact_dir") or "")) / "oof_predictions.csv.gz"


def record_has_prediction_inputs(record: dict[str, Any]) -> bool:
    return record_test_or_submission_path(record).exists()


def is_blend_record(record: dict[str, Any]) -> bool:
    text = " ".join(
        str(record.get(key) or "")
        for key in ("run", "hypothesis_id", "timestamp", "artifact_dir", "origin")
    ).lower()
    return any(token in text for token in ("blend", "stack", "ensemble", "manual"))


def read_prediction_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer")


def align_frame(
    frame: pd.DataFrame,
    *,
    id_col: str,
    base_ids: pd.Series,
) -> pd.DataFrame:
    if frame[id_col].equals(base_ids):
        return frame.reset_index(drop=True)
    aligned = frame.set_index(id_col).reindex(base_ids).reset_index()
    if aligned.isna().any(axis=None):
        raise ValueError(f"Cannot align predictions by {id_col!r}")
    return aligned


def is_numeric_series(values: pd.Series) -> bool:
    try:
        pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError):
        return False
    return True


def blend_categorical_columns(values: list[pd.Series], weights: tuple[float, ...]) -> pd.Series:
    arrays = [series.astype(str).to_numpy() for series in values]
    labels = np.array(
        sorted({label for array in arrays for label in pd.unique(array)}),
        dtype=object,
    )
    stacked = np.vstack(arrays)
    weight_array = np.asarray(weights, dtype=float)[:, None]
    label_scores = np.vstack(
        [((stacked == label) * weight_array).sum(axis=0) for label in labels]
    )
    return pd.Series(labels[np.argmax(label_scores, axis=0)])


def transform_numeric(values: pd.Series, mode: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if mode == "raw":
        return numeric
    if mode == "rank":
        return numeric.rank(method="average", pct=True)
    if mode == "logit":
        clipped = numeric.clip(1e-9, 1.0 - 1e-9)
        return np.log(clipped / (1.0 - clipped))
    raise ValueError(f"Unsupported numeric blend mode: {mode}")


def finish_numeric(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "logit":
        return 1.0 / (1.0 + np.exp(-values))
    return values


def blend_prediction_frames(
    frames: list[pd.DataFrame],
    *,
    weights: tuple[float, ...],
    id_col: str,
    target_col: str,
    mode: str,
) -> pd.DataFrame:
    base_ids = frames[0][id_col].copy()
    aligned = [align_frame(frame, id_col=id_col, base_ids=base_ids) for frame in frames]
    if all(is_numeric_series(frame[target_col]) for frame in aligned):
        blended = np.zeros(len(base_ids), dtype=float)
        for frame, weight in zip(aligned, weights, strict=True):
            blended += float(weight) * transform_numeric(frame[target_col], mode)
        target = pd.Series(finish_numeric(blended, mode))
    else:
        target = blend_categorical_columns([frame[target_col] for frame in aligned], weights)
    return pd.DataFrame({id_col: base_ids.reset_index(drop=True), target_col: target})


def blend_oof_frames(
    frames: list[pd.DataFrame],
    *,
    weights: tuple[float, ...],
    mode: str,
    metric_name: str,
) -> tuple[pd.DataFrame, float]:
    required = {"row", "target", "prediction"}
    if not required.issubset(frames[0].columns):
        raise ValueError(f"OOF file must contain {sorted(required)}")
    base_rows = frames[0]["row"].copy()
    base_target = frames[0]["target"].copy()
    aligned = []
    for frame in frames:
        if not required.issubset(frame.columns):
            raise ValueError(f"OOF file must contain {sorted(required)}")
        if frame["row"].equals(base_rows):
            item = frame.reset_index(drop=True)
        else:
            item = frame.set_index("row").reindex(base_rows).reset_index()
            if item[["target", "prediction"]].isna().any(axis=None):
                raise ValueError("Cannot align OOF predictions by 'row'")
        if not item["target"].astype(str).equals(base_target.astype(str)):
            raise ValueError("OOF target mismatch")
        aligned.append(item)

    if all(is_numeric_series(frame["prediction"]) for frame in aligned):
        blended = np.zeros(len(base_rows), dtype=float)
        for frame, weight in zip(aligned, weights, strict=True):
            blended += float(weight) * transform_numeric(frame["prediction"], mode)
        prediction = pd.Series(finish_numeric(blended, mode))
    else:
        prediction = blend_categorical_columns([frame["prediction"] for frame in aligned], weights)
    out = pd.DataFrame({"row": base_rows, "target": base_target, "prediction": prediction})
    return out, score_oof(out["target"], out["prediction"], metric_name)


def score_oof(target: pd.Series, prediction: pd.Series, metric_name: str) -> float:
    metric = (metric_name or "").lower()
    target_is_numeric = is_numeric_series(target)
    prediction_is_numeric = is_numeric_series(prediction)
    if not target_is_numeric or not prediction_is_numeric:
        target_values = target.astype(str).to_numpy()
        prediction_values = prediction.astype(str).to_numpy()
        if "balanced" in metric:
            recalls = []
            for label in pd.unique(target_values):
                mask = target_values == label
                if mask.any():
                    recalls.append(float(np.mean(prediction_values[mask] == label)))
            return float(np.mean(recalls)) if recalls else 0.0
        return float(np.mean(target_values == prediction_values))
    if "balanced" in metric:
        return float(balanced_accuracy_score(target, prediction))
    if "accuracy" in metric:
        return float(accuracy_score(target, prediction))
    if "roc" in metric or "auc" in metric:
        return float(roc_auc_score(target, pd.to_numeric(prediction, errors="raise")))
    error = pd.to_numeric(target) - pd.to_numeric(prediction)
    return float(-math.sqrt(float(np.mean(np.square(error)))))


def records_by_sha(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for record in records:
        sha = str(record.get("sha256") or "")
        if sha and sha not in out:
            out[sha] = record
    return out


def evaluate_blend_spec(
    spec: BlendSpec,
    records: list[dict[str, Any]],
    *,
    id_col: str,
    target_col: str,
    metric_name: str,
    cache: PredictionCache | None = None,
) -> BlendResult:
    cache = cache or PredictionCache()
    lookup = records_by_sha(records)
    selected = tuple(lookup[sha] for sha in spec.component_shas)
    test_frames = [cache.read(record_test_or_submission_path(record)) for record in selected]
    test_predictions = blend_prediction_frames(
        test_frames,
        weights=spec.weights,
        id_col=id_col,
        target_col=target_col,
        mode=spec.mode,
    )
    oof_predictions = None
    score = None
    if spec.blend_kind == "oof":
        oof_frames = [cache.read(record_oof_path(record)) for record in selected]
        oof_predictions, score = blend_oof_frames(
            oof_frames,
            weights=spec.weights,
            mode=spec.mode,
            metric_name=metric_name,
        )
    component_scores = [score_for_record(record) or 0.0 for record in selected]
    component_score = float(np.average(component_scores, weights=spec.weights))
    submission = test_predictions[[id_col, target_col]].copy()
    return BlendResult(
        spec=spec,
        records=selected,
        submission=submission,
        test_predictions=test_predictions,
        oof_predictions=oof_predictions,
        score=score,
        component_score=component_score,
        submission_sha256=spec.recipe_hash,
    )


def choose_modes(
    records: list[dict[str, Any]],
    *,
    target_col: str,
    cache: PredictionCache | None = None,
) -> list[str]:
    cache = cache or PredictionCache()
    for record in records:
        path = record_test_or_submission_path(record)
        if path.exists():
            frame = cache.read(path)
            if target_col in frame.columns and is_numeric_series(frame[target_col]):
                return ["raw", "rank", "logit"]
            break
    return ["vote"]


def score_weights(records: tuple[dict[str, Any], ...], *, power: float = 1.0) -> tuple[float, ...]:
    scores = [max(score_for_record(record) or 0.0, 0.0) ** power for record in records]
    if sum(scores) <= 0:
        return normalize_weights([1.0] * len(records))
    return normalize_weights(scores)


def portfolio_combinations(
    pool: list[dict[str, Any]],
    *,
    size: int,
    max_combos: int,
) -> list[tuple[dict[str, Any], ...]]:
    combos: list[tuple[dict[str, Any], ...]] = []
    seen: set[tuple[str, ...]] = set()

    def add(combo: tuple[dict[str, Any], ...]) -> None:
        key = tuple(str(record.get("sha256") or "") for record in combo)
        if len(combo) == size and key not in seen:
            seen.add(key)
            combos.append(combo)

    if len(pool) < size:
        return combos

    add(tuple(pool[:size]))
    for idx in range(size - 1, len(pool)):
        add(tuple([*pool[: size - 1], pool[idx]]))
        if len(combos) >= max_combos:
            return combos

    max_start = max(0, len(pool) - size)
    for start in range(1, max_start + 1):
        add(tuple(pool[start : start + size]))
        if len(combos) >= max_combos:
            return combos

    for combo in combinations(pool[: min(len(pool), size + 4)], size):
        add(combo)
        if len(combos) >= max_combos:
            return combos
    return combos


def candidate_specs(
    records: list[dict[str, Any]],
    *,
    blend_kind: str,
    target_col: str,
    top_k: int,
    cache: PredictionCache | None = None,
    max_combos_per_size: int = DEFAULT_MAX_COMBOS_PER_SIZE,
) -> list[BlendSpec]:
    pool = sorted(
        records,
        key=lambda record: (score_for_record(record) is not None, score_for_record(record) or 0.0),
        reverse=True,
    )[:top_k]
    if blend_kind == "oof":
        pool = [record for record in pool if record_oof_path(record).exists()]
    modes = choose_modes(pool, target_col=target_col, cache=cache)
    specs = []
    for size in (2, 3, 4, 5):
        if len(pool) < size:
            continue
        for combo in portfolio_combinations(
            pool,
            size=size,
            max_combos=max_combos_per_size,
        ):
            component_shas = tuple(str(record.get("sha256")) for record in combo)
            weighting_options = [
                ("uniform", normalize_weights([1.0] * size), {}),
                ("rank", normalize_weights(range(size, 0, -1)), {}),
                ("score", score_weights(combo), {}),
            ]
            if size <= 4:
                weighting_options.append(("power4", score_weights(combo, power=4.0), {"power": 4.0}))
            for mode in modes:
                for weighting, weights, params in weighting_options:
                    specs.append(
                        BlendSpec(
                            blend_kind=blend_kind,
                            mode=mode,
                            weighting=weighting,
                            component_shas=component_shas,
                            weights=weights,
                            params=params,
                        )
                    )
    return specs


def select_new_blends(
    records: list[dict[str, Any]],
    *,
    count: int,
    existing_submission_sha256: set[str],
    existing_recipe_hashes: set[str],
    id_col: str,
    target_col: str,
    metric_name: str,
    top_k: int = DEFAULT_TOP_K,
    jobs: int = 1,
    bad_component_sets: Iterable[frozenset[str]] = (),
    bad_blend_overlap: float = DEFAULT_BAD_BLEND_OVERLAP,
    progress_callback: Any | None = None,
) -> list[BlendResult]:
    ready = [
        record
        for record in records
        if record.get("status") == "ok"
        and not record.get("is_buggy")
        and record.get("sha256")
        and record_has_prediction_inputs(record)
    ]
    oof_quota = (count + 1) // 2
    submission_quota = count // 2
    selected: list[BlendResult] = []
    seen_sha = set(existing_submission_sha256)
    seen_recipe = set(existing_recipe_hashes)
    seen_evaluated_fingerprints: dict[str, set[str]] = {"oof": set(), "submission": set()}
    seen_selected_fingerprints: set[str] = set()
    cache = PredictionCache()

    evaluated: dict[str, list[BlendResult]] = {"oof": [], "submission": []}
    specs_by_kind = {
        blend_kind: candidate_specs(
            ready,
            blend_kind=blend_kind,
            target_col=target_col,
            top_k=top_k,
            cache=cache,
        )
        for blend_kind in ("oof", "submission")
    }
    total_specs = sum(len(specs) for specs in specs_by_kind.values())
    done_specs = 0

    def record_progress(kind: str) -> None:
        nonlocal done_specs
        done_specs += 1
        if progress_callback is not None:
            progress_callback(done_specs, total_specs, kind)

    def add_evaluated(kind: str, result: BlendResult) -> None:
        fingerprint = result.submission_sha256
        if fingerprint in seen_evaluated_fingerprints[kind]:
            return
        seen_evaluated_fingerprints[kind].add(fingerprint)
        evaluated[kind].append(result)

    for blend_kind in ("oof", "submission"):
        if jobs <= 1:
            for spec in specs_by_kind[blend_kind]:
                if spec.recipe_hash in seen_recipe or too_similar_to_bad_blend(
                    spec,
                    bad_component_sets,
                    overlap_threshold=bad_blend_overlap,
                ):
                    record_progress(blend_kind)
                    continue
                try:
                    result = evaluate_blend_spec(
                        spec,
                        ready,
                        id_col=id_col,
                        target_col=target_col,
                        metric_name=metric_name,
                        cache=cache,
                    )
                except Exception:
                    record_progress(blend_kind)
                    continue
                add_evaluated(blend_kind, result)
                record_progress(blend_kind)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {}
                for spec in specs_by_kind[blend_kind]:
                    if spec.recipe_hash in seen_recipe or too_similar_to_bad_blend(
                        spec,
                        bad_component_sets,
                        overlap_threshold=bad_blend_overlap,
                    ):
                        record_progress(blend_kind)
                        continue
                    futures[
                        executor.submit(
                            evaluate_blend_spec,
                            spec,
                            ready,
                            id_col=id_col,
                            target_col=target_col,
                            metric_name=metric_name,
                            cache=cache,
                        )
                    ] = spec
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception:
                        record_progress(blend_kind)
                        continue
                    add_evaluated(blend_kind, result)
                    record_progress(blend_kind)
        if blend_kind == "oof":
            evaluated[blend_kind].sort(
                key=lambda item: (
                    item.score is not None,
                    item.score if item.score is not None else float("-inf"),
                    item.component_score,
                    item.spec.recipe_hash,
                ),
                reverse=True,
            )
        else:
            evaluated[blend_kind].sort(
                key=lambda item: (item.component_score, item.spec.recipe_hash),
                reverse=True,
            )

    def take(kind: str, quota: int) -> None:
        nonlocal selected
        for result in evaluated[kind]:
            if len([item for item in selected if item.spec.blend_kind == kind]) >= quota:
                return
            if result.spec.recipe_hash in seen_recipe:
                continue
            fingerprint = fast_dataframe_fingerprint(result.submission)
            if fingerprint in seen_selected_fingerprints:
                continue
            exact_sha = sha256_dataframe_csv(result.submission)
            if exact_sha in seen_sha:
                continue
            result = replace(result, submission_sha256=exact_sha)
            selected.append(result)
            seen_recipe.add(result.spec.recipe_hash)
            seen_sha.add(result.submission_sha256)
            seen_selected_fingerprints.add(fingerprint)

    take("oof", oof_quota)
    take("submission", submission_quota)
    for result in sorted(
        evaluated["oof"] + evaluated["submission"],
        key=lambda item: (
            item.score if item.score is not None else item.component_score,
            item.component_score,
        ),
        reverse=True,
    ):
        if len(selected) >= count:
            break
        if result.spec.recipe_hash in seen_recipe:
            continue
        fingerprint = fast_dataframe_fingerprint(result.submission)
        if fingerprint in seen_selected_fingerprints:
            continue
        exact_sha = sha256_dataframe_csv(result.submission)
        if exact_sha in seen_sha:
            continue
        result = replace(result, submission_sha256=exact_sha)
        selected.append(result)
        seen_recipe.add(result.spec.recipe_hash)
        seen_sha.add(result.submission_sha256)
        seen_selected_fingerprints.add(fingerprint)
    return selected[:count]


def sample_submission_path(data_dir: Path) -> Path:
    for name in ("sample_submission.csv.gz", "sample_submission.csv"):
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing sample_submission.csv[.gz] in {data_dir}")


def existing_submission_sha256(index: dict[str, Any], registry: smart.SubmissionRegistry) -> set[str]:
    values = {
        str(record.get("sha256"))
        for record in index.get("records", [])
        if record.get("sha256")
    }
    values.update(str(entry.get("sha256")) for entry in registry.entries if entry.get("sha256"))
    return values


def existing_recipe_hashes(index: dict[str, Any]) -> set[str]:
    hashes = set()
    for record in index.get("records", []):
        recipe = record.get("blend_recipe_hash")
        if recipe:
            hashes.add(str(recipe))
    return hashes


def file_payload(path: Path, *, relative_to: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": sha256_file(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def write_solution_py(artifact_dir: Path, result: BlendResult) -> None:
    payload = {
        "blend_kind": result.spec.blend_kind,
        "blend_mode": result.spec.mode,
        "blend_weighting": result.spec.weighting,
        "blend_recipe_hash": result.spec.recipe_hash,
        "weights": dict(zip(result.spec.component_shas, result.spec.weights, strict=True)),
    }
    lines = [
        "# Auto prediction blend artifact generated by scripts/auto_blend_submissions.py.",
        f"BLEND_METADATA = {payload!r}",
        "",
    ]
    artifact_dir.joinpath("solution.py").write_text("\n".join(lines))


def safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "auto-blend"


def write_blend_artifact(
    *,
    result: BlendResult,
    args: argparse.Namespace,
    step: int,
    sample_path: Path,
    id_col: str,
    target_col: str,
    metric_name: str,
) -> str:
    created_at = dt.datetime.now(dt.timezone.utc)
    short_components = "-".join(sha[:6] for sha in result.spec.component_shas[:5])
    timestamp = (
        created_at.strftime("%Y%m%dT%H%M%S")
        + f"-auto-{result.spec.blend_kind}-{result.spec.weighting}-"
        + f"{result.spec.mode}-{short_components}-{result.spec.recipe_hash}"
    )
    timestamp = safe_label(timestamp)
    artifact_dir = args.logs_dir / args.output_run / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=False)

    if result.oof_predictions is not None:
        result.oof_predictions.to_csv(artifact_dir / "oof_predictions.csv.gz", index=False)
    result.test_predictions.to_csv(artifact_dir / "test_predictions.csv.gz", index=False)
    result.submission.to_csv(artifact_dir / "submission.csv", index=False)
    write_solution_py(artifact_dir, result)

    node_id = hashlib.md5(f"{timestamp}:{result.spec.recipe_hash}".encode()).hexdigest()
    validation_error = validate_submission_file(artifact_dir / "submission.csv", sample_path)
    labels = [str(record.get("sha256"))[:10] for record in result.records]
    source = result.records[0]
    metric_maximize = None if result.spec.blend_kind == "submission" else True
    manifest = {
        "schema_version": 1,
        "kind": "source_node",
        "competition": args.competition,
        "run": args.output_run,
        "timestamp": timestamp,
        "artifact_dir": str(artifact_dir),
        "created_at": created_at.isoformat(),
        "status": "ok" if validation_error is None else "error",
        "is_buggy": validation_error is not None,
        "local_score": result.score,
        "metric_maximize": metric_maximize,
        "eval_metric": metric_name,
        "hypothesis_id": safe_label(
            f"auto-blend-{result.spec.blend_kind}-{result.spec.weighting}-"
            f"{result.spec.mode}-{result.spec.recipe_hash}"
        ),
        "sha256": sha256_file(artifact_dir / "submission.csv"),
        "node": {
            "id": node_id,
            "parent_id": source.get("node_id"),
            "step": step,
            "ctime": created_at.timestamp(),
            "status": "ok" if validation_error is None else "error",
            "is_buggy": validation_error is not None,
            "metric": {"value": result.score, "maximize": metric_maximize, "name": metric_name},
            "analysis": (
                "Auto blend computed from OOF/test predictions."
                if result.spec.blend_kind == "oof"
                else "Auto submission-only blend computed from existing submissions."
            ),
            "plan": f"Auto blend recipe {result.spec.recipe_hash}: {result.spec.recipe_payload}",
            "origin": "auto_blend",
            "hypothesis_id": f"auto-blend-{result.spec.recipe_hash}",
            "validity_warning": validation_error,
        },
        "execution": {"exec_time": 0.0, "exc_type": None, "exc_info": None, "exc_stack": None},
        "run_stats": {
            "metric_name": metric_name,
            "cv_score": result.score,
            "submission_only": result.spec.blend_kind == "submission",
            "blend_kind": result.spec.blend_kind,
            "blend_mode": result.spec.mode,
            "blend_weighting": result.spec.weighting,
            "blend_recipe_hash": result.spec.recipe_hash,
            "blend_component_count": len(result.records),
            "blend_weights": dict(zip(labels, result.spec.weights, strict=True)),
            "components": labels,
            "component_sha256": dict(zip(labels, result.spec.component_shas, strict=True)),
            "recipe": result.spec.recipe_payload,
            "id_col": id_col,
            "target_col": target_col,
        },
        "files": {
            "submission": file_payload(artifact_dir / "submission.csv", relative_to=artifact_dir),
            "solution": file_payload(artifact_dir / "solution.py", relative_to=artifact_dir),
            "oof_predictions": (
                None
                if result.oof_predictions is None
                else file_payload(artifact_dir / "oof_predictions.csv.gz", relative_to=artifact_dir)
            ),
            "test_predictions": file_payload(
                artifact_dir / "test_predictions.csv.gz",
                relative_to=artifact_dir,
            ),
            "validation_predictions": None,
            "model_predictions": [],
            "error": validation_error,
        },
        "submission_validation": (
            {"status": "ok"} if validation_error is None else {"status": "error", "error": validation_error}
        ),
        "source": {
            "source_run": source.get("run"),
            "source_node_id": source.get("node_id"),
            "source_step": source.get("step"),
            "source_timestamp": source.get("timestamp"),
            "source_sha256": source.get("sha256"),
        },
    }
    artifact_dir.joinpath(lab.RESULT_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return str(artifact_dir)


def format_submit_command(sha_prefixes: list[str]) -> str:
    return "uv run python scripts/kaggle_submission_lab.py " + " ".join(
        f"--sha {sha}" for sha in sha_prefixes
    )


def render_created(console: Console, results: list[tuple[BlendResult, str]]) -> None:
    table = Table(title="Created auto blend artifacts")
    table.add_column("#", justify="right")
    table.add_column("kind")
    table.add_column("mode")
    table.add_column("weights")
    table.add_column("score", justify="right")
    table.add_column("sha")
    table.add_column("artifact")
    for idx, (result, artifact) in enumerate(results, start=1):
        table.add_row(
            str(idx),
            result.spec.blend_kind,
            result.spec.mode,
            result.spec.weighting,
            "-" if result.score is None else f"{result.score:.5f}",
            result.submission_sha256[:10],
            Path(artifact).name,
        )
    console.print(table)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create submit-ready auto blend artifacts and print a Kaggle submit queue."
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--competition", default=smart.DEFAULT_COMPETITION)
    parser.add_argument("--data-dir", type=Path, default=Path("aide/example_tasks") / smart.DEFAULT_COMPETITION)
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--index", type=Path, default=Path("logs/submission_index.json"))
    parser.add_argument("--registry", type=Path, default=smart.DEFAULT_REGISTRY)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--jobs", type=int, default=1, help="Parallel candidate evaluations")
    parser.add_argument(
        "--bad-blend-public-margin",
        type=float,
        default=DEFAULT_BAD_BLEND_PUBLIC_MARGIN,
        help="Public score gap from best known blend that marks a blend as bad feedback",
    )
    parser.add_argument(
        "--bad-blend-overlap",
        type=float,
        default=DEFAULT_BAD_BLEND_OVERLAP,
        help="Component overlap threshold used to skip candidates similar to bad public blends",
    )
    parser.add_argument(
        "--ignore-bad-public-blends",
        action="store_true",
        help="Disable negative public-score feedback from previous blends",
    )
    parser.add_argument("--output-run", default=DEFAULT_OUTPUT_RUN)
    parser.add_argument("--include-blends", action="store_true")
    parser.add_argument("--no-remote", action="store_true")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    console = Console()
    if args.count <= 0:
        console.print("[red]--count must be positive[/red]")
        return 2
    if args.jobs <= 0:
        console.print("[red]--jobs must be positive[/red]")
        return 2
    if not 0 <= args.bad_blend_overlap <= 1:
        console.print("[red]--bad-blend-overlap must be between 0 and 1[/red]")
        return 2

    registry = smart.SubmissionRegistry.load(args.registry)
    if not args.no_remote:
        lab.sync_registry_from_kaggle(
            console=console,
            registry=registry,
            competition=args.competition,
        )
        registry = smart.SubmissionRegistry.load(args.registry)
    run_filters = lab.parse_run_filters(args.run)
    index = lab.refresh_index(
        logs_dir=args.logs_dir,
        index_path=args.index,
        competition=args.competition,
        runs=run_filters or None,
        reindex=args.reindex,
    )
    records = lab.filter_records_by_run(list(index.get("records", [])), run_filters)
    records = [
        record
        for record in records
        if str(record.get("competition") or args.competition) == args.competition
        and (args.include_blends or not is_blend_record(record))
    ]
    all_records_with_scores = enrich_records_with_registry(
        list(index.get("records", [])),
        registry,
        competition=args.competition,
    )
    records = enrich_records_with_registry(records, registry, competition=args.competition)
    bad_components = (
        []
        if args.ignore_bad_public_blends
        else bad_blend_component_sets(
            all_records_with_scores,
            public_margin=args.bad_blend_public_margin,
        )
    )
    sample_path = sample_submission_path(args.data_dir)
    sample = pd.read_csv(sample_path, compression="infer", nrows=1)
    id_col = str(sample.columns[0])
    target_col = str(sample.columns[1])
    metric_name = next(
        (str(record.get("eval_metric")) for record in records if record.get("eval_metric")),
        "balanced_accuracy",
    )
    console.print(
        f"Evaluating auto blend candidates "
        f"(top_k={args.top_k}, count={args.count}, jobs={args.jobs}, "
        f"bad_blends={len(bad_components)})..."
    )
    progress_task: Any = None
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        def update_progress(done: int, total: int, kind: str) -> None:
            nonlocal progress_task
            if progress_task is None:
                progress_task = progress.add_task(
                    "Evaluating blend candidates",
                    total=total,
                )
            progress.update(
                progress_task,
                completed=done,
                total=total,
                description=f"Evaluating {kind} blend candidates",
            )

        selected = select_new_blends(
            records,
            count=args.count,
            existing_submission_sha256=existing_submission_sha256(index, registry),
            existing_recipe_hashes=existing_recipe_hashes(index),
            id_col=id_col,
            target_col=target_col,
            metric_name=metric_name,
            top_k=args.top_k,
            jobs=args.jobs,
            bad_component_sets=bad_components,
            bad_blend_overlap=args.bad_blend_overlap,
            progress_callback=update_progress,
        )
    if not selected:
        console.print("[yellow]No new blend artifacts selected.[/yellow]")
        return 1
    if args.dry_run:
        for result in selected:
            console.print(
                f"dry-run {result.spec.blend_kind} {result.spec.weighting} "
                f"{result.spec.mode} recipe={result.spec.recipe_hash} "
                f"score={result.score if result.score is not None else 'n/a'}"
            )
        return 0

    existing_steps = []
    for record in index.get("records", []):
        if str(record.get("run")) != args.output_run:
            continue
        try:
            existing_steps.append(int(record.get("step")))
        except (TypeError, ValueError):
            pass
    next_step = max(existing_steps, default=0) + 1
    created = []
    for offset, result in enumerate(selected):
        artifact = write_blend_artifact(
            result=result,
            args=args,
            step=next_step + offset,
            sample_path=sample_path,
            id_col=id_col,
            target_col=target_col,
            metric_name=metric_name,
        )
        created.append((result, artifact))
    index = lab.refresh_index(
        logs_dir=args.logs_dir,
        index_path=args.index,
        competition=args.competition,
        runs=[args.output_run],
        reindex=True,
    )
    created_shas = []
    for _result, artifact in created:
        matching = [
            record
            for record in index.get("records", [])
            if str(record.get("artifact_dir")) == artifact and record.get("sha256")
        ]
        if matching:
            created_shas.append(str(matching[0]["sha256"])[:10])
    render_created(console, created)
    console.print(format_submit_command(created_shas))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
