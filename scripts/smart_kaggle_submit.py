from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from aide.utils.submission_validation import (
    validate_submission_file as validate_submission_against_sample,
)
from aide.utils.path_portability import sanitize_persisted_payload, to_portable_path
from aide.utils.prediction_similarity import (
    DEFAULT_PREDICTION_ROUND_DECIMALS,
    DEFAULT_PREDICTION_SIMILARITY_MIN_COMMON_SAMPLE_SIZE,
    DEFAULT_PREDICTION_SIMILARITY_RMSE_THRESHOLD,
    DEFAULT_PREDICTION_SIMILARITY_SAMPLE_SIZE,
    DEFAULT_SCORE_ROUND_DECIMALS,
    submission_prediction_rmse,
)
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


DEFAULT_COMPETITION = "playground-series-s6e6"
DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_REGISTRY = Path("logs/submission_registry.json")


@dataclass(frozen=True)
class Candidate:
    competition: str
    run: str
    step: int
    node_id: str
    parent_node_id: str | None
    ancestor_node_ids: tuple[str, ...]
    timestamp: str
    ctime: float
    local_score: float | None
    metric_maximize: bool | None
    is_buggy: bool
    submission_path: Path
    sha256: str | None
    plan: str | None = None
    analysis: str | None = None
    validation_error: str | None = None
    algo: str | None = None
    eval_metric: str | None = None
    hypothesis_id: str | None = None
    source_sha256: str | None = None

    @property
    def is_submit_ready(self) -> bool:
        return (
            not self.is_buggy
            and self.local_score is not None
            and self.submission_path.exists()
            and self.sha256 is not None
            and self.validation_error is None
        )

    @property
    def status(self) -> str:
        return "ready" if self.is_submit_ready else "not-ready"


class SubmissionRegistry:
    def __init__(self, path: Path, entries: list[dict[str, Any]] | None = None):
        self.path = path
        self.entries = entries or []

    @classmethod
    def load(cls, path: Path) -> "SubmissionRegistry":
        if not path.exists():
            return cls(path)
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            entries = data.get("submissions", [])
        else:
            entries = data
        return cls(path, list(entries))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = sanitize_persisted_payload({"submissions": self.entries})
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def add(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        self.save()

    def is_submitted(
        self,
        *,
        competition: str,
        sha256: str | None,
        run: str,
        step: int,
        timestamp: str,
    ) -> bool:
        for entry in self.entries:
            if entry.get("competition") != competition:
                continue
            if sha256 is not None and _sha256_matches(entry.get("sha256"), sha256):
                return True
            if (
                entry.get("run") == run
                and entry.get("step") == step
                and entry.get("timestamp") == timestamp
            ):
                return True
        return False


def _sha256_matches(left: Any, right: Any, *, min_prefix_len: int = 10) -> bool:
    left_text = str(left or "").strip().lower()
    right_text = str(right or "").strip().lower()
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    if min(len(left_text), len(right_text)) < min_prefix_len:
        return False
    return left_text.startswith(right_text) or right_text.startswith(left_text)


def _timestamp_from_ctime(ctime: float) -> str:
    return dt.datetime.fromtimestamp(ctime).strftime("%Y%m%dT%H%M%S")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_candidate(candidate: Candidate, **updates: Any) -> Candidate:
    values = candidate.__dict__.copy()
    values.update(updates)
    return Candidate(**values)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _metric_value(node: dict[str, Any]) -> tuple[float | None, bool | None]:
    metric = node.get("metric") or {}
    value = metric.get("value")
    maximize = metric.get("maximize")
    if value is None:
        return None, maximize
    return float(value), maximize


def _metric_name(node: dict[str, Any]) -> str | None:
    metric = node.get("metric") or {}
    name = metric.get("name") or node.get("eval_metric")
    return str(name) if name else None


def _ancestor_node_ids(
    node_id: str,
    node2parent: dict[str, str],
) -> tuple[str, ...]:
    ancestors: list[str] = []
    seen = {node_id}
    parent_id = node2parent.get(node_id)
    while parent_id and parent_id not in seen:
        ancestors.append(parent_id)
        seen.add(parent_id)
        parent_id = node2parent.get(parent_id)
    return tuple(ancestors)


def collect_candidates(
    logs_dir: Path,
    *,
    competition: str = DEFAULT_COMPETITION,
    since: dt.datetime | None = None,
    progress: Any | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    journal_records = [
        (journal_path, _load_json(journal_path))
        for journal_path in sorted(logs_dir.glob("*/journal.json"))
    ]
    task_id = None
    if progress is not None:
        total = sum(len(journal.get("nodes", [])) for _, journal in journal_records)
        task_id = progress.add_task("Scanning local AIDE nodes", total=total)

    for journal_path, journal in journal_records:
        run = journal_path.parent.name
        node2parent = journal.get("node2parent") or {}
        for node in journal.get("nodes", []):
            try:
                ctime = node.get("ctime")
                if ctime is None:
                    continue
                if since is not None and dt.datetime.fromtimestamp(ctime) < since:
                    continue

                timestamp = _timestamp_from_ctime(float(ctime))
                submission_path = (
                    journal_path.parent / "artifacts" / timestamp / "submission.csv"
                )
                score, maximize = _metric_value(node)
                checksum = _sha256(submission_path) if submission_path.exists() else None
                node_id = str(node.get("id", ""))
                candidates.append(
                    Candidate(
                        competition=competition,
                        run=run,
                        step=int(node.get("step", -1)),
                        node_id=node_id,
                        parent_node_id=node2parent.get(node_id),
                        ancestor_node_ids=_ancestor_node_ids(node_id, node2parent),
                        timestamp=timestamp,
                        ctime=float(ctime),
                        local_score=score,
                        metric_maximize=maximize,
                        is_buggy=bool(node.get("is_buggy")),
                        submission_path=submission_path,
                        sha256=checksum,
                        plan=node.get("plan"),
                        analysis=node.get("analysis"),
                        eval_metric=_metric_name(node),
                    )
                )
            finally:
                if progress is not None and task_id is not None:
                    progress.advance(task_id)
    return candidates


def _sample_submission_candidates(candidate: Candidate, data_dir: Path | None) -> Iterable[Path]:
    if data_dir is not None:
        yield data_dir / "sample_submission.csv.gz"
        yield data_dir / "sample_submission.csv"

    run_dir = candidate.submission_path.parents[2]
    repo_root = run_dir.parent.parent
    input_dir = repo_root / "workspaces" / candidate.run / "input"
    yield input_dir / "sample_submission.csv.gz"
    yield input_dir / "sample_submission.csv"

    example_dir = repo_root / "aide" / "example_tasks" / candidate.competition
    yield example_dir / "sample_submission.csv.gz"
    yield example_dir / "sample_submission.csv"


def _find_sample_submission(candidate: Candidate, data_dir: Path | None) -> Path | None:
    for path in _sample_submission_candidates(candidate, data_dir):
        if path.exists():
            return path
    return None


def validate_submission_file(candidate: Candidate, data_dir: Path | None) -> str | None:
    sample_path = _find_sample_submission(candidate, data_dir)
    if sample_path is None:
        return "missing sample_submission for validation"
    return validate_submission_against_sample(candidate.submission_path, sample_path)


def validate_candidates(
    candidates: Iterable[Candidate],
    *,
    data_dir: Path | None,
    progress: Any | None = None,
) -> list[Candidate]:
    validated: list[Candidate] = []
    candidate_list = list(candidates)
    task_id = None
    if progress is not None:
        task_id = progress.add_task(
            "Validating local submissions",
            total=len(candidate_list),
        )
    for candidate in candidate_list:
        try:
            if (
                candidate.is_buggy
                or candidate.local_score is None
                or not candidate.submission_path.exists()
                or candidate.sha256 is None
            ):
                validated.append(candidate)
                continue
            validation_error = validate_submission_file(candidate, data_dir)
            validated.append(
                _replace_candidate(candidate, validation_error=validation_error)
            )
        finally:
            if progress is not None and task_id is not None:
                progress.advance(task_id)
    return validated


def _sort_key(candidate: Candidate) -> tuple[float, float]:
    if candidate.local_score is None:
        metric = float("-inf")
    elif candidate.metric_maximize is False:
        metric = -candidate.local_score
    else:
        metric = candidate.local_score
    return metric, candidate.ctime


def _rounded_score(candidate: Candidate, score_round_decimals: int) -> float | None:
    if candidate.local_score is None:
        return None
    return round(candidate.local_score, score_round_decimals)


def _normalized_score(candidate: Candidate) -> float | None:
    if candidate.local_score is None:
        return None
    if candidate.metric_maximize is False:
        return -candidate.local_score
    return candidate.local_score


def _lineage_node_ids(candidate: Candidate) -> set[str]:
    return {candidate.node_id, *candidate.ancestor_node_ids}


def _are_in_same_branch_family(left: Candidate, right: Candidate) -> bool:
    if left.run != right.run:
        return False
    return bool(_lineage_node_ids(left) & _lineage_node_ids(right))


def _has_ancestor_relation(left: Candidate, right: Candidate) -> bool:
    return (
        left.node_id in right.ancestor_node_ids
        or right.node_id in left.ancestor_node_ids
    )


def _rounded_normalized_score(
    candidate: Candidate,
    score_round_decimals: int,
) -> float | None:
    candidate_score = _normalized_score(candidate)
    if candidate_score is None:
        return None
    return round(candidate_score, score_round_decimals)


def _is_strictly_worse_than(
    candidate: Candidate,
    ancestor: Candidate,
    *,
    score_round_decimals: int,
) -> bool:
    candidate_score = _rounded_normalized_score(candidate, score_round_decimals)
    ancestor_score = _rounded_normalized_score(ancestor, score_round_decimals)
    if candidate_score is None or ancestor_score is None:
        return False
    return candidate_score < ancestor_score


def _has_same_rounded_score(
    left: Candidate,
    right: Candidate,
    *,
    score_round_decimals: int,
) -> bool:
    return _rounded_score(left, score_round_decimals) == _rounded_score(
        right,
        score_round_decimals,
    )


def _has_similar_predictions(
    left: Candidate,
    right: Candidate,
    *,
    prediction_round_decimals: int,
    prediction_similarity_sample_size: int,
    prediction_similarity_min_common_sample_size: int,
    prediction_similarity_rmse_threshold: float,
) -> bool:
    rmse = submission_prediction_rmse(
        left.submission_path,
        right.submission_path,
        prediction_round_decimals=prediction_round_decimals,
        sample_size=prediction_similarity_sample_size,
        min_common_sample_size=prediction_similarity_min_common_sample_size,
    )
    return rmse is not None and rmse <= prediction_similarity_rmse_threshold


def _should_collapse_related_candidate(
    candidate: Candidate,
    selected_candidate: Candidate,
    *,
    score_round_decimals: int,
    prediction_round_decimals: int,
    prediction_similarity_sample_size: int,
    prediction_similarity_min_common_sample_size: int,
    prediction_similarity_rmse_threshold: float,
) -> bool:
    if _has_ancestor_relation(candidate, selected_candidate):
        if selected_candidate.node_id in candidate.ancestor_node_ids:
            ancestor = selected_candidate
            descendant = candidate
        else:
            ancestor = candidate
            descendant = selected_candidate

        if _is_strictly_worse_than(
            descendant,
            ancestor,
            score_round_decimals=score_round_decimals,
        ):
            return True

    return _has_same_rounded_score(
        candidate,
        selected_candidate,
        score_round_decimals=score_round_decimals,
    ) and _has_similar_predictions(
        candidate,
        selected_candidate,
        prediction_round_decimals=prediction_round_decimals,
        prediction_similarity_sample_size=prediction_similarity_sample_size,
        prediction_similarity_min_common_sample_size=(
            prediction_similarity_min_common_sample_size
        ),
        prediction_similarity_rmse_threshold=prediction_similarity_rmse_threshold,
    )


def _prefer_ancestor_for_duplicate_related(
    candidates: list[Candidate],
    *,
    score_round_decimals: int,
    prediction_round_decimals: int,
    prediction_similarity_sample_size: int,
    prediction_similarity_min_common_sample_size: int,
    prediction_similarity_rmse_threshold: float,
    progress: Any | None = None,
) -> list[Candidate]:
    selected: list[Candidate] = []
    task_id = None
    if progress is not None:
        task_id = progress.add_task(
            "Filtering related submissions",
            total=len(candidates),
        )
    for candidate in candidates:
        try:
            related_indices = []
            for idx, selected_candidate in enumerate(list(selected)):
                if not _are_in_same_branch_family(candidate, selected_candidate):
                    continue
                if not _should_collapse_related_candidate(
                    candidate,
                    selected_candidate,
                    score_round_decimals=score_round_decimals,
                    prediction_round_decimals=prediction_round_decimals,
                    prediction_similarity_sample_size=prediction_similarity_sample_size,
                    prediction_similarity_min_common_sample_size=(
                        prediction_similarity_min_common_sample_size
                    ),
                    prediction_similarity_rmse_threshold=(
                        prediction_similarity_rmse_threshold
                    ),
                ):
                    continue
                related_indices.append(idx)

            if not related_indices:
                selected.append(candidate)
                continue

            if any(
                selected[idx].node_id in candidate.ancestor_node_ids
                for idx in related_indices
            ):
                continue

            if any(
                candidate.node_id in selected[idx].ancestor_node_ids
                for idx in related_indices
            ):
                selected = [
                    selected_candidate
                    for idx, selected_candidate in enumerate(selected)
                    if idx not in set(related_indices)
                ]
                selected.append(candidate)
        finally:
            if progress is not None and task_id is not None:
                progress.advance(task_id)
    return selected


def select_top_unsent_ready(
    candidates: Iterable[Candidate],
    *,
    registry: SubmissionRegistry,
    competition: str,
    limit: int,
    include_related: bool = False,
    score_round_decimals: int = DEFAULT_SCORE_ROUND_DECIMALS,
    prediction_round_decimals: int = DEFAULT_PREDICTION_ROUND_DECIMALS,
    prediction_similarity_sample_size: int = (
        DEFAULT_PREDICTION_SIMILARITY_SAMPLE_SIZE
    ),
    prediction_similarity_min_common_sample_size: int = (
        DEFAULT_PREDICTION_SIMILARITY_MIN_COMMON_SAMPLE_SIZE
    ),
    prediction_similarity_rmse_threshold: float = (
        DEFAULT_PREDICTION_SIMILARITY_RMSE_THRESHOLD
    ),
    progress: Any | None = None,
) -> list[Candidate]:
    ready = sorted(
        [
            candidate
            for candidate in candidates
            if candidate.is_submit_ready
        ],
        key=_sort_key,
        reverse=True,
    )
    if not include_related:
        ready = _prefer_ancestor_for_duplicate_related(
            ready,
            score_round_decimals=score_round_decimals,
            prediction_round_decimals=prediction_round_decimals,
            prediction_similarity_sample_size=prediction_similarity_sample_size,
            prediction_similarity_min_common_sample_size=(
                prediction_similarity_min_common_sample_size
            ),
            prediction_similarity_rmse_threshold=prediction_similarity_rmse_threshold,
            progress=progress,
        )

    unsent_ready = [
        candidate
        for candidate in ready
        if not registry.is_submitted(
            competition=competition,
            sha256=candidate.sha256,
            run=candidate.run,
            step=candidate.step,
            timestamp=candidate.timestamp,
        )
    ]
    return unsent_ready[:limit]


def parse_sha256_filters(values: list[str] | None) -> list[str]:
    filters: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip().lower()
            if not item:
                continue
            if not re.fullmatch(r"[0-9a-f]+", item):
                raise ValueError(
                    f"Invalid sha256 prefix '{item}'; use hexadecimal characters."
                )
            filters.append(item)
    return filters


def filter_candidates_by_sha256(
    candidates: list[Candidate],
    sha256_filters: list[str],
) -> list[Candidate]:
    if not sha256_filters:
        return candidates

    selected: list[Candidate] = []
    selected_hashes: set[str] = set()
    for sha_filter in sha256_filters:
        matches = [
            candidate
            for candidate in candidates
            if (candidate.sha256 or "").lower().startswith(sha_filter)
        ]
        if not matches:
            raise ValueError(
                f"No submit-ready candidate matches sha256 prefix: {sha_filter}"
            )

        matched_hashes = {candidate.sha256 for candidate in matches}
        if len(matched_hashes) > 1:
            preview = ", ".join(sorted(str(value)[:10] for value in matched_hashes))
            raise ValueError(
                f"Ambiguous sha256 prefix {sha_filter}; matches: {preview}"
            )

        candidate = matches[0]
        if candidate.sha256 not in selected_hashes:
            selected.append(candidate)
            if candidate.sha256 is not None:
                selected_hashes.add(candidate.sha256)

    return selected


def _not_ready_reason(candidate: Candidate) -> str:
    reasons = []
    if candidate.is_buggy:
        reasons.append("buggy")
    if candidate.local_score is None:
        reasons.append("no-score")
    if not candidate.submission_path.exists():
        reasons.append("missing-submission")
    if candidate.sha256 is None:
        reasons.append("missing-sha256")
    if candidate.validation_error is not None:
        reasons.append(f"invalid-submission: {candidate.validation_error}")
    return ", ".join(reasons) or "unknown"


def require_explicit_submit_ready_candidates(
    candidates: list[Candidate],
    *,
    registry: SubmissionRegistry,
    competition: str,
) -> list[Candidate]:
    selected: list[Candidate] = []
    for candidate in candidates:
        sha = (candidate.sha256 or "")[:10]
        if not candidate.is_submit_ready:
            raise ValueError(
                f"Candidate {sha} is not submit-ready: {_not_ready_reason(candidate)}"
            )
        if registry.is_submitted(
            competition=competition,
            sha256=candidate.sha256,
            run=candidate.run,
            step=candidate.step,
            timestamp=candidate.timestamp,
        ):
            raise ValueError(f"Candidate {sha} is already submitted.")
        selected.append(candidate)
    return selected


def build_kaggle_message(candidate: Candidate) -> str:
    score = "nan" if candidate.local_score is None else f"{candidate.local_score:.5f}"
    node = candidate.node_id[:8] if candidate.node_id else "unknown"
    algo = f" | algo={candidate.algo}" if candidate.algo else ""
    metric = f" | metric={candidate.eval_metric}" if candidate.eval_metric else ""
    return (
        f"cv={score} | run={candidate.run} | step={candidate.step} | "
        f"aide_ts={candidate.timestamp} | node={node} | "
        f"sha={(candidate.sha256 or '')[:10]}{algo}{metric}"
    )


def build_upload_filename(candidate: Candidate) -> str:
    score = "nan" if candidate.local_score is None else f"{candidate.local_score:.5f}"
    node = candidate.node_id[:8] if candidate.node_id else "unknown"
    sha = (candidate.sha256 or "nohash")[:10]
    return (
        f"sub_{candidate.timestamp}_step-{candidate.step}_node-{node}_"
        f"sha-{sha}_cv-{score}.csv"
    )


def prepare_upload_file(candidate: Candidate) -> Path:
    upload_path = candidate.submission_path.with_name(build_upload_filename(candidate))
    if upload_path != candidate.submission_path:
        shutil.copy2(candidate.submission_path, upload_path)
    return upload_path


def _response_to_jsonable(response: Any) -> Any:
    if response is None:
        return None
    if isinstance(response, (str, int, float, bool, list, dict)):
        return response
    if hasattr(response, "to_dict"):
        return response.to_dict()
    if hasattr(response, "__dict__"):
        return dict(response.__dict__)
    return repr(response)


def submit_candidates(
    candidates: Iterable[Candidate],
    *,
    registry: SubmissionRegistry,
    client: Any,
    competition: str,
) -> list[dict[str, Any]]:
    submitted: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.validation_error is not None:
            continue
        if registry.is_submitted(
            competition=competition,
            sha256=candidate.sha256,
            run=candidate.run,
            step=candidate.step,
            timestamp=candidate.timestamp,
        ):
            continue
        message = build_kaggle_message(candidate)
        upload_path = prepare_upload_file(candidate)
        response = client.competition_submit(
            str(upload_path),
            message,
            competition,
            quiet=False,
        )
        entry = {
            "competition": competition,
            "run": candidate.run,
            "step": candidate.step,
            "node_id": candidate.node_id,
            "timestamp": candidate.timestamp,
            "local_score": candidate.local_score,
            "metric_maximize": candidate.metric_maximize,
            "eval_metric": candidate.eval_metric,
            "submission_path": to_portable_path(candidate.submission_path),
            "upload_path": to_portable_path(upload_path),
            "uploaded_filename": upload_path.name,
            "sha256": candidate.sha256,
            "hypothesis_id": candidate.hypothesis_id,
            "source_sha256": candidate.source_sha256,
            "algo": candidate.algo,
            "kaggle_message": message,
            "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "response": _response_to_jsonable(response),
        }
        registry.add(entry)
        submitted.append(entry)
    return submitted


def _build_kaggle_client() -> Any:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def fetch_remote_submissions(client: Any, competition: str) -> list[Any]:
    submissions = client.competition_submissions(competition, page_size=200)
    return list(submissions or [])


def _remote_attr(remote: Any, snake_name: str) -> Any:
    if isinstance(remote, dict):
        return remote.get(snake_name) or remote.get(_snake_to_camel(snake_name))
    if hasattr(remote, snake_name):
        return getattr(remote, snake_name)
    private_name = f"_{snake_name}"
    if hasattr(remote, private_name):
        return getattr(remote, private_name)
    camel_name = _snake_to_camel(snake_name)
    if hasattr(remote, camel_name):
        return getattr(remote, camel_name)
    return None


def _snake_to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def _status_to_string(status: Any) -> str | None:
    if status is None:
        return None
    if hasattr(status, "name"):
        return str(status.name)
    return str(status)


def _date_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def parse_submission_description(description: str | None) -> dict[str, str]:
    if not description:
        return {}
    parsed: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=([^|]+)", description):
        key = match.group(1).strip()
        value = match.group(2).strip()
        parsed[key] = value
    if "aide_ts" in parsed and "timestamp" not in parsed:
        parsed["timestamp"] = parsed["aide_ts"]
    if "ts" in parsed and "timestamp" not in parsed:
        parsed["timestamp"] = parsed["ts"]
    return parsed


def _truthy_description_value(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _description_marks_local_invalid(parsed: dict[str, str]) -> bool:
    status = str(
        parsed.get("status")
        or parsed.get("manual_status")
        or parsed.get("local_status")
        or ""
    ).strip().lower()
    return (
        _truthy_description_value(parsed.get("ignore"))
        or _truthy_description_value(parsed.get("invalid"))
        or status in {"failed", "invalid", "ignore", "ignored", "failed_local_invalid"}
    )


def _entry_ref(entry: dict[str, Any]) -> int | None:
    ref = entry.get("kaggle_ref")
    if ref is None and isinstance(entry.get("response"), dict):
        ref = entry["response"].get("ref")
    return int(ref) if ref is not None else None


def _remote_ref(remote: Any) -> int | None:
    ref = _remote_attr(remote, "ref")
    return int(ref) if ref is not None else None


def _remote_matches_entry(
    remote: Any,
    *,
    entry: dict[str, Any],
    parsed_description: dict[str, str],
) -> bool:
    entry_ref = _entry_ref(entry)
    remote_ref = _remote_ref(remote)
    if entry_ref is not None and remote_ref is not None:
        return entry_ref == remote_ref

    timestamp = parsed_description.get("timestamp")
    if timestamp is None or timestamp != entry.get("timestamp"):
        return False

    run = parsed_description.get("run")
    if run is not None and run != entry.get("run"):
        return False

    step = parsed_description.get("step")
    if step is not None and str(step) != str(entry.get("step")):
        return False

    node = parsed_description.get("node")
    node_id = entry.get("node_id")
    if node is not None and node_id is not None:
        return str(node_id).startswith(node)

    return True


def _remote_registry_fields(remote: Any) -> dict[str, Any]:
    description = _remote_attr(remote, "description")
    parsed = parse_submission_description(description)
    fields = {
        "kaggle_ref": _remote_ref(remote),
        "remote_filename": _remote_attr(remote, "file_name"),
        "remote_date": _date_to_string(_remote_attr(remote, "date")),
        "remote_description": description,
        "remote_status": _status_to_string(_remote_attr(remote, "status")),
        "public_score": _remote_attr(remote, "public_score"),
        "private_score": _remote_attr(remote, "private_score"),
        "remote_url": _remote_attr(remote, "url"),
        "remote_total_bytes": _remote_attr(remote, "total_bytes"),
        "synced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if _description_marks_local_invalid(parsed):
        reason = parsed.get("reason") or parsed.get("invalid_reason") or "marked ignored in Kaggle description"
        fields.update(
            {
                "local_score": None,
                "public_score": "",
                "private_score": "",
                "remote_status": "FAILED_LOCAL_INVALID",
                "manual_status": "failed",
                "manual_invalid_reason": reason,
            }
        )
    return fields


def sync_registry_from_remote(
    *,
    registry: SubmissionRegistry,
    competition: str,
    remote_submissions: Iterable[Any],
) -> int:
    changed = 0
    for remote in remote_submissions:
        description = _remote_attr(remote, "description")
        parsed = parse_submission_description(description)
        for entry in registry.entries:
            if entry.get("competition") != competition:
                continue
            if not _remote_matches_entry(
                remote,
                entry=entry,
                parsed_description=parsed,
            ):
                continue
            if entry.get("manual_status") == "failed":
                break

            fields = _remote_registry_fields(remote)
            if fields.get("manual_status") == "failed":
                entry.setdefault("original_local_score", entry.get("local_score"))
                entry.setdefault("original_public_score", entry.get("public_score"))
            if any(entry.get(key) != value for key, value in fields.items()):
                entry.update(fields)
                changed += 1
            break
    if changed:
        registry.save()
    return changed


def _format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.5f}"


def _format_submission_date(timestamp: Any) -> str:
    text = str(timestamp or "")
    return text[:8] if len(text) >= 8 else text


def _parse_public_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _registry_sort_key(entry: dict[str, Any]) -> tuple[bool, float, str]:
    public_score = _parse_public_score(entry.get("public_score"))
    return (
        public_score is not None,
        public_score if public_score is not None else float("-inf"),
        str(entry.get("submitted_at") or entry.get("remote_date") or ""),
    )


def render_dry_run(
    *,
    console: Console,
    candidates: list[Candidate],
    selected: list[Candidate],
    registry: SubmissionRegistry,
    include_not_ready: bool,
    remote_submissions: list[Any] | None,
) -> None:
    table = Table(title="Top unsent submit-ready candidates")
    table.add_column("rank", justify="right")
    table.add_column("cv", justify="right")
    table.add_column("run")
    table.add_column("step", justify="right")
    table.add_column("timestamp", no_wrap=True)
    table.add_column("node", no_wrap=True)
    table.add_column("sha256", no_wrap=True)
    for rank, candidate in enumerate(selected, start=1):
        table.add_row(
            str(rank),
            _format_score(candidate.local_score),
            candidate.run,
            str(candidate.step),
            candidate.timestamp,
            candidate.node_id[:8],
            (candidate.sha256 or "")[:10],
        )
    console.print(table)

    invalid = [
        candidate
        for candidate in candidates
        if candidate.validation_error is not None
    ]
    if invalid:
        invalid_table = Table(title="Invalid local submissions skipped")
        invalid_table.add_column("cv", justify="right")
        invalid_table.add_column("reason")
        invalid_table.add_column("run")
        invalid_table.add_column("step", justify="right")
        invalid_table.add_column("timestamp", no_wrap=True)
        for candidate in sorted(invalid, key=_sort_key, reverse=True)[:10]:
            invalid_table.add_row(
                _format_score(candidate.local_score),
                candidate.validation_error or "",
                candidate.run,
                str(candidate.step),
                candidate.timestamp,
            )
        console.print(invalid_table)

    if include_not_ready:
        not_ready = [
            candidate for candidate in candidates if not candidate.is_submit_ready
        ]
        details = Table(title="Not submit-ready local results")
        details.add_column("cv", justify="right")
        details.add_column("reason")
        details.add_column("run")
        details.add_column("step", justify="right")
        details.add_column("timestamp")
        for candidate in not_ready:
            reasons = []
            if candidate.is_buggy:
                reasons.append("buggy")
            if candidate.local_score is None:
                reasons.append("no-score")
            if not candidate.submission_path.exists():
                reasons.append("missing-submission")
            if candidate.validation_error is not None:
                reasons.append(f"invalid-submission: {candidate.validation_error}")
            details.add_row(
                _format_score(candidate.local_score),
                ",".join(reasons) or "unknown",
                candidate.run,
                str(candidate.step),
                candidate.timestamp,
            )
        console.print(details)

    submitted = Table(title="Local submission registry")
    submitted.add_column("#", justify="right")
    submitted.add_column("cv", justify="right")
    submitted.add_column("metric")
    submitted.add_column("public", justify="right")
    submitted.add_column("status")
    submitted.add_column("run")
    submitted.add_column("step", justify="right")
    submitted.add_column("date")
    complete_rank = 0
    for entry in sorted(registry.entries, key=_registry_sort_key, reverse=True):
        remote_status = str(entry.get("remote_status") or "")
        if remote_status.upper() == "COMPLETE":
            complete_rank += 1
            display_rank = str(complete_rank)
        else:
            display_rank = "-"
        submitted.add_row(
            display_rank,
            _format_score(entry.get("local_score")),
            str(entry.get("eval_metric") or "-"),
            str(entry.get("public_score") or ""),
            remote_status,
            str(entry.get("run", "")),
            str(entry.get("step", "")),
            _format_submission_date(entry.get("timestamp")),
        )
    console.print(submitted)

    if remote_submissions is not None:
        console.print(f"Remote Kaggle submissions visible: {len(remote_submissions)}")


def parse_since(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run by default; submit top local AIDE submissions only with --submit."
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing sample_submission.csv(.gz) for submission validation.",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--since", help="Only consider nodes created on or after YYYY-MM-DD."
    )
    parser.add_argument(
        "--submit", action="store_true", help="Actually submit to Kaggle."
    )
    parser.add_argument(
        "--sha256",
        "--sha",
        dest="sha256",
        action="append",
        default=[],
        metavar="PREFIX",
        help=(
            "Only show or submit candidates matching a SHA-256 prefix. "
            "Repeat the option or pass comma-separated prefixes."
        ),
    )
    parser.add_argument(
        "--include-not-ready",
        action="store_true",
        help="Show scored or buggy local nodes that cannot be submitted.",
    )
    parser.add_argument(
        "--include-related",
        action="store_true",
        help=(
            "Include related branch candidates that are normally hidden by "
            "dominance or prediction-similarity filtering."
        ),
    )
    parser.add_argument(
        "--score-round-decimals",
        type=int,
        default=DEFAULT_SCORE_ROUND_DECIMALS,
        help="Decimal places used to group local CV scores for related candidates.",
    )
    parser.add_argument(
        "--prediction-round-decimals",
        type=int,
        default=DEFAULT_PREDICTION_ROUND_DECIMALS,
        help="Decimal places used before comparing submission predictions.",
    )
    parser.add_argument(
        "--prediction-similarity-sample-size",
        type=int,
        default=DEFAULT_PREDICTION_SIMILARITY_SAMPLE_SIZE,
        help="Number of leading submission rows sampled for prediction similarity.",
    )
    parser.add_argument(
        "--prediction-similarity-min-common-sample-size",
        type=int,
        default=DEFAULT_PREDICTION_SIMILARITY_MIN_COMMON_SAMPLE_SIZE,
        help="Minimum common ids required in the sampled rows before using RMSE.",
    )
    parser.add_argument(
        "--prediction-similarity-rmse-threshold",
        type=float,
        default=DEFAULT_PREDICTION_SIMILARITY_RMSE_THRESHOLD,
        help="RMSE threshold below which related submissions are treated as similar.",
    )
    return parser.parse_args(argv)


def _build_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
    try:
        sha256_filters = parse_sha256_filters(args.sha256)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    registry = SubmissionRegistry.load(args.registry)
    remote_submissions = None
    try:
        client = _build_kaggle_client()
        remote_submissions = fetch_remote_submissions(client, args.competition)
        synced = sync_registry_from_remote(
            registry=registry,
            competition=args.competition,
            remote_submissions=remote_submissions,
        )
        if synced:
            console.print(f"Synchronized {synced} Kaggle submission(s).")
    except Exception as exc:
        client = None
        console.print(f"Remote Kaggle submissions unavailable: {exc}")

    with _build_progress(console) as progress:
        candidates = collect_candidates(
            args.logs_dir,
            competition=args.competition,
            since=parse_since(args.since),
            progress=progress,
        )
        if sha256_filters:
            try:
                selected = filter_candidates_by_sha256(candidates, sha256_filters)
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                return 2
            selected = validate_candidates(
                selected,
                data_dir=args.data_dir,
                progress=progress,
            )
            try:
                selected = require_explicit_submit_ready_candidates(
                    selected,
                    registry=registry,
                    competition=args.competition,
                )
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")
                return 2
        else:
            candidates = validate_candidates(
                candidates,
                data_dir=args.data_dir,
                progress=progress,
            )
            selected = select_top_unsent_ready(
                candidates,
                registry=registry,
                competition=args.competition,
                limit=args.limit,
                include_related=args.include_related,
                score_round_decimals=args.score_round_decimals,
                prediction_round_decimals=args.prediction_round_decimals,
                prediction_similarity_sample_size=(
                    args.prediction_similarity_sample_size
                ),
                prediction_similarity_min_common_sample_size=(
                    args.prediction_similarity_min_common_sample_size
                ),
                prediction_similarity_rmse_threshold=(
                    args.prediction_similarity_rmse_threshold
                ),
                progress=progress,
            )

    if args.submit:
        if client is None:
            client = _build_kaggle_client()
        submitted = submit_candidates(
            track(selected, description="Submitting to Kaggle"),
            registry=registry,
            client=client,
            competition=args.competition,
        )
        console.print(f"Submitted {len(submitted)} candidate(s).")
    else:
        render_dry_run(
            console=console,
            candidates=candidates,
            selected=selected,
            registry=registry,
            include_not_ready=args.include_not_ready,
            remote_submissions=remote_submissions,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
