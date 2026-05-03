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

from rich.console import Console
from rich.progress import track
from rich.table import Table


DEFAULT_COMPETITION = "playground-series-s6e5"
DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_REGISTRY = Path("logs/submission_registry.json")


@dataclass(frozen=True)
class Candidate:
    competition: str
    run: str
    step: int
    node_id: str
    timestamp: str
    ctime: float
    local_score: float | None
    metric_maximize: bool | None
    is_buggy: bool
    submission_path: Path
    sha256: str | None
    plan: str | None = None
    analysis: str | None = None

    @property
    def is_submit_ready(self) -> bool:
        return (
            not self.is_buggy
            and self.local_score is not None
            and self.submission_path.exists()
            and self.sha256 is not None
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
        payload = {"submissions": self.entries}
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
            if sha256 is not None and entry.get("sha256") == sha256:
                return True
            if (
                entry.get("run") == run
                and entry.get("step") == step
                and entry.get("timestamp") == timestamp
            ):
                return True
        return False


def _timestamp_from_ctime(ctime: float) -> str:
    return dt.datetime.fromtimestamp(ctime).strftime("%Y%m%dT%H%M%S")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _metric_value(node: dict[str, Any]) -> tuple[float | None, bool | None]:
    metric = node.get("metric") or {}
    value = metric.get("value")
    maximize = metric.get("maximize")
    if value is None:
        return None, maximize
    return float(value), maximize


def collect_candidates(
    logs_dir: Path,
    *,
    competition: str = DEFAULT_COMPETITION,
    since: dt.datetime | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for journal_path in sorted(logs_dir.glob("*/journal.json")):
        run = journal_path.parent.name
        journal = _load_json(journal_path)
        for node in journal.get("nodes", []):
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
            candidates.append(
                Candidate(
                    competition=competition,
                    run=run,
                    step=int(node.get("step", -1)),
                    node_id=str(node.get("id", "")),
                    timestamp=timestamp,
                    ctime=float(ctime),
                    local_score=score,
                    metric_maximize=maximize,
                    is_buggy=bool(node.get("is_buggy")),
                    submission_path=submission_path,
                    sha256=checksum,
                    plan=node.get("plan"),
                    analysis=node.get("analysis"),
                )
            )
    return candidates


def _sort_key(candidate: Candidate) -> tuple[float, float]:
    if candidate.local_score is None:
        metric = float("-inf")
    elif candidate.metric_maximize is False:
        metric = -candidate.local_score
    else:
        metric = candidate.local_score
    return metric, candidate.ctime


def select_top_unsent_ready(
    candidates: Iterable[Candidate],
    *,
    registry: SubmissionRegistry,
    competition: str,
    limit: int,
) -> list[Candidate]:
    ready = [
        candidate
        for candidate in candidates
        if candidate.is_submit_ready
        and not registry.is_submitted(
            competition=competition,
            sha256=candidate.sha256,
            run=candidate.run,
            step=candidate.step,
            timestamp=candidate.timestamp,
        )
    ]
    return sorted(ready, key=_sort_key, reverse=True)[:limit]


def build_kaggle_message(candidate: Candidate) -> str:
    score = "nan" if candidate.local_score is None else f"{candidate.local_score:.5f}"
    node = candidate.node_id[:8] if candidate.node_id else "unknown"
    return (
        f"cv={score} | run={candidate.run} | step={candidate.step} | "
        f"aide_ts={candidate.timestamp} | node={node} | sha={(candidate.sha256 or '')[:10]}"
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
            "submission_path": str(candidate.submission_path),
            "upload_path": str(upload_path),
            "uploaded_filename": upload_path.name,
            "sha256": candidate.sha256,
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
    return {
        "kaggle_ref": _remote_ref(remote),
        "remote_filename": _remote_attr(remote, "file_name"),
        "remote_date": _date_to_string(_remote_attr(remote, "date")),
        "remote_description": _remote_attr(remote, "description"),
        "remote_status": _status_to_string(_remote_attr(remote, "status")),
        "public_score": _remote_attr(remote, "public_score"),
        "private_score": _remote_attr(remote, "private_score"),
        "remote_url": _remote_attr(remote, "url"),
        "remote_total_bytes": _remote_attr(remote, "total_bytes"),
        "synced_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


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

            fields = _remote_registry_fields(remote)
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
            details.add_row(
                _format_score(candidate.local_score),
                ",".join(reasons) or "unknown",
                candidate.run,
                str(candidate.step),
                candidate.timestamp,
            )
        console.print(details)

    submitted = Table(title="Local submission registry")
    submitted.add_column("cv", justify="right")
    submitted.add_column("public", justify="right")
    submitted.add_column("status")
    submitted.add_column("run")
    submitted.add_column("step", justify="right")
    submitted.add_column("date")
    for entry in sorted(registry.entries, key=_registry_sort_key, reverse=True):
        submitted.add_row(
            _format_score(entry.get("local_score")),
            str(entry.get("public_score") or ""),
            str(entry.get("remote_status") or ""),
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
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--since", help="Only consider nodes created on or after YYYY-MM-DD."
    )
    parser.add_argument(
        "--submit", action="store_true", help="Actually submit to Kaggle."
    )
    parser.add_argument(
        "--include-not-ready",
        action="store_true",
        help="Show scored or buggy local nodes that cannot be submitted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    console = Console()
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

    candidates = collect_candidates(
        args.logs_dir,
        competition=args.competition,
        since=parse_since(args.since),
    )
    selected = select_top_unsent_ready(
        candidates,
        registry=registry,
        competition=args.competition,
        limit=args.limit,
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
