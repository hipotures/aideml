from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.progress import Progress

from scripts.import_research_hypotheses import (
    _normalize_hypothesis,
    repo_root,
)


AGENT_MODES_BY_RUN_MODE = {
    "legacy": ["legacy"],
    "autogluon": ["legacy", "autogluon"],
    "autogluon_preprocess": ["legacy", "autogluon"],
}
HYPOTHESIS_FILENAME_RE = re.compile(r"^hypothesis-(\d{6})\.json$")
WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ArchivedImportResult:
    task: str
    target_dir: Path
    created_paths: tuple[Path, ...]
    dry_run: bool
    response_file_count: int
    candidate_count: int
    duplicate_count: int
    skipped_missing_config: int
    skipped_unknown_mode: int
    skipped_invalid: int
    duplicate_records: tuple[dict[str, Any], ...]

    @property
    def created_count(self) -> int:
        return len(self.created_paths)


@dataclass(frozen=True)
class _IndexedHypothesis:
    label: str
    fingerprint: str
    title_text: str
    content_tokens: frozenset[str]


@dataclass(frozen=True)
class _ArchivedCandidate:
    run_id: str
    checkpoint_step: str
    raw: dict[str, Any]
    agent_modes: list[str]
    source_path: Path


@dataclass(frozen=True)
class _SimilarityMatch:
    matched_label: str
    reason: str
    score: float


def import_archived_research_hypotheses(
    *,
    logs_dir: Path,
    task: str,
    repo_root: Path = repo_root(),
    enabled: bool = True,
    dry_run: bool = False,
    title_similarity_threshold: float = 0.86,
    token_jaccard_threshold: float = 0.72,
    progress_callback: Callable[[str, int, int | None], None] | None = None,
) -> ArchivedImportResult:
    target_dir = Path(repo_root) / "research_hypotheses" / task / "hypotheses"
    response_paths = sorted(Path(logs_dir).glob("*/research/checkpoint-*/response.json"))
    response_file_count = len(response_paths)
    candidates: list[_ArchivedCandidate] = []
    skipped_missing_config = 0
    skipped_unknown_mode = 0
    skipped_invalid = 0

    for index, response_path in enumerate(response_paths, start=1):
        _report(progress_callback, "Scanning archived research", index, response_file_count)
        run_dir = response_path.parents[2]
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            skipped_missing_config += 1
            continue
        run_mode = _read_run_agent_mode(config_path)
        agent_modes = AGENT_MODES_BY_RUN_MODE.get(run_mode)
        if agent_modes is None:
            skipped_unknown_mode += 1
            continue
        payload = _read_response_json(response_path)
        hypotheses = _extract_hypotheses(payload)
        for raw in hypotheses:
            if not _is_structured_archived_hypothesis(raw):
                skipped_invalid += 1
                continue
            candidates.append(
                _ArchivedCandidate(
                    run_id=run_dir.name,
                    checkpoint_step=response_path.parent.name.removeprefix(
                        "checkpoint-"
                    ),
                    raw=raw,
                    agent_modes=list(agent_modes),
                    source_path=response_path,
                )
            )

    existing = _load_existing_index(target_dir)
    accepted_index = list(existing)
    next_id = _next_hypothesis_number(target_dir)
    created_paths: list[Path] = []
    duplicate_records: list[dict[str, Any]] = []
    duplicate_count = 0

    for index, candidate in enumerate(candidates, start=1):
        _report(progress_callback, "Deduplicating hypotheses", index, len(candidates))
        raw_with_summary = {
            **candidate.raw,
            "summary": _summary_from_archived_hypothesis(candidate.raw),
        }
        try:
            payload = _normalize_hypothesis(
                raw_with_summary,
                enabled=enabled,
                agent_modes=candidate.agent_modes,
            )
        except ValueError:
            skipped_invalid += 1
            continue
        indexed = _index_hypothesis(
            label=f"{candidate.run_id}:{candidate.checkpoint_step}:{payload['title']}",
            payload=payload,
        )
        similarity_match = _find_similar_hypothesis(
            indexed,
            accepted_index,
            title_similarity_threshold=title_similarity_threshold,
            token_jaccard_threshold=token_jaccard_threshold,
        )
        if similarity_match is not None:
            duplicate_count += 1
            duplicate_records.append(
                {
                    "candidate_title": payload["title"],
                    "source_path": str(candidate.source_path),
                    "run_id": candidate.run_id,
                    "checkpoint_step": candidate.checkpoint_step,
                    "matched_label": similarity_match.matched_label,
                    "reason": similarity_match.reason,
                    "score": round(similarity_match.score, 6),
                }
            )
            continue

        output_path = target_dir / f"hypothesis-{next_id:06d}.json"
        next_id += 1
        created_paths.append(output_path)
        accepted_index.append(indexed)
        if dry_run:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return ArchivedImportResult(
        task=task,
        target_dir=target_dir,
        created_paths=tuple(created_paths),
        dry_run=dry_run,
        response_file_count=response_file_count,
        candidate_count=len(candidates),
        duplicate_count=duplicate_count,
        skipped_missing_config=skipped_missing_config,
        skipped_unknown_mode=skipped_unknown_mode,
        skipped_invalid=skipped_invalid,
        duplicate_records=tuple(duplicate_records),
    )


def _report(
    progress_callback: Callable[[str, int, int | None], None] | None,
    label: str,
    completed: int,
    total: int | None = None,
) -> None:
    if progress_callback is not None:
        progress_callback(label, completed, total)


def _read_response_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_hypotheses(payload: dict[str, Any]) -> list[dict[str, Any]]:
    parsed = payload.get("parsed_response")
    if isinstance(parsed, dict) and isinstance(parsed.get("hypotheses"), list):
        return [item for item in parsed["hypotheses"] if isinstance(item, dict)]
    raw_response = payload.get("raw_response")
    if isinstance(raw_response, str) and raw_response.strip():
        try:
            raw_payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return []
        if isinstance(raw_payload, dict) and isinstance(
            raw_payload.get("hypotheses"), list
        ):
            return [
                item for item in raw_payload["hypotheses"] if isinstance(item, dict)
            ]
    return []


def _is_structured_archived_hypothesis(raw: dict[str, Any]) -> bool:
    for field in (
        "title",
        "rationale",
        "implementation_hint",
        "expected_effect",
        "risk",
    ):
        if not isinstance(raw.get(field), str) or not raw[field].strip():
            return False
    return isinstance(raw.get("sources"), list)


def _summary_from_archived_hypothesis(raw: dict[str, Any]) -> str:
    rationale = str(raw.get("rationale") or "").strip()
    for match in re.finditer(r"(?<=[.!?])\s+", rationale):
        summary = rationale[: match.start()].strip()
        if summary:
            return summary
    if rationale:
        return rationale
    return str(raw.get("title") or "").strip()


def _read_run_agent_mode(config_path: Path) -> str | None:
    loader = _pathlib_tolerant_yaml_loader()
    try:
        payload = yaml.load(config_path.read_text(encoding="utf-8"), Loader=loader)
    except yaml.YAMLError:
        return None
    if not isinstance(payload, dict):
        return None
    agent = payload.get("agent")
    if not isinstance(agent, dict):
        return None
    mode = agent.get("mode")
    return mode.strip() if isinstance(mode, str) and mode.strip() else None


def _pathlib_tolerant_yaml_loader() -> type[yaml.SafeLoader]:
    class Loader(yaml.SafeLoader):
        pass

    def construct_path(loader: yaml.SafeLoader, node: yaml.Node) -> str:
        value = loader.construct_sequence(node)
        return str(Path(*value))

    Loader.add_constructor(
        "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
        construct_path,
    )
    return Loader


def _load_existing_index(target_dir: Path) -> list[_IndexedHypothesis]:
    if not target_dir.exists():
        return []
    indexed = []
    for path in sorted(target_dir.glob("hypothesis-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            indexed.append(_index_hypothesis(label=path.name, payload=payload))
    return indexed


def _next_hypothesis_number(target_dir: Path) -> int:
    max_id = 0
    if not target_dir.exists():
        return 1
    for path in target_dir.glob("hypothesis-*.json"):
        match = HYPOTHESIS_FILENAME_RE.match(path.name)
        if match is not None:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def _index_hypothesis(label: str, payload: dict[str, Any]) -> _IndexedHypothesis:
    content = " ".join(
        str(payload.get(field, ""))
        for field in (
            "title",
            "summary",
            "rationale",
            "implementation_hint",
            "expected_effect",
            "risk",
        )
    )
    fingerprint = _normalize_text(content)
    return _IndexedHypothesis(
        label=label,
        fingerprint=fingerprint,
        title_text=_normalize_text(str(payload.get("title", ""))),
        content_tokens=frozenset(_content_tokens(content)),
    )


def _normalize_text(value: str) -> str:
    return " ".join(WORD_RE.findall(value.lower()))


def _content_tokens(value: str) -> set[str]:
    return {token for token in WORD_RE.findall(value.lower()) if len(token) > 2}


def _find_similar_hypothesis(
    candidate: _IndexedHypothesis,
    existing: list[_IndexedHypothesis],
    *,
    title_similarity_threshold: float,
    token_jaccard_threshold: float,
) -> _SimilarityMatch | None:
    for other in existing:
        if candidate.fingerprint and candidate.fingerprint == other.fingerprint:
            return _SimilarityMatch(
                matched_label=other.label,
                reason="exact_content",
                score=1.0,
            )
        title_similarity = difflib.SequenceMatcher(
            None, candidate.title_text, other.title_text
        ).ratio()
        if title_similarity >= title_similarity_threshold:
            return _SimilarityMatch(
                matched_label=other.label,
                reason="title_similarity",
                score=title_similarity,
            )
        token_jaccard = _jaccard(candidate.content_tokens, other.content_tokens)
        if token_jaccard >= token_jaccard_threshold:
            return _SimilarityMatch(
                matched_label=other.label,
                reason="token_jaccard",
                score=token_jaccard,
            )
    return None


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import structured archived AIDE research checkpoints into the "
            "manual hypothesis library."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing AIDE run logs.",
    )
    parser.add_argument(
        "--task",
        default="playground-series-s6e5",
        help="Task slug under research_hypotheses/<task>/hypotheses.",
    )
    parser.add_argument(
        "--disabled",
        action="store_true",
        help="Import hypotheses with enabled=false.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show counts and paths without writing hypothesis files.",
    )
    parser.add_argument(
        "--title-similarity-threshold",
        type=float,
        default=0.86,
        help="Title similarity threshold used to skip near-duplicate hypotheses.",
    )
    parser.add_argument(
        "--token-jaccard-threshold",
        type=float,
        default=0.72,
        help="Token Jaccard threshold used to skip near-duplicate hypotheses.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print every output path. By default only a small sample is shown.",
    )
    parser.add_argument(
        "--path-sample-size",
        type=int,
        default=20,
        help="Number of created paths to print when --show-paths is not set.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional JSON report path for counts and skipped similar hypotheses.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    progress_state: dict[str, Any] = {"task_id": None, "label": None}

    with Progress() as progress:

        def progress_callback(label: str, completed: int, total: int | None) -> None:
            if progress_state["task_id"] is None or progress_state["label"] != label:
                progress_state["task_id"] = progress.add_task(label, total=total)
                progress_state["label"] = label
            progress.update(progress_state["task_id"], completed=completed, total=total)

        try:
            result = import_archived_research_hypotheses(
                logs_dir=args.logs_dir,
                task=args.task,
                enabled=not args.disabled,
                dry_run=args.dry_run,
                title_similarity_threshold=args.title_similarity_threshold,
                token_jaccard_threshold=args.token_jaccard_threshold,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            print(f"Archive import failed: {exc}", file=sys.stderr)
            return 1

    action = "Would create" if result.dry_run else "Created"
    print(
        f"{action} {result.created_count} hypotheses in "
        f"{result.target_dir.relative_to(repo_root())}"
    )
    print(f"Response files scanned: {result.response_file_count}")
    print(f"Structured candidates: {result.candidate_count}")
    print(f"Skipped similar hypotheses: {result.duplicate_count}")
    print(f"Skipped missing run config: {result.skipped_missing_config}")
    print(f"Skipped unknown run mode: {result.skipped_unknown_mode}")
    print(f"Skipped invalid hypotheses: {result.skipped_invalid}")
    paths_to_print = (
        result.created_paths
        if args.show_paths
        else result.created_paths[: max(args.path_sample_size, 0)]
    )
    for path in paths_to_print:
        print(f"- {path}")
    remaining_count = result.created_count - len(paths_to_print)
    if remaining_count > 0:
        print(f"... {remaining_count} more paths hidden; use --show-paths to print all")
    if args.report_path is not None:
        _write_report(args.report_path, result)
        print(f"Report written to {args.report_path}")
    return 0


def _write_report(path: Path, result: ArchivedImportResult) -> None:
    payload = {
        "task": result.task,
        "target_dir": str(result.target_dir),
        "dry_run": result.dry_run,
        "response_file_count": result.response_file_count,
        "candidate_count": result.candidate_count,
        "created_count": result.created_count,
        "duplicate_count": result.duplicate_count,
        "skipped_missing_config": result.skipped_missing_config,
        "skipped_unknown_mode": result.skipped_unknown_mode,
        "skipped_invalid": result.skipped_invalid,
        "created_paths": [str(path) for path in result.created_paths],
        "duplicate_records": list(result.duplicate_records),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
