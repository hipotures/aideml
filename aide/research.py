"""External Codex research advisor for long AIDE runs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import random
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .autogluon_preprocess import (
    extract_preprocess_source,
    is_autogluon_preprocess_mode,
)
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue

RESEARCH_PROMPT_INTRO = (
    "You are a research scientist and Kaggle competition strategist. Your job "
    "is to investigate this machine learning problem using live web search, "
    "compare public techniques and adjacent competition patterns, and propose "
    "concise, testable hypotheses for the existing AIDE code-search agent. Do "
    "not write a full solution script. Return only structured JSON matching "
    "the provided schema."
)


RESEARCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "rationale": {"type": "string"},
                    "implementation_hint": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "risk": {"type": "string"},
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "title",
                    "rationale",
                    "implementation_hint",
                    "expected_effect",
                    "risk",
                    "sources",
                ],
            },
        },
    },
    "required": ["summary", "hypotheses"],
}


Runner = Callable[..., subprocess.CompletedProcess[str]]
PROMPT_SCORE_DECIMALS = 5
REPO_ROOT = Path(__file__).resolve().parent.parent
MANUAL_HYPOTHESIS_PATTERN = re.compile(r"^hypothesis-(\d{6})\.json$")
MANUAL_HYPOTHESIS_AGENT_MODES = {"legacy", "autogluon"}
MANUAL_USAGE_RESEARCH_MODES = {"manual", "hypothesis"}
ROOT_CODE_PATTERN = re.compile(r"^(autogluon|legacy)-(\d{3})\.py$")


@dataclass(frozen=True)
class ManualHypothesis:
    id: str
    enabled: bool
    agent_modes: list[str]
    title: str
    summary: str
    rationale: str
    implementation_hint: str
    expected_effect: str
    risk: str
    sources: list[str]
    path: Path


@dataclass(frozen=True)
class ManualHypothesisLibrary:
    task_slug: str
    source_dir: Path
    source_hash: str
    hypotheses: list[ManualHypothesis]


@dataclass(frozen=True)
class ManualHypothesisSelection:
    completed_steps: int
    source_hash: str
    source_dir: Path
    hypotheses: list[ManualHypothesis]


@dataclass(frozen=True)
class HypothesisRootReservation:
    selection: ManualHypothesisSelection
    hypothesis_id: str
    completed_steps: int
    retry_attempts: int = 0


@dataclass(frozen=True)
class HypothesisRootCode:
    hypothesis_id: str
    agent_mode: str
    path: Path
    code: str
    version: int


@dataclass(frozen=True)
class ScoredHypothesisRootCode(HypothesisRootCode):
    score: float
    created_at: str | None
    source_node_id: str | None


def _json_default(value: Any) -> str:
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manual_task_slug(cfg: Config) -> str:
    return Path(cfg.data_dir).name


def _manual_library_dir(cfg: Config, *, repo_root: Path = REPO_ROOT) -> Path:
    return Path(repo_root) / "research_hypotheses" / _manual_task_slug(cfg)


def _manual_agent_mode_key(cfg: Config) -> str:
    mode = getattr(cfg.agent, "mode", "legacy")
    if mode in {"autogluon", "autogluon_preprocess"}:
        return "autogluon"
    return "legacy"


def _hypothesis_dir(source_dir: Path, hypothesis_id: str) -> Path:
    return source_dir / hypothesis_id


def _hypothesis_file_path(source_dir: Path, hypothesis_id: str) -> Path:
    return _hypothesis_dir(source_dir, hypothesis_id) / f"hypothesis-{hypothesis_id}.json"


def _new_layout_hypothesis_files(source_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in source_dir.glob("*/hypothesis-*.json"):
        if not path.parent.name.isdigit():
            continue
        match = MANUAL_HYPOTHESIS_PATTERN.match(path.name)
        if match is None or match.group(1) != path.parent.name:
            continue
        files.append(path)
    return sorted(files)


def _legacy_layout_hypothesis_files(source_dir: Path) -> list[Path]:
    return sorted((source_dir / "hypotheses").glob("hypothesis-*.json"))


def _manual_source_hash(source_dir: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _read_manual_hypothesis(path: Path) -> ManualHypothesis:
    match = MANUAL_HYPOTHESIS_PATTERN.match(path.name)
    if match is None:
        raise ValueError(
            f"Invalid manual research hypothesis filename: {path.name}. "
            "Expected hypothesis-000001.json."
        )
    hypothesis_id = match.group(1)
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid hypothesis JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Manual research hypothesis {path} must be a JSON object.")

    missing = []
    if "enabled" not in payload or not isinstance(payload.get("enabled"), bool):
        missing.append("enabled")
    raw_agent_modes = payload.get("agent_modes")
    if (
        not isinstance(raw_agent_modes, list)
        or not raw_agent_modes
        or not all(
            isinstance(mode, str) and mode in MANUAL_HYPOTHESIS_AGENT_MODES
            for mode in raw_agent_modes
        )
    ):
        missing.append("agent_modes")
    missing.extend(
        field
        for field in (
            "title",
            "summary",
            "rationale",
            "implementation_hint",
            "expected_effect",
            "risk",
        )
        if not isinstance(payload.get(field), str) or not payload.get(field).strip()
    )
    raw_sources = payload.get("sources", [])
    if raw_sources is None:
        raw_sources = []
    if not isinstance(raw_sources, list) or not all(
        isinstance(source, str) and source.strip() for source in raw_sources
    ):
        missing.append("sources")
    if missing:
        raise ValueError(
            f"Manual research hypothesis {path} missing required field(s): "
            + ", ".join(missing)
        )

    return ManualHypothesis(
        id=hypothesis_id,
        enabled=payload["enabled"],
        agent_modes=list(raw_agent_modes),
        title=payload["title"].strip(),
        summary=payload["summary"].strip(),
        rationale=payload["rationale"].strip(),
        implementation_hint=payload["implementation_hint"].strip(),
        expected_effect=payload["expected_effect"].strip(),
        risk=payload["risk"].strip(),
        sources=[source.strip() for source in raw_sources],
        path=path,
    )


def load_manual_hypothesis_library(
    cfg: Config,
    *,
    repo_root: Path = REPO_ROOT,
) -> ManualHypothesisLibrary:
    task_slug = _manual_task_slug(cfg)
    source_dir = _manual_library_dir(cfg, repo_root=repo_root)
    if not source_dir.exists():
        raise ValueError(f"Missing manual research library: {source_dir}")

    files = _new_layout_hypothesis_files(source_dir)
    if not files:
        files = _legacy_layout_hypothesis_files(source_dir)
    if not files:
        raise ValueError(
            "No manual research hypothesis files found in "
            f"{source_dir} using <id>/hypothesis-<id>.json or "
            "hypotheses/hypothesis-*.json."
        )

    hypotheses: list[ManualHypothesis] = []
    seen_ids: set[str] = set()
    for path in files:
        hypothesis = _read_manual_hypothesis(path)
        if hypothesis.id in seen_ids:
            raise ValueError(f"Duplicate manual research hypothesis id: {hypothesis.id}")
        seen_ids.add(hypothesis.id)
        hypotheses.append(hypothesis)

    return ManualHypothesisLibrary(
        task_slug=task_slug,
        source_dir=source_dir,
        source_hash=_manual_source_hash(source_dir, files),
        hypotheses=hypotheses,
    )


def _manifest_path(hypothesis_dir: Path) -> Path:
    return hypothesis_dir / "code_manifest.json"


def _load_code_manifest(hypothesis_dir: Path) -> dict[str, Any]:
    path = _manifest_path(hypothesis_dir)
    if not path.exists():
        return {}
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid code manifest JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Code manifest {path} must be a JSON object.")
    return payload


def _code_versions(hypothesis_dir: Path, agent_mode: str) -> dict[int, Path]:
    versions: dict[int, Path] = {}
    for path in hypothesis_dir.glob(f"{agent_mode}-*.py"):
        match = ROOT_CODE_PATTERN.match(path.name)
        if match is None or match.group(1) != agent_mode:
            continue
        versions[int(match.group(2))] = path
    return versions


def _manifest_active_file(manifest: dict[str, Any], agent_mode: str) -> str | None:
    active = manifest.get("active")
    if isinstance(active, dict) and isinstance(active.get(agent_mode), str):
        return active[agent_mode]
    active_versions = manifest.get("active_versions")
    if isinstance(active_versions, dict) and isinstance(
        active_versions.get(agent_mode),
        str,
    ):
        return active_versions[agent_mode]
    return None


def load_hypothesis_root_code(
    cfg: Config,
    hypothesis_id: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> HypothesisRootCode | None:
    """Load the active flat-library root code for a hypothesis and agent mode."""
    source_dir = _manual_library_dir(cfg, repo_root=repo_root)
    hypothesis_dir = _hypothesis_dir(source_dir, hypothesis_id)
    if not hypothesis_dir.exists():
        return None

    agent_mode = _manual_agent_mode_key(cfg)
    versions = _code_versions(hypothesis_dir, agent_mode)
    if not versions:
        return None

    manifest = _load_code_manifest(hypothesis_dir)
    active_file = _manifest_active_file(manifest, agent_mode)
    active_path = hypothesis_dir / active_file if active_file is not None else None
    if active_path is not None and active_path in versions.values():
        path = active_path
        version = next(number for number, candidate in versions.items() if candidate == path)
    else:
        version = max(versions)
        path = versions[version]
    entry = _manifest_entry_for_file(
        manifest,
        agent_mode=agent_mode,
        file_name=path.name,
    )
    if entry is not None and not _manifest_entry_is_loadable(entry):
        return None
    return HypothesisRootCode(
        hypothesis_id=hypothesis_id,
        agent_mode=agent_mode,
        path=path,
        code=path.read_text(encoding="utf-8"),
        version=version,
    )


def load_scored_hypothesis_root_code(
    cfg: Config,
    hypothesis_id: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> ScoredHypothesisRootCode | None:
    """Load active ROOT code only when its manifest proves it executed with a score."""
    root_code = load_hypothesis_root_code(
        cfg,
        hypothesis_id,
        repo_root=repo_root,
    )
    if root_code is None:
        return None

    manifest = _load_code_manifest(root_code.path.parent)
    entry = _manifest_entry_for_file(
        manifest,
        agent_mode=root_code.agent_mode,
        file_name=root_code.path.name,
    )
    if entry is None or not _manifest_entry_is_loadable(entry):
        return None
    score = _numeric_manifest_score(entry.get("score"))
    if score is None:
        return None
    created_at = entry.get("created_at")
    source_node_id = entry.get("node_id")
    return ScoredHypothesisRootCode(
        hypothesis_id=root_code.hypothesis_id,
        agent_mode=root_code.agent_mode,
        path=root_code.path,
        code=root_code.code,
        version=root_code.version,
        score=score,
        created_at=created_at if isinstance(created_at, str) else None,
        source_node_id=source_node_id if isinstance(source_node_id, str) else None,
    )


def _manifest_created_at_timestamp(created_at: str | None) -> float | None:
    if not created_at:
        return None
    try:
        normalized = created_at.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _scored_hypothesis_node(
    *,
    root_code: ScoredHypothesisRootCode,
    source_hash: str,
) -> Node:
    timestamp = _manifest_created_at_timestamp(root_code.created_at)
    node = Node(
        code=root_code.code,
        plan=(
            f"Seeded scored ROOT hypothesis {root_code.hypothesis_id} "
            f"from {root_code.agent_mode} {root_code.path.name}."
        ),
        **({"ctime": timestamp} if timestamp is not None else {}),
    )
    node.metric = MetricValue(root_code.score, maximize=True)
    node.is_buggy = False
    node.status = "ok"
    node._term_out = [
        (
            "Seeded from code_manifest.json; "
            f"score={root_code.score:.5f}; file={root_code.path.name}."
        )
    ]
    node.exec_time = 0.0
    node.exc_type = None
    node.exc_info = None
    node.exc_stack = None
    node.analysis = (
        "Seeded from previously executed hypothesis ROOT code with numeric "
        f"manifest score {root_code.score:.5f}."
    )
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = [root_code.hypothesis_id]
    node.research_source_hash = source_hash
    return node


def scored_hypothesis_root_nodes(
    cfg: Config,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[Node]:
    """Build completed ROOT nodes from scored active manifest code for this mode."""
    library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
    nodes: list[Node] = []
    for hypothesis in _compatible_manual_hypotheses(cfg, library):
        root_code = load_scored_hypothesis_root_code(
            cfg,
            hypothesis.id,
            repo_root=repo_root,
        )
        if root_code is None:
            continue
        nodes.append(
            _scored_hypothesis_node(
                root_code=root_code,
                source_hash=library.source_hash,
            )
        )
    return nodes


def _manifest_entry_for_file(
    manifest: dict[str, Any],
    *,
    agent_mode: str,
    file_name: str,
) -> dict[str, Any] | None:
    versions = manifest.get("versions")
    if not isinstance(versions, dict):
        return None
    mode_versions = versions.get(agent_mode)
    if not isinstance(mode_versions, list):
        return None
    for entry in mode_versions:
        if isinstance(entry, dict) and entry.get("file") == file_name:
            return entry
    return None


def _manifest_entry_is_loadable(entry: dict[str, Any]) -> bool:
    if entry.get("buggy") is True or entry.get("status") == "bug":
        return False
    if entry.get("status") == "recovered":
        return False
    if (
        entry.get("recovered_from") == "response.py"
        and entry.get("node_id") is None
        and _numeric_manifest_score(entry.get("score")) is None
    ):
        return False
    return True


def save_hypothesis_root_code(
    cfg: Config,
    *,
    hypothesis_id: str,
    code: str,
    is_buggy: bool,
    node_id: str | None = None,
    score: float | None = None,
    created_at: str | None = None,
    force_new_version: bool = False,
    activate: bool = True,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Save a generated root hypothesis node as the single flat-library file."""
    source_dir = _manual_library_dir(cfg, repo_root=repo_root)
    hypothesis_dir = _hypothesis_dir(source_dir, hypothesis_id)
    hypothesis_json = _hypothesis_file_path(source_dir, hypothesis_id)
    if not hypothesis_json.exists():
        raise ValueError(
            f"Cannot save root code for unknown hypothesis {hypothesis_id}: "
            f"{hypothesis_json} does not exist."
        )

    agent_mode = _manual_agent_mode_key(cfg)
    hypothesis_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_code_manifest(hypothesis_dir)
    versions_by_number = _code_versions(hypothesis_dir, agent_mode)
    highest_version = max(versions_by_number, default=0)
    highest_path = versions_by_number.get(highest_version)
    highest_entry = (
        _manifest_entry_for_file(
            manifest,
            agent_mode=agent_mode,
            file_name=highest_path.name,
        )
        if highest_path is not None
        else None
    )
    highest_is_buggy = (
        highest_entry.get("buggy") is True if highest_entry is not None else False
    )

    status = "generated" if not activate else "bug" if is_buggy else "ok"
    metadata = {
        "buggy": bool(is_buggy) if activate else None,
        "status": status,
        "node_id": node_id,
        "score": score,
        "created_at": created_at,
        "aux": bool(getattr(cfg.agent, "aux", False)),
    }
    versions = manifest.setdefault("versions", {})
    if not isinstance(versions, dict):
        manifest["versions"] = versions = {}
    mode_versions = versions.setdefault(agent_mode, [])
    if not isinstance(mode_versions, list):
        versions[agent_mode] = mode_versions = []

    existing_entry = None
    if node_id is not None:
        for entry in mode_versions:
            if isinstance(entry, dict) and entry.get("node_id") == node_id:
                existing_entry = entry
                break
    if existing_entry is not None and isinstance(existing_entry.get("file"), str):
        path = hypothesis_dir / existing_entry["file"]
    elif force_new_version:
        next_version = highest_version + 1 if highest_path is not None else 1
        path = hypothesis_dir / f"{agent_mode}-{next_version:03d}.py"
        path.write_text(code, encoding="utf-8")
    elif highest_path is not None and (is_buggy or not highest_is_buggy):
        scored_existing = (
            highest_entry is not None
            and _numeric_manifest_score(highest_entry.get("score")) is not None
        )
        if is_buggy and scored_existing:
            return highest_path
        path = highest_path
    else:
        next_version = highest_version + 1 if highest_path is not None else 1
        path = hypothesis_dir / f"{agent_mode}-{next_version:03d}.py"
        path.write_text(code, encoding="utf-8")

    previous_entry = _manifest_entry_for_file(
        manifest,
        agent_mode=agent_mode,
        file_name=path.name,
    )
    versions[agent_mode] = [
        entry
        for entry in mode_versions
        if not (isinstance(entry, dict) and entry.get("file") == path.name)
    ]
    preserved_metadata = previous_entry if previous_entry is not None else {}
    versions[agent_mode].append({"file": path.name, **preserved_metadata, **metadata})
    active = manifest.setdefault("active", {})
    if not isinstance(active, dict):
        manifest["active"] = active = {}
    if activate and not is_buggy:
        active[agent_mode] = path.name
    elif activate:
        active.pop(agent_mode, None)
    _write_json(_manifest_path(hypothesis_dir), manifest)
    return path


def _manual_run_dir(cfg: Config) -> Path:
    return Path(cfg.log_dir) / "research_hypotheses"


def _load_manual_usage(cfg: Config) -> dict[str, Any]:
    usage_path = _manual_run_dir(cfg) / "usage.json"
    if not usage_path.exists():
        return {}
    try:
        data = _read_json(usage_path)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_manual_usage(cfg: Config, usage: dict[str, Any]) -> None:
    _write_json(_manual_run_dir(cfg) / "usage.json", usage)


def _root_generation_failures_path(cfg: Config) -> Path:
    return _manual_run_dir(cfg) / "root_generation_failures.json"


def _load_root_generation_failures(cfg: Config) -> dict[str, Any]:
    path = _root_generation_failures_path(cfg)
    if not path.exists():
        return {}
    try:
        data = _read_json(path)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_root_generation_failures(cfg: Config, data: dict[str, Any]) -> None:
    _write_json(_root_generation_failures_path(cfg), data)


def record_hypothesis_root_generation_failure(
    cfg: Config,
    *,
    hypothesis_id: str,
    attempts: int,
    message: str,
) -> None:
    failures = _load_root_generation_failures(cfg)
    failures[hypothesis_id] = {
        "attempts": attempts,
        "message": message,
        "last_failed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    _write_root_generation_failures(cfg, failures)


def clear_hypothesis_root_generation_failure(cfg: Config, *, hypothesis_id: str) -> None:
    failures = _load_root_generation_failures(cfg)
    if hypothesis_id not in failures:
        return
    del failures[hypothesis_id]
    _write_root_generation_failures(cfg, failures)


def _write_manual_source_ref(
    *,
    cfg: Config,
    library: ManualHypothesisLibrary,
    created_at: str,
) -> None:
    enabled_count = sum(1 for hypothesis in library.hypotheses if hypothesis.enabled)
    agent_mode = _manual_agent_mode_key(cfg)
    compatible_hypotheses = _compatible_manual_hypotheses(cfg, library)
    compatible_count = len(compatible_hypotheses)
    compatible_ids = sorted(hypothesis.id for hypothesis in compatible_hypotheses)
    configured_root_limit = _configured_hypothesis_root_limit(cfg)
    _write_json(
        _manual_run_dir(cfg) / "source_ref.json",
        {
            "source_dir": str(library.source_dir),
            "source_hash": library.source_hash,
            "indexed_hypothesis_count": len(library.hypotheses),
            "enabled_hypothesis_count": enabled_count,
            "agent_mode": agent_mode,
            "compatible_hypothesis_count": compatible_count,
            "compatible_hypothesis_ids": compatible_ids,
            "configured_hypothesis_root_limit": configured_root_limit,
            "effective_hypothesis_root_limit": effective_hypothesis_root_limit(
                cfg,
                compatible_count=compatible_count,
            ),
            "hypothesis_root_order": getattr(
                cfg.research,
                "hypothesis_root_order",
                "default",
            ),
            "hypothesis_root_score_mode": getattr(
                cfg.research,
                "hypothesis_root_score_mode",
                "autogluon",
            ),
            "indexed_at": created_at,
            "filename_pattern": "<id>/hypothesis-<id>.json",
        },
    )


def _append_manual_offer(
    *,
    cfg: Config,
    completed_steps: int,
    offered_ids: list[str],
    source_hash: str,
    created_at: str,
) -> None:
    path = _manual_run_dir(cfg) / "offers.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_step": completed_steps,
        "offered": offered_ids,
        "source_hash": source_hash,
        "created_at": created_at,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _record_manual_offer_usage(
    *,
    cfg: Config,
    offered_ids: list[str],
    completed_steps: int,
    created_at: str,
) -> None:
    usage = _load_manual_usage(cfg)
    for hypothesis_id in offered_ids:
        entry = usage.setdefault(hypothesis_id, {})
        entry["offered_count"] = int(entry.get("offered_count", 0)) + 1
        steps = entry.setdefault("offered_checkpoint_steps", [])
        if isinstance(steps, list):
            steps.append(completed_steps)
        else:
            entry["offered_checkpoint_steps"] = [completed_steps]
        entry.setdefault("llm_claimed_used_count", 0)
        entry.setdefault("prompt_node_ids", [])
        entry.setdefault("llm_claimed_used_node_ids", [])
        entry["last_offered_at"] = created_at
    _write_manual_usage(cfg, usage)


def _manual_offer_sort_key(
    *,
    hypothesis: ManualHypothesis,
    usage: dict[str, Any],
    seed_text: str,
) -> tuple[int, float, str]:
    entry = usage.get(hypothesis.id, {})
    offered_count = (
        int(entry.get("offered_count", 0)) if isinstance(entry, dict) else 0
    )
    tie_break = random.Random(f"{seed_text}:{hypothesis.id}").random()
    return offered_count, tie_break, hypothesis.id


def select_manual_hypotheses(
    cfg: Config,
    *,
    completed_steps: int,
    repo_root: Path = REPO_ROOT,
) -> ManualHypothesisSelection:
    library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
    agent_mode = _manual_agent_mode_key(cfg)
    compatible_hypotheses = _compatible_manual_hypotheses(cfg, library)
    sample_size = int(cfg.research.manual_sample_size)
    if sample_size <= 0:
        raise ValueError("research.manual_sample_size must be greater than 0.")
    if sample_size > len(compatible_hypotheses):
        raise ValueError(
            "research.manual_sample_size cannot exceed compatible manual hypotheses "
            f"for agent mode {agent_mode} "
            f"({sample_size} requested, {len(compatible_hypotheses)} compatible)."
        )

    usage = _load_manual_usage(cfg)
    seed_text = f"{cfg.research.manual_seed}:{cfg.exp_name}:{completed_steps}"
    ordered = sorted(
        compatible_hypotheses,
        key=lambda hypothesis: _manual_offer_sort_key(
            hypothesis=hypothesis,
            usage=usage,
            seed_text=seed_text,
        ),
    )
    selected = sorted(ordered[:sample_size], key=lambda hypothesis: hypothesis.id)
    offered_ids = [hypothesis.id for hypothesis in selected]
    created_at = dt.datetime.now().isoformat(timespec="seconds")
    _write_manual_source_ref(cfg=cfg, library=library, created_at=created_at)
    _append_manual_offer(
        cfg=cfg,
        completed_steps=completed_steps,
        offered_ids=offered_ids,
        source_hash=library.source_hash,
        created_at=created_at,
    )
    _record_manual_offer_usage(
        cfg=cfg,
        offered_ids=offered_ids,
        completed_steps=completed_steps,
        created_at=created_at,
    )
    return ManualHypothesisSelection(
        completed_steps=completed_steps,
        source_hash=library.source_hash,
        source_dir=library.source_dir,
        hypotheses=selected,
    )


def _compatible_manual_hypotheses(
    cfg: Config,
    library: ManualHypothesisLibrary,
) -> list[ManualHypothesis]:
    if getattr(cfg.research, "ignore_hypothesis_agent_modes", False):
        return [hypothesis for hypothesis in library.hypotheses if hypothesis.enabled]
    agent_mode = _manual_agent_mode_key(cfg)
    return [
        hypothesis
        for hypothesis in library.hypotheses
        if hypothesis.enabled and agent_mode in hypothesis.agent_modes
    ]


def hypothesis_id_for_node(node: Node) -> str | None:
    if getattr(node, "research_mode", None) != "hypothesis":
        return None
    offered = getattr(node, "research_hypotheses_offered", []) or []
    if len(offered) != 1 or not isinstance(offered[0], str):
        return None
    return offered[0]


def _hypothesis_attempt_counts(cfg: Config, journal: Journal) -> dict[str, int]:
    counts: dict[str, int] = {}
    usage = _load_manual_usage(cfg)
    for hypothesis_id, entry in usage.items():
        if not isinstance(hypothesis_id, str) or not isinstance(entry, dict):
            continue
        try:
            offered_count = int(entry.get("offered_count", 0))
        except (TypeError, ValueError):
            continue
        if offered_count > 0:
            counts[hypothesis_id] = offered_count
    for node in journal.nodes:
        hypothesis_id = hypothesis_id_for_node(node)
        if hypothesis_id is None:
            continue
        counts[hypothesis_id] = max(counts.get(hypothesis_id, 0), 1)
    return counts


def _root_hypothesis_ids(journal: Journal) -> set[str]:
    ids: set[str] = set()
    for node in journal.nodes:
        if node.parent is not None:
            continue
        hypothesis_id = hypothesis_id_for_node(node)
        if hypothesis_id is not None:
            ids.add(hypothesis_id)
    return ids


def _unmaterialized_root_offer_ids(
    cfg: Config,
    *,
    journal: Journal,
    compatible_by_id: dict[str, ManualHypothesis],
    reserved_hypothesis_ids: set[str],
) -> list[str]:
    path = _manual_run_dir(cfg) / "offers.jsonl"
    if not path.exists():
        return []
    materialized_root_ids = _root_hypothesis_ids(journal)
    retry_ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        offered = payload.get("offered")
        if not isinstance(offered, list):
            continue
        for offered_id in offered:
            if not isinstance(offered_id, str):
                continue
            if offered_id in seen:
                continue
            if offered_id in materialized_root_ids:
                continue
            if offered_id in reserved_hypothesis_ids:
                continue
            if offered_id not in compatible_by_id:
                continue
            seen.add(offered_id)
            retry_ids.append(offered_id)
    return retry_ids


def _configured_hypothesis_root_limit(cfg: Config) -> int:
    try:
        return int(getattr(cfg.research, "hypothesis_root_limit", 100))
    except (TypeError, ValueError):
        return 100


def effective_hypothesis_root_limit(cfg: Config, *, compatible_count: int) -> int:
    compatible_count = max(int(compatible_count), 0)
    if compatible_count == 0:
        return 0
    configured_limit = _configured_hypothesis_root_limit(cfg)
    if configured_limit <= 0:
        return compatible_count
    return min(configured_limit, compatible_count)


def _hypothesis_root_pool_complete(
    cfg: Config,
    *,
    journal: Journal,
    compatible: list[ManualHypothesis],
) -> bool:
    if not compatible:
        return True
    compatible_ids = {hypothesis.id for hypothesis in compatible}
    used_root_ids = _root_hypothesis_ids(journal)
    used_compatible_root_count = len(compatible_ids & used_root_ids)
    root_limit = effective_hypothesis_root_limit(
        cfg,
        compatible_count=len(compatible),
    )
    if used_compatible_root_count >= root_limit:
        return True
    return compatible_ids.issubset(used_root_ids)


def _ancestor_hypothesis_ids(parent_node: Node) -> set[str]:
    ids: set[str] = set()
    node: Node | None = parent_node
    while node is not None:
        hypothesis_id = hypothesis_id_for_node(node)
        if hypothesis_id is not None:
            ids.add(hypothesis_id)
        node = node.parent
    return ids


def _direct_child_hypothesis_ids(parent_node: Node, journal: Journal) -> set[str]:
    journal_nodes = set(journal.nodes)
    ids: set[str] = set()
    for child in parent_node.children:
        if child not in journal_nodes:
            continue
        hypothesis_id = hypothesis_id_for_node(child)
        if hypothesis_id is not None:
            ids.add(hypothesis_id)
    return ids


def hypothesis_root_pool_exhausted(
    cfg: Config,
    *,
    journal: Journal,
    repo_root: Path = REPO_ROOT,
) -> bool:
    library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
    compatible = _compatible_manual_hypotheses(cfg, library)
    return _hypothesis_root_pool_complete(
        cfg,
        journal=journal,
        compatible=compatible,
    )


def _hypothesis_candidates_for_node_from_library(
    cfg: Config,
    *,
    journal: Journal,
    parent_node: Node | None,
    library: ManualHypothesisLibrary,
) -> list[ManualHypothesis]:
    by_id = {hypothesis.id: hypothesis for hypothesis in library.hypotheses}
    forced_hypothesis = getattr(cfg.agent.search, "forced_hypothesis", None)
    compatible = _compatible_manual_hypotheses(cfg, library)
    if forced_hypothesis is not None:
        compatible = [
            hypothesis
            for hypothesis in compatible
            if hypothesis.id == forced_hypothesis
        ]

    if parent_node is not None and parent_node.is_buggy:
        inherited_id = hypothesis_id_for_node(parent_node)
        if inherited_id is None:
            return []
        if forced_hypothesis is not None and inherited_id != forced_hypothesis:
            return []
        inherited = by_id.get(inherited_id)
        return [inherited] if inherited is not None else []

    if parent_node is None:
        if _hypothesis_root_pool_complete(
            cfg,
            journal=journal,
            compatible=compatible,
        ):
            return []
        blocked_ids = _root_hypothesis_ids(journal)
    else:
        if forced_hypothesis is not None:
            return []
        blocked_ids = _ancestor_hypothesis_ids(parent_node)
        blocked_ids |= _direct_child_hypothesis_ids(parent_node, journal)

    return [
        hypothesis
        for hypothesis in compatible
        if hypothesis.id not in blocked_ids
    ]


def hypothesis_candidates_for_node(
    cfg: Config,
    *,
    journal: Journal,
    parent_node: Node | None,
    repo_root: Path = REPO_ROOT,
) -> list[ManualHypothesis]:
    library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
    return _hypothesis_candidates_for_node_from_library(
        cfg,
        journal=journal,
        parent_node=parent_node,
        library=library,
    )


def hypothesis_has_candidates_for_node(
    cfg: Config,
    journal: Journal,
    parent_node: Node | None,
    *,
    repo_root: Path = REPO_ROOT,
) -> bool:
    return bool(
        hypothesis_candidates_for_node(
            cfg,
            journal=journal,
            parent_node=parent_node,
            repo_root=repo_root,
        )
    )


def filter_hypothesis_candidate_parents(
    cfg: Config,
    *,
    journal: Journal,
    parent_nodes: list[Node],
    repo_root: Path = REPO_ROOT,
) -> list[Node]:
    if not parent_nodes:
        return []
    library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
    return [
        node
        for node in parent_nodes
        if _hypothesis_candidates_for_node_from_library(
            cfg,
            journal=journal,
            parent_node=node,
            library=library,
        )
    ]


def _hypothesis_sort_key(
    *,
    hypothesis: ManualHypothesis,
    attempts: dict[str, int],
    seed_text: str,
) -> tuple[int, float, str]:
    tie_break = random.Random(f"{seed_text}:{hypothesis.id}").random()
    return attempts.get(hypothesis.id, 0), tie_break, hypothesis.id


def _numeric_manifest_score(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def _best_manifest_score_for_mode(
    *,
    source_dir: Path,
    hypothesis_id: str,
    agent_mode: str,
) -> float | None:
    hypothesis_dir = _hypothesis_dir(source_dir, hypothesis_id)
    try:
        manifest = _load_code_manifest(hypothesis_dir)
    except ValueError:
        return None
    versions = manifest.get("versions")
    if not isinstance(versions, dict):
        return None
    mode_versions = versions.get(agent_mode)
    if not isinstance(mode_versions, list):
        return None
    scores = [
        score
        for entry in mode_versions
        if isinstance(entry, dict) and entry.get("buggy") is not True
        if (score := _numeric_manifest_score(entry.get("score"))) is not None
    ]
    return max(scores, default=None)


def _manifest_scores_for_hypotheses(
    *,
    source_dir: Path,
    hypotheses: list[ManualHypothesis],
    agent_mode: str,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for hypothesis in hypotheses:
        score = _best_manifest_score_for_mode(
            source_dir=source_dir,
            hypothesis_id=hypothesis.id,
            agent_mode=agent_mode,
        )
        if score is not None:
            scores[hypothesis.id] = score
    return scores


def _hypothesis_root_manifest_score_sort_key(
    *,
    hypothesis: ManualHypothesis,
    manifest_scores: dict[str, float],
    attempts: dict[str, int],
    seed_text: str,
) -> tuple[int, int, float, float, str]:
    score = manifest_scores.get(hypothesis.id)
    tie_break = random.Random(f"{seed_text}:{hypothesis.id}").random()
    if score is not None:
        return (attempts.get(hypothesis.id, 0), 0, -score, tie_break, hypothesis.id)
    return (attempts.get(hypothesis.id, 0), 1, 0.0, tie_break, hypothesis.id)


def _metric_for_hypothesis_ranking(node: Node) -> float:
    assert node.metric is not None and node.metric.value is not None
    value = float(node.metric.value)
    return -value if node.metric.maximize is False else value


def _root_hypothesis_score_ranks(journal: Journal) -> dict[str, tuple[float, str]]:
    ranks: dict[str, tuple[float, str]] = {}
    for node in journal.nodes:
        if node.parent is not None or node.is_buggy:
            continue
        hypothesis_id = hypothesis_id_for_node(node)
        if hypothesis_id is None or node.metric is None or node.metric.value is None:
            continue
        score = _metric_for_hypothesis_ranking(node)
        previous = ranks.get(hypothesis_id)
        if previous is None or score > previous[0]:
            ranks[hypothesis_id] = (score, node.id)
    return ranks


def _hypothesis_child_root_score_sort_key(
    *,
    hypothesis: ManualHypothesis,
    root_scores: dict[str, tuple[float, str]],
    attempts: dict[str, int],
    seed_text: str,
) -> tuple[int, float, int, float, str]:
    root_score = root_scores.get(hypothesis.id)
    tie_break = random.Random(f"{seed_text}:{hypothesis.id}").random()
    if root_score is not None:
        score, root_node_id = root_score
        return (0, -score, attempts.get(hypothesis.id, 0), tie_break, root_node_id)
    return (1, 0.0, attempts.get(hypothesis.id, 0), tie_break, hypothesis.id)


def select_hypothesis_for_node(
    cfg: Config,
    *,
    journal: Journal,
    parent_node: Node | None,
    completed_steps: int,
    repo_root: Path = REPO_ROOT,
) -> ManualHypothesisSelection:
    library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
    candidates = _hypothesis_candidates_for_node_from_library(
        cfg,
        journal=journal,
        parent_node=parent_node,
        library=library,
    )
    if not candidates:
        agent_mode = _manual_agent_mode_key(cfg)
        stage = (
            "root"
            if parent_node is None
            else "debug"
            if parent_node.is_buggy
            else "child"
        )
        raise ValueError(
            "No compatible hypothesis candidates available for "
            f"{stage} selection in agent mode {agent_mode}."
        )

    attempts = _hypothesis_attempt_counts(cfg, journal)
    seed_text = (
        f"{cfg.research.manual_seed}:{cfg.exp_name}:"
        f"{completed_steps}:{parent_node.id if parent_node is not None else 'root'}"
    )
    if (
        parent_node is not None
        and not parent_node.is_buggy
        and getattr(cfg.agent.search, "hypothesis_child_order", "root_score")
        == "root_score"
    ):
        root_scores = _root_hypothesis_score_ranks(journal)
        selected = sorted(
            candidates,
            key=lambda hypothesis: _hypothesis_child_root_score_sort_key(
                hypothesis=hypothesis,
                root_scores=root_scores,
                attempts=attempts,
                seed_text=seed_text,
            ),
        )[0]
    elif (
        parent_node is None
        and getattr(cfg.research, "hypothesis_root_order", "default")
        == "manifest_score"
    ):
        score_mode = str(
            getattr(cfg.research, "hypothesis_root_score_mode", "autogluon")
            or "autogluon"
        )
        manifest_scores = _manifest_scores_for_hypotheses(
            source_dir=library.source_dir,
            hypotheses=candidates,
            agent_mode=score_mode,
        )
        selected = sorted(
            candidates,
            key=lambda hypothesis: _hypothesis_root_manifest_score_sort_key(
                hypothesis=hypothesis,
                manifest_scores=manifest_scores,
                attempts=attempts,
                seed_text=seed_text,
            ),
        )[0]
    else:
        selected = sorted(
            candidates,
            key=lambda hypothesis: _hypothesis_sort_key(
                hypothesis=hypothesis,
                attempts=attempts,
                seed_text=seed_text,
            ),
        )[0]
    created_at = dt.datetime.now().isoformat(timespec="seconds")
    _write_manual_source_ref(cfg=cfg, library=library, created_at=created_at)
    _append_manual_offer(
        cfg=cfg,
        completed_steps=completed_steps,
        offered_ids=[selected.id],
        source_hash=library.source_hash,
        created_at=created_at,
    )
    _record_manual_offer_usage(
        cfg=cfg,
        offered_ids=[selected.id],
        completed_steps=completed_steps,
        created_at=created_at,
    )
    return ManualHypothesisSelection(
        completed_steps=completed_steps,
        source_hash=library.source_hash,
        source_dir=library.source_dir,
        hypotheses=[selected],
    )


def _append_reserved_placeholder(
    journal: Journal,
    *,
    hypothesis_id: str,
) -> None:
    placeholder = Node(code="", plan="reserved hypothesis root")
    placeholder.research_mode = "hypothesis"
    placeholder.research_hypotheses_offered = [hypothesis_id]
    journal.append(placeholder)


def reserve_hypothesis_roots(
    cfg: Config,
    *,
    journal: Journal,
    count: int,
    completed_steps: int,
    reserved_hypothesis_ids: set[str] | None = None,
    forced_hypothesis_ids: tuple[str, ...] | list[str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> list[HypothesisRootReservation]:
    reservations: list[HypothesisRootReservation] = []
    working_journal = Journal(nodes=list(journal.nodes))
    reserved_hypothesis_ids = set(reserved_hypothesis_ids or set())
    for reserved_id in sorted(reserved_hypothesis_ids):
        _append_reserved_placeholder(working_journal, hypothesis_id=reserved_id)
    failures = _load_root_generation_failures(cfg)
    library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
    compatible_by_id = {
        hypothesis.id: hypothesis for hypothesis in _compatible_manual_hypotheses(cfg, library)
    }

    if forced_hypothesis_ids:
        ordered_forced_ids = list(dict.fromkeys(forced_hypothesis_ids))
        missing_ids = [
            hypothesis_id
            for hypothesis_id in ordered_forced_ids
            if hypothesis_id not in compatible_by_id
        ]
        if missing_ids:
            raise ValueError(
                "Requested generate-only hypothesis id(s) are not available for "
                f"agent mode {getattr(cfg.agent, 'mode', 'legacy')!r}: "
                + ", ".join(missing_ids)
            )
        materialized_root_ids = _root_hypothesis_ids(journal)
        for hypothesis_id in ordered_forced_ids:
            if hypothesis_id in materialized_root_ids:
                continue
            if hypothesis_id in reserved_hypothesis_ids:
                continue
            if len(reservations) >= count:
                break
            hypothesis = compatible_by_id[hypothesis_id]
            step = completed_steps + len(reservations)
            created_at = dt.datetime.now().isoformat(timespec="seconds")
            _write_manual_source_ref(cfg=cfg, library=library, created_at=created_at)
            _append_manual_offer(
                cfg=cfg,
                completed_steps=step,
                offered_ids=[hypothesis.id],
                source_hash=library.source_hash,
                created_at=created_at,
            )
            _record_manual_offer_usage(
                cfg=cfg,
                offered_ids=[hypothesis.id],
                completed_steps=step,
                created_at=created_at,
            )
            selection = ManualHypothesisSelection(
                completed_steps=step,
                source_hash=library.source_hash,
                source_dir=library.source_dir,
                hypotheses=[hypothesis],
            )
            failure = failures.get(hypothesis.id, {})
            reservations.append(
                HypothesisRootReservation(
                    selection=selection,
                    hypothesis_id=hypothesis.id,
                    completed_steps=step,
                    retry_attempts=(
                        int(failure.get("attempts", 0))
                        if isinstance(failure, dict)
                        else 0
                    ),
                )
            )
            _append_reserved_placeholder(working_journal, hypothesis_id=hypothesis.id)
        return reservations

    retry_ids = sorted(failures)
    retry_ids.extend(
        retry_id
        for retry_id in _unmaterialized_root_offer_ids(
            cfg,
            journal=journal,
            compatible_by_id=compatible_by_id,
            reserved_hypothesis_ids=reserved_hypothesis_ids,
        )
        if retry_id not in failures
    )

    for retry_id in retry_ids:
        if len(reservations) >= count:
            break
        if retry_id in reserved_hypothesis_ids:
            continue
        retry = compatible_by_id.get(retry_id)
        if retry is None:
            continue
        step = completed_steps + len(reservations)
        created_at = dt.datetime.now().isoformat(timespec="seconds")
        _write_manual_source_ref(cfg=cfg, library=library, created_at=created_at)
        _append_manual_offer(
            cfg=cfg,
            completed_steps=step,
            offered_ids=[retry.id],
            source_hash=library.source_hash,
            created_at=created_at,
        )
        _record_manual_offer_usage(
            cfg=cfg,
            offered_ids=[retry.id],
            completed_steps=step,
            created_at=created_at,
        )
        selection = ManualHypothesisSelection(
            completed_steps=step,
            source_hash=library.source_hash,
            source_dir=library.source_dir,
            hypotheses=[retry],
        )
        failure = failures.get(retry.id, {})
        reservations.append(
            HypothesisRootReservation(
                selection=selection,
                hypothesis_id=retry.id,
                completed_steps=step,
                retry_attempts=(
                    int(failure.get("attempts", 0))
                    if isinstance(failure, dict)
                    else 0
                ),
            )
        )
        _append_reserved_placeholder(working_journal, hypothesis_id=retry.id)

    while len(reservations) < count:
        try:
            selection = select_hypothesis_for_node(
                cfg,
                journal=working_journal,
                parent_node=None,
                completed_steps=completed_steps + len(reservations),
                repo_root=repo_root,
            )
        except ValueError:
            break
        if len(selection.hypotheses) != 1:
            raise ValueError("Hypothesis mode requires exactly one selected root.")
        hypothesis_id = selection.hypotheses[0].id
        reservations.append(
            HypothesisRootReservation(
                selection=selection,
                hypothesis_id=hypothesis_id,
                completed_steps=selection.completed_steps,
            )
        )
        _append_reserved_placeholder(working_journal, hypothesis_id=hypothesis_id)

    return reservations


def _latest_manual_offer(cfg: Config) -> dict[str, Any] | None:
    path = _manual_run_dir(cfg) / "offers.jsonl"
    if not path.exists():
        return None
    latest: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            latest = payload
    return latest


def _manual_offer_exists(cfg: Config, completed_steps: int) -> bool:
    path = _manual_run_dir(cfg) / "offers.jsonl"
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, dict)
            and int(payload.get("checkpoint_step", -1)) == completed_steps
        ):
            return True
    return False


def load_latest_manual_research_hints(
    cfg: Config,
    *,
    repo_root: Path = REPO_ROOT,
) -> ManualHypothesisSelection | None:
    offer = _latest_manual_offer(cfg)
    if offer is None:
        return None
    offered_ids = offer.get("offered")
    if not isinstance(offered_ids, list):
        return None
    library = load_manual_hypothesis_library(cfg, repo_root=repo_root)
    by_id = {hypothesis.id: hypothesis for hypothesis in library.hypotheses}
    hypotheses = [
        by_id[hypothesis_id]
        for hypothesis_id in offered_ids
        if isinstance(hypothesis_id, str) and hypothesis_id in by_id
    ]
    if not hypotheses:
        return None
    return ManualHypothesisSelection(
        completed_steps=int(offer.get("checkpoint_step", 0)),
        source_hash=str(offer.get("source_hash") or library.source_hash),
        source_dir=library.source_dir,
        hypotheses=hypotheses,
    )


def record_manual_prompt_node(cfg: Config, node: Node) -> None:
    if getattr(node, "research_mode", None) not in MANUAL_USAGE_RESEARCH_MODES:
        return
    offered_ids = getattr(node, "research_hypotheses_offered", []) or []
    if not offered_ids:
        return
    usage = _load_manual_usage(cfg)
    for hypothesis_id in offered_ids:
        if not isinstance(hypothesis_id, str):
            continue
        entry = usage.setdefault(hypothesis_id, {})
        node_ids = entry.setdefault("prompt_node_ids", [])
        if isinstance(node_ids, list) and node.id not in node_ids:
            node_ids.append(node.id)
        elif not isinstance(node_ids, list):
            entry["prompt_node_ids"] = [node.id]
        entry.setdefault("offered_count", 0)
        entry.setdefault("llm_claimed_used_count", 0)
        entry.setdefault("llm_claimed_used_node_ids", [])
    _write_manual_usage(cfg, usage)


def record_manual_claimed_usage(cfg: Config, node: Node) -> None:
    if getattr(node, "research_mode", None) not in MANUAL_USAGE_RESEARCH_MODES:
        return
    claimed_ids = getattr(node, "research_hypotheses_llm_claimed_used", []) or []
    if not claimed_ids:
        return
    offered = set(getattr(node, "research_hypotheses_offered", []) or [])
    timestamp = dt.datetime.now().isoformat(timespec="seconds")
    usage = _load_manual_usage(cfg)
    for hypothesis_id in claimed_ids:
        if not isinstance(hypothesis_id, str) or hypothesis_id not in offered:
            continue
        entry = usage.setdefault(hypothesis_id, {})
        entry["llm_claimed_used_count"] = (
            int(entry.get("llm_claimed_used_count", 0)) + 1
        )
        node_ids = entry.setdefault("llm_claimed_used_node_ids", [])
        if isinstance(node_ids, list) and node.id not in node_ids:
            node_ids.append(node.id)
        elif not isinstance(node_ids, list):
            entry["llm_claimed_used_node_ids"] = [node.id]
        entry["last_llm_claimed_used_at"] = timestamp
        entry.setdefault("offered_count", 0)
        entry.setdefault("prompt_node_ids", [])
    _write_manual_usage(cfg, usage)


def format_manual_research_hints_for_prompt(
    selection: ManualHypothesisSelection,
) -> str:
    lines = [
        "Manual research hypotheses offered for this experiment.",
        "Treat them as hypotheses to test, not as proven facts.",
        (
            "You were offered manual research hypotheses with ids. If your "
            "solution intentionally uses any of them, mention the ids in your "
            "plan/rationale. If none are relevant, say that no manual research "
            "hypothesis was used."
        ),
        f"Research source hash: {selection.source_hash}",
        "",
        "Offered hypotheses:",
    ]
    for hypothesis in selection.hypotheses:
        lines.append(f"{hypothesis.id}. {hypothesis.title}")
        lines.append(f"   Summary: {_compact_prompt_text(hypothesis.summary, 320)}")
        rationale = _compact_prompt_text(hypothesis.rationale, 360)
        if rationale:
            lines.append(f"   Why: {rationale}")
        implementation_hint = _compact_prompt_text(
            hypothesis.implementation_hint, 520
        )
        if implementation_hint:
            lines.append(f"   Implementation: {implementation_hint}")
        expected_effect = _compact_prompt_text(hypothesis.expected_effect, 260)
        if expected_effect:
            lines.append(f"   Expected effect: {expected_effect}")
        risk = _compact_prompt_text(hypothesis.risk, 260)
        if risk:
            lines.append(f"   Risk: {risk}")
        if hypothesis.sources:
            lines.append(
                "   Sources: " + _compact_prompt_text(", ".join(hypothesis.sources), 360)
            )
    return "\n".join(lines)


def format_hypothesis_for_prompt(
    selection: ManualHypothesisSelection,
) -> str:
    if len(selection.hypotheses) != 1:
        raise ValueError("Hypothesis mode requires exactly one selected hypothesis.")
    hypothesis = selection.hypotheses[0]
    lines = [
        "Hypothesis verification contract.",
        f"Hypothesis ID: {hypothesis.id}",
        (
            "Implement this exact hypothesis. Do not choose another hypothesis, "
            "do not combine it with unrelated ideas, and do not ignore it."
        ),
        (
            "Your solution sketch must explicitly mention this hypothesis ID. "
            "Do not add hypothesis-id bookkeeping variables or result fields to "
            "the generated code."
        ),
        "",
        f"Title: {hypothesis.title}",
        f"Summary: {_compact_prompt_text(hypothesis.summary, 420)}",
    ]
    rationale = _compact_prompt_text(hypothesis.rationale, 520)
    if rationale:
        lines.append(f"Rationale: {rationale}")
    implementation_hint = _compact_prompt_text(
        hypothesis.implementation_hint,
        700,
    )
    if implementation_hint:
        lines.append(f"Implementation: {implementation_hint}")
    expected_effect = _compact_prompt_text(hypothesis.expected_effect, 320)
    if expected_effect:
        lines.append(f"Expected effect: {expected_effect}")
    risk = _compact_prompt_text(hypothesis.risk, 320)
    if risk:
        lines.append(f"Risk: {risk}")
    if hypothesis.sources:
        lines.append(
            "Sources: " + _compact_prompt_text(", ".join(hypothesis.sources), 420)
        )
    return "\n".join(lines)


def format_hypothesis_for_log_panel(
    selection: ManualHypothesisSelection,
) -> str:
    hypothesis = selection.hypotheses[0]
    lines = [
        f"Hypothesis {hypothesis.id}",
        f"Title: {_compact_prompt_text(hypothesis.title, 180)}",
        f"Summary: {_compact_prompt_text(hypothesis.summary, 260)}",
        f"Try: {_compact_prompt_text(hypothesis.implementation_hint, 360)}",
    ]
    return "\n".join(line for line in lines if line.strip())


def _metric_value(node: Node) -> float | None:
    return None if node.metric is None else node.metric.value


def _prompt_score(value: float | None) -> float | None:
    return None if value is None else round(float(value), PROMPT_SCORE_DECIMALS)


def _timestamp_from_ctime(ctime: float) -> str:
    return dt.datetime.fromtimestamp(ctime).strftime("%Y%m%dT%H%M%S")


def count_scored_working_nodes(journal: Journal) -> int:
    return sum(1 for node in journal.good_nodes if _metric_value(node) is not None)


def _compact_prompt_text(value: Any, max_chars: int = 500) -> str:
    text = (
        str(value or "")
        .replace("\\n", " ")
        .replace("\\r", " ")
        .replace("\\t", " ")
    )
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def build_data_overview(cfg: Config) -> str | None:
    candidate_dirs = [Path(cfg.data_dir)]
    workspace_input_dir = Path(cfg.workspace_dir) / "input"
    if workspace_input_dir.exists():
        candidate_dirs.append(workspace_input_dir)
    else:
        candidate_dirs.append(Path(cfg.workspace_dir))

    seen: set[Path] = set()
    for base_dir in candidate_dirs:
        resolved = base_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            try:
                return data_preview.generate(resolved)
            except Exception:  # noqa: BLE001 - research should not stop the run
                continue
    return None


def _node_payload(
    *,
    node: Node,
    registry_entries: list[dict[str, Any]],
    run_id: str,
    preprocess_only: bool,
) -> dict[str, Any] | None:
    code = node.code
    if preprocess_only:
        try:
            code = extract_preprocess_source(code)
        except ValueError:
            return None
    payload = {
        "local_cv_score": _prompt_score(_metric_value(node)),
        "code": code,
    }
    public_score = _public_score_for_node(
        registry_entries=registry_entries,
        run_id=run_id,
        node=node,
    )
    if public_score is not None:
        payload["kaggle_public_score"] = _prompt_score(public_score)
    return payload


def _node_payloads(
    *,
    nodes: list[Node],
    registry_entries: list[dict[str, Any]],
    run_id: str,
    preprocess_only: bool,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for node in nodes:
        payload = _node_payload(
            node=node,
            registry_entries=registry_entries,
            run_id=run_id,
            preprocess_only=preprocess_only,
        )
        if payload is not None:
            payloads.append(payload)
    return payloads


def _sort_best(nodes: list[Node]) -> list[Node]:
    return sorted(nodes, key=lambda n: n.metric, reverse=True)


def _sort_worst(nodes: list[Node]) -> list[Node]:
    return sorted(nodes, key=lambda n: n.metric)


def _checkpoint_name(completed_steps: int) -> str:
    return f"checkpoint-{completed_steps:06d}"


def _checkpoint_label(checkpoint: Path) -> str:
    return checkpoint.name.removeprefix("checkpoint-")


def checkpoint_dir_for(cfg: Config, completed_steps: int) -> Path:
    return Path(cfg.log_dir) / "research" / _checkpoint_name(completed_steps)


def _checkpoint_step(checkpoint: Path) -> int:
    return int(_checkpoint_label(checkpoint))


def _load_submission_registry(cfg: Config) -> list[dict[str, Any]]:
    registry_path = Path(cfg.log_dir).resolve().parent / "submission_registry.json"
    if not registry_path.exists():
        return []
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = data.get("submissions", []) if isinstance(data, dict) else data
    return entries if isinstance(entries, list) else []


def _parse_public_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _public_score_for_node(
    *,
    registry_entries: list[dict[str, Any]],
    run_id: str,
    node: Node,
) -> float | None:
    timestamp = _timestamp_from_ctime(node.ctime)
    for entry in registry_entries:
        if entry.get("run") != run_id:
            continue
        if str(entry.get("remote_status", "")).upper() != "COMPLETE":
            continue
        public_score = _parse_public_score(entry.get("public_score"))
        if public_score is None:
            continue

        if entry.get("node_id") == node.id:
            return public_score
        if (
            str(entry.get("step")) == str(node.step)
            and entry.get("timestamp") == timestamp
        ):
            return public_score
    return None


def _scored_nodes_with_counts(journal: Journal) -> list[tuple[int, Node]]:
    scored: list[tuple[int, Node]] = []
    count = 0
    for node in journal.nodes:
        if node in journal.good_nodes and _metric_value(node) is not None:
            count += 1
            scored.append((count, node))
    return scored


def _max_local_score(nodes: list[Node]) -> float | None:
    if not nodes:
        return None
    best = max(nodes, key=lambda node: node.metric)
    return _prompt_score(_metric_value(best))


def _max_public_score(
    *,
    registry_entries: list[dict[str, Any]],
    run_id: str,
    nodes: list[Node],
) -> float | None:
    scores = [
        score
        for node in nodes
        if (
            score := _public_score_for_node(
                registry_entries=registry_entries,
                run_id=run_id,
                node=node,
            )
        )
        is not None
    ]
    return _prompt_score(max(scores)) if scores else None


def _completed_research_checkpoints(
    *, cfg: Config, before_step: int
) -> list[tuple[int, Path]]:
    research_dir = Path(cfg.log_dir) / "research"
    if not research_dir.exists():
        return []
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint in sorted(research_dir.glob("checkpoint-*")):
        if _checkpoint_status(checkpoint) != "completed":
            continue
        step = _checkpoint_step(checkpoint)
        if step < before_step:
            checkpoints.append((step, checkpoint))
    return checkpoints


def collect_previous_research_summaries(
    *,
    cfg: Config,
    journal: Journal,
    completed_steps: int,
) -> list[dict[str, Any]]:
    limit = max(0, int(cfg.research.previous_summary_count))
    if limit == 0:
        return []

    checkpoints = _completed_research_checkpoints(
        cfg=cfg,
        before_step=completed_steps,
    )
    if not checkpoints:
        return []

    registry_entries = _load_submission_registry(cfg)
    scored_nodes = _scored_nodes_with_counts(journal)
    selected_checkpoints = checkpoints[-limit:]
    summaries: list[dict[str, Any]] = []
    for index, (checkpoint_step, checkpoint) in enumerate(
        reversed(selected_checkpoints),
        start=1,
    ):
        next_step = (
            completed_steps if index == 1 else selected_checkpoints[-index + 1][0]
        )
        window_nodes = [
            node
            for scored_count, node in scored_nodes
            if checkpoint_step < scored_count <= next_step
        ]
        try:
            response = _read_json(checkpoint / "response.json")
        except (OSError, json.JSONDecodeError):
            continue
        parsed = response.get("parsed_response", {})
        if not isinstance(parsed, dict):
            continue
        summary = _compact_prompt_text(parsed.get("summary"), max_chars=700)
        if not summary:
            continue
        summaries.append(
            {
                "checkpoint": checkpoint.name,
                "summary": summary,
                "max_local_cv_score_after": _max_local_score(window_nodes),
                "max_kaggle_public_score_after": _max_public_score(
                    registry_entries=registry_entries,
                    run_id=cfg.exp_name,
                    nodes=window_nodes,
                ),
            }
        )
    return summaries


def collect_research_context(
    *,
    cfg: Config,
    task_desc: Any,
    journal: Journal,
    completed_steps: int,
) -> dict[str, Any]:
    good_nodes = [
        node for node in journal.good_nodes if _metric_value(node) is not None
    ]
    top_best = _sort_best(good_nodes)[: cfg.research.top_k_best]
    top_best_ids = {node.id for node in top_best}
    worst_candidates = [
        node for node in _sort_worst(good_nodes) if node.id not in top_best_ids
    ]
    top_worst = worst_candidates[: cfg.research.top_k_worst]
    best_node = journal.get_best_node()
    registry_entries = _load_submission_registry(cfg)
    preprocess_only = is_autogluon_preprocess_mode(cfg)
    return {
        "run_id": cfg.exp_name,
        "checkpoint_step": completed_steps,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "task_desc": task_desc,
        "data_overview": build_data_overview(cfg),
        "metric_direction": (
            None
            if best_node is None or best_node.metric is None
            else ("maximize" if best_node.metric.maximize else "minimize")
        ),
        "best_working_solutions": _node_payloads(
            nodes=top_best,
            registry_entries=registry_entries,
            run_id=cfg.exp_name,
            preprocess_only=preprocess_only,
        ),
        "worst_working_solutions": _node_payloads(
            nodes=top_worst,
            registry_entries=registry_entries,
            run_id=cfg.exp_name,
            preprocess_only=preprocess_only,
        ),
        "previous_research_summaries": collect_previous_research_summaries(
            cfg=cfg,
            journal=journal,
            completed_steps=completed_steps,
        ),
    }


def build_research_prompt(context: dict[str, Any]) -> str:
    prompt_context = {
        key: context.get(key)
        for key in [
            "task_desc",
            "data_overview",
            "metric_direction",
            "best_working_solutions",
            "worst_working_solutions",
            "previous_research_summaries",
        ]
        if key in context and context.get(key) is not None
    }
    context_json = json.dumps(
        prompt_context, indent=2, ensure_ascii=False, default=_json_default
    )
    return (
        f"{RESEARCH_PROMPT_INTRO}\n\n"
        "# Research task\n"
        "Use live web search to identify techniques, validation traps, feature "
        "engineering ideas, model families, and ensemble/calibration strategies "
        "that are relevant to this competition or closely related machine "
        "learning problems.\n\n"
        "# Output contract\n"
        "Return exactly 5 concise new solution ideas. Do not target a specific "
        "previous node or code block. Use the prior results only to avoid "
        "repeating approaches that have already been tried. Do not debug broken "
        "code.\n\n"
        "# Prior research history\n"
        "If previous_research_summaries is present, it lists recent completed "
        "research proposals. Each entry includes its summary plus the maximum "
        "local CV score and Kaggle public score observed afterwards when "
        "available. Try to propose ideas that are unique relative to those "
        "earlier summaries, or explicitly develop the strongest methods from "
        "them into a new testable direction.\n\n"
        "# Context field meanings\n"
        "best_working_solutions contains the highest-scoring code snippets that "
        "ran successfully. worst_working_solutions contains the lowest-scoring "
        "code snippets that still ran successfully. local_cv_score is the "
        "validation metric. kaggle_public_score is included only when a "
        "completed Kaggle public leaderboard score is available for that exact "
        "node. Use these examples only to understand what has already been tried "
        "and what performed well or poorly.\n\n"
        "# Required JSON output shape\n"
        "Return JSON with: summary; hypotheses[].title; hypotheses[].rationale; "
        "hypotheses[].implementation_hint; hypotheses[].expected_effect; "
        "hypotheses[].risk; hypotheses[].sources. The hypotheses array must "
        "contain exactly 5 items.\n\n"
        "# Compact AIDE run context\n"
        f"```json\n{context_json}\n```\n"
    )


def _codex_profile_text(model: str, reasoning_effort: str | None) -> str:
    lines = [f'model = "{model}"']
    if reasoning_effort is not None:
        lines.append(f'model_reasoning_effort = "{reasoning_effort}"')
    lines.extend(
        [
            'approval_policy = "never"',
            'sandbox_mode = "read-only"',
            "# This profile is archival. The actual invocation uses --ignore-user-config",
            "# plus explicit CLI overrides so no global MCP servers are loaded.",
        ]
    )
    return "\n".join(lines) + "\n"


def _codex_command(cfg: Config, checkpoint_dir: Path) -> list[str]:
    command = [
        "codex",
        "--search",
        "--ask-for-approval",
        "never",
        "exec",
        "--ignore-user-config",
        "--sandbox",
        "read-only",
        "--cd",
        str(checkpoint_dir),
        "--model",
        cfg.research.model,
        "--output-schema",
        "schema.json",
        "--output-last-message",
        "response_raw.txt",
        "--json",
        "-",
    ]
    if cfg.research.reasoning_effort is not None:
        command[command.index("--output-schema") : command.index("--output-schema")] = [
            "-c",
            f'model_reasoning_effort="{cfg.research.reasoning_effort}"',
        ]
    return command


def _parse_response(raw_response: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def run_research_checkpoint(
    *,
    cfg: Config,
    context: dict[str, Any],
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    completed_steps = int(context["checkpoint_step"])
    checkpoint_dir = checkpoint_dir_for(cfg, completed_steps)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    timings_seconds: dict[str, float] = dict(context.get("timings_seconds", {}))

    phase_started = time.monotonic()
    prompt = build_research_prompt(context)
    command = _codex_command(cfg, checkpoint_dir)
    timings_seconds["build_prompt"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    _write_json(
        checkpoint_dir / "status.json",
        {
            "status": "running",
            "run_id": cfg.exp_name,
            "checkpoint_step": completed_steps,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        },
    )
    _write_json(checkpoint_dir / "context.json", context)
    (checkpoint_dir / "request.md").write_text(prompt, encoding="utf-8")
    _write_json(
        checkpoint_dir / "request.json",
        {
            "run_id": cfg.exp_name,
            "checkpoint_step": completed_steps,
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "command": command,
            "model": cfg.research.model,
            "reasoning_effort": cfg.research.reasoning_effort,
            "prompt": prompt,
        },
    )
    _write_json(checkpoint_dir / "schema.json", RESEARCH_RESPONSE_SCHEMA)
    (checkpoint_dir / "codex_profile.toml").write_text(
        _codex_profile_text(cfg.research.model, cfg.research.reasoning_effort),
        encoding="utf-8",
    )
    timings_seconds["write_inputs"] = time.monotonic() - phase_started

    exit_code: int | None = None
    stderr = ""
    stdout = ""
    error: str | None = None
    phase_started = time.monotonic()
    try:
        completed = runner(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=cfg.research.timeout,
            cwd=checkpoint_dir,
        )
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        error = f"Codex research timed out after {cfg.research.timeout} seconds."
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    except Exception as exc:  # noqa: BLE001 - external tool failure must not stop AIDE
        error = f"{exc.__class__.__name__}: {exc}"
    timings_seconds["codex_subprocess"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    (checkpoint_dir / "codex_events.jsonl").write_text(stdout, encoding="utf-8")
    (checkpoint_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    raw_response_path = checkpoint_dir / "response_raw.txt"
    raw_response = (
        raw_response_path.read_text(encoding="utf-8")
        if raw_response_path.exists()
        else ""
    )
    timings_seconds["read_response"] = time.monotonic() - phase_started

    phase_started = time.monotonic()
    parsed_response = _parse_response(raw_response)
    if isinstance(parsed_response, dict):
        raw_response_path.write_text(
            _format_research_response_for_file(
                checkpoint_name=_checkpoint_name(completed_steps),
                parsed_response=parsed_response,
            ),
            encoding="utf-8",
        )
    timings_seconds["parse_response"] = time.monotonic() - phase_started
    status = "completed" if exit_code == 0 and parsed_response is not None else "failed"
    if error is not None:
        status = "failed"
    if error is None and exit_code not in (None, 0):
        error = f"Codex exited with status {exit_code}."
    if error is None and parsed_response is None:
        error = "Codex response was not valid JSON."

    duration = time.monotonic() - started_at
    timings_seconds["total"] = duration
    response_payload = {
        "status": status,
        "run_id": cfg.exp_name,
        "checkpoint_step": completed_steps,
        "exit_code": exit_code,
        "duration_seconds": duration,
        "timings_seconds": timings_seconds,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "stderr": stderr,
        "error": error,
    }
    _write_json(checkpoint_dir / "response.json", response_payload)
    _write_json(
        checkpoint_dir / "status.json",
        {
            "status": status,
            "run_id": cfg.exp_name,
            "checkpoint_step": completed_steps,
            "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": duration,
            "timings_seconds": timings_seconds,
            "exit_code": exit_code,
            "error": error,
        },
    )
    return {
        "status": status,
        "checkpoint_dir": str(checkpoint_dir),
        "response": response_payload,
    }


def _checkpoint_status(path: Path) -> str | None:
    status_path = path / "status.json"
    if not status_path.exists():
        return None
    try:
        status = _read_json(status_path)
    except json.JSONDecodeError:
        return "unknown"
    return status.get("status") if isinstance(status, dict) else "unknown"


def _latest_checkpoint_with_status(log_dir: Path | str) -> tuple[Path, str] | None:
    research_dir = Path(log_dir) / "research"
    if not research_dir.exists():
        return None

    candidates: list[tuple[Path, str]] = []
    for checkpoint in sorted(research_dir.glob("checkpoint-*")):
        status = _checkpoint_status(checkpoint)
        if status is not None:
            candidates.append((checkpoint, status))
    return candidates[-1] if candidates else None


def load_latest_research_hints(log_dir: Path | str) -> dict[str, Any] | None:
    research_dir = Path(log_dir) / "research"
    if not research_dir.exists():
        return None
    completed: list[Path] = []
    for checkpoint in sorted(research_dir.glob("checkpoint-*")):
        if _checkpoint_status(checkpoint) == "completed":
            completed.append(checkpoint)
    if not completed:
        return None

    latest = completed[-1]
    response = _read_json(latest / "response.json")
    parsed = response.get("parsed_response", {})
    if not isinstance(parsed, dict):
        return None
    return {
        "checkpoint": latest.name,
        "summary": parsed.get("summary", ""),
        "hypotheses": parsed.get("hypotheses", []),
    }


def format_research_hints_for_prompt(hints: dict[str, Any]) -> str:
    checkpoint = str(hints.get("checkpoint", "")).removeprefix("checkpoint-")
    lines = [
        "Use these external Codex research hints only when relevant.",
        "Treat them as hypotheses to test, not as proven facts.",
    ]
    if checkpoint:
        lines.append(f"Research checkpoint: {checkpoint}")

    summary = _compact_prompt_text(hints.get("summary"), max_chars=800)
    if summary:
        lines.extend(["", f"Summary: {summary}"])

    hypotheses = hints.get("hypotheses", [])
    if isinstance(hypotheses, list) and hypotheses:
        lines.extend(["", "Prioritized hypotheses:"])
        for idx, hypothesis in enumerate(hypotheses[:8], start=1):
            if not isinstance(hypothesis, dict):
                lines.append(f"{idx}. {_compact_prompt_text(hypothesis)}")
                continue

            title = _compact_prompt_text(hypothesis.get("title"), max_chars=160)
            lines.append(f"{idx}. {title}")

            rationale = _compact_prompt_text(hypothesis.get("rationale"), max_chars=260)
            if rationale:
                lines.append(f"   Why: {rationale}")

            implementation = _compact_prompt_text(
                hypothesis.get("implementation_hint"), max_chars=360
            )
            if implementation:
                lines.append(f"   Try: {implementation}")

            expected = _compact_prompt_text(
                hypothesis.get("expected_effect"), max_chars=220
            )
            if expected:
                lines.append(f"   Expected: {expected}")

            risk = _compact_prompt_text(hypothesis.get("risk"), max_chars=220)
            if risk:
                lines.append(f"   Risk: {risk}")

    return "\n".join(lines)


def _format_research_response_for_file(
    *,
    checkpoint_name: str,
    parsed_response: dict[str, Any],
) -> str:
    return (
        format_research_hints_for_prompt(
            {
                "checkpoint": checkpoint_name,
                "summary": parsed_response.get("summary", ""),
                "hypotheses": parsed_response.get("hypotheses", []),
            }
        )
        + "\n"
    )


class ResearchAdvisor:
    def __init__(
        self,
        *,
        cfg: Config,
        task_desc: Any,
        runner: Runner = subprocess.run,
        repo_root: Path = REPO_ROOT,
    ):
        self.cfg = cfg
        self.task_desc = task_desc
        self.runner = runner
        self.repo_root = repo_root
        self._threads: list[threading.Thread] = []

    def maybe_start(self, *, journal: Journal, completed_steps: int) -> bool:
        if not self.cfg.research.enabled:
            return False
        if self.cfg.research.every_steps <= 0:
            return False
        if completed_steps <= 0 or completed_steps % self.cfg.research.every_steps != 0:
            return False

        research_mode = getattr(self.cfg.research, "mode", "llm")
        if research_mode == "manual":
            if _manual_offer_exists(self.cfg, completed_steps):
                return False
            select_manual_hypotheses(
                self.cfg,
                completed_steps=completed_steps,
                repo_root=self.repo_root,
            )
            return True
        if research_mode == "hypothesis":
            return False

        checkpoint_dir = checkpoint_dir_for(self.cfg, completed_steps)
        if _checkpoint_status(checkpoint_dir) is not None:
            return False

        context_started = time.monotonic()
        context = collect_research_context(
            cfg=self.cfg,
            task_desc=self.task_desc,
            journal=journal,
            completed_steps=completed_steps,
        )
        context["timings_seconds"] = {
            "collect_context": time.monotonic() - context_started
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            checkpoint_dir / "status.json",
            {
                "status": "queued",
                "run_id": self.cfg.exp_name,
                "checkpoint_step": completed_steps,
                "queued_at": dt.datetime.now().isoformat(timespec="seconds"),
            },
        )
        thread = threading.Thread(
            target=run_research_checkpoint,
            kwargs={"cfg": self.cfg, "context": context, "runner": self.runner},
            daemon=True,
        )
        thread.start()
        self._threads.append(thread)
        return True

    def status_text(self) -> str:
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        research_mode = getattr(self.cfg.research, "mode", "llm")
        if research_mode == "manual":
            latest_offer = _latest_manual_offer(self.cfg)
            if latest_offer is None:
                return "[dim]Research: ○ manual"
            step = int(latest_offer.get("checkpoint_step", 0))
            return f"[green]Research: ✓ manual {step:06d}"
        if research_mode == "hypothesis":
            latest_offer = _latest_manual_offer(self.cfg)
            if latest_offer is None:
                return "[dim]Research: ○ hypothesis"
            step = int(latest_offer.get("checkpoint_step", 0))
            offered = latest_offer.get("offered")
            if (
                isinstance(offered, list)
                and len(offered) == 1
                and isinstance(offered[0], str)
            ):
                return f"[green]Research: ✓ {step:06d} @ {offered[0]}"
            return f"[green]Research: ✓ {step:06d}"

        latest_checkpoint = _latest_checkpoint_with_status(self.cfg.log_dir)
        latest_name = (
            _checkpoint_label(latest_checkpoint[0])
            if latest_checkpoint is not None
            else None
        )
        if self._threads:
            suffix = f" {latest_name}" if latest_name is not None else ""
            return f"[cyan]Research: ▶{suffix}"

        latest = load_latest_research_hints(self.cfg.log_dir)
        if latest is not None:
            return (
                f"[green]Research: ✓ {latest['checkpoint'].removeprefix('checkpoint-')}"
            )

        if latest_checkpoint is not None:
            checkpoint, status = latest_checkpoint
            if status in {"queued", "running"}:
                icon = "…" if status == "queued" else "▶"
                return f"[cyan]Research: {icon} {_checkpoint_label(checkpoint)}"
            if status == "failed":
                return f"[red]Research: ✗ {_checkpoint_label(checkpoint)}"
            return f"[yellow]Research: ? {_checkpoint_label(checkpoint)}"
        return "[dim]Research: ○"
