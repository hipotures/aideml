"""External Codex solution synthesis for long AIDE runs."""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .autogluon_preprocess import (
    AGENT_MODE as AUTOGLUON_PREPROCESS_MODE,
    build_autogluon_wrapper,
    extract_preprocess_source,
    infer_sample_submission_columns,
    is_autogluon_preprocess_mode,
    sanitize_preprocess_prompt_text,
    validate_preprocess_source,
)
from .journal import Journal, Node
from .research import (
    _checkpoint_label,
    _checkpoint_name,
    _checkpoint_status,
    _codex_profile_text,
    _json_default,
    _metric_value,
    _prompt_score,
    _read_json,
    _write_json,
    build_data_overview,
)
from .utils import serialize
from .utils.config import Config
from .utils.metric import WorstMetricValue
from .utils.prediction_similarity import submission_prediction_rmse
from .utils.response import extract_code, extract_text_up_to_code, is_valid_python_script

SYNTHESIS_PROMPT_INTRO = (
    "You are a Kaggle grandmaster and senior machine learning engineer. Your "
    "job is to use live web search when useful, study the strongest successful "
    "AIDE solution scripts, and produce one coherent Python solution that "
    "combines the best compatible ideas."
)
SYNTHESIS_PREPROCESS_PROMPT_INTRO = (
    "You are a Kaggle grandmaster and senior machine learning engineer. Your "
    "job is to study the strongest successful AIDE preprocess functions for a "
    "fixed AutoGluon wrapper and produce one coherent, leakage-safe "
    "preprocess(df) function that combines the best compatible feature ideas."
)
SYNTHESIS_PLAN_PREFIX = "External Codex synthesis checkpoint"
TARGET_LEAKAGE_PATTERNS = (
    "next_pitstop",
    "next_pit_stop",
    "next_pit",
    "next pitstop",
    "next pit stop",
    "pitstop_known",
)


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SynthesisNode:
    node: Node
    completed_steps: int
    checkpoint_dir: Path
    ready_for_execution: bool = True


def checkpoint_dir_for(cfg: Config, completed_steps: int) -> Path:
    return Path(cfg.log_dir) / "synthesis" / _checkpoint_name(completed_steps)


def _synthesis_dir(log_dir: Path | str) -> Path:
    return Path(log_dir) / "synthesis"


def _load_journal(path: Path) -> Journal | None:
    try:
        return serialize.load_json(path, Journal)
    except Exception:  # noqa: BLE001 - stale runs must not stop synthesis
        return None


def _candidate_run_ids(cfg: Config) -> list[str]:
    source_runs = list(cfg.synthesis.source_runs or [])
    if source_runs:
        return source_runs

    source_scope = str(getattr(cfg.synthesis, "source_scope", "current"))
    if source_scope == "current":
        return [cfg.exp_name]
    if source_scope != "all":
        raise ValueError(
            f"Unsupported synthesis.source_scope={source_scope!r}; "
            "expected 'current' or 'all'"
        )

    top_log_dir = Path(cfg.log_dir).resolve().parent
    if not top_log_dir.exists():
        return [cfg.exp_name]

    run_ids = sorted(path.parent.name for path in top_log_dir.glob("*/journal.json"))
    if cfg.exp_name not in run_ids:
        run_ids.append(cfg.exp_name)
    return run_ids


def _journal_for_run(
    *, cfg: Config, current_journal: Journal, run_id: str
) -> Journal | None:
    if run_id == cfg.exp_name:
        return current_journal
    return _load_journal(Path(cfg.log_dir).resolve().parent / run_id / "journal.json")


def _working_nodes_with_metrics(journal: Journal) -> list[Node]:
    return [
        node
        for node in journal.good_nodes
        if node.metric is not None and _metric_value(node) is not None
    ]


def _timestamp_from_ctime(ctime: float) -> str:
    return dt.datetime.fromtimestamp(ctime).strftime("%Y%m%dT%H%M%S")


def _target_leakage_reasons(code: str) -> list[str]:
    lowered = code.lower()
    reasons: list[str] = []
    for pattern in TARGET_LEAKAGE_PATTERNS:
        if pattern in lowered:
            reasons.append(f"suspicious token '{pattern}'")

    lines = code.splitlines()
    for idx, line in enumerate(lines):
        normalized_line = re.sub(r"\s+", "", line.lower())
        if (
            "shift(-1" not in normalized_line
            and "shift(periods=-1" not in normalized_line
        ):
            continue
        window = "\n".join(lines[max(0, idx - 4) : idx + 5]).lower()
        if "pitstop" in window or "pit_stop" in window:
            reasons.append("future PitStop shift(-1)")

    return list(dict.fromkeys(reasons))


def _has_target_leakage_pattern(code: str) -> bool:
    return bool(_target_leakage_reasons(code))


def _validate_synthesis_code_for_injection(code: str) -> None:
    reasons = _target_leakage_reasons(code)
    if reasons:
        raise ValueError(
            "Codex synthesis response contains target leakage risk: "
            + "; ".join(reasons)
        )


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


def _completed_public_score_for_node(
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


def _solution_payload(
    *,
    cfg: Config,
    registry_entries: list[dict[str, Any]],
    run_id: str,
    node: Node,
    preprocess_only: bool = False,
) -> dict[str, Any]:
    code = node.code
    source_metadata = _source_metadata_for_node(cfg=cfg, run_id=run_id, node=node)
    payload = {
        **source_metadata,
        "local_cv_score": _prompt_score(_metric_value(node)),
    }
    if preprocess_only:
        payload["response"] = _preprocess_candidate_response(
            cfg=cfg,
            run_id=run_id,
            node=node,
        )
    else:
        payload["code"] = code
    public_score = _completed_public_score_for_node(
        registry_entries=registry_entries,
        run_id=run_id,
        node=node,
    )
    if public_score is not None:
        payload["kaggle_public_score"] = _prompt_score(public_score)
    return payload


def _read_text_if_exists(path: Path) -> str | None:
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            return text if text.strip() else None
    except OSError:
        return None
    return None


def _node_timestamp(node: Node) -> str:
    return _timestamp_from_ctime(node.ctime)


def _synthesis_status_for_node(
    *,
    cfg: Config,
    run_id: str,
    node: Node,
) -> tuple[Path, dict[str, Any]] | None:
    synthesis_dir = Path(cfg.log_dir).resolve().parent / run_id / "synthesis"
    if not synthesis_dir.exists():
        return None

    for status_path in sorted(synthesis_dir.glob("checkpoint-*/status.json")):
        status = _read_json(status_path)
        if not isinstance(status, dict):
            continue
        node_id_matches = node.id in {
            status.get("recorded_node_id"),
            status.get("injected_node_id"),
        }
        node_step_matches = node.step is not None and str(node.step) in {
            str(status.get("recorded_node_step")),
            str(status.get("injected_node_step")),
        }
        if not node_id_matches and not node_step_matches:
            continue
        return status_path, status
    return None


def _checkpoint_step_from_path(checkpoint_dir: Path) -> int | None:
    match = re.search(r"checkpoint-(\d+)$", checkpoint_dir.name)
    return int(match.group(1)) if match else None


def _source_metadata_for_node(
    *,
    cfg: Config,
    run_id: str,
    node: Node,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    synthesis_status = _synthesis_status_for_node(cfg=cfg, run_id=run_id, node=node)
    if synthesis_status is not None:
        status_path, _status = synthesis_status
        metadata["source_kind"] = "synthesis"
        checkpoint_step = _checkpoint_step_from_path(status_path.parent)
        if checkpoint_step is not None:
            metadata["source_checkpoint_step"] = checkpoint_step
    return metadata


def _raw_synthesis_response_for_node(
    *,
    cfg: Config,
    run_id: str,
    node: Node,
) -> str | None:
    synthesis_status = _synthesis_status_for_node(cfg=cfg, run_id=run_id, node=node)
    if synthesis_status is None:
        return None
    status_path, _status = synthesis_status
    return _read_text_if_exists(status_path.parent / "response_raw.txt")


def _raw_generation_response_for_node(
    *,
    cfg: Config,
    run_id: str,
    node: Node,
) -> str | None:
    raw_response_path = (
        Path(cfg.log_dir).resolve().parent
        / run_id
        / "artifacts"
        / _node_timestamp(node)
        / "response_raw.txt"
    )
    return _read_text_if_exists(raw_response_path)


def _preprocess_candidate_response(
    *,
    cfg: Config,
    run_id: str,
    node: Node,
) -> str:
    raw_response = _raw_synthesis_response_for_node(
        cfg=cfg,
        run_id=run_id,
        node=node,
    ) or _raw_generation_response_for_node(
        cfg=cfg,
        run_id=run_id,
        node=node,
    )
    if raw_response is not None:
        return raw_response

    preprocess_source = extract_preprocess_source(node.code)
    plan = str(node.plan or "").strip()
    if not plan:
        return f"```python\n{preprocess_source}\n```\n"
    return f"{plan}\n\n```python\n{preprocess_source}\n```\n"


def _rounded_score(node: Node, score_round_decimals: int) -> float | None:
    value = _metric_value(node)
    return None if value is None else round(value, score_round_decimals)


def _normalized_score(node: Node) -> float | None:
    value = _metric_value(node)
    if value is None or node.metric is None:
        return None
    return value if node.metric.maximize else -value


def _rounded_normalized_score(
    node: Node,
    score_round_decimals: int,
) -> float | None:
    value = _normalized_score(node)
    return None if value is None else round(value, score_round_decimals)


def _ancestor_nodes(node: Node) -> list[Node]:
    ancestors: list[Node] = []
    parent = node.parent
    seen = {node.id}
    while parent is not None and parent.id not in seen:
        ancestors.append(parent)
        seen.add(parent.id)
        parent = parent.parent
    return ancestors


def _lineage_nodes(node: Node) -> set[Node]:
    return {node, *_ancestor_nodes(node)}


def _are_in_same_branch_family(left: Node, right: Node) -> bool:
    return bool(_lineage_nodes(left) & _lineage_nodes(right))


def _has_ancestor_relation(left: Node, right: Node) -> bool:
    return left in _ancestor_nodes(right) or right in _ancestor_nodes(left)


def _is_strictly_worse_than(
    node: Node,
    ancestor: Node,
    *,
    score_round_decimals: int,
) -> bool:
    node_score = _rounded_normalized_score(node, score_round_decimals)
    ancestor_score = _rounded_normalized_score(ancestor, score_round_decimals)
    if node_score is None or ancestor_score is None:
        return False
    return node_score < ancestor_score


def _submission_path_for_node(cfg: Config, run_id: str, node: Node) -> Path:
    timestamp = _timestamp_from_ctime(node.ctime)
    return (
        Path(cfg.log_dir).resolve().parent
        / run_id
        / "artifacts"
        / timestamp
        / "submission.csv"
    )


def _has_similar_submission_predictions(
    *,
    cfg: Config,
    left_run_id: str,
    left: Node,
    right_run_id: str,
    right: Node,
) -> bool:
    rmse = submission_prediction_rmse(
        _submission_path_for_node(cfg, left_run_id, left),
        _submission_path_for_node(cfg, right_run_id, right),
        prediction_round_decimals=cfg.synthesis.prediction_round_decimals,
        sample_size=cfg.synthesis.prediction_similarity_sample_size,
        min_common_sample_size=cfg.synthesis.prediction_similarity_min_common_sample_size,
    )
    return (
        rmse is not None
        and rmse <= cfg.synthesis.prediction_similarity_rmse_threshold
    )


def _should_collapse_related_node(
    *,
    cfg: Config,
    run_id: str,
    node: Node,
    selected_node: Node,
) -> bool:
    if _has_ancestor_relation(node, selected_node):
        if selected_node in _ancestor_nodes(node):
            ancestor = selected_node
            descendant = node
        else:
            ancestor = node
            descendant = selected_node

        if _is_strictly_worse_than(
            descendant,
            ancestor,
            score_round_decimals=cfg.synthesis.score_round_decimals,
        ):
            return True

    return _rounded_score(
        node,
        cfg.synthesis.score_round_decimals,
    ) == _rounded_score(
        selected_node,
        cfg.synthesis.score_round_decimals,
    ) and _has_similar_submission_predictions(
        cfg=cfg,
        left_run_id=run_id,
        left=node,
        right_run_id=run_id,
        right=selected_node,
    )


def _prefer_ancestor_for_related_predictions(
    cfg: Config,
    candidates: list[tuple[str, Node]],
) -> list[tuple[str, Node]]:
    selected: list[tuple[str, Node]] = []
    for run_id, node in candidates:
        related_indices = []
        for idx, (selected_run_id, selected_node) in enumerate(selected):
            if run_id != selected_run_id:
                continue
            if not _are_in_same_branch_family(node, selected_node):
                continue
            if not _should_collapse_related_node(
                cfg=cfg,
                run_id=run_id,
                node=node,
                selected_node=selected_node,
            ):
                continue
            related_indices.append(idx)

        if not related_indices:
            selected.append((run_id, node))
            continue

        if any(selected[idx][1] in _ancestor_nodes(node) for idx in related_indices):
            continue

        if any(node in _ancestor_nodes(selected[idx][1]) for idx in related_indices):
            related_index_set = set(related_indices)
            selected = [
                item
                for idx, item in enumerate(selected)
                if idx not in related_index_set
            ]
            selected.append((run_id, node))
    return selected


def collect_top_synthesis_solutions(
    *,
    cfg: Config,
    journal: Journal,
) -> list[dict[str, Any]]:
    registry_entries = _load_submission_registry(cfg)
    selected: list[tuple[str, Node]] = []
    preprocess_only = is_autogluon_preprocess_mode(cfg)
    for run_id in _candidate_run_ids(cfg):
        source_journal = _journal_for_run(
            cfg=cfg,
            current_journal=journal,
            run_id=run_id,
        )
        if source_journal is None:
            continue
        for node in _working_nodes_with_metrics(source_journal):
            if _has_target_leakage_pattern(node.code):
                continue
            if preprocess_only:
                try:
                    extract_preprocess_source(node.code)
                except ValueError:
                    continue
            selected.append((run_id, node))

    selected.sort(key=lambda item: item[1].metric, reverse=True)
    selected = _prefer_ancestor_for_related_predictions(cfg, selected)
    return [
        _solution_payload(
            cfg=cfg,
            registry_entries=registry_entries,
            run_id=run_id,
            node=node,
            preprocess_only=preprocess_only,
        )
        for run_id, node in selected[: cfg.synthesis.top_k]
    ]


def _preprocess_unavailable_columns(cfg: Config) -> list[str]:
    for base_dir in (Path(cfg.workspace_dir) / "input", Path(cfg.data_dir)):
        columns = infer_sample_submission_columns(base_dir)
        if columns is not None:
            return [column for column in columns if column]
    return []


def collect_synthesis_context(
    *,
    cfg: Config,
    task_desc: Any,
    journal: Journal,
    completed_steps: int,
) -> dict[str, Any]:
    best_node = journal.get_best_node()
    task_desc_for_prompt = task_desc
    data_overview_for_prompt = build_data_overview(cfg)
    if is_autogluon_preprocess_mode(cfg):
        unavailable_columns = _preprocess_unavailable_columns(cfg)
        task_desc_for_prompt = sanitize_preprocess_prompt_text(
            task_desc,
            unavailable_columns=unavailable_columns,
        )
        data_overview_for_prompt = sanitize_preprocess_prompt_text(
            data_overview_for_prompt,
            unavailable_columns=unavailable_columns,
        )
    return {
        "run_id": cfg.exp_name,
        "checkpoint_step": completed_steps,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "agent_mode": getattr(cfg.agent, "mode", "legacy"),
        "task_desc": task_desc_for_prompt,
        "data_overview": data_overview_for_prompt,
        "metric_direction": (
            None
            if best_node is None or best_node.metric is None
            else ("maximize" if best_node.metric.maximize else "minimize")
        ),
        "best_working_solutions": collect_top_synthesis_solutions(
            cfg=cfg,
            journal=journal,
        ),
    }


def build_synthesis_prompt(context: dict[str, Any]) -> str:
    prompt_context = {
        key: context.get(key)
        for key in [
            "task_desc",
            "data_overview",
            "metric_direction",
            "best_working_solutions",
        ]
        if key in context and context.get(key) is not None
    }
    context_json = json.dumps(
        prompt_context, indent=2, ensure_ascii=False, default=_json_default
    )
    if context.get("agent_mode") == AUTOGLUON_PREPROCESS_MODE:
        return (
            f"{SYNTHESIS_PREPROCESS_PROMPT_INTRO}\n\n"
            "# Synthesis task\n"
            "Create one coherent leakage-safe feature preprocessing function for "
            "a fixed AutoGluon training wrapper. You are given successful "
            "preprocess functions with local CV scores and, when available, Kaggle "
            "public leaderboard scores. Treat them as evidence about useful "
            "feature-engineering strategies, not as code snippets to concatenate. "
            "Keep strong consensus feature families, reconcile compatible "
            "divergent ideas, remove redundant aliases, include only distinct "
            "signal, and add a small number of logically motivated new features "
            "when supported by the task, data columns, or research context.\n\n"
            "# Internal synthesis procedure\n"
            "Perform this analysis internally before writing code. Do not output "
            "the analysis. First, rank evidence: use kaggle_public_score as the "
            "strongest generalization signal when present and local_cv_score as "
            "secondary evidence. Treat tiny local score differences as weak "
            "evidence; do not respond with micro-threshold tuning. Second, build "
            "a strategy map across candidates: identify consensus feature "
            "families, divergent families, near-duplicates, redundant ratios, "
            "risky leakage-prone ideas, and safe reusable implementation "
            "patterns. Third, decide what to keep, merge, add, and drop: keep "
            "robust consensus ideas, merge similar ideas into canonical features, "
            "add divergent ideas only when they provide different signal, add new "
            "features only as deterministic transformations of available columns, "
            "and drop duplicate aliases, row-order hacks, leakage-prone features, "
            "and pure cosmetic changes.\n\n"
            "# Reject trivial synthesis\n"
            "A valid synthesis must include a strategy-level change compared with "
            "the source candidates. Invalid synthesis includes only changing "
            "numeric thresholds or constants, only renaming columns, only "
            "reordering code, only adding near-duplicate ratios, pasting all "
            "candidate features together without pruning, or copying the top "
            "candidate with minor edits.\n\n"
            "# Avoid feature bloat\n"
            "Do not include every possible feature. Merge repeated transformations "
            "into one canonical feature with a clear name. Avoid synonymous ratios "
            "and aliases that encode the same information. Add a feature family "
            "only when it likely contributes distinct signal.\n\n"
            "# Current task-specific feature strategy\n"
            "The task is to predict whether an F1 driver will pit on the next lap. "
            "Favor feature families that represent tyre age and wear pressure "
            "relative to compound-specific expected life; compound pit windows "
            "and old-tyre flags; degradation rate, lap-time loss, and performance "
            "drop; race phase, estimated total laps, and estimated laps remaining; "
            "stint progress and current pit-stop state; relative field context "
            "within Year/Race/LapNumber groups; safe driver-race chronological "
            "state using only current or past rows; dry-compound usage pressure "
            "computed safely from observed current/past compounds; and stable "
            "categorical interactions or frequency encodings useful to "
            "AutoGluon.\n\n"
            "# Output contract\n"
            "Return the same two-part structure as ordinary AIDE code generation: "
            "first a short natural-language design paragraph, then exactly one "
            "fenced Python code block. The design paragraph should be 2-4 "
            "sentences describing the feature-engineering strategy and why it is "
            "different from the source candidates. The code block must define "
            "exactly one top-level function: "
            "`def preprocess(df: pd.DataFrame) -> pd.DataFrame`. Do not include "
            "imports, helper functions, top-level constants, executable top-level "
            "statements, JSON, extra explanations, titles, or comments outside "
            "the design paragraph and code block. `pd` is already available from "
            "the fixed wrapper, and `np` is also available. Nested helper "
            "functions inside preprocess are allowed only when they keep the "
            "implementation clear and deterministic. Do not read files, write "
            "files, train models, call AutoGluon, create validation splits, or "
            "save submissions. The fixed AIDE wrapper handles all model training, "
            "metric reporting, and submission generation.\n\n"
            "# Data contract\n"
            "`df` contains concatenated train features followed by Kaggle "
            "prediction/test features, with only model feature columns present. "
            "Use only columns visible in the sanitized feature overview or in "
            "candidate preprocess functions. Do not add defensive cleanup for "
            "hidden wrapper columns or columns that are not present in "
            "preprocess(df). Row order must be preserved. "
            "Return a DataFrame with the same row count. Do not infer train/test "
            "split from row counts, index ranges, sorted order, or hidden "
            "assumptions.\n\n"
            "# Leakage rules\n"
            "Use only deterministic transformations of model feature columns. "
            "Do not create label-derived features, encodings, filters, groups, "
            "or pseudo-labels inside preprocess(df). "
            "Do not use future PitStop values, next-lap PitStop reconstruction, shift(-1) on PitStop, "
            "next_PitStop-like features, or test PitStop values to overwrite "
            "predictions. Use PitStop only as a normal current-row historical "
            "feature. Safe chronological features may be computed only from rows "
            "at the current or earlier LapNumber within the same Year/Race/Driver "
            "group. If sorting is needed, sort a copy, compute only non-future "
            "features such as shift(1), diff(), cumsum(), or cumulative flags, "
            "then reindex back to the original df index.\n\n"
            "# Implementation quality\n"
            "The function should be deterministic, vectorized where practical, "
            "and robust to missing values, singleton groups, unseen categories, "
            "and zero denominators. Avoid slow per-row loops over the full "
            "dataset unless there is no vectorized alternative. Use stable dtypes "
            "for categorical columns and preserve useful original categorical "
            "columns as category dtype when appropriate. For group statistics, "
            "fill or guard standard deviations to avoid inf/nan z-scores. At the "
            "end, replace inf and -inf with NaN or safe finite values if needed.\n\n"
            "# Context field meanings\n"
            "best_working_solutions contains the highest-scoring preprocess "
            "functions that ran successfully through the fixed AutoGluon wrapper. "
            "source_kind appears only when a candidate was created by an earlier "
            "external synthesis checkpoint; use source_kind=synthesis to avoid "
            "repeating the same prior synthesis strategy. source_checkpoint_step "
            "identifies that earlier synthesis checkpoint. "
            "local_cv_score is the AutoGluon validation score. kaggle_public_score "
            "is included only when the local submission registry has a completed "
            "Kaggle public leaderboard score for that exact node. Each solution "
            "also includes response, containing the source candidate's design "
            "paragraph plus a fenced preprocess(df) code block when available.\n\n"
            "# Compact AIDE synthesis context\n"
            f"```json\n{context_json}\n```\n"
        )

    return (
        f"{SYNTHESIS_PROMPT_INTRO}\n\n"
        "# Synthesis task\n"
        "Create one complete, self-contained Python script. Combine compatible "
        "high-performing ideas from the provided successful solutions, but do "
        "not merely paste them together. You may use live web search to check "
        "competition-specific or closely related modeling tactics before "
        "writing the script.\n\n"
        "# Output contract\n"
        "Return the same two-part structure as ordinary AIDE code generation: "
        "first a short natural-language design paragraph, then exactly one "
        "fenced Python code block. The design paragraph should be 2-4 sentences "
        "describing the modeling strategy and why it is different from the source "
        "candidates. The code block must contain a complete Python script. Do not "
        "include JSON, extra explanations, titles, or comments outside the design "
        "paragraph and code block. The code must read all inputs from ./input, "
        "print a hold-out or cross-validation metric, and save the final test "
        "predictions as ./working/submission.csv when test data exists. The "
        "script must be executable as a single file within the configured AIDE "
        "timeout. Design the implementation for strong time and memory "
        "efficiency: avoid unnecessary full-data copies, unbounded feature "
        "explosions, oversized intermediate objects, and excessively expensive "
        "model searches.\n\n"
        "# Leakage rules\n"
        "Do not use target leakage. Do not use future PitStop values, next-lap "
        "PitStop reconstruction, shift(-1) on PitStop, next_PitStop-like "
        "features, or test PitStop values to overwrite predictions. Use PitStop "
        "only as a normal current-row historical feature.\n\n"
        "# Context field meanings\n"
        "best_working_solutions contains the highest-scoring scripts that ran "
        "successfully. source_kind appears only when a candidate was created by "
        "an earlier external synthesis checkpoint. source_checkpoint_step "
        "identifies that earlier synthesis checkpoint. local_cv_score is the "
        "AIDE validation score. "
        "kaggle_public_score is included only when the local submission registry "
        "has a completed Kaggle public leaderboard score for that exact node. "
        "Each solution also includes code. Use these examples to identify strong "
        "modeling and feature engineering patterns to merge into a better root "
        "solution.\n\n"
        "# Compact AIDE synthesis context\n"
        f"```json\n{context_json}\n```\n"
    )


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
        cfg.synthesis.model,
        "--output-last-message",
        "response_raw.txt",
        "--json",
        "-",
    ]
    if cfg.synthesis.reasoning_effort is not None:
        command[command.index("--output-last-message") : command.index(
            "--output-last-message"
        )] = [
            "-c",
            f'model_reasoning_effort="{cfg.synthesis.reasoning_effort}"',
        ]
    return command


def parse_synthesis_code(raw_response: str) -> str:
    code = extract_code(raw_response)
    if code and is_valid_python_script(code):
        return code

    stripped = raw_response.strip()
    if stripped and is_valid_python_script(stripped):
        return stripped
    raise ValueError("Codex synthesis response did not contain valid Python code.")


def parse_synthesis_response(raw_response: str) -> tuple[str, str]:
    code = parse_synthesis_code(raw_response)
    plan = extract_text_up_to_code(raw_response).strip()
    if not plan:
        plan = "External Codex synthesis generated a new root solution."
    return plan, code


def run_synthesis_checkpoint(
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
    prompt = build_synthesis_prompt(context)
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
            "model": cfg.synthesis.model,
            "reasoning_effort": cfg.synthesis.reasoning_effort,
            "prompt": prompt,
        },
    )
    (checkpoint_dir / "codex_profile.toml").write_text(
        _codex_profile_text(cfg.synthesis.model, cfg.synthesis.reasoning_effort),
        encoding="utf-8",
    )
    timings_seconds["write_inputs"] = time.monotonic() - phase_started

    exit_code: int | None = None
    stderr = ""
    stdout = ""
    error: str | None = None
    code: str | None = None
    plan: str | None = None
    phase_started = time.monotonic()
    try:
        completed = runner(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=cfg.synthesis.timeout,
            cwd=checkpoint_dir,
        )
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        error = f"Codex synthesis timed out after {cfg.synthesis.timeout} seconds."
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
    if error is None and exit_code not in (None, 0):
        error = f"Codex exited with status {exit_code}."
    if error is None:
        try:
            if context.get("agent_mode") == AUTOGLUON_PREPROCESS_MODE:
                plan = extract_text_up_to_code(raw_response).strip()
                if not plan:
                    plan = "External Codex synthesis generated a new preprocess root."
                preprocess_source = extract_preprocess_source(raw_response)
                columns = infer_sample_submission_columns(
                    Path(cfg.workspace_dir) / "input"
                )
                validate_preprocess_source(
                    preprocess_source,
                    target_col=columns[1] if columns is not None else None,
                )
                code = build_autogluon_wrapper(preprocess_source, cfg)
            else:
                plan, code = parse_synthesis_response(raw_response)
                _validate_synthesis_code_for_injection(code)
            (checkpoint_dir / "response.py").write_text(code, encoding="utf-8")
        except ValueError as exc:
            error = str(exc)
    timings_seconds["parse_response"] = time.monotonic() - phase_started

    status = "ready" if error is None and code is not None else "failed"
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
        "plan": plan,
        "code": code,
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


def _latest_checkpoint_with_status(log_dir: Path | str) -> tuple[Path, str] | None:
    synthesis_dir = _synthesis_dir(log_dir)
    if not synthesis_dir.exists():
        return None

    candidates: list[tuple[Path, str]] = []
    for checkpoint in sorted(synthesis_dir.glob("checkpoint-*")):
        status = _checkpoint_status(checkpoint)
        if status is not None:
            candidates.append((checkpoint, status))
    return candidates[-1] if candidates else None


def _oldest_ready_checkpoint(log_dir: Path | str) -> Path | None:
    synthesis_dir = _synthesis_dir(log_dir)
    if not synthesis_dir.exists():
        return None
    for checkpoint in sorted(synthesis_dir.glob("checkpoint-*")):
        if (
            _checkpoint_status(checkpoint) == "ready"
            and (checkpoint / "response.py").exists()
        ):
            return checkpoint
    return None


def _checkpoint_step(checkpoint: Path) -> int:
    return int(_checkpoint_label(checkpoint))


def _mark_checkpoint_failed(checkpoint: Path, error: str) -> None:
    status = _read_json(checkpoint / "status.json")
    if not isinstance(status, dict):
        status = {}
    status.update(
        {
            "status": "failed",
            "failed_at": dt.datetime.now().isoformat(timespec="seconds"),
            "error": error,
        }
    )
    _write_json(checkpoint / "status.json", status)


def _checkpoint_already_recorded(checkpoint: Path) -> bool:
    status = _read_json(checkpoint / "status.json")
    return isinstance(status, dict) and bool(status.get("recorded_node_id"))


def _failed_checkpoint_error(checkpoint: Path) -> str:
    status = _read_json(checkpoint / "status.json")
    if isinstance(status, dict) and status.get("error"):
        return str(status["error"])

    response = _read_json(checkpoint / "response.json")
    if isinstance(response, dict) and response.get("error"):
        return str(response["error"])

    return "Synthesis checkpoint failed before producing injectable code."


def _failed_checkpoint_code(checkpoint: Path) -> str:
    response_code = checkpoint / "response.py"
    if response_code.exists():
        code = response_code.read_text(encoding="utf-8").strip()
        if code:
            return code

    raw_response = checkpoint / "response_raw.txt"
    if raw_response.exists():
        code = raw_response.read_text(encoding="utf-8").strip()
        if code:
            return code
    return "# Failed synthesis checkpoint did not produce code.\n"


def _checkpoint_plan(checkpoint: Path, *, fallback_step: int) -> str:
    response = _read_json(checkpoint / "response.json")
    if isinstance(response, dict):
        plan = response.get("plan")
        if isinstance(plan, str) and plan.strip():
            return plan.strip()
    return f"{SYNTHESIS_PLAN_PREFIX} {fallback_step:06d}"


def _failed_synthesis_node(checkpoint: Path) -> SynthesisNode | None:
    if _checkpoint_already_recorded(checkpoint):
        return None

    checkpoint_step = _checkpoint_step(checkpoint)
    error = _failed_checkpoint_error(checkpoint)
    node = Node(
        plan=f"{SYNTHESIS_PLAN_PREFIX} {checkpoint_step:06d}",
        code=_failed_checkpoint_code(checkpoint),
    )
    node.metric = WorstMetricValue()
    node.is_buggy = True
    node.status = "failed"
    node.analysis = error
    node._term_out = [f"Failed: {error}\n"]
    node.exec_time = 0.0
    node.exc_type = "Failed"
    node.exc_info = {"error": error}
    node.exc_stack = None
    return SynthesisNode(
        node=node,
        completed_steps=checkpoint_step,
        checkpoint_dir=checkpoint,
        ready_for_execution=False,
    )


def _read_valid_ready_checkpoint_code(checkpoint: Path) -> str | None:
    code = (checkpoint / "response.py").read_text(encoding="utf-8")
    try:
        _validate_synthesis_code_for_injection(code)
    except ValueError as exc:
        _mark_checkpoint_failed(checkpoint, str(exc))
        return None
    return code


class SynthesisAdvisor:
    def __init__(
        self,
        *,
        cfg: Config,
        task_desc: Any,
        runner: Runner = subprocess.run,
    ):
        self.cfg = cfg
        self.task_desc = task_desc
        self.runner = runner
        self._active_checkpoint: int | None = None

    def _is_due(self, completed_steps: int) -> bool:
        return (
            self.cfg.synthesis.enabled
            and self.cfg.synthesis.every_scored_steps > 0
            and completed_steps > 0
            and completed_steps % self.cfg.synthesis.every_scored_steps == 0
        )

    def generate_node_if_due(
        self,
        *,
        journal: Journal,
        completed_steps: int,
    ) -> SynthesisNode | None:
        if not self.cfg.synthesis.enabled:
            return None

        ready_checkpoint = _oldest_ready_checkpoint(self.cfg.log_dir)
        if ready_checkpoint is not None:
            checkpoint_step = _checkpoint_step(ready_checkpoint)
            code = _read_valid_ready_checkpoint_code(ready_checkpoint)
            if code is None:
                return _failed_synthesis_node(ready_checkpoint)
            return SynthesisNode(
                node=Node(
                    plan=_checkpoint_plan(
                        ready_checkpoint,
                        fallback_step=checkpoint_step,
                    ),
                    code=code,
                ),
                completed_steps=checkpoint_step,
                checkpoint_dir=ready_checkpoint,
            )

        if not self._is_due(completed_steps):
            return None

        checkpoint_dir = checkpoint_dir_for(self.cfg, completed_steps)
        status = _checkpoint_status(checkpoint_dir)
        plan: str | None = None
        if status == "injected":
            return None
        if status == "failed":
            return _failed_synthesis_node(checkpoint_dir)
        if status == "ready":
            code = _read_valid_ready_checkpoint_code(checkpoint_dir)
            if code is None:
                return _failed_synthesis_node(checkpoint_dir)
        elif status is None:
            self._active_checkpoint = completed_steps
            try:
                context_started = time.monotonic()
                context = collect_synthesis_context(
                    cfg=self.cfg,
                    task_desc=self.task_desc,
                    journal=journal,
                    completed_steps=completed_steps,
                )
                context["timings_seconds"] = {
                    "collect_context": time.monotonic() - context_started
                }
                result = run_synthesis_checkpoint(
                    cfg=self.cfg,
                    context=context,
                    runner=self.runner,
                )
            finally:
                self._active_checkpoint = None
            if result["status"] != "ready":
                return _failed_synthesis_node(Path(result["checkpoint_dir"]))
            code = result["response"]["code"]
            plan = result["response"].get("plan")
        else:
            return None

        return SynthesisNode(
            node=Node(
                plan=(
                    plan.strip()
                    if plan
                    else _checkpoint_plan(checkpoint_dir, fallback_step=completed_steps)
                ),
                code=code,
            ),
            completed_steps=completed_steps,
            checkpoint_dir=checkpoint_dir,
        )

    def mark_recorded(self, synthesis_node: SynthesisNode, *, node: Node) -> None:
        status = _read_json(synthesis_node.checkpoint_dir / "status.json")
        if not isinstance(status, dict):
            status = {}
        status.update(
            {
                "recorded_at": dt.datetime.now().isoformat(timespec="seconds"),
                "recorded_node_id": node.id,
                "recorded_node_step": node.step,
            }
        )
        _write_json(synthesis_node.checkpoint_dir / "status.json", status)

    def mark_injected(self, synthesis_node: SynthesisNode, *, node: Node) -> None:
        status = _read_json(synthesis_node.checkpoint_dir / "status.json")
        if not isinstance(status, dict):
            status = {}
        injected_failed = node.is_buggy is True or node.status == "failed"
        status.update(
            {
                "status": "failed" if injected_failed else "injected",
                "injected_at": dt.datetime.now().isoformat(timespec="seconds"),
                "injected_node_id": node.id,
                "injected_node_step": node.step,
                "recorded_node_id": node.id,
                "recorded_node_step": node.step,
            }
        )
        if injected_failed:
            status["failed_at"] = dt.datetime.now().isoformat(timespec="seconds")
            status["error"] = (
                node.analysis
                or "".join(node.term_out).strip()
                or "Injected synthesis node failed during execution or review."
            )
        _write_json(synthesis_node.checkpoint_dir / "status.json", status)

    def status_text(self) -> str:
        if self._active_checkpoint is not None:
            return f"[cyan]Synthesis: ▶ {self._active_checkpoint:06d}"

        latest_checkpoint = _latest_checkpoint_with_status(self.cfg.log_dir)
        if latest_checkpoint is None:
            return "[dim]Synthesis: ○"

        checkpoint, status = latest_checkpoint
        label = _checkpoint_label(checkpoint)
        if status == "running":
            return f"[cyan]Synthesis: ▶ {label}"
        if status == "ready":
            return f"[yellow]Synthesis: … {label}"
        if status == "injected":
            return f"[green]Synthesis: ✓ {label}"
        if status == "failed":
            return f"[red]Synthesis: ✗ {label}"
        return f"[yellow]Synthesis: ? {label}"
