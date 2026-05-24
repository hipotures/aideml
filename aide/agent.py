import datetime as dt
import logging
import json
import math
import random
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

import humanize
from .autogluon_preprocess import (
    BASELINE_PLAN_PREFIX,
    baseline_preprocess_source,
    build_autogluon_wrapper,
    extract_preprocess_source,
    infer_sample_submission_columns,
    is_autogluon_preprocess_mode,
    parse_result_marker,
    sanitize_preprocess_prompt_text,
    validate_preprocess_source,
)
from .backend import FunctionSpec, query
from .backend.utils import write_llm_response_code
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .research import (
    build_data_overview,
    filter_hypothesis_candidate_parents,
    format_hypothesis_for_log_panel,
    format_hypothesis_for_prompt,
    format_manual_research_hints_for_prompt,
    format_research_hints_for_prompt,
    hypothesis_root_pool_exhausted,
    hypothesis_id_for_node,
    ManualHypothesisSelection,
    load_latest_manual_research_hints,
    load_latest_research_hints,
    load_hypothesis_root_code,
    record_manual_claimed_usage,
    record_manual_prompt_node,
    save_hypothesis_root_code,
    select_hypothesis_for_node,
)
from .telegram_notifications import append_node_with_best_score_notification
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.node_artifacts import node_artifact_dir as artifact_dir_for_node
from .utils.response import (
    extract_code,
    extract_jsons,
    extract_text_up_to_code,
    wrap_code,
)

logger = logging.getLogger("aide")


ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true only if the output log shows a technical execution failure, missing/invalid result, unusable metric, or another issue that makes the run result invalid. Do not set this for methodological concerns such as possible leakage, overfitting, weak validation, or risky feature engineering when the code ran and reported a valid metric.",
            },
            "summary": {
                "type": "string",
                "description": "If there is a technical bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.",
            },
            "metric": {
                "type": ["number", "null"],
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
            "validity_warning": {
                "type": ["string", "null"],
                "description": "Use this for non-fatal methodological concerns such as possible leakage, overfitting, non-grouped validation, questionable feature availability, or other reasons the reported metric may not generalize. Leave null when there is no such concern. A validity warning is not a technical bug by itself.",
            },
            "research_hypotheses_llm_claimed_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Research hypothesis ids that the implementation intentionally used. Include only ids that were explicitly offered in the prompt. In hypothesis mode, return exactly the single assigned id if and only if the implementation verifies that hypothesis.",
            },
            "research_usage_note": {
                "type": ["string", "null"],
                "description": "Short explanation of how the offered manual research hypotheses were used, or null when none were used.",
            },
        },
        "required": [
            "is_bug",
            "summary",
            "metric",
            "lower_is_better",
            "validity_warning",
            "research_hypotheses_llm_claimed_used",
            "research_usage_note",
        ],
        "additionalProperties": False,
    },
    description="Submit a review evaluating the output of the training script.",
)


def _parse_review_response(response: Any) -> dict[str, Any] | None:
    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        for parsed in extract_jsons(response):
            if isinstance(parsed, dict):
                return parsed

    return None


def _mark_invalid_review_response(node: Node, response: Any) -> None:
    node.analysis = (
        "Invalid review response from feedback model; marking this node as buggy "
        f"so the run can continue. Response type: {type(response).__name__}."
    )
    node.is_buggy = True
    node.metric = WorstMetricValue()


def _review_validity_warning(response: dict[str, Any], *, summary: str) -> str | None:
    warning = response.get("validity_warning")
    if isinstance(warning, str) and warning.strip():
        return warning.strip()
    if bool(response.get("is_bug")):
        return summary.strip() or "Feedback model flagged a non-fatal validity concern."
    return None


def _manual_research_ids_claimed_in_plan(node: Node) -> list[str]:
    if node.research_mode != "manual":
        return []
    plan = str(node.plan or "")
    return [
        hypothesis_id
        for hypothesis_id in node.research_hypotheses_offered
        if hypothesis_id in plan
    ]


def _record_plan_claimed_manual_research_usage(cfg: Config, node: Node) -> None:
    claimed = _manual_research_ids_claimed_in_plan(node)
    if not claimed:
        return
    node.research_hypotheses_llm_claimed_used = claimed
    node.research_usage_note = (
        "Plan text mentioned offered manual research hypothesis id(s): "
        + ", ".join(claimed)
        + "."
    )
    record_manual_claimed_usage(cfg, node)


def _raw_claimed_research_ids(response: dict[str, Any]) -> list[str]:
    claimed = response.get("research_hypotheses_llm_claimed_used", [])
    if not isinstance(claimed, list):
        return []
    return [item for item in claimed if isinstance(item, str)]


def _offered_research_claims(node: Node, raw_claimed: list[str]) -> list[str]:
    offered = set(node.research_hypotheses_offered)
    return [item for item in raw_claimed if item in offered]


def _mark_hypothesis_protocol_failure(
    node: Node,
    *,
    expected_id: str,
    raw_claimed: list[str],
) -> None:
    claimed_text = ", ".join(raw_claimed) if raw_claimed else "none"
    previous = str(node.analysis or "").strip()
    failure = (
        "Hypothesis verification failed: "
        f"expected hypothesis id {expected_id}, got {claimed_text}."
    )
    node.analysis = f"{previous}\n\n{failure}".strip()
    node.status = "failed"
    node.is_buggy = True
    node.metric = WorstMetricValue()
    node.research_hypotheses_llm_claimed_used = []
    node.research_usage_note = failure


def _record_hypothesis_claim_missing_after_bug(
    node: Node,
    *,
    expected_id: str,
    raw_claimed: list[str],
) -> None:
    claimed_text = ", ".join(raw_claimed) if raw_claimed else "none"
    note = (
        "Hypothesis claim missing after technical execution failure: "
        f"expected hypothesis id {expected_id}, got {claimed_text}. "
        "Keeping this node as a debuggable bug instead of terminal failed."
    )
    previous = str(node.analysis or "").strip()
    node.analysis = f"{previous}\n\n{note}".strip()
    node.status = None
    node.research_usage_note = note


def _apply_research_claims_from_response(
    cfg: Config,
    node: Node,
    response: dict[str, Any],
) -> bool:
    raw_claimed = _raw_claimed_research_ids(response)
    node.research_hypotheses_llm_claimed_used = _offered_research_claims(
        node,
        raw_claimed,
    )
    usage_note = response.get("research_usage_note")
    node.research_usage_note = (
        usage_note.strip()
        if isinstance(usage_note, str) and usage_note.strip()
        else None
    )

    if node.research_mode == "hypothesis":
        expected = (
            node.research_hypotheses_offered[0]
            if len(node.research_hypotheses_offered) == 1
            else None
        )
        if expected is None or raw_claimed != [expected]:
            if node.is_buggy:
                _record_hypothesis_claim_missing_after_bug(
                    node,
                    expected_id=expected or "<missing>",
                    raw_claimed=raw_claimed,
                )
                return True
            _mark_hypothesis_protocol_failure(
                node,
                expected_id=expected or "<missing>",
                raw_claimed=raw_claimed,
            )
            return False

    record_manual_claimed_usage(cfg, node)
    return True


def _metadata_hypothesis_id(metadata: dict[str, Any] | None) -> str | None:
    if not metadata or metadata.get("research_mode") != "hypothesis":
        return None
    offered = metadata.get("research_hypotheses_offered", [])
    if len(offered) == 1 and isinstance(offered[0], str):
        return offered[0]
    return None


def _configured_forced_hypothesis_root(search_cfg: Any) -> str | None:
    value = getattr(search_cfg, "forced_root", None)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit() and len(text) < 6:
        return text.zfill(6)
    return text


def _hypothesis_root_node(node: Node) -> Node:
    while node.parent is not None:
        node = node.parent
    return node


def _is_in_forced_hypothesis_root(node: Node, forced_root: str | None) -> bool:
    if forced_root is None:
        return True
    root = _hypothesis_root_node(node)
    return hypothesis_id_for_node(root) == forced_root


def _find_forced_hypothesis_root(journal: Journal, forced_root: str) -> Node | None:
    for node in journal.nodes:
        if node.parent is not None:
            continue
        if hypothesis_id_for_node(node) == forced_root:
            return node
    return None


def _metric_for_search(node: Node) -> float:
    assert node.metric is not None and node.metric.value is not None
    value = float(node.metric.value)
    return -value if node.metric.maximize is False else value


def _node_improves_parent(
    child: Node,
    parent: Node,
    *,
    epsilon: float = 0.0,
) -> bool:
    if child.metric is None or child.metric.value is None:
        return False
    if parent.metric is None or parent.metric.value is None:
        return False
    child_value = float(child.metric.value)
    parent_value = float(parent.metric.value)
    epsilon = max(0.0, float(epsilon))
    if child.metric.maximize is False:
        return child_value < parent_value - epsilon
    return child_value > parent_value + epsilon


def _nearest_scored_ancestor(node: Node) -> Node | None:
    parent = node.parent
    while parent is not None:
        if parent.metric is not None and parent.metric.value is not None:
            return parent
        parent = parent.parent
    return None


def _node_improves_nearest_scored_ancestor(
    node: Node,
    *,
    epsilon: float = 0.0,
) -> bool:
    ancestor = _nearest_scored_ancestor(node)
    if ancestor is None:
        return False
    return _node_improves_parent(node, ancestor, epsilon=epsilon)


def _is_scored_non_improving_child(
    child: Node,
    parent: Node,
    *,
    epsilon: float = 0.0,
) -> bool:
    if child.is_buggy or child.is_terminal_failure:
        return False
    if child.metric is None or child.metric.value is None:
        return False
    return not _node_improves_parent(child, parent, epsilon=epsilon)


def _has_improving_child(node: Node, *, epsilon: float = 0.0) -> bool:
    return any(
        _node_improves_parent(child, node, epsilon=epsilon)
        for child in node.children
    )


def _non_improving_child_count(node: Node, *, epsilon: float = 0.0) -> int:
    return sum(
        1
        for child in node.children
        if _is_scored_non_improving_child(child, node, epsilon=epsilon)
    )


def _is_hypothesis_parent_saturated(
    node: Node,
    *,
    limit: int,
    epsilon: float = 0.0,
) -> bool:
    if limit <= 0:
        return False
    if _has_improving_child(node, epsilon=epsilon):
        return False
    return _non_improving_child_count(node, epsilon=epsilon) >= limit


def _is_hypothesis_branch_candidate(node: Node, *, epsilon: float = 0.0) -> bool:
    if node.parent is None:
        return True
    if _node_improves_parent(node, node.parent, epsilon=epsilon):
        return True
    if node.parent.metric is None or node.parent.metric.value is None:
        return _node_improves_nearest_scored_ancestor(node, epsilon=epsilon)
    return False


def _metric_score_range(nodes: list[Node]) -> tuple[dict[Node, float], float, float, float]:
    metric_values = {node: _metric_for_search(node) for node in nodes}
    low = min(metric_values.values())
    high = max(metric_values.values())
    span = high - low
    return metric_values, low, high, span


def _search_exploration_bonus(
    *,
    child_count: int,
    total_good_nodes: int,
    exploration_weight: float,
) -> float:
    return exploration_weight * math.sqrt(
        math.log(total_good_nodes + 1) / (child_count + 1)
    )


def _search_exploration_score(
    node: Node,
    *,
    normalized_metric: float,
    total_good_nodes: int,
    exploration_weight: float,
) -> float:
    child_count = sum(1 for child in node.children if not child.is_terminal_failure)
    return normalized_metric + _search_exploration_bonus(
        child_count=child_count,
        total_good_nodes=total_good_nodes,
        exploration_weight=exploration_weight,
    )


def _metric_value(node: Node | None) -> float | None:
    if node is None or node.metric is None:
        return None
    value = node.metric.value
    return float(value) if value is not None else None


def _is_debuggable_failed_hypothesis_root(node: Node) -> bool:
    return (
        node.status == "failed"
        and node.parent is None
        and node.code
        and hypothesis_id_for_node(node) is not None
        and not node.is_submission_contract_error
    )


def _search_node_payload(node: Node | None) -> dict[str, Any] | None:
    if node is None:
        return None
    parent = node.parent
    return {
        "node_id": node.id,
        "step": node.step,
        "hypothesis_id": hypothesis_id_for_node(node),
        "metric": _metric_value(node),
        "is_buggy": node.is_buggy,
        "status": node.status,
        "parent_id": parent.id if parent is not None else None,
        "parent_step": parent.step if parent is not None else None,
        "parent_metric": _metric_value(parent),
        "parent_is_buggy": parent.is_buggy if parent is not None else None,
        "child_count": _search_child_count(node),
    }


def _search_policy_payload(
    node: Node,
    *,
    normalized_metric: float,
    policy_score: float,
    total_good_nodes: int,
    exploration_weight: float,
) -> dict[str, Any]:
    payload = _search_node_payload(node) or {}
    child_count = int(payload.get("child_count") or 0)
    payload.update(
        {
            "normalized_metric": normalized_metric,
            "exploration_bonus": _search_exploration_bonus(
                child_count=child_count,
                total_good_nodes=total_good_nodes,
                exploration_weight=exploration_weight,
            ),
            "policy_score": policy_score,
        }
    )
    return payload


def _fresh_child_metric_threshold(
    *,
    best_node: Node,
    best_policy_score: float,
    metric_low: float,
    metric_span: float,
    total_good_nodes: int,
    exploration_weight: float,
) -> dict[str, Any] | None:
    if metric_span <= 0:
        return None
    fresh_child_count = 0
    fresh_bonus = _search_exploration_bonus(
        child_count=fresh_child_count,
        total_good_nodes=total_good_nodes,
        exploration_weight=exploration_weight,
    )
    required_normalized_metric = best_policy_score - fresh_bonus
    required_search_metric = metric_low + required_normalized_metric * metric_span
    maximize = best_node.metric is None or best_node.metric.maximize is not False
    required_metric = required_search_metric if maximize else -required_search_metric
    return {
        "child_count": fresh_child_count,
        "direction": ">=" if maximize else "<=",
        "metric": required_metric,
        "normalized_metric": required_normalized_metric,
    }


def _search_child_count(node: Node) -> int:
    return sum(1 for child in node.children if not child.is_terminal_failure)


def _best_scored_search_node(journal: Journal) -> Node | None:
    candidates = [
        node
        for node in journal.good_nodes
        if node.metric is not None and node.metric.value is not None
    ]
    return max(candidates, key=lambda node: node.metric, default=None)


def _branch_candidate_rejection_reason(
    node: Node,
    *,
    epsilon: float = 0.0,
) -> str:
    if node.parent is None:
        return "accepted_root"
    if node.metric is None or node.metric.value is None:
        return "metric_missing"
    if node.parent.metric is None or node.parent.metric.value is None:
        if _nearest_scored_ancestor(node) is None:
            return "parent_metric_missing"
        return "does_not_improve_nearest_scored_ancestor"
    if not _node_improves_parent(node, node.parent, epsilon=epsilon):
        if _metric_for_search(node) > _metric_for_search(node.parent):
            return "does_not_clear_min_improvement_epsilon"
        return "does_not_improve_parent"
    return "accepted"


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.active_parent_node: Node | None = None
        self.active_node: Node | None = None
        self.active_stage: str | None = None
        self.active_research_hypothesis_id: str | None = None
        self.active_research_hypothesis_log_hint: str | None = None
        self.active_stage_started_at: float | None = None
        self._pending_node_ctime: float | None = None
        self._pending_llm_log_dir: Path | None = None
        self.last_search_decision: dict[str, Any] | None = None

    def _record_search_decision(self, trace: dict[str, Any]) -> None:
        self.last_search_decision = trace
        try:
            path = Path(self.cfg.log_dir) / "search_decisions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, sort_keys=True, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001 - diagnostics must not stop the run
            logger.debug("Failed to write search decision trace: %s", exc)

    def set_active_stage(self, stage: str | None) -> None:
        self.active_stage = stage
        self.active_stage_started_at = time.monotonic() if stage is not None else None

    def _node_artifact_dir(self, node: Node) -> Path:
        return artifact_dir_for_node(self.cfg.log_dir, node)

    def _new_node(
        self,
        *,
        plan: str,
        code: str,
        parent: Node | None = None,
    ) -> Node:
        kwargs: dict[str, Any] = {"plan": plan, "code": code, "parent": parent}
        if self._pending_node_ctime is not None:
            kwargs["ctime"] = self._pending_node_ctime
        return Node(**kwargs)

    def _apply_research_metadata(
        self,
        node: Node,
        metadata: dict[str, Any] | None,
    ) -> Node:
        if not metadata:
            return node
        node.research_mode = metadata.get("research_mode")
        node.research_hypotheses_offered = list(
            metadata.get("research_hypotheses_offered", [])
        )
        node.research_source_hash = metadata.get("research_source_hash")
        record_manual_prompt_node(self.cfg, node)
        return node

    def _generation_log_context(self) -> dict[str, Any]:
        parent = self.active_parent_node
        return {
            "phase": "generate",
            "run_id": self.cfg.exp_name,
            "parent_node_id": parent.id if parent is not None else None,
            "parent_stage": parent.stage_name if parent is not None else None,
            "agent_mode": self.acfg.mode,
            "node_ctime": self._pending_node_ctime,
        }

    def _review_log_context(self, node: Node) -> dict[str, Any]:
        return {
            "phase": "review",
            "run_id": self.cfg.exp_name,
            "node_id": node.id,
            "node_step": node.step,
            "node_stage": node.stage_name,
            "node_ctime": node.ctime,
            "agent_mode": self.acfg.mode,
        }

    def _is_hypothesis_mode(self) -> bool:
        return (
            self.cfg.research.enabled
            and getattr(self.cfg.research, "mode", "llm") == "hypothesis"
        )

    def _should_open_hypothesis_root(self) -> bool:
        if not self._is_hypothesis_mode():
            return False
        if _configured_forced_hypothesis_root(self.acfg.search) is not None:
            return False
        return not hypothesis_root_pool_exhausted(self.cfg, journal=self.journal)

    def _select_debuggable_node(
        self,
        *,
        forced_hypothesis_root: str | None = None,
    ) -> Node | None:
        search_cfg = self.acfg.search
        if search_cfg.debug_prob <= 0:
            return None
        if random.random() >= search_cfg.debug_prob:
            return None

        debuggable_nodes = [
            n
            for n in self.journal.buggy_nodes
            if (
                n.is_leaf
                and n.debug_depth < search_cfg.max_debug_depth
                and not n.is_submission_contract_error
                and not n.is_terminal_failure
                and _is_in_forced_hypothesis_root(n, forced_hypothesis_root)
            )
        ]
        if debuggable_nodes:
            logger.debug("[search policy] debugging")
            return random.choice(debuggable_nodes)
        logger.debug("[search policy] not debugging by chance")
        return None

    def _select_buggy_hypothesis_root(
        self,
        *,
        forced_hypothesis_root: str | None = None,
    ) -> Node | None:
        search_cfg = self.acfg.search
        candidates = [
            n
            for n in self.journal.buggy_nodes
            if (
                n.parent is None
                and hypothesis_id_for_node(n) is not None
                and n.is_leaf
                and n.debug_depth < search_cfg.max_debug_depth
                and not n.is_submission_contract_error
                and (
                    not n.is_terminal_failure
                    or _is_debuggable_failed_hypothesis_root(n)
                )
                and _is_in_forced_hypothesis_root(n, forced_hypothesis_root)
            )
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda n: n.step)

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search
        trace: dict[str, Any] = {
            "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "step": len(self.journal.nodes),
            "mode": "hypothesis" if self._is_hypothesis_mode() else "standard",
            "agent_mode": self.acfg.mode,
            "forced_hypothesis_root": None,
            "counts": {},
            "rejections": {},
            "top_candidates": [],
            "best_node": None,
            "selected": None,
            "reason": None,
        }
        forced_hypothesis_root = (
            _configured_forced_hypothesis_root(search_cfg)
            if self._is_hypothesis_mode()
            else None
        )
        trace["forced_hypothesis_root"] = forced_hypothesis_root

        def finish(selected: Node | None, reason: str) -> Node | None:
            trace["selected"] = _search_node_payload(selected)
            trace["reason"] = reason
            best_node = _best_scored_search_node(self.journal)
            best_payload = _search_node_payload(best_node)
            if best_payload is not None:
                best_payload["selected"] = (
                    selected is not None
                    and best_node is not None
                    and selected.id == best_node.id
                )
                if not best_payload["selected"]:
                    rejection = trace["rejections"].get(best_node.id) if best_node else None
                    if rejection is not None:
                        best_payload["rejected_at"] = rejection["stage"]
                        best_payload["reason"] = rejection["reason"]
                    elif selected is not None:
                        best_payload["rejected_at"] = "policy_score"
                        best_payload["reason"] = reason
            trace["best_node"] = best_payload
            self._record_search_decision(trace)
            return selected

        if self._is_hypothesis_mode() and self._should_open_hypothesis_root():
            if len(self.journal.draft_nodes) < search_cfg.num_drafts:
                logger.debug("[search policy] drafting new hypothesis root")
                return finish(None, "open_hypothesis_root_before_num_drafts")
            logger.debug("[search policy] drafting new hypothesis root")
            return finish(None, "open_hypothesis_root")

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return finish(None, "not_enough_drafts")

        if self._is_hypothesis_mode():
            root_debug_node = self._select_buggy_hypothesis_root(
                forced_hypothesis_root=forced_hypothesis_root,
            )
            if root_debug_node is not None:
                logger.debug("[search policy] debugging buggy hypothesis root")
                return finish(root_debug_node, "debugging_buggy_hypothesis_root")

        # debugging
        debug_node = self._select_debuggable_node(
            forced_hypothesis_root=forced_hypothesis_root,
        )
        if debug_node is not None:
            return finish(debug_node, "debugging")

        # back to drafting if no nodes to improve
        all_good_nodes = list(self.journal.good_nodes)
        good_nodes = [
            n
            for n in all_good_nodes
            if not n.is_in_submission_contract_error_branch
            and (
                not search_cfg.disable_oom_saturated_parents
                or not n.is_oom_blocked_parent
            )
        ]
        trace["counts"]["good_nodes"] = len(all_good_nodes)
        trace["counts"]["after_base_filters"] = len(good_nodes)
        for node in all_good_nodes:
            if node not in good_nodes:
                trace["rejections"][node.id] = {
                    "stage": "base_filters",
                    "reason": "submission_contract_or_oom_branch",
                }
        if self._is_hypothesis_mode():
            if forced_hypothesis_root is not None:
                before_forced_scope = list(good_nodes)
                good_nodes = [
                    node
                    for node in good_nodes
                    if _is_in_forced_hypothesis_root(node, forced_hypothesis_root)
                ]
                trace["counts"]["after_forced_root_scope"] = len(good_nodes)
                for node in before_forced_scope:
                    if node not in good_nodes:
                        trace["rejections"][node.id] = {
                            "stage": "forced_root_scope",
                            "reason": f"outside_forced_root_{forced_hypothesis_root}",
                        }
            improvement_epsilon = float(
                getattr(search_cfg, "hypothesis_min_improvement_epsilon", 0.0)
            )
            before_child_candidates = list(good_nodes)
            good_nodes = filter_hypothesis_candidate_parents(
                self.cfg,
                journal=self.journal,
                parent_nodes=good_nodes,
            )
            trace["counts"]["after_hypothesis_child_candidates"] = len(good_nodes)
            for node in before_child_candidates:
                if node not in good_nodes:
                    trace["rejections"][node.id] = {
                        "stage": "hypothesis_child_candidates",
                        "reason": "no_unused_child_hypothesis_available",
                    }
            before_branch_candidates = list(good_nodes)
            good_nodes = [
                node
                for node in good_nodes
                if _is_hypothesis_branch_candidate(
                    node,
                    epsilon=improvement_epsilon,
                )
            ]
            trace["counts"]["after_branch_candidate"] = len(good_nodes)
            for node in before_branch_candidates:
                if node not in good_nodes:
                    trace["rejections"][node.id] = {
                        "stage": "branch_candidate",
                        "reason": _branch_candidate_rejection_reason(
                            node,
                            epsilon=improvement_epsilon,
                        ),
                    }
            limit = int(
                getattr(
                    search_cfg,
                    "hypothesis_max_non_improving_children_per_parent",
                    3,
                )
            )
            if limit > 0:
                before_saturation = list(good_nodes)
                good_nodes = [
                    node
                    for node in good_nodes
                    if not _is_hypothesis_parent_saturated(
                        node,
                        limit=limit,
                        epsilon=improvement_epsilon,
                    )
                ]
                trace["counts"]["after_saturation"] = len(good_nodes)
                for node in before_saturation:
                    if node not in good_nodes:
                        trace["rejections"][node.id] = {
                            "stage": "saturation",
                            "reason": (
                                f"no_improving_child_and_at_least_{limit}_"
                                "non_improving_children"
                            ),
                        }
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return finish(None, "no_good_nodes_after_filters")

        exploration_weight = float(getattr(search_cfg, "exploration_weight", 0.0))
        if exploration_weight <= 0:
            greedy_node = max(good_nodes, key=lambda n: n.metric)
            ranked = sorted(good_nodes, key=lambda n: n.metric, reverse=True)
            trace["top_candidates"] = [
                {
                    **(_search_node_payload(node) or {}),
                    "rank": index + 1,
                    "policy_score": _metric_value(node),
                }
                for index, node in enumerate(ranked[:8])
            ]
            logger.debug("[search policy] greedy node selected")
            return finish(greedy_node, "highest_metric_after_filters")

        metric_values, metric_low, metric_high, metric_span = _metric_score_range(good_nodes)
        if metric_span <= 0:
            normalized_scores = {node: 1.0 for node in good_nodes}
        else:
            normalized_scores = {
                node: (value - metric_low) / metric_span
                for node, value in metric_values.items()
            }
        policy_scores = {
            node: _search_exploration_score(
                node,
                normalized_metric=normalized_scores[node],
                total_good_nodes=len(good_nodes),
                exploration_weight=exploration_weight,
            )
            for node in good_nodes
        }
        best_policy_node = max(good_nodes, key=lambda node: node.metric)
        min_best_children = int(
            getattr(search_cfg, "best_score_min_children_before_exploration", 0)
        )
        selection_override: dict[str, Any] | None = None
        if min_best_children > 0:
            best_child_count = _search_child_count(best_policy_node)
            if best_child_count < min_best_children:
                selection_override = {
                    "reason": "best_score_min_children_before_exploration",
                    "best_child_count": best_child_count,
                    "min_children": min_best_children,
                }
        selected_node = (
            best_policy_node
            if selection_override is not None
            else max(good_nodes, key=lambda node: policy_scores[node])
        )
        ranked = sorted(good_nodes, key=lambda node: policy_scores[node], reverse=True)
        selected_policy_payload = _search_policy_payload(
            selected_node,
            normalized_metric=normalized_scores[selected_node],
            policy_score=policy_scores[selected_node],
            total_good_nodes=len(good_nodes),
            exploration_weight=exploration_weight,
        )
        best_policy_payload = _search_policy_payload(
            best_policy_node,
            normalized_metric=normalized_scores[best_policy_node],
            policy_score=policy_scores[best_policy_node],
            total_good_nodes=len(good_nodes),
            exploration_weight=exploration_weight,
        )
        trace["policy_diagnostics"] = {
            "candidate_count": len(good_nodes),
            "exploration_weight": exploration_weight,
            "metric_min": metric_low,
            "metric_max": metric_high,
            "metric_span": metric_span,
            "selected": selected_policy_payload,
            "best": best_policy_payload,
            "selected_minus_best_policy_score": (
                policy_scores[selected_node] - policy_scores[best_policy_node]
            ),
            "selected_minus_best_metric": (
                _metric_value(selected_node) - _metric_value(best_policy_node)
                if (
                    _metric_value(selected_node) is not None
                    and _metric_value(best_policy_node) is not None
                )
                else None
            ),
            "fresh_child_metric_threshold": _fresh_child_metric_threshold(
                best_node=best_policy_node,
                best_policy_score=policy_scores[best_policy_node],
                metric_low=metric_low,
                metric_span=metric_span,
                total_good_nodes=len(good_nodes),
                exploration_weight=exploration_weight,
            ),
        }
        if selection_override is not None:
            trace["policy_diagnostics"]["selection_override"] = selection_override
        trace["top_candidates"] = [
            {
                **(_search_node_payload(node) or {}),
                "rank": index + 1,
                "normalized_metric": normalized_scores[node],
                "policy_score": policy_scores[node],
            }
            for index, node in enumerate(ranked[:8])
        ]
        logger.debug("[search policy] exploration node selected")
        if selection_override is not None:
            return finish(selected_node, selection_override["reason"])
        return finish(selected_node, "highest_policy_score_after_filters")

    @property
    def _prompt_environment(self):
        candidate_pkgs = [
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("scikit-learn", "sklearn"),
            ("statsmodels", "statsmodels"),
            ("xgboost", "xgboost"),
            ("catboost", "catboost"),
            ("autogluon", "autogluon"),
            ("lightgbm", "lightgbm"),
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("torch-geometric", "torch_geometric"),
            ("bayesian-optimization", "bayes_opt"),
            ("timm", "timm"),
        ]
        pkgs = [
            display_name
            for display_name, import_name in candidate_pkgs
            if find_spec(import_name) is not None
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])
        neural_hint = (
            " For neural networks, PyTorch is importable and can be used."
            if "torch" in pkgs
            else ""
        )

        env_prompt = {
            "Installed Packages": (
                "Detected importable machine learning packages include: "
                f"{pkg_str}. Other packages may be available, but verify imports "
                f"before relying on them.{neural_hint}"
            )
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            'If you run a post-CV blend-weight search with many independent AUC/ROC-AUC evaluations, do not evaluate thousands of `roc_auc_score` calls serially. Use `joblib.Parallel` with `n_jobs=min(16, os.cpu_count() or 1)` and `prefer="threads"` to avoid copying large OOF arrays; print "Evaluating N blend candidates with M workers" before starting. Keep candidate grids bounded, and fall back to a simple 1D blend if joblib is unavailable.',
            "If you can preserve the intended behavior while reducing code size, memory use, or runtime, make that optimization instead of emitting verbose or redundant code.",
            'All the provided input data is stored in "./input" directory.',
            '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'When you train with cross-validation, also save leak-free out-of-fold predictions to gzip-compressed `./working/oof_predictions.csv.gz` with columns `row`, `target`, and `prediction`; save full test probabilities/predictions to `./working/test_predictions.csv.gz` using the sample-submission id and target columns. If you only use a holdout split, save holdout predictions to `./working/validation_predictions.csv.gz`.',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        if self.acfg.gpu:
            impl_guideline.extend(
                [
                    "A CUDA-capable NVIDIA GPU is available. Prefer GPU-enabled training for tabular tree models when the chosen library supports it.",
                    'For CatBoost, use `task_type="GPU"`, `devices="0"`, and `gpu_ram_part=0.8` when training on GPU.',
                    'For XGBoost, use `tree_method="hist"` with `device="cuda"` when training on GPU.',
                    'For LightGBM, do not use `device_type="gpu"` or `device="gpu"`; that selects the OpenCL backend. If you explicitly use LightGBM on GPU, use the CUDA backend with `device_type="cuda"` or `device="cuda"`.',
                    "If a GPU-specific implementation fails, the next debug attempt may fall back to CPU.",
                ]
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    @property
    def _prompt_autogluon_preprocess_guideline(self):
        return {
            "AutoGluon preprocess mode contract": [
                "You are writing only the feature preprocessing function for a fixed AutoGluon training wrapper.",
                "Return a single markdown code block containing exactly one top-level function: def preprocess(df: pd.DataFrame) -> pd.DataFrame.",
                "The df argument contains concatenated train features followed by Kaggle prediction/test features, with only model feature columns present.",
                "Your returned preprocess function must replace the previous preprocess function. It is not composed with the previous function automatically.",
                "Do not call `globals().get(\"preprocess\")`, do not add extra preprocess arguments, and do not assume any previous preprocess function is callable at runtime.",
                "Columns created by a previous preprocess function are not present in df at entry. If your new feature needs previous derived columns, preserve or recompute the code that creates them inside the returned function.",
                "Use only columns visible in the sanitized feature overview or in previous preprocess functions.",
                "Do not add defensive cleanup for hidden wrapper columns or columns that are not present in preprocess(df).",
                "Do not read files, write files, train models, create validation splits, save submissions, or call AutoGluon. The fixed wrapper does all of that.",
                "Do not change row count or reorder rows.",
                "If your intended algorithm would remove rows, such as outlier filtering, do not drop them. Preserve all rows and instead add features such as an outlier flag, clipped/winsorized value, imputed clean value, anomaly score, or distance-from-normal feature.",
                "Create deterministic, leakage-safe feature engineering only. Shared train+test operations like dtype cleanup, frequency encoding, and category normalization are allowed if they use only model feature columns.",
                "If you can preserve the intended behavior while reducing code size, memory use, or runtime, make that optimization instead of emitting verbose or redundant code.",
                f"preprocess(df) has a dedicated timeout of {int(getattr(self.cfg.agent.autogluon, 'preprocess_timeout', 180))} seconds before AutoGluon training starts.",
                "Avoid expensive Python callbacks over rows, groups, or rolling windows, especially `groupby.apply`, `rolling.apply`, and `np.polyfit` on full train+test data. Prefer bounded vectorized `groupby().transform`, `shift`, `rolling().mean/std/min/max`, and simple arithmetic features.",
            ]
        }

    def _add_research_hints(
        self,
        prompt: dict[str, Any],
        *,
        parent_node: Node | None = None,
    ) -> dict[str, Any] | None:
        if not self.cfg.research.enabled:
            return None
        research_mode = getattr(self.cfg.research, "mode", "llm")
        if research_mode == "manual":
            selection = load_latest_manual_research_hints(self.cfg)
            if selection is None:
                return None
            prompt["External research hints"] = format_manual_research_hints_for_prompt(
                selection
            )
            return {
                "research_mode": "manual",
                "research_hypotheses_offered": [
                    hypothesis.id for hypothesis in selection.hypotheses
                ],
                "research_source_hash": selection.source_hash,
            }
        if research_mode == "hypothesis":
            selection = select_hypothesis_for_node(
                self.cfg,
                journal=self.journal,
                parent_node=parent_node,
                completed_steps=len(self.journal.nodes),
            )
            return self._add_hypothesis_selection(prompt, selection)
        hints = load_latest_research_hints(self.cfg.log_dir)
        if hints is not None:
            prompt["External research hints"] = format_research_hints_for_prompt(hints)
        return None

    def _add_hypothesis_selection(
        self,
        prompt: dict[str, Any],
        selection: Any,
    ) -> dict[str, Any]:
        self.active_research_hypothesis_id = (
            selection.hypotheses[0].id if selection.hypotheses else None
        )
        self.active_research_hypothesis_log_hint = (
            format_hypothesis_for_log_panel(selection)
        )
        prompt["Hypothesis under verification"] = format_hypothesis_for_prompt(
            selection
        )
        return {
            "research_mode": "hypothesis",
            "research_hypotheses_offered": [
                hypothesis.id for hypothesis in selection.hypotheses
            ],
            "research_source_hash": selection.source_hash,
        }

    def _add_memory_or_branch_context(
        self,
        prompt: dict[str, Any],
        *,
        parent_node: Node | None,
        include_global_memory: bool = True,
    ) -> None:
        if self._is_hypothesis_mode():
            if parent_node is not None:
                prompt["Branch context"] = self.journal.generate_branch_context(
                    parent_node
                )
            return
        if include_global_memory:
            prompt["Memory"] = self.journal.generate_summary()

    def _autogluon_unavailable_columns(self) -> list[str]:
        columns = infer_sample_submission_columns(self.cfg.workspace_dir / "input")
        if columns is None:
            return []
        return [column for column in columns if column]

    def _autogluon_target_column(self) -> str | None:
        columns = infer_sample_submission_columns(self.cfg.workspace_dir / "input")
        return columns[1] if columns is not None else None

    def _autogluon_prompt_text(self, text: Any) -> str:
        return sanitize_preprocess_prompt_text(
            text,
            unavailable_columns=self._autogluon_unavailable_columns(),
        )

    def _add_autogluon_context(self, prompt: dict[str, Any]) -> None:
        prompt["Fixed AutoGluon wrapper context"] = (
            "The fixed wrapper passes only model feature columns to preprocess(df). "
            "Use only columns visible in the sanitized feature overview or in "
            "previous preprocess functions. Do not add defensive cleanup for "
            "columns that are not present in preprocess(df)."
        )

    def _wrap_autogluon_preprocess_node(
        self,
        *,
        plan: str,
        code: str,
        parent: Node | None = None,
        research_metadata: dict[str, Any] | None = None,
    ) -> Node:
        try:
            preprocess_source = extract_preprocess_source(code)
            validate_preprocess_source(
                preprocess_source,
                target_col=self._autogluon_target_column(),
            )
            wrapped_code = build_autogluon_wrapper(
                preprocess_source,
                self.cfg,
                research_hypothesis_id=_metadata_hypothesis_id(research_metadata),
            )
        except ValueError as exc:
            wrapped_code = f"raise ValueError({str(exc)!r})\n"
        return self._new_node(plan=plan, code=wrapped_code, parent=parent)

    def _autogluon_raw_baseline(self) -> Node:
        code = build_autogluon_wrapper(baseline_preprocess_source(), self.cfg)
        return self._new_node(
            plan=(
                f"{BASELINE_PLAN_PREFIX}: raw features with the configured "
                "fixed AutoGluon runner."
            ),
            code=code,
        )

    def _previous_preprocess_source(self, parent_node: Node) -> str:
        try:
            return extract_preprocess_source(parent_node.code)
        except ValueError:
            return parent_node.code

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                reasoning_effort=self.acfg.code.reasoning_effort,
                temperature=self.acfg.code.temp,
                llm_log_dir=self._pending_llm_log_dir,
                llm_log_context=self._generation_log_context(),
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                write_llm_response_code(
                    log_dir=self._pending_llm_log_dir,
                    code=code,
                )
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self, *, hypothesis_selection: Any | None = None) -> Node:
        if is_autogluon_preprocess_mode(self.cfg):
            return self._draft_autogluon_preprocess(
                hypothesis_selection=hypothesis_selection,
            )

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Instructions": {},
        }
        self._add_memory_or_branch_context(prompt, parent_node=None)
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                "Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        if hypothesis_selection is not None:
            research_metadata = self._add_hypothesis_selection(
                prompt,
                hypothesis_selection,
            )
        else:
            research_metadata = self._add_research_hints(prompt, parent_node=None)
        plan, code = self.plan_and_code_query(prompt)
        return self._apply_research_metadata(
            self._new_node(plan=plan, code=code),
            research_metadata,
        )

    def _draft_autogluon_preprocess(
        self,
        *,
        hypothesis_selection: Any | None = None,
    ) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "A fixed AutoGluon runner will handle model training, validation, "
                "and submission generation. Your job is to design leakage-safe "
                "feature preprocessing for that runner."
            ),
            "Task description": self._autogluon_prompt_text(self.task_desc),
            "Instructions": {},
        }
        self._add_memory_or_branch_context(prompt, parent_node=None)
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Preprocessing sketch guideline": [
                "The solution sketch should be 3-5 sentences describing the feature engineering idea.",
                "Keep the first solution relatively simple and deterministic.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_autogluon_preprocess_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self._autogluon_prompt_text(self.data_preview)

        self._add_autogluon_context(prompt)
        if hypothesis_selection is not None:
            research_metadata = self._add_hypothesis_selection(
                prompt,
                hypothesis_selection,
            )
        else:
            research_metadata = self._add_research_hints(prompt, parent_node=None)
        plan, code = self.plan_and_code_query(prompt)
        return self._apply_research_metadata(
            self._wrap_autogluon_preprocess_node(
                plan=plan,
                code=code,
                research_metadata=research_metadata,
            ),
            research_metadata,
        )

    def _improve(self, parent_node: Node) -> Node:
        if is_autogluon_preprocess_mode(self.cfg):
            return self._improve_autogluon_preprocess(parent_node)

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Instructions": {},
        }
        self._add_memory_or_branch_context(prompt, parent_node=parent_node)
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        research_metadata = self._add_research_hints(prompt, parent_node=parent_node)
        plan, code = self.plan_and_code_query(prompt)
        return self._apply_research_metadata(
            self._new_node(
                plan=plan,
                code=code,
                parent=parent_node,
            ),
            research_metadata,
        )

    def _improve_autogluon_preprocess(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are improving only the feature preprocessing function for a "
                "fixed AutoGluon training wrapper. Keep the wrapper behavior "
                "unchanged and make one atomic, leakage-safe feature improvement."
            ),
            "Task description": self._autogluon_prompt_text(self.task_desc),
            "Previous preprocess function": wrap_code(
                self._previous_preprocess_source(parent_node)
            ),
            "Instructions": {},
        }
        self._add_memory_or_branch_context(prompt, parent_node=parent_node)
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Preprocessing improvement sketch guideline": [
                "The solution sketch should describe one specific feature engineering improvement.",
                "Make the change atomic so the AutoGluon wrapper can evaluate its effect.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_autogluon_preprocess_guideline
        self._add_autogluon_context(prompt)
        research_metadata = self._add_research_hints(prompt, parent_node=parent_node)
        plan, code = self.plan_and_code_query(prompt)
        return self._apply_research_metadata(
            self._wrap_autogluon_preprocess_node(
                plan=plan,
                code=code,
                parent=parent_node,
                research_metadata=research_metadata,
            ),
            research_metadata,
        )

    def _debug(self, parent_node: Node) -> Node:
        if is_autogluon_preprocess_mode(self.cfg):
            return self._debug_autogluon_preprocess(parent_node)

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        self._add_memory_or_branch_context(
            prompt,
            parent_node=parent_node,
            include_global_memory=False,
        )
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        if parent_node.exc_type == "TimeoutError":
            prompt["Instructions"]["Timeout fix guideline"] = [
                f"The previous implementation exceeded the execution timeout of {humanize.naturaldelta(self.cfg.exec.timeout)}.",
                "Treat this as a runtime efficiency failure. Preserve the intended approach where possible, but simplify or limit the expensive parts so the script completes within the timeout.",
                "Do not assume a specific failing operation unless it is visible in the execution output.",
            ]
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        research_metadata = self._add_research_hints(prompt, parent_node=parent_node)
        plan, code = self.plan_and_code_query(prompt)
        return self._apply_research_metadata(
            self._new_node(plan=plan, code=code, parent=parent_node),
            research_metadata,
        )

    def _debug_autogluon_preprocess(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are fixing a buggy feature preprocessing function used by a "
                "fixed AutoGluon training wrapper. Revise only preprocess(df); "
                "do not write a full model pipeline."
            ),
            "Task description": self._autogluon_prompt_text(self.task_desc),
            "Previous preprocess function": wrap_code(
                self._previous_preprocess_source(parent_node)
            ),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        self._add_memory_or_branch_context(
            prompt,
            parent_node=parent_node,
            include_global_memory=False,
        )
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix preprocessing sketch guideline": [
                "Describe the cause of the preprocessing failure and the narrow fix.",
                "Keep the function deterministic and leakage-safe.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_autogluon_preprocess_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self._autogluon_prompt_text(self.data_preview)

        self._add_autogluon_context(prompt)
        research_metadata = self._add_research_hints(prompt, parent_node=parent_node)
        plan, code = self.plan_and_code_query(prompt)
        return self._apply_research_metadata(
            self._wrap_autogluon_preprocess_node(
                plan=plan,
                code=code,
                parent=parent_node,
                research_metadata=research_metadata,
            ),
            research_metadata,
        )

    def update_data_preview(
        self,
    ):
        workspace_input_dir = Path(self.cfg.workspace_dir) / "input"
        data_dir = workspace_input_dir if workspace_input_dir.exists() else Path(self.cfg.data_dir)
        if data_dir.exists():
            self.data_preview = data_preview.generate(
                data_dir,
                include_file_tree=False,
            )
        else:
            self.data_preview = build_data_overview(self.cfg)

    def _draft_hypothesis_root(
        self,
        selection: ManualHypothesisSelection | None = None,
    ) -> Node:
        if selection is None:
            selection = select_hypothesis_for_node(
                self.cfg,
                journal=self.journal,
                parent_node=None,
                completed_steps=len(self.journal.nodes),
            )
        if len(selection.hypotheses) != 1:
            raise ValueError("Hypothesis mode requires exactly one selected root.")
        hypothesis_id = selection.hypotheses[0].id
        self.active_research_hypothesis_id = hypothesis_id
        self.active_research_hypothesis_log_hint = format_hypothesis_for_log_panel(
            selection
        )
        root_code = load_hypothesis_root_code(self.cfg, hypothesis_id)
        if root_code is not None:
            metadata = {
                "research_mode": "hypothesis",
                "research_hypotheses_offered": [hypothesis_id],
                "research_source_hash": selection.source_hash,
            }
            plan = (
                f"Loaded library {root_code.agent_mode} root code for "
                f"hypothesis {hypothesis_id} from {root_code.path.name}."
            )
            return self._apply_research_metadata(
                self._new_node(plan=plan, code=root_code.code),
                metadata,
            )
        return self._draft(hypothesis_selection=selection)

    def prepare_step(self) -> Node | None:
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        self.active_parent_node = parent_node
        return parent_node

    def generate_node(
        self,
        parent_node: Node | None,
        *,
        node_ctime: float | None = None,
        llm_log_dir: Path | None = None,
    ) -> Node:
        self.set_active_stage("generating")
        self.active_research_hypothesis_id = None
        self.active_research_hypothesis_log_hint = None
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")
        previous_ctime = self._pending_node_ctime
        previous_log_dir = self._pending_llm_log_dir
        self._pending_node_ctime = node_ctime
        self._pending_llm_log_dir = llm_log_dir

        try:
            if parent_node is None and self._is_hypothesis_mode():
                forced_root = _configured_forced_hypothesis_root(self.acfg.search)
                if forced_root is not None:
                    parent_node = _find_forced_hypothesis_root(
                        self.journal,
                        forced_root,
                    )
                    if parent_node is None:
                        raise ValueError(
                            "Configured forced hypothesis root "
                            f"{forced_root!r} was not found in the journal."
                        )
                    self.active_parent_node = parent_node
            if (
                parent_node is None
                and is_autogluon_preprocess_mode(self.cfg)
                and not self.journal.nodes
            ):
                return self._autogluon_raw_baseline()
            if parent_node is None and self._is_hypothesis_mode():
                return self._draft_hypothesis_root()
            if parent_node is None:
                return self._draft()
            if parent_node.is_buggy:
                return self._debug(parent_node)
            return self._improve(parent_node)
        finally:
            self._pending_node_ctime = previous_ctime
            self._pending_llm_log_dir = previous_log_dir

    def generate_preselected_hypothesis_root(
        self,
        selection: ManualHypothesisSelection,
        *,
        node_ctime: float,
        llm_log_dir: Path,
        artifact_dir_name: str,
    ) -> Node:
        self.set_active_stage("generating")
        self.active_parent_node = None
        self.active_research_hypothesis_id = selection.hypotheses[0].id
        self.active_research_hypothesis_log_hint = format_hypothesis_for_log_panel(
            selection
        )
        previous_ctime = self._pending_node_ctime
        previous_log_dir = self._pending_llm_log_dir
        self._pending_node_ctime = node_ctime
        self._pending_llm_log_dir = llm_log_dir
        try:
            node = self._draft_hypothesis_root(selection)
            node.artifact_dir_name = artifact_dir_name
            return node
        finally:
            self._pending_node_ctime = previous_ctime
            self._pending_llm_log_dir = previous_log_dir

    def execute_node(
        self, node: Node, exec_callback: ExecCallbackType
    ) -> ExecutionResult:
        self.active_node = node
        self.set_active_stage("executing")
        return exec_callback(node.code, True)

    def review_node(self, node: Node, exec_result: ExecutionResult) -> None:
        self.set_active_stage("reviewing")
        self.parse_exec_result(
            node=node,
            exec_result=exec_result,
        )
        if node.status == "generated":
            node.status = "bug" if node.is_buggy else "ok"
        self._save_reviewed_hypothesis_root_code(node)

    def _save_reviewed_hypothesis_root_code(self, node: Node) -> None:
        self.save_hypothesis_root_code_for_node(node)

    def save_hypothesis_root_code_for_node(
        self,
        node: Node,
        *,
        activate: bool = True,
    ) -> None:
        if node.research_mode != "hypothesis":
            return
        force_new_version = False
        if node.parent is not None:
            root = node.parent
            while root.parent is not None:
                root = root.parent
            root_hypothesis_id = hypothesis_id_for_node(root)
            node_hypothesis_id = hypothesis_id_for_node(node)
            root_is_buggy = (
                bool(root.is_buggy)
                or root.status == "bug"
                or root.is_submission_contract_error
            )
            if (
                not root_is_buggy
                or root_hypothesis_id is None
                or node_hypothesis_id != root_hypothesis_id
            ):
                return
            force_new_version = True
        hypothesis_id = hypothesis_id_for_node(node)
        if hypothesis_id is None:
            return
        score = None if node.metric is None else node.metric.value
        created_at = dt.datetime.fromtimestamp(node.ctime).isoformat(timespec="seconds")
        save_hypothesis_root_code(
            self.cfg,
            hypothesis_id=hypothesis_id,
            code=node.code,
            is_buggy=bool(node.is_buggy),
            node_id=node.id,
            score=score,
            created_at=created_at,
            force_new_version=force_new_version,
            activate=activate,
        )

    def clear_active_step(self) -> None:
        self.active_parent_node = None
        self.active_node = None
        self.active_research_hypothesis_id = None
        self.active_research_hypothesis_log_hint = None
        self.set_active_stage(None)

    def step(self, exec_callback: ExecCallbackType):
        parent_node = self.prepare_step()

        try:
            result_node = self.generate_node(parent_node)
            exec_result = self.execute_node(result_node, exec_callback)
            self.review_node(result_node, exec_result)
            append_node_with_best_score_notification(
                journal=self.journal,
                node=result_node,
                experiment_id=self.cfg.exp_name,
            )
        finally:
            self.clear_active_step()

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)
        marker_response = parse_result_marker(node.term_out)
        if marker_response is not None:
            metric = marker_response.get("metric")
            if not isinstance(metric, (float, int)) or isinstance(metric, bool):
                metric = None
            node.analysis = str(marker_response.get("summary", ""))
            node.validity_warning = _review_validity_warning(
                marker_response,
                summary=node.analysis,
            )
            run_stats = marker_response.get("run_stats")
            node.run_stats = run_stats if isinstance(run_stats, dict) else None
            node.is_buggy = (
                node.exc_type is not None
                or metric is None
                or (bool(marker_response.get("is_bug")) and metric is None)
            )
            if node.is_buggy:
                node.metric = WorstMetricValue()
            else:
                node.metric = MetricValue(
                    metric,
                    maximize=not bool(marker_response.get("lower_is_better")),
                )
            if node.research_mode == "manual":
                _record_plan_claimed_manual_research_usage(self.cfg, node)
            elif not _apply_research_claims_from_response(
                self.cfg,
                node,
                marker_response,
            ):
                return
            return

        prompt = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any technical bugs as well as report the empirical findings. "
                "Use is_bug only for technical failures that make the result invalid. "
                "For non-fatal methodological concerns, such as possible leakage or weak validation, keep is_bug false when a valid metric was reported and put the concern in validity_warning."
            ),
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }
        if node.research_mode in {"manual", "hypothesis"} and (
            node.research_hypotheses_offered
        ):
            prompt_title = (
                "Hypothesis verification required"
                if node.research_mode == "hypothesis"
                else "Manual research hypotheses offered"
            )
            instruction = (
                "Report exactly this id in "
                "research_hypotheses_llm_claimed_used. If the implementation "
                "did not implement this assigned hypothesis, leave the list "
                "empty so the node can be marked failed."
                if node.research_mode == "hypothesis"
                else (
                    "Report only offered ids in "
                    "research_hypotheses_llm_claimed_used. Use an empty list if "
                    "the implementation did not intentionally use any offered "
                    "manual research hypothesis."
                )
            )
            prompt[prompt_title] = {
                "ids": node.research_hypotheses_offered,
                "instruction": instruction,
            }

        response = query(
            system_message=prompt,
            user_message=None,
            func_spec=review_func_spec,
            model=self.acfg.feedback.model,
            reasoning_effort=self.acfg.feedback.reasoning_effort,
            temperature=self.acfg.feedback.temp,
            llm_log_dir=self._node_artifact_dir(node),
            llm_log_prefix="review",
            llm_log_context=self._review_log_context(node),
        )
        parsed_response = _parse_review_response(response)
        if parsed_response is None:
            _mark_invalid_review_response(node, response)
            return

        required_keys = {"is_bug", "summary", "metric", "lower_is_better"}
        if not required_keys.issubset(parsed_response):
            _mark_invalid_review_response(node, response)
            return

        # if the metric isn't a float then fill the metric with the worst metric
        metric = parsed_response["metric"]
        if not isinstance(metric, (float, int)) or isinstance(metric, bool):
            metric = None

        node.analysis = str(parsed_response["summary"])
        node.validity_warning = _review_validity_warning(
            parsed_response,
            summary=node.analysis,
        )
        node.is_buggy = (
            node.exc_type is not None
            or metric is None
            or (bool(parsed_response["is_bug"]) and metric is None)
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                metric, maximize=not bool(parsed_response["lower_is_better"])
            )
        if not _apply_research_claims_from_response(self.cfg, node, parsed_response):
            return
