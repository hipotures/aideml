from __future__ import annotations

from aide.journal import Journal, Node
from aide.utils.metric import MetricValue
from aide.utils.plateau import (
    DEFAULT_PLATEAU_BLOCK_EPSILON,
    is_plateau_blocked_descendant,
)
from aide.utils.public_scores import public_adjusted_oriented_score

from .state import WebTreeLine


def _node_order_key(journal: Journal, node: Node) -> tuple[bool, int, float, str]:
    return (
        node.step is None,
        node.step if node.step is not None else len(journal.nodes),
        node.ctime,
        node.id,
    )


def _hypothesis_or_step_suffix(node: Node) -> str:
    if len(node.research_hypotheses_offered) == 1:
        return f"·{node.research_hypotheses_offered[0]}"
    if node.step is not None:
        return f"·{node.step}"
    return ""


def _runtime_suffix(node: Node) -> str:
    try:
        seconds = float(node.exec_time)
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    minutes = int(round(seconds / 60.0))
    if minutes <= 0:
        return ""
    return f"·{minutes}m"


def _node_public_score_bonus_active(
    node: Node,
    *,
    public_scores_by_node_id: dict[str, float],
    weight: float,
    cap: float,
) -> bool:
    if node.metric is None or node.metric.value is None:
        return False
    local_score = float(node.metric.value)
    adjusted = public_adjusted_oriented_score(
        local_score=local_score,
        public_score=public_scores_by_node_id.get(node.id),
        maximize=node.metric.maximize is not False,
        weight=weight,
        cap=cap,
    )
    oriented_local = local_score if node.metric.maximize is not False else -local_score
    return adjusted > oriented_local


def _node_has_public_score(
    node: Node,
    *,
    public_scores_by_node_id: dict[str, float],
) -> bool:
    return node.id in public_scores_by_node_id


def _best_scored_node(
    journal: Journal,
    *,
    plateau_block_epsilon: float,
) -> Node | None:
    candidates = [
        node
        for node in journal.good_nodes
        if node.metric is not None and node.metric.value is not None
        and not is_plateau_blocked_descendant(
            node,
            epsilon=plateau_block_epsilon,
        )
    ]
    return max(candidates, key=lambda node: node.metric, default=None)


def _public_best_scored_node(
    journal: Journal,
    *,
    plateau_block_epsilon: float,
    public_scores_by_node_id: dict[str, float],
) -> Node | None:
    if not public_scores_by_node_id:
        return None
    candidates = [
        node
        for node in journal.good_nodes
        if node.metric is not None and node.metric.value is not None
        and node.id in public_scores_by_node_id
        and not is_plateau_blocked_descendant(
            node,
            epsilon=plateau_block_epsilon,
        )
    ]
    return max(
        candidates,
        key=lambda node: (
            public_scores_by_node_id[node.id]
            if node.metric.maximize is not False
            else -public_scores_by_node_id[node.id]
        ),
        default=None,
    )


def _metric_text(metric: MetricValue | None) -> str | None:
    if metric is None or metric.value is None:
        return None
    return f"{metric.value:.5f}"


def _line_for_node(
    node: Node,
    *,
    best_node: Node | None,
    public_best_node: Node | None,
    plateau_block_epsilon: float,
    public_node_ids: set[str],
    public_bonus_node_ids: set[str],
) -> tuple[str, str]:
    suffix = _hypothesis_or_step_suffix(node)
    runtime_suffix = _runtime_suffix(node)
    if node.status == "generated":
        return f"generated{suffix}", "generated"
    if node.is_terminal_failure or node.status == "failed":
        return f"failed{suffix}{runtime_suffix}", "bug"
    if node.is_buggy:
        return f"bug{suffix}{runtime_suffix}", "bug"

    metric = _metric_text(node.metric)
    if metric is None:
        return f"bug{suffix}{runtime_suffix}", "bug"
    if is_plateau_blocked_descendant(node, epsilon=plateau_block_epsilon):
        return f"{metric}{suffix}{runtime_suffix}", "blocked"
    is_public = node.id in public_node_ids
    is_public_bonus = node.id in public_bonus_node_ids
    is_public_best = node is public_best_node
    kinds: list[str] = []
    if node is best_node:
        kinds.append("best")
    if is_public:
        kinds.append("public")
    if is_public and not is_public_bonus and not is_public_best:
        kinds.append("public-worse")
    if is_public_bonus and not is_public_best:
        kinds.append("public-bonus")
    if is_public_best:
        kinds.append("public-best")
    return f"{metric}{suffix}{runtime_suffix}", " ".join(kinds) or "ok"


def build_web_tree_lines(
    journal: Journal,
    *,
    active_parent_node: Node | None = None,
    active_stage: str | None = None,
    active_hypothesis_id: str | None = None,
    plateau_block_epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
    public_scores_by_node_id: dict[str, float] | None = None,
    public_score_bonus_weight: float = 0.0,
    public_score_bonus_cap: float = 0.0,
) -> list[WebTreeLine]:
    journal_nodes = set(journal.nodes)
    public_scores_by_node_id = public_scores_by_node_id or {}
    public_node_ids = {
        node.id
        for node in journal.good_nodes
        if _node_has_public_score(
            node,
            public_scores_by_node_id=public_scores_by_node_id,
        )
    }
    public_bonus_node_ids = {
        node.id
        for node in journal.good_nodes
        if _node_public_score_bonus_active(
            node,
            public_scores_by_node_id=public_scores_by_node_id,
            weight=public_score_bonus_weight,
            cap=public_score_bonus_cap,
        )
    }
    best_node = _best_scored_node(
        journal,
        plateau_block_epsilon=plateau_block_epsilon,
    )
    public_best_node = _public_best_scored_node(
        journal,
        plateau_block_epsilon=plateau_block_epsilon,
        public_scores_by_node_id=public_scores_by_node_id,
    )
    lines: list[WebTreeLine] = []

    def visible_children(node: Node) -> list[Node]:
        return sorted(
            (
                child
                for child in node.children
                if child in journal_nodes and not child.is_terminal_failure
            ),
            key=lambda child: _node_order_key(journal, child),
        )

    def active_label() -> str:
        stage = active_stage or "running"
        if stage == "generating":
            label = "generating..."
        elif stage == "executing":
            label = "executing..."
        elif stage == "reviewing":
            label = "reviewing result..."
        else:
            label = f"{stage}..."
        if active_hypothesis_id:
            return f"{label}·{active_hypothesis_id}"
        return label

    def append_active(prefix: str, desktop_prefix: str, is_last: bool) -> None:
        if active_stage is None:
            return
        branch = "└" if is_last else "├"
        lines.append(
            WebTreeLine(
                prefix=f"{prefix}{branch}",
                label=active_label(),
                kind="active",
                desktop_prefix=f"{desktop_prefix}{branch}── ",
            )
        )

    def append_rec(
        node: Node,
        prefix: str,
        desktop_prefix: str,
        is_last: bool,
    ) -> None:
        label, kind = _line_for_node(
            node,
            best_node=best_node,
            public_best_node=public_best_node,
            plateau_block_epsilon=plateau_block_epsilon,
            public_node_ids=public_node_ids,
            public_bonus_node_ids=public_bonus_node_ids,
        )
        branch = "└" if is_last else "├"
        lines.append(
            WebTreeLine(
                prefix=f"{prefix}{branch}",
                label=label,
                kind=kind,
                desktop_prefix=f"{desktop_prefix}{branch}── ",
            )
        )
        children = visible_children(node)
        has_active_child = node is active_parent_node and active_stage is not None
        next_prefix = f"{prefix}{' ' if is_last else '│'}"
        next_desktop_prefix = f"{desktop_prefix}{'    ' if is_last else '│   '}"
        for index, child in enumerate(children):
            append_rec(
                child,
                next_prefix,
                next_desktop_prefix,
                index == len(children) - 1 and not has_active_child,
            )
        if has_active_child:
            append_active(next_prefix, next_desktop_prefix, True)

    roots = sorted(journal.draft_nodes, key=lambda node: _node_order_key(journal, node))
    has_root_active = active_parent_node is None and active_stage is not None
    for index, node in enumerate(roots):
        append_rec(node, "", "", index == len(roots) - 1 and not has_root_active)
    if has_root_active:
        append_active("", "", True)
    return lines
