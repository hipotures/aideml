from __future__ import annotations

from aide.journal import Journal, Node
from aide.utils.metric import MetricValue
from aide.utils.plateau import (
    DEFAULT_PLATEAU_BLOCK_EPSILON,
    is_plateau_blocked_descendant,
)

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


def _best_scored_node(journal: Journal, *, plateau_block_epsilon: float) -> Node | None:
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


def _metric_text(metric: MetricValue | None) -> str | None:
    if metric is None or metric.value is None:
        return None
    return f"{metric.value:.5f}"


def _line_for_node(
    node: Node,
    *,
    best_node: Node | None,
    plateau_block_epsilon: float,
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
    if node is best_node:
        return f"{metric}{suffix}{runtime_suffix}", "best"
    return f"{metric}{suffix}{runtime_suffix}", "ok"


def build_web_tree_lines(
    journal: Journal,
    *,
    active_parent_node: Node | None = None,
    active_stage: str | None = None,
    active_hypothesis_id: str | None = None,
    plateau_block_epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
) -> list[WebTreeLine]:
    journal_nodes = set(journal.nodes)
    best_node = _best_scored_node(
        journal,
        plateau_block_epsilon=plateau_block_epsilon,
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

    def append_active(prefix: str, is_last: bool) -> None:
        if active_stage is None:
            return
        lines.append(
            WebTreeLine(
                prefix=f"{prefix}{'└' if is_last else '├'}",
                label=active_label(),
                kind="active",
            )
        )

    def append_rec(node: Node, prefix: str, is_last: bool) -> None:
        label, kind = _line_for_node(
            node,
            best_node=best_node,
            plateau_block_epsilon=plateau_block_epsilon,
        )
        lines.append(
            WebTreeLine(
                prefix=f"{prefix}{'└' if is_last else '├'}",
                label=label,
                kind=kind,
            )
        )
        children = visible_children(node)
        has_active_child = node is active_parent_node and active_stage is not None
        next_prefix = f"{prefix}{' ' if is_last else '│'}"
        for index, child in enumerate(children):
            append_rec(
                child,
                next_prefix,
                index == len(children) - 1 and not has_active_child,
            )
        if has_active_child:
            append_active(next_prefix, True)

    roots = sorted(journal.draft_nodes, key=lambda node: _node_order_key(journal, node))
    has_root_active = active_parent_node is None and active_stage is not None
    for index, node in enumerate(roots):
        append_rec(node, "", index == len(roots) - 1 and not has_root_active)
    if has_root_active:
        append_active("", True)
    return lines
