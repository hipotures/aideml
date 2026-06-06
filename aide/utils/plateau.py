from __future__ import annotations

from aide.journal import Node

DEFAULT_PLATEAU_BLOCK_EPSILON = 0.00001


def nearest_scored_ancestor(node: Node) -> Node | None:
    parent = node.parent
    while parent is not None:
        if parent.metric is not None and parent.metric.value is not None:
            return parent
        parent = parent.parent
    return None


def _metric_maximize(node: Node, ancestor: Node) -> bool:
    if node.metric is not None and node.metric.maximize is not None:
        return node.metric.maximize is not False
    if ancestor.metric is not None and ancestor.metric.maximize is not None:
        return ancestor.metric.maximize is not False
    return True


def _oriented_metric_value(node: Node, *, maximize: bool) -> float | None:
    if node.metric is None or node.metric.value is None:
        return None
    value = float(node.metric.value)
    return value if maximize else -value


def plateau_delta_to_nearest_scored_ancestor(node: Node) -> float | None:
    ancestor = nearest_scored_ancestor(node)
    if ancestor is None:
        return None
    maximize = _metric_maximize(node, ancestor)
    node_value = _oriented_metric_value(node, maximize=maximize)
    ancestor_value = _oriented_metric_value(ancestor, maximize=maximize)
    if node_value is None or ancestor_value is None:
        return None
    return node_value - ancestor_value


def is_plateau_blocked_descendant(
    node: Node,
    *,
    epsilon: float = DEFAULT_PLATEAU_BLOCK_EPSILON,
) -> bool:
    if node.parent is None:
        return False
    if node.is_buggy or node.is_terminal_failure:
        return False
    delta = plateau_delta_to_nearest_scored_ancestor(node)
    if delta is None:
        return False
    epsilon = max(0.0, float(epsilon))
    return -epsilon <= delta <= 0.0
