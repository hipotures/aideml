from pathlib import Path

from aide.agent import Agent
from aide.journal import Journal, Node
from aide.research import write_forced_child_hypothesis_queue
from aide.synthesis import SYNTHESIS_PLAN_PREFIX
from aide.utils.config import _load_cfg, prep_cfg
from aide.utils.metric import MetricValue, WorstMetricValue


def _cfg(tmp_path: Path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "search-policy-test"
    cfg = prep_cfg(cfg)
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.max_debug_depth = 3
    return cfg


def _good_node(score: float, parent: Node | None = None) -> Node:
    node = Node(code="print('ok')", plan="ok", parent=parent)
    node.metric = MetricValue(score, maximize=True)
    node.is_buggy = False
    node.analysis = "ok"
    node._term_out = ["ok"]
    return node


def _bug_node(parent: Node | None = None) -> Node:
    node = Node(code="raise RuntimeError('bug')", plan="bug", parent=parent)
    node.metric = WorstMetricValue()
    node.is_buggy = True
    node.analysis = "bug"
    node._term_out = ["RuntimeError: bug"]
    node.exc_type = "RuntimeError"
    return node


def _hypothesis_node(node: Node, hypothesis_id: str) -> Node:
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = [hypothesis_id]
    return node


def _submission_bug_node(parent: Node | None = None) -> Node:
    node = _bug_node(parent=parent)
    node.exc_type = "SubmissionValidationError"
    node.exc_info = {"args": ["duplicate id rows: 1"]}
    node.submission_validation = {
        "status": "error",
        "error": "duplicate id rows: 1",
    }
    return node


def _failed_node(parent: Node | None = None) -> Node:
    node = Node(code="# Failed checkpoint did not produce code.\n", plan="failed", parent=parent)
    node.status = "failed"
    node.metric = WorstMetricValue()
    node.is_buggy = True
    node.analysis = "failed"
    node._term_out = ["Failed: checkpoint failed"]
    node.exc_type = "Failed"
    return node


def _oom_bug_node(parent: Node | None = None) -> Node:
    node = _bug_node(parent=parent)
    node.status = "bug"
    node.analysis = (
        "REPL child process died unexpectedly\n\n"
        "CatBoost GPU ran out of memory while the REPL child process was executing."
    )
    node._term_out = [
        "RuntimeError: REPL child process died unexpectedly\n"
        "CatBoost GPU ran out of memory while the REPL child process was executing."
    ]
    return node


def test_search_policy_does_not_debug_invalid_submission_leaf(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 1.0
    journal = Journal()
    invalid = _submission_bug_node()
    normal_bug = _bug_node()
    journal.append(invalid)
    journal.append(normal_bug)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is normal_bug


def test_search_policy_does_not_debug_catboost_gpu_oom_leaf(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 1.0
    journal = Journal()
    oom = _oom_bug_node()
    normal_bug = _bug_node()
    journal.append(oom)
    journal.append(normal_bug)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is normal_bug


def test_search_policy_does_not_debug_failed_leaf(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 1.0
    journal = Journal()
    failed = _failed_node()
    normal_bug = _bug_node()
    journal.append(failed)
    journal.append(normal_bug)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is normal_bug


def test_search_policy_ignores_good_descendants_of_invalid_submission_branch(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    journal = Journal()
    invalid = _submission_bug_node()
    unsafe_good = _good_node(0.99, parent=invalid)
    safe_good = _good_node(0.90)
    journal.append(invalid)
    journal.append(unsafe_good)
    journal.append(safe_good)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is safe_good


def test_search_policy_does_not_debug_past_configured_max_depth(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 1.0
    cfg.agent.search.max_debug_depth = 3
    journal = Journal()
    root = _bug_node()
    depth_1 = _bug_node(parent=root)
    depth_2 = _bug_node(parent=depth_1)
    depth_3 = _bug_node(parent=depth_2)
    journal.append(root)
    journal.append(depth_1)
    journal.append(depth_2)
    journal.append(depth_3)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is None


def test_search_policy_does_not_prioritize_synthesis_leaf_over_better_node(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    journal = Journal()
    best = _good_node(0.95)
    synthesis_leaf = _good_node(0.94)
    synthesis_leaf.plan = f"{SYNTHESIS_PLAN_PREFIX} 000015"
    journal.append(best)
    journal.append(synthesis_leaf)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is best


def test_search_policy_explores_underexpanded_good_node(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.05
    cfg.agent.search.best_score_min_children_before_exploration = 0
    journal = Journal()
    saturated_best = _good_node(0.95110)
    underexpanded = _good_node(0.951095)
    journal.append(saturated_best)
    journal.append(underexpanded)
    for _ in range(30):
        journal.append(_good_node(0.95090, parent=saturated_best))
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is underexpanded


def test_search_policy_ignores_terminal_failure_children_for_exploration(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.05
    cfg.agent.search.best_score_min_children_before_exploration = 0
    journal = Journal()
    active_best = _good_node(0.95110)
    underexpanded = _good_node(0.951095)
    baseline = _good_node(0.95090)
    journal.append(active_best)
    journal.append(underexpanded)
    journal.append(baseline)
    for _ in range(30):
        journal.append(_failed_node(parent=active_best))
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is active_best


def test_hypothesis_search_skips_parent_after_non_improving_child_limit(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.hypothesis_max_non_improving_children_per_parent = 2

    journal = Journal()
    saturated = _good_node(0.9510)
    fallback = _good_node(0.9500)
    worse_a = _good_node(0.9501, parent=saturated)
    worse_b = _good_node(0.9502, parent=saturated)
    for node, hypothesis_id in [
        (saturated, "000001"),
        (fallback, "000002"),
        (worse_a, "000003"),
        (worse_b, "000004"),
    ]:
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda cfg, journal: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda cfg, journal, parent_nodes: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is fallback


def test_standard_search_keeps_best_parent_after_non_improving_child_limit(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = False
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.hypothesis_max_non_improving_children_per_parent = 2
    cfg.agent.search.hypothesis_min_improvement_epsilon = 0.00006

    journal = Journal()
    saturated = _good_node(0.95100)
    fallback = _good_node(0.95000)
    near_equal = _good_node(0.95099, parent=saturated)
    worse = _good_node(0.95090, parent=saturated)
    for node in [saturated, fallback, near_equal, worse]:
        journal.append(node)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is saturated
    trace = agent.last_search_decision
    assert trace is not None
    assert saturated.id not in trace["rejections"]


def test_hypothesis_search_keeps_parent_with_improving_child_available(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.hypothesis_max_non_improving_children_per_parent = 2

    journal = Journal()
    parent = _good_node(0.9510)
    fallback = _good_node(0.9500)
    worse = _good_node(0.9501, parent=parent)
    better = _good_node(0.9512, parent=parent)
    for node, hypothesis_id in [
        (parent, "000001"),
        (fallback, "000002"),
        (worse, "000003"),
        (better, "000004"),
    ]:
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda cfg, journal: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda cfg, journal, parent_nodes: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is better


def test_hypothesis_search_requires_epsilon_improvement_for_child_candidate(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.hypothesis_min_improvement_epsilon = 0.0001

    journal = Journal()
    parent = _good_node(0.95100)
    tiny_gain = _good_node(0.95105, parent=parent)
    for node, hypothesis_id in [
        (parent, "000001"),
        (tiny_gain, "000002"),
    ]:
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda cfg, journal: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda cfg, journal, parent_nodes: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is parent
    trace = agent.last_search_decision
    assert trace is not None
    assert trace["rejections"][tiny_gain.id]["stage"] == "branch_candidate"


def test_hypothesis_saturation_treats_tiny_gain_as_non_improving(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.hypothesis_max_non_improving_children_per_parent = 1
    cfg.agent.search.hypothesis_min_improvement_epsilon = 0.0001

    journal = Journal()
    parent = _good_node(0.95100)
    fallback = _good_node(0.95090)
    tiny_gain = _good_node(0.95105, parent=parent)
    for node, hypothesis_id in [
        (parent, "000001"),
        (fallback, "000002"),
        (tiny_gain, "000003"),
    ]:
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda cfg, journal: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda cfg, journal, parent_nodes: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is fallback
    trace = agent.last_search_decision
    assert trace is not None
    assert trace["rejections"][parent.id]["stage"] == "saturation"


def test_hypothesis_non_improving_limit_ignores_bug_children(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.hypothesis_max_non_improving_children_per_parent = 1

    journal = Journal()
    parent = _good_node(0.9510)
    fallback = _good_node(0.9500)
    bug = _bug_node(parent=parent)
    for node, hypothesis_id in [
        (parent, "000001"),
        (fallback, "000002"),
        (bug, "000003"),
    ]:
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis_id]
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda cfg, journal: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda cfg, journal, parent_nodes: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is parent


def test_search_policy_keeps_oom_saturated_parent_active_by_default(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    journal = Journal()
    saturated_parent = _good_node(0.95110)
    good_child = _good_node(0.95109, parent=saturated_parent)
    fallback = _good_node(0.95090)
    journal.append(saturated_parent)
    journal.append(good_child)
    journal.append(fallback)
    for _ in range(3):
        journal.append(_oom_bug_node(parent=saturated_parent))
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is saturated_parent


def test_search_policy_blocks_parent_after_three_oom_children_when_enabled(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.disable_oom_saturated_parents = True
    journal = Journal()
    blocked_parent = _good_node(0.95110)
    good_child = _good_node(0.95109, parent=blocked_parent)
    fallback = _good_node(0.95090)
    journal.append(blocked_parent)
    journal.append(good_child)
    journal.append(fallback)
    for _ in range(3):
        journal.append(_oom_bug_node(parent=blocked_parent))
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is good_child


def test_search_policy_zero_exploration_keeps_greedy_selection(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    journal = Journal()
    saturated_best = _good_node(0.95110)
    underexpanded = _good_node(0.951095)
    journal.append(saturated_best)
    journal.append(underexpanded)
    for _ in range(30):
        journal.append(_good_node(0.95090, parent=saturated_best))
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is saturated_best


def test_hypothesis_mode_opens_new_roots_until_root_pool_is_complete(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    journal = Journal()
    for score in [0.90, 0.91]:
        journal.append(_good_node(score))
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: False,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is None


def test_hypothesis_mode_defers_debug_until_root_sweep_is_complete(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 1.0
    journal = Journal()
    bug = _bug_node()
    journal.append(bug)
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: False,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is None


def test_hypothesis_search_debugs_buggy_root_before_branching_after_root_sweep(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0

    journal = Journal()
    buggy_root = _hypothesis_node(_bug_node(), "000101")
    good_root = _hypothesis_node(_good_node(0.9510), "000102")
    for node in [buggy_root, good_root]:
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is buggy_root


def test_hypothesis_search_debugs_failed_root_before_branching_after_root_sweep(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0

    journal = Journal()
    failed_root = _hypothesis_node(_failed_node(), "000101")
    good_root = _hypothesis_node(_good_node(0.9510), "000102")
    for node in [failed_root, good_root]:
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is failed_root


def test_hypothesis_search_does_not_force_non_root_bug_before_branching(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0

    journal = Journal()
    good_root = _hypothesis_node(_good_node(0.9510), "000102")
    child_bug = _hypothesis_node(_bug_node(parent=good_root), "000103")
    fallback = _hypothesis_node(_good_node(0.9500), "000104")
    for node in [good_root, child_bug, fallback]:
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is good_root


def test_hypothesis_forced_root_scope_disables_opening_new_roots(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.forced_root = "000365"
    journal = Journal()
    root = _hypothesis_node(_good_node(0.90), "000365")
    journal.append(root)
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is root


def test_hypothesis_forced_root_scope_ignores_num_drafts(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 5
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.forced_root = "000365"
    journal = Journal()
    root = _hypothesis_node(_good_node(0.90), "000365")
    journal.append(root)
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is root
    assert agent.last_search_decision["reason"] != "not_enough_drafts"


def test_hypothesis_forced_root_scope_selects_only_descendants(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    cfg.agent.search.forced_root = "000365"
    journal = Journal()
    forced_root = _hypothesis_node(_good_node(0.9500), "000365")
    forced_child = _hypothesis_node(_good_node(0.9510, parent=forced_root), "000020")
    outside_root = _hypothesis_node(_good_node(0.9600), "000777")
    outside_child = _hypothesis_node(_good_node(0.9700, parent=outside_root), "000888")
    for node in [forced_root, forced_child, outside_root, outside_child]:
        journal.append(node)
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is forced_child
    trace = agent.last_search_decision
    assert trace is not None
    assert trace["forced_hypothesis_root"] == "000365"
    assert trace["best_node"]["node_id"] == outside_child.id
    assert trace["best_node"]["selected"] is False
    assert trace["best_node"]["rejected_at"] == "forced_root_scope"


def test_hypothesis_forced_hypothesis_does_not_expand_child_hypotheses(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.forced_hypothesis = "000941"
    journal = Journal()
    root = _hypothesis_node(_good_node(0.95226), "000941")
    journal.append(root)
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: [],
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is None
    trace = agent.last_search_decision
    assert trace is not None
    assert trace["reason"] == "no_good_nodes_after_filters"


def test_hypothesis_search_prioritizes_forced_child_queue_root(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    write_forced_child_hypothesis_queue(
        cfg,
        root_hypothesis="001172",
        children=("001176",),
    )
    journal = Journal()
    root = _hypothesis_node(_good_node(0.950), "001172")
    better_child = _hypothesis_node(_good_node(0.960, parent=root), "000806")
    journal.append(root)
    journal.append(better_child)
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.forced_child_hypothesis_ids_for_node",
        lambda _cfg, _journal, node, **_kwargs: ["001176"] if node is root else [],
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is root
    assert agent.last_search_decision["reason"] == "forced_child_hypothesis_queue"


def test_hypothesis_search_prioritizes_forced_child_queue_nonroot_parent(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.forced_root = "001189"
    write_forced_child_hypothesis_queue(
        cfg,
        root_hypothesis="001189",
        children=("001193",),
    )
    journal = Journal()
    root = _hypothesis_node(_good_node(0.950), "001172")
    branch_parent = _hypothesis_node(_good_node(0.954, parent=root), "001189")
    sibling = _hypothesis_node(_good_node(0.960, parent=root), "000806")
    journal.append(root)
    journal.append(branch_parent)
    journal.append(sibling)
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.forced_child_hypothesis_ids_for_node",
        lambda _cfg, _journal, node, **_kwargs: (
            ["001193"] if node is branch_parent else []
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is branch_parent
    trace = agent.last_search_decision
    assert trace is not None
    assert trace["reason"] == "forced_child_hypothesis_queue"
    assert trace["forced_hypothesis_root"] == "001189"


def test_hypothesis_forced_root_scope_debugs_only_descendants(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 1.0
    cfg.agent.search.forced_root = "000365"
    journal = Journal()
    forced_root = _hypothesis_node(_good_node(0.9500), "000365")
    forced_bug = _hypothesis_node(_bug_node(parent=forced_root), "000020")
    outside_root = _hypothesis_node(_good_node(0.9600), "000777")
    outside_bug = _hypothesis_node(_bug_node(parent=outside_root), "000888")
    for node in [outside_root, outside_bug, forced_root, forced_bug]:
        journal.append(node)
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: False,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is forced_bug


def test_hypothesis_mode_does_not_open_root_when_root_pool_is_exhausted(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.research.every_steps = 3
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    journal = Journal()
    best = _good_node(0.92)
    for node in [_good_node(0.90), _good_node(0.91), best]:
        journal.append(node)
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is best


def test_hypothesis_search_trace_marks_top_debug_fix_as_selected(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0

    journal = Journal()
    fallback = _hypothesis_node(_good_node(0.95193), "000011")
    bug = _hypothesis_node(_bug_node(parent=fallback), "000002")
    best = _hypothesis_node(_good_node(0.95239, parent=bug), "000002")
    for node in [fallback, bug, best]:
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is best
    trace = agent.last_search_decision
    assert trace is not None
    assert trace["selected"]["node_id"] == best.id
    assert trace["best_node"]["node_id"] == best.id
    assert trace["best_node"]["selected"] is True


def test_hypothesis_search_can_select_top_debug_fix_after_bug_parent(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0

    journal = Journal()
    ancestor = _hypothesis_node(_good_node(0.95209), "000746")
    fallback = _hypothesis_node(_good_node(0.95193), "000011")
    bug = _hypothesis_node(_bug_node(parent=ancestor), "000002")
    best = _hypothesis_node(_good_node(0.95239, parent=bug), "000002")
    for node in [ancestor, fallback, bug, best]:
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is best


def test_search_policy_appends_decision_jsonl(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    journal = Journal()
    selected = _good_node(0.92)
    journal.append(selected)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    assert agent.search_policy() is selected

    decision_log = Path(cfg.log_dir) / "search_decisions.jsonl"
    assert decision_log.exists()
    records = decision_log.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    assert selected.id in records[0]


def test_hypothesis_mode_filters_parents_without_child_candidates(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.research.every_steps = 0
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.0
    journal = Journal()
    exhausted_parent = _good_node(0.95)
    available_parent = _good_node(0.94)
    journal.append(exhausted_parent)
    journal.append(available_parent)

    def fake_filter(_cfg, *, journal, parent_nodes, **_kwargs):
        return [node for node in parent_nodes if node is available_parent]

    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        fake_filter,
    )
    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is available_parent


def test_search_policy_exploration_respects_minimization_metrics(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.05
    journal = Journal()
    saturated_best = _good_node(0.100)
    saturated_best.metric = MetricValue(0.100, maximize=False)
    underexpanded = _good_node(0.100005)
    underexpanded.metric = MetricValue(0.100005, maximize=False)
    journal.append(saturated_best)
    journal.append(underexpanded)
    for _ in range(30):
        child = _good_node(0.102, parent=saturated_best)
        child.metric = MetricValue(0.102, maximize=False)
        journal.append(child)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is underexpanded


def test_search_policy_trace_explains_exploration_threshold(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.05
    cfg.agent.search.best_score_min_children_before_exploration = 0
    journal = Journal()
    best = _good_node(0.95264)
    selected = _good_node(0.95263)
    low = _good_node(0.94939)
    child = _good_node(0.95200, parent=best)
    for node in [best, selected, low, child]:
        journal.append(node)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    assert agent.search_policy() is selected

    trace = agent.last_search_decision
    assert trace is not None
    diagnostics = trace["policy_diagnostics"]
    assert diagnostics["candidate_count"] == 4
    assert diagnostics["exploration_weight"] == 0.05
    assert diagnostics["best"]["node_id"] == best.id
    assert diagnostics["selected"]["node_id"] == selected.id
    assert diagnostics["selected_minus_best_policy_score"] > 0
    assert diagnostics["selected_minus_best_metric"] < 0
    assert diagnostics["fresh_child_metric_threshold"]["child_count"] == 0
    assert diagnostics["fresh_child_metric_threshold"]["direction"] == ">="
    assert diagnostics["fresh_child_metric_threshold"]["metric"] < best.metric.value


def test_search_policy_exploits_best_score_until_min_child_count(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.05
    cfg.agent.search.best_score_min_children_before_exploration = 3
    journal = Journal()
    best = _good_node(0.95264)
    fresh_lower = _good_node(0.95125)
    outlier = _good_node(0.49985)
    best_child = _good_node(0.95237, parent=best)
    for node in [best, fresh_lower, outlier, best_child]:
        journal.append(node)
    for index in range(96):
        journal.append(_good_node(0.94939 + index * 0.00001))
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    assert agent.search_policy() is best

    trace = agent.last_search_decision
    assert trace is not None
    assert trace["reason"] == "best_score_min_children_before_exploration"
    diagnostics = trace["policy_diagnostics"]
    assert diagnostics["selection_override"]["best_child_count"] == 1
    assert diagnostics["selection_override"]["min_children"] == 3
    assert diagnostics["selected"]["node_id"] == best.id
    assert diagnostics["fresh_child_metric_threshold"]["metric"] < fresh_lower.metric.value


def test_search_policy_exploits_best_remaining_score_after_top_saturates(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.search.num_drafts = 0
    cfg.agent.search.debug_prob = 0.0
    cfg.agent.search.exploration_weight = 0.05
    cfg.agent.search.best_score_min_children_before_exploration = 3
    cfg.agent.search.hypothesis_max_non_improving_children_per_parent = 3

    journal = Journal()
    lower_parent = _hypothesis_node(_good_node(0.95237), "000011")
    next_best = _hypothesis_node(_good_node(0.95249, parent=lower_parent), "000703")
    saturated_top = _hypothesis_node(_good_node(0.95264, parent=next_best), "000459")
    stale_children = [
        _hypothesis_node(_good_node(0.95237, parent=saturated_top), "000749"),
        _hypothesis_node(_good_node(0.95257, parent=saturated_top), "000052"),
        _hypothesis_node(_good_node(0.95219, parent=saturated_top), "000234"),
    ]
    fresh_lower = _hypothesis_node(_good_node(0.95093), "000634")
    for node in [
        lower_parent,
        next_best,
        saturated_top,
        *stale_children,
        fresh_lower,
    ]:
        journal.append(node)

    monkeypatch.setattr(
        "aide.agent.hypothesis_root_pool_exhausted",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "aide.agent.filter_hypothesis_candidate_parents",
        lambda _cfg, *, journal, parent_nodes, **_kwargs: parent_nodes,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    assert agent.search_policy() is next_best

    trace = agent.last_search_decision
    assert trace is not None
    assert trace["best_node"]["node_id"] == saturated_top.id
    assert trace["best_node"]["rejected_at"] == "saturation"
    diagnostics = trace["policy_diagnostics"]
    assert diagnostics["best"]["node_id"] == next_best.id
    assert diagnostics["selection_override"]["best_child_count"] == 1
