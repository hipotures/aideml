from pathlib import Path

from aide.agent import Agent
from aide.journal import Journal, Node
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
