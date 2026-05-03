from pathlib import Path

from aide.agent import Agent
from aide.journal import Journal, Node
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
