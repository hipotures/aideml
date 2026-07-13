import json
from pathlib import Path

from aide.agent import Agent
from aide.journal import Journal, Node
from aide.run import validate_code_ahead, validate_hypothesis_root_generate_workers
from aide.utils.config import _load_cfg, prep_cfg


def _cfg(tmp_path: Path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "branch-session-test"
    cfg.agent.code.codex_branch_sessions = True
    return prep_cfg(cfg)


def test_new_parent_forks_its_generation_turn_then_resumes_group(tmp_path):
    cfg = _cfg(tmp_path)
    parent = Node(
        code="print('parent')",
        plan="parent",
        codex_thread_id="thread-root",
        codex_turn_id="turn-parent",
    )
    journal = Journal()
    journal.append(parent)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    agent.active_parent_node = parent

    assert agent._codex_session_kwargs() == {
        "codex_fork_thread_id": "thread-root",
        "codex_fork_turn_id": "turn-parent",
    }

    agent._record_codex_generation(
        {"thread_id": "thread-children", "turn_id": "turn-child-1"}
    )

    assert agent._codex_session_kwargs() == {"codex_thread_id": "thread-children"}
    registry = json.loads((Path(cfg.log_dir) / "codex_sessions.json").read_text())
    assert registry["groups"][parent.id]["thread_id"] == "thread-children"


def test_root_group_uses_latest_existing_root_when_registry_is_missing(tmp_path):
    cfg = _cfg(tmp_path)
    journal = Journal()
    first = Node(
        code="print(0)",
        plan="first",
        codex_thread_id="thread-0",
        codex_turn_id="turn-0",
    )
    latest = Node(
        code="print(1)",
        plan="latest",
        codex_thread_id="thread-1",
        codex_turn_id="turn-1",
    )
    journal.append(first)
    journal.append(latest)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    assert agent._codex_session_kwargs() == {"codex_thread_id": "thread-1"}


def test_legacy_exec_node_backfills_fork_from_rollout_path(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    parent = Node(code="print('parent')", plan="parent")
    parent.artifact_dir_name = "parent-artifact"
    journal = Journal()
    journal.append(parent)
    artifact_dir = Path(cfg.log_dir) / "artifacts" / parent.artifact_dir_name
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "codex_events.jsonl").write_text(
        '{"type":"thread.started","thread_id":"legacy-thread"}\n'
    )
    rollout = tmp_path / "rollout-legacy-thread.jsonl"
    rollout.write_text("{}\n")
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    agent.active_parent_node = parent
    monkeypatch.setattr(agent, "_legacy_codex_rollout", lambda _thread_id: rollout)

    assert agent._codex_session_kwargs() == {
        "codex_fork_thread_id": "legacy-thread",
        "codex_fork_path": str(rollout),
    }


def test_parse_retry_resumes_thread_created_by_previous_attempt(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    calls = []

    def fake_query_with_info(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return "not code", {"thread_id": "thread-1", "turn_id": "turn-1"}
        return (
            "Plan.\n```python\nprint('ok')\n```",
            {"thread_id": "thread-1", "turn_id": "turn-2"},
        )

    monkeypatch.setitem(
        agent.plan_and_code_query.__globals__,
        "query_with_info",
        fake_query_with_info,
    )

    plan, code = agent.plan_and_code_query({"Instructions": {}})

    assert plan == "Plan."
    assert 'print("ok")' in code
    assert "codex_thread_id" not in calls[0]
    assert calls[1]["codex_thread_id"] == "thread-1"
    assert agent._pending_codex_turn_id == "turn-2"


def test_branch_sessions_force_sequential_generation(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.search.code_ahead = 3
    cfg.research.hypothesis_root_generate_workers = 4

    assert validate_code_ahead(cfg) == 0
    assert validate_hypothesis_root_generate_workers(cfg) == 1
    assert cfg.agent.search.code_ahead == 0
    assert cfg.research.hypothesis_root_generate_workers == 1
