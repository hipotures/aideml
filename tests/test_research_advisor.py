import json
import subprocess
from pathlib import Path

from aide.agent import Agent
from aide.journal import Journal, Node
from aide.research import (
    RESEARCH_PROMPT_INTRO,
    ResearchAdvisor,
    build_research_prompt,
    collect_research_context,
    load_latest_research_hints,
    run_research_checkpoint,
)
from aide.utils.config import _load_cfg, prep_cfg
from aide.utils.metric import MetricValue, WorstMetricValue


def _cfg(tmp_path: Path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "Predict next-lap pit stop probability"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "research-test"
    cfg.research.enabled = True
    cfg = prep_cfg(cfg)
    return cfg


def _node(score: float | None, *, code: str, plan: str, buggy: bool = False) -> Node:
    node = Node(code=code, plan=plan)
    node.metric = (
        WorstMetricValue() if score is None else MetricValue(score, maximize=True)
    )
    node.is_buggy = buggy or score is None
    node.analysis = "analysis"
    node._term_out = ["output"]
    node.exec_time = 1.0
    node.exc_type = "RuntimeError" if node.is_buggy else None
    return node


def test_collect_research_context_selects_top_best_and_top_worst_nodes(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.research.top_k_best = 2
    cfg.research.top_k_worst = 2
    journal = Journal()
    for node in [
        _node(0.81, code="print('mid')", plan="mid"),
        _node(0.95, code="print('best')", plan="best"),
        _node(0.10, code="print('weak')", plan="weak"),
        _node(None, code="raise RuntimeError('bug')", plan="bug"),
    ]:
        journal.append(node)

    context = collect_research_context(
        cfg=cfg,
        task_desc="task",
        journal=journal,
        completed_steps=10,
    )

    assert [n["plan"] for n in context["top_best_nodes"][:2]] == ["best", "mid"]
    assert context["top_worst_nodes"][0]["is_buggy"] is True
    assert context["top_worst_nodes"][1]["plan"] == "weak"
    assert context["selected_node_ids"] == [
        journal.nodes[1].id,
        journal.nodes[0].id,
        journal.nodes[3].id,
        journal.nodes[2].id,
    ]


def test_research_prompt_starts_with_researcher_instruction(tmp_path):
    cfg = _cfg(tmp_path)
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')", plan="ok"))
    context = collect_research_context(
        cfg=cfg,
        task_desc="task",
        journal=journal,
        completed_steps=10,
    )

    prompt = build_research_prompt(context)

    assert prompt.startswith(RESEARCH_PROMPT_INTRO)
    assert "Return only structured JSON" in prompt
    assert "task" in prompt


def test_run_research_checkpoint_logs_request_and_response(tmp_path):
    cfg = _cfg(tmp_path)
    context = {
        "run_id": cfg.exp_name,
        "checkpoint_step": 10,
        "selected_node_ids": ["node-a"],
        "task_desc": "task",
        "top_best_nodes": [],
        "top_worst_nodes": [],
    }
    seen = {}

    def fake_runner(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["stdin"] = kwargs["input"]
        checkpoint_dir = Path(cmd[cmd.index("--cd") + 1])
        (checkpoint_dir / "response_raw.txt").write_text(
            json.dumps(
                {
                    "summary": "researched",
                    "hypotheses": [
                        {
                            "target": "root",
                            "parent_node_id": None,
                            "title": "Try calibrated LightGBM",
                            "rationale": "AUC often benefits from calibration checks.",
                            "implementation_hint": "Add calibrated CV probabilities.",
                            "expected_effect": "small AUC gain",
                            "risk": "overfitting",
                            "sources": ["https://example.com"],
                        }
                    ],
                }
            )
        )
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"event":"done"}\n', stderr=""
        )

    result = run_research_checkpoint(
        cfg=cfg,
        context=context,
        runner=fake_runner,
    )

    checkpoint_dir = Path(result["checkpoint_dir"])
    command = seen["cmd"]

    assert "--ignore-user-config" in command
    assert "--search" in command
    assert command[command.index("--ask-for-approval") + 1] == "never"
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-5.5"
    assert 'model_reasoning_effort="medium"' in command
    assert seen["stdin"].startswith(RESEARCH_PROMPT_INTRO)
    assert (checkpoint_dir / "request.json").exists()
    assert (checkpoint_dir / "request.md").exists()
    assert (
        (checkpoint_dir / "codex_profile.toml")
        .read_text()
        .startswith('model = "gpt-5.5"')
    )
    response = json.loads((checkpoint_dir / "response.json").read_text())
    assert response["parsed_response"]["summary"] == "researched"
    assert response["exit_code"] == 0
    status = json.loads((checkpoint_dir / "status.json").read_text())
    assert status["status"] == "completed"


def test_research_advisor_does_not_duplicate_existing_checkpoint(tmp_path):
    cfg = _cfg(tmp_path)
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')", plan="ok"))
    checkpoint_dir = Path(cfg.log_dir) / "research" / "checkpoint-000010"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "status.json").write_text('{"status": "completed"}')

    advisor = ResearchAdvisor(cfg=cfg, task_desc="task", runner=lambda *_a, **_k: None)

    assert advisor.maybe_start(journal=journal, completed_steps=10) is False


def test_research_advisor_status_text_shows_checkpoint_status(tmp_path):
    cfg = _cfg(tmp_path)
    checkpoint_dir = Path(cfg.log_dir) / "research" / "checkpoint-000010"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "status.json").write_text('{"status": "queued"}')
    advisor = ResearchAdvisor(cfg=cfg, task_desc="task", runner=lambda *_a, **_k: None)

    assert advisor.status_text() == "[cyan]Research: … 000010"


def test_load_latest_research_hints_returns_latest_completed_checkpoint(tmp_path):
    cfg = _cfg(tmp_path)
    older = Path(cfg.log_dir) / "research" / "checkpoint-000010"
    newer = Path(cfg.log_dir) / "research" / "checkpoint-000020"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "status.json").write_text('{"status": "completed"}')
    (older / "response.json").write_text(
        json.dumps({"parsed_response": {"summary": "old", "hypotheses": []}})
    )
    (newer / "status.json").write_text('{"status": "completed"}')
    (newer / "response.json").write_text(
        json.dumps({"parsed_response": {"summary": "new", "hypotheses": []}})
    )

    hints = load_latest_research_hints(cfg.log_dir)

    assert hints["checkpoint"] == "checkpoint-000020"
    assert hints["summary"] == "new"


def test_agent_includes_latest_research_hints_in_draft_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.data_preview = False
    checkpoint = Path(cfg.log_dir) / "research" / "checkpoint-000010"
    checkpoint.mkdir(parents=True)
    (checkpoint / "status.json").write_text('{"status": "completed"}')
    (checkpoint / "response.json").write_text(
        json.dumps(
            {
                "parsed_response": {
                    "summary": "research summary",
                    "hypotheses": [{"title": "Use tire-age feature"}],
                }
            }
        )
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    assert (
        captured["prompt"]["External research hints"]["summary"] == "research summary"
    )
