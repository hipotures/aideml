import json
import subprocess
from pathlib import Path

import pytest

from aide.journal import Journal, Node
from aide.synthesis import (
    SYNTHESIS_PROMPT_INTRO,
    SynthesisAdvisor,
    build_synthesis_prompt,
    collect_synthesis_context,
    collect_top_synthesis_solutions,
    parse_synthesis_code,
    run_synthesis_checkpoint,
)
from aide.utils import serialize
from aide.utils.config import _load_cfg, prep_cfg
from aide.utils.metric import MetricValue, WorstMetricValue


def _cfg(tmp_path: Path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "Predict next-lap pit stop probability"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "synthesis-test"
    cfg.synthesis.enabled = True
    cfg = prep_cfg(cfg)
    return cfg


def _node(score: float | None, *, code: str, buggy: bool = False) -> Node:
    node = Node(code=code, plan="plan")
    node.metric = (
        WorstMetricValue() if score is None else MetricValue(score, maximize=True)
    )
    node.is_buggy = buggy or score is None
    node.analysis = "analysis"
    node._term_out = ["output"]
    node.exec_time = 1.0
    node.exc_type = "RuntimeError" if node.is_buggy else None
    return node


def _write_journal(log_dir: Path, run_id: str, journal: Journal) -> None:
    run_dir = log_dir / run_id
    run_dir.mkdir(parents=True)
    serialize.dump_json(journal, run_dir / "journal.json")


def test_collect_top_synthesis_solutions_uses_best_scored_nodes_across_runs(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.synthesis.top_k = 3
    current = Journal()
    current.append(_node(0.90, code="print('current')"))
    current.append(_node(None, code="raise RuntimeError('bug')"))

    other = Journal()
    other.append(_node(0.95, code="print('best')"))
    other.append(_node(0.70, code="print('weak')"))
    _write_journal(Path(cfg.log_dir).parent, "1-other-run", other)

    solutions = collect_top_synthesis_solutions(cfg=cfg, journal=current)

    assert [solution["metric"] for solution in solutions] == [0.95, 0.9, 0.7]
    assert [solution["code"] for solution in solutions] == [
        "print('best')",
        "print('current')",
        "print('weak')",
    ]
    assert all(set(solution) == {"metric", "code"} for solution in solutions)


def test_collect_top_synthesis_solutions_honors_explicit_source_runs(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.synthesis.top_k = 5
    cfg.synthesis.source_runs = ["2-included-run"]
    current = Journal()
    current.append(_node(0.99, code="print('current')"))

    included = Journal()
    included.append(_node(0.80, code="print('included')"))
    excluded = Journal()
    excluded.append(_node(0.95, code="print('excluded')"))
    _write_journal(Path(cfg.log_dir).parent, "2-included-run", included)
    _write_journal(Path(cfg.log_dir).parent, "3-excluded-run", excluded)

    solutions = collect_top_synthesis_solutions(cfg=cfg, journal=current)

    assert solutions == [{"metric": 0.8, "code": "print('included')"}]


def test_synthesis_prompt_contains_only_relevant_context(tmp_path):
    cfg = _cfg(tmp_path)
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')"))

    context = collect_synthesis_context(
        cfg=cfg,
        task_desc="task",
        journal=journal,
        completed_steps=15,
    )
    prompt = build_synthesis_prompt(context)

    assert prompt.startswith(SYNTHESIS_PROMPT_INTRO)
    assert "Return only Python code" in prompt
    assert "task" in prompt
    assert '"best_working_solutions"' in prompt
    assert '"metric"' in prompt
    assert '"code"' in prompt
    assert '"run_id"' not in prompt
    assert '"checkpoint_step"' not in prompt
    assert '"created_at"' not in prompt
    assert '"step"' not in prompt
    assert '"stage"' not in prompt
    assert '"plan"' not in prompt
    assert '"analysis"' not in prompt


def test_parse_synthesis_code_accepts_raw_and_fenced_python():
    raw = "value = 1\nprint(value)\n"
    fenced = "```python\nvalue = 2\nprint(value)\n```\n"

    assert parse_synthesis_code(raw) == raw
    assert parse_synthesis_code(fenced) == "value = 2\nprint(value)\n"

    with pytest.raises(ValueError):
        parse_synthesis_code("not python prose")


def test_run_synthesis_checkpoint_logs_request_and_python_response(tmp_path):
    cfg = _cfg(tmp_path)
    context = {
        "run_id": cfg.exp_name,
        "checkpoint_step": 15,
        "task_desc": "task",
        "best_working_solutions": [{"metric": 0.9, "code": "print('old')"}],
    }
    seen = {}

    def fake_runner(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["stdin"] = kwargs["input"]
        checkpoint_dir = Path(cmd[cmd.index("--cd") + 1])
        (checkpoint_dir / "response_raw.txt").write_text("value = 1\nprint(value)\n")
        return subprocess.CompletedProcess(
            cmd, 0, stdout='{"event":"done"}\n', stderr=""
        )

    result = run_synthesis_checkpoint(
        cfg=cfg,
        context=context,
        runner=fake_runner,
    )

    checkpoint_dir = Path(result["checkpoint_dir"])
    command = seen["cmd"]

    assert command[:6] == [
        "codex",
        "--search",
        "--ask-for-approval",
        "never",
        "exec",
        "--ignore-user-config",
    ]
    assert "--output-schema" not in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert command[command.index("--model") + 1] == "gpt-5.5"
    assert 'model_reasoning_effort="medium"' in command
    assert seen["stdin"].startswith(SYNTHESIS_PROMPT_INTRO)
    assert (checkpoint_dir / "request.json").exists()
    assert (checkpoint_dir / "request.md").exists()
    assert (checkpoint_dir / "response.py").read_text() == "value = 1\nprint(value)\n"
    response = json.loads((checkpoint_dir / "response.json").read_text())
    assert response["code"] == "value = 1\nprint(value)\n"
    status = json.loads((checkpoint_dir / "status.json").read_text())
    assert status["status"] == "ready"


def test_synthesis_advisor_generates_root_node_once_per_checkpoint(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.synthesis.every_scored_steps = 2
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')"))
    journal.append(_node(None, code="raise RuntimeError('bug')"))

    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append(cmd)
        checkpoint_dir = Path(cmd[cmd.index("--cd") + 1])
        (checkpoint_dir / "response_raw.txt").write_text("value = 2\nprint(value)\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    advisor = SynthesisAdvisor(cfg=cfg, task_desc="task", runner=fake_runner)

    assert advisor.generate_node_if_due(journal=journal, completed_steps=1) is None

    journal.append(_node(0.8, code="print('ok2')"))
    synthesized = advisor.generate_node_if_due(journal=journal, completed_steps=2)

    assert synthesized is not None
    assert synthesized.node.parent is None
    assert synthesized.node.code == "value = 2\nprint(value)\n"
    assert len(calls) == 1

    journal.append(synthesized.node)
    advisor.mark_injected(synthesized, node=synthesized.node)

    assert advisor.generate_node_if_due(journal=journal, completed_steps=2) is None
    assert len(calls) == 1
    assert "Synthesis: ✓ 000002" in advisor.status_text()
