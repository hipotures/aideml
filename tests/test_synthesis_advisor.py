import datetime as dt
import json
import subprocess
from pathlib import Path

import pytest

from aide.journal import Journal, Node
from aide.autogluon_preprocess import AGENT_MODE, build_autogluon_wrapper
from aide.synthesis import (
    SYNTHESIS_PROMPT_INTRO,
    SYNTHESIS_PREPROCESS_PROMPT_INTRO,
    SYNTHESIS_PLAN_PREFIX,
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


def _node(
    score: float | None,
    *,
    code: str,
    buggy: bool = False,
    parent: Node | None = None,
    ctime: float | None = None,
) -> Node:
    kwargs = {"ctime": ctime} if ctime is not None else {}
    node = Node(code=code, plan="plan", parent=parent, **kwargs)
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


def _write_submission(cfg, node: Node, body: str, *, run_id: str | None = None) -> None:
    timestamp = dt.datetime.fromtimestamp(node.ctime).strftime("%Y%m%dT%H%M%S")
    artifact_dir = (
        Path(cfg.log_dir).parent
        / (run_id or cfg.exp_name)
        / "artifacts"
        / timestamp
    )
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "submission.csv").write_text(body)


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

    assert [solution["local_cv_score"] for solution in solutions] == [0.95, 0.9, 0.7]
    assert [solution["code"] for solution in solutions] == [
        "print('best')",
        "print('current')",
        "print('weak')",
    ]
    assert all(set(solution) == {"local_cv_score", "code"} for solution in solutions)


def test_collect_top_synthesis_solutions_adds_completed_kaggle_public_score(tmp_path):
    cfg = _cfg(tmp_path)
    journal = Journal()
    node = _node(0.900006, code="print('submitted')")
    journal.append(node)
    registry_path = Path(cfg.log_dir).parent / "submission_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "submissions": [
                    {
                        "run": cfg.exp_name,
                        "step": node.step,
                        "timestamp": dt.datetime.fromtimestamp(node.ctime).strftime(
                            "%Y%m%dT%H%M%S"
                        ),
                        "node_id": node.id,
                        "remote_status": "COMPLETE",
                        "public_score": "0.812344",
                    }
                ]
            }
        )
    )

    solutions = collect_top_synthesis_solutions(cfg=cfg, journal=journal)

    assert solutions == [
        {
            "local_cv_score": 0.90001,
            "kaggle_public_score": 0.81234,
            "code": "print('submitted')",
        }
    ]


def test_collect_top_synthesis_solutions_rounds_scores_for_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    journal = Journal()
    node = _node(0.9507496188899213, code="print('rounding')")
    journal.append(node)

    solutions = collect_top_synthesis_solutions(cfg=cfg, journal=journal)

    assert solutions[0]["local_cv_score"] == 0.95075


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

    assert solutions == [{"local_cv_score": 0.8, "code": "print('included')"}]


def test_collect_top_synthesis_solutions_filters_similar_related_predictions(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    cfg.synthesis.top_k = 5
    journal = Journal()
    parent = _node(0.948596, code="print('parent')", ctime=1777716000.0)
    child = _node(
        0.948604,
        code="print('child')",
        parent=parent,
        ctime=1777716060.0,
    )
    sibling = _node(
        0.948604,
        code="print('sibling')",
        parent=parent,
        ctime=1777716120.0,
    )
    unrelated = _node(0.948604, code="print('unrelated')", ctime=1777716180.0)
    journal.append(parent)
    journal.append(child)
    journal.append(sibling)
    journal.append(unrelated)
    _write_submission(
        cfg,
        parent,
        "id,target\n1,0.100000\n2,0.200000\n",
    )
    _write_submission(
        cfg,
        child,
        "id,target\n1,0.105000\n2,0.205000\n",
    )
    _write_submission(
        cfg,
        sibling,
        "id,target\n1,0.800000\n2,0.900000\n",
    )
    _write_submission(
        cfg,
        unrelated,
        "id,target\n1,0.110000\n2,0.210000\n",
    )

    solutions = collect_top_synthesis_solutions(cfg=cfg, journal=journal)

    assert "print('child')" not in [solution["code"] for solution in solutions]
    assert {solution["code"] for solution in solutions} == {
        "print('unrelated')",
        "print('sibling')",
        "print('parent')",
    }


def test_collect_top_synthesis_solutions_keeps_child_when_rounded_score_improves(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    cfg.synthesis.top_k = 5
    journal = Journal()
    parent = _node(0.94859, code="print('parent')")
    child = _node(0.94861, code="print('child')", parent=parent)
    journal.append(parent)
    journal.append(child)

    solutions = collect_top_synthesis_solutions(cfg=cfg, journal=journal)

    assert [solution["code"] for solution in solutions] == [
        "print('child')",
        "print('parent')",
    ]


def test_collect_top_synthesis_solutions_excludes_next_pitstop_leakage(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.synthesis.top_k = 2
    journal = Journal()
    leaky = _node(
        0.99,
        code=(
            "test_sorted['next_PitStop'] = "
            "test_sorted.groupby(['Race', 'Driver'])['PitStop'].shift(-1)\n"
            "test_preds[mask] = test_sorted['next_PitStop'][mask]\n"
        ),
    )
    clean = _node(0.90, code="print('clean model')\n")
    journal.append(leaky)
    journal.append(clean)

    solutions = collect_top_synthesis_solutions(cfg=cfg, journal=journal)

    assert solutions == [{"local_cv_score": 0.9, "code": "print('clean model')\n"}]


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
    assert "strong time and memory efficiency" in prompt
    assert "avoid unnecessary full-data copies" in prompt
    assert "Do not use target leakage" in prompt
    assert "future PitStop" in prompt
    assert "task" in prompt
    assert '"best_working_solutions"' in prompt
    assert '"local_cv_score"' in prompt
    assert '"code"' in prompt
    assert '"metric":' not in prompt
    assert '"run_id"' not in prompt
    assert '"checkpoint_step"' not in prompt
    assert '"created_at"' not in prompt
    assert '"step"' not in prompt
    assert '"stage"' not in prompt
    assert '"plan"' not in prompt
    assert '"analysis"' not in prompt


def test_synthesis_prompt_switches_to_preprocess_contract_in_autogluon_mode(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.mode = AGENT_MODE
    journal = Journal()
    journal.append(
        _node(
            0.9,
            code=build_autogluon_wrapper(
                "def preprocess(df):\n"
                "    df = df.copy()\n"
                "    df['x'] = 1\n"
                "    return df\n",
                cfg,
            ),
        )
    )

    context = collect_synthesis_context(
        cfg=cfg,
        task_desc="task",
        journal=journal,
        completed_steps=15,
    )
    prompt = build_synthesis_prompt(context)

    assert prompt.startswith(SYNTHESIS_PREPROCESS_PROMPT_INTRO)
    assert not prompt.startswith(SYNTHESIS_PROMPT_INTRO)
    assert "Internal synthesis procedure" in prompt
    assert "Reject trivial synthesis" in prompt
    assert "kaggle_public_score as the strongest generalization signal" in prompt
    assert "Avoid feature bloat" in prompt
    assert "F1 driver will pit on the next lap" in prompt
    assert "def preprocess(df: pd.DataFrame)" in prompt
    assert "Return only Python code defining exactly one top-level function" in prompt
    assert "Do not include imports, helper functions, top-level constants" in prompt
    assert "`pd` is already available from the fixed wrapper" in prompt
    assert "Do not read files, write files, train models" in prompt
    assert "solution scripts" not in prompt
    assert "live web search" not in prompt
    assert "TabularPredictor" not in prompt
    assert '"code": "def preprocess' in prompt


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
        "best_working_solutions": [{"local_cv_score": 0.9, "code": "print('old')"}],
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


def test_run_synthesis_checkpoint_rejects_generated_next_pitstop_leakage(tmp_path):
    cfg = _cfg(tmp_path)
    context = {
        "run_id": cfg.exp_name,
        "checkpoint_step": 15,
        "task_desc": "task",
        "best_working_solutions": [{"local_cv_score": 0.9, "code": "print('old')"}],
    }

    def fake_runner(cmd, **kwargs):
        checkpoint_dir = Path(cmd[cmd.index("--cd") + 1])
        (checkpoint_dir / "response_raw.txt").write_text(
            "import pandas as pd\n"
            "test = pd.read_csv('./input/test.csv.gz')\n"
            "test['next_PitStop'] = test.groupby(['Race', 'Driver'])['PitStop'].shift(-1)\n"
            "print(test['next_PitStop'].mean())\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    result = run_synthesis_checkpoint(
        cfg=cfg,
        context=context,
        runner=fake_runner,
    )

    checkpoint_dir = Path(result["checkpoint_dir"])
    response = json.loads((checkpoint_dir / "response.json").read_text())
    status = json.loads((checkpoint_dir / "status.json").read_text())

    assert result["status"] == "failed"
    assert status["status"] == "failed"
    assert "target leakage" in response["error"]
    assert not (checkpoint_dir / "response.py").exists()


def test_run_synthesis_checkpoint_wraps_preprocess_in_autogluon_mode(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.mode = AGENT_MODE
    context = {
        "run_id": cfg.exp_name,
        "checkpoint_step": 15,
        "agent_mode": AGENT_MODE,
        "task_desc": "task",
        "best_working_solutions": [
            {
                "local_cv_score": 0.9,
                "code": "def preprocess(df):\n    return df\n",
            }
        ],
    }

    def fake_runner(cmd, **kwargs):
        checkpoint_dir = Path(cmd[cmd.index("--cd") + 1])
        (checkpoint_dir / "response_raw.txt").write_text(
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['feature'] = 1\n"
            "    return df\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    result = run_synthesis_checkpoint(
        cfg=cfg,
        context=context,
        runner=fake_runner,
    )

    checkpoint_dir = Path(result["checkpoint_dir"])
    response_code = (checkpoint_dir / "response.py").read_text()
    response = json.loads((checkpoint_dir / "response.json").read_text())

    assert result["status"] == "ready"
    assert "TabularPredictor" in response_code
    assert "def preprocess(df):" in response_code
    assert "feature" in response_code
    assert response["code"] == response_code


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
    assert synthesized.node.plan == f"{SYNTHESIS_PLAN_PREFIX} 000002"
    assert synthesized.node.code == "value = 2\nprint(value)\n"
    assert len(calls) == 1

    journal.append(synthesized.node)
    advisor.mark_injected(synthesized, node=synthesized.node)

    assert advisor.generate_node_if_due(journal=journal, completed_steps=2) is None
    assert len(calls) == 1
    assert "Synthesis: ✓ 000002" in advisor.status_text()


def test_synthesis_advisor_injects_existing_ready_checkpoint_even_after_count_moves_on(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')"))
    checkpoint = Path(cfg.log_dir) / "synthesis" / "checkpoint-000010"
    checkpoint.mkdir(parents=True)
    (checkpoint / "status.json").write_text('{"status": "ready"}')
    (checkpoint / "response.py").write_text("value = 10\nprint(value)\n")
    advisor = SynthesisAdvisor(cfg=cfg, task_desc="task", runner=lambda *_a, **_k: None)

    synthesized = advisor.generate_node_if_due(journal=journal, completed_steps=23)

    assert synthesized is not None
    assert synthesized.completed_steps == 10
    assert synthesized.checkpoint_dir == checkpoint
    assert synthesized.node.parent is None
    assert synthesized.node.plan == f"{SYNTHESIS_PLAN_PREFIX} 000010"
    assert synthesized.node.code == "value = 10\nprint(value)\n"


def test_synthesis_advisor_rejects_existing_ready_checkpoint_with_leakage(tmp_path):
    cfg = _cfg(tmp_path)
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')"))
    checkpoint = Path(cfg.log_dir) / "synthesis" / "checkpoint-000010"
    checkpoint.mkdir(parents=True)
    (checkpoint / "status.json").write_text('{"status": "ready"}')
    (checkpoint / "response.py").write_text(
        "test['next_PitStop'] = test.groupby(['Race'])['PitStop'].shift(-1)\n"
    )
    advisor = SynthesisAdvisor(cfg=cfg, task_desc="task", runner=lambda *_a, **_k: None)

    synthesized = advisor.generate_node_if_due(journal=journal, completed_steps=23)
    status = json.loads((checkpoint / "status.json").read_text())

    assert synthesized is None
    assert status["status"] == "failed"
    assert "target leakage" in status["error"]
