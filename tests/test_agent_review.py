from pathlib import Path

from aide.agent import Agent, review_func_spec
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.utils.config import _load_cfg, prep_cfg
from aide.utils.metric import MetricValue


def _cfg(tmp_path: Path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "review-test"
    return prep_cfg(cfg)


def test_review_schema_is_valid_for_codex_structured_output():
    assert review_func_spec.json_schema["additionalProperties"] is False
    assert review_func_spec.json_schema["properties"]["metric"]["type"] == [
        "number",
        "null",
    ]


def test_journal_summary_formats_validation_metric_to_five_decimals():
    journal = Journal()
    node = Node(code="print('ok')", plan="plan")
    node.analysis = (
        "AutoGluon preprocess wrapper completed with roc_auc=0.950479 "
        "using presets=medium_quality."
    )
    node.metric = MetricValue(0.9504787907447247, maximize=True)
    journal.append(node)

    summary = journal.generate_summary()

    assert "Results: AutoGluon preprocess wrapper completed." not in summary
    assert "Validation Metric: 0.95048" in summary
    assert "0.9504787907447247" not in summary
    assert "presets=medium_quality" not in summary


def test_journal_summary_hides_technical_synthesis_checkpoint_ids():
    journal = Journal()
    node = Node(code="print('ok')", plan="External Codex synthesis checkpoint 000027")
    node.analysis = ""
    node.metric = MetricValue(0.950327, maximize=True)
    journal.append(node)

    summary = journal.generate_summary()

    assert "Design: External Codex synthesis generated a new root solution" in summary
    assert "checkpoint 000027" not in summary


def test_parse_exec_result_marks_node_buggy_when_review_response_is_not_dict(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    exec_result = ExecutionResult(
        term_out=["CV AUC: 0.9\n"],
        exec_time=1.0,
        exc_type=None,
    )

    monkeypatch.setattr("aide.agent.query", lambda **_kwargs: "not json")

    agent.parse_exec_result(node, exec_result)

    assert node.is_buggy is True
    assert node.metric.is_worst
    assert "Invalid review response" in node.analysis


def test_parse_exec_result_accepts_json_string_review_response(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    exec_result = ExecutionResult(
        term_out=["CV AUC: 0.9\n"],
        exec_time=1.0,
        exc_type=None,
    )
    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: (
            '{"is_bug": false, "summary": "good run", '
            '"metric": 0.9, "lower_is_better": false}'
        ),
    )

    agent.parse_exec_result(node, exec_result)

    assert node.is_buggy is False
    assert node.metric.value == 0.9
    assert node.metric.maximize is True
    assert node.analysis == "good run"
