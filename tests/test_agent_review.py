import json
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
    assert set(review_func_spec.json_schema["required"]) == set(
        review_func_spec.json_schema["properties"]
    )
    assert review_func_spec.json_schema["properties"]["validity_warning"]["type"] == [
        "string",
        "null",
    ]
    assert review_func_spec.json_schema["properties"][
        "research_hypotheses_llm_claimed_used"
    ]["items"]["type"] == "string"
    assert review_func_spec.json_schema["properties"]["research_usage_note"][
        "type"
    ] == [
        "string",
        "null",
    ]
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
    node.validity_warning = "Possible leakage in grouped features."
    node.metric = MetricValue(0.9504787907447247, maximize=True)
    journal.append(node)

    summary = journal.generate_summary()

    assert "Results: AutoGluon preprocess wrapper completed." not in summary
    assert "Validation Metric: 0.95048" in summary
    assert "Validity warning: Possible leakage in grouped features." in summary
    assert "0.9504787907447247" not in summary
    assert "presets=medium_quality" not in summary


def test_journal_summary_compacts_entries_older_than_recent_step_window():
    journal = Journal()

    for step in range(101):
        node = Node(code=f"print({step})", plan=f"plan {step}")
        node.analysis = f"results {step}"
        node.validity_warning = f"warning {step}"
        node.metric = MetricValue(0.9 + step / 1000, maximize=True)
        node.is_buggy = False
        journal.append(node)

    summary = journal.generate_summary(recent_steps=100, full_recent_steps=20)
    sections = summary.split("\n-------------------------------\n")
    compact_section = next(section for section in sections if "Design: plan 1" in section)
    compact_last_section = next(
        section for section in sections if "Design: plan 80" in section
    )
    full_first_section = next(
        section for section in sections if "Design: plan 81" in section
    )
    full_last_section = next(
        section for section in sections if "Design: plan 100" in section
    )

    assert "Design: plan 0" not in summary
    assert "Results: results 1" not in compact_section
    assert "Validity warning: warning 1" not in compact_section
    assert "Validation Metric: 0.90100" in compact_section
    assert "Results: results 80" not in compact_last_section
    assert "Validity warning: warning 80" not in compact_last_section
    assert "Results: results 81" in full_first_section
    assert "Validity warning: warning 81" in full_first_section
    assert "Results: results 100" in full_last_section
    assert "Validity warning: warning 100" in full_last_section


def test_journal_summary_hides_technical_synthesis_checkpoint_ids():
    journal = Journal()
    node = Node(code="print('ok')", plan="External Codex synthesis checkpoint 000027")
    node.analysis = ""
    node.metric = MetricValue(0.950327, maximize=True)
    journal.append(node)

    summary = journal.generate_summary()

    assert "Design: External Codex synthesis generated a new root solution" in summary
    assert "checkpoint 000027" not in summary


def test_legacy_agent_memory_uses_configured_recent_and_full_windows(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.data_preview = False
    cfg.agent.memory_recent_steps = 2
    cfg.agent.memory_full_recent_steps = 1
    journal = Journal()
    for step in range(3):
        node = Node(code=f"print({step})", plan=f"plan {step}")
        node.analysis = f"results {step}"
        node.metric = MetricValue(0.9 + step / 1000, maximize=True)
        node.is_buggy = False
        journal.append(node)

    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    memory = captured["prompt"]["Memory"]
    assert "Design: plan 0" not in memory
    assert "Design: plan 1" in memory
    assert "Results: results 1" not in memory
    assert "Design: plan 2" in memory
    assert "Results: results 2" in memory


def test_agent_data_preview_excludes_workspace_working_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("id,x,y\n1,2,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,x\n2,3\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,y\n2,0\n", encoding="utf-8")

    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(data_dir)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "review-test"
    cfg = prep_cfg(cfg)
    metadata_dir = Path(cfg.workspace_dir) / "working" / "autogluon_model"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "metadata.json").write_text(
        '{"packages": {"nvidia-cudnn-cu13": "9.0"}}', encoding="utf-8"
    )
    (metadata_dir / "version.txt").write_text("1.5.0\n", encoding="utf-8")

    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    agent.update_data_preview()

    assert agent.data_preview is not None
    assert "train.csv" in agent.data_preview
    assert not agent.data_preview.startswith("```")
    assert "working/" not in agent.data_preview


def test_journal_generates_branch_context_from_root_to_parent_only():
    journal = Journal()
    root = Node(code="root", plan="Root hypothesis plan")
    root.metric = MetricValue(0.91, maximize=True)
    root.is_buggy = False
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000101"]
    journal.append(root)

    child = Node(code="child", plan="Child hypothesis plan", parent=root)
    child.metric = MetricValue(0.92, maximize=True)
    child.is_buggy = False
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["000202"]
    journal.append(child)

    unrelated = Node(code="other", plan="Unrelated root plan")
    unrelated.metric = MetricValue(0.99, maximize=True)
    unrelated.is_buggy = False
    unrelated.research_mode = "hypothesis"
    unrelated.research_hypotheses_offered = ["000999"]
    journal.append(unrelated)

    grandchild = Node(code="grandchild", plan="Grandchild plan", parent=child)
    grandchild.metric = MetricValue(0.93, maximize=True)
    grandchild.is_buggy = False
    grandchild.research_mode = "hypothesis"
    grandchild.research_hypotheses_offered = ["000303"]

    context = journal.generate_branch_context(child)

    assert "ancestor nodes of this parent, ordered from root to direct parent" in context
    assert "Branch path:\n000101 -> 000202" in context
    assert "Ancestor 1 / root:" in context
    assert "Hypothesis ID: 000101" in context
    assert "Design: Root hypothesis plan" in context
    assert "Validation Metric: 0.91000" in context
    assert "Ancestor 2 / direct parent:" in context
    assert "Hypothesis ID: 000202" in context
    assert "Design: Child hypothesis plan" in context
    assert "Validation Metric: 0.92000" in context
    assert "000999" not in context
    assert "Unrelated root plan" not in context
    assert "000303" not in context
    assert "Grandchild plan" not in context


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


def test_review_node_clears_generated_status_after_execution(tmp_path):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan", status="generated")
    exec_result = ExecutionResult(
        term_out=[
            'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
            '"metric": 0.9, "lower_is_better": false}\n'
        ],
        exec_time=1.0,
        exc_type=None,
    )

    agent.review_node(node, exec_result)

    assert node.status == "ok"
    assert node.is_buggy is False
    assert node.metric.value == 0.9


def test_parse_exec_result_keeps_metric_when_review_reports_validity_warning(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    exec_result = ExecutionResult(
        term_out=["OOF blended ROC AUC: 0.949967\nSaved submission\n"],
        exec_time=1.0,
        exc_type=None,
    )
    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: {
            "is_bug": True,
            "summary": "Run completed, but same-lap aggregates may leak.",
            "metric": 0.949967,
            "lower_is_better": False,
            "validity_warning": "Possible leakage from same-lap aggregate features.",
        },
    )

    agent.parse_exec_result(node, exec_result)

    assert node.is_buggy is False
    assert node.metric.value == 0.949967
    assert node.metric.maximize is True
    assert node.validity_warning == "Possible leakage from same-lap aggregate features."


def test_parse_exec_result_records_manual_research_claimed_usage(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "manual"
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    node.research_mode = "manual"
    node.research_hypotheses_offered = ["000001", "000002"]
    node.research_source_hash = "sha256:test"
    exec_result = ExecutionResult(
        term_out=["CV AUC: 0.91\n"],
        exec_time=1.0,
        exc_type=None,
    )
    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: {
            "is_bug": False,
            "summary": "Used grouped validation research.",
            "metric": 0.91,
            "lower_is_better": False,
            "validity_warning": None,
            "research_hypotheses_llm_claimed_used": ["000001"],
            "research_usage_note": "The review claims 000001 influenced validation.",
        },
    )

    agent.parse_exec_result(node, exec_result)

    assert node.research_hypotheses_llm_claimed_used == ["000001"]
    assert node.research_usage_note == "The review claims 000001 influenced validation."
    usage = json.loads((Path(cfg.log_dir) / "research_hypotheses" / "usage.json").read_text())
    assert usage["000001"]["llm_claimed_used_count"] == 1
    assert node.id in usage["000001"]["llm_claimed_used_node_ids"]


def test_parse_result_marker_records_manual_research_claim_from_plan(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "manual"
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(
        code="print('ok')",
        plan="I intentionally use manual research hypothesis 000002.",
    )
    node.research_mode = "manual"
    node.research_hypotheses_offered = ["000001", "000002"]
    node.research_source_hash = "sha256:test"
    exec_result = ExecutionResult(
        term_out=[
            'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
            '"metric": 0.92, "lower_is_better": false}\n'
        ],
        exec_time=1.0,
        exc_type=None,
    )

    agent.parse_exec_result(node, exec_result)

    assert node.research_hypotheses_llm_claimed_used == ["000002"]
    assert node.research_usage_note == (
        "Plan text mentioned offered manual research hypothesis id(s): 000002."
    )


def test_parse_exec_result_accepts_hypothesis_node_when_claim_missing(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000123"]
    exec_result = ExecutionResult(
        term_out=["CV AUC: 0.91\n"],
        exec_time=1.0,
        exc_type=None,
    )
    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: {
            "is_bug": False,
            "summary": "Ran a valid metric but omitted the hypothesis id.",
            "metric": 0.91,
            "lower_is_better": False,
            "validity_warning": None,
            "research_hypotheses_llm_claimed_used": [],
            "research_usage_note": None,
        },
    )

    agent.parse_exec_result(node, exec_result)

    assert node.status is None
    assert node.is_buggy is False
    assert node.metric.value == 0.91
    assert node.research_hypotheses_llm_claimed_used == []
    assert "expected hypothesis id 000123" not in node.analysis


def test_parse_exec_result_keeps_hypothesis_technical_failure_debuggable_without_claim(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="raise ValueError('bad')", plan="plan")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000123"]
    exec_result = ExecutionResult(
        term_out=["ValueError: bad\n"],
        exec_time=1.0,
        exc_type="ValueError",
    )
    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: {
            "is_bug": True,
            "summary": "Technical preprocessing failure before metric.",
            "metric": None,
            "lower_is_better": False,
            "validity_warning": None,
            "research_hypotheses_llm_claimed_used": [],
            "research_usage_note": None,
        },
    )

    agent.parse_exec_result(node, exec_result)

    assert node.status is None
    assert node.is_terminal_failure is False
    assert node.is_buggy is True
    assert node.metric.is_worst
    assert node.research_hypotheses_llm_claimed_used == []
    assert "Keeping this node as a debuggable bug" not in node.analysis
    assert "Technical preprocessing failure" in node.analysis


def test_parse_exec_result_keeps_hypothesis_preprocess_timeout_as_bug(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('timeout')", plan="plan")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000123"]
    exec_result = ExecutionResult(
        term_out=[
            "PreprocessTimeoutError: AIDE AutoGluon preprocess exceeded the dedicated timeout\n"
        ],
        exec_time=180.0,
        exc_type="PreprocessTimeoutError",
    )
    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: {
            "is_bug": True,
            "summary": "Preprocess exceeded the dedicated timeout and must be simplified.",
            "metric": None,
            "lower_is_better": False,
            "validity_warning": None,
            "research_hypotheses_llm_claimed_used": ["000123"],
            "research_usage_note": "Implemented 000123 but preprocessing timed out.",
        },
    )

    agent.parse_exec_result(node, exec_result)

    assert node.status is None
    assert node.is_terminal_failure is False
    assert node.is_buggy is True
    assert node.metric.is_worst
    assert node.research_hypotheses_llm_claimed_used == []
    assert "must be simplified" in node.analysis


def test_parse_exec_result_ignores_hypothesis_claim(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000123"]
    exec_result = ExecutionResult(
        term_out=["CV AUC: 0.91\n"],
        exec_time=1.0,
        exc_type=None,
    )
    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: {
            "is_bug": False,
            "summary": "Verified the assigned hypothesis.",
            "metric": 0.91,
            "lower_is_better": False,
            "validity_warning": None,
            "research_hypotheses_llm_claimed_used": ["000123"],
            "research_usage_note": "Implemented 000123.",
        },
    )

    agent.parse_exec_result(node, exec_result)

    assert node.status is None
    assert node.is_buggy is False
    assert node.metric.value == 0.91
    assert node.research_hypotheses_llm_claimed_used == []
    usage_path = Path(cfg.log_dir) / "research_hypotheses" / "usage.json"
    assert not usage_path.exists()


def test_parse_result_marker_accepts_missing_hypothesis_claim(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000123"]
    exec_result = ExecutionResult(
        term_out=[
            'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
            '"metric": 0.92, "lower_is_better": false}\n'
        ],
        exec_time=1.0,
        exc_type=None,
    )

    agent.parse_exec_result(node, exec_result)

    assert node.status is None
    assert node.is_terminal_failure is False
    assert node.is_buggy is False
    assert node.metric.value == 0.92
