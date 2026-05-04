from pathlib import Path

import pytest

from aide.agent import Agent
from aide.autogluon_preprocess import (
    AGENT_MODE,
    build_autogluon_wrapper,
    extract_preprocess_source,
    parse_result_marker,
    validate_preprocess_source,
)
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
    cfg.exp_name = "ag-preprocess-test"
    cfg.agent.mode = AGENT_MODE
    cfg.agent.search.num_drafts = 0
    cfg = prep_cfg(cfg)
    input_dir = Path(cfg.workspace_dir) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "sample_submission.csv").write_text("id,PitNextLap\n10,0.0\n")
    return cfg


def test_extract_preprocess_source_from_markdown_code_block():
    source = extract_preprocess_source(
        "plan\n```python\n"
        "def preprocess(df):\n"
        "    df = df.copy()\n"
        "    return df\n"
        "```"
    )

    assert source.startswith("def preprocess(df):")
    assert "return df" in source


def test_validate_preprocess_source_rejects_target_reference():
    with pytest.raises(ValueError, match="forbidden column"):
        validate_preprocess_source(
            "def preprocess(df):\n"
            "    df['target_copy'] = df['PitNextLap']\n"
            "    return df\n",
            target_col="PitNextLap",
        )


def test_validate_preprocess_source_rejects_split_marker_reference():
    with pytest.raises(ValueError, match="__is_train__"):
        validate_preprocess_source(
            "def preprocess(df):\n"
            "    df['split_feature'] = df['__is_train__'].astype(int)\n"
            "    return df\n",
            target_col="PitNextLap",
        )


def test_validate_preprocess_source_rejects_row_id_reference():
    with pytest.raises(ValueError, match="__aide_row_id__"):
        validate_preprocess_source(
            "def preprocess(df):\n"
            "    df['row_feature'] = df['__aide_row_id__']\n"
            "    return df\n",
            target_col="PitNextLap",
        )


def test_build_autogluon_wrapper_compiles_and_preserves_preprocess(tmp_path):
    cfg = _cfg(tmp_path)

    code = build_autogluon_wrapper(
        "def preprocess(df):\n"
        "    df = df.copy()\n"
        "    return df\n",
        cfg,
    )

    compile(code, "<generated_autogluon_wrapper>", "exec")
    assert "TabularPredictor" in code
    assert "def preprocess(df):" in code
    assert "AIDE_RESULT_JSON:" in code
    assert "'time_limit': 600" in code
    assert "train_features = train_df.drop(columns=[target_col, id_col]" in code
    assert "_make_combined_frame(train_features, test_features)" in code
    assert "df[HELPER_ROW_ID]" not in code
    assert "FORBIDDEN_ROW_ID in after.columns" in code
    assert "verbosity=2" in code
    assert 'os.environ.get("AIDE_NODE_ARTIFACT_DIR"' in code
    assert "class _AutoFlushWriter" in code
    assert 'print("AIDE AutoGluon: starting fit", flush=True)' in code
    assert 'if __name__ == "__main__"' not in code
    assert code.rstrip().endswith("main()")


def test_agent_autogluon_draft_wraps_preprocess_response(tmp_path):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    captured = {}

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return (
            "add a simple numeric ratio",
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['TyreLife_x2'] = df.get('TyreLife', 0) * 2\n"
            "    return df\n",
        )

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent._draft()

    assert "AutoGluon preprocess mode contract" in captured["prompt"]["Instructions"]
    assert "TabularPredictor" in node.code
    assert "TyreLife_x2" in node.code


def test_agent_autogluon_improve_prompt_uses_previous_preprocess(tmp_path):
    cfg = _cfg(tmp_path)
    parent = Node(
        plan="base",
        code=build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['base_feature'] = 1\n"
            "    return df\n",
            cfg,
        ),
    )
    parent.metric = MetricValue(0.9, maximize=True)
    parent.is_buggy = False
    journal = Journal()
    journal.append(parent)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    captured = {}

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return (
            "improve feature",
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['base_feature'] = 2\n"
            "    return df\n",
        )

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent._improve(parent)

    assert "base_feature" in captured["prompt"]["Previous preprocess function"]
    assert node.parent is parent
    assert "base_feature" in node.code


def test_parse_result_marker_short_circuits_feedback_review(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('ok')", plan="plan")
    exec_result = ExecutionResult(
        term_out=[
            'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ag ok", '
            '"metric": 0.91, "lower_is_better": false}\n'
        ],
        exec_time=1.0,
        exc_type=None,
    )

    monkeypatch.setattr(
        "aide.agent.query",
        lambda **_kwargs: pytest.fail("feedback LLM should not be called"),
    )

    agent.parse_exec_result(node, exec_result)

    assert node.is_buggy is False
    assert node.metric.value == 0.91
    assert node.analysis == "ag ok"


def test_parse_result_marker_uses_latest_valid_marker():
    parsed = parse_result_marker(
        'AIDE_RESULT_JSON: {"metric": 0.1}\n'
        'AIDE_RESULT_JSON: {"metric": 0.2, "summary": "latest"}\n'
    )

    assert parsed == {"metric": 0.2, "summary": "latest"}
