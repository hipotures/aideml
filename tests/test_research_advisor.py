import datetime as dt
import json
import subprocess
from pathlib import Path

from aide.agent import Agent
from aide.autogluon_preprocess import AGENT_MODE, build_autogluon_wrapper
from aide.journal import Journal, Node
from aide.research import (
    RESEARCH_PROMPT_INTRO,
    ResearchAdvisor,
    build_data_overview,
    build_research_prompt,
    collect_previous_research_summaries,
    collect_research_context,
    count_scored_working_nodes,
    format_research_hints_for_prompt,
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


def _write_research_checkpoint(
    cfg,
    step: int,
    *,
    summary: str,
    status: str = "completed",
) -> Path:
    checkpoint = Path(cfg.log_dir) / "research" / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "status.json").write_text(json.dumps({"status": status}))
    (checkpoint / "response.json").write_text(
        json.dumps(
            {
                "parsed_response": {
                    "summary": summary,
                    "hypotheses": [],
                }
            }
        )
    )
    return checkpoint


def test_build_data_overview_includes_compact_column_schema(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "train.csv").write_text(
        "id,Driver,Race,TyreLife,PitNextLap\n"
        "1,HAM,Bahrain,12,0\n"
        "2,VER,Bahrain,24,1\n",
        encoding="utf-8",
    )
    (input_dir / "test.csv").write_text(
        "id,Driver,Race,TyreLife\n" "3,HAM,Bahrain,13\n",
        encoding="utf-8",
    )
    (input_dir / "sample_submission.csv").write_text(
        "id,PitNextLap\n3,0\n",
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path)

    overview = build_data_overview(cfg)

    assert "train.csv (3 lines)" in overview
    assert "test.csv (2 lines)" in overview
    assert "sample_submission.csv (2 lines)" in overview
    assert "-> input/train.csv has 2 rows and 5 columns." in overview
    assert "Driver (object) has 2 unique values" in overview
    assert "TyreLife (int64)" in overview


def test_build_data_overview_prefers_data_dir_over_workspace_working(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("id,x,y\n1,2,0\n", encoding="utf-8")
    (data_dir / "test.csv").write_text("id,x\n2,3\n", encoding="utf-8")
    (data_dir / "sample_submission.csv").write_text("id,y\n2,0\n", encoding="utf-8")

    workspace_dir = tmp_path / "workspace"
    metadata_dir = workspace_dir / "working" / "autogluon_model"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "metadata.json").write_text('{"packages": {"noise": "1"}}')

    cfg = _cfg(tmp_path)
    cfg.data_dir = str(data_dir)
    cfg.workspace_dir = str(workspace_dir)

    overview = build_data_overview(cfg)

    assert "train.csv" in overview
    assert "working/" not in overview
    assert "metadata.json" not in overview
    assert "packages" not in overview


def test_collect_research_context_selects_top_best_and_worst_scored_nodes(tmp_path):
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
    context["data_overview"] = {"columns": ["feature"]}

    assert [n["local_cv_score"] for n in context["best_working_solutions"][:2]] == [
        0.95,
        0.81,
    ]
    assert [n["local_cv_score"] for n in context["worst_working_solutions"][:2]] == [
        0.1
    ]
    assert all(
        n["local_cv_score"] is not None for n in context["worst_working_solutions"]
    )
    assert context["run_id"] == cfg.exp_name
    assert context["checkpoint_step"] == 10
    assert "created_at" in context
    assert "selected_steps" not in context
    assert "selected_node_ids" not in context
    assert "recent_nodes" not in context

    serialized = json.dumps(context)
    assert '"step"' not in serialized
    assert "stage" not in serialized
    assert "plan" not in serialized
    assert "analysis" not in serialized
    assert journal.nodes[0].id not in serialized
    assert journal.nodes[1].id not in serialized
    assert journal.nodes[2].id not in serialized
    assert journal.nodes[3].id not in serialized
    assert "parent_id" not in serialized
    assert "ctime" not in serialized
    assert "is_buggy" not in serialized
    assert "terminal_output" not in serialized
    assert "exec_time" not in serialized
    assert "exc_type" not in serialized
    assert "exc_info" not in serialized


def test_collect_research_context_uses_preprocess_only_in_autogluon_mode(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.mode = AGENT_MODE
    journal = Journal()
    journal.append(
        _node(
            0.95,
            code=build_autogluon_wrapper(
                "def preprocess(df: pd.DataFrame) -> pd.DataFrame:\n"
                "    df = df.copy()\n"
                "    df['x2'] = df['x'] * 2\n"
                "    return df\n",
                cfg,
            ),
            plan="best",
        )
    )

    context = collect_research_context(
        cfg=cfg,
        task_desc="task",
        journal=journal,
        completed_steps=10,
    )

    payload = context["best_working_solutions"][0]
    assert payload["local_cv_score"] == 0.95
    assert payload["code"].startswith("def preprocess")
    assert "TabularPredictor" not in payload["code"]
    assert "AIDE_AG_CONFIG" not in payload["code"]
    assert '"code"' in json.dumps(context)


def test_count_scored_working_nodes_ignores_buggy_nodes(tmp_path):
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')", plan="ok"))
    journal.append(_node(None, code="raise RuntimeError('bug')", plan="bug"))
    journal.append(_node(0.8, code="print('ok2')", plan="ok2"))

    assert count_scored_working_nodes(journal) == 2


def test_collect_previous_research_summaries_includes_scores_after_each_checkpoint(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    cfg.research.previous_summary_count = 2
    journal = Journal()
    for idx, score in enumerate(
        [
            0.70,
            0.71,
            0.72,
            0.73,
            0.74,
            0.81,
            0.84,
            0.82,
            0.83,
            0.80,
            0.91,
            0.90,
        ],
        start=1,
    ):
        journal.append(_node(score, code=f"print({idx})", plan=f"node {idx}"))
    _write_research_checkpoint(cfg, 5, summary="older research summary")
    _write_research_checkpoint(cfg, 10, summary="newer research summary")
    public_node = journal.nodes[10]
    (Path(cfg.log_dir).parent / "submission_registry.json").write_text(
        json.dumps(
            {
                "submissions": [
                    {
                        "run": cfg.exp_name,
                        "step": public_node.step,
                        "timestamp": dt.datetime.fromtimestamp(
                            public_node.ctime
                        ).strftime("%Y%m%dT%H%M%S"),
                        "node_id": public_node.id,
                        "remote_status": "COMPLETE",
                        "public_score": "0.70124",
                    }
                ]
            }
        )
    )

    summaries = collect_previous_research_summaries(
        cfg=cfg,
        journal=journal,
        completed_steps=12,
    )

    assert summaries == [
        {
            "checkpoint": "checkpoint-000010",
            "summary": "newer research summary",
            "max_local_cv_score_after": 0.91,
            "max_kaggle_public_score_after": 0.70124,
        },
        {
            "checkpoint": "checkpoint-000005",
            "summary": "older research summary",
            "max_local_cv_score_after": 0.84,
            "max_kaggle_public_score_after": None,
        },
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
    assert '"data_overview"' in prompt
    assert '"run_id"' not in prompt
    assert '"checkpoint_step"' not in prompt
    assert '"created_at"' not in prompt
    assert '"step"' not in prompt
    assert '"stage"' not in prompt
    assert '"additionalProperties"' not in prompt
    assert '"parent_node_id"' not in prompt
    assert '"parent_step"' not in prompt
    assert "hypotheses[].target" not in prompt
    assert "Return exactly 5 concise new solution ideas" in prompt
    assert "Do not target a specific previous node or code block" in prompt


def test_research_prompt_includes_previous_research_summaries(tmp_path):
    context = {
        "task_desc": "task",
        "best_working_solutions": [],
        "worst_working_solutions": [],
        "previous_research_summaries": [
            {
                "checkpoint": "checkpoint-000010",
                "summary": "Try pit-window features",
                "max_local_cv_score_after": 0.91,
                "max_kaggle_public_score_after": 0.70124,
            }
        ],
    }

    prompt = build_research_prompt(context)

    assert '"previous_research_summaries"' in prompt
    assert '"label"' not in prompt
    assert "Try pit-window features" in prompt
    assert "unique relative to those earlier summaries" in prompt


def test_run_research_checkpoint_logs_request_and_response(tmp_path):
    cfg = _cfg(tmp_path)
    context = {
        "run_id": cfg.exp_name,
        "checkpoint_step": 10,
        "task_desc": "task",
        "best_working_solutions": [],
        "worst_working_solutions": [],
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

    assert command[:6] == [
        "codex",
        "--search",
        "--ask-for-approval",
        "never",
        "exec",
        "--ignore-user-config",
    ]
    assert "--ignore-user-config" in command
    assert "--search" in command
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
    assert response["raw_response"].startswith('{"summary":')
    readable_response = (checkpoint_dir / "response_raw.txt").read_text()
    assert readable_response.startswith(
        "Use these external Codex research hints only when relevant."
    )
    assert "Research checkpoint: 000010" in readable_response
    assert "Summary: researched" in readable_response
    assert "Try calibrated LightGBM" in readable_response
    assert '{"summary":' not in readable_response
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


def test_research_advisor_uses_scored_working_count_for_checkpoints(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.research.every_steps = 2
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')", plan="ok"))
    journal.append(_node(None, code="raise RuntimeError('bug')", plan="bug"))
    advisor = ResearchAdvisor(cfg=cfg, task_desc="task", runner=lambda *_a, **_k: None)

    assert (
        advisor.maybe_start(
            journal=journal,
            completed_steps=count_scored_working_nodes(journal),
        )
        is False
    )

    journal.append(_node(0.8, code="print('ok2')", plan="ok2"))

    assert (
        advisor.maybe_start(
            journal=journal,
            completed_steps=count_scored_working_nodes(journal),
        )
        is True
    )
    assert (Path(cfg.log_dir) / "research" / "checkpoint-000002").exists()


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


def test_format_research_hints_for_prompt_renders_concise_human_hints():
    rendered = format_research_hints_for_prompt(
        {
            "checkpoint": "checkpoint-000010",
            "summary": "research summary",
            "hypotheses": [
                {
                    "target": "node",
                    "parent_node_id": "dfe8126b1b4c46d68446bcb513e51d10",
                    "title": "Use tire-age feature",
                    "rationale": "Tyre age matters.",
                    "implementation_hint": "Add TyreLife rolling features.",
                    "expected_effect": "Better pit-window ranking.",
                    "risk": "May overfit.",
                    "sources": ["https://example.com/source"],
                }
            ],
        }
    )

    assert "Research checkpoint: 000010" in rendered
    assert "Summary: research summary" in rendered
    assert "Use tire-age feature" in rendered
    assert "Try: Add TyreLife rolling features." in rendered
    assert "parent_node_id" not in rendered
    assert "dfe8126b1b4c46d68446bcb513e51d10" not in rendered
    assert "https://example.com/source" not in rendered
    assert "```json" not in rendered


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

    assert "research summary" in captured["prompt"]["External research hints"]
    assert "Use tire-age feature" in captured["prompt"]["External research hints"]


def test_agent_includes_latest_research_hints_in_debug_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.data_preview = False
    checkpoint = Path(cfg.log_dir) / "research" / "checkpoint-000010"
    checkpoint.mkdir(parents=True)
    (checkpoint / "status.json").write_text('{"status": "completed"}')
    (checkpoint / "response.json").write_text(
        json.dumps(
            {
                "parsed_response": {
                    "summary": "debug research summary",
                    "hypotheses": [{"title": "Fix tire-age leakage"}],
                }
            }
        )
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    parent = _node(None, code="raise RuntimeError('bug')", plan="bug")

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._debug(parent)

    assert "debug research summary" in captured["prompt"]["External research hints"]
    assert "Fix tire-age leakage" in captured["prompt"]["External research hints"]


def test_agent_includes_serialized_research_hints_in_improve_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.data_preview = False
    checkpoint = Path(cfg.log_dir) / "research" / "checkpoint-000010"
    checkpoint.mkdir(parents=True)
    (checkpoint / "status.json").write_text('{"status": "completed"}')
    (checkpoint / "response.json").write_text(
        json.dumps(
            {
                "parsed_response": {
                    "summary": "improve research summary",
                    "hypotheses": [{"title": "Add race-driver sequential features"}],
                }
            }
        )
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    parent = _node(0.94, code="print('baseline')", plan="baseline")

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(parent)

    hints = captured["prompt"]["External research hints"]
    assert isinstance(hints, str)
    assert "improve research summary" in hints
    assert "Add race-driver sequential features" in hints
