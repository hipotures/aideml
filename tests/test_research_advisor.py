import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import pytest

import aide.agent as agent_module
import aide.research as research
from aide.agent import Agent
from aide.autogluon_preprocess import AGENT_MODE, build_autogluon_wrapper
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.run import (
    ParallelRootJob,
    format_hypothesis_only_finish_message,
    generate_reserved_hypothesis_root,
)
from aide.research import (
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
    cfg.agent.gpu = False
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


def _write_manual_hypothesis(
    root: Path,
    task_slug: str,
    hypothesis_id: str,
    *,
    title: str = "Grouped validation",
    summary: str = "Use grouped validation to reduce public/CV mismatch.",
    rationale: str = "Race/year grouping should reduce public/CV mismatch.",
    implementation_hint: str = "Build Race_Year groups and compare grouped CV.",
    expected_effect: str = "Better validation stability.",
    risk: str = "Grouped CV may be pessimistic.",
    sources: list[str] | None = None,
    enabled: bool = True,
    agent_modes: list[str] | None = None,
) -> Path:
    path = (
        root
        / "research_hypotheses"
        / task_slug
        / hypothesis_id
        / f"hypothesis-{hypothesis_id}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "enabled": enabled,
                "agent_modes": (
                    agent_modes if agent_modes is not None else ["legacy", "autogluon"]
                ),
                "title": title,
                "summary": summary,
                "rationale": rationale,
                "implementation_hint": implementation_hint,
                "expected_effect": expected_effect,
                "risk": risk,
                **({"sources": sources} if sources is not None else {}),
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_code_manifest(
    root: Path,
    task_slug: str,
    hypothesis_id: str,
    *,
    agent_mode: str,
    file_name: str,
    score: float | None,
    buggy: bool = False,
) -> None:
    hypothesis_dir = root / "research_hypotheses" / task_slug / hypothesis_id
    hypothesis_dir.mkdir(parents=True, exist_ok=True)
    (hypothesis_dir / file_name).write_text("print('root code')\n", encoding="utf-8")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "versions": {
                    agent_mode: [
                        {
                            "file": file_name,
                            "buggy": buggy,
                            "node_id": f"node-{hypothesis_id}",
                            "score": score,
                            "created_at": "2026-05-23T00:00:00",
                        }
                    ]
                },
                "active": {agent_mode: file_name} if not buggy else {},
            }
        ),
        encoding="utf-8",
    )


def _generated_hypothesis_payload(
    *,
    title: str = "Wide tabular root",
    summary: str = "Build a broad tabular baseline with rich feature coverage.",
) -> dict[str, object]:
    return {
        "title": title,
        "summary": summary,
        "feature_family": "broad_tabular_feature_expansion",
        "feature_strategy": (
            "Build numerical transformations, categorical counts, missingness "
            "indicators, and fold-safe statistical features."
        ),
        "baseline_model_panel": (
            "Use a compact panel of simple tree-boosting, bagging, and linear "
            "baselines when compatible with the task."
        ),
        "model_panel_rationale": (
            "Boosting and bagging baselines expose whether the feature family has "
            "nonlinear or robust tree signal."
        ),
        "validation_strategy": "Use leak-free 5-fold CV appropriate for the task.",
        "materialization_hint": (
            "Materialize as staged load, feature build, per-fold model panel, "
            "diagnostics, and standard AIDE outputs."
        ),
        "expected_signal": (
            "At least one simple tree model should improve over raw-feature roots."
        ),
        "novelty_confidence": "high",
        "risk": "The broad feature set can overfit if validation is weak.",
        "sources": ["https://example.com/tabular-root"],
    }


def _manual_cfg(tmp_path: Path):
    cfg = _cfg(tmp_path)
    data_dir = tmp_path / "playground-series-s6e5"
    data_dir.mkdir(exist_ok=True)
    cfg.data_dir = data_dir
    return cfg


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


def test_manual_hypothesis_library_indexes_json_files_from_task_slug(tmp_path):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000002")
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        title="Compact option-richness block",
        summary="Prune alias-heavy features from the option-richness block.",
    )

    library = research.load_manual_hypothesis_library(cfg, repo_root=tmp_path)

    assert library.task_slug == "playground-series-s6e5"
    assert library.source_dir == (
        tmp_path / "research_hypotheses" / "playground-series-s6e5"
    )
    assert [hypothesis.id for hypothesis in library.hypotheses] == [
        "000001",
        "000002",
    ]
    assert library.hypotheses[0].title == "Compact option-richness block"
    assert library.hypotheses[0].summary == (
        "Prune alias-heavy features from the option-richness block."
    )
    assert library.hypotheses[0].enabled is True
    assert library.hypotheses[0].agent_modes == ["legacy", "autogluon"]
    assert library.hypotheses[0].rationale
    assert library.hypotheses[0].implementation_hint
    assert library.hypotheses[0].expected_effect
    assert library.hypotheses[0].risk
    assert library.hypotheses[0].sources == []
    assert library.source_hash.startswith("sha256:")


def test_manual_hypothesis_library_uses_project_name_env_for_default_input(
    tmp_path,
    monkeypatch,
):
    cfg = _manual_cfg(tmp_path)
    cfg.data_dir = tmp_path / "input"
    Path(cfg.data_dir).mkdir(exist_ok=True)
    monkeypatch.setenv("AIDE_PROJECT_NAME", "playground-series-s6e6")
    _write_manual_hypothesis(tmp_path, "playground-series-s6e6", "000001")

    library = research.load_manual_hypothesis_library(cfg, repo_root=tmp_path)

    assert library.task_slug == "playground-series-s6e6"
    assert library.source_dir == (
        tmp_path / "research_hypotheses" / "playground-series-s6e6"
    )


def test_manual_hypothesis_library_loads_optional_sources(tmp_path):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        sources=[
            "https://example.com/research-a",
            "https://example.com/research-b",
        ],
    )

    library = research.load_manual_hypothesis_library(cfg, repo_root=tmp_path)

    assert library.hypotheses[0].sources == [
        "https://example.com/research-a",
        "https://example.com/research-b",
    ]


def test_playground_manual_hypotheses_do_not_reference_run_specific_ids():
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = "aide/example_tasks/playground-series-s6e5"
    cfg.goal = "Predict next-lap pit stop probability"
    cfg = prep_cfg(cfg)

    library = research.load_manual_hypothesis_library(cfg)
    forbidden_patterns = [
        r"\bstep\s+\d+\b",
        r"\bnode\s+[0-9a-f]{8,}\b",
        r"\b[0-9a-f]{32}\b",
        r"\bCV\s+0\.\d+",
        r"\bpublic\s+0\.\d+",
    ]

    for hypothesis in library.hypotheses:
        text = "\n".join(
            [
                hypothesis.title,
                hypothesis.summary,
                hypothesis.rationale,
                hypothesis.implementation_hint,
                hypothesis.expected_effect,
                hypothesis.risk,
            ]
        )
        for pattern in forbidden_patterns:
            assert not re.search(pattern, text, re.IGNORECASE), (
                f"{hypothesis.id} contains run-specific reference matching {pattern}"
            )


def test_manual_hypothesis_library_rejects_missing_hypothesis_files(tmp_path):
    cfg = _manual_cfg(tmp_path)
    (tmp_path / "research_hypotheses" / "playground-series-s6e5").mkdir(
        parents=True
    )

    try:
        research.load_manual_hypothesis_library(cfg, repo_root=tmp_path)
    except ValueError as exc:
        assert "No manual research hypothesis files found" in str(exc)
    else:
        raise AssertionError("Expected missing hypothesis files to fail")


def test_manual_hypothesis_library_rejects_missing_required_fields(tmp_path):
    cfg = _manual_cfg(tmp_path)
    path = (
        tmp_path
        / "research_hypotheses"
        / "playground-series-s6e5"
        / "hypotheses"
        / "hypothesis-000001.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"title": "Only title"}), encoding="utf-8")

    try:
        research.load_manual_hypothesis_library(cfg, repo_root=tmp_path)
    except ValueError as exc:
        assert "missing required field" in str(exc)
        assert "summary" in str(exc)
    else:
        raise AssertionError("Expected missing required field to fail")


def test_select_manual_hypotheses_prefers_under_offered_and_records_usage(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.manual_sample_size = 2
    cfg.research.manual_seed = 123
    for idx in range(1, 5):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
            summary=f"Summary {idx}",
            implementation_hint=f"Implementation {idx}",
        )
    usage_dir = Path(cfg.log_dir) / "research_hypotheses"
    usage_dir.mkdir(parents=True)
    (usage_dir / "usage.json").write_text(
        json.dumps(
            {
                "000001": {"offered_count": 2},
                "000002": {"offered_count": 1},
            }
        ),
        encoding="utf-8",
    )

    selection = research.select_manual_hypotheses(
        cfg,
        completed_steps=10,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == [
        "000003",
        "000004",
    ]
    source_ref = json.loads((usage_dir / "source_ref.json").read_text())
    assert source_ref["indexed_hypothesis_count"] == 4
    assert source_ref["source_hash"].startswith("sha256:")
    offers = (usage_dir / "offers.jsonl").read_text().splitlines()
    assert len(offers) == 1
    offer = json.loads(offers[0])
    assert offer["checkpoint_step"] == 10
    assert offer["offered"] == ["000003", "000004"]
    updated_usage = json.loads((usage_dir / "usage.json").read_text())
    assert updated_usage["000003"]["offered_count"] == 1
    assert updated_usage["000004"]["offered_count"] == 1
    assert updated_usage["000003"]["offered_checkpoint_steps"] == [10]


def test_format_manual_research_hints_for_prompt_includes_usage_instruction(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.manual_sample_size = 1
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        title="Grouped validation",
        summary="Use grouped validation to reduce public/CV mismatch.",
        rationale="Random holdout can mix race context across splits.",
        implementation_hint=(
            "Build Race_Year groups and compare grouped CV against the current holdout."
        ),
        expected_effect="More reliable model selection.",
        risk="Grouped CV can be noisy on small groups.",
        sources=["https://example.com/grouped-validation"],
    )

    selection = research.select_manual_hypotheses(
        cfg,
        completed_steps=10,
        repo_root=tmp_path,
    )
    rendered = research.format_manual_research_hints_for_prompt(selection)

    assert "Manual research hypotheses offered" in rendered
    assert "If your solution intentionally uses any of them" in rendered
    assert "000001. Grouped validation" in rendered
    assert "Use grouped validation to reduce public/CV mismatch." in rendered
    assert "Why: Random holdout can mix race context" in rendered
    assert "Build Race_Year groups" in rendered
    assert "Expected effect: More reliable model selection." in rendered
    assert "Risk: Grouped CV can be noisy" in rendered
    assert "Sources:" not in rendered
    assert "https://example.com/grouped-validation" not in rendered


def test_select_manual_hypotheses_skips_disabled_hypotheses(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.manual_sample_size = 1
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        title="Disabled hypothesis",
        enabled=False,
    )
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000002",
        title="Enabled hypothesis",
        enabled=True,
    )

    selection = research.select_manual_hypotheses(
        cfg,
        completed_steps=10,
        repo_root=tmp_path,
    )

    assert selection.hypotheses[0].id in {"000002", "000003"}
    source_ref = json.loads(
        (Path(cfg.log_dir) / "research_hypotheses" / "source_ref.json").read_text()
    )
    assert source_ref["indexed_hypothesis_count"] == 2
    assert source_ref["enabled_hypothesis_count"] == 1
    assert source_ref["compatible_hypothesis_count"] == 1


def test_select_manual_hypotheses_filters_by_agent_mode(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.mode = AGENT_MODE
    cfg.research.manual_sample_size = 1
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        title="Legacy-only hypothesis",
        agent_modes=["legacy"],
    )
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000002",
        title="AutoGluon-compatible hypothesis",
        agent_modes=["autogluon"],
    )

    selection = research.select_manual_hypotheses(
        cfg,
        completed_steps=10,
        repo_root=tmp_path,
    )

    assert selection.hypotheses[0].id in {"000002", "000003"}
    source_ref = json.loads(
        (Path(cfg.log_dir) / "research_hypotheses" / "source_ref.json").read_text()
    )
    assert source_ref["agent_mode"] == "autogluon"
    assert source_ref["indexed_hypothesis_count"] == 2
    assert source_ref["enabled_hypothesis_count"] == 2
    assert source_ref["compatible_hypothesis_count"] == 1


def test_select_manual_hypotheses_can_ignore_agent_modes(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.mode = AGENT_MODE
    cfg.research.manual_sample_size = 2
    cfg.research.ignore_hypothesis_agent_modes = True
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        title="Legacy-only hypothesis",
        agent_modes=["legacy"],
    )
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000002",
        title="AutoGluon-compatible hypothesis",
        agent_modes=["autogluon"],
    )

    selection = research.select_manual_hypotheses(
        cfg,
        completed_steps=10,
        repo_root=tmp_path,
    )

    assert {hypothesis.id for hypothesis in selection.hypotheses} == {
        "000001",
        "000002",
    }
    source_ref = json.loads(
        (Path(cfg.log_dir) / "research_hypotheses" / "source_ref.json").read_text()
    )
    assert source_ref["agent_mode"] == "autogluon"
    assert source_ref["enabled_hypothesis_count"] == 2
    assert source_ref["compatible_hypothesis_count"] == 2
    assert source_ref["compatible_hypothesis_ids"] == ["000001", "000002"]


def test_select_hypothesis_for_root_excludes_root_ids_and_detects_exhaustion(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    for idx in range(1, 3):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )
    journal = Journal()
    used_root = _node(0.9, code="print('ok')", plan="root")
    used_root.research_mode = "hypothesis"
    used_root.research_hypotheses_offered = ["000001"]
    journal.append(used_root)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=None,
        completed_steps=1,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000002"]
    assert (
        research.hypothesis_root_pool_exhausted(
            cfg,
            journal=journal,
            repo_root=tmp_path,
        )
        is False
    )


def test_agent_hypotheses_enables_research_hypothesis_pipeline(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = "hypothesis-pipeline"
    cfg.agent.hypotheses = 3

    cfg = prep_cfg(cfg)

    assert cfg.agent.hypotheses == 3
    assert cfg.research.enabled is True
    assert cfg.research.mode == "hypothesis"
    assert cfg.research.materialize is True
    assert cfg.research.execute is True


def test_agent_hypotheses_caps_effective_root_limit(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.hypotheses = 2
    cfg.research.hypothesis_root_limit = 100

    assert research.effective_hypothesis_root_limit(cfg, compatible_count=7) == 2


def test_record_hypothesis_only_selection_writes_artifact_without_node(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    journal = Journal()
    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=None,
        completed_steps=0,
        repo_root=tmp_path,
    )

    path = research.record_hypothesis_only_selection(
        cfg=cfg,
        selection=selection,
        parent_node=None,
        completed_steps=0,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["hypotheses"][0]["id"] == "000001"
    assert payload["materialized"] is False
    assert payload["executed"] is False
    assert len(journal.nodes) == 0


def test_format_hypothesis_only_finish_message_lists_created_hypotheses(tmp_path):
    source_dir = tmp_path / "research_hypotheses" / "playground-series-s6e5"
    hypotheses = [
        research.ManualHypothesis(
            id="000001",
            enabled=True,
            agent_modes=["legacy"],
            title="Broad photometry ratios",
            summary="summary",
            rationale="rationale",
            implementation_hint="hint",
            expected_effect="effect",
            risk="risk",
            sources=[],
            path=source_dir / "000001" / "hypothesis-000001.json",
        ),
        research.ManualHypothesis(
            id="000002",
            enabled=True,
            agent_modes=["legacy"],
            title="Fold-safe group stats",
            summary="summary",
            rationale="rationale",
            implementation_hint="hint",
            expected_effect="effect",
            risk="risk",
            sources=[],
            path=source_dir / "000002" / "hypothesis-000002.json",
        ),
    ]

    message = format_hypothesis_only_finish_message(
        2,
        hypotheses,
        repo_root=tmp_path,
    )

    assert "creating 2 initial hypotheses" in message
    assert "ROOT" not in message
    assert "- 000001: Broad photometry ratios -> research_hypotheses" in message
    assert "- 000002: Fold-safe group stats -> research_hypotheses" in message


def test_store_generated_research_hypotheses_persists_library_and_run_record(
    tmp_path,
):
    cfg = _manual_cfg(tmp_path)
    checkpoint_dir = Path(cfg.log_dir) / "research" / "checkpoint-000000"
    parsed_response = {
        "summary": "Generated root hypotheses.",
        "hypotheses": [
            _generated_hypothesis_payload(),
            _generated_hypothesis_payload(
                title="Alternative categorical root",
                summary="Test categorical-heavy preprocessing with robust CV.",
            ),
        ],
    }

    selection = research.store_generated_research_hypotheses(
        cfg=cfg,
        parsed_response=parsed_response,
        completed_steps=0,
        checkpoint_dir=checkpoint_dir,
        count=2,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == [
        "000001",
        "000002",
    ]
    first_path = (
        tmp_path
        / "research_hypotheses"
        / "playground-series-s6e5"
        / "000001"
        / "hypothesis-000001.json"
    )
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert first_payload["enabled"] is True
    assert first_payload["agent_modes"] == ["legacy"]
    assert first_payload["summary"] == (
        "Build a broad tabular baseline with rich feature coverage."
    )
    assert first_payload["feature_family"] == "broad_tabular_feature_expansion"
    assert "simple tree-boosting" in first_payload["baseline_model_panel"]
    assert first_payload["implementation_hint"] == first_payload["materialization_hint"]
    assert first_payload["expected_effect"] == first_payload["expected_signal"]
    assert first_payload["novelty_confidence"] == "high"
    assert "Feature family: broad_tabular_feature_expansion" in first_payload["rationale"]
    generated_log = (
        Path(cfg.log_dir) / "research_hypotheses" / "generated_hypotheses.jsonl"
    )
    records = generated_log.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1
    record = json.loads(records[0])
    assert record["hypothesis_ids"] == ["000001", "000002"]
    assert record["checkpoint_step"] == 0


def test_generate_research_hypotheses_writes_per_hypothesis_prompt_and_response(
    tmp_path,
):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.gpu = True
    cfg.agent.aux = "star_classification.csv"
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        title="AutoGluon-only idea",
        summary="This must not enter a legacy hypothesis prompt.",
        agent_modes=["autogluon"],
    )
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000002",
        title="Disabled idea",
        summary="This must not enter any hypothesis prompt.",
        enabled=False,
    )
    seen: list[tuple[str, str]] = []

    def fake_runner(cmd, *, input, text, capture_output, timeout, cwd):
        del text, capture_output, timeout
        hypothesis_id = Path(cwd).name
        seen.append((hypothesis_id, input))
        response = {
            "summary": f"generated {hypothesis_id}",
            "hypotheses": [
                _generated_hypothesis_payload(
                    title=f"Generated {hypothesis_id}",
                    summary=f"Summary {hypothesis_id}",
                )
            ],
        }
        (Path(cwd) / "response_raw.txt").write_text(
            json.dumps(response),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    selection = research.generate_research_hypotheses_for_pipeline(
        cfg=cfg,
        task_desc="task",
        journal=Journal(),
        completed_steps=0,
        count=2,
        runner=fake_runner,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == [
        "000003",
        "000004",
    ]
    assert [hypothesis_id for hypothesis_id, _prompt in seen] == [
        "000003",
        "000004",
    ]
    assert "Generated 000003" not in seen[0][1]
    assert "Generated 000003" in seen[1][1]
    assert "AutoGluon-only idea" not in seen[1][1]
    assert "Disabled idea" not in seen[1][1]
    assert '"id"' not in seen[1][1]
    assert '"enabled"' not in seen[1][1]
    assert '"agent_modes"' not in seen[1][1]

    first_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000003"
    )
    second_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000004"
    )
    for hypothesis_dir, hypothesis_id in [
        (first_dir, "000003"),
        (second_dir, "000004"),
    ]:
        assert (hypothesis_dir / "request.md").exists()
        assert (hypothesis_dir / "request.json").exists()
        assert (hypothesis_dir / "response_raw.txt").exists()
        assert (hypothesis_dir / "response.json").exists()
        hypothesis_path = hypothesis_dir / f"hypothesis-{hypothesis_id}.json"
        payload = json.loads(hypothesis_path.read_text(encoding="utf-8"))
        assert payload["source_request_path"].endswith(
            f"research_hypotheses/playground-series-s6e5/{hypothesis_id}/request.md"
        )
        assert payload["source_response_path"].endswith(
            f"research_hypotheses/playground-series-s6e5/{hypothesis_id}/response.json"
        )
        request = json.loads((hypothesis_dir / "request.json").read_text())
        assert request["cfg_snapshot"]["agent"]["gpu"] is True
        assert request["cfg_snapshot"]["agent"]["aux"] == "star_classification.csv"
        assert request["cfg_snapshot"]["research"]["materialize"] is True
        assert request["cfg_snapshot"]["research"]["execute"] is True

    generated_log = (
        Path(cfg.log_dir) / "research_hypotheses" / "generated_hypotheses.jsonl"
    )
    records = [
        json.loads(line)
        for line in generated_log.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["hypothesis_ids"] for record in records] == [
        ["000003"],
        ["000004"],
    ]
    assert records[0]["checkpoint_dirs"][0].endswith(
        "research_hypotheses/playground-series-s6e5/000003"
    )


def test_select_hypothesis_for_root_prioritizes_generated_run_hypotheses(
    tmp_path,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    checkpoint_dir = Path(cfg.log_dir) / "research" / "checkpoint-000000"
    research.store_generated_research_hypotheses(
        cfg=cfg,
        parsed_response={
            "summary": "Generated root hypotheses.",
            "hypotheses": [_generated_hypothesis_payload(title="Generated root")],
        },
        completed_steps=0,
        checkpoint_dir=checkpoint_dir,
        count=1,
        repo_root=tmp_path,
    )

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=Journal(),
        parent_node=None,
        completed_steps=0,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000002"]


def test_select_hypothesis_for_node_uses_forced_disabled_child_queue(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "001172")
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "001176",
        enabled=False,
    )
    research.write_forced_child_hypothesis_queue(
        cfg,
        root_hypothesis="001172",
        children=("001176",),
    )
    journal = Journal()
    root = _node(0.95, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["001172"]
    journal.append(root)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=root,
        completed_steps=1,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["001176"]


def test_forced_child_queue_skips_already_materialized_child(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "001172")
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "001176",
        enabled=False,
    )
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "001173",
        enabled=False,
    )
    research.write_forced_child_hypothesis_queue(
        cfg,
        root_hypothesis="001172",
        children=("001176", "001173"),
    )
    journal = Journal()
    root = _node(0.95, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["001172"]
    journal.append(root)
    child = _node(0.94, code="print('child')", plan="child")
    child.parent = root
    root.children.add(child)
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["001176"]
    journal.append(child)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=root,
        completed_steps=2,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["001173"]

    second_root = _node(0.91, code="print('ok')", plan="root")
    second_root.research_mode = "hypothesis"
    second_root.research_hypotheses_offered = ["000002"]
    journal.append(second_root)

    assert (
        research.hypothesis_root_pool_exhausted(
            cfg,
            journal=journal,
            repo_root=tmp_path,
        )
        is True
    )


def test_select_hypothesis_for_node_uses_forced_disabled_child_queue_for_nonroot_parent(
    tmp_path,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "001172")
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "001189")
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "001193",
        enabled=False,
    )
    research.write_forced_child_hypothesis_queue(
        cfg,
        root_hypothesis="001189",
        children=("001193",),
    )
    journal = Journal()
    root = _node(0.95, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["001172"]
    journal.append(root)
    branch_parent = _node(0.954, code="print('branch')", plan="branch")
    branch_parent.parent = root
    root.children.add(branch_parent)
    branch_parent.research_mode = "hypothesis"
    branch_parent.research_hypotheses_offered = ["001189"]
    journal.append(branch_parent)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=branch_parent,
        completed_steps=2,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["001193"]


def test_hypothesis_root_pool_respects_configured_root_limit(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.research.hypothesis_root_limit = 1
    for idx in range(1, 4):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )
    journal = Journal()
    used_root = _node(0.9, code="print('ok')", plan="root")
    used_root.research_mode = "hypothesis"
    used_root.research_hypotheses_offered = ["000001"]
    journal.append(used_root)

    assert (
        research.hypothesis_root_pool_exhausted(
            cfg,
            journal=journal,
            repo_root=tmp_path,
        )
        is True
    )
    cfg.research.hypothesis_root_limit = 2

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=None,
        completed_steps=1,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000002"]


def test_hypothesis_root_limit_is_capped_by_compatible_library_size(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.research.hypothesis_root_limit = 1000
    for idx in range(1, 3):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )
    journal = Journal()
    for hypothesis_id in ["000001", "000002"]:
        used_root = _node(0.9, code="print('ok')", plan="root")
        used_root.research_mode = "hypothesis"
        used_root.research_hypotheses_offered = [hypothesis_id]
        journal.append(used_root)

    assert research.effective_hypothesis_root_limit(cfg, compatible_count=2) == 2
    assert (
        research.hypothesis_root_pool_exhausted(
            cfg,
            journal=journal,
            repo_root=tmp_path,
        )
        is True
    )


def test_hypothesis_child_selection_uses_full_library_after_root_limit(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.research.hypothesis_root_limit = 1
    for idx in range(1, 4):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )
    journal = Journal()
    root = _node(0.9, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000001"]
    journal.append(root)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=root,
        completed_steps=1,
        repo_root=tmp_path,
    )

    assert selection.hypotheses[0].id in {"000002", "000003"}


def test_select_hypothesis_avoids_previously_offered_interrupted_candidate(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    for idx in range(1, 3):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )
    usage_dir = Path(cfg.log_dir) / "research_hypotheses"
    usage_dir.mkdir(parents=True)
    (usage_dir / "usage.json").write_text(
        json.dumps({"000001": {"offered_count": 1}}),
        encoding="utf-8",
    )

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=Journal(),
        parent_node=None,
        completed_steps=0,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000002"]


def test_select_hypothesis_for_root_can_prioritize_manifest_scores(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.mode = "legacy"
    cfg.research.hypothesis_root_order = "manifest_score"
    for idx in range(1, 4):
        hypothesis_id = f"{idx:06d}"
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            hypothesis_id,
            title=f"Hypothesis {idx}",
        )
    _write_code_manifest(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        agent_mode="autogluon",
        file_name="autogluon-001.py",
        score=0.91,
    )
    _write_code_manifest(
        tmp_path,
        "playground-series-s6e5",
        "000002",
        agent_mode="autogluon",
        file_name="autogluon-001.py",
        score=0.95,
    )

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=Journal(),
        parent_node=None,
        completed_steps=0,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000002"]
    source_ref = json.loads(
        (Path(cfg.log_dir) / "research_hypotheses" / "source_ref.json").read_text()
    )
    assert source_ref["agent_mode"] == "legacy"
    assert source_ref["hypothesis_root_order"] == "manifest_score"
    assert source_ref["hypothesis_root_score_mode"] == "autogluon"


def test_reserve_hypothesis_roots_returns_unique_manifest_score_order(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.research.hypothesis_root_order = "manifest_score"
    cfg.research.hypothesis_root_score_mode = "autogluon"
    for hypothesis_id, score in [
        ("000001", 0.91),
        ("000002", 0.95),
        ("000003", 0.93),
    ]:
        _write_manual_hypothesis(tmp_path, "playground-series-s6e5", hypothesis_id)
        _write_code_manifest(
            tmp_path,
            "playground-series-s6e5",
            hypothesis_id,
            agent_mode="autogluon",
            file_name="autogluon-001.py",
            score=score,
        )

    reservations = research.reserve_hypothesis_roots(
        cfg,
        journal=Journal(),
        count=2,
        completed_steps=0,
        repo_root=tmp_path,
    )

    assert [reservation.hypothesis_id for reservation in reservations] == [
        "000002",
        "000003",
    ]
    assert [reservation.completed_steps for reservation in reservations] == [0, 1]


def test_reserve_hypothesis_roots_uses_forced_ids_and_ignores_root_limit(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.research.hypothesis_root_limit = 1
    for hypothesis_id in ["000001", "000002", "000003"]:
        _write_manual_hypothesis(tmp_path, "playground-series-s6e5", hypothesis_id)

    reservations = research.reserve_hypothesis_roots(
        cfg,
        journal=Journal(),
        count=3,
        completed_steps=0,
        forced_hypothesis_ids=("000003", "000001"),
        repo_root=tmp_path,
    )

    assert [reservation.hypothesis_id for reservation in reservations] == [
        "000003",
        "000001",
    ]
    assert [reservation.completed_steps for reservation in reservations] == [0, 1]


def test_reserve_hypothesis_roots_uses_forced_disabled_ids(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        enabled=False,
        agent_modes=["legacy"],
    )
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000002",
        enabled=False,
        agent_modes=["autogluon"],
    )

    reservations = research.reserve_hypothesis_roots(
        cfg,
        journal=Journal(),
        count=2,
        completed_steps=0,
        forced_hypothesis_ids=("000001",),
        repo_root=tmp_path,
    )

    assert [reservation.hypothesis_id for reservation in reservations] == ["000001"]

    with pytest.raises(ValueError, match="000002"):
        research.reserve_hypothesis_roots(
            cfg,
            journal=Journal(),
            count=2,
            completed_steps=0,
            forced_hypothesis_ids=("000002",),
            repo_root=tmp_path,
        )


def test_reserve_hypothesis_roots_retries_failed_generation_first(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000002")
    research.record_hypothesis_root_generation_failure(
        cfg,
        hypothesis_id="000002",
        attempts=3,
        message="RuntimeError: network",
    )

    reservations = research.reserve_hypothesis_roots(
        cfg,
        journal=Journal(),
        count=1,
        completed_steps=0,
        repo_root=tmp_path,
    )

    assert [reservation.hypothesis_id for reservation in reservations] == ["000002"]


def test_reserve_hypothesis_roots_retries_unmaterialized_offers_first(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    for hypothesis_id in ["000001", "000002", "000003"]:
        _write_manual_hypothesis(tmp_path, "playground-series-s6e5", hypothesis_id)

    usage_dir = Path(cfg.log_dir) / "research_hypotheses"
    usage_dir.mkdir(parents=True)
    (usage_dir / "offers.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "checkpoint_step": 0,
                        "offered": ["000001"],
                        "source_hash": "sha256:test",
                        "created_at": "2026-05-23T00:00:00",
                    }
                ),
                json.dumps(
                    {
                        "checkpoint_step": 1,
                        "offered": ["000002"],
                        "source_hash": "sha256:test",
                        "created_at": "2026-05-23T00:01:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (usage_dir / "usage.json").write_text(
        json.dumps(
            {
                "000001": {"offered_count": 1},
                "000002": {"offered_count": 1},
            }
        ),
        encoding="utf-8",
    )
    journal = Journal()
    materialized = _node(0.9, code="print('ok')", plan="root")
    materialized.research_mode = "hypothesis"
    materialized.research_hypotheses_offered = ["000001"]
    journal.append(materialized)

    reservations = research.reserve_hypothesis_roots(
        cfg,
        journal=journal,
        count=2,
        completed_steps=1,
        repo_root=tmp_path,
    )

    assert [reservation.hypothesis_id for reservation in reservations] == [
        "000002",
        "000003",
    ]


def test_reserve_hypothesis_roots_uses_generated_hypotheses_not_generated_nodes(
    tmp_path,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    checkpoint_dir = tmp_path / "logs" / "research" / "checkpoint-000000"
    checkpoint_dir.mkdir(parents=True)
    parsed = {
        "hypotheses": [
            {
                "title": f"Root {idx}",
                "summary": f"Summary {idx}",
                "rationale": f"Rationale {idx}",
                "feature_family": f"Feature family {idx}",
                "feature_strategy": f"Feature strategy {idx}",
                "baseline_model_panel": "CatBoost, LightGBM, XGBoost.",
                "model_panel_rationale": "Use a simple GPU-capable tree panel.",
                "validation_strategy": "5-fold stratified CV.",
                "materialization_hint": "Create feature-first root code.",
                "expected_signal": f"Effect {idx}",
                "novelty_confidence": "medium",
                "risk": f"Risk {idx}",
                "sources": [],
            }
            for idx in range(3)
        ]
    }
    selection = research.store_generated_research_hypotheses(
        cfg=cfg,
        parsed_response=parsed,
        completed_steps=0,
        checkpoint_dir=checkpoint_dir,
        count=3,
        repo_root=tmp_path,
    )
    journal = Journal()
    for hypothesis in selection.hypotheses:
        node = Node(code="", plan="generated")
        node.research_mode = "hypothesis"
        node.research_hypotheses_offered = [hypothesis.id]
        node.status = "generated"
        journal.append(node)

    reservations = research.reserve_hypothesis_roots(
        cfg,
        journal=journal,
        count=3,
        completed_steps=0,
        repo_root=tmp_path,
    )

    assert [reservation.hypothesis_id for reservation in reservations] == [
        "000001",
        "000002",
        "000003",
    ]


def test_reserve_hypothesis_roots_excludes_inflight_reserved_ids(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.research.hypothesis_root_order = "manifest_score"
    cfg.research.hypothesis_root_score_mode = "autogluon"
    for hypothesis_id, score in [
        ("000001", 0.95),
        ("000002", 0.94),
        ("000003", 0.93),
    ]:
        _write_manual_hypothesis(tmp_path, "playground-series-s6e5", hypothesis_id)
        _write_code_manifest(
            tmp_path,
            "playground-series-s6e5",
            hypothesis_id,
            agent_mode="autogluon",
            file_name="autogluon-001.py",
            score=score,
        )

    reservations = research.reserve_hypothesis_roots(
        cfg,
        journal=Journal(),
        count=1,
        completed_steps=0,
        reserved_hypothesis_ids={"000001", "000002"},
        repo_root=tmp_path,
    )

    assert [reservation.hypothesis_id for reservation in reservations] == ["000003"]


def test_select_hypothesis_for_child_excludes_ancestors_and_siblings(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    for idx in range(1, 5):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )
    journal = Journal()
    root = _node(0.9, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000001"]
    sibling_a = _node(0.91, code="print('child')", plan="child")
    sibling_a.parent = root
    root.children.add(sibling_a)
    sibling_a.research_mode = "hypothesis"
    sibling_a.research_hypotheses_offered = ["000002"]
    sibling_b = _node(0.92, code="print('child')", plan="child")
    sibling_b.parent = root
    root.children.add(sibling_b)
    sibling_b.research_mode = "hypothesis"
    sibling_b.research_hypotheses_offered = ["000003"]
    journal.append(root)
    journal.append(sibling_a)
    journal.append(sibling_b)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=root,
        completed_steps=3,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000004"]


def test_select_hypothesis_for_child_prefers_root_score_ranking(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.search.hypothesis_child_order = "root_score"
    for idx in range(1, 6):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )

    journal = Journal()
    root = _node(0.91, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000001"]

    root_rank_2 = _node(0.92, code="print('h2')", plan="h2")
    root_rank_2.research_mode = "hypothesis"
    root_rank_2.research_hypotheses_offered = ["000002"]
    root_rank_1 = _node(0.94, code="print('h3')", plan="h3")
    root_rank_1.research_mode = "hypothesis"
    root_rank_1.research_hypotheses_offered = ["000003"]
    root_rank_3 = _node(0.90, code="print('h4')", plan="h4")
    root_rank_3.research_mode = "hypothesis"
    root_rank_3.research_hypotheses_offered = ["000004"]

    journal.append(root)
    journal.append(root_rank_2)
    journal.append(root_rank_1)
    journal.append(root_rank_3)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=root,
        completed_steps=4,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000003"]


def test_select_hypothesis_for_child_keeps_untested_hypotheses_available(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.search.hypothesis_child_order = "root_score"
    for idx in range(1, 5):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )

    journal = Journal()
    root = _node(0.91, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000001"]

    scored_root = _node(0.92, code="print('h2')", plan="h2")
    scored_root.research_mode = "hypothesis"
    scored_root.research_hypotheses_offered = ["000002"]
    used_child = _node(0.90, code="print('child')", plan="child")
    used_child.parent = root
    root.children.add(used_child)
    used_child.research_mode = "hypothesis"
    used_child.research_hypotheses_offered = ["000002"]

    journal.append(root)
    journal.append(scored_root)
    journal.append(used_child)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=root,
        completed_steps=3,
        repo_root=tmp_path,
    )

    assert selection.hypotheses[0].id in {"000003", "000004"}


def test_child_ranking_uses_new_root_scores_after_root_limit_extension(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.search.hypothesis_child_order = "root_score"
    for idx in range(1, 5):
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            f"{idx:06d}",
            title=f"Hypothesis {idx}",
        )

    journal = Journal()
    parent = _node(0.91, code="print('parent')", plan="parent")
    parent.research_mode = "hypothesis"
    parent.research_hypotheses_offered = ["000001"]
    old_root = _node(0.92, code="print('old')", plan="old root")
    old_root.research_mode = "hypothesis"
    old_root.research_hypotheses_offered = ["000002"]
    new_root = _node(0.95, code="print('new')", plan="new root")
    new_root.research_mode = "hypothesis"
    new_root.research_hypotheses_offered = ["000003"]
    journal.append(parent)
    journal.append(old_root)
    journal.append(new_root)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=parent,
        completed_steps=3,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000003"]


def test_select_hypothesis_for_debug_inherits_buggy_parent_hypothesis(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000007",
        title="Buggy parent hypothesis",
    )
    journal = Journal()
    parent = _node(None, code="raise RuntimeError('bug')", plan="bug")
    parent.research_mode = "hypothesis"
    parent.research_hypotheses_offered = ["000007"]
    journal.append(parent)

    selection = research.select_hypothesis_for_node(
        cfg,
        journal=journal,
        parent_node=parent,
        completed_steps=1,
        repo_root=tmp_path,
    )

    assert [hypothesis.id for hypothesis in selection.hypotheses] == ["000007"]


def test_format_hypothesis_for_prompt_omits_self_report_contract(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        title="Grouped validation",
        summary="Use grouped validation to reduce public/CV mismatch.",
    )
    selection = research.select_hypothesis_for_node(
        cfg,
        journal=Journal(),
        parent_node=None,
        completed_steps=0,
        repo_root=tmp_path,
    )

    rendered = research.format_hypothesis_for_prompt(selection)

    assert "Hypothesis verification contract" in rendered
    assert "Hypothesis ID: 000001" in rendered
    assert "Implement this exact hypothesis" in rendered
    assert "research_hypotheses_llm_claimed_used" not in rendered
    assert "Do not add hypothesis-id bookkeeping" in rendered
    assert "Research source hash" not in rendered
    assert "Sources:" not in rendered


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


def test_collect_research_context_rounds_scores_for_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    journal = Journal()
    node = _node(
        0.9507496188899213,
        code="print('score rounding')",
        plan="rounding",
    )
    journal.append(node)
    (Path(cfg.log_dir).parent / "submission_registry.json").write_text(
        json.dumps(
            {
                "submissions": [
                    {
                        "run": cfg.exp_name,
                        "step": node.step,
                        "timestamp": dt.datetime.fromtimestamp(
                            node.ctime
                        ).strftime("%Y%m%dT%H%M%S"),
                        "node_id": node.id,
                        "remote_status": "COMPLETE",
                        "public_score": "0.948831234",
                    }
                ]
            }
        )
    )

    context = collect_research_context(
        cfg=cfg,
        task_desc="task",
        journal=journal,
        completed_steps=10,
    )

    payload = context["best_working_solutions"][0]
    assert payload["local_cv_score"] == 0.95075
    assert payload["kaggle_public_score"] == 0.94883


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
                        "public_score": "0.7012449",
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
    compact_prompt = " ".join(prompt.split())

    assert prompt.startswith(
        "You are a research scientist and Kaggle competition strategist."
    )
    assert "Return only structured JSON" in compact_prompt
    assert "task" in prompt
    assert "## Data overview" in prompt
    assert '"data_overview"' not in prompt
    assert "```json" not in prompt
    assert '"run_id"' not in prompt
    assert '"checkpoint_step"' not in prompt
    assert '"created_at"' not in prompt
    assert '"step"' not in prompt
    assert '"stage"' not in prompt
    assert '"additionalProperties"' not in prompt
    assert '"parent_node_id"' not in prompt
    assert '"parent_step"' not in prompt
    assert "hypotheses[].target" not in prompt
    assert "Generate initial hypotheses for feature search" in prompt
    assert (
        "Return exactly 5 concise new initial feature-search hypotheses"
        in compact_prompt
    )
    assert "hypotheses[].summary" in prompt
    assert "hypotheses[].feature_family" in prompt
    assert "hypotheses[].baseline_model_panel" in prompt
    assert "hypotheses[].novelty_confidence" in prompt
    assert "Do not force weak novelty" in prompt
    assert "novelty_confidence to \"low\"" in prompt
    assert "# Novelty dimensions" in prompt
    assert "feature representation family" in prompt
    assert "physical or domain-specific representation" in prompt
    assert "Changing only the model panel" in prompt
    assert "only the number of bins is not" in prompt
    assert "basic algorithm families fit the engineered features best" in compact_prompt
    assert "not as a fixed magic list of model names" in prompt
    assert "baseline model panel should normally be reused" in prompt
    assert "not from changing panel composition" in prompt
    assert "Do not propose heavy ensembling" in prompt
    assert "Do not target a specific previous node or code block" in prompt


def test_research_prompt_uses_requested_hypothesis_count():
    prompt = build_research_prompt(
        {
            "task_desc": "task",
            "best_working_solutions": [],
            "worst_working_solutions": [],
            "hypothesis_count": 3,
        }
    )
    compact_prompt = " ".join(prompt.split())

    assert (
        "Return exactly 3 concise new initial feature-search hypotheses"
        in compact_prompt
    )
    assert "contain exactly 3 items" in prompt


def test_research_context_lists_existing_hypotheses_as_text_only(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.mode = "legacy"
    for hypothesis_id, title in [
        ("000001", "First feature family"),
        ("000002", "Second feature family"),
        ("000003", "Third feature family"),
    ]:
        _write_manual_hypothesis(
            tmp_path,
            "playground-series-s6e5",
            hypothesis_id,
            title=title,
        )
    _write_code_manifest(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        agent_mode="legacy",
        file_name="legacy-001.py",
        score=0.91234,
    )

    context = collect_research_context(
        cfg=cfg,
        task_desc="task",
        journal=Journal(),
        completed_steps=0,
        repo_root=tmp_path,
    )
    prompt = build_research_prompt(context)

    assert len(context["existing_hypotheses"]) == 3
    assert "First feature family" in context["existing_hypotheses"][0]
    assert "Second feature family" in context["existing_hypotheses"][1]
    assert "Third feature family" in context["existing_hypotheses"][2]
    assert "Validation metric" not in prompt
    assert "Evidence type" not in prompt
    assert "Status:" not in prompt
    assert "## Existing hypotheses" in prompt
    assert "## Executed hypotheses" not in prompt
    assert "## Buggy hypotheses" not in prompt
    assert "## Unexecuted hypotheses" not in prompt

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

    assert "## Previous research summaries" in prompt
    assert '"previous_research_summaries"' not in prompt
    assert "```json" not in prompt
    assert '"label"' not in prompt
    assert "Try pit-window features" in prompt
    assert "context for choosing a distinct next direction" in " ".join(prompt.split())


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
                            "summary": "Test calibrated LightGBM probabilities.",
                            "feature_family": "probability_shape_features",
                            "feature_strategy": "Build probability-shape features.",
                            "baseline_model_panel": (
                                "Use a compact mix of simple tree and linear "
                                "baselines when compatible."
                            ),
                            "model_panel_rationale": (
                                "Different simple model families expose whether "
                                "the feature representation is broadly useful."
                            ),
                            "validation_strategy": "Use leak-free 5-fold CV.",
                            "materialization_hint": "Build staged features and model panel.",
                            "expected_signal": "small AUC gain",
                            "novelty_confidence": "medium",
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

    assert checkpoint_dir == (
        Path(cfg.log_dir) / "artifacts" / "research-checkpoint-000010"
    )
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
    assert command[command.index("--model") + 1] == "gpt-5.4-mini"
    assert 'model_reasoning_effort="low"' in command
    assert seen["stdin"].startswith(
        "You are a research scientist and Kaggle competition strategist."
    )
    assert (checkpoint_dir / "request.json").exists()
    assert (checkpoint_dir / "request.md").exists()
    assert (
        (checkpoint_dir / "codex_profile.toml")
        .read_text()
        .startswith('model = "gpt-5.4-mini"')
    )
    response = json.loads((checkpoint_dir / "response.json").read_text())
    assert response["parsed_response"]["summary"] == "researched"
    assert response["raw_response"].startswith('{"summary":')
    assert set(response["timings_seconds"]) >= {
        "build_prompt",
        "write_inputs",
        "codex_subprocess",
        "read_response",
        "parse_response",
        "total",
    }
    assert all(value >= 0 for value in response["timings_seconds"].values())
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
    assert status["timings_seconds"] == response["timings_seconds"]


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


def test_manual_research_advisor_records_offer_without_codex_runner(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "manual"
    cfg.research.every_steps = 2
    cfg.research.manual_sample_size = 1
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        title="Grouped validation",
    )
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')", plan="ok"))
    journal.append(_node(0.8, code="print('ok2')", plan="ok2"))

    def fail_runner(*_args, **_kwargs):
        raise AssertionError("manual mode must not invoke Codex research runner")

    advisor = ResearchAdvisor(
        cfg=cfg,
        task_desc="task",
        runner=fail_runner,
        repo_root=tmp_path,
    )

    assert (
        advisor.maybe_start(
            journal=journal,
            completed_steps=count_scored_working_nodes(journal),
        )
        is True
    )
    assert (Path(cfg.log_dir) / "research_hypotheses" / "offers.jsonl").exists()
    assert not (Path(cfg.log_dir) / "research" / "checkpoint-000002").exists()


def test_manual_research_advisor_does_not_duplicate_existing_offer(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "manual"
    cfg.research.every_steps = 1
    cfg.research.manual_sample_size = 1
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')", plan="ok"))
    advisor = ResearchAdvisor(
        cfg=cfg,
        task_desc="task",
        runner=lambda *_args, **_kwargs: None,
        repo_root=tmp_path,
    )

    assert advisor.maybe_start(journal=journal, completed_steps=1) is True
    assert advisor.maybe_start(journal=journal, completed_steps=1) is False

    offers = (
        Path(cfg.log_dir) / "research_hypotheses" / "offers.jsonl"
    ).read_text().splitlines()
    assert len(offers) == 1


def test_manual_research_advisor_status_text_shows_latest_offer(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "manual"
    cfg.research.every_steps = 1
    cfg.research.manual_sample_size = 1
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')", plan="ok"))
    advisor = ResearchAdvisor(
        cfg=cfg,
        task_desc="task",
        runner=lambda *_args, **_kwargs: None,
        repo_root=tmp_path,
    )

    advisor.maybe_start(journal=journal, completed_steps=1)

    assert advisor.status_text() == "[green]Research: ✓ manual 000001"


def test_hypothesis_research_advisor_does_not_start_llm_checkpoint(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.research.every_steps = 1
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    journal = Journal()
    journal.append(_node(0.9, code="print('ok')", plan="ok"))

    def fail_runner(*_args, **_kwargs):
        raise AssertionError("hypothesis mode must not invoke Codex research runner")

    advisor = ResearchAdvisor(
        cfg=cfg,
        task_desc="task",
        runner=fail_runner,
        repo_root=tmp_path,
    )

    assert advisor.maybe_start(journal=journal, completed_steps=1) is False
    assert not (Path(cfg.log_dir) / "research" / "checkpoint-000001").exists()


def test_hypothesis_search_policy_debugs_buggy_root_before_opening_next_root(
    tmp_path,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.mode = "legacy"
    cfg.agent.hypotheses = 2
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000002")
    bug = _node(None, code="raise RuntimeError('bug')", plan="bug")
    bug.research_mode = "hypothesis"
    bug.research_hypotheses_offered = ["000001"]
    journal = Journal()
    journal.append(bug)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    selected = agent.search_policy()

    assert selected is bug
    assert agent.last_search_decision["reason"] == "debugging_buggy_hypothesis_root"


def test_filter_hypothesis_candidate_parents_blocks_disabled_root_branch(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(
        tmp_path,
        "playground-series-s6e5",
        "000013",
        enabled=False,
    )
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000014")
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000015")
    disabled_root = _node(0.96321, code="print('ok')", plan="disabled")
    disabled_root.research_mode = "hypothesis"
    disabled_root.research_hypotheses_offered = ["000013"]
    enabled_root = _node(0.96113, code="print('ok')", plan="enabled")
    enabled_root.research_mode = "hypothesis"
    enabled_root.research_hypotheses_offered = ["000014"]
    journal = Journal()
    journal.append(disabled_root)
    journal.append(enabled_root)

    parents = research.filter_hypothesis_candidate_parents(
        cfg,
        journal=journal,
        parent_nodes=[disabled_root, enabled_root],
        repo_root=tmp_path,
    )

    assert disabled_root not in parents
    assert enabled_root in parents


def test_hypothesis_research_advisor_status_text_shows_latest_hypothesis(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    research.select_hypothesis_for_node(
        cfg,
        journal=Journal(),
        parent_node=None,
        completed_steps=12,
        repo_root=tmp_path,
    )
    advisor = ResearchAdvisor(
        cfg=cfg,
        task_desc="task",
        runner=lambda *_args, **_kwargs: None,
        repo_root=tmp_path,
    )

    assert advisor.status_text() == "[green]Research: ✓ 000012 @ 000001"


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


def test_agent_includes_latest_manual_research_hints_in_draft_prompt(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "manual"
    cfg.agent.data_preview = False
    selection = research.ManualHypothesisSelection(
        completed_steps=10,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000001",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Grouped validation",
                summary="Use grouped validation.",
                rationale="Random holdout can mix race context.",
                implementation_hint="Build Race_Year groups.",
                expected_effect="More reliable validation.",
                risk="Grouped CV may be pessimistic.",
                sources=[],
                path=tmp_path / "hypothesis-000001.json",
            )
        ],
    )
    monkeypatch.setattr(
        "aide.agent.load_latest_manual_research_hints",
        lambda _cfg: selection,
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent._draft()

    assert "Manual research hypotheses offered" in captured["prompt"][
        "External research hints"
    ]
    assert "000001. Grouped validation" in captured["prompt"][
        "External research hints"
    ]
    assert node.research_mode == "manual"
    assert node.research_hypotheses_offered == ["000001"]
    assert node.research_source_hash == "sha256:test"


def test_agent_includes_hard_hypothesis_contract_in_draft_prompt(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    selection = research.ManualHypothesisSelection(
        completed_steps=0,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000001",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Grouped validation",
                summary="Use grouped validation.",
                rationale="Random holdout can mix race context.",
                implementation_hint="Build Race_Year groups.",
                expected_effect="More reliable validation.",
                risk="Grouped CV may be pessimistic.",
                sources=[],
                path=tmp_path / "hypothesis-000001.json",
            )
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "I will verify hypothesis 000001.", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent._draft()

    assert "Hypothesis verification contract" in captured["prompt"][
        "Hypothesis under verification"
    ]
    assert "Hypothesis ID: 000001" in captured["prompt"][
        "Hypothesis under verification"
    ]
    assert "Research source hash" not in captured["prompt"][
        "Hypothesis under verification"
    ]
    assert node.research_mode == "hypothesis"
    assert node.research_hypotheses_offered == ["000001"]
    assert node.research_source_hash == "sha256:test"


def test_hypothesis_root_code_loader_uses_active_manifest_file(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.mode = AGENT_MODE
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "autogluon-001.py").write_text("print('one')\n")
    (hypothesis_dir / "autogluon-002.py").write_text("print('two')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps({"active": {"autogluon": "autogluon-001.py"}}),
        encoding="utf-8",
    )

    root_code = research.load_hypothesis_root_code(
        cfg,
        "000001",
        repo_root=tmp_path,
    )

    assert root_code is not None
    assert root_code.path.name == "autogluon-001.py"
    assert root_code.code == "print('one')\n"


def test_hypothesis_root_code_loader_requires_matching_gpu_variant(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.gpu = True
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("print('cpu')\n")
    (hypothesis_dir / "legacy-002.py").write_text("print('gpu')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "active": {"legacy": "legacy-001.py"},
                "versions": {
                    "legacy": [
                        {"file": "legacy-001.py", "status": "ok", "gpu": False},
                        {"file": "legacy-002.py", "status": "ok", "gpu": True},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    root_code = research.load_hypothesis_root_code(
        cfg,
        "000001",
        repo_root=tmp_path,
    )

    assert root_code is not None
    assert root_code.path.name == "legacy-002.py"
    assert root_code.gpu is True
    assert root_code.code == "print('gpu')\n"


def test_hypothesis_root_code_loader_rejects_cpu_only_code_when_gpu_enabled(
    tmp_path,
):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.gpu = True
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("print('cpu')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "active": {"legacy": "legacy-001.py"},
                "versions": {
                    "legacy": [
                        {"file": "legacy-001.py", "status": "ok", "gpu": False}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        research.load_hypothesis_root_code(cfg, "000001", repo_root=tmp_path)
        is None
    )


def test_hypothesis_root_code_loader_uses_highest_legacy_file(tmp_path):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("print('one')\n")
    (hypothesis_dir / "legacy-002.py").write_text("print('two')\n")

    root_code = research.load_hypothesis_root_code(
        cfg,
        "000001",
        repo_root=tmp_path,
    )

    assert root_code is not None
    assert root_code.path.name == "legacy-002.py"
    assert root_code.code == "print('two')\n"


def test_hypothesis_root_code_loader_ignores_unexecuted_recovered_response(
    tmp_path,
):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.gpu = False
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("print('old ok')\n")
    (hypothesis_dir / "legacy-002.py").write_text("print('recovered')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "active": {"legacy": "legacy-002.py"},
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": False,
                            "status": "ok",
                        },
                        {
                            "file": "legacy-002.py",
                            "buggy": False,
                            "score": None,
                            "node_id": None,
                            "recovered_from": "response.py",
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    root_code = research.load_hypothesis_root_code(
        cfg,
        "000001",
        repo_root=tmp_path,
    )

    assert root_code is None


def test_hypothesis_root_code_loader_loads_single_unmanifested_file(tmp_path):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("print('one')\n")

    root_code = research.load_hypothesis_root_code(
        cfg,
        "000001",
        repo_root=tmp_path,
    )

    assert root_code is not None
    assert root_code.path.name == "legacy-001.py"
    assert root_code.code == "print('one')\n"


def test_hypothesis_root_code_loader_ignores_buggy_manifest_entry(tmp_path):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("raise RuntimeError('bug')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": True,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    root_code = research.load_hypothesis_root_code(
        cfg,
        "000001",
        repo_root=tmp_path,
    )

    assert root_code is None


def test_hypothesis_root_code_loader_ignores_buggy_highest_version(tmp_path):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("print('old ok')\n")
    (hypothesis_dir / "legacy-002.py").write_text("raise RuntimeError('bug')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": False,
                        },
                        {
                            "file": "legacy-002.py",
                            "buggy": True,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    root_code = research.load_hypothesis_root_code(
        cfg,
        "000001",
        repo_root=tmp_path,
    )

    assert root_code is None


def test_scored_hypothesis_root_nodes_use_current_agent_mode(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.mode = "legacy"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text(
        "print('legacy scored')\n",
        encoding="utf-8",
    )
    (hypothesis_dir / "autogluon-001.py").write_text(
        "print('autogluon scored')\n",
        encoding="utf-8",
    )
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "active": {
                    "legacy": "legacy-001.py",
                    "autogluon": "autogluon-001.py",
                },
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": False,
                            "status": "ok",
                            "node_id": "legacy-node",
                            "score": 0.951,
                            "created_at": "2026-05-23T00:00:00",
                        }
                    ],
                    "autogluon": [
                        {
                            "file": "autogluon-001.py",
                            "buggy": False,
                            "status": "ok",
                            "node_id": "autogluon-node",
                            "score": 0.962,
                            "created_at": "2026-05-24T00:00:00",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    legacy_nodes = research.scored_hypothesis_root_nodes(
        cfg,
        repo_root=tmp_path,
    )
    cfg.agent.mode = "autogluon_preprocess"
    autogluon_nodes = research.scored_hypothesis_root_nodes(
        cfg,
        repo_root=tmp_path,
    )

    assert len(legacy_nodes) == 1
    assert legacy_nodes[0].code == "print('legacy scored')\n"
    assert legacy_nodes[0].metric.value == 0.951
    assert legacy_nodes[0].research_hypotheses_offered == ["000001"]
    assert legacy_nodes[0].is_buggy is False
    assert legacy_nodes[0].status == "ok"
    assert len(autogluon_nodes) == 1
    assert autogluon_nodes[0].code == "print('autogluon scored')\n"
    assert autogluon_nodes[0].metric.value == 0.962


def test_scored_hypothesis_root_node_recovers_exec_time_from_journal(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.mode = "legacy"
    cfg.log_dir = tmp_path / "logs" / "2-current-run"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    _write_code_manifest(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        agent_mode="legacy",
        file_name="legacy-001.py",
        score=0.951,
    )
    journal_dir = tmp_path / "logs" / "2-previous-run"
    journal_dir.mkdir(parents=True)
    (journal_dir / "journal.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "node-000001",
                        "exec_time": 377.2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    nodes = research.scored_hypothesis_root_nodes(cfg, repo_root=tmp_path)

    assert len(nodes) == 1
    assert nodes[0].exec_time == 377.2


def test_scored_hypothesis_root_node_recovers_term_out_from_journal(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.mode = "legacy"
    cfg.log_dir = tmp_path / "logs" / "2-current-run"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    _write_code_manifest(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        agent_mode="legacy",
        file_name="legacy-001.py",
        score=0.951,
    )
    journal_dir = tmp_path / "logs" / "2-previous-run"
    journal_dir.mkdir(parents=True)
    (journal_dir / "journal.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "node-000001",
                        "_term_out": [
                            "Fold 1 balanced_accuracy=0.95\n",
                            "OOF balanced_accuracy=0.951\n",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    nodes = research.scored_hypothesis_root_nodes(cfg, repo_root=tmp_path)

    assert len(nodes) == 1
    assert nodes[0]._term_out == [
        "Fold 1 balanced_accuracy=0.95\n",
        "OOF balanced_accuracy=0.951\n",
    ]
    assert nodes[0].run_stats["source_process_stdout_recovered"] is True


def test_scored_hypothesis_root_nodes_skip_unscored_or_buggy_code(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000002")
    _write_code_manifest(
        tmp_path,
        "playground-series-s6e5",
        "000001",
        agent_mode="legacy",
        file_name="legacy-001.py",
        score=None,
    )
    _write_code_manifest(
        tmp_path,
        "playground-series-s6e5",
        "000002",
        agent_mode="legacy",
        file_name="legacy-001.py",
        score=0.94,
        buggy=True,
    )

    nodes = research.scored_hypothesis_root_nodes(cfg, repo_root=tmp_path)

    assert nodes == []


def test_agent_loads_library_hypothesis_root_without_llm(tmp_path, monkeypatch):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("print('from library')\n")
    selection = research.ManualHypothesisSelection(
        completed_steps=0,
        source_hash="sha256:test",
        source_dir=hypothesis_dir.parent,
        hypotheses=[
            research.load_manual_hypothesis_library(
                cfg,
                repo_root=tmp_path,
            ).hypotheses[0]
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        "aide.agent.load_hypothesis_root_code",
        lambda _cfg, hypothesis_id: research.load_hypothesis_root_code(
            _cfg,
            hypothesis_id,
            repo_root=tmp_path,
        ),
    )

    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fail_plan_and_code(_prompt):
        raise AssertionError("LLM should not be called for a library root")

    agent.plan_and_code_query = fail_plan_and_code  # type: ignore[method-assign]

    node = agent.generate_node(None)

    assert node.code == "print('from library')\n"
    assert node.research_mode == "hypothesis"
    assert node.research_hypotheses_offered == ["000001"]


def test_agent_missing_library_root_still_uses_llm(tmp_path, monkeypatch):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    selection = research.ManualHypothesisSelection(
        completed_steps=0,
        source_hash="sha256:test",
        source_dir=tmp_path / "research_hypotheses" / "playground-series-s6e5",
        hypotheses=[
            research.load_manual_hypothesis_library(
                cfg,
                repo_root=tmp_path,
            ).hypotheses[0]
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        "aide.agent.load_hypothesis_root_code",
        lambda _cfg, hypothesis_id: research.load_hypothesis_root_code(
            _cfg,
            hypothesis_id,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    captured = {}

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "I will verify hypothesis 000001.", "print('from llm')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent.generate_node(None)

    assert "Hypothesis under verification" in captured["prompt"]
    assert node.code == "print('from llm')"
    assert node.research_hypotheses_offered == ["000001"]


def test_agent_generates_preselected_hypothesis_root_without_selector(
    tmp_path,
    monkeypatch,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    library = research.load_manual_hypothesis_library(cfg, repo_root=tmp_path)
    selection = research.ManualHypothesisSelection(
        completed_steps=0,
        source_hash=library.source_hash,
        source_dir=library.source_dir,
        hypotheses=[library.hypotheses[0]],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("selector called")
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(prompt):
        return "I will implement hypothesis 000001.", "print('root')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent.generate_preselected_hypothesis_root(
        selection,
        node_ctime=1_779_492_701.0,
        llm_log_dir=tmp_path / "artifact",
        artifact_dir_name="20260523T220603-a1b2c3d4",
    )

    assert node.code == "print('root')"
    assert node.ctime == 1_779_492_701.0
    assert node.artifact_dir_name == "20260523T220603-a1b2c3d4"
    assert node.research_hypotheses_offered == ["000001"]


def test_parallel_root_worker_initializes_missing_data_preview(
    tmp_path,
    monkeypatch,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = True
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    library = research.load_manual_hypothesis_library(cfg, repo_root=tmp_path)
    selection = research.ManualHypothesisSelection(
        completed_steps=0,
        source_hash=library.source_hash,
        source_dir=library.source_dir,
        hypotheses=[library.hypotheses[0]],
    )
    reservation = research.HypothesisRootReservation(
        selection=selection,
        hypothesis_id="000001",
        completed_steps=0,
    )
    job = ParallelRootJob(
        reservation=reservation,
        node_ctime=1_779_492_701.0,
        artifact_dir_name="20260523T220603-a1b2c3d4",
        artifact_dir=tmp_path / "artifact",
        launched_index=1,
    )
    base_agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    assert base_agent.data_preview is None
    captured: dict[str, object] = {}

    def fake_plan_and_code(self, prompt):
        captured["data_overview"] = prompt.get("Data Overview")
        return "I will implement hypothesis 000001.", "print('root')"

    monkeypatch.setattr(Agent, "plan_and_code_query", fake_plan_and_code)

    result = generate_reserved_hypothesis_root(
        base_agent=base_agent,
        journal=Journal(),
        job=job,
    )

    assert result.node.code == "print('root')"
    assert captured["data_overview"] is not None


def test_agent_buggy_library_root_still_uses_llm(tmp_path, monkeypatch):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("raise RuntimeError('bug')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": True,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    selection = research.ManualHypothesisSelection(
        completed_steps=0,
        source_hash="sha256:test",
        source_dir=hypothesis_dir.parent,
        hypotheses=[
            research.load_manual_hypothesis_library(
                cfg,
                repo_root=tmp_path,
            ).hypotheses[0]
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    monkeypatch.setattr(
        "aide.agent.load_hypothesis_root_code",
        lambda _cfg, hypothesis_id: research.load_hypothesis_root_code(
            _cfg,
            hypothesis_id,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(_prompt):
        return "I will replace buggy hypothesis root code.", "print('from llm')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent.generate_node(None)

    assert node.code == "print('from llm')"
    assert node.research_hypotheses_offered == ["000001"]


def test_reviewed_llm_hypothesis_root_saves_single_file(tmp_path, monkeypatch):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    monkeypatch.setattr(
        "aide.agent.save_hypothesis_root_code",
        lambda _cfg, **kwargs: research.save_hypothesis_root_code(
            _cfg,
            **kwargs,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('new root')\n", plan="root")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000001"]

    agent.review_node(
        node,
        ExecutionResult(
            term_out=[
                'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
                '"metric": 0.91, "lower_is_better": false, '
                '"research_hypotheses_llm_claimed_used": ["000001"]}'
            ],
            exec_time=1.0,
            exc_type=None,
        ),
    )

    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    assert (hypothesis_dir / "legacy-001.py").read_text() == "print('new root')\n"
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    assert manifest["active"]["legacy"] == "legacy-001.py"


def test_generated_only_hypothesis_root_saves_single_file(tmp_path, monkeypatch):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    monkeypatch.setattr(
        "aide.agent.save_hypothesis_root_code",
        lambda _cfg, **kwargs: research.save_hypothesis_root_code(
            _cfg,
            **kwargs,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('generated root')\n", plan="root")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000001"]

    agent.save_hypothesis_root_code_for_node(node, activate=False)

    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    assert (hypothesis_dir / "legacy-001.py").read_text() == "print('generated root')\n"
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    entry = manifest["versions"]["legacy"][0]
    assert entry["buggy"] is None
    assert entry["status"] == "generated"
    assert manifest.get("active", {}).get("legacy") is None


def test_buggy_hypothesis_root_manifest_stores_exception_context(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.gpu = True
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")

    message = (
        "REPL child process died unexpectedly\n\n"
        "LightGBM CUDA native crash likely terminated the REPL child process "
        "before Python could raise an exception."
    )
    research.save_hypothesis_root_code(
        cfg,
        hypothesis_id="000001",
        code="print('buggy root')\n",
        is_buggy=True,
        node_id="node-bug",
        score=None,
        created_at="2026-06-10T00:00:00",
        exception_type="RuntimeError",
        exception_info={"args": [message]},
        terminal_output=f"RuntimeError: {message}",
        analysis=message,
        activate=True,
        repo_root=tmp_path,
    )

    failed = research.load_failed_hypothesis_root_code(
        cfg,
        "000001",
        repo_root=tmp_path,
    )

    assert failed is not None
    assert failed.code == "print('buggy root')\n"
    assert failed.exception_type == "RuntimeError"
    assert failed.exception_info == {"args": [message]}
    assert "LightGBM CUDA native crash" in (failed.terminal_output or "")
    assert "LightGBM CUDA native crash" in (failed.analysis or "")


def test_hypothesis_materialization_prompt_includes_previous_bug_context(
    tmp_path,
    monkeypatch,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.gpu = True
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    message = (
        "REPL child process died unexpectedly\n\n"
        "LightGBM CUDA native crash likely terminated the REPL child process "
        "before Python could raise an exception."
    )
    research.save_hypothesis_root_code(
        cfg,
        hypothesis_id="000001",
        code="print('buggy root')\n",
        is_buggy=True,
        node_id="node-bug",
        score=None,
        created_at="2026-06-10T00:00:00",
        exception_type="RuntimeError",
        exception_info={"args": [message]},
        terminal_output=f"RuntimeError: {message}",
        analysis=message,
        activate=True,
        repo_root=tmp_path,
    )
    selection = research.select_hypothesis_by_id(
        cfg,
        hypothesis_id="000001",
        completed_steps=0,
        repo_root=tmp_path,
    )
    captured: dict[str, object] = {}

    def fake_plan_and_code_query(prompt):
        captured["prompt"] = prompt
        return "plan", "print('fixed')\n"

    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    monkeypatch.setattr(agent, "plan_and_code_query", fake_plan_and_code_query)

    agent._draft(hypothesis_selection=selection)

    prompt = captured["prompt"]
    assert isinstance(prompt, dict)
    failed_context = prompt["Previous failed implementation for assigned hypothesis"]
    assert "Exception type:\nRuntimeError" in failed_context
    assert "Exception info:" in failed_context
    assert "Terminal output:" in failed_context
    assert "Analysis:" in failed_context
    assert "LightGBM CUDA native crash" in failed_context
    assert "Previous failed code:" in failed_context
    assert "print('buggy root')" in failed_context
    assert "Bug-fix instruction for the latest failure:" in failed_context
    assert "keep LightGBM on CPU" in failed_context
    assert 'device="cuda"' in failed_context
    assert "Previous failure GPU/CUDA hint" not in prompt


def test_hypothesis_materialization_prompt_does_not_infer_cuda_from_failed_code(
    tmp_path,
    monkeypatch,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.gpu = True
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    research.save_hypothesis_root_code(
        cfg,
        hypothesis_id="000001",
        code='print("previous code mentions LightGBM CUDA but did not crash there")\n',
        is_buggy=True,
        node_id="node-bug",
        score=None,
        created_at="2026-06-10T00:00:00",
        exception_type="TypeError",
        exception_info={"args": ["Cannot interpret 'string[python]' as a data type"]},
        terminal_output="TypeError: Cannot interpret 'string[python]' as a data type",
        analysis="XGBoost could not ingest a pandas string dtype column.",
        activate=True,
        repo_root=tmp_path,
    )
    selection = research.select_hypothesis_by_id(
        cfg,
        hypothesis_id="000001",
        completed_steps=0,
        repo_root=tmp_path,
    )
    captured: dict[str, object] = {}

    def fake_plan_and_code_query(prompt):
        captured["prompt"] = prompt
        return "plan", "print('fixed')\n"

    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    monkeypatch.setattr(agent, "plan_and_code_query", fake_plan_and_code_query)

    agent._draft(hypothesis_selection=selection)

    prompt = captured["prompt"]
    assert isinstance(prompt, dict)
    failed_context = prompt["Previous failed implementation for assigned hypothesis"]
    assert "Exception type:\nTypeError" in failed_context
    assert "Cannot interpret 'string[python]' as a data type" in failed_context
    assert "previous code mentions LightGBM CUDA" in failed_context
    assert "Bug-fix instruction for the latest failure:" not in failed_context
    assert "keep LightGBM on CPU" not in failed_context
    assert "Previous failure GPU/CUDA hint" not in prompt


def test_hypothesis_root_code_save_creates_new_version_when_gpu_changes(
    tmp_path,
):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")

    cpu_path = research.save_hypothesis_root_code(
        cfg,
        hypothesis_id="000001",
        code="print('cpu')\n",
        is_buggy=False,
        node_id="node-cpu",
        score=None,
        created_at="2026-06-10T00:00:00",
        activate=False,
        repo_root=tmp_path,
    )
    cfg.agent.gpu = True
    gpu_path = research.save_hypothesis_root_code(
        cfg,
        hypothesis_id="000001",
        code="print('gpu')\n",
        is_buggy=False,
        node_id="node-gpu",
        score=None,
        created_at="2026-06-10T00:01:00",
        activate=False,
        repo_root=tmp_path,
    )

    assert cpu_path.name == "legacy-001.py"
    assert gpu_path.name == "legacy-002.py"
    assert cpu_path.read_text() == "print('cpu')\n"
    assert gpu_path.read_text() == "print('gpu')\n"
    manifest = json.loads((gpu_path.parent / "code_manifest.json").read_text())
    entries = manifest["versions"]["legacy"]
    assert entries[0]["gpu"] is False
    assert entries[1]["gpu"] is True


def test_hypothesis_candidates_ignore_mismatched_gpu_root(tmp_path):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.gpu = True
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    journal = Journal()
    node = Node(code="print('cpu')", plan="generated")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000001"]
    node.research_runtime_config = {"gpu": False}
    node.status = "generated"
    journal.append(node)

    candidates = research.hypothesis_candidates_for_node(
        cfg,
        journal=journal,
        parent_node=None,
        repo_root=tmp_path,
    )

    assert [candidate.id for candidate in candidates] == ["000001"]


def test_reviewed_generated_hypothesis_root_updates_manifest_score(
    tmp_path,
    monkeypatch,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    monkeypatch.setattr(
        "aide.agent.save_hypothesis_root_code",
        lambda _cfg, **kwargs: research.save_hypothesis_root_code(
            _cfg,
            **kwargs,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('generated root')\n", plan="root")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000001"]

    agent.save_hypothesis_root_code_for_node(node, activate=False)
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    assert manifest["versions"]["legacy"][0]["status"] == "generated"
    assert manifest["versions"]["legacy"][0]["score"] is None

    agent.review_node(
        node,
        ExecutionResult(
            term_out=[
                'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
                '"metric": 0.94783, "lower_is_better": false, '
                '"research_hypotheses_llm_claimed_used": ["000001"]}'
            ],
            exec_time=1.0,
            exc_type=None,
        ),
    )

    assert list(hypothesis_dir.glob("legacy-*.py")) == [hypothesis_dir / "legacy-001.py"]
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    entry = manifest["versions"]["legacy"][0]
    assert entry["file"] == "legacy-001.py"
    assert entry["buggy"] is False
    assert entry["node_id"] == node.id
    assert entry["score"] == 0.94783
    assert entry["exec_time"] == 1.0
    assert manifest["active"]["legacy"] == "legacy-001.py"


def test_hypothesis_root_manifest_entry_records_aux_flag(tmp_path, monkeypatch):
    cfg = _manual_cfg(tmp_path)
    cfg.agent.aux = True
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    node = _node(0.91, code="print('new root')\n", plan="plan")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000001"]
    node.ctime = 1000.0

    research.save_hypothesis_root_code(
        cfg,
        hypothesis_id="000001",
        code=node.code,
        is_buggy=False,
        node_id=node.id,
        score=0.91,
        created_at="2026-05-25T00:00:00",
        repo_root=tmp_path,
    )

    manifest = json.loads(
        (
            tmp_path
            / "research_hypotheses"
            / "playground-series-s6e5"
            / "000001"
            / "code_manifest.json"
        ).read_text()
    )
    assert manifest["versions"]["legacy"][0]["aux"] is True


def test_hypothesis_root_manifest_update_preserves_source_provenance(tmp_path):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("print('promoted')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "active": {"legacy": "legacy-001.py"},
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": False,
                            "score": 0.9,
                            "source_run_id": "run-1",
                            "source_node_id": "branch-a",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    node = _node(0.92, code="print('promoted')\n", plan="plan")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000001"]

    research.save_hypothesis_root_code(
        cfg,
        hypothesis_id="000001",
        code=node.code,
        is_buggy=False,
        node_id=node.id,
        score=0.92,
        created_at="2026-05-25T00:00:00",
        repo_root=tmp_path,
    )

    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    entry = manifest["versions"]["legacy"][0]
    assert entry["score"] == 0.92
    assert entry["source_run_id"] == "run-1"
    assert entry["source_node_id"] == "branch-a"


def test_generated_hypothesis_root_can_be_saved_without_activating(
    tmp_path,
    monkeypatch,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    monkeypatch.setattr(
        "aide.agent.save_hypothesis_root_code",
        lambda _cfg, **kwargs: research.save_hypothesis_root_code(
            _cfg,
            **kwargs,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('generated root')\n", plan="root")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000001"]

    agent.save_hypothesis_root_code_for_node(node, activate=False)

    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    assert (hypothesis_dir / "legacy-001.py").read_text() == "print('generated root')\n"
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    entry = manifest["versions"]["legacy"][0]
    assert entry["buggy"] is None
    assert entry["status"] == "generated"
    assert entry["score"] is None
    assert manifest.get("active", {}).get("legacy") is None

    root_code = research.load_hypothesis_root_code(
        cfg,
        "000001",
        repo_root=tmp_path,
    )

    assert root_code is not None
    assert root_code.path.name == "legacy-001.py"

    agent.review_node(
        node,
        ExecutionResult(
            term_out=[
                'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
                '"metric": 0.94783, "lower_is_better": false, '
                '"research_hypotheses_llm_claimed_used": ["000001"]}'
            ],
            exec_time=1.0,
            exc_type=None,
        ),
    )

    assert list(hypothesis_dir.glob("legacy-*.py")) == [hypothesis_dir / "legacy-001.py"]
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    entry = manifest["versions"]["legacy"][0]
    assert entry["buggy"] is False
    assert entry["status"] == "ok"
    assert entry["score"] == 0.94783
    assert manifest["active"]["legacy"] == "legacy-001.py"


def test_reviewed_llm_hypothesis_root_after_buggy_highest_saves_next_version(
    tmp_path,
    monkeypatch,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("print('old ok')\n")
    (hypothesis_dir / "legacy-002.py").write_text("raise RuntimeError('bug')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": False,
                        },
                        {
                            "file": "legacy-002.py",
                            "buggy": True,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "aide.agent.save_hypothesis_root_code",
        lambda _cfg, **kwargs: research.save_hypothesis_root_code(
            _cfg,
            **kwargs,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    node = Node(code="print('fixed root')\n", plan="root")
    node.research_mode = "hypothesis"
    node.research_hypotheses_offered = ["000001"]

    agent.review_node(
        node,
        ExecutionResult(
            term_out=[
                'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
                '"metric": 0.92, "lower_is_better": false, '
                '"research_hypotheses_llm_claimed_used": ["000001"]}'
            ],
            exec_time=1.0,
            exc_type=None,
        ),
    )

    assert (hypothesis_dir / "legacy-003.py").read_text() == "print('fixed root')\n"
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    assert manifest["active"]["legacy"] == "legacy-003.py"
    assert [entry["file"] for entry in manifest["versions"]["legacy"]] == [
        "legacy-001.py",
        "legacy-002.py",
        "legacy-003.py",
    ]


def test_reviewed_buggy_root_repair_saves_next_root_version(tmp_path, monkeypatch):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("raise RuntimeError('bug')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": True,
                            "node_id": "bug-root",
                            "score": None,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "aide.agent.save_hypothesis_root_code",
        lambda _cfg, **kwargs: research.save_hypothesis_root_code(
            _cfg,
            **kwargs,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    root = Node(code="raise RuntimeError('bug')\n", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000001"]
    root.is_buggy = True
    child = Node(code="print('fixed root')\n", plan="fix", parent=root)
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["000001"]

    agent.review_node(
        child,
        ExecutionResult(
            term_out=[
                'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
                '"metric": 0.92, "lower_is_better": false, '
                '"research_hypotheses_llm_claimed_used": ["000001"]}'
            ],
            exec_time=1.0,
            exc_type=None,
        ),
    )

    assert (hypothesis_dir / "legacy-001.py").read_text() == (
        "raise RuntimeError('bug')\n"
    )
    assert (hypothesis_dir / "legacy-002.py").read_text() == "print('fixed root')\n"
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    assert manifest["active"]["legacy"] == "legacy-002.py"
    assert [entry["file"] for entry in manifest["versions"]["legacy"]] == [
        "legacy-001.py",
        "legacy-002.py",
    ]


def test_reviewed_status_bug_root_repair_saves_next_root_version(
    tmp_path,
    monkeypatch,
):
    cfg = _manual_cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    (hypothesis_dir / "legacy-001.py").write_text("raise RuntimeError('bug')\n")
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": True,
                            "node_id": "bug-root",
                            "score": None,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "aide.agent.save_hypothesis_root_code",
        lambda _cfg, **kwargs: research.save_hypothesis_root_code(
            _cfg,
            **kwargs,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    root = Node(code="raise RuntimeError('bug')\n", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000001"]
    root.status = "bug"
    root.is_buggy = False
    child = Node(code="print('fixed root')\n", plan="fix", parent=root)
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["000001"]

    agent.review_node(
        child,
        ExecutionResult(
            term_out=[
                'AIDE_RESULT_JSON: {"is_bug": false, "summary": "ok", '
                '"metric": 0.92, "lower_is_better": false, '
                '"research_hypotheses_llm_claimed_used": ["000001"]}'
            ],
            exec_time=1.0,
            exc_type=None,
        ),
    )

    assert (hypothesis_dir / "legacy-002.py").read_text() == "print('fixed root')\n"


def test_branch_hypothesis_node_does_not_save_library_root(tmp_path, monkeypatch):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    monkeypatch.setattr(
        "aide.agent.save_hypothesis_root_code",
        lambda _cfg, **kwargs: research.save_hypothesis_root_code(
            _cfg,
            **kwargs,
            repo_root=tmp_path,
        ),
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    parent = Node(code="print('parent')", plan="parent")
    child = Node(code="print('branch')", plan="branch", parent=parent)
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["000001"]
    child.is_buggy = False

    agent._save_reviewed_hypothesis_root_code(child)

    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    assert not list(hypothesis_dir.glob("legacy-*.py"))


def test_buggy_duplicate_root_does_not_deactivate_existing_manifest(tmp_path):
    cfg = _manual_cfg(tmp_path)
    _write_manual_hypothesis(tmp_path, "playground-series-s6e5", "000001")
    path = research.save_hypothesis_root_code(
        cfg,
        hypothesis_id="000001",
        code="print('same root')\n",
        is_buggy=False,
        node_id="ok-node",
        score=0.91,
        repo_root=tmp_path,
    )

    duplicate = research.save_hypothesis_root_code(
        cfg,
        hypothesis_id="000001",
        code="print('same root')\n",
        is_buggy=True,
        node_id="bug-node",
        score=None,
        repo_root=tmp_path,
    )

    assert duplicate == path
    manifest = json.loads((path.parent / "code_manifest.json").read_text())
    assert manifest["active"]["legacy"] == "legacy-001.py"
    assert manifest["versions"]["legacy"][0]["buggy"] is False
    assert manifest["versions"]["legacy"][0]["node_id"] == "ok-node"


def test_agent_exposes_active_hypothesis_log_hint_during_generation(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    selection = research.ManualHypothesisSelection(
        completed_steps=0,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000122",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Rival-relative pit-wave features",
                summary="Use current-lap peer pit context.",
                rationale="Pit decisions often react to nearby rivals.",
                implementation_hint="Add current-lap rival aggregate features.",
                expected_effect="Improves reactive stop timing signal.",
                risk="Avoid future laps and target-derived aggregates.",
                sources=[],
                path=tmp_path / "hypothesis-000122.json",
            )
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    captured_hint = {}

    def fake_plan_and_code(_prompt):
        captured_hint["value"] = agent.active_research_hypothesis_log_hint
        captured_hint["hypothesis_id"] = agent.active_research_hypothesis_id
        return "I will verify hypothesis 000122.", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    hint = captured_hint["value"]
    assert "Hypothesis 000122" in hint
    assert "Title: Rival-relative pit-wave features" in hint
    assert "Summary: Use current-lap peer pit context." in hint
    assert "Try: Add current-lap rival aggregate features." in hint
    assert captured_hint["hypothesis_id"] == "000122"


def test_agent_generates_under_forced_root_when_search_returns_none(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    cfg.agent.search.forced_root = "000405"
    root = _node(0.95232, code="print('root')", plan="root")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000405"]
    journal = Journal()
    journal.append(root)
    captured = {}
    selection = research.ManualHypothesisSelection(
        completed_steps=1,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000941",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Use stronger tyre-age interactions",
                summary="Add tyre-age interaction features.",
                rationale="Pit timing depends on tyre degradation.",
                implementation_hint="Cross tyre age with stint context.",
                expected_effect="Better pit probability ranking.",
                risk="May overfit rare strategy windows.",
                sources=[],
                path=tmp_path / "hypothesis-000941.json",
            )
        ],
    )

    def fake_select(*_args, **kwargs):
        captured["parent_node"] = kwargs["parent_node"]
        return selection

    monkeypatch.setattr("aide.agent.select_hypothesis_for_node", fake_select)
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(_prompt):
        return "I will verify hypothesis 000941.", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    node = agent.generate_node(None)

    assert captured["parent_node"] is root
    assert node.parent is root
    assert node.research_mode == "hypothesis"
    assert node.research_hypotheses_offered == ["000941"]


def test_hypothesis_log_hint_normalizes_literal_json_newlines(tmp_path):
    selection = research.ManualHypothesisSelection(
        completed_steps=10,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000515",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Template distance",
                summary="First sentence.\\nSecond sentence.",
                rationale="Strategy templates.",
                implementation_hint="Use prior stops only.\\nAdd template distance.",
                expected_effect="Better planned-stop timing.",
                risk="Avoid fold leakage.",
                sources=[],
                path=tmp_path / "hypothesis-000515.json",
            )
        ],
    )

    hint = research.format_hypothesis_for_log_panel(selection)

    assert "\\n" not in hint
    assert "Summary: First sentence. Second sentence." in hint
    assert "Try: Use prior stops only. Add template distance." in hint


def test_hypothesis_root_prompt_omits_global_memory_and_branch_context(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    previous = _node(
        0.99,
        code="print('unrelated')",
        plan="Unrelated global winner",
    )
    journal = Journal()
    journal.append(previous)
    selection = research.ManualHypothesisSelection(
        completed_steps=1,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000122",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Rival-relative pit-wave features",
                summary="Use current-lap peer pit context.",
                rationale="Pit decisions often react to nearby rivals.",
                implementation_hint="Add current-lap rival aggregate features.",
                expected_effect="Improves reactive stop timing signal.",
                risk="Avoid future laps and target-derived aggregates.",
                sources=[],
                path=tmp_path / "hypothesis-000122.json",
            )
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "I will verify hypothesis 000122.", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    assert "Memory" not in captured["prompt"]
    assert "Branch context" not in captured["prompt"]
    assert "Hypothesis under verification" in captured["prompt"]


def test_preselected_hypothesis_root_prompt_omits_global_memory(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "llm"
    cfg.agent.data_preview = False
    previous = _node(
        0.99,
        code="print('unrelated')",
        plan="Unrelated global winner",
    )
    journal = Journal()
    journal.append(previous)
    selection = research.ManualHypothesisSelection(
        completed_steps=1,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000015",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Assigned feature hypothesis",
                summary="Add fold-safe categorical and binned features.",
                rationale="This validates the selected root feature family.",
                implementation_hint="Build categorical bins inside each fold.",
                expected_effect="May improve balanced accuracy.",
                risk="Avoid target leakage in encodings.",
                sources=[],
                path=tmp_path / "hypothesis-000015.json",
            )
        ],
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "I will verify the assigned feature hypothesis.", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent.generate_preselected_hypothesis_root(
        selection,
        node_ctime=1.0,
        llm_log_dir=tmp_path / "artifacts" / "node",
        artifact_dir_name="node",
    )

    assert "Memory" not in captured["prompt"]
    assert "Branch context" not in captured["prompt"]
    assert "Hypothesis under verification" in captured["prompt"]
    sketch_guideline = captured["prompt"]["Instructions"]["Solution sketch guideline"]
    assert not any("Memory section" in item for item in sketch_guideline)


def test_hypothesis_child_prompt_uses_branch_context_not_global_memory(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    root = _node(0.91, code="root code", plan="Root branch plan")
    root.research_mode = "hypothesis"
    root.research_hypotheses_offered = ["000101"]
    child = _node(0.92, code="child code", plan="Child branch plan")
    child.parent = root
    root.children.add(child)
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["000202"]
    unrelated = _node(0.99, code="other code", plan="Unrelated global winner")
    unrelated.research_mode = "hypothesis"
    unrelated.research_hypotheses_offered = ["000999"]
    journal = Journal()
    journal.append(root)
    journal.append(child)
    journal.append(unrelated)
    selection = research.ManualHypothesisSelection(
        completed_steps=3,
        source_hash="sha256:test",
        source_dir=tmp_path,
        hypotheses=[
            research.ManualHypothesis(
                id="000303",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Assigned child hypothesis",
                summary="Add one assigned child change.",
                rationale="This validates the selected hypothesis.",
                implementation_hint="Add the assigned feature block.",
                expected_effect="May improve ROC AUC.",
                risk="Keep it leakage-safe.",
                sources=[],
                path=tmp_path / "hypothesis-000303.json",
            )
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "I will verify hypothesis 000303.", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(child)

    assert "Memory" not in captured["prompt"]
    assert "Branch context" in captured["prompt"]
    branch_context = captured["prompt"]["Branch context"]
    assert "Branch path:\n000101 -> 000202" in branch_context
    assert "Ancestor 1 / root:" in branch_context
    assert "Ancestor 2 / direct parent:" in branch_context
    assert "Unrelated global winner" not in branch_context
    assert "Hypothesis under verification" in captured["prompt"]
    assert "Hypothesis ID: 000303" in captured["prompt"]["Hypothesis under verification"]


def test_hypothesis_child_prompt_includes_legacy_reference_code(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.data_preview = False
    task_slug = Path(cfg.data_dir).name
    _write_manual_hypothesis(tmp_path, task_slug, "000303")
    hypothesis_dir = tmp_path / "research_hypotheses" / task_slug / "000303"
    (hypothesis_dir / "legacy-001.py").write_text(
        "def helper_from_assigned_hypothesis():\n"
        "    return 'reference legacy code'\n",
        encoding="utf-8",
    )
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "active": {"legacy": "legacy-001.py"},
                "versions": {
                    "legacy": [
                        {
                            "file": "legacy-001.py",
                            "buggy": False,
                            "score": 0.91,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    parent = _node(0.91, code="print('parent')", plan="parent")
    journal = Journal()
    journal.append(parent)
    selection = research.ManualHypothesisSelection(
        completed_steps=1,
        source_hash="sha256:test",
        source_dir=tmp_path / "research_hypotheses" / task_slug,
        hypotheses=[
            research.ManualHypothesis(
                id="000303",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Grouped validation",
                summary="Use grouped validation.",
                rationale="Race/year grouping should reduce mismatch.",
                implementation_hint="Build Race_Year groups.",
                expected_effect="Better validation stability.",
                risk="Grouped CV may be pessimistic.",
                sources=[],
                path=hypothesis_dir / "hypothesis-000303.json",
            )
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "I will verify hypothesis 000303.", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(parent)

    reference = captured["prompt"]["Reference implementation for assigned hypothesis"]
    assert "stored implementation of the newly assigned hypothesis 000303" in reference
    assert "Previous solution" in reference
    assert "optional implementation context" in reference
    assert "change the model family" in reference
    assert "Do not replace the parent training loop" not in reference
    assert "helper_from_assigned_hypothesis" in reference
    assert "reference legacy code" in reference


def test_hypothesis_child_prompt_includes_autogluon_preprocess_reference_only(
    tmp_path,
    monkeypatch,
):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "hypothesis"
    cfg.agent.mode = AGENT_MODE
    cfg.agent.data_preview = False
    task_slug = Path(cfg.data_dir).name
    _write_manual_hypothesis(tmp_path, task_slug, "000303")
    hypothesis_dir = tmp_path / "research_hypotheses" / task_slug / "000303"
    (hypothesis_dir / "autogluon-001.py").write_text(
        build_autogluon_wrapper(
            "def preprocess(df):\n"
            "    df = df.copy()\n"
            "    df['assigned_feature'] = 1\n"
            "    return df\n",
            cfg,
        ),
        encoding="utf-8",
    )
    (hypothesis_dir / "code_manifest.json").write_text(
        json.dumps(
            {
                "active": {"autogluon": "autogluon-001.py"},
                "versions": {
                    "autogluon": [
                        {
                            "file": "autogluon-001.py",
                            "buggy": False,
                            "score": 0.91,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    parent = _node(
        0.91,
        code=build_autogluon_wrapper("def preprocess(df):\n    return df\n", cfg),
        plan="parent",
    )
    journal = Journal()
    journal.append(parent)
    selection = research.ManualHypothesisSelection(
        completed_steps=1,
        source_hash="sha256:test",
        source_dir=tmp_path / "research_hypotheses" / task_slug,
        hypotheses=[
            research.ManualHypothesis(
                id="000303",
                enabled=True,
                agent_modes=["legacy", "autogluon"],
                title="Grouped validation",
                summary="Use grouped validation.",
                rationale="Race/year grouping should reduce mismatch.",
                implementation_hint="Build Race_Year groups.",
                expected_effect="Better validation stability.",
                risk="Grouped CV may be pessimistic.",
                sources=[],
                path=hypothesis_dir / "hypothesis-000303.json",
            )
        ],
    )
    monkeypatch.setattr(
        "aide.agent.select_hypothesis_for_node",
        lambda *_args, **_kwargs: selection,
    )
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return (
            "I will verify hypothesis 000303.",
            "def preprocess(df):\n    return df\n",
        )

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(parent)

    reference = captured["prompt"]["Reference implementation for assigned hypothesis"]
    assert "stored implementation of the newly assigned hypothesis 000303" in reference
    assert "optional implementation context" in reference
    assert "change the model family" in reference
    assert "Do not replace the parent training loop" not in reference
    assert "assigned_feature" in reference
    assert "def preprocess" in reference
    assert "TabularPredictor" not in reference
    assert "AIDE_AG_CONFIG" not in reference


def test_non_hypothesis_draft_prompt_keeps_global_memory(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.research.mode = "llm"
    cfg.agent.data_preview = False
    previous = _node(
        0.95,
        code="print('previous')",
        plan="Previous global design",
    )
    journal = Journal()
    journal.append(previous)
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    assert "Memory" in captured["prompt"]
    assert "Previous global design" in captured["prompt"]["Memory"]
    assert "Branch context" not in captured["prompt"]


def test_legacy_agent_gpu_prompt_is_opt_in(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.data_preview = False
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    default_guidelines = captured["prompt"]["Instructions"]["Implementation guideline"]
    assert not any("CUDA-capable NVIDIA GPU" in line for line in default_guidelines)

    cfg.agent.gpu = True
    captured.clear()
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    gpu_guidelines = captured["prompt"]["Instructions"]["Implementation guideline"]
    assert any(
        "CUDA-capable NVIDIA GPU" in line
        and "Use GPU-enabled training" in line
        for line in gpu_guidelines
    )
    assert any(
        "Do not silently switch a GPU-capable model to CPU" in line
        for line in gpu_guidelines
    )
    assert not any(line.startswith("If a GPU-specific") for line in gpu_guidelines)
    assert not any("If you cannot isolate" in line for line in gpu_guidelines)
    assert any('task_type="GPU"' in line for line in gpu_guidelines)
    assert any(
        'tree_method="hist"' in line and 'device="cuda"' in line
        for line in gpu_guidelines
    )
    assert any(
        "For LightGBM" in line
        and "try GPU training first" in line
        and "CPU LightGBM fallback is allowed only after" in line
        and "previous failed implementation for this same hypothesis" in line
        for line in gpu_guidelines
    )
    assert any(
        'device_type="cuda"' in line
        and 'device="cuda"' in line
        for line in gpu_guidelines
    )
    assert any(
        "falls back from GPU to CPU" in line and "print" in line
        for line in gpu_guidelines
    )


def test_legacy_agent_prompt_forbids_data_directory_discovery(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.data_preview = False
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    guidelines = captured["prompt"]["Instructions"]["Implementation guideline"]
    contract = "\n".join(guidelines)
    assert "aide_solution_helpers" in contract
    assert "load_competition_data" in contract
    assert "working_dir" in contract
    assert "aide_stage" in contract
    assert "log_stage" in contract
    assert 'with aide_stage("fit_predict_fold_stage")' in contract
    assert "Before each long model fit" in contract
    assert "flush=True" in contract
    assert "write_submission" in contract
    assert "write_validation_predictions" in contract
    assert "train, test, sample_sub = load_competition_data()" in contract
    assert "do not call `to_csv()`" in contract
    assert "submission.csv" in contract
    assert "oof_predictions.csv.gz" in contract
    assert "test_predictions.csv.gz" in contract
    assert "validation_predictions.csv.gz" in contract
    assert "write each required artifact at most once" in contract
    assert "call `write_test_predictions(...)` exactly once" in contract
    assert "Do not read train/test/sample_submission manually" in contract
    assert "data-directory discovery code" in contract
    assert "find_data_dir()" in contract
    assert "Path.cwd()" in contract
    assert "../input" in contract
    assert "logs/" in contract
    assert "workspaces/" in contract


def test_agent_prompt_only_lists_importable_packages(tmp_path, monkeypatch):
    available = {"numpy", "pandas", "sklearn", "catboost"}
    monkeypatch.setattr(
        agent_module,
        "find_spec",
        lambda name: object() if name in available else None,
    )
    cfg = _cfg(tmp_path)
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    installed = agent._prompt_environment["Installed Packages"]

    assert "`numpy`" in installed
    assert "`pandas`" in installed
    assert "`scikit-learn`" in installed
    assert "`catboost`" in installed
    assert "`torch`" not in installed
    assert "PyTorch" not in installed
    assert "all packages are already installed" not in installed


def test_legacy_agent_prompt_omits_auc_blend_guideline(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.data_preview = False
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    guidelines = captured["prompt"]["Instructions"]["Implementation guideline"]
    assert not any("roc_auc_score" in line for line in guidelines)
    assert not any("AUC/ROC-AUC" in line for line in guidelines)


def test_legacy_agent_prompt_prefers_behavior_preserving_optimization(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.data_preview = False
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._draft()

    response_format = captured["prompt"]["Instructions"]["Response format"]
    assert "Your response must contain exactly" in response_format
    assert "Exactly one markdown Python code block" in response_format
    assert "Do not include headings" in response_format

    guidelines = captured["prompt"]["Instructions"]["Implementation guideline"]
    assert any(
        "same validation protocol as the parent solution" in line
        for line in guidelines
    )
    assert any(
        "prefer 5-fold stratified CV" in line
        for line in guidelines
    )
    assert not any(
        "minimal semantic patch over the parent solution" in line
        for line in guidelines
    )
    assert not any(
        "Do not replace the parent training loop" in line
        for line in guidelines
    )
    assert any(
        "Same-lap covariate aggregates may use all rows available at prediction time" in line
        for line in guidelines
    )
    assert any(
        "Mechanical simplifications are allowed only" in line
        for line in guidelines
    )


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


def test_agent_legacy_debug_prompt_calls_out_timeout_as_efficiency_failure(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.agent.data_preview = False
    cfg.exec.timeout = 1800
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    parent = _node(None, code="while True: pass", plan="slow")
    parent.exc_type = "TimeoutError"
    parent._term_out = ["TimeoutError: Execution exceeded the time limit of 30 minutes"]

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._debug(parent)

    timeout_guideline = captured["prompt"]["Instructions"]["Timeout fix guideline"]
    assert any("exceeded the execution timeout" in line for line in timeout_guideline)
    assert any("runtime efficiency failure" in line for line in timeout_guideline)
    assert any("Do not assume a specific failing operation" in line for line in timeout_guideline)


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


def test_agent_includes_data_overview_in_improve_prompt(tmp_path):
    cfg = _cfg(tmp_path)
    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=Journal())
    agent.data_preview = "Position (int64) has range: 1.00 - 20.00\n"
    parent = _node(0.94, code="print('baseline')", plan="baseline")

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(parent)

    assert "Data Overview" in captured["prompt"]
    assert "Position" in captured["prompt"]["Data Overview"]


def test_standard_improve_prompt_includes_prior_child_attempts(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = False
    cfg.agent.data_preview = False
    parent = _node(
        0.954674,
        code="print('strong parent')",
        plan="two-seed RealMLP plus CatBoost parent",
    )
    near_copy = Node(
        code="print('rank blend')",
        plan="Add rank blend mode to the existing blend candidates.",
        parent=parent,
    )
    near_copy.metric = MetricValue(0.954672, maximize=True)
    near_copy.is_buggy = False
    near_copy.analysis = "rank blend did not improve"
    worse = Node(
        code="print('remove features')",
        plan="Remove count encoding features.",
        parent=parent,
    )
    worse.metric = MetricValue(0.953900, maximize=True)
    worse.is_buggy = False
    worse.analysis = "feature removal hurt validation"
    journal = Journal()
    for node in [parent, near_copy, worse]:
        journal.append(node)

    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(parent)

    attempts = captured["prompt"]["Previous attempts from this parent"]
    assert "These direct children already tried changes on the same parent" in attempts
    assert "Do not repeat these attempted changes" in attempts
    assert "Add rank blend mode" in attempts
    assert "Remove count encoding features" in attempts
    assert "0.954672" in attempts
    assert "did_not_improve" in attempts


def test_standard_improve_prompt_omits_prior_child_attempts_without_children(
    tmp_path,
):
    cfg = _cfg(tmp_path)
    cfg.research.enabled = False
    cfg.agent.data_preview = False
    parent = _node(0.954674, code="print('strong parent')", plan="parent")
    journal = Journal()
    journal.append(parent)

    captured = {}
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)

    def fake_plan_and_code(prompt):
        captured["prompt"] = prompt
        return "plan", "print('ok')"

    agent.plan_and_code_query = fake_plan_and_code  # type: ignore[method-assign]

    agent._improve(parent)

    assert "Previous attempts from this parent" not in captured["prompt"]
