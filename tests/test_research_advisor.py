import datetime as dt
import json
import re
import subprocess
from pathlib import Path

import aide.agent as agent_module
import aide.research as research
from aide.agent import Agent
from aide.autogluon_preprocess import AGENT_MODE, build_autogluon_wrapper
from aide.interpreter import ExecutionResult
from aide.journal import Journal, Node
from aide.run import (
    ParallelRootJob,
    generate_reserved_hypothesis_root,
)
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
    assert "Sources: https://example.com/grouped-validation" in rendered


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


def test_format_hypothesis_for_prompt_is_hard_contract_without_source_hash(tmp_path):
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
    assert "research_hypotheses_llm_claimed_used" in rendered
    assert "Research source hash" not in rendered


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
    assert command[command.index("--model") + 1] == "gpt-5.4-mini"
    assert 'model_reasoning_effort="low"' in command
    assert seen["stdin"].startswith(RESEARCH_PROMPT_INTRO)
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


def test_hypothesis_root_code_loader_uses_highest_numbered_file(tmp_path):
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
    assert root_code.path.name == "autogluon-002.py"
    assert root_code.code == "print('two')\n"


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

    agent.save_hypothesis_root_code_for_node(node)

    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    assert (hypothesis_dir / "legacy-001.py").read_text() == "print('generated root')\n"
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
    assert manifest["active"]["legacy"] == "legacy-001.py"


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

    agent.save_hypothesis_root_code_for_node(node)
    hypothesis_dir = (
        tmp_path / "research_hypotheses" / "playground-series-s6e5" / "000001"
    )
    manifest = json.loads((hypothesis_dir / "code_manifest.json").read_text())
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
    assert any("CUDA-capable NVIDIA GPU" in line for line in gpu_guidelines)
    assert any('task_type="GPU"' in line for line in gpu_guidelines)
    assert any('device="cuda"' in line for line in gpu_guidelines)
    assert any('device_type="gpu"' in line for line in gpu_guidelines)
    assert any('device_type="cuda"' in line for line in gpu_guidelines)


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


def test_legacy_agent_prompt_parallelizes_expensive_blend_search(tmp_path):
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
    assert any("blend-weight search" in line for line in guidelines)
    assert any("joblib.Parallel" in line for line in guidelines)
    assert any("n_jobs=min(16, os.cpu_count() or 1)" in line for line in guidelines)
    assert any("prefer=\"threads\"" in line for line in guidelines)
    assert any("Evaluating N blend candidates with M workers" in line for line in guidelines)


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

    guidelines = captured["prompt"]["Instructions"]["Implementation guideline"]
    assert any(
        "reducing code size, memory use, or runtime" in line
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
