import os
import time
import datetime as dt
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aide.agent import Agent
from aide.journal import Journal, Node
from aide.run import (
    allocate_node_artifact_slot,
    ensure_node_artifact_slot,
    find_latest_run_id,
    load_resume_state,
    mark_node_generated_only,
    next_generated_only_node,
    parse_runtime_args,
    parse_resume_args,
    ParallelRootFailureState,
    record_generated_only_node,
    recover_generated_only_root_artifacts,
    validate_hypothesis_root_generate_workers,
)
from aide.utils.config import _load_cfg, prep_cfg, save_run
from aide.utils.node_artifacts import node_artifact_dir, node_artifact_submission_path
from aide.utils.metric import MetricValue
from aide.utils import serialize


def _write_run(tmp_path: Path, run_id: str, *, steps: int, mtime: float) -> None:
    log_dir = tmp_path / "logs" / run_id
    workspace_dir = tmp_path / "workspaces" / run_id
    (workspace_dir / "input").mkdir(parents=True)
    (workspace_dir / "working").mkdir(parents=True)

    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspaces")
    cfg.exp_name = run_id.split("-", 1)[1]
    cfg.agent.steps = steps
    cfg = prep_cfg(cfg)
    cfg.exp_name = run_id
    cfg.log_dir = log_dir
    cfg.workspace_dir = workspace_dir

    journal = Journal()
    node = Node(code="print('ok')", plan="ok")
    node.metric = MetricValue(0.9, maximize=True)
    node.is_buggy = False
    node._term_out = ["score 0.9"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ok"
    journal.append(node)
    save_run(cfg, journal, current_node=node)
    os.utime(log_dir / "journal.json", (mtime, mtime))


def test_parse_resume_args_accepts_run_id_and_keeps_omegaconf_overrides():
    resume, remaining = parse_resume_args(
        [
            "--resume",
            "2-judicious-unbreakable-hoatzin",
            "agent.steps=200",
            "generate_report=False",
        ]
    )

    assert resume.requested is True
    assert resume.run_id == "2-judicious-unbreakable-hoatzin"
    assert resume.use_latest is False
    assert remaining == ["agent.steps=200", "generate_report=False"]


def test_parse_resume_args_without_run_id_uses_latest_and_keeps_next_override():
    resume, remaining = parse_resume_args(["--resume", "agent.steps=200"])

    assert resume.requested is True
    assert resume.run_id is None
    assert resume.use_latest is True
    assert remaining == ["agent.steps=200"]


def test_parse_runtime_args_extracts_submission_flags_from_omegaconf_overrides():
    resume, runtime, remaining = parse_runtime_args(
        [
            "--resume",
            "2-example-run",
            "--show-invalid-submission-branches",
            "--force-check-submissions",
            "--telegram-test-message",
            "--debug",
            "agent.steps=200",
        ]
    )

    assert resume.run_id == "2-example-run"
    assert runtime.show_invalid_submission_branches is True
    assert runtime.force_check_submissions is True
    assert runtime.telegram_test_message is True
    assert runtime.debug is True
    assert remaining == ["agent.steps=200"]


def test_parse_runtime_args_extracts_skip_execution_flag():
    resume, runtime, remaining = parse_runtime_args(
        [
            "--resume",
            "2-example-run",
            "--skip-execution",
            "agent.steps=200",
        ]
    )

    assert resume.run_id == "2-example-run"
    assert runtime.skip_execution is True
    assert remaining == ["agent.steps=200"]


@pytest.mark.parametrize("workers", [0, -1, 9, "four"])
def test_validate_hypothesis_root_generate_workers_rejects_invalid_values(workers):
    cfg = _load_cfg(use_cli_args=False)
    cfg.research.hypothesis_root_generate_workers = workers

    with pytest.raises(ValueError, match="hypothesis_root_generate_workers"):
        validate_hypothesis_root_generate_workers(cfg)


def test_generated_only_nodes_are_pending_until_evaluated():
    journal = Journal()
    executed = Node(code="print('ok')", plan="ok")
    executed.metric = MetricValue(0.9, maximize=True)
    executed.is_buggy = False
    pending = Node(code="print('later')", plan="later")
    journal.append(executed)
    journal.append(pending)

    mark_node_generated_only(pending)

    assert pending.status == "generated"
    assert pending.is_buggy is False
    assert pending.metric is None
    assert next_generated_only_node(journal) is pending

    pending.status = "ok"
    assert next_generated_only_node(journal) is None


def test_generated_only_selection_respects_forced_root_scope():
    journal = Journal()
    outside_root = Node(code="print('outside')", plan="outside")
    outside_root.research_mode = "hypothesis"
    outside_root.research_hypotheses_offered = ["000111"]
    forced_root = Node(code="print('forced')", plan="forced")
    forced_root.research_mode = "hypothesis"
    forced_root.research_hypotheses_offered = ["000222"]
    journal.append(outside_root)
    journal.append(forced_root)

    mark_node_generated_only(outside_root)
    mark_node_generated_only(forced_root)

    assert next_generated_only_node(journal, forced_root="000222") is forced_root


def test_generated_only_selection_respects_forced_hypothesis_exact_id():
    journal = Journal()
    forced_root = Node(code="print('forced')", plan="forced")
    forced_root.research_mode = "hypothesis"
    forced_root.research_hypotheses_offered = ["000941"]
    child = Node(code="print('child')", plan="child", parent=forced_root)
    child.research_mode = "hypothesis"
    child.research_hypotheses_offered = ["000365"]
    journal.append(forced_root)
    journal.append(child)

    mark_node_generated_only(forced_root)
    mark_node_generated_only(child)

    assert (
        next_generated_only_node(journal, forced_hypothesis="000941") is forced_root
    )


def test_record_generated_only_node_marks_saves_and_appends():
    journal = Journal()
    node = Node(code="print('generated')", plan="generated")

    class AgentStub:
        saved_node: Node | None = None

        def save_hypothesis_root_code_for_node(self, node: Node) -> None:
            self.saved_node = node

    agent = AgentStub()

    record_generated_only_node(
        agent=agent,
        journal=journal,
        node=node,
        experiment_id="test-run",
    )

    assert node.status == "generated"
    assert journal.nodes == [node]
    assert agent.saved_node is node


def test_recover_generated_only_root_artifacts_materializes_completed_orphans(
    tmp_path,
    monkeypatch,
):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path / "playground-series-s6e5")
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "2-existing-run"
    cfg.workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    cfg.exp_name = "2-existing-run"
    cfg.research.enabled = True
    cfg.research.mode = "hypothesis"
    cfg.agent.mode = "legacy"
    cfg = prep_cfg(cfg)
    cfg.log_dir.mkdir(parents=True)
    cfg.workspace_dir.mkdir(parents=True)

    run_research_dir = cfg.log_dir / "research_hypotheses"
    run_research_dir.mkdir(parents=True)
    (run_research_dir / "offers.jsonl").write_text(
        json.dumps(
            {
                "checkpoint_step": 0,
                "offered": ["000123"],
                "source_hash": "sha256:test",
                "created_at": "2026-05-23T00:00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    artifact_dir = cfg.log_dir / "artifacts" / "20260523T233540-abcdef12"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "context.json").write_text(
        json.dumps(
            {
                "phase": "generate",
                "parent_node_id": None,
                "agent_mode": "legacy",
                "node_ctime": 1779575740.0,
            }
        ),
        encoding="utf-8",
    )
    (artifact_dir / "status.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    (artifact_dir / "request.md").write_text(
        "# Hypothesis under verification\n\n"
        "Hypothesis ID: 000123\n"
        "Implement this exact hypothesis.\n",
        encoding="utf-8",
    )
    (artifact_dir / "response_raw.txt").write_text(
        "Hypothesis ID 000123 recovered plan.\n\n```python\nprint('recovered')\n```\n",
        encoding="utf-8",
    )
    (artifact_dir / "response.py").write_text(
        "print('recovered')\n",
        encoding="utf-8",
    )

    journal = Journal()
    agent = Agent(task_desc="task", cfg=cfg, journal=journal)
    monkeypatch.setattr(agent, "save_hypothesis_root_code_for_node", lambda _node: None)

    recovered = recover_generated_only_root_artifacts(
        cfg=cfg,
        journal=journal,
        agent=agent,
    )

    assert recovered == 1
    assert len(journal.nodes) == 1
    node = journal.nodes[0]
    assert node.status == "generated"
    assert node.code == "print('recovered')\n"
    assert node.plan == "Hypothesis ID 000123 recovered plan."
    assert node.artifact_dir_name == "20260523T233540-abcdef12"
    assert node.research_hypotheses_offered == ["000123"]
    assert node.research_source_hash == "sha256:test"
    usage = json.loads((run_research_dir / "usage.json").read_text())
    assert usage["000123"]["prompt_node_ids"] == [node.id]


def test_generated_only_nodes_are_not_scored_good_candidates():
    journal = Journal()
    executed = Node(code="print('ok')", plan="ok")
    executed.metric = MetricValue(0.9, maximize=True)
    executed.is_buggy = False
    pending = Node(code="print('later')", plan="later")
    journal.append(executed)
    journal.append(pending)

    mark_node_generated_only(pending)

    assert journal.good_nodes == [executed]
    assert journal.get_best_node() is executed
    assert journal.get_best_node(only_good=False) is executed


def test_node_artifact_dir_uses_explicit_name(tmp_path):
    node = Node(
        code="print('x')",
        plan="x",
        ctime=1_779_492_701.0,
        artifact_dir_name="20260523T220603-a1b2c3d4",
    )

    assert node_artifact_dir(tmp_path, node) == (
        tmp_path / "artifacts" / "20260523T220603-a1b2c3d4"
    )
    assert node_artifact_submission_path(tmp_path, node) == (
        tmp_path / "artifacts" / "20260523T220603-a1b2c3d4" / "submission.csv"
    )


def test_node_artifact_dir_falls_back_to_legacy_ctime_timestamp(tmp_path):
    node = Node(code="print('x')", plan="x", ctime=1_779_492_701.0)

    assert node_artifact_dir(tmp_path, node).name == "20260523T013141"


def test_allocate_node_artifact_slot_sets_unique_explicit_name(tmp_path):
    first_ctime, first_dir_name, first_dir = allocate_node_artifact_slot(tmp_path)
    second_ctime, second_dir_name, second_dir = allocate_node_artifact_slot(tmp_path)

    assert first_ctime <= second_ctime
    assert first_dir_name != second_dir_name
    assert first_dir.name == first_dir_name
    assert second_dir.name == second_dir_name
    assert first_dir.exists()
    assert second_dir.exists()
    assert len(first_dir_name.split("-")[-1]) == 8


def test_ensure_node_artifact_slot_assigns_hash_name_to_legacy_node(tmp_path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path)
    cfg.goal = "test goal"
    cfg.log_dir = tmp_path / "logs" / "run"
    cfg = prep_cfg(cfg)
    node = Node(code="print('x')", plan="x", ctime=1_779_492_701.0)

    artifact_dir = ensure_node_artifact_slot(cfg, node)

    assert node.artifact_dir_name is not None
    assert node.artifact_dir_name.startswith("20260523T013141-")
    assert len(node.artifact_dir_name.split("-")[-1]) == 8
    assert artifact_dir == cfg.log_dir / "artifacts" / node.artifact_dir_name
    assert artifact_dir.exists()

    same_dir = ensure_node_artifact_slot(cfg, node)

    assert same_dir == artifact_dir


def test_generation_retry_policy_stops_refill_after_three_failures():
    state = ParallelRootFailureState()

    assert state.record_failure("000405", RuntimeError("network")) is False
    assert state.record_failure("000405", RuntimeError("network")) is False
    assert state.record_failure("000405", RuntimeError("network")) is True
    assert state.stop_refill is True


def test_parse_runtime_args_extracts_seed_options():
    resume, runtime, remaining = parse_runtime_args(
        [
            "--seed-from-sha",
            "abcdef12",
            "--seed-source-run=2-source-run",
            "agent.steps=200",
        ]
    )

    assert resume.requested is False
    assert runtime.seed_sha_prefix == "abcdef12"
    assert runtime.seed_source_run == "2-source-run"
    assert remaining == ["agent.steps=200"]


def test_parse_runtime_args_rejects_resume_with_seed():
    with pytest.raises(ValueError, match="cannot be combined"):
        parse_runtime_args(["--resume", "--seed-from-sha=abcdef12"])


def test_find_latest_run_id_uses_newest_journal_mtime(tmp_path):
    _write_run(tmp_path, "1-old-run", steps=10, mtime=time.time() - 100)
    _write_run(tmp_path, "2-new-run", steps=20, mtime=time.time())

    assert find_latest_run_id(tmp_path / "logs") == "2-new-run"


def test_load_resume_state_uses_existing_paths_and_cli_overrides(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())

    cfg, journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.steps=25", "generate_report=False"],
    )

    assert cfg.exp_name == "2-existing-run"
    assert Path(cfg.log_dir) == tmp_path / "logs" / "2-existing-run"
    assert Path(cfg.workspace_dir) == tmp_path / "workspaces" / "2-existing-run"
    assert cfg.agent.steps == 25
    assert cfg.generate_report is False
    assert len(journal.nodes) == 1
    assert journal.nodes[0].metric.value == 0.9


def test_load_resume_state_clears_saved_forced_root_without_cli_override(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.search.forced_root = "000405"
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.agent.search.forced_root is None


def test_load_resume_state_preserves_unquoted_forced_root_cli_id(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.search.forced_root=000405"],
    )

    assert cfg.agent.search.forced_root == "000405"


def test_load_resume_state_accepts_short_forced_root_cli_alias(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["forced_root=000405"],
    )

    assert cfg.agent.search.forced_root == "000405"


def test_resume_profile_override_clears_old_autogluon_model_list(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.autogluon.included_model_types = ["XGB", "GBM", "CAT"]
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.autogluon.profile=fast_boost"],
    )

    assert cfg.agent.autogluon.profile == "fast_boost"
    assert cfg.agent.autogluon.included_model_types is None


def test_resume_explicit_model_list_override_wins_over_profile(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.autogluon.included_model_types = ["XGB", "GBM", "CAT"]
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[
            "agent.autogluon.profile=fast_boost",
            "agent.autogluon.included_model_types=[CAT]",
        ],
    )

    assert cfg.agent.autogluon.profile == "fast_boost"
    assert cfg.agent.autogluon.included_model_types == ["CAT"]


def test_load_resume_state_persists_submission_contract_revalidation(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    log_dir = tmp_path / "logs" / "2-existing-run"
    workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    (workspace_dir / "input" / "sample_submission.csv").write_text(
        "id,PitNextLap\n1,0.0\n2,0.0\n"
    )

    journal = serialize.load_json(log_dir / "journal.json", Journal)
    timestamp = dt.datetime.fromtimestamp(journal.nodes[0].ctime).strftime(
        "%Y%m%dT%H%M%S"
    )
    artifact_dir = log_dir / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "submission.csv").write_text("id,PitNextLap\n1,0.8\n1,0.9\n")

    _cfg, loaded = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )
    persisted = serialize.load_json(log_dir / "journal.json", Journal)

    assert loaded.nodes[0].is_buggy is True
    assert loaded.nodes[0].metric.value is None
    assert persisted.nodes[0].is_buggy is True
    assert persisted.nodes[0].metric.value is None
    assert persisted.nodes[0].exc_type == "SubmissionValidationError"


def test_load_resume_state_does_not_submission_validate_generated_nodes(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    log_dir = tmp_path / "logs" / "2-existing-run"
    workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    (workspace_dir / "input" / "sample_submission.csv").write_text(
        "id,PitNextLap\n1,0.0\n"
    )

    journal = Journal()
    generated = Node(code="print('generated')", plan="generated")
    journal.append(generated)
    mark_node_generated_only(generated)
    generated.is_buggy = True
    generated.exc_type = "SubmissionValidationError"
    generated.exc_info = {"args": ["missing artifact submission.csv"]}
    generated._term_out = [
        "SubmissionValidationError: missing artifact submission.csv\n"
    ]
    serialize.dump_json(journal, log_dir / "journal.json")

    _cfg, loaded = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )
    persisted = serialize.load_json(log_dir / "journal.json", Journal)

    assert loaded.nodes[0].status == "generated"
    assert loaded.nodes[0].is_buggy is False
    assert loaded.nodes[0].exc_type is None
    assert loaded.nodes[0]._term_out == []
    assert persisted.nodes[0].status == "generated"
    assert persisted.nodes[0].is_buggy is False
    assert persisted.nodes[0].exc_type is None


def test_load_resume_state_force_revalidates_cached_submission(tmp_path, monkeypatch):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    log_dir = tmp_path / "logs" / "2-existing-run"
    workspace_dir = tmp_path / "workspaces" / "2-existing-run"
    (workspace_dir / "input" / "sample_submission.csv").write_text(
        "id,PitNextLap\n1,0.0\n"
    )

    journal = serialize.load_json(log_dir / "journal.json", Journal)
    timestamp = dt.datetime.fromtimestamp(journal.nodes[0].ctime).strftime(
        "%Y%m%dT%H%M%S"
    )
    artifact_dir = log_dir / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True, exist_ok=True)
    submission_path = artifact_dir / "submission.csv"
    submission_path.write_text("id,PitNextLap\n1,0.8\n")
    journal.nodes[0].submission_validation = {
        "status": "ok",
        "sample_signature": {"size": 1, "mtime_ns": 1},
        "submission_signature": {"size": 1, "mtime_ns": 1},
    }
    serialize.dump_json(journal, log_dir / "journal.json")
    calls = []

    def fake_validate(_submission_path, _sample_path):
        calls.append((_submission_path, _sample_path))
        return None

    monkeypatch.setattr("aide.run.validate_submission_file", fake_validate)

    load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
        force_check_submissions=True,
    )

    assert calls


def test_load_resume_state_rejects_missing_workspace(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    OmegaConf.load(tmp_path / "logs" / "2-existing-run" / "config.yaml")
    missing_workspace = tmp_path / "workspaces" / "2-existing-run"
    missing_workspace.rename(tmp_path / "workspaces" / "moved-run")

    with pytest.raises(FileNotFoundError, match="workspace"):
        load_resume_state(
            run_id="2-existing-run",
            top_log_dir=tmp_path / "logs",
            top_workspace_dir=tmp_path / "workspaces",
            cli_overrides=[],
        )


def test_load_resume_state_defaults_research_for_older_configs(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    del cfg_data["research"]
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert cfg.research.enabled is False
    assert cfg.research.model == "gpt-5.4-mini"
    assert cfg.research.reasoning_effort == "low"
    assert cfg.research.previous_summary_count == 5
    assert cfg.synthesis.enabled is False
    assert cfg.synthesis.model == "gpt-5.4-mini"
    assert cfg.synthesis.reasoning_effort == "low"
    assert cfg.synthesis.every_scored_steps == 15
    assert cfg.exec.memory_limit_gb == 80.0
    assert cfg.agent.mode == "legacy"
    assert cfg.agent.autogluon.profile == "full_boost"
    assert cfg.agent.autogluon.time_limit == 600
    assert cfg.agent.autogluon.included_model_types is None


def test_load_resume_state_normalizes_autogluon_mode_alias(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=["agent.mode=autogluon"],
    )

    assert cfg.agent.mode == "autogluon_preprocess"


def test_load_resume_state_ignores_deprecated_seeded_base_limit(tmp_path):
    _write_run(tmp_path, "2-existing-run", steps=20, mtime=time.time())
    config_path = tmp_path / "logs" / "2-existing-run" / "config.yaml"
    cfg_data = OmegaConf.load(config_path)
    cfg_data.agent.search.seeded_base_max_children = 3
    OmegaConf.save(cfg_data, config_path)

    cfg, _journal = load_resume_state(
        run_id="2-existing-run",
        top_log_dir=tmp_path / "logs",
        top_workspace_dir=tmp_path / "workspaces",
        cli_overrides=[],
    )

    assert not hasattr(cfg.agent.search, "seeded_base_max_children")
