import os
import time
import datetime as dt
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aide.journal import Journal, Node
from aide.run import (
    find_latest_run_id,
    load_resume_state,
    parse_runtime_args,
    parse_resume_args,
)
from aide.utils.config import _load_cfg, prep_cfg, save_run
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
