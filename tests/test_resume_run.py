import os
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from aide.journal import Journal, Node
from aide.run import (
    find_latest_run_id,
    load_resume_state,
    parse_resume_args,
)
from aide.utils.config import _load_cfg, prep_cfg, save_run
from aide.utils.metric import MetricValue


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
    assert cfg.research.model == "gpt-5.5"
    assert cfg.research.previous_summary_count == 5
    assert cfg.synthesis.enabled is False
    assert cfg.synthesis.model == "gpt-5.5"
    assert cfg.synthesis.every_scored_steps == 15
    assert cfg.exec.memory_limit_gb == 80.0
