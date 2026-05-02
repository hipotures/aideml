from pathlib import Path
from dataclasses import dataclass
import os

from aide.journal import Journal, Node
from aide.utils.config import save_run
from aide.utils.metric import MetricValue


@dataclass
class DummyConfig:
    log_dir: Path
    workspace_dir: Path
    exp_name: str = "test-run"


def test_save_run_archives_current_node_code_and_submission_with_same_timestamp(
    tmp_path,
):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    working_dir = workspace_dir / "working"
    working_dir.mkdir(parents=True)
    (working_dir / "submission.csv").write_text("id,PitNextLap\n1,0.7\n")

    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    node = Node(
        code="print('current node')",
        plan="current node plan",
        ctime=1777750547.0057797,
    )
    node.metric = MetricValue(0.9473, maximize=True)
    node.is_buggy = False
    node._term_out = ["CV ROC AUC: 0.9473\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"
    journal.append(node)

    save_run(cfg, journal, current_node=node)

    artifact_dirs = sorted((log_dir / "artifacts").iterdir())

    assert len(artifact_dirs) == 1
    assert artifact_dirs[0].name == "20260502T213547"
    assert (artifact_dirs[0] / "solution.py").read_text() == "print('current node')"
    assert (artifact_dirs[0] / "submission.csv").read_text() == "id,PitNextLap\n1,0.7\n"
    assert (log_dir / "best_solution.py").read_text() == "print('current node')"


def test_save_run_does_not_archive_submission_when_missing(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    (workspace_dir / "working").mkdir(parents=True)

    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    node = Node(
        code="print('no submission')",
        plan="no submission plan",
        ctime=1777750547.0057797,
    )
    node.metric = MetricValue(0.5, maximize=True)
    node.is_buggy = False
    node._term_out = ["CV ROC AUC: 0.5000\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"
    journal.append(node)

    save_run(cfg, journal, current_node=node)

    artifact_dir = log_dir / "artifacts" / "20260502T213547"

    assert (artifact_dir / "solution.py").exists()
    assert not (artifact_dir / "submission.csv").exists()


def test_save_run_does_not_archive_stale_submission_from_previous_node(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    working_dir = workspace_dir / "working"
    working_dir.mkdir(parents=True)
    submission_path = working_dir / "submission.csv"
    submission_path.write_text("id,PitNextLap\n1,0.1\n")
    os.utime(submission_path, (1777750000.0, 1777750000.0))

    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    node = Node(
        code="raise RuntimeError('bug')",
        plan="buggy node plan",
        ctime=1777750547.0057797,
    )
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = True
    node._term_out = ["RuntimeError: bug\n"]
    node.exec_time = 1.0
    node.exc_type = "RuntimeError"
    node.analysis = "buggy"
    journal.append(node)

    save_run(cfg, journal, current_node=node)

    artifact_dir = log_dir / "artifacts" / "20260502T213547"

    assert (artifact_dir / "solution.py").exists()
    assert not (artifact_dir / "submission.csv").exists()
