from pathlib import Path
from dataclasses import dataclass
import os

import pytest

from aide.journal import Journal, Node
from aide.run import mark_node_generated_only
from aide.utils import serialize
from aide.utils.config import save_run
from aide.utils.metric import MetricValue
import json


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
    assert artifact_dirs[0].name.startswith("20260502T213547-")
    assert (artifact_dirs[0] / "solution.py").read_text() == "print('current node')"
    assert (artifact_dirs[0] / "submission.csv").read_text() == "id,PitNextLap\n1,0.7\n"
    manifest = json.loads((artifact_dirs[0] / "aide_result.json").read_text())
    assert manifest["kind"] == "source_node"
    assert manifest["run"] == "run"
    assert manifest["timestamp"] == artifact_dirs[0].name
    assert manifest["status"] == "ok"
    assert manifest["local_score"] == 0.9473
    assert manifest["node"]["id"] == node.id
    assert manifest["node"]["step"] == 0
    assert manifest["execution"]["exec_time"] == 1.0
    assert manifest["files"]["submission"]["path"] == "submission.csv"
    assert not (artifact_dirs[0] / "error.txt").exists()
    assert (log_dir / "best_solution.py").read_text() == "print('current node')"
    saved = json.loads((log_dir / "journal.json").read_text())
    assert saved["__version"] == "3"
    assert saved["nodes"][0]["code"] == ""
    assert saved["nodes"][0]["code_path"] == (
        f"artifacts/{artifact_dirs[0].name}/solution.py"
    )


def test_save_run_uses_explicit_artifact_dir_name(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    (workspace_dir / "input").mkdir(parents=True)
    (workspace_dir / "working").mkdir(parents=True)

    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    node = Node(
        code="print('current node')",
        plan="plan",
        ctime=1_779_492_701.0,
        artifact_dir_name="20260523T220603-a1b2c3d4",
    )
    node.is_buggy = False
    node._term_out = ["ok\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"
    journal.append(node)

    save_run(cfg, journal, current_node=node)

    artifact_dir = cfg.log_dir / "artifacts" / "20260523T220603-a1b2c3d4"
    assert (artifact_dir / "solution.py").read_text() == "print('current node')"
    assert (artifact_dir / "input").is_symlink()
    assert (artifact_dir / "input").resolve() == (workspace_dir / "input").resolve()


def test_save_run_archives_solution_helper_when_available(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    (workspace_dir / "working").mkdir(parents=True)
    (workspace_dir / "aide_solution_helpers.py").write_text(
        "def load_competition_data():\n    pass\n",
        encoding="utf-8",
    )

    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    node = Node(code="print('current node')", plan="plan")
    node.is_buggy = False
    node._term_out = ["ok\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"
    journal.append(node)

    save_run(cfg, journal, current_node=node)

    artifact_dir = next((log_dir / "artifacts").iterdir())
    helper_path = artifact_dir / "aide_solution_helpers.py"
    assert helper_path.exists()
    assert "load_competition_data" in helper_path.read_text(encoding="utf-8")


def test_load_json_hydrates_code_from_solution_artifact(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    (workspace_dir / "working").mkdir(parents=True)
    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    node = Node(code="print('original')", plan="plan")
    node.is_buggy = False
    node._term_out = ["ok\n"]
    journal.append(node)

    save_run(cfg, journal, current_node=node)
    artifact_dir = next((log_dir / "artifacts").iterdir())
    (artifact_dir / "solution.py").write_text("print('edited')", encoding="utf-8")

    loaded = serialize.load_json(log_dir / "journal.json", Journal)

    assert loaded.nodes[0].code == "print('edited')"


def test_load_json_rejects_missing_solution_artifact(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    log_dir.mkdir(parents=True)
    (log_dir / "journal.json").write_text(
        json.dumps(
            {
                "__version": "3",
                "nodes": [
                    {
                        "id": "node-1",
                        "code": "",
                        "code_path": "artifacts/node-1/solution.py",
                        "artifact_dir_name": "node-1",
                    }
                ],
                "node2parent": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="solution artifact"):
        serialize.load_json(log_dir / "journal.json", Journal)


def test_load_json_recovers_failed_node_with_removed_solution_artifact(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    log_dir.mkdir(parents=True)
    journal_path = log_dir / "journal.json"
    journal_path.write_text(
        json.dumps(
            {
                "__version": "3",
                "nodes": [
                    {
                        "id": "failed-node",
                        "code": "",
                        "code_path": "artifacts/failed-node/solution.py",
                        "artifact_dir_name": "failed-node",
                        "status": "failed",
                        "plan": "Generation failed: missing Codex metadata.",
                    }
                ],
                "node2parent": {},
            }
        ),
        encoding="utf-8",
    )

    loaded = serialize.load_json(journal_path, Journal)
    serialized = json.loads(serialize.dumps_json(loaded, base_dir=log_dir))

    assert loaded.nodes[0].status == "failed"
    assert "missing Codex metadata" in loaded.nodes[0].code
    assert loaded.nodes[0].code_path is None
    assert loaded.nodes[0].artifact_dir_name is None
    assert serialized["nodes"][0]["code_path"] is None
    assert serialized["nodes"][0]["artifact_dir_name"] is None


def test_save_run_writes_node_run_stats_to_manifest(tmp_path):
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
    node.run_stats = {
        "feature_count": 42,
        "preprocess_time": 1.2,
        "training_time": 3.4,
        "total_exec_time": 5.0,
        "models": [{"model": "WeightedEnsemble_L2", "score_val": 0.95}],
    }
    journal.append(node)

    save_run(cfg, journal, current_node=node)

    artifact_dir = next((log_dir / "artifacts").iterdir())
    manifest = json.loads((artifact_dir / "aide_result.json").read_text())

    expected = dict(node.run_stats)
    expected["total_exec_time"] = 1.0
    assert manifest["run_stats"] == expected


def test_save_run_handles_generated_only_current_node_without_metric(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    (workspace_dir / "working").mkdir(parents=True)

    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    scored = Node(code="print('scored')", plan="scored", ctime=1777750000.0)
    scored.metric = MetricValue(0.5, maximize=True)
    scored.is_buggy = False
    scored._term_out = ["CV ROC AUC: 0.5000\n"]
    scored.exec_time = 1.0
    scored.exc_type = None
    scored.analysis = "ran successfully"
    generated = Node(code="print('generated')", plan="generated", ctime=1777750547.0)
    mark_node_generated_only(generated)
    journal.append(scored)
    journal.append(generated)

    save_run(cfg, journal, current_node=generated)

    assert (log_dir / "best_solution.py").read_text() == "print('scored')"
    artifact_dir = next((log_dir / "artifacts").iterdir())
    manifest = json.loads((artifact_dir / "aide_result.json").read_text())
    assert manifest["status"] == "generated"
    assert manifest["local_score"] is None


def test_save_run_persists_journal_when_progress_callback_fails(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    (workspace_dir / "working").mkdir(parents=True)

    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    node = Node(code="print('must persist')", plan="persist")
    mark_node_generated_only(node)
    journal.append(node)

    def failing_progress_callback(_message: str) -> None:
        raise RuntimeError("ui render failed")

    save_run(
        cfg,
        journal,
        current_node=node,
        progress_callback=failing_progress_callback,
    )

    assert (log_dir / "journal.json").exists()
    saved = json.loads((log_dir / "journal.json").read_text())
    assert saved["nodes"][0]["status"] == "generated"


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

    artifact_dir = next((log_dir / "artifacts").iterdir())

    assert (artifact_dir / "solution.py").exists()
    assert not (artifact_dir / "submission.csv").exists()
    manifest = json.loads((artifact_dir / "aide_result.json").read_text())
    assert manifest["status"] == "ok"
    assert manifest["local_score"] == 0.5
    assert manifest["files"]["submission"] is None


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
    node.exc_info = {"args": ["bug"]}
    node.exc_stack = [("runfile.py", 1, "<module>", "raise RuntimeError('bug')")]
    node.analysis = "buggy"
    journal.append(node)

    save_run(cfg, journal, current_node=node)

    artifact_dir = next((log_dir / "artifacts").iterdir())

    assert (artifact_dir / "solution.py").exists()
    assert not (artifact_dir / "submission.csv").exists()
    manifest = json.loads((artifact_dir / "aide_result.json").read_text())
    assert manifest["status"] == "bug"
    assert manifest["local_score"] is None
    assert manifest["files"]["error"]["path"] == "error.txt"
    assert manifest["execution"]["exc_type"] == "RuntimeError"
    error_text = (artifact_dir / "error.txt").read_text()
    assert "Exception type:\nRuntimeError" in error_text
    assert '"args": [' in error_text
    assert "Terminal output:\nRuntimeError: bug" in error_text
    assert "Analysis:\nbuggy" in error_text


def test_save_run_reports_progress_for_each_save_stage(tmp_path):
    log_dir = tmp_path / "logs" / "run"
    workspace_dir = tmp_path / "workspaces" / "run"
    (workspace_dir / "working").mkdir(parents=True)

    cfg = DummyConfig(log_dir=log_dir, workspace_dir=workspace_dir)
    journal = Journal()
    node = Node(
        code="print('progress')",
        plan="progress node plan",
        ctime=1777750547.0057797,
    )
    node.metric = MetricValue(0.5, maximize=True)
    node.is_buggy = False
    node._term_out = ["CV ROC AUC: 0.5000\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"
    journal.append(node)
    messages = []

    save_run(cfg, journal, current_node=node, progress_callback=messages.append)

    assert messages == [
        "Preparing log directory",
        "Saving node artifacts",
        "Saving journal",
        "Saving config",
        "Rendering tree HTML",
        "Saving best solution",
    ]
