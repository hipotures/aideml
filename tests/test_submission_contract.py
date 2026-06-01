from dataclasses import dataclass
import gzip
import os
from pathlib import Path

from aide.journal import Journal, Node
import aide.run as run_module
import aide.utils.submission_validation as submission_validation
from aide.run import enforce_journal_submission_contract, enforce_submission_contract
from aide.utils.artifact_manifest import artifact_timestamp_from_ctime
from aide.utils.metric import MetricValue
from aide.utils.submission_validation import validate_workspace_submission


@dataclass
class DummyConfig:
    workspace_dir: Path
    log_dir: Path | None = None


def _write_sample(workspace_dir: Path) -> None:
    input_dir = workspace_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "sample_submission.csv").write_text("id,PitNextLap\n1,0.0\n2,0.0\n")


def _write_submission(workspace_dir: Path, body: str) -> None:
    working_dir = workspace_dir / "working"
    working_dir.mkdir(parents=True)
    (working_dir / "submission.csv").write_text(body)


def test_validate_workspace_submission_rejects_duplicate_ids(tmp_path):
    workspace_dir = tmp_path / "workspace"
    _write_sample(workspace_dir)
    _write_submission(workspace_dir, "id,PitNextLap\n1,0.8\n1,0.9\n")

    error = validate_workspace_submission(workspace_dir)

    assert error == "duplicate id rows: 1"


def test_validate_workspace_submission_streams_without_pandas(tmp_path):
    workspace_dir = tmp_path / "workspace"
    _write_sample(workspace_dir)
    _write_submission(workspace_dir, "id,PitNextLap\n1,0.8\n2,0.9\n")

    assert not hasattr(submission_validation, "pd")
    assert validate_workspace_submission(workspace_dir) is None


def test_validate_workspace_submission_accepts_categorical_labels(tmp_path):
    workspace_dir = tmp_path / "workspace"
    input_dir = workspace_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "sample_submission.csv").write_text(
        "id,class\n1,GALAXY\n2,GALAXY\n"
    )
    _write_submission(workspace_dir, "id,class\n1,STAR\n2,QSO\n")

    assert validate_workspace_submission(workspace_dir) is None


def test_validate_workspace_submission_rejects_empty_categorical_labels(tmp_path):
    workspace_dir = tmp_path / "workspace"
    input_dir = workspace_dir / "input"
    input_dir.mkdir(parents=True)
    (input_dir / "sample_submission.csv").write_text(
        "id,class\n1,GALAXY\n2,GALAXY\n"
    )
    _write_submission(workspace_dir, "id,class\n1,STAR\n2,\n")

    assert (
        validate_workspace_submission(workspace_dir)
        == "class contains empty or null class labels"
    )


def test_validate_workspace_submission_supports_gzipped_sample(tmp_path):
    workspace_dir = tmp_path / "workspace"
    input_dir = workspace_dir / "input"
    input_dir.mkdir(parents=True)
    with gzip.open(input_dir / "sample_submission.csv.gz", "wt", encoding="utf-8") as f:
        f.write("id,PitNextLap\n1,0.0\n2,0.0\n")
    _write_submission(workspace_dir, "id,PitNextLap\n1,0.8\n2,0.9\n")

    assert validate_workspace_submission(workspace_dir) is None


def test_enforce_submission_contract_marks_invalid_successful_node_as_bug(tmp_path):
    workspace_dir = tmp_path / "workspace"
    _write_sample(workspace_dir)
    node = Node(code="print('ok')", plan="ok")
    _write_submission(workspace_dir, "id,PitNextLap\n1,0.8\n1,0.9\n")
    node.metric = MetricValue(0.99, maximize=True)
    node.is_buggy = False
    node._term_out = ["CV AUC: 0.99\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"

    changed = enforce_submission_contract(DummyConfig(workspace_dir=workspace_dir), node)

    assert changed is True
    assert node.is_buggy is True
    assert node.metric.value is None
    assert node.exc_type == "SubmissionValidationError"
    assert "duplicate id rows: 1" in "".join(node._term_out)
    assert "Submission validation failed" in node.analysis


def test_enforce_submission_contract_accepts_valid_submission(tmp_path):
    workspace_dir = tmp_path / "workspace"
    _write_sample(workspace_dir)
    node = Node(code="print('ok')", plan="ok")
    _write_submission(workspace_dir, "id,PitNextLap\n1,0.8\n2,0.9\n")
    node.metric = MetricValue(0.99, maximize=True)
    node.is_buggy = False
    node._term_out = ["CV AUC: 0.99\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"

    changed = enforce_submission_contract(DummyConfig(workspace_dir=workspace_dir), node)

    assert changed is True
    assert node.is_buggy is False
    assert node.metric.value == 0.99
    assert node.exc_type is None
    assert node.submission_validation["status"] == "ok"


def test_enforce_submission_contract_rejects_stale_workspace_submission(tmp_path):
    workspace_dir = tmp_path / "workspace"
    _write_sample(workspace_dir)
    _write_submission(workspace_dir, "id,PitNextLap\n1,0.8\n2,0.9\n")
    submission_path = workspace_dir / "working" / "submission.csv"
    node = Node(code="print('no output')", plan="bad", ctime=1778290000.0)
    os.utime(submission_path, (node.ctime - 10, node.ctime - 10))
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = True
    node._term_out = []
    node.exec_time = 0.25
    node.analysis = "no result marker"

    changed = enforce_submission_contract(DummyConfig(workspace_dir=workspace_dir), node)

    assert changed is True
    assert node.is_buggy is True
    assert node.exc_type == "SubmissionValidationError"
    assert node.submission_validation is None
    assert "stale working/submission.csv" in "".join(node._term_out)


def test_enforce_journal_submission_contract_marks_saved_invalid_artifact(tmp_path):
    workspace_dir = tmp_path / "workspace"
    log_dir = tmp_path / "logs" / "run"
    _write_sample(workspace_dir)
    node = Node(code="print('ok')", plan="ok", ctime=1777750547.0057797)
    node.metric = MetricValue(0.99, maximize=True)
    node.is_buggy = False
    node._term_out = ["CV AUC: 0.99\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"
    journal = Journal()
    journal.append(node)
    artifact_dir = log_dir / "artifacts" / artifact_timestamp_from_ctime(node.ctime)
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "submission.csv").write_text("id,PitNextLap\n1,0.8\n1,0.9\n")

    changed = enforce_journal_submission_contract(
        DummyConfig(workspace_dir=workspace_dir, log_dir=log_dir),
        journal,
    )

    assert changed == 1
    assert node.is_buggy is True
    assert node.metric.value is None
    assert node.exc_type == "SubmissionValidationError"


def test_enforce_journal_submission_contract_caches_valid_artifact(tmp_path, monkeypatch):
    workspace_dir = tmp_path / "workspace"
    log_dir = tmp_path / "logs" / "run"
    _write_sample(workspace_dir)
    node = Node(code="print('ok')", plan="ok", ctime=1777750547.0057797)
    node.metric = MetricValue(0.99, maximize=True)
    node.is_buggy = False
    node._term_out = ["CV AUC: 0.99\n"]
    node.exec_time = 1.0
    node.exc_type = None
    node.analysis = "ran successfully"
    journal = Journal()
    journal.append(node)
    artifact_dir = log_dir / "artifacts" / artifact_timestamp_from_ctime(node.ctime)
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "submission.csv").write_text("id,PitNextLap\n1,0.8\n2,0.9\n")
    cfg = DummyConfig(workspace_dir=workspace_dir, log_dir=log_dir)

    first_changed = enforce_journal_submission_contract(cfg, journal)

    assert first_changed == 1
    assert node.submission_validation["status"] == "ok"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("cached validation should skip reading submission")

    monkeypatch.setattr(run_module, "validate_submission_file", fail_if_called)

    second_changed = enforce_journal_submission_contract(cfg, journal)

    assert second_changed == 0
    assert node.is_buggy is False
    assert node.metric.value == 0.99


def test_force_recheck_can_restore_fixed_submission_validation_error(tmp_path):
    workspace_dir = tmp_path / "workspace"
    log_dir = tmp_path / "logs" / "run"
    _write_sample(workspace_dir)
    node = Node(code="print('ok')", plan="ok", ctime=1777750547.0057797)
    node.metric = MetricValue(None, maximize=True)
    node.is_buggy = True
    node._term_out = ["CV AUC: 0.99\nSubmission saved successfully.\n"]
    node.exec_time = 1.0
    node.exc_type = "SubmissionValidationError"
    node.exc_info = {"args": ["duplicate id rows: 1"]}
    node.analysis = "Submission validation failed: duplicate id rows: 1"
    node.submission_validation = {
        "status": "error",
        "error": "duplicate id rows: 1",
        "previous_metric": {"value": 0.99, "maximize": True},
    }
    journal = Journal()
    journal.append(node)
    artifact_dir = log_dir / "artifacts" / artifact_timestamp_from_ctime(node.ctime)
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "submission.csv").write_text("id,PitNextLap\n1,0.8\n2,0.9\n")

    changed = enforce_journal_submission_contract(
        DummyConfig(workspace_dir=workspace_dir, log_dir=log_dir),
        journal,
        force_check_submissions=True,
    )

    assert changed == 1
    assert node.is_buggy is False
    assert node.metric.value == 0.99
    assert node.exc_type is None
    assert node.submission_validation["status"] == "ok"
