import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "smart_kaggle_submit.py"
SPEC = importlib.util.spec_from_file_location("smart_kaggle_submit", MODULE_PATH)
smart_kaggle_submit = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = smart_kaggle_submit
SPEC.loader.exec_module(smart_kaggle_submit)

SubmissionRegistry = smart_kaggle_submit.SubmissionRegistry
collect_candidates = smart_kaggle_submit.collect_candidates
select_top_unsent_ready = smart_kaggle_submit.select_top_unsent_ready
submit_candidates = smart_kaggle_submit.submit_candidates


def _ctime(timestamp: str) -> float:
    return dt.datetime.strptime(timestamp, "%Y%m%dT%H%M%S").timestamp()


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> None:
    run_dir = logs_dir / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "journal.json").write_text(
        json.dumps({"__version": "test", "node2parent": {}, "nodes": nodes})
    )


def _write_artifact(logs_dir: Path, run_name: str, timestamp: str, body: str) -> Path:
    artifact_dir = logs_dir / run_name / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True)
    submission_path = artifact_dir / "submission.csv"
    submission_path.write_text(body)
    return submission_path


def test_collect_candidates_reads_scores_and_marks_submit_ready(tmp_path):
    logs_dir = tmp_path / "logs"
    _write_journal(
        logs_dir,
        "run-a",
        [
            {
                "step": 0,
                "id": "node-ready-low",
                "ctime": _ctime("20260502T100000"),
                "metric": {"value": 0.8, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 1,
                "id": "node-not-ready",
                "ctime": _ctime("20260502T101000"),
                "metric": {"value": 0.9, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 2,
                "id": "node-buggy",
                "ctime": _ctime("20260502T102000"),
                "metric": {"value": None, "maximize": True},
                "is_buggy": True,
            },
        ],
    )
    _write_artifact(logs_dir, "run-a", "20260502T100000", "id,target\n1,0.8\n")

    candidates = collect_candidates(logs_dir)

    assert [(c.step, c.local_score, c.is_submit_ready) for c in candidates] == [
        (0, 0.8, True),
        (1, 0.9, False),
        (2, None, False),
    ]
    assert candidates[0].sha256 is not None
    assert candidates[0].submission_path == (
        logs_dir / "run-a" / "artifacts" / "20260502T100000" / "submission.csv"
    )


def test_select_top_unsent_ready_sorts_by_metric_and_skips_registry_duplicates(
    tmp_path,
):
    logs_dir = tmp_path / "logs"
    _write_journal(
        logs_dir,
        "run-a",
        [
            {
                "step": 0,
                "id": "node-low",
                "ctime": _ctime("20260502T100000"),
                "metric": {"value": 0.8, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 1,
                "id": "node-high",
                "ctime": _ctime("20260502T101000"),
                "metric": {"value": 0.9, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 2,
                "id": "node-mid",
                "ctime": _ctime("20260502T102000"),
                "metric": {"value": 0.85, "maximize": True},
                "is_buggy": False,
            },
        ],
    )
    _write_artifact(logs_dir, "run-a", "20260502T100000", "id,target\n1,0.8\n")
    _write_artifact(logs_dir, "run-a", "20260502T101000", "id,target\n1,0.9\n")
    _write_artifact(logs_dir, "run-a", "20260502T102000", "id,target\n1,0.85\n")
    candidates = collect_candidates(logs_dir)

    registry = SubmissionRegistry(
        tmp_path / "registry.json",
        [
            {
                "competition": "playground-series-s6e5",
                "run": "run-a",
                "step": 1,
                "timestamp": "20260502T101000",
                "sha256": candidates[1].sha256,
            }
        ],
    )

    selected = select_top_unsent_ready(
        candidates,
        registry=registry,
        competition="playground-series-s6e5",
        limit=5,
    )

    assert [(c.step, c.local_score) for c in selected] == [(2, 0.85), (0, 0.8)]


def test_registry_round_trip_and_duplicate_detection(tmp_path):
    registry_path = tmp_path / "submission_registry.json"
    registry = SubmissionRegistry(registry_path)
    entry = {
        "competition": "playground-series-s6e5",
        "run": "run-a",
        "step": 0,
        "timestamp": "20260502T100000",
        "sha256": "abc123",
    }

    registry.add(entry)

    reloaded = SubmissionRegistry.load(registry_path)
    assert reloaded.is_submitted(
        competition="playground-series-s6e5",
        sha256="abc123",
        run="other-run",
        step=9,
        timestamp="20260502T111111",
    )
    assert reloaded.is_submitted(
        competition="playground-series-s6e5",
        sha256="different",
        run="run-a",
        step=0,
        timestamp="20260502T100000",
    )


class FakeKaggleClient:
    def __init__(self):
        self.calls = []

    def competition_submit(self, file_name, message, competition, quiet=False):
        self.calls.append(
            {
                "file_name": file_name,
                "message": message,
                "competition": competition,
                "quiet": quiet,
            }
        )
        return {"ok": True}


def test_submit_candidates_records_each_successful_submission(tmp_path):
    logs_dir = tmp_path / "logs"
    _write_journal(
        logs_dir,
        "run-a",
        [
            {
                "step": 0,
                "id": "abcdef0123456789",
                "ctime": _ctime("20260502T100000"),
                "metric": {"value": 0.8, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 1,
                "id": "fedcba9876543210",
                "ctime": _ctime("20260502T101000"),
                "metric": {"value": 0.9, "maximize": True},
                "is_buggy": False,
            },
        ],
    )
    _write_artifact(logs_dir, "run-a", "20260502T100000", "id,target\n1,0.8\n")
    _write_artifact(logs_dir, "run-a", "20260502T101000", "id,target\n1,0.9\n")
    candidates = collect_candidates(logs_dir)
    selected = select_top_unsent_ready(
        candidates,
        registry=SubmissionRegistry(tmp_path / "registry.json"),
        competition="playground-series-s6e5",
        limit=2,
    )
    registry = SubmissionRegistry(tmp_path / "registry.json")
    client = FakeKaggleClient()

    submitted = submit_candidates(
        selected,
        registry=registry,
        client=client,
        competition="playground-series-s6e5",
    )

    assert [entry["step"] for entry in submitted] == [1, 0]
    assert len(client.calls) == 2
    assert "cv=0.90000" in client.calls[0]["message"]
    assert "run=run-a" in client.calls[0]["message"]
    assert "step=1" in client.calls[0]["message"]
    assert SubmissionRegistry.load(tmp_path / "registry.json").is_submitted(
        competition="playground-series-s6e5",
        sha256=selected[0].sha256,
        run="run-a",
        step=1,
        timestamp="20260502T101000",
    )
