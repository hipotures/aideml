import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

from rich.console import Console

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
sync_registry_from_remote = smart_kaggle_submit.sync_registry_from_remote
render_dry_run = smart_kaggle_submit.render_dry_run
validate_candidates = smart_kaggle_submit.validate_candidates


class FakeProgress:
    def __init__(self):
        self.tasks = []
        self.advances = {}

    def add_task(self, description, *, total=None):
        task_id = len(self.tasks)
        self.tasks.append({"description": description, "total": total})
        self.advances[task_id] = 0
        return task_id

    def advance(self, task_id, advance=1):
        self.advances[task_id] += advance


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


def test_collect_candidates_reports_progress_per_node(tmp_path):
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
        ],
    )
    progress = FakeProgress()

    collect_candidates(logs_dir, progress=progress)

    assert progress.tasks == [
        {"description": "Scanning local AIDE nodes", "total": 2}
    ]
    assert progress.advances == {0: 2}


def test_validate_candidates_reports_progress_per_candidate(tmp_path):
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
                "id": "node-buggy",
                "ctime": _ctime("20260502T101000"),
                "metric": {"value": None, "maximize": True},
                "is_buggy": True,
            },
        ],
    )
    _write_artifact(logs_dir, "run-a", "20260502T100000", "id,target\n1,0.8\n")
    candidates = collect_candidates(logs_dir)
    progress = FakeProgress()

    validate_candidates(candidates, data_dir=None, progress=progress)

    assert progress.tasks == [
        {"description": "Validating local submissions", "total": 2}
    ]
    assert progress.advances == {0: 2}


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


def test_select_top_unsent_ready_filters_similar_related_predictions(
    tmp_path,
):
    logs_dir = tmp_path / "logs"
    _write_journal(
        logs_dir,
        "run-a",
        [
            {
                "step": 0,
                "id": "parent-node",
                "ctime": _ctime("20260502T100000"),
                "metric": {"value": 0.948596, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 1,
                "id": "child-node",
                "ctime": _ctime("20260502T101000"),
                "metric": {"value": 0.948604, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 2,
                "id": "sibling-node",
                "ctime": _ctime("20260502T102000"),
                "metric": {"value": 0.948604, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 3,
                "id": "bug-node",
                "ctime": _ctime("20260502T103000"),
                "metric": {"value": None, "maximize": True},
                "is_buggy": True,
            },
            {
                "step": 4,
                "id": "grandchild-node",
                "ctime": _ctime("20260502T104000"),
                "metric": {"value": 0.948604, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 5,
                "id": "unrelated-node",
                "ctime": _ctime("20260502T105000"),
                "metric": {"value": 0.948604, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 6,
                "id": "worse-grandchild-node",
                "ctime": _ctime("20260502T110000"),
                "metric": {"value": 0.94850, "maximize": True},
                "is_buggy": False,
            },
        ],
    )
    journal = json.loads((logs_dir / "run-a" / "journal.json").read_text())
    journal["node2parent"] = {
        "child-node": "parent-node",
        "sibling-node": "parent-node",
        "bug-node": "parent-node",
        "grandchild-node": "bug-node",
        "worse-grandchild-node": "child-node",
    }
    (logs_dir / "run-a" / "journal.json").write_text(json.dumps(journal))
    _write_artifact(
        logs_dir,
        "run-a",
        "20260502T100000",
        "id,target\n1,0.100000\n2,0.200000\n",
    )
    _write_artifact(
        logs_dir,
        "run-a",
        "20260502T101000",
        "id,target\n1,0.105000\n2,0.205000\n",
    )
    _write_artifact(
        logs_dir,
        "run-a",
        "20260502T102000",
        "id,target\n1,0.800000\n2,0.900000\n",
    )
    _write_artifact(
        logs_dir,
        "run-a",
        "20260502T104000",
        "id,target\n1,0.110000\n2,0.210000\n",
    )
    _write_artifact(logs_dir, "run-a", "20260502T105000", "id,target\n1,0.87\n")
    _write_artifact(logs_dir, "run-a", "20260502T110000", "id,target\n1,0.88\n")
    candidates = collect_candidates(logs_dir)

    selected = select_top_unsent_ready(
        candidates,
        registry=SubmissionRegistry(tmp_path / "registry.json"),
        competition="playground-series-s6e5",
        limit=5,
    )
    selected_with_related = select_top_unsent_ready(
        candidates,
        registry=SubmissionRegistry(tmp_path / "registry.json"),
        competition="playground-series-s6e5",
        limit=5,
        include_related=True,
    )

    assert [(c.step, c.node_id) for c in selected] == [
        (5, "unrelated-node"),
        (2, "sibling-node"),
        (0, "parent-node"),
    ]
    assert [(c.step, c.node_id) for c in selected_with_related] == [
        (5, "unrelated-node"),
        (4, "grandchild-node"),
        (2, "sibling-node"),
        (1, "child-node"),
        (0, "parent-node"),
    ]


def test_select_top_unsent_ready_reports_filtering_progress(tmp_path):
    logs_dir = tmp_path / "logs"
    _write_journal(
        logs_dir,
        "run-a",
        [
            {
                "step": 0,
                "id": "node-a",
                "ctime": _ctime("20260502T100000"),
                "metric": {"value": 0.8, "maximize": True},
                "is_buggy": False,
            },
            {
                "step": 1,
                "id": "node-b",
                "ctime": _ctime("20260502T101000"),
                "metric": {"value": 0.9, "maximize": True},
                "is_buggy": False,
            },
        ],
    )
    _write_artifact(logs_dir, "run-a", "20260502T100000", "id,target\n1,0.8\n")
    _write_artifact(logs_dir, "run-a", "20260502T101000", "id,target\n1,0.9\n")
    candidates = collect_candidates(logs_dir)
    progress = FakeProgress()

    select_top_unsent_ready(
        candidates,
        registry=SubmissionRegistry(tmp_path / "registry.json"),
        competition="playground-series-s6e5",
        limit=5,
        progress=progress,
    )

    assert progress.tasks == [
        {"description": "Filtering related submissions", "total": 2}
    ]
    assert progress.advances == {0: 2}


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


class FakeRemoteSubmission:
    def __init__(
        self,
        *,
        ref=52271267,
        description="cv=0.90000 | run=run-a | step=1 | ts=20260502T101000 | node=fedcba98",
        file_name="submission.csv",
        public_score="0.87654",
        private_score=None,
        status="COMPLETE",
        url="/submissions/52271267/52271267.raw",
        total_bytes=123,
    ):
        self.ref = ref
        self.description = description
        self.file_name = file_name
        self.public_score = public_score
        self.private_score = private_score
        self.status = status
        self.url = url
        self.total_bytes = total_bytes
        self.date = dt.datetime(2026, 5, 2, 20, 32, 12, 733000)


def test_sync_registry_from_remote_updates_public_score_by_ref(tmp_path):
    registry = SubmissionRegistry(
        tmp_path / "registry.json",
        [
            {
                "competition": "playground-series-s6e5",
                "response": {"ref": 52271267},
                "run": "run-a",
                "step": 1,
                "timestamp": "20260502T101000",
                "sha256": "abc123",
            }
        ],
    )

    changed = sync_registry_from_remote(
        registry=registry,
        competition="playground-series-s6e5",
        remote_submissions=[FakeRemoteSubmission()],
    )

    reloaded = SubmissionRegistry.load(tmp_path / "registry.json")
    assert changed == 1
    assert reloaded.entries[0]["kaggle_ref"] == 52271267
    assert reloaded.entries[0]["public_score"] == "0.87654"
    assert reloaded.entries[0]["remote_filename"] == "submission.csv"
    assert reloaded.entries[0]["remote_status"] == "COMPLETE"
    assert reloaded.entries[0]["remote_url"] == "/submissions/52271267/52271267.raw"


def test_sync_registry_from_remote_updates_public_score_by_timestamp_description(
    tmp_path,
):
    registry = SubmissionRegistry(
        tmp_path / "registry.json",
        [
            {
                "competition": "playground-series-s6e5",
                "run": "run-a",
                "step": 1,
                "timestamp": "20260502T101000",
                "node_id": "fedcba9876543210",
                "sha256": "abc123",
            }
        ],
    )

    changed = sync_registry_from_remote(
        registry=registry,
        competition="playground-series-s6e5",
        remote_submissions=[
            FakeRemoteSubmission(
                description=(
                    "cv=0.90000 | run=run-a | step=1 | "
                    "aide_ts=20260502T101000 | node=fedcba98"
                )
            )
        ],
    )

    reloaded = SubmissionRegistry.load(tmp_path / "registry.json")
    assert changed == 1
    assert reloaded.entries[0]["public_score"] == "0.87654"
    assert reloaded.entries[0]["remote_description"].startswith("cv=0.90000")


def test_render_dry_run_sorts_registry_by_public_score_desc(tmp_path):
    registry = SubmissionRegistry(
        tmp_path / "registry.json",
        [
            {
                "competition": "playground-series-s6e5",
                "run": "run-low",
                "step": 1,
                "timestamp": "20260502T101000",
                "local_score": 0.95,
                "public_score": "0.70124",
                "remote_status": "COMPLETE",
                "uploaded_filename": "low.csv",
                "sha256": "lowhash",
            },
            {
                "competition": "playground-series-s6e5",
                "run": "run-missing",
                "step": 2,
                "timestamp": "20260502T102000",
                "local_score": 0.99,
                "remote_status": "SUBMITTED",
                "uploaded_filename": "missing.csv",
                "sha256": "missinghash",
            },
            {
                "competition": "playground-series-s6e5",
                "run": "run-high",
                "step": 3,
                "timestamp": "20260502T103000",
                "local_score": 0.90,
                "public_score": "0.81234",
                "remote_status": "COMPLETE",
                "uploaded_filename": "high.csv",
                "sha256": "highhash",
            },
        ],
    )
    console = Console(record=True, width=160, color_system=None)

    render_dry_run(
        console=console,
        candidates=[],
        selected=[],
        registry=registry,
        include_not_ready=False,
        remote_submissions=None,
    )
    output = console.export_text()

    assert output.index("run-high") < output.index("run-low")
    assert output.index("run-low") < output.index("run-missing")


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
    assert (
        Path(client.calls[0]["file_name"]).name
        == "sub_20260502T101000_step-1_node-fedcba98_sha-142e531fbf_cv-0.90000.csv"
    )
    assert "cv=0.90000" in client.calls[0]["message"]
    assert "run=run-a" in client.calls[0]["message"]
    assert "step=1" in client.calls[0]["message"]
    assert submitted[0]["uploaded_filename"] == (
        "sub_20260502T101000_step-1_node-fedcba98_sha-142e531fbf_cv-0.90000.csv"
    )
    assert SubmissionRegistry.load(tmp_path / "registry.json").is_submitted(
        competition="playground-series-s6e5",
        sha256=selected[0].sha256,
        run="run-a",
        step=1,
        timestamp="20260502T101000",
    )
