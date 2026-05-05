import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_submission_lab.py"
SPEC = importlib.util.spec_from_file_location("kaggle_submission_lab", MODULE_PATH)
kaggle_submission_lab = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = kaggle_submission_lab
SPEC.loader.exec_module(kaggle_submission_lab)


def _ctime(timestamp: str) -> float:
    return dt.datetime.strptime(timestamp, "%Y%m%dT%H%M%S").timestamp()


def _write_artifact(
    logs_dir: Path,
    run_name: str,
    timestamp: str,
    *,
    code: str,
    submission: str | None = "id,target\n1,0.8\n",
) -> Path:
    artifact_dir = logs_dir / run_name / "artifacts" / timestamp
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "solution.py").write_text(code)
    if submission is not None:
        (artifact_dir / "submission.csv").write_text(submission)
    return artifact_dir


def test_refresh_index_records_result_manifests_without_journal(tmp_path):
    logs_dir = tmp_path / "logs"
    artifact = _write_artifact(
        logs_dir,
        "remote-run",
        "20260506T120000",
        code=(
            "AIDE_AG_CONFIG = {'profile': 'full_boost_gpu', "
            "'included_model_types': ['XGB', 'GBM', 'CAT'], "
            "'presets': 'medium_quality', 'time_limit': 600}\n"
            "def preprocess(df):\n    return df\n"
        ),
        submission="id,target\n1,0.9\n",
    )
    submission_sha = kaggle_submission_lab.sha256_file(artifact / "submission.csv")
    (artifact / "aide_result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "source_node",
                "competition": "playground-series-s6e5",
                "run": "remote-run",
                "timestamp": "20260506T120000",
                "artifact_dir": str(artifact),
                "status": "ok",
                "local_score": 0.95098,
                "metric_maximize": True,
                "is_buggy": False,
                "sha256": submission_sha,
                "profile": "full_boost_gpu",
                "autogluon_presets": "medium_quality",
                "included_model_types": ["XGB", "GBM", "CAT"],
                "time_limit": 600,
                "node": {
                    "id": "node-remote",
                    "step": 4,
                    "ctime": _ctime("20260506T120000"),
                    "parent_id": "node-parent",
                    "metric": {"value": 0.95098, "maximize": True},
                    "is_buggy": False,
                    "plan": "remote plan",
                    "analysis": "remote analysis",
                },
                "execution": {"exec_time": 123.0},
                "source": {
                    "source_run": None,
                    "source_node_id": None,
                    "source_step": None,
                    "source_timestamp": None,
                    "source_sha256": None,
                },
            }
        )
    )

    index = kaggle_submission_lab.refresh_index(
        logs_dir=logs_dir,
        index_path=logs_dir / "submission_index.json",
        competition="playground-series-s6e5",
        reindex=True,
    )

    assert len(index["records"]) == 1
    record = index["records"][0]
    assert record["kind"] == "source_node"
    assert record["run"] == "remote-run"
    assert record["step"] == 4
    assert record["node_id"] == "node-remote"
    assert record["parent_node_id"] == "node-parent"
    assert record["local_score"] == 0.95098
    assert record["sha256"] == submission_sha
    assert record["profile"] == "full_boost_gpu"
    assert record["artifact_dir"] == str(artifact)

def test_refresh_index_skips_unchanged_runs_without_reindex(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    artifact = _write_artifact(
        logs_dir,
        "run-a",
        "20260504T100000",
        code="AIDE_AG_CONFIG = {'included_model_types': ['XGB']}\n",
    )
    (artifact / "aide_result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "source_node",
                "run": "run-a",
                "timestamp": "20260504T100000",
                "status": "ok",
                "local_score": 0.9,
                "metric_maximize": True,
                "is_buggy": False,
                "node": {
                    "id": "node-a",
                    "step": 0,
                    "metric": {"value": 0.9, "maximize": True},
                },
                "execution": {},
                "source": {},
            }
        )
    )
    index_path = logs_dir / "submission_index.json"
    first = kaggle_submission_lab.refresh_index(
        logs_dir=logs_dir,
        index_path=index_path,
        competition="playground-series-s6e5",
        reindex=True,
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("unchanged run should not be rebuilt")

    monkeypatch.setattr(kaggle_submission_lab, "build_run_records", fail_if_called)

    second = kaggle_submission_lab.refresh_index(
        logs_dir=logs_dir,
        index_path=index_path,
        competition="playground-series-s6e5",
    )

    assert second["records"] == first["records"]


def test_select_top_records_deduplicates_same_submission_hash(tmp_path):
    records = [
        {
            "kind": "source_node",
            "competition": "playground-series-s6e5",
            "run": "run-a",
            "step": 1,
            "timestamp": "20260504T100000",
            "artifact_dir": str(tmp_path),
            "submission_path": str(tmp_path / "submission.csv"),
            "local_score": 0.90,
            "metric_maximize": True,
            "status": "ok",
            "is_buggy": False,
            "sha256": "samehash",
        },
        {
            "kind": "source_node",
            "competition": "playground-series-s6e5",
            "run": "run-a",
            "step": 2,
            "timestamp": "20260504T101000",
            "artifact_dir": str(tmp_path),
            "submission_path": str(tmp_path / "submission.csv"),
            "local_score": 0.91,
            "metric_maximize": True,
            "status": "ok",
            "is_buggy": False,
            "sha256": "samehash",
        },
    ]
    (tmp_path / "submission.csv").write_text("id,target\n1,0.8\n")

    selected = kaggle_submission_lab.select_top_records(
        records,
        registry=kaggle_submission_lab.smart.SubmissionRegistry(
            tmp_path / "registry.json"
        ),
        competition="playground-series-s6e5",
        limit=5,
    )

    assert len(selected) == 1
    assert selected[0]["step"] == 2


def test_render_table_hides_source_column_when_no_profile_evals(tmp_path):
    console = kaggle_submission_lab.Console(record=True, width=160, color_system=None)
    records = [
        {
            "kind": "source_node",
            "run": "2-intelligent-amber-bandicoot",
            "step": 1,
            "timestamp": "20260504T134159",
            "local_score": 0.95026,
            "profile": "full_boost",
            "included_model_types": ["XGB", "GBM", "CAT"],
            "sha256": "13bc36ab26abcdef",
        }
    ]

    kaggle_submission_lab.render_table(console, records)

    output = console.export_text()
    assert "src" not in output
    assert "prof" not in output
    assert "20260504" in output


def test_render_table_shows_profile_only_in_full_view(tmp_path):
    records = [
        {
            "kind": "source_node",
            "run": "2-intelligent-amber-bandicoot",
            "step": 1,
            "timestamp": "20260504T134159",
            "local_score": 0.95026,
            "profile": "full_best_30m",
            "included_model_types": ["XGB", "GBM", "CAT"],
            "sha256": "13bc36ab26abcdef",
        }
    ]
    compact = kaggle_submission_lab.Console(record=True, width=160, color_system=None)
    full = kaggle_submission_lab.Console(record=True, width=160, color_system=None)

    kaggle_submission_lab.render_table(compact, records)
    kaggle_submission_lab.render_table(full, records, full_view=True)

    assert "prof" not in compact.export_text()
    full_output = full.export_text()
    assert "prof" in full_output
    assert "full_best_30m" in full_output


def test_render_registry_table_numbers_only_complete_submissions(tmp_path):
    registry = kaggle_submission_lab.smart.SubmissionRegistry(
        tmp_path / "registry.json",
        entries=[
            {
                "competition": "playground-series-s6e5",
                "run": "run-a",
                "step": 1,
                "timestamp": "20260504T100000",
                "local_score": 0.95,
                "sha256": "aaaabbbbcccc",
                "remote_status": "ERROR",
                "public_score": None,
            },
            {
                "competition": "playground-series-s6e5",
                "run": "run-b",
                "step": 2,
                "timestamp": "20260504T110000",
                "local_score": 0.96,
                "sha256": "dddd11112222",
                "remote_status": "COMPLETE",
                "public_score": "0.90123",
            },
        ],
    )
    console = kaggle_submission_lab.Console(record=True, width=160, color_system=None)

    kaggle_submission_lab.render_registry_table(console, registry)

    output = console.export_text()
    assert "Submission registry" in output
    assert "0.90123" in output
    assert "COMPLETE" in output
    assert "ERROR" in output


def test_sync_registry_from_kaggle_updates_public_score(tmp_path, monkeypatch):
    registry = kaggle_submission_lab.smart.SubmissionRegistry(
        tmp_path / "registry.json",
        entries=[
            {
                "competition": "playground-series-s6e5",
                "run": "run-a",
                "step": 1,
                "timestamp": "20260504T100000",
                "node_id": "node-source",
                "sha256": "aaaabbbbcccc",
            }
        ],
    )

    class FakeRemote:
        ref = 123
        file_name = "submission.csv"
        description = "cv=0.95000 | run=run-a | step=1 | aide_ts=20260504T100000 | node=node-sour | sha=aaaabbbbcc"
        status = "COMPLETE"
        public_score = "0.91234"
        private_score = None
        url = None
        total_bytes = 100
        date = "2026-05-04"

    fake_client = object()
    monkeypatch.setattr(
        kaggle_submission_lab.smart,
        "_build_kaggle_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        kaggle_submission_lab.smart,
        "fetch_remote_submissions",
        lambda client, competition: [FakeRemote()],
    )
    console = kaggle_submission_lab.Console(record=True, width=160, color_system=None)

    client, remote = kaggle_submission_lab.sync_registry_from_kaggle(
        console=console,
        registry=registry,
        competition="playground-series-s6e5",
    )

    assert client is fake_client
    assert len(remote) == 1
    assert registry.entries[0]["public_score"] == "0.91234"
    assert registry.entries[0]["remote_status"] == "COMPLETE"


def test_render_registry_table_includes_remote_only_submissions(tmp_path):
    registry = kaggle_submission_lab.smart.SubmissionRegistry(
        tmp_path / "registry.json",
        entries=[],
    )

    class FakeRemote:
        file_name = "submission_autogluon.csv"
        description = "Ensemble Weights: should not be rendered"
        status = "COMPLETE"
        public_score = "0.94895"
        date = "2026-05-04T18:04:00Z"

    console = kaggle_submission_lab.Console(record=True, width=160, color_system=None)

    kaggle_submission_lab.render_registry_table(console, registry, [FakeRemote()])

    output = console.export_text()
    assert "Submission registry" in output
    assert "0.94895" in output
    assert "COMPLETE" in output
    assert "20260504" in output
    assert "submission_autogluon.csv" in output
    assert "Ensemble Weights" not in output


def test_render_registry_table_parses_remote_only_aide_description(tmp_path):
    registry = kaggle_submission_lab.smart.SubmissionRegistry(
        tmp_path / "registry.json",
        entries=[],
    )

    class FakeRemote:
        file_name = "sub_20260505T204454_step--1_node-profile-_sha-165e7caae1_cv-0.95100.csv"
        description = (
            "cv=0.95100 | run=2-remote-server-run | step=-1 | "
            "aide_ts=20260505T204454 | node=profile- | sha=165e7caae1"
        )
        status = "COMPLETE"
        public_score = "0.95072"
        date = "2026-05-05T20:44:54Z"

    console = kaggle_submission_lab.Console(record=True, width=180, color_system=None)

    kaggle_submission_lab.render_registry_table(console, registry, [FakeRemote()])

    output = console.export_text()
    assert "0.95100" in output
    assert "0.95072" in output
    assert "2-remote-server-run" in output
    assert "20260505" in output
    assert "165e7caae1" in output
    assert "sub_20260505T204454" not in output
