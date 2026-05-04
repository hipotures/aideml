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


def _write_journal(logs_dir: Path, run_name: str, nodes: list[dict]) -> Path:
    run_dir = logs_dir / run_name
    run_dir.mkdir(parents=True)
    journal_path = run_dir / "journal.json"
    journal_path.write_text(
        json.dumps({"__version": "test", "node2parent": {}, "nodes": nodes})
    )
    return journal_path


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


def test_refresh_index_records_source_nodes_and_profile_evals(tmp_path):
    logs_dir = tmp_path / "logs"
    source_code = (
        "AIDE_AG_CONFIG = {'profile': 'fast_boost', "
        "'included_model_types': ['XGB', 'GBM'], 'presets': 'medium_quality', "
        "'time_limit': 300}\n"
        "def preprocess(df):\n    return df\n"
    )
    _write_journal(
        logs_dir,
        "run-a",
        [
            {
                "step": 2,
                "id": "node-source",
                "ctime": _ctime("20260504T100000"),
                "metric": {"value": 0.95, "maximize": True},
                "is_buggy": False,
                "exec_time": 12.5,
            }
        ],
    )
    source_artifact = _write_artifact(
        logs_dir,
        "run-a",
        "20260504T100000",
        code=source_code,
    )
    eval_artifact = _write_artifact(
        logs_dir,
        "run-a",
        "20260504T110000",
        code=source_code.replace("fast_boost", "full_boost"),
        submission="id,target\n1,0.9\n",
    )
    (eval_artifact / "submission_eval.json").write_text(
        json.dumps(
            {
                "kind": "profile_eval",
                "competition": "playground-series-s6e5",
                "source_run": "run-a",
                "source_node_id": "node-source",
                "source_step": 2,
                "source_timestamp": "20260504T100000",
                "source_sha256": kaggle_submission_lab.sha256_file(
                    source_artifact / "submission.csv"
                ),
                "profile": "full_boost",
                "autogluon_presets": "best_quality",
                "included_model_types": ["XGB", "GBM", "CAT"],
                "time_limit": 600,
                "local_score": 0.951,
                "exec_time": 42.0,
                "status": "ok",
            }
        )
    )

    index = kaggle_submission_lab.refresh_index(
        logs_dir=logs_dir,
        index_path=logs_dir / "submission_index.json",
        competition="playground-series-s6e5",
        reindex=True,
    )

    assert [record["kind"] for record in index["records"]] == [
        "source_node",
        "profile_eval",
    ]
    assert index["records"][0]["artifact_dir"] == str(source_artifact)
    assert index["records"][0]["profile"] == "fast_boost"
    assert index["records"][0]["autogluon_presets"] == "medium_quality"
    assert index["records"][0]["included_model_types"] == ["XGB", "GBM"]
    assert index["records"][1]["autogluon_presets"] == "best_quality"
    assert index["records"][1]["source_sha256"] == index["records"][0]["sha256"]
    assert index["records"][1]["artifact_dir"] == str(eval_artifact)


def test_refresh_index_skips_unchanged_runs_without_reindex(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    _write_journal(
        logs_dir,
        "run-a",
        [
            {
                "step": 0,
                "id": "node-a",
                "ctime": _ctime("20260504T100000"),
                "metric": {"value": 0.9, "maximize": True},
                "is_buggy": False,
            }
        ],
    )
    _write_artifact(
        logs_dir,
        "run-a",
        "20260504T100000",
        code="AIDE_AG_CONFIG = {'included_model_types': ['XGB']}\n",
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
