import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from aide.journal import Journal, Node
from aide.utils import serialize
from aide.utils.metric import MetricValue


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
                    "metric": {
                        "value": 0.95098,
                        "maximize": True,
                        "name": "balanced_accuracy",
                    },
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
    assert record["eval_metric"] == "balanced_accuracy"
    assert record["sha256"] == submission_sha
    assert record["profile"] == "full_boost_gpu"
    assert record["algo"] == "AG"
    assert record["artifact_dir"] == str(artifact)


def test_refresh_index_reads_eval_metric_from_autogluon_resolved_settings(tmp_path):
    logs_dir = tmp_path / "logs"
    artifact = _write_artifact(
        logs_dir,
        "resolved-settings-run",
        "20260506T130000",
        code="def preprocess(df):\n    return df\n",
        submission="id,target\n1,0.9\n",
    )
    submission_sha = kaggle_submission_lab.sha256_file(artifact / "submission.csv")
    (artifact / "aide_result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "source_node",
                "competition": "playground-series-s6e5",
                "run": "resolved-settings-run",
                "timestamp": "20260506T130000",
                "artifact_dir": str(artifact),
                "status": "ok",
                "local_score": 0.95098,
                "metric_maximize": True,
                "is_buggy": False,
                "sha256": submission_sha,
                "node": {
                    "id": "node-remote",
                    "step": 4,
                    "ctime": _ctime("20260506T130000"),
                    "metric": {
                        "value": 0.95098,
                        "maximize": True,
                    },
                    "is_buggy": False,
                },
                "autogluon": {
                    "resolved_settings": {
                        "eval_metric": "balanced_accuracy",
                    },
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

    assert index["records"][0]["eval_metric"] == "balanced_accuracy"


def test_refresh_index_backfills_legacy_journal_artifacts(tmp_path):
    logs_dir = tmp_path / "logs"
    run_name = "legacy-run"
    timestamp = "20260505T235122"
    code = (
        "AIDE_AG_CONFIG = {'profile': 'full_boost_gpu', "
        "'included_model_types': ['XGB', 'GBM', 'CAT'], "
        "'presets': 'medium_quality', 'time_limit': 600}\n"
        "def preprocess(df):\n    return df\n"
    )
    artifact = _write_artifact(
        logs_dir,
        run_name,
        timestamp,
        code=code,
        submission="id,target\n1,0.9\n",
    )
    submission_sha = kaggle_submission_lab.sha256_file(artifact / "submission.csv")
    run_dir = logs_dir / run_name
    journal = Journal()
    journal.append(
        Node(
            code=code,
            plan="legacy plan",
            id="node-legacy",
            ctime=_ctime(timestamp),
            status="ok",
            exec_time=12.5,
            analysis="legacy analysis",
            metric=MetricValue(0.95098, maximize=True),
            is_buggy=False,
            research_mode="hypothesis",
            research_hypotheses_offered=["001234"],
        )
    )
    serialize.dump_json(journal, run_dir / "journal.json")

    assert not (artifact / "aide_result.json").exists()

    index = kaggle_submission_lab.refresh_index(
        logs_dir=logs_dir,
        index_path=logs_dir / "submission_index.json",
        competition="playground-series-s6e5",
        reindex=True,
    )

    assert (artifact / "aide_result.json").exists()
    assert len(index["records"]) == 1
    record = index["records"][0]
    assert record["kind"] == "source_node"
    assert record["run"] == run_name
    assert record["step"] == 0
    assert record["node_id"] == "node-legacy"
    assert record["local_score"] == 0.95098
    assert record["sha256"] == submission_sha
    assert record["hypothesis_id"] == "001234"
    assert record["profile"] == "full_boost_gpu"
    assert record["included_model_types"] == ["XGB", "GBM", "CAT"]
    assert record["algo"] == "AG"


def test_refresh_index_marks_non_autogluon_artifacts_as_legacy(tmp_path):
    logs_dir = tmp_path / "logs"
    artifact = _write_artifact(
        logs_dir,
        "legacy-run",
        "20260506T130000",
        code="print('legacy model')\n",
        submission="id,target\n1,0.9\n",
    )
    submission_sha = kaggle_submission_lab.sha256_file(artifact / "submission.csv")
    (artifact / "aide_result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "source_node",
                "run": "legacy-run",
                "timestamp": "20260506T130000",
                "status": "ok",
                "local_score": 0.94000,
                "metric_maximize": True,
                "is_buggy": False,
                "sha256": submission_sha,
                "node": {
                    "id": "node-legacy",
                    "step": 3,
                    "metric": {"value": 0.94000, "maximize": True},
                    "is_buggy": False,
                },
                "execution": {},
                "autogluon": {
                    "profile": None,
                    "presets": None,
                    "included_model_types": None,
                    "time_limit": None,
                    "resolved_settings": {},
                },
                "source": {},
            }
        )
    )

    index = kaggle_submission_lab.refresh_index(
        logs_dir=logs_dir,
        index_path=logs_dir / "submission_index.json",
        competition="playground-series-s6e5",
        reindex=True,
    )

    assert index["records"][0]["algo"] == "Leg"


def test_refresh_index_can_be_limited_to_specific_run(tmp_path):
    logs_dir = tmp_path / "logs"
    for run_name, score in [("run-a", 0.91), ("run-b", 0.93)]:
        artifact = _write_artifact(
            logs_dir,
            run_name,
            "20260506T130000",
            code="print('legacy model')\n",
            submission=f"id,target\n1,{score}\n",
        )
        submission_sha = kaggle_submission_lab.sha256_file(artifact / "submission.csv")
        (artifact / "aide_result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "source_node",
                    "run": run_name,
                    "timestamp": "20260506T130000",
                    "status": "ok",
                    "local_score": score,
                    "metric_maximize": True,
                    "is_buggy": False,
                    "sha256": submission_sha,
                    "node": {
                        "id": f"node-{run_name}",
                        "step": 3,
                        "metric": {"value": score, "maximize": True},
                        "is_buggy": False,
                    },
                    "execution": {},
                    "source": {},
                }
            )
        )

    index = kaggle_submission_lab.refresh_index(
        logs_dir=logs_dir,
        index_path=logs_dir / "submission_index.json",
        competition="playground-series-s6e5",
        runs=["run-b"],
        reindex=True,
    )

    assert [record["run"] for record in index["records"]] == ["run-b"]
    assert list(index["runs"]) == ["run-b"]


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


def test_refresh_index_rebuilds_old_index_version_for_eval_metric_backfill(tmp_path):
    logs_dir = tmp_path / "logs"
    artifact = _write_artifact(
        logs_dir,
        "run-a",
        "20260504T100000",
        code="AIDE_AG_CONFIG = {'included_model_types': ['XGB']}\n",
    )
    submission_sha = kaggle_submission_lab.sha256_file(artifact / "submission.csv")
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
                "sha256": submission_sha,
                "node": {
                    "id": "node-a",
                    "step": 0,
                    "metric": {"value": 0.9, "maximize": True},
                },
                "autogluon": {
                    "resolved_settings": {
                        "eval_metric": "balanced_accuracy",
                    },
                },
                "execution": {},
                "source": {},
            }
        )
    )
    index_path = logs_dir / "submission_index.json"
    index_path.write_text(
        json.dumps(
            {
                "version": 1,
                "competition": "playground-series-s6e5",
                "records": [
                    {
                        "kind": "source_node",
                        "run": "run-a",
                        "step": 0,
                        "timestamp": "20260504T100000",
                        "local_score": 0.9,
                        "metric_maximize": True,
                        "is_buggy": False,
                        "sha256": submission_sha,
                        "submission_path": str(artifact / "submission.csv"),
                        "algo": "AG",
                        "eval_metric": None,
                    }
                ],
                "runs": {
                    "run-a": kaggle_submission_lab.run_scan_signature(logs_dir / "run-a"),
                },
            }
        )
    )

    refreshed = kaggle_submission_lab.refresh_index(
        logs_dir=logs_dir,
        index_path=index_path,
        competition="playground-series-s6e5",
    )

    assert refreshed["version"] == kaggle_submission_lab.INDEX_VERSION
    assert refreshed["records"][0]["eval_metric"] == "balanced_accuracy"


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
    console = kaggle_submission_lab.Console(record=True, width=260, color_system=None)
    records = [
        {
            "kind": "source_node",
            "run": "2-intelligent-amber-bandicoot",
            "step": 1,
            "timestamp": "20260504T134159",
            "local_score": 0.95026,
            "exec_time": 142.4,
            "profile": "full_boost",
            "included_model_types": ["XGB", "GBM", "CAT"],
            "sha256": "13bc36ab26abcdef",
            "algo": "AG",
            "hypothesis_id": "001234",
        }
    ]

    kaggle_submission_lab.render_table(console, records)

    output = console.export_text()
    assert "src" not in output
    assert "prof" not in output
    assert "Algo" in output
    assert "AG" in output
    assert "time" in output
    assert "2.4m" in output
    assert "hyp" in output
    assert "001234" in output
    assert "20260504" in output


def test_candidate_display_table_shows_eval_metric_for_submit_ready_records():
    records = [
        {
            "kind": "profile_eval",
            "run": "2-text-run",
            "step": None,
            "timestamp": "20260504T134159",
            "local_score": 0.95026,
            "exec_time": 142.4,
            "sha256": "13bc36ab26abcdef",
            "algo": "AG",
            "eval_metric": "balanced_accuracy",
        }
    ]

    columns, rows = kaggle_submission_lab.candidate_display_table(records)

    metric_index = columns.index("metric")
    assert rows[0][metric_index] == "balanced_accuracy"


def test_parse_args_defaults_to_rich_output_format():
    args = kaggle_submission_lab.parse_args([])

    assert args.output_format == "rich"


def test_parse_args_treats_sha_as_sha256_alias():
    args = kaggle_submission_lab.parse_args(["--sha", "abc123", "--sha256", "def456"])

    assert args.sha256 == ["abc123", "def456"]


def test_parse_args_help_lists_sha_alias(capsys):
    with pytest.raises(SystemExit):
        kaggle_submission_lab.parse_args(["--help"])

    help_text = capsys.readouterr().out
    assert "--sha " in help_text


def test_render_text_table_uses_plain_rows_without_box_frames():
    console = kaggle_submission_lab.Console(record=True, width=260, color_system=None)
    records = [
        {
            "kind": "source_node",
            "run": "2-text-run",
            "step": 7,
            "timestamp": "20260504T134159",
            "local_score": 0.95026,
            "sha256": "13bc36ab26abcdef",
            "algo": "Leg",
            "hypothesis_id": "001234",
        }
    ]

    kaggle_submission_lab.render_text_table(
        console,
        "Plain candidates",
        *kaggle_submission_lab.candidate_display_table(records),
    )

    output = console.export_text()
    assert "Plain candidates" in output
    assert "2-text-run" in output
    assert "001234" in output
    assert "┏" not in output
    assert "└" not in output
    assert "┃" not in output


def test_build_json_output_payload_includes_selected_and_registry_rows(tmp_path):
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
                "remote_status": "COMPLETE",
                "public_score": "0.90123",
            }
        ],
    )
    selected = [
        {
            "kind": "source_node",
            "competition": "playground-series-s6e5",
            "run": "run-a",
            "step": 1,
            "timestamp": "20260504T100000",
            "local_score": 0.95,
            "metric_maximize": True,
            "is_buggy": False,
            "submission_path": str(tmp_path / "submission.csv"),
            "sha256": "aaaabbbbcccc",
            "hypothesis_id": "001234",
        }
    ]

    payload = kaggle_submission_lab.build_output_payload(
        selected=selected,
        registry=registry,
        remote_submissions=None,
        records=selected,
        full_view=False,
        registry_limit=20,
        run_filters=None,
    )

    assert payload["selected"][0]["hypothesis_id"] == "001234"
    assert payload["selected"][0]["sha256"] == "aaaabbbbcccc"
    assert payload["registry"][0]["run"] == "run-a"
    assert payload["registry"][0]["public_score"] == "0.90123"


def test_json_safe_serializes_datetime_values():
    value = dt.datetime(2026, 5, 27, 13, 40, 1)

    assert kaggle_submission_lab._json_safe({"when": value}) == {
        "when": "2026-05-27T13:40:01"
    }


def test_render_table_derives_legacy_algo_for_old_index_records(tmp_path):
    console = kaggle_submission_lab.Console(record=True, width=260, color_system=None)
    records = [
        {
            "kind": "source_node",
            "run": "2-legacy-run",
            "step": 9,
            "timestamp": "20260504T134159",
            "local_score": 0.94000,
            "sha256": "13bc36ab26abcdef",
        }
    ]

    kaggle_submission_lab.render_table(console, records)

    output = console.export_text()
    assert "Algo" in output
    assert "Leg" in output


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
                "algo": "Leg",
                "timestamp": "20260504T100000",
                "local_score": 0.95,
                "sha256": "aaaabbbbcccc",
                "remote_status": "ERROR",
                "public_score": None,
                "eval_metric": "balanced_accuracy",
            },
            {
                "competition": "playground-series-s6e5",
                "run": "run-b",
                "step": 2,
                "algo": "AG",
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
    assert "metric" in output
    assert "balanced_accuracy" in output
    assert "Algo" in output
    assert "AG" in output
    assert "Leg" in output
    assert "0.90123" in output
    assert "COMPLETE" in output
    assert "ERROR" in output


def test_registry_display_table_shows_metric_or_dash(tmp_path):
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
                "remote_status": "COMPLETE",
                "public_score": "0.90123",
                "eval_metric": "balanced_accuracy",
            },
            {
                "competition": "playground-series-s6e5",
                "run": "run-b",
                "step": 2,
                "timestamp": "20260504T110000",
                "local_score": 0.94,
                "sha256": "dddd11112222",
                "remote_status": "COMPLETE",
                "public_score": "0.90111",
            },
        ],
    )

    columns, rows = kaggle_submission_lab.registry_display_table(registry)

    metric_index = columns.index("metric")
    assert rows[0][metric_index] == "balanced_accuracy"
    assert rows[1][metric_index] == "-"


def test_registry_display_table_shows_exec_time_from_records_or_dash(tmp_path):
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
                "remote_status": "COMPLETE",
                "public_score": "0.90123",
            },
            {
                "competition": "playground-series-s6e5",
                "run": "run-b",
                "step": 2,
                "timestamp": "20260504T110000",
                "local_score": 0.94,
                "sha256": "dddd11112222",
                "remote_status": "COMPLETE",
                "public_score": "0.90111",
            },
        ],
    )
    records = [
        {
            "kind": "source_node",
            "run": "run-a",
            "step": 1,
            "timestamp": "20260504T100000",
            "sha256": "aaaabbbbcccc",
            "exec_time": 142.4,
        },
        {
            "kind": "source_node",
            "run": "seeded-run",
            "step": 0,
            "timestamp": "20260504T120000",
            "sha256": "aaaabbbbcccc",
            "artifact_dir": str(tmp_path / "seeded"),
            "exec_time": 142.4,
        }
    ]

    columns, rows = kaggle_submission_lab.registry_display_table(
        registry,
        records=records,
    )

    time_index = columns.index("time")
    assert rows[0][time_index] == "2.4m"
    assert rows[1][time_index] == "-"


def test_render_registry_table_limits_rows_by_default(tmp_path):
    registry = kaggle_submission_lab.smart.SubmissionRegistry(
        tmp_path / "registry.json",
        entries=[
            {
                "competition": "playground-series-s6e5",
                "run": f"run-{idx:02d}",
                "step": idx,
                "timestamp": f"20260504T10{idx:04d}",
                "local_score": 0.90 + idx / 1000,
                "sha256": f"{idx:012d}",
                "remote_status": "COMPLETE",
                "public_score": f"0.{90000 + idx}",
            }
            for idx in range(25)
        ],
    )
    console = kaggle_submission_lab.Console(record=True, width=160, color_system=None)

    kaggle_submission_lab.render_registry_table(console, registry)

    output = console.export_text()
    assert "run-24" in output
    assert "run-05" in output
    assert "run-04" not in output


def test_render_registry_table_filters_rows_by_run(tmp_path):
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
                "remote_status": "COMPLETE",
                "public_score": "0.90123",
            },
            {
                "competition": "playground-series-s6e5",
                "run": "run-b",
                "step": 2,
                "timestamp": "20260504T110000",
                "local_score": 0.96,
                "sha256": "dddd11112222",
                "remote_status": "COMPLETE",
                "public_score": "0.90111",
            },
        ],
    )

    class FakeRemote:
        file_name = "sub_20260504T120000_step-3_node-node-c_sha-eeeeffff00_cv-0.97000.csv"
        description = (
            "cv=0.97000 | run=run-c | step=3 | "
            "aide_ts=20260504T120000 | node=node-c | sha=eeeeffff00"
        )
        status = "COMPLETE"
        public_score = "0.90199"
        date = "2026-05-04T12:00:00Z"

    console = kaggle_submission_lab.Console(record=True, width=180, color_system=None)

    kaggle_submission_lab.render_registry_table(
        console,
        registry,
        [FakeRemote()],
        run_filters=["run-b"],
    )

    output = console.export_text()
    assert "run-b" in output
    assert "run-a" not in output
    assert "run-c" not in output


def test_render_registry_table_accepts_unlimited_rows(tmp_path):
    registry = kaggle_submission_lab.smart.SubmissionRegistry(
        tmp_path / "registry.json",
        entries=[
            {
                "competition": "playground-series-s6e5",
                "run": f"run-{idx:02d}",
                "step": idx,
                "timestamp": f"20260504T10{idx:04d}",
                "local_score": 0.90 + idx / 1000,
                "sha256": f"{idx:012d}",
                "remote_status": "COMPLETE",
                "public_score": f"0.{90000 + idx}",
            }
            for idx in range(25)
        ],
    )
    console = kaggle_submission_lab.Console(record=True, width=160, color_system=None)

    kaggle_submission_lab.render_registry_table(console, registry, limit=None)

    output = console.export_text()
    assert "run-24" in output
    assert "run-00" in output


def test_render_registry_table_backfills_algo_from_local_records(tmp_path):
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
                "remote_status": "COMPLETE",
                "public_score": "0.90123",
            },
            {
                "competition": "playground-series-s6e5",
                "run": "run-b",
                "step": 2,
                "timestamp": "20260504T110000",
                "local_score": 0.94,
                "sha256": "dddd11112222",
                "remote_status": "COMPLETE",
                "public_score": "0.90111",
            },
        ],
    )
    records = [
        {
            "kind": "source_node",
            "run": "run-a",
            "step": 1,
            "timestamp": "20260504T100000",
            "sha256": "aaaabbbbcccc",
            "artifact_dir": str(tmp_path / "logs" / "run-a" / "artifacts" / "20260504T100000"),
            "profile": "full_boost",
            "included_model_types": ["XGB", "GBM", "CAT"],
            "hypothesis_id": "001234",
        },
        {
            "kind": "source_node",
            "run": "run-b",
            "step": 2,
            "timestamp": "20260504T110000",
            "sha256": "dddd11112222",
            "artifact_dir": str(tmp_path / "logs" / "run-b" / "artifacts" / "20260504T110000"),
        },
    ]
    console = kaggle_submission_lab.Console(record=True, width=260, color_system=None)

    kaggle_submission_lab.render_registry_table(
        console,
        registry,
        records=records,
        full_view=True,
    )

    output = console.export_text()
    assert "AG" in output
    assert "Leg" in output
    assert "001234" in output
    assert "artifact" in output
    assert str(tmp_path / "logs" / "run-a" / "artifacts" / "20260504T100000") in output
    assert "?" not in output


def test_render_registry_table_full_view_uses_registry_submission_path(tmp_path):
    submission_path = tmp_path / "logs" / "run-a" / "artifacts" / "20260504T100000" / "submission.csv"
    submission_path.parent.mkdir(parents=True)
    submission_path.write_text("id,target\n1,0.8\n")
    (submission_path.parent / "solution.py").write_text(
        "AIDE_AG_CONFIG = {'included_model_types': ['XGB', 'GBM', 'CAT']}\n",
        encoding="utf-8",
    )
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
                "remote_status": "COMPLETE",
                "public_score": "0.90123",
                "submission_path": str(submission_path),
            },
        ],
    )
    console = kaggle_submission_lab.Console(record=True, width=180, color_system=None)

    kaggle_submission_lab.render_registry_table(console, registry, full_view=True)

    output = console.export_text()
    assert "artifact" in output
    assert "logs/run-a/artifacts" in output
    assert "AG" in output


def test_record_to_candidate_preserves_algo_for_kaggle_message_and_registry(tmp_path):
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.8\n")
    record = {
        "kind": "source_node",
        "competition": "playground-series-s6e5",
        "run": "2-ag-run",
        "step": 12,
        "node_id": "node-ag",
        "timestamp": "20260504T134159",
        "local_score": 0.95026,
        "metric_maximize": True,
        "is_buggy": False,
        "submission_path": str(submission_path),
        "sha256": "13bc36ab26abcdef",
        "profile": "full_boost",
        "included_model_types": ["XGB", "GBM", "CAT"],
    }

    candidate = kaggle_submission_lab._record_to_candidate(record)

    assert candidate.algo == "AG"


def test_record_to_candidate_accepts_suffixed_artifact_timestamp(tmp_path):
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,target\n1,0.8\n")
    record = {
        "kind": "source_node",
        "competition": "playground-series-s6e5",
        "run": "2-ag-run",
        "step": 12,
        "node_id": "node-ag",
        "timestamp": "20260504T134159-838a0027",
        "local_score": 0.95026,
        "metric_maximize": True,
        "is_buggy": False,
        "submission_path": str(submission_path),
        "sha256": "13bc36ab26abcdef",
    }

    candidate = kaggle_submission_lab._record_to_candidate(record)

    assert candidate.timestamp == "20260504T134159-838a0027"
    assert candidate.ctime == _ctime("20260504T134159")


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


def test_render_registry_table_backfills_remote_only_rows_from_sha_prefix(tmp_path):
    registry = kaggle_submission_lab.smart.SubmissionRegistry(
        tmp_path / "registry.json",
        entries=[],
    )
    artifact_dir = tmp_path / "logs" / "2-remote-server-run" / "artifacts" / "20260505T204454"

    class FakeRemote:
        file_name = "sub_20260505T204454_step--1_node-profile-_sha-165e7caae1_cv-0.95100.csv"
        description = (
            "cv=0.95100 | run=2-remote-server-run | step=-1 | "
            "aide_ts=20260505T204454 | node=profile- | sha=165e7caae1"
        )
        status = "COMPLETE"
        public_score = "0.95072"
        date = "2026-05-05T20:44:54Z"

    records = [
        {
            "kind": "profile_eval",
            "run": "2-remote-server-run",
            "step": None,
            "timestamp": "20260505T204454",
            "sha256": "165e7caae1abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "algo": "AG",
            "artifact_dir": str(artifact_dir),
        }
    ]
    console = kaggle_submission_lab.Console(record=True, width=280, color_system=None)

    kaggle_submission_lab.render_registry_table(
        console,
        registry,
        [FakeRemote()],
        records=records,
        full_view=True,
    )

    output = console.export_text()
    assert "AG" in output
    assert str(artifact_dir) in output


def test_render_registry_table_hides_remote_only_duplicate_by_sha_prefix(tmp_path):
    full_sha = "0aa5d277ee10f54230913379457b7695150ba7d9ec61df650f1b11d381187bd9"
    registry = kaggle_submission_lab.smart.SubmissionRegistry(
        tmp_path / "registry.json",
        entries=[
            {
                "competition": "playground-series-s6e5",
                "run": "2-nuthatch-of-lucky-tact",
                "step": 0,
                "timestamp": "20260510T021544",
                "sha256": full_sha,
                "local_score": 0.95224,
                "public_score": "0.95168",
                "remote_status": "COMPLETE",
            }
        ],
    )

    class FakeRemote:
        file_name = "sub_20260506T094019_step--1_node-profile-_sha-0aa5d277ee_cv-0.95224.csv"
        description = (
            "cv=0.95224 | run=2-enthusiastic-crane-of-completion | step=-1 | "
            "aide_ts=20260506T094019 | node=profile- | sha=0aa5d277ee"
        )
        status = "COMPLETE"
        public_score = "0.95168"
        date = "2026-05-06T09:40:19Z"

    console = kaggle_submission_lab.Console(record=True, width=180, color_system=None)

    kaggle_submission_lab.render_registry_table(console, registry, [FakeRemote()])

    output = console.export_text()
    assert "2-nuthatch-of-lucky-tact" in output
    assert "2-enthusiastic-crane-of-completion" not in output
