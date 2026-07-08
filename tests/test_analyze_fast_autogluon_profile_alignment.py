import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_fast_autogluon_profile_alignment.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_fast_autogluon_profile_alignment", MODULE_PATH
)
analyze_fast_alignment = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = analyze_fast_alignment
SPEC.loader.exec_module(analyze_fast_alignment)


def test_source_table_filters_submitted_ag_balanced_accuracy_rows():
    payload = {
        "registry": [
            {
                "algo": "AG",
                "remote_status": "COMPLETE",
                "eval_metric": "balanced_accuracy",
                "run": "run-a",
                "step": 3,
                "sha256": "source-a",
                "local_score": "0.95",
                "public_score": "0.94",
                "exec_time": "0.8m",
                "artifact_dir": "logs/run-a/artifacts/a",
                "has_source_rerun": False,
                "source_sha256": None,
                "source_solution_path": None,
                "manual_status": None,
                "manual_invalid_reason": None,
            },
            {
                "algo": "AG",
                "remote_status": "PENDING",
                "eval_metric": "balanced_accuracy",
                "sha256": "pending-source",
                "local_score": "0.96",
                "public_score": "0.95",
            },
            {
                "algo": "Leg",
                "remote_status": "COMPLETE",
                "eval_metric": "balanced_accuracy",
                "sha256": "legacy-source",
                "local_score": "0.96",
                "public_score": "0.95",
            },
        ]
    }

    rows = analyze_fast_alignment.source_table(payload, index_payload={"records": []})

    assert len(rows) == 1
    assert rows[0]["run"] == "run-a"
    assert rows[0]["step"] == 3
    assert rows[0]["sha256"] == "source-a"
    assert rows[0]["algo"] == "AG"
    assert rows[0]["local_score"] == 0.95
    assert rows[0]["public_score"] == 0.94
    assert rows[0]["signed_gap"] == 0.010000000000000009
    assert rows[0]["exec_time_seconds"] == 48.0


def test_profile_rows_classify_medium_fast_candidates_and_full_references():
    lab_payload = {
        "registry": [
            {
                "algo": "AG",
                "remote_status": "COMPLETE",
                "eval_metric": "balanced_accuracy",
                "run": "source-run",
                "step": 1,
                "sha256": "source-a",
                "local_score": 0.95,
                "public_score": "0.94",
                "artifact_dir": "logs/source/artifacts/a",
            }
        ]
    }
    index_payload = {
        "records": [
            {
                "kind": "profile_eval",
                "status": "ok",
                "competition": "playground-series-s6e7",
                "profile": "fast_medium",
                "autogluon_presets": "medium_quality",
                "time_limit": 600,
                "source_sha256": "source-a",
                "sha256": "eval-a",
                "local_score": 0.941,
                "eval_metric": "balanced_accuracy",
                "exec_time": 75.0,
            },
            {
                "kind": "profile_eval",
                "status": "ok",
                "competition": "playground-series-s6e7",
                "profile": "best_reference",
                "autogluon_presets": "best",
                "time_limit": 3600,
                "source_sha256": "source-a",
                "sha256": "eval-best",
                "local_score": 0.942,
                "eval_metric": "balanced_accuracy",
                "exec_time": 900.0,
            },
            {
                "kind": "profile_eval",
                "status": "ok",
                "competition": "playground-series-s6e6",
                "profile": "wrong_competition",
                "autogluon_presets": "medium_quality",
                "time_limit": 600,
                "source_sha256": "source-a",
                "sha256": "eval-wrong",
                "local_score": 0.943,
                "eval_metric": "balanced_accuracy",
                "exec_time": 80.0,
            },
        ]
    }

    payload = analyze_fast_alignment.build_analysis_payload(
        lab_payload=lab_payload,
        index_payload=index_payload,
        competition="playground-series-s6e7",
    )

    assert [row["profile"] for row in payload["new_fast_candidate_rows"]] == [
        "fast_medium"
    ]
    assert [row["profile"] for row in payload["historical_full_reference_rows"]] == [
        "best_reference"
    ]
    assert [row["profile"] for row in payload["excluded_profile_rows"]] == [
        "wrong_competition"
    ]


def test_task_start_time_accepts_timezone_aware_start_and_compact_index_timestamp():
    lab_payload = {
        "registry": [
            {
                "algo": "AG",
                "remote_status": "COMPLETE",
                "eval_metric": "balanced_accuracy",
                "run": "source-run",
                "step": 1,
                "sha256": "source-a",
                "local_score": 0.95,
                "public_score": "0.94",
                "artifact_dir": "logs/source/artifacts/a",
            }
        ]
    }
    index_payload = {
        "records": [
            {
                "kind": "profile_eval",
                "status": "ok",
                "competition": "playground-series-s6e7",
                "profile": "fast_medium",
                "autogluon_presets": "medium_quality",
                "time_limit": 600,
                "source_sha256": "source-a",
                "sha256": "eval-a",
                "local_score": 0.941,
                "eval_metric": "balanced_accuracy",
                "exec_time": 75.0,
                "timestamp": "20260708T011600",
            }
        ]
    }

    payload = analyze_fast_alignment.build_analysis_payload(
        lab_payload=lab_payload,
        index_payload=index_payload,
        competition="playground-series-s6e7",
        task_start_time="2026-07-08T01:15:58+02:00",
    )

    assert [row["profile"] for row in payload["current_task_fast_candidate_rows"]] == [
        "fast_medium"
    ]


def test_profile_summary_reports_rank_error_runtime_and_topk_metrics():
    rows = [
        {
            "profile": "candidate",
            "source_sha256": "source-a",
            "local_score": 0.91,
            "public_score": 0.91,
            "exec_time": 10.0,
        },
        {
            "profile": "candidate",
            "source_sha256": "source-b",
            "local_score": 0.92,
            "public_score": 0.93,
            "exec_time": 20.0,
        },
        {
            "profile": "candidate",
            "source_sha256": "source-c",
            "local_score": 0.93,
            "public_score": 0.92,
            "exec_time": 30.0,
        },
    ]

    summary = analyze_fast_alignment.summarize_profiles(rows)[0]

    assert summary["profile"] == "candidate"
    assert summary["n"] == 3
    assert summary["source_sha256s"] == ["source-a", "source-b", "source-c"]
    assert summary["pearson"] == 0.5
    assert summary["spearman"] == 0.5
    assert summary["mae"] == 0.006666666666666672
    assert summary["bias"] == 0.0
    assert summary["median_absolute_error"] == pytest.approx(0.01)
    assert summary["top_2_hit_rate"] == 1.0
    assert summary["avg_runtime_seconds"] == 20.0
    assert summary["max_runtime_seconds"] == 30.0
    assert summary["failure_rate"] == 0.0
    assert summary["worst_over_optimistic"][0]["source_sha256"] == "source-c"
    assert summary["worst_under_optimistic"][0]["source_sha256"] == "source-b"


def test_profile_summary_reports_bias_corrected_mae():
    rows = [
        {
            "profile": "candidate",
            "source_sha256": "source-a",
            "local_score": 0.9510,
            "public_score": 0.9500,
        },
        {
            "profile": "candidate",
            "source_sha256": "source-b",
            "local_score": 0.9522,
            "public_score": 0.9510,
        },
        {
            "profile": "candidate",
            "source_sha256": "source-c",
            "local_score": 0.9531,
            "public_score": 0.9520,
        },
    ]

    summary = analyze_fast_alignment.summarize_profiles(rows)[0]

    assert summary["mae"] == pytest.approx(0.0011)
    assert summary["bias"] == pytest.approx(0.0011)
    assert summary["bias_corrected_mae"] == pytest.approx(0.00006666666666667024)
    assert summary["loo_bias_corrected_mae"] == pytest.approx(0.0001)


def test_cli_writes_json_and_csv_outputs(tmp_path):
    lab_json = tmp_path / "lab.json"
    index_json = tmp_path / "index.json"
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "summary.csv"
    sources_csv = tmp_path / "sources.csv"
    lab_json.write_text(
        json.dumps(
            {
                "registry": [
                    {
                        "algo": "AG",
                        "remote_status": "COMPLETE",
                        "eval_metric": "balanced_accuracy",
                        "run": "source-run",
                        "step": 1,
                        "sha256": "source-a",
                        "local_score": 0.95,
                        "public_score": "0.94",
                        "artifact_dir": "logs/source/artifacts/a",
                    }
                ]
            }
        )
    )
    index_json.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "kind": "profile_eval",
                        "status": "ok",
                        "competition": "playground-series-s6e7",
                        "profile": "fast_medium",
                        "autogluon_presets": "medium_quality",
                        "time_limit": 600,
                        "source_sha256": "source-a",
                        "sha256": "eval-a",
                        "local_score": 0.941,
                        "eval_metric": "balanced_accuracy",
                        "exec_time": 75.0,
                    }
                ]
            }
        )
    )

    result = analyze_fast_alignment.main(
        [
            "--lab-json",
            str(lab_json),
            "--index",
            str(index_json),
            "--competition",
            "playground-series-s6e7",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--sources-csv",
            str(sources_csv),
        ]
    )

    assert result == 0
    payload = json.loads(output_json.read_text())
    assert payload["new_fast_candidate_profiles"][0]["profile"] == "fast_medium"
    assert "profile,n,pearson,spearman" in output_csv.read_text()
    assert "run,step,sha256" in sources_csv.read_text()
