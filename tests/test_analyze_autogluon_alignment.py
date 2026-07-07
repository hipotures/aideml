import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_autogluon_alignment.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_autogluon_alignment", MODULE_PATH)
analyze_autogluon_alignment = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = analyze_autogluon_alignment
SPEC.loader.exec_module(analyze_autogluon_alignment)


def test_usable_registry_rows_filter_ag_complete_balanced_accuracy():
    payload = {
        "registry": [
            {
                "algo": "AG",
                "remote_status": "COMPLETE",
                "eval_metric": "balanced_accuracy",
                "local_score": 0.95,
                "public_score": "0.94",
                "sha256": "source-a",
            },
            {
                "algo": "Leg",
                "remote_status": "COMPLETE",
                "eval_metric": "balanced_accuracy",
                "local_score": 0.96,
                "public_score": "0.95",
                "sha256": "legacy-a",
            },
            {
                "algo": "AG",
                "remote_status": "PENDING",
                "eval_metric": "balanced_accuracy",
                "local_score": 0.97,
                "public_score": "0.96",
                "sha256": "pending-a",
            },
            {
                "algo": "AG",
                "remote_status": "COMPLETE",
                "eval_metric": "log_loss",
                "local_score": 0.98,
                "public_score": "0.97",
                "sha256": "wrong-metric",
            },
        ]
    }

    rows = analyze_autogluon_alignment.usable_registry_rows(payload)

    assert len(rows) == 1
    assert rows[0]["sha256"] == "source-a"
    assert rows[0]["local_score"] == 0.95
    assert rows[0]["public_score"] == 0.94


def test_profile_eval_rows_join_known_public_score_by_source_sha():
    registry_rows = [
        {
            "algo": "AG",
            "remote_status": "COMPLETE",
            "eval_metric": "balanced_accuracy",
            "local_score": 0.95,
            "public_score": "0.94",
            "sha256": "source-a",
            "run": "run-a",
            "step": 1,
        },
        {
            "algo": "AG",
            "remote_status": "COMPLETE",
            "eval_metric": "balanced_accuracy",
            "local_score": 0.96,
            "public_score": "0.955",
            "sha256": "source-b",
            "run": "run-b",
            "step": 2,
        },
    ]
    index_payload = {
        "records": [
            {
                "kind": "profile_eval",
                "status": "ok",
                "profile": "candidate_cv",
                "source_sha256": "source-a",
                "sha256": "eval-a",
                "local_score": 0.941,
                "eval_metric": "balanced_accuracy",
                "exec_time": 120.0,
                "time_limit": 600,
                "artifact_dir": "logs/run-a/artifacts/eval-a",
            },
            {
                "kind": "profile_eval",
                "status": "error",
                "profile": "candidate_cv",
                "source_sha256": "source-b",
                "sha256": "eval-b",
                "local_score": 0.956,
                "eval_metric": "balanced_accuracy",
            },
            {
                "kind": "profile_eval",
                "status": "ok",
                "profile": "candidate_cv",
                "source_sha256": "missing-source",
                "sha256": "eval-missing",
                "local_score": 0.956,
                "eval_metric": "balanced_accuracy",
            },
        ]
    }

    rows = analyze_autogluon_alignment.profile_eval_rows(
        registry_rows=registry_rows,
        index_records=index_payload["records"],
    )

    assert len(rows) == 1
    assert rows[0]["profile"] == "candidate_cv"
    assert rows[0]["sha256"] == "eval-a"
    assert rows[0]["source_sha256"] == "source-a"
    assert rows[0]["public_score"] == 0.94
    assert rows[0]["source_public_rank"] == 2


def test_summarize_agreement_computes_metrics_and_average_runtime():
    rows = [
        {"local_score": 0.91, "public_score": 0.91, "exec_time": 10.0},
        {"local_score": 0.92, "public_score": 0.92, "exec_time": 20.0},
        {"local_score": 0.93, "public_score": 0.93, "exec_time": 30.0},
    ]

    summary = analyze_autogluon_alignment.summarize_agreement(rows)

    assert summary["n"] == 3
    assert summary["pearson"] == 1.0
    assert summary["spearman"] == 1.0
    assert summary["mae"] == 0.0
    assert summary["bias"] == 0.0
    assert summary["avg_runtime_seconds"] == 20.0


def test_cli_writes_profile_summary_json(tmp_path):
    lab_json = tmp_path / "lab.json"
    index_json = tmp_path / "submission_index.json"
    output_json = tmp_path / "alignment.json"
    lab_json.write_text(
        json.dumps(
            {
                "registry": [
                    {
                        "algo": "AG",
                        "remote_status": "COMPLETE",
                        "eval_metric": "balanced_accuracy",
                        "local_score": 0.95,
                        "public_score": "0.94",
                        "sha256": "source-a",
                    },
                    {
                        "algo": "AG",
                        "remote_status": "COMPLETE",
                        "eval_metric": "balanced_accuracy",
                        "local_score": 0.96,
                        "public_score": "0.95",
                        "sha256": "source-b",
                    },
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
                        "profile": "candidate_cv",
                        "source_sha256": "source-a",
                        "sha256": "eval-a",
                        "local_score": 0.941,
                        "eval_metric": "balanced_accuracy",
                        "exec_time": 120.0,
                    },
                    {
                        "kind": "profile_eval",
                        "status": "ok",
                        "profile": "candidate_cv",
                        "source_sha256": "source-b",
                        "sha256": "eval-b",
                        "local_score": 0.951,
                        "eval_metric": "balanced_accuracy",
                        "exec_time": 180.0,
                    },
                ]
            }
        )
    )

    result = analyze_autogluon_alignment.main(
        [
            "--lab-json",
            str(lab_json),
            "--index",
            str(index_json),
            "--output",
            str(output_json),
        ]
    )

    assert result == 0
    payload = json.loads(output_json.read_text())
    assert payload["baseline"]["n"] == 2
    assert payload["profiles"][0]["profile"] == "candidate_cv"
    assert payload["profiles"][0]["n"] == 2
    assert len(payload["profile_rows"]) == 2
