import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "autogluon_fast_profile_calibration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "autogluon_fast_profile_calibration", MODULE_PATH
)
calibration = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def _completed_event(
    *,
    session_id: str,
    profile: str,
    source: str,
    local: float,
    public: float,
    valid: bool = True,
    trained: bool = True,
    eligible: bool | None = None,
    panel: str = "development",
):
    eligible = trained if eligible is None else eligible
    trained_types = calibration.REQUIRED_MODEL_FAMILY if trained else []
    eligible_types = calibration.REQUIRED_MODEL_FAMILY if eligible else []
    return {
        "session_id": session_id,
        "event_type": "completed",
        "profile": profile,
        "source_sha256": source,
        "source_public_score": public,
        "local_cv": local,
        "source_feature_signature": {"panel": panel},
        "status": "ok",
        "valid": valid,
        "all_required_model_types_trained": trained,
        "configured_model_types": calibration.REQUIRED_MODEL_FAMILY,
        "successfully_trained_model_types": trained_types,
        "eligible_for_selection_model_types": eligible_types,
        "preset": "medium_quality",
        "time_limit": 600,
        "source_code_unchanged": True,
        "wall_clock_seconds": 60.0,
    }


def _write_source_sets(
    session_dir: Path, *, development: list[str], confirmation: list[str]
) -> None:
    session_dir.joinpath("source_sets.json").write_text(
        json.dumps(
            {
                "panels": {
                    "development": {
                        "sources": [
                            {"source_sha256": source} for source in development
                        ]
                    },
                    "confirmation": {
                        "sources": [
                            {"source_sha256": source} for source in confirmation
                        ]
                    },
                }
            }
        )
    )


def test_session_metrics_exclude_historical_and_invalid_reruns(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session_id = "fresh-session"
    (session_dir / "state.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "task_start_time": "2026-07-09T20:44:43Z",
                "task_deadline": "2026-07-10T08:44:43Z",
            }
        )
    )
    _write_source_sets(
        session_dir,
        development=["source-a", "source-b", "source-c"],
        confirmation=["confirmation-source"],
    )
    events = [
        _completed_event(
            session_id=session_id,
            profile="profile-a",
            source="source-a",
            local=0.91,
            public=0.90,
        ),
        _completed_event(
            session_id=session_id,
            profile="profile-a",
            source="source-b",
            local=0.92,
            public=0.92,
        ),
        _completed_event(
            session_id=session_id,
            profile="profile-a",
            source="source-c",
            local=0.93,
            public=0.91,
        ),
        _completed_event(
            session_id=session_id,
            profile="profile-a",
            source="invalid-source",
            local=1.0,
            public=0.0,
            valid=False,
            trained=False,
        ),
        _completed_event(
            session_id="historical-session",
            profile="historical-winner",
            source="historical-source",
            local=1.0,
            public=1.0,
        ),
    ]
    (session_dir / "experiments.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    metrics = calibration.refresh_session_outputs(session_dir)
    summary = metrics["profiles"][0]
    state = json.loads((session_dir / "state.json").read_text())

    assert metrics["completed_experiments"] == 4
    assert [profile["profile"] for profile in metrics["profiles"]] == ["profile-a"]
    assert summary["valid"] == 3
    assert summary["invalid"] == 1
    assert summary["required_model_training_success_rate"] == 0.75
    assert summary["top_3_overlap"] == 1.0
    assert summary["top_5_overlap"] is None
    assert state["experiments"]["attempted"] == 4
    assert state["experiments"]["invalid"] == 1
    assert summary["complete_frozen_panel"] is True
    assert state["current_best_development_profile"] is None


def test_profile_metrics_report_all_required_ranking_and_runtime_fields():
    events = [
        _completed_event(
            session_id="fresh",
            profile="profile-a",
            source=f"source-{index}",
            local=0.90 + index / 1000,
            public=0.89 + index / 1000,
        )
        for index in range(5)
    ]

    payload = calibration._profile_metrics(events, session_id="fresh")
    summary = payload["profiles"][0]

    assert summary["pearson"] is not None
    assert summary["spearman"] is not None
    assert summary["top_3_overlap"] == 1.0
    assert summary["top_5_overlap"] == 1.0
    assert summary["mae"] is not None
    assert summary["median_absolute_error"] is not None
    assert summary["signed_bias"] is not None
    assert summary["mean_runtime_seconds"] == 60.0
    assert summary["median_runtime_seconds"] == 60.0
    assert summary["max_runtime_seconds"] == 60.0
    assert summary["source_subset_sensitivity"]["method"] == "leave_one_source_out"
    assert summary["spearman_uncertainty"]["method"] == "paired_bootstrap"


def test_metrics_reject_valid_flag_when_a_required_family_is_not_eligible():
    event = _completed_event(
        session_id="fresh",
        profile="profile-a",
        source="source-a",
        local=0.91,
        public=0.90,
        valid=True,
        trained=True,
        eligible=False,
    )

    payload = calibration._profile_metrics([event], session_id="fresh")
    summary = payload["profiles"][0]

    assert summary["attempted"] == 1
    assert summary["valid"] == 0
    assert summary["invalid"] == 1


def test_confirmation_and_incomplete_profiles_do_not_select_development_winner(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session_id = "fresh-session"
    session_dir.joinpath("state.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "task_start_time": "2026-07-09T20:44:43Z",
                "task_deadline": "2026-07-10T08:44:43Z",
                "frozen_finalists": [],
            }
        )
    )
    _write_source_sets(
        session_dir,
        development=["dev-a", "dev-b"],
        confirmation=["confirm-a", "confirm-b"],
    )
    events = [
        _completed_event(
            session_id=session_id,
            profile="profile-complete",
            source="dev-a",
            local=0.91,
            public=0.90,
        ),
        _completed_event(
            session_id=session_id,
            profile="profile-complete",
            source="dev-b",
            local=0.92,
            public=0.91,
        ),
        _completed_event(
            session_id=session_id,
            profile="profile-incomplete",
            source="dev-a",
            local=0.99,
            public=0.90,
        ),
        _completed_event(
            session_id=session_id,
            profile="profile-confirmation-only",
            source="confirm-a",
            local=0.99,
            public=0.90,
            panel="confirmation",
        ),
        _completed_event(
            session_id=session_id,
            profile="profile-confirmation-only",
            source="confirm-b",
            local=1.00,
            public=0.91,
            panel="confirmation",
        ),
    ]
    session_dir.joinpath("experiments.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    metrics = calibration.refresh_session_outputs(session_dir)
    state = json.loads(session_dir.joinpath("state.json").read_text())

    assert [summary["profile"] for summary in metrics["profiles"]] == [
        "profile-complete",
        "profile-incomplete",
    ]
    assert [summary["profile"] for summary in metrics["confirmation"]["profiles"]] == [
        "profile-confirmation-only"
    ]
    assert state["development_comparison_ready"] is False
    assert state["current_best_development_profile"] is None
    assert state["confirmation_status"] == "protocol_breach_unfrozen_profile"


def test_refresh_fails_closed_without_session_id(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session_dir.joinpath("state.json").write_text("{}")

    with pytest.raises(ValueError, match="session_id"):
        calibration.refresh_session_outputs(session_dir)


def test_reconcile_eligibility_evidence_uses_can_infer_metadata(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session_id = "fresh-session"
    session_dir.joinpath("state.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "task_start_time": "2026-07-09T20:44:43Z",
                "task_deadline": "2026-07-10T08:44:43Z",
            }
        )
    )
    _write_source_sets(
        session_dir,
        development=["source-a"],
        confirmation=["confirmation-source"],
    )
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    artifact_dir.joinpath("submission_eval.json").write_text(
        json.dumps(
            {
                "artifact_role": "profile_calibration_rerun",
                "profile_calibration_session_id": session_id,
                "status": "ok",
                "source_code_unchanged": True,
                "run_stats": {
                    "models": [
                        {"model": "XGBoost", "can_infer": True},
                        {"model": "LightGBM", "can_infer": True},
                        {"model": "CatBoost", "can_infer": True},
                    ]
                },
            }
        )
    )
    event = _completed_event(
        session_id=session_id,
        profile="profile-a",
        source="source-a",
        local=0.91,
        public=0.90,
    )
    event.pop("successfully_trained_model_types")
    event.pop("eligible_for_selection_model_types")
    event["experiment_id"] = "fresh-session-exp-001"
    event["artifact_dir"] = str(artifact_dir)
    session_dir.joinpath("experiments.jsonl").write_text(json.dumps(event) + "\n")

    assert calibration.reconcile_eligibility_evidence(session_dir) == 1
    metrics = calibration.refresh_session_outputs(session_dir)

    assert metrics["development"]["profiles"][0]["valid"] == 1
    assert calibration.reconcile_eligibility_evidence(session_dir) == 0


def test_freezing_finalist_records_config_hash_and_confirmation_state(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session_id = "fresh-session"
    profile = (
        "s6e7_calibration_reference_holdout20_unweighted_gpu_cuda_"
        "fairone_seed1729_10m"
    )
    session_dir.joinpath("state.json").write_text(
        json.dumps(
            {
                "session_id": session_id,
                "task_start_time": "2026-07-09T20:44:43Z",
                "task_deadline": "2026-07-10T08:44:43Z",
                "frozen_finalists": [],
            }
        )
    )
    _write_source_sets(
        session_dir,
        development=["source-a", "source-b"],
        confirmation=["confirmation-source"],
    )
    events = [
        _completed_event(
            session_id=session_id,
            profile=profile,
            source="source-a",
            local=0.91,
            public=0.90,
        ),
        _completed_event(
            session_id=session_id,
            profile=profile,
            source="source-b",
            local=0.92,
            public=0.91,
        ),
    ]
    session_dir.joinpath("experiments.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    calibration.freeze_finalists(session_dir, profiles=[profile])
    state = json.loads(session_dir.joinpath("state.json").read_text())

    assert state["frozen_finalists"] == [profile]
    assert state["confirmation_status"] == "awaiting_confirmation"
    assert state["frozen_finalist_profile_config_hashes"][profile]


def test_set_reference_profile_validates_and_records_state(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session_dir.joinpath("state.json").write_text(
        json.dumps({"session_id": "fresh-session"})
    )
    profile = "s6e7_calibration_reference_holdout20_unweighted_cpu_fairone_seed1729_10m"

    calibration.set_reference_profile(
        session_dir,
        profile=profile,
        competition="playground-series-s6e7",
    )

    state = json.loads(session_dir.joinpath("state.json").read_text())
    assert state["reference_profile"] == profile


def test_source_lookup_uses_frozen_solution_path_for_duplicate_submission_sha(tmp_path):
    first_solution = tmp_path / "first.py"
    second_solution = tmp_path / "second.py"
    first_solution.write_text("def preprocess(df):\n    return df\n")
    second_solution.write_text("def preprocess(df):\n    return df\n")
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "competition": "playground-series-s6e7",
                        "kind": "source_node",
                        "sha256": "duplicate-submission",
                        "run": "first-run",
                        "solution_path": str(first_solution),
                    },
                    {
                        "competition": "playground-series-s6e7",
                        "kind": "source_node",
                        "sha256": "duplicate-submission",
                        "run": "second-run",
                        "solution_path": str(second_solution),
                    },
                ]
            }
        )
    )

    record = calibration._source_record(
        index_path=index_path,
        source_sha256="duplicate-submission",
        source_solution_path=str(second_solution),
        competition="playground-series-s6e7",
    )

    assert record["run"] == "second-run"
