import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

from scripts import kaggle_submission_lab


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "auto_blend_submissions.py"
SPEC = importlib.util.spec_from_file_location("auto_blend_submissions", MODULE_PATH)
auto_blend_submissions = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = auto_blend_submissions
SPEC.loader.exec_module(auto_blend_submissions)


def _record(
    tmp_path: Path,
    *,
    name: str,
    sha: str,
    score: float,
    target_values: list[str],
    pred_values: list[str],
    test_values: list[str],
    include_oof: bool = True,
) -> dict:
    artifact_dir = tmp_path / "logs" / f"run-{name}" / "artifacts" / f"20260601T00000{name}"
    artifact_dir.mkdir(parents=True)
    pd.DataFrame({"id": [1, 2, 3], "class": test_values}).to_csv(
        artifact_dir / "test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame({"id": [1, 2, 3], "class": test_values}).to_csv(
        artifact_dir / "submission.csv",
        index=False,
    )
    if include_oof:
        pd.DataFrame(
            {
                "row": [0, 1, 2],
                "target": target_values,
                "prediction": pred_values,
            }
        ).to_csv(artifact_dir / "oof_predictions.csv.gz", index=False, compression="gzip")
    return {
        "competition": "playground-series-s6e6",
        "run": f"run-{name}",
        "timestamp": f"20260601T00000{name}",
        "artifact_dir": str(artifact_dir),
        "status": "ok",
        "is_buggy": False,
        "local_score": score,
        "metric_maximize": True,
        "eval_metric": "balanced_accuracy",
        "step": int(name),
        "node_id": f"node-{name}",
        "sha256": sha,
        "submission_path": str(artifact_dir / "submission.csv"),
    }


def test_categorical_oof_blend_uses_weighted_vote_and_scores_accuracy(tmp_path):
    records = [
        _record(
            tmp_path,
            name="1",
            sha="a" * 64,
            score=0.9,
            target_values=["A", "B", "B"],
            pred_values=["A", "B", "B"],
            test_values=["A", "B", "B"],
        ),
        _record(
            tmp_path,
            name="2",
            sha="b" * 64,
            score=0.8,
            target_values=["A", "B", "B"],
            pred_values=["A", "A", "B"],
            test_values=["A", "A", "B"],
        ),
    ]

    spec = auto_blend_submissions.BlendSpec(
        blend_kind="oof",
        mode="vote",
        weighting="uniform",
        component_shas=tuple(record["sha256"] for record in records),
        weights=(0.7, 0.3),
        params={},
    )
    result = auto_blend_submissions.evaluate_blend_spec(
        spec,
        records,
        id_col="id",
        target_col="class",
        metric_name="balanced_accuracy",
    )

    assert result.score == 1.0
    assert result.submission["class"].tolist() == ["A", "B", "B"]


def test_select_new_blends_splits_count_and_skips_existing_recipe(tmp_path):
    records = [
        _record(tmp_path, name="1", sha="a" * 64, score=0.91, target_values=["A", "B", "B"], pred_values=["A", "B", "B"], test_values=["A", "B", "B"]),
        _record(tmp_path, name="2", sha="b" * 64, score=0.89, target_values=["A", "B", "B"], pred_values=["A", "B", "A"], test_values=["A", "B", "A"]),
        _record(tmp_path, name="3", sha="c" * 64, score=0.88, target_values=["A", "B", "B"], pred_values=["A", "A", "B"], test_values=["A", "A", "B"]),
        _record(tmp_path, name="4", sha="d" * 64, score=0.87, target_values=["A", "B", "B"], pred_values=["B", "B", "B"], test_values=["B", "B", "B"], include_oof=False),
    ]
    existing = auto_blend_submissions.BlendSpec(
        blend_kind="oof",
        mode="vote",
        weighting="uniform",
        component_shas=("a" * 64, "b" * 64),
        weights=(0.5, 0.5),
        params={},
    ).recipe_hash

    selected = auto_blend_submissions.select_new_blends(
        records,
        count=3,
        existing_submission_sha256=set(),
        existing_recipe_hashes={existing},
        id_col="id",
        target_col="class",
        metric_name="balanced_accuracy",
    )

    assert len(selected) == 3
    assert selected[0].spec.blend_kind == "oof"
    assert any(item.spec.blend_kind == "submission" for item in selected)
    assert existing not in {item.spec.recipe_hash for item in selected}


def test_enrich_records_with_registry_adds_public_score(tmp_path):
    registry = auto_blend_submissions.smart.SubmissionRegistry(
        tmp_path / "registry.json",
        [
            {
                "competition": "playground-series-s6e6",
                "sha256": "a" * 64,
                "public_score": "0.96842",
                "remote_status": "COMPLETE",
                "blend_component_sha256": {
                    "a": "b" * 64,
                    "c": "d" * 64,
                },
            }
        ],
    )

    records = auto_blend_submissions.enrich_records_with_registry(
        [{"competition": "playground-series-s6e6", "sha256": "a" * 64}],
        registry,
        competition="playground-series-s6e6",
    )

    assert records[0]["public_score"] == "0.96842"
    assert records[0]["remote_status"] == "COMPLETE"
    assert records[0]["blend_component_sha256"] == {
        "a": "b" * 64,
        "c": "d" * 64,
    }


def test_bad_blend_component_sets_use_public_score_feedback():
    good_components = {"a": "a" * 64, "b": "b" * 64}
    bad_components = {"a": "a" * 64, "c": "c" * 64}
    records = [
        {
            "run": "blended",
            "origin": "auto_blend",
            "public_score": "0.96840",
            "blend_component_sha256": good_components,
        },
        {
            "run": "blended",
            "origin": "auto_blend",
            "public_score": "0.96790",
            "blend_component_sha256": bad_components,
        },
    ]

    blocked = auto_blend_submissions.bad_blend_component_sets(records, public_margin=0.00005)

    assert blocked == [frozenset(bad_components.values())]


def test_select_new_blends_skips_candidates_similar_to_bad_public_blend(tmp_path):
    records = [
        _record(tmp_path, name="1", sha="a" * 64, score=0.91, target_values=["A", "B", "B"], pred_values=["A", "B", "B"], test_values=["A", "B", "B"]),
        _record(tmp_path, name="2", sha="b" * 64, score=0.89, target_values=["A", "B", "B"], pred_values=["A", "B", "A"], test_values=["A", "B", "A"]),
        _record(tmp_path, name="3", sha="c" * 64, score=0.88, target_values=["A", "B", "B"], pred_values=["A", "A", "B"], test_values=["A", "A", "B"]),
    ]
    bad_components = [frozenset([records[0]["sha256"], records[1]["sha256"]])]

    selected = auto_blend_submissions.select_new_blends(
        records,
        count=2,
        existing_submission_sha256=set(),
        existing_recipe_hashes=set(),
        id_col="id",
        target_col="class",
        metric_name="balanced_accuracy",
        bad_component_sets=bad_components,
        bad_blend_overlap=1.0,
    )

    assert selected
    assert all(
        frozenset(result.spec.component_shas) != bad_components[0]
        for result in selected
    )


def test_select_new_blends_reports_progress(tmp_path):
    records = [
        _record(tmp_path, name="1", sha="a" * 64, score=0.91, target_values=["A", "B", "B"], pred_values=["A", "B", "B"], test_values=["A", "B", "B"]),
        _record(tmp_path, name="2", sha="b" * 64, score=0.89, target_values=["A", "B", "B"], pred_values=["A", "B", "A"], test_values=["A", "B", "A"]),
        _record(tmp_path, name="3", sha="c" * 64, score=0.88, target_values=["A", "B", "B"], pred_values=["A", "A", "B"], test_values=["A", "A", "B"]),
    ]
    events = []

    auto_blend_submissions.select_new_blends(
        records,
        count=2,
        existing_submission_sha256=set(),
        existing_recipe_hashes=set(),
        id_col="id",
        target_col="class",
        metric_name="balanced_accuracy",
        progress_callback=lambda done, total, kind: events.append((done, total, kind)),
    )

    assert events
    assert events[-1][0] == events[-1][1]
    assert {event[2] for event in events} == {"oof", "submission"}


def test_select_new_blends_caches_prediction_file_reads(tmp_path, monkeypatch):
    records = [
        _record(tmp_path, name="1", sha="a" * 64, score=0.91, target_values=["A", "B", "B"], pred_values=["A", "B", "B"], test_values=["A", "B", "B"]),
        _record(tmp_path, name="2", sha="b" * 64, score=0.89, target_values=["A", "B", "B"], pred_values=["A", "B", "A"], test_values=["A", "B", "A"]),
        _record(tmp_path, name="3", sha="c" * 64, score=0.88, target_values=["A", "B", "B"], pred_values=["A", "A", "B"], test_values=["A", "A", "B"]),
    ]
    original = auto_blend_submissions.read_prediction_file
    reads = {}

    def counted(path):
        reads[str(path)] = reads.get(str(path), 0) + 1
        return original(path)

    monkeypatch.setattr(auto_blend_submissions, "read_prediction_file", counted)

    auto_blend_submissions.select_new_blends(
        records,
        count=2,
        existing_submission_sha256=set(),
        existing_recipe_hashes=set(),
        id_col="id",
        target_col="class",
        metric_name="balanced_accuracy",
        jobs=2,
    )

    assert reads
    assert max(reads.values()) == 1


def test_categorical_balanced_accuracy_uses_fast_path(monkeypatch):
    def fail_sklearn(*args, **kwargs):
        raise AssertionError("sklearn balanced_accuracy_score should not be used")

    monkeypatch.setattr(
        auto_blend_submissions,
        "balanced_accuracy_score",
        fail_sklearn,
    )

    score = auto_blend_submissions.score_oof(
        pd.Series(["A", "A", "B", "B"]),
        pd.Series(["A", "B", "B", "B"]),
        "balanced_accuracy",
    )

    assert score == 0.75


def test_categorical_blend_avoids_sorting_full_stacked_array(monkeypatch):
    def fail_unique(*args, **kwargs):
        raise AssertionError("np.unique should not sort the full prediction matrix")

    monkeypatch.setattr(auto_blend_submissions.np, "unique", fail_unique)

    blended = auto_blend_submissions.blend_categorical_columns(
        [
            pd.Series(["B", "A", "A", "B"]),
            pd.Series(["A", "A", "B", "B"]),
            pd.Series(["A", "B", "B", "B"]),
        ],
        (0.2, 0.3, 0.5),
    )

    assert blended.tolist() == ["A", "A", "B", "B"]


def test_evaluate_blend_spec_does_not_hash_full_candidate_dataframe(tmp_path, monkeypatch):
    records = [
        _record(tmp_path, name="1", sha="a" * 64, score=0.91, target_values=["A", "B", "B"], pred_values=["A", "B", "B"], test_values=["A", "B", "B"]),
        _record(tmp_path, name="2", sha="b" * 64, score=0.89, target_values=["A", "B", "B"], pred_values=["A", "B", "A"], test_values=["A", "B", "A"]),
    ]

    def fail_fingerprint(*args, **kwargs):
        raise AssertionError("candidate dataframe fingerprints should not be computed")

    monkeypatch.setattr(auto_blend_submissions, "fast_dataframe_fingerprint", fail_fingerprint)

    spec = auto_blend_submissions.BlendSpec(
        blend_kind="submission",
        mode="vote",
        weighting="uniform",
        component_shas=tuple(record["sha256"] for record in records),
        weights=(0.5, 0.5),
        params={},
    )
    result = auto_blend_submissions.evaluate_blend_spec(
        spec,
        records,
        id_col="id",
        target_col="class",
        metric_name="balanced_accuracy",
    )

    assert result.submission_sha256 == spec.recipe_hash


def test_select_new_blends_only_computes_csv_sha_for_finalists(tmp_path, monkeypatch):
    records = [
        _record(tmp_path, name="1", sha="a" * 64, score=0.91, target_values=["A", "B", "B"], pred_values=["A", "B", "B"], test_values=["A", "B", "B"]),
        _record(tmp_path, name="2", sha="b" * 64, score=0.89, target_values=["A", "B", "B"], pred_values=["A", "B", "A"], test_values=["A", "B", "A"]),
        _record(tmp_path, name="3", sha="c" * 64, score=0.88, target_values=["A", "B", "B"], pred_values=["A", "A", "B"], test_values=["A", "A", "B"]),
        _record(tmp_path, name="4", sha="d" * 64, score=0.87, target_values=["A", "B", "B"], pred_values=["B", "B", "B"], test_values=["B", "B", "B"]),
    ]
    calls = 0
    original = auto_blend_submissions.sha256_dataframe_csv

    def counted(frame):
        nonlocal calls
        calls += 1
        return original(frame)

    monkeypatch.setattr(auto_blend_submissions, "sha256_dataframe_csv", counted)

    auto_blend_submissions.select_new_blends(
        records,
        count=2,
        existing_submission_sha256=set(),
        existing_recipe_hashes=set(),
        id_col="id",
        target_col="class",
        metric_name="balanced_accuracy",
    )

    assert calls <= 4


def test_format_submit_command_uses_repeated_sha_flags():
    command = auto_blend_submissions.format_submit_command(["abc123def4", "9876543210"])

    assert command == (
        "uv run python scripts/kaggle_submission_lab.py "
        "--sha abc123def4 --sha 9876543210"
    )


def test_write_submission_only_artifact_refreshes_as_submit_ready(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample_submission.csv").write_text("id,class\n1,A\n2,A\n3,A\n")
    records = [
        _record(tmp_path, name="1", sha="a" * 64, score=0.91, target_values=["A", "B", "B"], pred_values=["A", "B", "B"], test_values=["A", "B", "B"], include_oof=False),
        _record(tmp_path, name="2", sha="b" * 64, score=0.88, target_values=["A", "B", "B"], pred_values=["A", "A", "B"], test_values=["A", "A", "B"], include_oof=False),
    ]
    spec = auto_blend_submissions.BlendSpec(
        blend_kind="submission",
        mode="vote",
        weighting="uniform",
        component_shas=tuple(record["sha256"] for record in records),
        weights=(0.5, 0.5),
        params={},
    )
    result = auto_blend_submissions.evaluate_blend_spec(
        spec,
        records,
        id_col="id",
        target_col="class",
        metric_name="balanced_accuracy",
    )
    args = argparse.Namespace(
        logs_dir=tmp_path / "logs",
        output_run="blended",
        competition="playground-series-s6e6",
    )

    artifact = auto_blend_submissions.write_blend_artifact(
        result=result,
        args=args,
        step=1,
        sample_path=data_dir / "sample_submission.csv",
        id_col="id",
        target_col="class",
        metric_name="balanced_accuracy",
    )
    index = kaggle_submission_lab.refresh_index(
        logs_dir=tmp_path / "logs",
        index_path=tmp_path / "logs" / "submission_index.json",
        competition="playground-series-s6e6",
        reindex=True,
    )
    record = index["records"][0]

    assert str(record["artifact_dir"]) == artifact
    assert record["status"] == "ok"
    assert not record["is_buggy"]
    assert record["sha256"]
    assert Path(record["submission_path"]).exists()
    assert record["origin"] == "auto_blend"
    assert record["submission_only"] is True
    assert record["blend_recipe_hash"] == spec.recipe_hash
    assert set(record["blend_component_sha256"].values()) == set(spec.component_shas)
    assert kaggle_submission_lab._record_is_submit_ready(record), record
