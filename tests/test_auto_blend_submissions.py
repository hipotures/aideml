import importlib.util
import argparse
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
    assert kaggle_submission_lab._record_is_submit_ready(record), record
