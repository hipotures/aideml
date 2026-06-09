import gzip
from pathlib import Path

import pandas as pd
import pytest

from aide.solution_helpers import (
    log_stage,
    load_competition_data,
    load_input_csv,
    stage,
    working_dir,
    write_oof_predictions,
    write_submission,
    write_test_predictions,
)


def _write_csv_gz(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content)


def test_solution_helpers_load_standard_competition_data(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "train.csv").write_text("id,x,y\n1,2,0\n", encoding="utf-8")
    _write_csv_gz(input_dir / "test.csv.gz", "id,x\n2,3\n")
    (input_dir / "sample_submission.csv").write_text("id,y\n2,0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    train, test, sample = load_competition_data()

    assert train.to_dict("records") == [{"id": 1, "x": 2, "y": 0}]
    assert test.to_dict("records") == [{"id": 2, "x": 3}]
    assert sample.to_dict("records") == [{"id": 2, "y": 0}]


def test_solution_helpers_log_load_data_stage(tmp_path, monkeypatch, capsys):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "train.csv").write_text("id,x,y\n1,2,0\n", encoding="utf-8")
    (input_dir / "test.csv").write_text("id,x\n2,3\n", encoding="utf-8")
    (input_dir / "sample_submission.csv").write_text("id,y\n2,0\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    load_competition_data()

    output = capsys.readouterr().out
    assert "AIDE_STAGE|event=start|stage=load_data_stage" in output
    assert "AIDE_STAGE|event=end|stage=load_data_stage|elapsed_s=" in output


def test_solution_helpers_stage_logs_failure(capsys):
    with pytest.raises(RuntimeError):
        with stage("fit_predict_fold_stage"):
            raise RuntimeError("boom")

    output = capsys.readouterr().out
    assert "AIDE_STAGE|event=start|stage=fit_predict_fold_stage" in output
    assert (
        "AIDE_STAGE|event=failed|stage=fit_predict_fold_stage|elapsed_s="
        in output
    )
    assert "error_type=RuntimeError" in output


def test_solution_helpers_log_stage_flushes_marker(capsys):
    log_stage("event=progress|stage=fit_predict_fold_stage|fold=1")

    assert (
        "AIDE_STAGE|event=progress|stage=fit_predict_fold_stage|fold=1"
        in capsys.readouterr().out
    )


def test_solution_helpers_do_not_search_outside_input(tmp_path, monkeypatch):
    (tmp_path / "train.csv").write_text("id,x,y\n1,2,0\n", encoding="utf-8")
    (tmp_path / "input").mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="under input"):
        load_input_csv("train")


def test_solution_helpers_write_standard_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = pd.DataFrame({"id": [1], "target": [0]})

    write_submission(frame)
    write_oof_predictions(frame)
    write_test_predictions(frame)

    assert (tmp_path / "working" / "submission.csv").exists()
    assert (tmp_path / "working" / "oof_predictions.csv.gz").exists()
    assert (tmp_path / "working" / "test_predictions.csv.gz").exists()
    assert working_dir() == Path("./working")
