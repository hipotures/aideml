import gzip
from pathlib import Path

import pandas as pd
import pytest

from aide.utils.config import _load_cfg, load_task_desc, prep_agent_workspace, prep_cfg


def _write_csv_gz(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content)


def _cfg(tmp_path: Path):
    cfg = _load_cfg(use_cli_args=False)
    cfg.data_dir = str(tmp_path / "data" / "playground-series-s6e5")
    cfg.goal = "Predict next-lap pit stop probability"
    cfg.log_dir = str(tmp_path / "logs")
    cfg.workspace_dir = str(tmp_path / "workspace")
    cfg.exp_name = "aux-test"
    cfg.preprocess_data = False
    cfg.copy_data = True
    return prep_cfg(cfg)


def test_agent_aux_materializes_merged_train_and_hides_raw_aux_dataset(tmp_path):
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    _write_csv_gz(
        data_dir / "train.csv.gz",
        "id,Driver,LapNumber,TyreLife,PitNextLap\n1,A,10,5,0\n2,B,20,8,1\n",
    )
    _write_csv_gz(
        data_dir / "train-aux.csv.gz",
        "id,Driver,LapNumber,TyreLife,PitNextLap\n"
        "1,A,10,5,0\n"
        "2,B,20,8,1\n"
        "-1,C,40,12,0\n",
    )
    _write_csv_gz(
        data_dir / "test.csv.gz",
        "id,Driver,LapNumber,TyreLife\n10,A,30,9\n",
    )
    _write_csv_gz(
        data_dir / "test-aux.csv.gz",
        "id,Driver,LapNumber,TyreLife\n"
        "10,A,30,9\n",
    )
    _write_csv_gz(
        data_dir / "sample_submission.csv.gz",
        "id,PitNextLap\n10,0\n",
    )
    (data_dir / "f1_strategy_dataset_v4.csv").write_text(
        "Driver,LapNumber,TyreLife,PitNextLap,Normalized_TyreLife\n"
        "C,40,12,0,0.3\n",
        encoding="utf-8",
    )

    cfg = _cfg(tmp_path)
    cfg.agent.aux = True

    prep_agent_workspace(cfg)

    input_dir = Path(cfg.workspace_dir) / "input"
    train = pd.read_csv(input_dir / "train.csv.gz")
    test = pd.read_csv(input_dir / "test.csv.gz")

    assert len(train) == 3
    assert list(train.columns) == [
        "id",
        "Driver",
        "LapNumber",
        "TyreLife",
        "PitNextLap",
    ]
    assert list(test.columns) == ["id", "Driver", "LapNumber", "TyreLife"]
    assert "source_is_aux" not in train.columns
    assert "source_is_aux" not in test.columns
    assert "Normalized_TyreLife" not in train.columns
    assert "Normalized_TyreLife" not in test.columns
    assert sorted(path.name for path in input_dir.iterdir()) == [
        "sample_submission.csv.gz",
        "test.csv.gz",
        "train.csv.gz",
    ]


def test_agent_aux_merged_string_uses_existing_merged_flow(tmp_path):
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    _write_csv_gz(
        data_dir / "train-aux.csv.gz",
        "id,x,target\n1,2,0\n2,3,1\n",
    )
    _write_csv_gz(data_dir / "test-aux.csv.gz", "id,x\n10,4\n")
    _write_csv_gz(data_dir / "sample_submission.csv.gz", "id,target\n10,0\n")

    cfg = _cfg(tmp_path)
    cfg.agent.aux = "merged"

    prep_agent_workspace(cfg)

    input_dir = Path(cfg.workspace_dir) / "input"
    assert sorted(path.name for path in input_dir.iterdir()) == [
        "sample_submission.csv.gz",
        "test.csv.gz",
        "train.csv.gz",
    ]
    assert pd.read_csv(input_dir / "train.csv.gz").shape == (2, 3)


def test_agent_aux_false_keeps_competition_train_and_original_dataset_only(tmp_path):
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    _write_csv_gz(
        data_dir / "train.csv.gz",
        "id,Driver,LapNumber,TyreLife,PitNextLap\n1,A,10,5,0\n",
    )
    _write_csv_gz(
        data_dir / "train-aux.csv.gz",
        "id,Driver,LapNumber,TyreLife,PitNextLap,source_is_aux\n"
        "1,A,10,5,0,0\n"
        "-1,C,40,12,0,1\n",
    )
    _write_csv_gz(data_dir / "test.csv.gz", "id,Driver,LapNumber,TyreLife\n10,A,30,9\n")
    (data_dir / "f1_strategy_dataset_v4.csv").write_text(
        "Driver,LapNumber,TyreLife,PitNextLap\nC,40,12,0\n",
        encoding="utf-8",
    )

    cfg = _cfg(tmp_path)

    prep_agent_workspace(cfg)

    input_dir = Path(cfg.workspace_dir) / "input"
    train = pd.read_csv(input_dir / "train.csv.gz")

    assert len(train) == 1
    assert "source_is_aux" not in train.columns
    assert (input_dir / "f1_strategy_dataset_v4.csv").exists()
    assert not (input_dir / "train-aux.csv.gz").exists()
    assert not (input_dir / "test-aux.csv.gz").exists()


def test_agent_aux_file_copies_single_raw_dataset_flat_without_merging(tmp_path):
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    _write_csv_gz(data_dir / "train.csv.gz", "id,x,target\n1,2,0\n")
    _write_csv_gz(data_dir / "test.csv.gz", "id,x\n10,4\n")
    _write_csv_gz(data_dir / "sample_submission.csv.gz", "id,target\n10,0\n")
    nested_dir = data_dir / "original"
    nested_dir.mkdir(parents=True)
    (nested_dir / "external.csv").write_text(
        "x,target,source_col\n5,1,a\n6,0,b\n",
        encoding="utf-8",
    )

    cfg = _cfg(tmp_path)
    cfg.agent.aux = "external.csv"

    prep_agent_workspace(cfg)

    input_dir = Path(cfg.workspace_dir) / "input"
    assert (input_dir / "external.csv").exists()
    assert not (input_dir / "original" / "external.csv").exists()
    assert pd.read_csv(input_dir / "train.csv.gz").shape == (1, 3)
    assert pd.read_csv(input_dir / "external.csv").shape == (2, 3)


def test_agent_aux_file_is_copied_even_when_copy_data_false(tmp_path):
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    _write_csv_gz(data_dir / "train.csv.gz", "id,x,target\n1,2,0\n")
    _write_csv_gz(data_dir / "test.csv.gz", "id,x\n10,4\n")
    _write_csv_gz(data_dir / "sample_submission.csv.gz", "id,target\n10,0\n")
    nested_dir = data_dir / "original"
    nested_dir.mkdir(parents=True)
    (nested_dir / "external.csv").write_text("x,target\n5,1\n", encoding="utf-8")

    cfg = _cfg(tmp_path)
    cfg.copy_data = False
    cfg.agent.aux = "external.csv"

    prep_agent_workspace(cfg)

    aux_link = Path(cfg.workspace_dir) / "input" / "external.csv"
    assert not aux_link.is_symlink()
    assert pd.read_csv(aux_link).shape == (1, 2)


def test_agent_aux_file_appends_txt_description_from_source_directory(tmp_path):
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task\n", encoding="utf-8")
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    source_dir = data_dir / "original"
    source_dir.mkdir(parents=True)
    (source_dir / "external.csv").write_text("x,target\n5,1\n", encoding="utf-8")
    (source_dir / "external.md").write_text("markdown description\n", encoding="utf-8")
    (source_dir / "external.txt").write_text("text description\n", encoding="utf-8")

    cfg = _cfg(tmp_path)
    cfg.desc_file = desc_file
    cfg.agent.aux = "external.csv"

    task_desc = load_task_desc(cfg)

    assert "Additional auxiliary data description for `external.csv`" in task_desc
    assert "text description" in task_desc
    assert "markdown description" not in task_desc


def test_agent_aux_file_uses_md_description_when_txt_missing(tmp_path):
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task\n", encoding="utf-8")
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    source_dir = data_dir / "original"
    source_dir.mkdir(parents=True)
    (source_dir / "external.csv.gz").write_text("not actually read here\n", encoding="utf-8")
    (source_dir / "external.md").write_text("markdown description\n", encoding="utf-8")

    cfg = _cfg(tmp_path)
    cfg.desc_file = desc_file
    cfg.agent.aux = "external.csv.gz"

    task_desc = load_task_desc(cfg)

    assert "Additional auxiliary data description for `external.csv.gz`" in task_desc
    assert "markdown description" in task_desc


def test_agent_aux_file_without_description_adds_no_aux_description(tmp_path):
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task\n", encoding="utf-8")
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    (data_dir / "external.csv").parent.mkdir(parents=True, exist_ok=True)
    (data_dir / "external.csv").write_text("x,target\n5,1\n", encoding="utf-8")

    cfg = _cfg(tmp_path)
    cfg.desc_file = desc_file
    cfg.agent.aux = "external.csv"

    assert load_task_desc(cfg) == "task\n"


def test_agent_aux_appends_merged_external_data_note_to_task_desc(tmp_path):
    desc_file = tmp_path / "task.md"
    desc_file.write_text(
        "## Data description\n- **train.csv.gz** - competition train\n",
        encoding="utf-8",
    )

    cfg = _cfg(tmp_path)
    cfg.desc_file = desc_file
    cfg.agent.aux = True

    task_desc = load_task_desc(cfg)

    assert "original/external F1 strategy dataset" in task_desc
    assert "synthetic Kaggle Playground tabular data" in task_desc
    assert "source/provenance-only columns" in task_desc
    assert "same feature columns as the competition train file" in task_desc
    assert "`test.csv.gz` remains the competition test set only" in task_desc


def test_agent_aux_false_does_not_append_external_data_note(tmp_path):
    desc_file = tmp_path / "task.md"
    desc_file.write_text("task\n", encoding="utf-8")

    cfg = _cfg(tmp_path)
    cfg.desc_file = desc_file

    task_desc = load_task_desc(cfg)

    assert "original/external F1 strategy dataset" not in task_desc
    assert "source_is_aux" not in task_desc


@pytest.mark.parametrize(
    "aux_value",
    [
        "[a.csv,b.csv]",
        "a.csv,b.csv",
        "foo.parquet",
        "dir/file.csv",
        "../file.csv",
        "train.csv",
    ],
)
def test_agent_aux_file_rejects_invalid_values(tmp_path, aux_value):
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    _write_csv_gz(data_dir / "train.csv.gz", "id,x,target\n1,2,0\n")
    _write_csv_gz(data_dir / "test.csv.gz", "id,x\n10,4\n")
    _write_csv_gz(data_dir / "sample_submission.csv.gz", "id,target\n10,0\n")

    cfg = _cfg(tmp_path)
    cfg.agent.aux = aux_value

    with pytest.raises(ValueError):
        prep_agent_workspace(cfg)


def test_agent_aux_file_rejects_missing_file(tmp_path):
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    _write_csv_gz(data_dir / "train.csv.gz", "id,x,target\n1,2,0\n")

    cfg = _cfg(tmp_path)
    cfg.agent.aux = "external.csv"

    with pytest.raises(FileNotFoundError):
        prep_agent_workspace(cfg)


def test_agent_aux_file_rejects_ambiguous_filename(tmp_path):
    data_dir = tmp_path / "data" / "playground-series-s6e5"
    _write_csv_gz(data_dir / "train.csv.gz", "id,x,target\n1,2,0\n")
    for parent in (data_dir / "a", data_dir / "b"):
        parent.mkdir(parents=True)
        (parent / "external.csv").write_text("x,target\n5,1\n", encoding="utf-8")

    cfg = _cfg(tmp_path)
    cfg.agent.aux = "external.csv"

    with pytest.raises(ValueError, match="matched multiple files"):
        prep_agent_workspace(cfg)
