import gzip
from pathlib import Path

import pandas as pd

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


def test_agent_aux_false_keeps_competition_train_only_and_hides_raw_aux(tmp_path):
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
    assert not (input_dir / "f1_strategy_dataset_v4.csv").exists()
    assert not (input_dir / "train-aux.csv.gz").exists()


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
