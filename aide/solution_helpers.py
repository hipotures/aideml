from __future__ import annotations

from pathlib import Path

import pandas as pd


def input_dir() -> Path:
    return Path("./input")


def working_dir() -> Path:
    path = Path("./working")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _csv_path(name: str) -> Path:
    base = input_dir()
    requested = base / name
    if requested.exists():
        return requested

    if requested.suffix:
        raise FileNotFoundError(f"Could not find {name} under {base}")

    for suffix in (".csv", ".csv.gz"):
        candidate = base / f"{name}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find {name}.csv or {name}.csv.gz under {base}")


def load_input_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_csv_path(name))


def load_competition_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        load_input_csv("train"),
        load_input_csv("test"),
        load_input_csv("sample_submission"),
    )


def write_submission(frame: pd.DataFrame) -> None:
    frame.to_csv(working_dir() / "submission.csv", index=False)


def write_oof_predictions(frame: pd.DataFrame) -> None:
    frame.to_csv(
        working_dir() / "oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )


def write_test_predictions(frame: pd.DataFrame) -> None:
    frame.to_csv(
        working_dir() / "test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )


def write_validation_predictions(frame: pd.DataFrame) -> None:
    frame.to_csv(
        working_dir() / "validation_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
