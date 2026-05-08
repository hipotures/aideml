from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCORE_ROUND_DECIMALS = 5
DEFAULT_PREDICTION_ROUND_DECIMALS = 5
DEFAULT_PREDICTION_SIMILARITY_RMSE_THRESHOLD = 0.015
DEFAULT_PREDICTION_SIMILARITY_SAMPLE_SIZE = 200
DEFAULT_PREDICTION_SIMILARITY_MIN_COMMON_SAMPLE_SIZE = 100


def submission_prediction_rmse(
    left_path: Path,
    right_path: Path,
    *,
    prediction_round_decimals: int = DEFAULT_PREDICTION_ROUND_DECIMALS,
    sample_size: int = DEFAULT_PREDICTION_SIMILARITY_SAMPLE_SIZE,
    min_common_sample_size: int = DEFAULT_PREDICTION_SIMILARITY_MIN_COMMON_SAMPLE_SIZE,
) -> float | None:
    if not left_path.exists() or not right_path.exists():
        return None

    try:
        left = pd.read_csv(left_path)
        right = pd.read_csv(right_path)
    except Exception:  # noqa: BLE001 - unreadable artifacts should not stop selection
        return None

    if "id" not in left.columns or "id" not in right.columns:
        return None

    left_targets = [column for column in left.columns if column != "id"]
    right_targets = [column for column in right.columns if column != "id"]
    if len(left_targets) != 1 or left_targets != right_targets:
        return None

    target = left_targets[0]
    if sample_size <= 0 or min_common_sample_size <= 0:
        return None
    left = left[["id", target]].iloc[:sample_size].reset_index(drop=True)
    right = right[["id", target]].iloc[:sample_size].reset_index(drop=True)
    if left["id"].duplicated().any() or right["id"].duplicated().any():
        return None
    try:
        paired = left.merge(
            right,
            on="id",
            how="inner",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
    except Exception:  # noqa: BLE001 - malformed artifacts should not stop selection
        return None
    if len(paired) < min_common_sample_size:
        return None

    left_predictions = pd.to_numeric(paired[f"{target}_left"], errors="coerce")
    right_predictions = pd.to_numeric(paired[f"{target}_right"], errors="coerce")
    if left_predictions.isna().any() or right_predictions.isna().any():
        return None

    left_values = np.round(
        left_predictions.to_numpy(dtype=float),
        prediction_round_decimals,
    )
    right_values = np.round(
        right_predictions.to_numpy(dtype=float),
        prediction_round_decimals,
    )
    return math.sqrt(float(np.mean((left_values - right_values) ** 2)))
