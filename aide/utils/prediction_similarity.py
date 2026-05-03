from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCORE_ROUND_DECIMALS = 5
DEFAULT_PREDICTION_ROUND_DECIMALS = 5
DEFAULT_PREDICTION_SIMILARITY_RMSE_THRESHOLD = 0.015


def submission_prediction_rmse(
    left_path: Path,
    right_path: Path,
    *,
    prediction_round_decimals: int = DEFAULT_PREDICTION_ROUND_DECIMALS,
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
    left = left[["id", target]].sort_values("id").reset_index(drop=True)
    right = right[["id", target]].sort_values("id").reset_index(drop=True)
    if len(left) != len(right) or not left["id"].equals(right["id"]):
        return None

    left_predictions = pd.to_numeric(left[target], errors="coerce")
    right_predictions = pd.to_numeric(right[target], errors="coerce")
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
