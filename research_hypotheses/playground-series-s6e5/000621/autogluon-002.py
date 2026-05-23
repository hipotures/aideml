from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    keys = ["Year", "Race", "Compound", "Stint"]
    race_keys = ["Year", "Race"]
    has_pit = "PitStop" in out.columns
    pit = out["PitStop"].fillna(0).clip(0, 1).astype(float) if has_pit else 0.0

    out["_pit_window_source"] = pit
    out["_tyre_life_for_window"] = out["TyreLife"].astype(float)
    out["_lap_for_window"] = out["LapNumber"].astype(float)

    group = out.groupby(keys, sort=False, observed=True)
    fallback = out.groupby(["Compound", "Stint"], sort=False, observed=True)
    race_group = out.groupby(race_keys, sort=False, observed=True)

    event_weight = out["_pit_window_source"]
    weighted_tyre = out["_tyre_life_for_window"] * event_weight
    weighted_lap = out["_lap_for_window"] * event_weight

    event_count = group["_pit_window_source"].transform("sum").astype(float)
    fallback_count = fallback["_pit_window_source"].transform("sum").astype(float)
    group_rows = group["_pit_window_source"].transform("count").astype(float)
    fallback_rows = fallback["_pit_window_source"].transform("count").astype(float)

    group_event_tyre = weighted_tyre.groupby([out[c] for c in keys], sort=False).transform(
        "sum"
    ) / event_count.replace(0, np.nan)
    fallback_event_tyre = weighted_tyre.groupby(
        [out["Compound"], out["Stint"]], sort=False
    ).transform("sum") / fallback_count.replace(0, np.nan)
    group_event_lap = weighted_lap.groupby([out[c] for c in keys], sort=False).transform(
        "sum"
    ) / event_count.replace(0, np.nan)
    fallback_event_lap = weighted_lap.groupby(
        [out["Compound"], out["Stint"]], sort=False
    ).transform("sum") / fallback_count.replace(0, np.nan)

    unsup_group_tyre = group["TyreLife"].transform("median")
    unsup_fallback_tyre = fallback["TyreLife"].transform("median")
    unsup_group_lap = group["LapNumber"].transform("median")
    unsup_fallback_lap = fallback["LapNumber"].transform("median")

    center_tyre = (
        group_event_tyre.fillna(fallback_event_tyre)
        .fillna(unsup_group_tyre)
        .fillna(unsup_fallback_tyre)
        .fillna(out["TyreLife"].median())
    )
    center_lap = (
        group_event_lap.fillna(fallback_event_lap)
        .fillna(unsup_group_lap)
        .fillna(unsup_fallback_lap)
        .fillna(out["LapNumber"].median())
    )

    race_laps = race_group["LapNumber"].transform("max").clip(lower=1)
    smooth_rate = (event_count + 2.0) / (group_rows + 20.0)
    fallback_rate = (fallback_count + 2.0) / (fallback_rows + 20.0)

    out["pit_window_event_rate"] = smooth_rate.fillna(fallback_rate).astype("float32")
    out["pit_window_center_tyre_life"] = center_tyre.astype("float32")
    out["pit_window_center_lap"] = center_lap.astype("float32")
    out["pit_window_tyre_distance"] = (out["TyreLife"] - center_tyre).astype("float32")
    out["pit_window_abs_tyre_distance"] = out["pit_window_tyre_distance"].abs()
    out["pit_window_lap_distance"] = (out["LapNumber"] - center_lap).astype("float32")
    out["pit_window_abs_lap_distance"] = out["pit_window_lap_distance"].abs()
    out["pit_window_norm_tyre_distance"] = (
        out["pit_window_tyre_distance"] / (center_tyre.abs() + 1.0)
    ).astype("float32")
    out["pit_window_norm_lap_distance"] = (
        out["pit_window_lap_distance"] / (race_laps + eps)
    ).astype("float32")
    out["pit_window_near_core"] = (
        out["pit_window_abs_tyre_distance"] <= 2.0
    ).astype("int8")
    out["pit_window_near_wide"] = (
        out["pit_window_abs_tyre_distance"] <= 5.0
    ).astype("int8")
    out["pit_window_past_core"] = (out["pit_window_tyre_distance"] > 3.0).astype("int8")
    out["pit_window_future_core"] = (
        out["pit_window_tyre_distance"] < -3.0
    ).astype("int8")

    return out.drop(columns=["_pit_window_source", "_tyre_life_for_window", "_lap_for_window"])
