from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    eps = 1e-6

    out["_pit_window_source"] = out["PitStop"].fillna(0).clip(0, 1).astype(float)
    out["_tyre_life"] = out["TyreLife"].astype(float)
    out["_lap_number"] = out["LapNumber"].astype(float)
    out["_race_progress"] = out["RaceProgress"].astype(float)

    primary_keys = ["Year", "Race", "Compound", "Stint"]
    fallback_keys = ["Compound", "Stint"]
    primary = out.groupby(primary_keys, sort=False, observed=True)
    fallback = out.groupby(fallback_keys, sort=False, observed=True)

    event_weight = out["_pit_window_source"]
    weighted_tyre = out["_tyre_life"] * event_weight
    weighted_lap = out["_lap_number"] * event_weight
    weighted_progress = out["_race_progress"] * event_weight

    event_count_primary = primary["_pit_window_source"].transform("sum").astype(float)
    event_count_fallback = fallback["_pit_window_source"].transform("sum").astype(float)
    row_count_primary = primary["_pit_window_source"].transform("count").astype(float)
    row_count_fallback = fallback["_pit_window_source"].transform("count").astype(float)

    primary_tyre_center = weighted_tyre.groupby(
        [out[c] for c in primary_keys], sort=False
    ).transform("sum") / event_count_primary.replace(0, np.nan)
    fallback_tyre_center = weighted_tyre.groupby(
        [out[c] for c in fallback_keys], sort=False
    ).transform("sum") / event_count_fallback.replace(0, np.nan)
    primary_lap_center = weighted_lap.groupby(
        [out[c] for c in primary_keys], sort=False
    ).transform("sum") / event_count_primary.replace(0, np.nan)
    fallback_lap_center = weighted_lap.groupby(
        [out[c] for c in fallback_keys], sort=False
    ).transform("sum") / event_count_fallback.replace(0, np.nan)
    primary_progress_center = weighted_progress.groupby(
        [out[c] for c in primary_keys], sort=False
    ).transform("sum") / event_count_primary.replace(0, np.nan)
    fallback_progress_center = weighted_progress.groupby(
        [out[c] for c in fallback_keys], sort=False
    ).transform("sum") / event_count_fallback.replace(0, np.nan)

    tyre_center = (
        primary_tyre_center.fillna(fallback_tyre_center)
        .fillna(primary["TyreLife"].transform("median"))
        .fillna(fallback["TyreLife"].transform("median"))
        .fillna(out["TyreLife"].median())
    )
    lap_center = (
        primary_lap_center.fillna(fallback_lap_center)
        .fillna(primary["LapNumber"].transform("median"))
        .fillna(fallback["LapNumber"].transform("median"))
        .fillna(out["LapNumber"].median())
    )
    progress_center = (
        primary_progress_center.fillna(fallback_progress_center)
        .fillna(primary["RaceProgress"].transform("median"))
        .fillna(fallback["RaceProgress"].transform("median"))
        .fillna(out["RaceProgress"].median())
    )

    q25 = primary["TyreLife"].transform(lambda s: s.quantile(0.25))
    q75 = primary["TyreLife"].transform(lambda s: s.quantile(0.75))
    q90 = primary["TyreLife"].transform(lambda s: s.quantile(0.90))
    fb_q25 = fallback["TyreLife"].transform(lambda s: s.quantile(0.25))
    fb_q75 = fallback["TyreLife"].transform(lambda s: s.quantile(0.75))
    fb_q90 = fallback["TyreLife"].transform(lambda s: s.quantile(0.90))
    q25 = q25.fillna(fb_q25).fillna(tyre_center - 3.0)
    q75 = q75.fillna(fb_q75).fillna(tyre_center + 3.0)
    q90 = q90.fillna(fb_q90).fillna(tyre_center + 6.0)

    smooth_primary = (event_count_primary + 1.0) / (row_count_primary + 10.0)
    smooth_fallback = (event_count_fallback + 1.0) / (row_count_fallback + 10.0)

    out["distance_to_pit_window"] = (out["TyreLife"] - tyre_center).astype("float32")
    out["abs_distance_to_pit_window"] = out["distance_to_pit_window"].abs()
    out["lap_distance_to_pit_window"] = (out["LapNumber"] - lap_center).astype(
        "float32"
    )
    out["progress_distance_to_pit_window"] = (
        out["RaceProgress"] - progress_center
    ).astype("float32")
    out["normalized_tyre_age_vs_window"] = (
        out["distance_to_pit_window"] / (tyre_center.abs() + 1.0)
    ).astype("float32")
    out["inside_25_75_window"] = (
        out["_tyre_life"].between(q25, q75, inclusive="both")
    ).astype("int8")
    out["past_p90_window"] = (out["_tyre_life"] > q90).astype("int8")
    out["window_event_rate_smoothed"] = smooth_primary.fillna(smooth_fallback).astype(
        "float32"
    )
    out["window_distance_x_degradation"] = (
        out["distance_to_pit_window"] * out["Cumulative_Degradation"]
    ).astype("float32")
    out["window_distance_x_position_loss"] = (
        out["distance_to_pit_window"] * (-out["Position_Change"]).clip(lower=0)
    ).astype("float32")
    out["window_distance_x_progress"] = (
        out["distance_to_pit_window"] * out["RaceProgress"]
    ).astype("float32")
    out["window_center_lap_frac"] = (lap_center / (out["LapNumber"].abs() + 1.0)).astype(
        "float32"
    )

    return out.drop(
        columns=["_pit_window_source", "_tyre_life", "_lap_number", "_race_progress"]
    )
