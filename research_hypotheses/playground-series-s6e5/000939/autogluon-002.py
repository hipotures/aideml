from __future__ import annotations

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    race = out["Race"].astype("string").str.lower().fillna("")
    compound = out["Compound"].astype("string").str.upper().fillna("UNKNOWN")
    lap = pd.to_numeric(out["LapNumber"], errors="coerce").fillna(1.0).clip(lower=1.0)
    progress = (
        pd.to_numeric(out["RaceProgress"], errors="coerce")
        .fillna(0.5)
        .clip(lower=0.01, upper=1.0)
    )
    tyre = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(0.0).clip(lower=0.0)
    stint = pd.to_numeric(out["Stint"], errors="coerce").fillna(1.0).clip(lower=1.0)
    lap_delta = pd.to_numeric(out["LapTime_Delta"], errors="coerce").fillna(0.0)
    degradation = pd.to_numeric(
        out["Cumulative_Degradation"], errors="coerce"
    ).fillna(0.0)
    position = pd.to_numeric(out["Position"], errors="coerce").fillna(10.0)

    est_total_laps = (lap / progress).clip(lower=lap, upper=120.0)
    laps_remaining = (est_total_laps - lap).clip(lower=0.0)

    expected_life = compound.map(
        {
            "SOFT": 18.0,
            "MEDIUM": 27.0,
            "HARD": 38.0,
            "INTERMEDIATE": 20.0,
            "WET": 16.0,
        }
    ).fillna(26.0)
    degradation_rate = (degradation / tyre.clip(lower=1.0)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    race_delta_scale = (
        lap_delta.groupby([out["Year"], out["Race"]], sort=False)
        .transform("std")
        .fillna(lap_delta.abs().median())
        .fillna(0.0)
        .abs()
    )
    pit_loss = pd.Series(22.0, index=out.index, dtype="float64")
    pit_loss += np.where(
        race.str.contains("monaco|singapore|hungar", regex=True), 4.0, 0.0
    )
    pit_loss += np.where(race.str.contains("monza|italian|vegas", regex=True), -3.0, 0.0)
    pit_loss += np.where(race.str.contains("austria|jeddah|saudi", regex=True), -1.5, 0.0)
    pit_loss += 0.8 * race_delta_scale.clip(0.0, 8.0)
    pit_loss = pit_loss.clip(10.0, 45.0)

    fresh_tyre_gain = (
        lap_delta.clip(lower=0.0)
        + 0.015 * degradation.clip(lower=0.0)
        + 0.35 * (tyre / expected_life.clip(lower=1.0))
    ).clip(0.05, 8.0)
    loss_recovery_laps = (pit_loss / fresh_tyre_gain).clip(2.0, 80.0)
    optimal_stint_lap = (expected_life - 0.35 * loss_recovery_laps).clip(3.0, 70.0)
    stint_gap = tyre - optimal_stint_lap

    out["pit_loss_est_seconds"] = pit_loss.astype("float32")
    out["fresh_tyre_gain_proxy"] = fresh_tyre_gain.astype("float32")
    out["pit_loss_recovery_laps"] = loss_recovery_laps.astype("float32")
    out["optimal_stint_lap_proxy"] = optimal_stint_lap.astype("float32")
    out["optimal_stint_gap"] = stint_gap.astype("float32")
    out["optimal_stint_gap_x_progress"] = (stint_gap * progress).astype("float32")
    out["optimal_stint_gap_x_laps_remaining"] = (
        stint_gap / (laps_remaining + 1.0)
    ).astype("float32")
    out["late_rule_pressure_proxy"] = (
        ((stint <= 1.0) & (progress >= 0.65)).astype("int8")
    )
    out["track_position_loss_pressure"] = (
        pit_loss / (3.0 + (21.0 - position).clip(lower=0.0) / 10.0)
    ).astype("float32")
    out["degradation_rate_x_gap"] = (degradation_rate * stint_gap).astype("float32")

    return out
