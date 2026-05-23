from __future__ import annotations

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    lap = pd.to_numeric(out["LapNumber"], errors="coerce").fillna(0.0)
    progress = (
        pd.to_numeric(out["RaceProgress"], errors="coerce")
        .clip(lower=0.01, upper=1.0)
        .fillna(0.01)
    )
    tyre_life = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(0.0)
    stint = pd.to_numeric(out["Stint"], errors="coerce").fillna(1.0).clip(lower=1.0)
    degradation = pd.to_numeric(
        out["Cumulative_Degradation"], errors="coerce"
    ).fillna(0.0)

    est_total_laps = (lap / progress).replace([np.inf, -np.inf], np.nan)
    est_total_laps = est_total_laps.fillna(lap).clip(lower=lap)
    laps_remaining = (est_total_laps - lap).clip(lower=0.0)

    compound_life = {
        "SOFT": 18.0,
        "MEDIUM": 28.0,
        "HARD": 38.0,
        "INTERMEDIATE": 14.0,
        "WET": 12.0,
    }
    compound = out["Compound"].astype(str).str.upper()
    expected_life = compound.map(compound_life).fillna(26.0)
    expected_remaining = (expected_life - tyre_life).clip(lower=0.0)

    out["aft_laps_remaining_est"] = laps_remaining.astype("float32")
    out["aft_expected_stint_remaining"] = expected_remaining.astype("float32")
    out["aft_censoring_pressure"] = np.minimum(
        laps_remaining, expected_remaining
    ).astype("float32")
    out["aft_stint_progress_ratio"] = (
        tyre_life / expected_life.clip(lower=1.0)
    ).clip(0.0, 3.0).astype("float32")
    out["aft_late_race_old_tyre"] = (progress * tyre_life).astype("float32")
    out["aft_degradation_rate"] = (
        degradation / tyre_life.clip(lower=1.0)
    ).astype("float32")
    out["aft_hazard_proxy"] = (
        1.0
        / (
            1.0
            + np.exp(
                -(
                    2.0 * out["aft_stint_progress_ratio"]
                    + 0.7 * progress
                    + 0.15 * stint
                    - 2.2
                )
            )
        )
    ).astype("float32")

    return out
