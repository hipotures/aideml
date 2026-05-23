from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_row_order_000624"] = np.arange(len(out))

    sort_cols = ["Year", "Race", "Driver", "LapNumber", "_row_order_000624"]
    ordered = out.sort_values(sort_cols, kind="mergesort").copy()
    group_cols = ["Year", "Race", "Driver"]
    stint_cols = ["Year", "Race", "Driver", "Stint"]
    g = ordered.groupby(group_cols, sort=False, observed=True)
    gs = ordered.groupby(stint_cols, sort=False, observed=True)

    pit = ordered["PitStop"].fillna(0).clip(0, 1).astype(float)
    ordered["_pit_current"] = pit
    ordered["_pit_prev1"] = g["_pit_current"].shift(1).fillna(0.0)
    ordered["_pit_prev2"] = g["_pit_current"].shift(2).fillna(0.0)
    ordered["_pit_prev3"] = g["_pit_current"].shift(3).fillna(0.0)
    ordered["_prior_stop_count"] = g["_pit_current"].cumsum() - ordered["_pit_current"]

    ordered["post_pit_cooldown_1"] = ordered["_pit_prev1"].astype("float32")
    ordered["post_pit_cooldown_3"] = (
        ordered["_pit_prev1"] + ordered["_pit_prev2"] + ordered["_pit_prev3"]
    ).clip(0, 1).astype("float32")
    ordered["prior_stop_count"] = ordered["_prior_stop_count"].astype("float32")

    ordered["stint_lap_rank_pct"] = gs["LapNumber"].rank(pct=True).astype("float32")
    ordered["stint_rows_seen_proxy"] = gs.cumcount().astype("float32")
    ordered["stint_len_proxy"] = gs["LapNumber"].transform("count").astype("float32")
    ordered["stint_remaining_rank_proxy"] = (
        ordered["stint_len_proxy"] - ordered["stint_rows_seen_proxy"] - 1.0
    ).clip(lower=0).astype("float32")

    tyre = ordered["TyreLife"].astype(float)
    progress = ordered["RaceProgress"].astype(float)
    ordered["temporal_window_pressure"] = (
        tyre * (1.0 - ordered["post_pit_cooldown_1"]) * (0.25 + progress)
    ).astype("float32")
    ordered["temporal_plausible_peak"] = (
        (tyre >= 8)
        & (progress.between(0.12, 0.94))
        & (ordered["post_pit_cooldown_1"] == 0)
    ).astype("int8")
    ordered["temporal_late_stint_peak"] = (
        (ordered["stint_lap_rank_pct"] >= 0.65)
        & (ordered["post_pit_cooldown_3"] == 0)
    ).astype("int8")
    ordered["temporal_recent_reset_x_tyre"] = (
        ordered["post_pit_cooldown_3"] * tyre
    ).astype("float32")

    keep_cols = [
        "_row_order_000624",
        "post_pit_cooldown_1",
        "post_pit_cooldown_3",
        "prior_stop_count",
        "stint_lap_rank_pct",
        "stint_rows_seen_proxy",
        "stint_len_proxy",
        "stint_remaining_rank_proxy",
        "temporal_window_pressure",
        "temporal_plausible_peak",
        "temporal_late_stint_peak",
        "temporal_recent_reset_x_tyre",
    ]
    ordered = ordered.sort_values("_row_order_000624")
    for col in keep_cols[1:]:
        out[col] = ordered[col].to_numpy()

    return out.drop(columns=["_row_order_000624"])
