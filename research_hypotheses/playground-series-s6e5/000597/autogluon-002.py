from __future__ import annotations

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()
    race_keys = ["Year", "Race"]
    lap_keys = race_keys + ["LapNumber"]
    comp_keys = race_keys + ["Compound", "LapNumber"]

    pit_mask = out["PitStop"].eq(1)
    out["_prior_pit_cnt_src"] = pit_mask.astype("int16")
    out["_prior_pit_gain_src"] = out["Position_Change"].where(pit_mask, 0.0)
    out["_prior_pit_loss_src"] = out["LapTime_Delta"].clip(-60, 300).where(
        pit_mask, 0.0
    )

    lap_agg = (
        out.groupby(lap_keys, as_index=False, sort=True)
        .agg(
            pit_cnt=("_prior_pit_cnt_src", "sum"),
            pit_gain=("_prior_pit_gain_src", "sum"),
            pit_loss=("_prior_pit_loss_src", "sum"),
        )
        .sort_values(lap_keys, kind="mergesort")
    )
    race_group = lap_agg.groupby(race_keys, sort=False)
    race_group_keys = [lap_agg["Year"], lap_agg["Race"]]
    pit_csum = race_group["pit_cnt"].cumsum()
    gain_csum = race_group["pit_gain"].cumsum()
    loss_csum = race_group["pit_loss"].cumsum()
    lap_agg["prior_race_pit_count"] = pit_csum.groupby(
        race_group_keys, sort=False
    ).shift(fill_value=0)
    lap_agg["prior_race_pit_gain_sum"] = gain_csum.groupby(
        race_group_keys, sort=False
    ).shift(fill_value=0.0)
    lap_agg["prior_race_pit_loss_sum"] = loss_csum.groupby(
        race_group_keys, sort=False
    ).shift(fill_value=0.0)
    lap_agg["prior_race_pit_gain_mean"] = lap_agg[
        "prior_race_pit_gain_sum"
    ] / lap_agg["prior_race_pit_count"].replace(0, np.nan)
    lap_agg["prior_race_pit_loss_mean"] = lap_agg[
        "prior_race_pit_loss_sum"
    ] / lap_agg["prior_race_pit_count"].replace(0, np.nan)

    out = out.merge(
        lap_agg[
            lap_keys
            + [
                "prior_race_pit_count",
                "prior_race_pit_gain_mean",
                "prior_race_pit_loss_mean",
            ]
        ],
        on=lap_keys,
        how="left",
        sort=False,
    )

    comp_agg = (
        out.groupby(comp_keys, as_index=False, sort=True)
        .agg(comp_pit_cnt=("_prior_pit_cnt_src", "sum"))
        .sort_values(comp_keys, kind="mergesort")
    )
    comp_group = comp_agg.groupby(race_keys + ["Compound"], sort=False)
    comp_group_keys = [comp_agg["Year"], comp_agg["Race"], comp_agg["Compound"]]
    comp_csum = comp_group["comp_pit_cnt"].cumsum()
    comp_agg["prior_compound_pit_count"] = comp_csum.groupby(
        comp_group_keys, sort=False
    ).shift(fill_value=0)
    out = out.merge(
        comp_agg[comp_keys + ["prior_compound_pit_count"]],
        on=comp_keys,
        how="left",
        sort=False,
    )

    for col in [
        "prior_race_pit_count",
        "prior_race_pit_gain_mean",
        "prior_race_pit_loss_mean",
        "prior_compound_pit_count",
    ]:
        out[col] = out[col].fillna(0.0).astype("float32")

    tyre_life = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(0.0)
    progress = pd.to_numeric(out["RaceProgress"], errors="coerce").fillna(0.0)
    position = pd.to_numeric(out["Position"], errors="coerce").fillna(20.0)
    out["prior_pit_evidence_x_tyre"] = (
        out["prior_race_pit_count"] * tyre_life
    ).astype("float32")
    out["prior_gain_x_progress"] = (
        out["prior_race_pit_gain_mean"] * progress
    ).astype("float32")
    out["prior_loss_x_tyre"] = (
        out["prior_race_pit_loss_mean"] * tyre_life
    ).astype("float32")
    out["prior_compound_pressure"] = (
        out["prior_compound_pit_count"] / (1.0 + position)
    ).astype("float32")

    out = out.drop(
        columns=["_prior_pit_cnt_src", "_prior_pit_gain_src", "_prior_pit_loss_src"]
    )
    return out
