from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keys = [c for c in ["Year", "Race", "LapNumber"] if c in out.columns]
    if not keys:
        return out

    grp = out.groupby(keys, sort=False)
    lap_size = grp[out.columns[0]].transform("size").clip(lower=1)

    if "PitStop" in out.columns:
        own_pit = pd.to_numeric(out["PitStop"], errors="coerce").fillna(0)
        pit_sum = grp["PitStop"].transform("sum")
        out["field_other_pitting_count"] = (pit_sum - own_pit).clip(lower=0).astype("float32")
        out["field_other_pitting_share"] = (
            out["field_other_pitting_count"] / (lap_size - 1).clip(lower=1)
        ).astype("float32")

    if "Compound" in out.columns:
        comp = out["Compound"].astype("string").fillna("UNKNOWN")
        for value in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
            flag = comp.eq(value).astype("int8")
            out[f"field_share_{value.lower()}"] = (
                flag.groupby([out[k] for k in keys], sort=False).transform("sum") / lap_size
            ).astype("float32")
        wet_flag = comp.isin(["INTERMEDIATE", "WET"]).astype("int8")
        out["field_wet_weather_share"] = (
            wet_flag.groupby([out[k] for k in keys], sort=False).transform("sum") / lap_size
        ).astype("float32")

    if "TyreLife" in out.columns:
        tyre = pd.to_numeric(out["TyreLife"], errors="coerce")
        out["field_median_tyrelife"] = grp["TyreLife"].transform("median")
        out["own_tyrelife_lap_rank_pct"] = grp["TyreLife"].rank(pct=True).astype("float32")
        out["own_tyrelife_vs_lap_median"] = (
            tyre - pd.to_numeric(out["field_median_tyrelife"], errors="coerce")
        ).astype("float32")

    if "LapTime_Delta" in out.columns:
        delta = pd.to_numeric(out["LapTime_Delta"], errors="coerce")
        delta_mean = grp["LapTime_Delta"].transform("mean")
        delta_std = grp["LapTime_Delta"].transform("std").replace(0, np.nan)
        out["own_laptime_delta_lap_z"] = (
            (delta - delta_mean) / delta_std
        ).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

    if "Position_Change" in out.columns:
        out["lap_position_change_volatility"] = (
            grp["Position_Change"].transform("std").fillna(0).astype("float32")
        )

    if {"Position", "PitStop"}.issubset(out.columns):
        n = len(out)
        row_id = np.arange(n, dtype=np.int64)
        neighbor = out[keys + ["Position", "PitStop"]].copy()
        if "Position_Change" in out.columns:
            neighbor["Position_Change"] = out["Position_Change"]
        neighbor["_row_id"] = row_id

        asc = neighbor.sort_values(keys + ["Position", "_row_id"], kind="mergesort")
        asc_grp = asc.groupby(keys, sort=False)
        asc_rows = asc["_row_id"].to_numpy(dtype=np.int64)
        ahead_pit = np.zeros(n, dtype=np.int8)
        ahead_pit[asc_rows] = (
            asc_grp["PitStop"].shift(1).fillna(0).astype(np.int8).to_numpy()
        )

        desc = neighbor.sort_values(
            keys + ["Position", "_row_id"],
            ascending=[True] * len(keys) + [False, True],
            kind="mergesort",
        )
        desc_grp = desc.groupby(keys, sort=False)
        desc_rows = desc["_row_id"].to_numpy(dtype=np.int64)
        behind_pit = np.zeros(n, dtype=np.int8)
        behind_pit[desc_rows] = (
            desc_grp["PitStop"].shift(1).fillna(0).astype(np.int8).to_numpy()
        )
        out["ahead_position_pitted_same_lap"] = ahead_pit
        out["behind_position_pitted_same_lap"] = behind_pit
        out["nearby_position_pitted_same_lap"] = (
            (ahead_pit + behind_pit) > 0
        ).astype(np.int8)

        if "Position_Change" in neighbor.columns:
            ahead_change = np.zeros(n, dtype=np.float32)
            behind_change = np.zeros(n, dtype=np.float32)
            ahead_change[asc_rows] = (
                asc_grp["Position_Change"].shift(1).fillna(0).to_numpy(dtype=np.float32)
            )
            behind_change[desc_rows] = (
                desc_grp["Position_Change"].shift(1).fillna(0).to_numpy(dtype=np.float32)
            )
            out["nearby_position_change_pressure"] = ahead_change - behind_change

    return out
