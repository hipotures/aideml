from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    sort_cols = [c for c in ["Year", "Race", "Driver", "LapNumber"] if c in out.columns]
    if not sort_cols:
        return out

    tmp = out[sort_cols].copy()
    tmp["_pos"] = np.arange(n, dtype=np.int64)
    order = tmp.sort_values(sort_cols + ["_pos"], kind="mergesort")["_pos"].to_numpy()

    next_order = np.full(n, -1, dtype=np.int64)
    if n > 1:
        next_order[:-1] = order[1:]
    next_pos = np.full(n, -1, dtype=np.int64)
    next_pos[order] = next_order

    valid_next = next_pos >= 0
    same_driver_race = valid_next.copy()
    for c in ["Year", "Race", "Driver"]:
        if c in out.columns:
            vals = out[c].to_numpy()
            eq = np.zeros(n, dtype=bool)
            eq[valid_next] = vals[valid_next] == vals[next_pos[valid_next]]
            same_driver_race &= eq

    if "LapNumber" in out.columns:
        lap = pd.to_numeric(out["LapNumber"], errors="coerce").to_numpy(dtype=float)
        gap = np.full(n, -1.0, dtype=np.float32)
        gap[valid_next] = lap[next_pos[valid_next]] - lap[valid_next]
        out["SequenceNextLapGap"] = gap
        out["SameDriverRaceNextLap"] = (same_driver_race & (gap == 1)).astype(np.int8)

    out["SameDriverRaceNextRow"] = same_driver_race.astype(np.int8)
    out["EndOfDriverRaceSequence"] = (~same_driver_race).astype(np.int8)

    if "PitStop" in out.columns:
        pit = pd.to_numeric(out["PitStop"], errors="coerce").fillna(0).to_numpy()
        prev_order = np.full(n, -1, dtype=np.int64)
        if n > 1:
            prev_order[1:] = order[:-1]
        prev_pos = np.full(n, -1, dtype=np.int64)
        prev_pos[order] = prev_order
        valid_prev = prev_pos >= 0

        same_prev = valid_prev.copy()
        for c in ["Year", "Race", "Driver"]:
            if c in out.columns:
                vals = out[c].to_numpy()
                eq = np.zeros(n, dtype=bool)
                eq[valid_prev] = vals[valid_prev] == vals[prev_pos[valid_prev]]
                same_prev &= eq

        prev_pit = np.zeros(n, dtype=np.int8)
        prev_pit[same_prev] = pit[prev_pos[same_prev]].astype(np.int8)
        out["PrevPitStop"] = prev_pit

        hist = pd.DataFrame({"_pos": order, "pit": pit[order]})
        hist_cols = []
        for c in ["Year", "Race", "Driver"]:
            if c in out.columns:
                hist[c] = out[c].to_numpy()[order]
                hist_cols.append(c)
        if hist_cols:
            prior = hist.groupby(hist_cols, sort=False)["pit"].cumsum() - hist["pit"]
        else:
            prior = hist["pit"].cumsum() - hist["pit"]
        prior_count = np.zeros(n, dtype=np.float32)
        prior_count[hist["_pos"].to_numpy(dtype=np.int64)] = prior.to_numpy(dtype=np.float32)
        out["PriorPitStopCount"] = prior_count

    return out
