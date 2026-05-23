from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["Year", "Race", "Driver", "Stint"]
    sort_cols = group_cols + ["LapNumber"]

    out["_orig_order"] = np.arange(len(out))
    s = out.sort_values(sort_cols, kind="mergesort").copy()
    g = s.groupby(group_cols, sort=False)

    lap = s["LapNumber"].astype(float)
    lt = s["LapTime (s)"].astype(float)
    deg = s["Cumulative_Degradation"].astype(float)

    s["lt_prev"] = g["LapTime (s)"].shift(1)
    s["deg_prev"] = g["Cumulative_Degradation"].shift(1)
    s["lap_prev"] = g["LapNumber"].shift(1).astype(float)

    s["lap_time_delta_past"] = s["lt_prev"] - g["LapTime (s)"].shift(2)
    s["acceleration_of_laptime"] = s["lap_time_delta_past"] - g["LapTime (s)"].shift(
        2
    ).sub(g["LapTime (s)"].shift(3))

    x_prev = s["lap_prev"]
    y_prev = s["lt_prev"]
    d_prev = s["deg_prev"]

    for w in (3, 5):
        mx = (
            x_prev.groupby([s[c] for c in group_cols], sort=False)
            .rolling(w, min_periods=2)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
        my = (
            y_prev.groupby([s[c] for c in group_cols], sort=False)
            .rolling(w, min_periods=2)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
        mxy = (
            (x_prev * y_prev)
            .groupby([s[c] for c in group_cols], sort=False)
            .rolling(w, min_periods=2)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
        mx2 = (
            (x_prev * x_prev)
            .groupby([s[c] for c in group_cols], sort=False)
            .rolling(w, min_periods=2)
            .mean()
            .reset_index(level=list(range(len(group_cols))), drop=True)
        )
        s[f"lap_time_slope_{w}"] = (mxy - mx * my) / (mx2 - mx * mx + 1e-6)

    mx = (
        x_prev.groupby([s[c] for c in group_cols], sort=False)
        .expanding(min_periods=2)
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    md = (
        d_prev.groupby([s[c] for c in group_cols], sort=False)
        .expanding(min_periods=2)
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    mxd = (
        (x_prev * d_prev)
        .groupby([s[c] for c in group_cols], sort=False)
        .expanding(min_periods=2)
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    mx2 = (
        (x_prev * x_prev)
        .groupby([s[c] for c in group_cols], sort=False)
        .expanding(min_periods=2)
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    s["degradation_slope"] = (mxd - mx * md) / (mx2 - mx * mx + 1e-6)

    my = (
        y_prev.groupby([s[c] for c in group_cols], sort=False)
        .expanding(min_periods=2)
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    mxy = (
        (x_prev * y_prev)
        .groupby([s[c] for c in group_cols], sort=False)
        .expanding(min_periods=2)
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    slope = (mxy - mx * my) / (mx2 - mx * mx + 1e-6)
    intercept = my - slope * mx
    s["residual_vs_stint_linear_fit"] = lt - (intercept + slope * lap)

    race_lap_med = out.groupby(["Year", "Race", "LapNumber"])["LapTime (s)"].transform(
        "median"
    )
    out["pace_residual_race_lap"] = out["LapTime (s)"] - race_lap_med
    s = s.merge(
        out[["_orig_order", "pace_residual_race_lap"]],
        on="_orig_order",
        how="left",
        suffixes=("", "_drop"),
    )
    g = s.groupby(group_cols, sort=False)
    s["pace_residual_prev"] = g["pace_residual_race_lap"].shift(1)
    pr_mean = (
        s["pace_residual_prev"]
        .groupby([s[c] for c in group_cols], sort=False)
        .expanding(min_periods=2)
        .mean()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    pr_std = (
        s["pace_residual_prev"]
        .groupby([s[c] for c in group_cols], sort=False)
        .expanding(min_periods=2)
        .std()
        .reset_index(level=list(range(len(group_cols))), drop=True)
    )
    s["normalized_residual_race_lap"] = (s["pace_residual_race_lap"] - pr_mean) / (
        pr_std.fillna(0) + 1.0
    )

    new_cols = [
        "lap_time_slope_3",
        "lap_time_slope_5",
        "degradation_slope",
        "residual_vs_stint_linear_fit",
        "acceleration_of_laptime",
        "pace_residual_race_lap",
        "normalized_residual_race_lap",
    ]
    s = s.sort_values("_orig_order")
    for c in new_cols:
        out[c] = s[c].to_numpy()
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out = out.drop(columns=["_orig_order"])
    return out
