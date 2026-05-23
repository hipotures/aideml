def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()
    eps = 1e-6

    for c in ["Race", "Compound", "Driver"]:
        out[c] = out[c].astype("string").fillna("UNK")
        out[c + "_freq"] = out[c].map(out[c].value_counts(normalize=True)).astype(
            "float32"
        )
        out[c + "_code"] = out[c].astype("category").cat.codes.astype("int16")

    num_cols = ["TyreLife", "LapTime_Delta", "Cumulative_Degradation", "Stint"]
    race_lap_key = ["Year", "Race", "LapNumber"]
    race_lap_comp_key = ["Year", "Race", "LapNumber", "Compound"]
    race_lap = out.groupby(race_lap_key, sort=False)
    race_lap_comp = out.groupby(race_lap_comp_key, sort=False)

    out["race_lap_cars"] = race_lap["Driver"].transform("count").astype("int16")
    out["compound_lap_cars"] = (
        race_lap_comp["Driver"].transform("count").astype("int16")
    )

    for c in num_cols:
        out[f"{c}_race_lap_pct"] = race_lap[c].rank(pct=True).astype("float32")
        out[f"{c}_compound_lap_pct"] = race_lap_comp[c].rank(pct=True).astype(
            "float32"
        )
        rl_mean = race_lap[c].transform("mean")
        rl_std = race_lap[c].transform("std").fillna(0.0)
        cl_mean = race_lap_comp[c].transform("mean")
        cl_std = race_lap_comp[c].transform("std").fillna(0.0)
        out[f"{c}_race_lap_z"] = ((out[c] - rl_mean) / (rl_std + eps)).astype(
            "float32"
        )
        out[f"{c}_compound_lap_z"] = ((out[c] - cl_mean) / (cl_std + eps)).astype(
            "float32"
        )

    out["older_tyre_rivals"] = (
        (race_lap["TyreLife"].transform("sum") - out["TyreLife"])
        .div(out["TyreLife"].replace(0, np.nan))
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .astype("float32")
    )
    out["older_tyre_share"] = (1.0 - out["TyreLife_race_lap_pct"]).astype("float32")
    out["younger_tyre_share"] = out["TyreLife_race_lap_pct"].astype("float32")
    out["alt_compound_rivals"] = (
        out["race_lap_cars"] - out["compound_lap_cars"]
    ).astype("int16")
    out["alt_compound_share"] = (
        out["alt_compound_rivals"] / (out["race_lap_cars"] - 1 + eps)
    ).astype("float32")

    same_stint_counts = out.groupby(
        ["Year", "Race", "LapNumber", "Stint"], sort=False
    )["Driver"].transform("count")
    out["alt_stint_rivals"] = (out["race_lap_cars"] - same_stint_counts).astype(
        "int16"
    )
    out["alt_stint_share"] = (
        out["alt_stint_rivals"] / (out["race_lap_cars"] - 1 + eps)
    ).astype("float32")

    out["_orig_order_000439"] = np.arange(len(out))
    s = out.sort_values(["Year", "Race", "Driver", "LapNumber", "_orig_order_000439"])
    gd = s.groupby(["Year", "Race", "Driver"], sort=False)
    s["_pit_prev1"] = gd["PitStop"].shift(1).fillna(0)
    s["_pit_prev2"] = gd["PitStop"].shift(2).fillna(0)
    s["_pit_prev3"] = gd["PitStop"].shift(3).fillna(0)
    s["_pit_cum_before"] = gd["PitStop"].cumsum() - s["PitStop"]
    recent = s[
        [
            "_orig_order_000439",
            "_pit_prev1",
            "_pit_prev2",
            "_pit_prev3",
            "_pit_cum_before",
        ]
    ].set_index("_orig_order_000439")
    out = out.join(recent, on="_orig_order_000439")

    out["recent_pit_3"] = (
        out["_pit_prev1"] + 0.7 * out["_pit_prev2"] + 0.4 * out["_pit_prev3"]
    ).astype("float32")
    out["prior_stops"] = out["_pit_cum_before"].astype("float32")

    race_lap = out.groupby(race_lap_key, sort=False)
    out["already_stopped_rivals"] = (
        race_lap["_pit_cum_before"].transform("sum") - out["_pit_cum_before"]
    ).astype("float32")
    out["current_pit_rivals"] = (
        race_lap["PitStop"].transform("sum") - out["PitStop"]
    ).astype("float32")
    out["recent_pit_rivals"] = (
        race_lap["recent_pit_3"].transform("sum") - out["recent_pit_3"]
    ).astype("float32")
    out["stops_vs_field"] = (
        out["prior_stops"] - race_lap["_pit_cum_before"].transform("mean")
    ).astype("float32")

    pos_gap = (out["Position"] - race_lap["Position"].transform("median")).abs()
    close_weight = (1.0 / (1.0 + pos_gap)).astype("float32")
    out["nearby_recent_pit_pressure"] = (
        out["recent_pit_rivals"] / (out["race_lap_cars"] - 1 + eps) * close_weight
    ).astype("float32")
    out["undercut_weak_score"] = (
        out["nearby_recent_pit_pressure"]
        * (1 - out["PitStop"].clip(0, 1))
        * (0.5 + 0.5 * out["TyreLife_race_lap_pct"])
    ).astype("float32")
    out["overcut_weak_score"] = (
        out["recent_pit_3"]
        * out["older_tyre_share"]
        * (0.5 + 0.5 * out["LapTime_Delta_race_lap_pct"])
    ).astype("float32")
    prior_stop_frac = (out["prior_stops"] / 3.0).clip(0, 1).astype("float32")
    out["go_long_weak_score"] = (
        out["TyreLife_race_lap_pct"]
        * (1.0 - prior_stop_frac)
        * (0.5 + 0.5 * out["RaceProgress"])
    ).astype("float32")

    out["tyrelife_x_progress"] = (out["TyreLife"] * out["RaceProgress"]).astype(
        "float32"
    )
    out["degradation_per_tyre_lap"] = (
        out["Cumulative_Degradation"] / (out["TyreLife"] + 1.0)
    ).astype("float32")
    out["lapdelta_degradation_gap"] = (
        out["LapTime_Delta"] - out["Cumulative_Degradation"]
    ).astype("float32")
    out["stint_progress_gap"] = (out["Stint"] - out["prior_stops"] - 1).astype(
        "float32"
    )

    out = out.drop(
        columns=[
            "_orig_order_000439",
            "_pit_prev1",
            "_pit_prev2",
            "_pit_prev3",
            "_pit_cum_before",
        ]
    )
    return out
