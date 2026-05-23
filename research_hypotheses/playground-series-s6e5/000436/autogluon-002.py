def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()
    out["_row_order_000436"] = np.arange(len(out))

    compound = out["Compound"].astype("string").str.upper().str.strip()
    lap = pd.to_numeric(out["LapNumber"], errors="coerce").astype("float32")
    progress = pd.to_numeric(out["RaceProgress"], errors="coerce").clip(0.01, 1.0)
    tyre_life = pd.to_numeric(out["TyreLife"], errors="coerce").astype("float32")
    est_total = (lap / progress).clip(lower=lap, upper=90)
    laps_remaining = (est_total - lap).clip(0, 90)

    out["prior_laps_remaining_est_000436"] = laps_remaining.astype("float32")
    out["prior_is_wet_000436"] = compound.isin(["INTERMEDIATE", "WET"]).astype("int8")
    out["prior_is_slick_000436"] = compound.isin(["SOFT", "MEDIUM", "HARD"]).astype(
        "int8"
    )
    out["prior_degradation_per_life_000436"] = (
        pd.to_numeric(out["Cumulative_Degradation"], errors="coerce") / (tyre_life + 1.0)
    ).astype("float32")

    def cut_int(series, bins):
        return (
            pd.cut(series, bins=bins, labels=False, include_lowest=True)
            .astype("float32")
            .fillna(-1)
            .astype("int16")
        )

    out["prior_tyre_life_bin_000436"] = cut_int(
        tyre_life, [-np.inf, 3, 8, 15, 25, 40, np.inf]
    )
    out["prior_laps_remaining_bin_000436"] = cut_int(
        laps_remaining, [-np.inf, 3, 8, 15, 25, 40, np.inf]
    )
    out["prior_progress_bin_000436"] = cut_int(
        out["RaceProgress"], [-np.inf, 0.15, 0.35, 0.6, 0.8, np.inf]
    )
    out["prior_stint_bin_000436"] = cut_int(out["Stint"], [0, 1, 2, 3, 4, np.inf])

    fine_key = [
        "Race",
        "Compound",
        "prior_stint_bin_000436",
        "prior_tyre_life_bin_000436",
        "prior_progress_bin_000436",
    ]
    broad_key = ["Race", "Compound", "prior_tyre_life_bin_000436"]
    year = pd.to_numeric(out["Year"], errors="coerce")
    pit = pd.to_numeric(out["PitStop"], errors="coerce").fillna(0.0).astype("float32")

    out["prior_year_hazard_fine_000436"] = 0.0
    out["prior_year_hazard_broad_000436"] = 0.0
    out["prior_year_count_fine_000436"] = 0.0
    out["prior_year_count_broad_000436"] = 0.0

    for yy in sorted(year.dropna().unique()):
        current_mask = year == yy
        history_mask = year < yy
        if not history_mask.any():
            continue

        history = out.loc[history_mask].copy()
        history["_pit_proxy_000436"] = pit.loc[history_mask].to_numpy()

        for keys, mean_col, count_col in (
            (fine_key, "prior_year_hazard_fine_000436", "prior_year_count_fine_000436"),
            (
                broad_key,
                "prior_year_hazard_broad_000436",
                "prior_year_count_broad_000436",
            ),
        ):
            stats = history.groupby(keys, dropna=False, sort=False)[
                "_pit_proxy_000436"
            ].agg(["mean", "count"])
            idx = pd.MultiIndex.from_frame(out.loc[current_mask, keys])
            out.loc[current_mask, mean_col] = (
                idx.map(stats["mean"]).astype("float32").fillna(0.0).to_numpy()
            )
            out.loc[current_mask, count_col] = (
                idx.map(stats["count"]).astype("float32").fillna(0.0).to_numpy()
            )

    sparse = out["prior_year_count_fine_000436"] < 10
    out["prior_year_hazard_blend_000436"] = np.where(
        sparse,
        out["prior_year_hazard_broad_000436"],
        out["prior_year_hazard_fine_000436"],
    ).astype("float32")
    out["prior_year_log_count_000436"] = np.log1p(
        np.where(
            sparse,
            out["prior_year_count_broad_000436"],
            out["prior_year_count_fine_000436"],
        )
    ).astype("float32")
    out["prior_hazard_x_tyre_life_000436"] = (
        out["prior_year_hazard_blend_000436"] * tyre_life
    ).astype("float32")
    out["prior_hazard_x_laps_remaining_000436"] = (
        out["prior_year_hazard_blend_000436"] * laps_remaining
    ).astype("float32")

    out = out.sort_values("_row_order_000436", kind="mergesort").drop(
        columns=["_row_order_000436"]
    )
    return out
