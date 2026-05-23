def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    X = df.copy()
    X["_row_order_000435"] = np.arange(len(X))

    group_cols = ["Year", "Race", "Driver", "Stint"]
    X = X.sort_values(
        group_cols + ["LapNumber", "_row_order_000435"], kind="mergesort"
    ).reset_index(drop=True)

    compound = X["Compound"].astype("string").str.upper().str.strip()
    X["wet_flag"] = compound.isin(["INTERMEDIATE", "WET"]).astype("int8")
    X["slick_flag"] = 1 - X["wet_flag"]

    lap = pd.to_numeric(X["LapNumber"], errors="coerce").astype("float32")
    progress = pd.to_numeric(X["RaceProgress"], errors="coerce").clip(0.01, 1.0)
    tyre_life = pd.to_numeric(X["TyreLife"], errors="coerce").astype("float32")
    est_total_laps = (lap / progress).clip(lower=lap, upper=90)
    X["Estimated_Total_Laps"] = est_total_laps.astype("float32")
    X["Laps_Remaining_Est"] = (est_total_laps - lap).clip(0, 90).astype("float32")

    expected_life = compound.map(
        {
            "SOFT": 18.0,
            "MEDIUM": 25.0,
            "HARD": 35.0,
            "INTERMEDIATE": 20.0,
            "WET": 16.0,
        }
    ).fillna(24.0)
    X["Stint_Remaining_Proxy"] = (expected_life - tyre_life).clip(0, 90).astype(
        "float32"
    )

    g = X.groupby(group_cols, sort=False)
    lap_gap = g["LapNumber"].diff().replace(0, np.nan)
    X["LapTimeDelta_Slope"] = (
        (g["LapTime_Delta"].diff() / lap_gap)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )
    X["Degradation_Slope"] = (
        (g["Cumulative_Degradation"].diff() / lap_gap)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )
    X["Position_Change_Slope"] = (
        (g["Position_Change"].diff() / lap_gap)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype("float32")
    )
    X["LapTimeDelta_Slope_3"] = (
        X.groupby(group_cols, sort=False)["LapTimeDelta_Slope"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
        .astype("float32")
    )
    X["Degradation_Slope_3"] = (
        X.groupby(group_cols, sort=False)["Degradation_Slope"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
        .astype("float32")
    )

    def cut_int(series, bins):
        return (
            pd.cut(series, bins=bins, labels=False, include_lowest=True)
            .astype("float32")
            .fillna(-1)
            .astype("int16")
        )

    X["TyreLife_Bin"] = cut_int(tyre_life, [-np.inf, 3, 8, 15, 25, 40, np.inf])
    X["LapsRemaining_Bin"] = cut_int(
        X["Laps_Remaining_Est"], [-np.inf, 3, 8, 15, 25, 40, np.inf]
    )
    X["RaceProgress_Bin"] = cut_int(
        X["RaceProgress"], [-np.inf, 0.15, 0.35, 0.6, 0.8, np.inf]
    )
    X["Position_Bin"] = cut_int(X["Position"], [0, 3, 6, 10, 15, 20, np.inf])
    X["LapTimeSlope_Bin"] = cut_int(
        X["LapTimeDelta_Slope_3"].clip(-30, 30), [-np.inf, -3, -1, 1, 3, np.inf]
    )
    X["DegSlope_Bin"] = cut_int(
        X["Degradation_Slope_3"].clip(-30, 30), [-np.inf, -3, -1, 1, 3, np.inf]
    )

    fine_key = [
        "Compound",
        "wet_flag",
        "TyreLife_Bin",
        "LapsRemaining_Bin",
        "RaceProgress_Bin",
        "Position_Bin",
        "LapTimeSlope_Bin",
        "DegSlope_Bin",
    ]
    broad_key = [
        "Compound",
        "wet_flag",
        "TyreLife_Bin",
        "LapsRemaining_Bin",
        "RaceProgress_Bin",
        "Position_Bin",
    ]

    def add_prior_mean(keys, value_col, prefix):
        values = pd.to_numeric(X[value_col], errors="coerce").fillna(0.0)
        grp = X.groupby(keys, dropna=False, sort=False)
        cnt = grp.cumcount().astype("float32")
        csum = values.groupby([X[k] for k in keys], sort=False).cumsum() - values
        prior = csum / cnt.where(cnt > 0, np.nan)
        fallback = float(values.mean()) if len(values) else 0.0
        X[f"{prefix}_{value_col}_prior_mean"] = prior.fillna(fallback).astype(
            "float32"
        )
        X[f"{prefix}_prior_count"] = cnt.clip(0, 10000).astype("float32")

    for key, prefix in ((fine_key, "retrieval_fine"), (broad_key, "retrieval_broad")):
        add_prior_mean(key, "PitStop", prefix)
        add_prior_mean(key, "TyreLife", prefix)
        add_prior_mean(key, "Laps_Remaining_Est", prefix)
        add_prior_mean(key, "LapTime_Delta", prefix)
        add_prior_mean(key, "Degradation_Slope_3", prefix)

    sparse = X["retrieval_fine_prior_count"] < 25
    X["retrieval_pit_rate"] = np.where(
        sparse,
        X["retrieval_broad_PitStop_prior_mean"],
        X["retrieval_fine_PitStop_prior_mean"],
    ).astype("float32")
    X["retrieval_count"] = np.where(
        sparse, X["retrieval_broad_prior_count"], X["retrieval_fine_prior_count"]
    ).astype("float32")
    X["retrieval_density"] = np.log1p(X["retrieval_count"].fillna(0)).astype("float32")
    X["retrieval_dist_tyre_life"] = (
        tyre_life - X["retrieval_fine_TyreLife_prior_mean"]
    ).abs().astype("float32")
    X["retrieval_dist_laps_remaining"] = (
        X["Laps_Remaining_Est"] - X["retrieval_fine_Laps_Remaining_Est_prior_mean"]
    ).abs().astype("float32")
    X["retrieval_dist_lap_delta"] = (
        X["LapTime_Delta"] - X["retrieval_fine_LapTime_Delta_prior_mean"]
    ).abs().astype("float32")
    X["retrieval_dist_deg_slope"] = (
        X["Degradation_Slope_3"] - X["retrieval_fine_Degradation_Slope_3_prior_mean"]
    ).abs().astype("float32")
    X["retrieval_tyre_life_vs_regime"] = (
        tyre_life / (1.0 + X["retrieval_fine_TyreLife_prior_mean"].abs())
    ).astype("float32")
    X["retrieval_remaining_vs_regime"] = (
        X["Laps_Remaining_Est"]
        / (1.0 + X["retrieval_fine_Laps_Remaining_Est_prior_mean"].abs())
    ).astype("float32")

    X = X.sort_values("_row_order_000435", kind="mergesort").drop(
        columns=["_row_order_000435"]
    )
    return X
