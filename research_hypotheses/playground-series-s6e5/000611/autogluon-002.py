from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    X["RaceProgressBin"] = np.clip((X["RaceProgress"] * 10).astype(int), 0, 9).astype(
        "int16"
    )
    X["LapNumberBin"] = ((X["LapNumber"] - 1) // 5).astype("int16")
    X["TyreLifeBin"] = pd.cut(
        X["TyreLife"],
        bins=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 100],
        labels=False,
        include_lowest=True,
    ).astype("int16")

    X["TyreLifeOverLap"] = X["TyreLife"] / X["LapNumber"].clip(lower=1)
    X["TyreLifeOverRaceProgress"] = X["TyreLife"] / (X["RaceProgress"] + 0.02)
    X["DegPerTyreLap"] = X["Cumulative_Degradation"] / X["TyreLife"].clip(lower=1)
    X["LateStintPressure"] = X["TyreLife"] * (1.0 + X["RaceProgress"]) * X["Stint"]
    X["PositionLossPressure"] = (-X["Position_Change"]).clip(lower=0) * X["TyreLife"]
    X["PitFlagTyreLife"] = X["PitStop"] * X["TyreLife"]
    X["PitFlagDeg"] = X["PitStop"] * X["Cumulative_Degradation"]

    X["key_compound_tlife"] = (
        X["Compound"].astype(str) + "_" + X["TyreLifeBin"].astype(str)
    ).astype("category")
    X["key_compound_stint_prog"] = (
        X["Compound"].astype(str)
        + "_"
        + X["Stint"].astype(str)
        + "_"
        + X["RaceProgressBin"].astype(str)
    ).astype("category")
    X["key_race_lap"] = (
        X["Race"].astype(str) + "_" + X["LapNumberBin"].astype(str)
    ).astype("category")
    X["key_driver_stint"] = (
        X["Driver"].astype(str) + "_" + X["Stint"].astype(str)
    ).astype("category")
    X["key_year_race_stint"] = (
        X["Year"].astype(str)
        + "_"
        + X["Race"].astype(str)
        + "_"
        + X["Stint"].astype(str)
    ).astype("category")

    def add_freq_features(frame, key_col, parent_key, prefix):
        key_count = (
            frame.groupby(key_col, observed=True)[key_col]
            .transform("size")
            .astype("float32")
        )
        if isinstance(parent_key, str):
            parent_count = (
                frame.groupby(parent_key, observed=True)[parent_key]
                .transform("size")
                .astype("float32")
            )
        else:
            parent_count = (
                frame.groupby(parent_key, observed=True)[key_col]
                .transform("size")
                .astype("float32")
            )
        global_n = float(len(frame))

        frame[f"{prefix}_count"] = key_count
        frame[f"{prefix}_log_count"] = np.log1p(key_count).astype("float32")
        frame[f"{prefix}_share_in_parent"] = (
            key_count / parent_count.clip(lower=1)
        ).astype("float32")
        frame[f"{prefix}_global_freq"] = (key_count / global_n).astype("float32")
        frame[f"{prefix}_shrunk_freq"] = ((key_count + 5.0) / (global_n + 50.0)).astype(
            "float32"
        )

    add_freq_features(X, "key_compound_tlife", "Compound", "hz_compound_tlife")
    add_freq_features(
        X,
        "key_compound_stint_prog",
        (X["Compound"].astype(str) + "_" + X["Stint"].astype(str)).astype("category"),
        "hz_compound_stint_prog",
    )
    add_freq_features(X, "key_race_lap", "Race", "hz_race_lap")
    add_freq_features(X, "key_driver_stint", "Driver", "hz_driver_stint")
    add_freq_features(
        X,
        "key_year_race_stint",
        (X["Year"].astype(str) + "_" + X["Race"].astype(str)).astype("category"),
        "hz_year_race_stint",
    )

    compound_tlife_median = X.groupby("Compound", observed=True)["TyreLife"].transform(
        "median"
    )
    compound_deg_median = X.groupby("Compound", observed=True)[
        "Cumulative_Degradation"
    ].transform("median")
    X["TyreLifeVsCompoundMedian"] = (X["TyreLife"] - compound_tlife_median).astype(
        "float32"
    )
    X["DegVsCompoundMedian"] = (
        X["Cumulative_Degradation"] - compound_deg_median
    ).astype("float32")

    race_max_lap = (
        X.groupby("Race", observed=True)["LapNumber"].transform("max").clip(lower=1)
    )
    X["RaceLapFrac"] = (X["LapNumber"] / race_max_lap).astype("float32")
    X["TyreLifeMinusExpectedLap"] = (
        X["TyreLife"] - X["RaceLapFrac"] * race_max_lap
    ).astype("float32")

    return X
