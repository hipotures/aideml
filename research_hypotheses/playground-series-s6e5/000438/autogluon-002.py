def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()
    out["_row_order_000438"] = np.arange(len(out))

    group_cols = ["Year", "Race", "Driver", "Stint"]
    out = out.sort_values(
        group_cols + ["LapNumber", "_row_order_000438"], kind="mergesort"
    ).reset_index(drop=True)
    grp = out.groupby(group_cols, sort=False)

    stint_idx = grp.cumcount()
    stint_seen = (stint_idx + 1).astype("float32")
    out["StintLapIndex"] = stint_seen.astype("int16")
    out["StintLength"] = stint_seen.astype("int16")
    out["StintProgressLocal"] = (stint_seen / (stint_seen + 5.0)).astype("float32")
    out["JustPitted"] = (stint_idx == 0).astype("int8")
    out["WarmupLap2"] = (stint_idx <= 1).astype("int8")
    out["WarmupLap3"] = (stint_idx <= 2).astype("int8")
    out["WarmupDecay"] = (1.0 / stint_seen).astype("float32")

    metric_map = {
        "LapTime_Delta": "LapTimeDelta",
        "LapTime (s)": "LapTimeS",
        "Cumulative_Degradation": "CumDeg",
    }

    for col, prefix in metric_map.items():
        values = pd.to_numeric(out[col], errors="coerce").astype("float32")
        g = values.groupby([out[c] for c in group_cols], sort=False)
        lag1 = g.shift(1)
        lag2 = g.shift(2)
        diff1 = values - lag1
        diff2 = values - lag2
        accel = values - 2.0 * lag1 + lag2

        roll_slope3 = g.transform(lambda x: x.diff().rolling(3, min_periods=1).mean())
        roll_slope5 = g.transform(lambda x: x.diff().rolling(5, min_periods=1).mean())
        ewm_mean = g.transform(lambda x: x.ewm(alpha=0.35, adjust=False).mean())
        roll_mean3 = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        roll_std5 = g.transform(lambda x: x.rolling(5, min_periods=2).std())

        out[f"{prefix}_Lag1"] = lag1.fillna(values).astype("float32")
        out[f"{prefix}_Diff1"] = diff1.fillna(0.0).astype("float32")
        out[f"{prefix}_Diff2"] = diff2.fillna(0.0).astype("float32")
        out[f"{prefix}_Accel"] = accel.fillna(0.0).astype("float32")
        out[f"{prefix}_RollSlope3"] = roll_slope3.fillna(0.0).astype("float32")
        out[f"{prefix}_RollSlope5"] = roll_slope5.fillna(0.0).astype("float32")
        out[f"{prefix}_EwmResid"] = (values - ewm_mean).fillna(0.0).astype("float32")
        out[f"{prefix}_ChangeZ5"] = (
            ((values - roll_mean3) / (roll_std5 + 1e-6))
            .fillna(0.0)
            .clip(-10, 10)
            .astype("float32")
        )

    out["LatentPaceProxy"] = (
        pd.to_numeric(out["LapTime_Delta"], errors="coerce")
        .astype("float32")
        .groupby([out[c] for c in group_cols], sort=False)
        .transform(lambda x: x.ewm(alpha=0.20, adjust=False).mean())
        .astype("float32")
    )
    out["LatentPaceGap"] = (
        pd.to_numeric(out["LapTime_Delta"], errors="coerce").astype("float32")
        - out["LatentPaceProxy"]
    ).astype("float32")
    prev_latent = out.groupby(group_cols, sort=False)["LatentPaceProxy"].shift(1)
    out["LatentPaceStep"] = (
        out["LatentPaceProxy"] - prev_latent
    ).fillna(0.0).astype("float32")

    out["TyreCliffScore"] = (
        out["LapTimeDelta_RollSlope3"]
        + 0.5 * out["LapTimeDelta_Accel"]
        + 0.35 * out["CumDeg_RollSlope3"]
        + 0.35 * out["CumDeg_Accel"]
        + 0.5 * out["LatentPaceGap"]
    ).astype("float32")
    out["TyreCliffWarm"] = (out["TyreCliffScore"] * out["WarmupDecay"]).astype(
        "float32"
    )
    out["TyreLife_x_Cliff"] = (
        pd.to_numeric(out["TyreLife"], errors="coerce").astype("float32")
        * out["TyreCliffScore"]
    ).astype("float32")
    out["RaceProgress_x_Cliff"] = (
        pd.to_numeric(out["RaceProgress"], errors="coerce").astype("float32")
        * out["TyreCliffScore"]
    ).astype("float32")

    out = out.sort_values("_row_order_000438", kind="mergesort").drop(
        columns=["_row_order_000438"]
    )
    return out
