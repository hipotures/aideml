def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()
    out["_row_order_000433"] = np.arange(len(out))

    group_cols = ["Year", "Race", "Driver", "Stint"]
    base_cols = ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "TyreLife"]
    for col in base_cols + ["LapNumber", "RaceProgress"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")

    out = out.sort_values(
        group_cols + ["LapNumber", "_row_order_000433"], kind="mergesort"
    ).reset_index(drop=True)
    g = out.groupby(group_cols, sort=False)
    stint_seen = g.cumcount().astype("float32") + 1.0
    out["stint_lap_idx_000433"] = stint_seen.astype("float32")
    out["stint_progress_seen_000433"] = (stint_seen / (stint_seen + 5.0)).astype(
        "float32"
    )

    for col in base_cols:
        safe = (
            col.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        series = out[col]
        prev1 = g[col].shift(1)
        prev2 = g[col].shift(2)
        prev5 = g[col].shift(5)

        out[f"{safe}_diff1_000433"] = (series - prev1).fillna(0.0).astype("float32")
        out[f"{safe}_prior_diff1_000433"] = (
            prev1 - prev2
        ).fillna(0.0).astype("float32")
        out[f"{safe}_curvature2_000433"] = (
            series - 2.0 * prev1 + prev2
        ).fillna(0.0).astype("float32")
        out[f"{safe}_slope5_000433"] = (
            (series - prev5) / 5.0
        ).fillna(0.0).astype("float32")

        prior = prev1.groupby([out[c] for c in group_cols], sort=False)
        for window in (2, 3, 5):
            roll_mean = prior.transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            roll_std = prior.transform(
                lambda s: s.rolling(window, min_periods=2).std()
            )
            out[f"{safe}_trailmean{window}_000433"] = (
                roll_mean.fillna(series).astype("float32")
            )
            out[f"{safe}_trailstd{window}_000433"] = (
                roll_std.fillna(0.0).astype("float32")
            )
            out[f"{safe}_surprise{window}_000433"] = (
                (series - roll_mean).fillna(0.0).astype("float32")
            )
            out[f"{safe}_zsurprise{window}_000433"] = (
                ((series - roll_mean) / (roll_std + 1e-6))
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
                .clip(-10, 10)
                .astype("float32")
            )

        ewm3 = prior.transform(lambda s: s.ewm(span=3, adjust=False).mean())
        ewm5 = prior.transform(lambda s: s.ewm(span=5, adjust=False).mean())
        out[f"{safe}_ewm3_prior_000433"] = ewm3.fillna(series).astype("float32")
        out[f"{safe}_ewm5_prior_000433"] = ewm5.fillna(series).astype("float32")
        out[f"{safe}_ewm_gap3_000433"] = (
            (series - ewm3).fillna(0.0).astype("float32")
        )
        out[f"{safe}_ewm_gap5_000433"] = (
            (series - ewm5).fillna(0.0).astype("float32")
        )

    tyre_life = out["TyreLife"].clip(lower=0)
    out["deg_per_tyre_life_000433"] = (
        out["Cumulative_Degradation"] / (tyre_life + 1.0)
    ).astype("float32")
    out["lap_delta_per_tyre_life_000433"] = (
        out["LapTime_Delta"] / (tyre_life + 1.0)
    ).astype("float32")

    prior_delta = g["LapTime_Delta"].shift(1) - g["LapTime_Delta"].shift(2)
    out["latent_pace_loss_proxy_000433"] = (
        prior_delta.clip(lower=0)
        .groupby([out[c] for c in group_cols], sort=False)
        .transform(lambda s: s.ewm(span=4, adjust=False).mean())
        .fillna(0.0)
        .astype("float32")
    )

    out = out.sort_values("_row_order_000433", kind="mergesort").drop(
        columns=["_row_order_000433"]
    )
    return out
