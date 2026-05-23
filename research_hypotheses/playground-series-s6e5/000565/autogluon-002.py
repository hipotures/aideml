from __future__ import annotations

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()
    out["_row_id_000565"] = np.arange(len(out), dtype=np.int64)
    sort_cols = ["Year", "Race", "Driver", "LapNumber", "_row_id_000565"]
    ordered = out.sort_values(sort_cols, kind="mergesort").copy()
    group_cols = ["Year", "Race", "Driver"]
    grp = ordered.groupby(group_cols, sort=False)

    hist_cols = [
        "TyreLife",
        "Cumulative_Degradation",
        "LapTime_Delta",
        "Position",
        "Position_Change",
        "RaceProgress",
        "PitStop",
    ]
    for col in hist_cols:
        safe = col.replace(" ", "_").replace("(", "").replace(")", "")
        numeric = pd.to_numeric(ordered[col], errors="coerce")
        for lag in (1, 2, 3, 5):
            ordered[f"seq_{safe}_lag{lag}"] = grp[col].shift(lag)
        prev = numeric.groupby([ordered[c] for c in group_cols], sort=False).shift(1)
        ordered[f"seq_{safe}_roll3_mean"] = (
            prev.groupby([ordered[c] for c in group_cols], sort=False)
            .rolling(3, min_periods=1)
            .mean()
            .reset_index(level=[0, 1, 2], drop=True)
        )
        ordered[f"seq_{safe}_roll5_std"] = (
            prev.groupby([ordered[c] for c in group_cols], sort=False)
            .rolling(5, min_periods=2)
            .std()
            .reset_index(level=[0, 1, 2], drop=True)
        )

    ordered["seq_lap_delta_accel"] = (
        ordered["seq_LapTime_Delta_lag1"] - ordered["seq_LapTime_Delta_lag3"]
    )
    ordered["seq_degradation_momentum"] = (
        ordered["Cumulative_Degradation"]
        - ordered["seq_Cumulative_Degradation_roll3_mean"]
    )
    ordered["seq_recent_pit_count"] = (
        ordered["seq_PitStop_lag1"].fillna(0)
        + ordered["seq_PitStop_lag2"].fillna(0)
        + ordered["seq_PitStop_lag3"].fillna(0)
    ).astype("float32")

    ordered = ordered.sort_values("_row_id_000565", kind="mergesort")
    ordered = ordered.drop(columns=["_row_id_000565"])
    new_cols = [c for c in ordered.columns if c not in out.columns]
    out = out.join(ordered[new_cols])
    return out
