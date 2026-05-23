from __future__ import annotations

import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()
    n = len(out)

    compound = out["Compound"].astype("string").str.upper().fillna("UNKNOWN")
    tyre = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(0.0).clip(0.0, 90.0)
    progress = (
        pd.to_numeric(out["RaceProgress"], errors="coerce")
        .fillna(0.5)
        .clip(0.0, 1.0)
    )
    stint = pd.to_numeric(out["Stint"], errors="coerce").fillna(1.0).clip(1.0, 6.0)
    pit = pd.to_numeric(out["PitStop"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    out["hazard_tyre_bin"] = pd.cut(
        tyre,
        bins=[-0.1, 3, 6, 9, 12, 16, 22, 30, 45, 90],
        labels=False,
    ).astype("int16")
    out["hazard_progress_bin"] = pd.cut(
        progress,
        bins=[-0.01, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.01],
        labels=False,
    ).astype("int16")
    out["hazard_stint_bin"] = stint.clip(1, 4).astype("int16")

    order_frame = pd.DataFrame(
        {
            "Year": out["Year"].to_numpy(),
            "Race": out["Race"].astype("string").to_numpy(),
            "LapNumber": out["LapNumber"].to_numpy(),
            "Driver": out["Driver"].astype("string").to_numpy(),
            "_pos": np.arange(n, dtype=np.int64),
        }
    )
    ordered_pos = order_frame.sort_values(
        ["Year", "Race", "LapNumber", "Driver", "_pos"], kind="mergesort"
    )["_pos"].to_numpy()

    pit_ord = pd.Series(pit.to_numpy(dtype=np.float64)[ordered_pos], dtype="float64")
    global_count = np.arange(n, dtype=np.float64)
    global_sum = np.cumsum(pit_ord.to_numpy()) - pit_ord.to_numpy()
    global_prior = np.divide(
        global_sum,
        global_count,
        out=np.zeros(n, dtype=np.float64),
        where=global_count > 0,
    )

    def add_ordered_hazard(col: str, smooth: float = 40.0) -> None:
        key_ord = pd.Series(
            out[col].astype("string").iloc[ordered_pos].to_numpy(), dtype="string"
        )
        grp = pit_ord.groupby(key_ord, sort=False)
        cnt = grp.cumcount().to_numpy(dtype=np.float64)
        prev_sum = (grp.cumsum() - pit_ord).to_numpy(dtype=np.float64)
        prior_ord = (prev_sum + smooth * global_prior) / (cnt + smooth)

        prior = np.empty(n, dtype=np.float32)
        hist_count = np.empty(n, dtype=np.float32)
        prior[ordered_pos] = prior_ord.astype(np.float32)
        hist_count[ordered_pos] = np.log1p(cnt).astype(np.float32)
        out[f"{col}_ordered_hazard"] = prior
        out[f"{col}_ordered_hazard_count"] = hist_count

    out["hazard_compound_stint"] = (
        compound + "|S" + out["hazard_stint_bin"].astype("string")
    ).astype("category")
    out["hazard_compound_tyre"] = (
        compound + "|T" + out["hazard_tyre_bin"].astype("string")
    ).astype("category")
    out["hazard_race_progress"] = (
        out["Race"].astype("string") + "|P" + out["hazard_progress_bin"].astype("string")
    ).astype("category")
    out["hazard_compound_stint_tyre"] = (
        compound
        + "|S"
        + out["hazard_stint_bin"].astype("string")
        + "|T"
        + out["hazard_tyre_bin"].astype("string")
    ).astype("category")

    for col in [
        "Compound",
        "hazard_compound_stint",
        "hazard_compound_tyre",
        "hazard_race_progress",
        "hazard_compound_stint_tyre",
    ]:
        add_ordered_hazard(col)

    hazard_cols = [c for c in out.columns if c.endswith("_ordered_hazard")]
    out["ordered_hazard_mean"] = out[hazard_cols].mean(axis=1).astype("float32")
    out["ordered_hazard_max"] = out[hazard_cols].max(axis=1).astype("float32")
    out["ordered_hazard_x_tyre"] = (out["ordered_hazard_mean"] * tyre).astype("float32")
    out["ordered_hazard_x_progress"] = (
        out["ordered_hazard_mean"] * progress
    ).astype("float32")

    return out
