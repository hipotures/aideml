# Top 10 AutoGluon Codes

Source: `logs/submission_index.json` plus Kaggle/public-score registry from `scripts/kaggle_submission_lab.py --output-format json`.
Ranking: top AutoGluon records by CV, deduplicated by submission SHA so identical prediction files are not repeated.
Code excerpt: only `preprocess()` from each `solution.py`, to keep the report focused.

| # | hypothesis | CV | public | run | step | timestamp | submission sha | code sha | solution |
|---:|---|---:|---:|---|---:|---|---|---|---|
| 1 | `000459` | 0.952642035592 | - | `2-delectable-curvy-dolphin` | 314 | `20260518T154051` | `400e890e00` | `f025672a01` | `logs/2-delectable-curvy-dolphin/artifacts/20260518T154051/solution.py` |
| 2 | `000459` | 0.952622651654 | 0.95073 | `2-delectable-curvy-dolphin` | 454 | `20260519T140926` | `b26e3bd90c` | `929341ee41` | `logs/2-delectable-curvy-dolphin/artifacts/20260519T140926/solution.py` |
| 3 | `000443` | 0.952619527955 | 0.95058 | `2-delectable-curvy-dolphin` | 409 | `20260519T073718` | `1895c6f925` | `1180aa5aab` | `logs/2-delectable-curvy-dolphin/artifacts/20260519T073718/solution.py` |
| 4 | `000015` | 0.952617822344 | - | `2-delectable-curvy-dolphin` | 403 | `20260519T063613` | `24f69e7eaa` | `7fb4c85886` | `logs/2-delectable-curvy-dolphin/artifacts/20260519T063613/solution.py` |
| 5 | `000979` | 0.952614874329 | 0.95063 | `2-delectable-curvy-dolphin` | 727 | `20260521T123053` | `b59a1b8f86` | `c4b8d2788d` | `logs/2-delectable-curvy-dolphin/artifacts/20260521T123053/solution.py` |
| 6 | `000017` | 0.952612763260 | 0.95070 | `2-delectable-curvy-dolphin` | 726 | `20260521T121925` | `e0fdb4d7e3` | `1ce7e2975e` | `logs/2-delectable-curvy-dolphin/artifacts/20260521T121925/solution.py` |
| 7 | `000904` | 0.952597044718 | 0.95067 | `2-delectable-curvy-dolphin` | 430 | `20260519T111046` | `8052cadeb7` | `8be0c1ee85` | `logs/2-delectable-curvy-dolphin/artifacts/20260519T111046/solution.py` |
| 8 | `000316` | 0.952593343941 | - | `2-delectable-curvy-dolphin` | 418 | `20260519T090536` | `a868e30f7b` | `948679c85d` | `logs/2-delectable-curvy-dolphin/artifacts/20260519T090536/solution.py` |
| 9 | `000257` | 0.952566795543 | - | `2-delectable-curvy-dolphin` | 390 | `20260519T042240` | `99ce4a444e` | `06117aec2d` | `logs/2-delectable-curvy-dolphin/artifacts/20260519T042240/solution.py` |
| 10 | `000052` | 0.952566612538 | 0.95051 | `2-delectable-curvy-dolphin` | 320 | `20260518T164418` | `eae711286c` | `91a2fd2fa0` | `logs/2-delectable-curvy-dolphin/artifacts/20260518T164418/solution.py` |

## 1. `000459` - CV 0.952642035592, public -

- run: `2-delectable-curvy-dolphin`
- step: `314`
- timestamp: `20260518T154051`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260518T154051/solution.py`
- submission sha: `400e890e002897bc6c25bc204e823b101da9382da753abba0a1b85abd5095a59`
- code sha: `f025672a01d8f1845e8dffe9741f79fddf3692b5a54b296da3e2ed2c1b9e71a6`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )

    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    if pd.isna(race_median_lap).any():
        race_median_lap = race_median_lap.fillna(
            out.groupby("Race")["LapTime (s)"].transform("median")
        )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask],
        [0.18, 0.28, 0.40],
        default=0.34,
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )

    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt

    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    eps = 1e-3

    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    if pd.isna(race_year_median_delta).any():
        race_year_median_delta = race_year_median_delta.fillna(
            out.groupby("Race")["LapTime_Delta"].transform("median")
        )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    if pd.isna(race_year_median_deg).any():
        race_year_median_deg = race_year_median_deg.fillna(
            out.groupby("Race")["Cumulative_Degradation"].transform("median")
        )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    if pd.isna(race_year_median_tyre_life).any():
        race_year_median_tyre_life = race_year_median_tyre_life.fillna(
            out.groupby("Race")["TyreLife"].transform("median")
        )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    if pd.isna(race_year_median_wear_pressure).any():
        race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
            pd.Series(wear_pressure, index=out.index)
            .groupby(out["Race"])
            .transform("median")
        )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    if pd.isna(race_year_median_stop_adv).any():
        race_year_median_stop_adv = race_year_median_stop_adv.fillna(
            out.groupby("Race")["best_stop_now_advantage"].transform("median")
        )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = lap_delta_pos / (
        race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / (
        race_year_median_deg.abs().clip(lower=eps)
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / (
        race_year_median_tyre_life.clip(lower=1.0)
    )
    out["wear_pressure_vs_race_year_median"] = wear_pressure / (
        race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = out["best_stop_now_advantage"] - (
        race_year_median_stop_adv
    )

    # Hypothesis 000459: regularize sparse identifiers so downstream models defer
    # brittle ID memorization and use dense race-state signals first.
    for col, rare_thresh in [("Driver", 8), ("Race_Year", 12), ("Race", 12)]:
        counts = out[col].value_counts(dropna=False)
        freq = out[col].map(counts).astype("float32")
        out[f"{col}_freq"] = freq
        out[f"{col}_log_freq"] = np.log1p(freq).astype("float32")
        out[f"{col}_is_rare"] = (freq <= rare_thresh).astype("int8")
        out[col] = out[col].where(freq > rare_thresh, f"RARE_{col}")

    driver_raceyear = (
        out["Driver"].astype("string") + "__" + out["Race_Year"].astype("string")
    )
    driver_raceyear_counts = driver_raceyear.value_counts(dropna=False)
    driver_raceyear_freq = driver_raceyear.map(driver_raceyear_counts).astype("float32")
    out["Driver_RaceYear_freq"] = driver_raceyear_freq
    out["Driver_RaceYear_log_freq"] = np.log1p(driver_raceyear_freq).astype("float32")
    out["Driver_RaceYear_is_rare"] = (driver_raceyear_freq <= 3).astype("int8")

    return out
```

## 2. `000459` - CV 0.952622651654, public 0.95073

- run: `2-delectable-curvy-dolphin`
- step: `454`
- timestamp: `20260519T140926`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260519T140926/solution.py`
- submission sha: `b26e3bd90cce8f9029c5a2be13b325ed73b3022a6446e76f91f376f2f9e6e053`
- code sha: `929341ee419db7fd766f1ca45f3f7047b00f6282bdb565290661e0c3976fa83b`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )

    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    if pd.isna(race_median_lap).any():
        race_median_lap = race_median_lap.fillna(
            out.groupby("Race")["LapTime (s)"].transform("median")
        )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask],
        [0.18, 0.28, 0.40],
        default=0.34,
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )

    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt

    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    eps = 1e-3

    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    if pd.isna(race_year_median_delta).any():
        race_year_median_delta = race_year_median_delta.fillna(
            out.groupby("Race")["LapTime_Delta"].transform("median")
        )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    if pd.isna(race_year_median_deg).any():
        race_year_median_deg = race_year_median_deg.fillna(
            out.groupby("Race")["Cumulative_Degradation"].transform("median")
        )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    if pd.isna(race_year_median_tyre_life).any():
        race_year_median_tyre_life = race_year_median_tyre_life.fillna(
            out.groupby("Race")["TyreLife"].transform("median")
        )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    if pd.isna(race_year_median_wear_pressure).any():
        race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
            pd.Series(wear_pressure, index=out.index)
            .groupby(out["Race"])
            .transform("median")
        )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    if pd.isna(race_year_median_stop_adv).any():
        race_year_median_stop_adv = race_year_median_stop_adv.fillna(
            out.groupby("Race")["best_stop_now_advantage"].transform("median")
        )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = lap_delta_pos / (
        race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / (
        race_year_median_deg.abs().clip(lower=eps)
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / (
        race_year_median_tyre_life.clip(lower=1.0)
    )
    out["wear_pressure_vs_race_year_median"] = wear_pressure / (
        race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = out["best_stop_now_advantage"] - (
        race_year_median_stop_adv
    )

    service_window_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 14.0,
                "MEDIUM": 20.0,
                "HARD": 26.0,
                "INTERMEDIATE": 18.0,
                "WET": 16.0,
            }
        )
        .astype("float32")
        .fillna(18.0)
    )
    service_window_template = np.minimum(
        service_window_template, conservative_compound_life - 2.0
    ).clip(lower=6.0)

    service_age_distance = tyre_life - service_window_template
    service_window_pressure = (
        service_age_distance / service_window_template.clip(lower=1.0)
    ).clip(lower=-2.0, upper=3.0)

    tyre_age_at_finish_est = current_tyre_age_to_finish
    finish_stop_pressure = (
        (-current_tyre_finish_margin_signed)
        / conservative_compound_life.clip(lower=1.0)
    ).clip(lower=0.0, upper=3.0)
    cannot_finish_on_current_tyre = current_tyre_cannot_finish

    laps_since_last_pit = tyre_life
    freshness_suppression = 1.0 / (1.0 + laps_since_last_pit)
    just_pitted_fresh_tyre = (laps_since_last_pit <= 2).astype("int8")

    out["Service_Age_Distance"] = service_age_distance
    out["Service_Window_Pressure"] = service_window_pressure
    out["Service_Window_Overage"] = service_age_distance.clip(lower=0.0)
    out["TyreAge_At_Finish_Est"] = tyre_age_at_finish_est
    out["Finish_Stop_Pressure"] = finish_stop_pressure
    out["Cannot_Finish_On_Current_Tyre"] = cannot_finish_on_current_tyre
    out["Laps_Since_Last_Pit"] = laps_since_last_pit
    out["Fresh_Tyre_Suppression"] = freshness_suppression
    out["Just_Pitted_Fresh_Tyre"] = just_pitted_fresh_tyre
    out["Monotone_Pit_Urgency_Proxy"] = (
        0.40 * service_window_pressure.clip(lower=0.0)
        + 0.85 * finish_stop_pressure
        + 0.25 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0.0)
        + 0.20 * out["remaining_dry_compound_debt"]
        - 0.75 * freshness_suppression
        - 0.35 * just_pitted_fresh_tyre
    )

    driver_freq = out.groupby("Driver")["Driver"].transform("size").astype("float32")
    race_year_freq = (
        out.groupby("Race_Year")["Race_Year"].transform("size").astype("float32")
    )
    driver_race_year_freq = (
        out.groupby(["Driver", "Race_Year"])["Driver"]
        .transform("size")
        .astype("float32")
    )

    out["Driver_DataSupport"] = driver_freq
    out["RaceYear_DataSupport"] = race_year_freq
    out["DriverRaceYear_DataSupport"] = driver_race_year_freq
    out["Driver_Is_Rare"] = (driver_freq <= 4).astype("int8")
    out["RaceYear_Is_Rare"] = (race_year_freq <= 8).astype("int8")
    out["DriverRaceYear_Is_Rare"] = (driver_race_year_freq <= 2).astype("int8")

    out["Driver"] = out["Driver"].where(driver_freq >= 5, "RARE_DRIVER")
    out["Race_Year"] = out["Race_Year"].where(race_year_freq >= 10, "RARE_RACE_YEAR")
    out["Driver_Race_Year_Group"] = (
        out["Driver"].astype("string") + "__" + out["Race_Year"].astype("string")
    ).where(driver_race_year_freq >= 3, "RARE_DRIVER_RACE_YEAR")

    return out
```

## 3. `000443` - CV 0.952619527955, public 0.95058

- run: `2-delectable-curvy-dolphin`
- step: `409`
- timestamp: `20260519T073718`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260519T073718/solution.py`
- submission sha: `1895c6f925af8758931b0421c75ba869d700a18a16176425abd86425f4abef4b`
- code sha: `1180aa5aabc4e2333b9f37c1487a53fb6ec162092cb0845366aeb9c8f71f35af`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )

    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    if pd.isna(race_median_lap).any():
        race_median_lap = race_median_lap.fillna(
            out.groupby("Race")["LapTime (s)"].transform("median")
        )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask],
        [0.18, 0.28, 0.40],
        default=0.34,
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )

    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt

    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    eps = 1e-3

    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    if pd.isna(race_year_median_delta).any():
        race_year_median_delta = race_year_median_delta.fillna(
            out.groupby("Race")["LapTime_Delta"].transform("median")
        )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    if pd.isna(race_year_median_deg).any():
        race_year_median_deg = race_year_median_deg.fillna(
            out.groupby("Race")["Cumulative_Degradation"].transform("median")
        )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    if pd.isna(race_year_median_tyre_life).any():
        race_year_median_tyre_life = race_year_median_tyre_life.fillna(
            out.groupby("Race")["TyreLife"].transform("median")
        )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    if pd.isna(race_year_median_wear_pressure).any():
        race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
            pd.Series(wear_pressure, index=out.index)
            .groupby(out["Race"])
            .transform("median")
        )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    if pd.isna(race_year_median_stop_adv).any():
        race_year_median_stop_adv = race_year_median_stop_adv.fillna(
            out.groupby("Race")["best_stop_now_advantage"].transform("median")
        )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = lap_delta_pos / (
        race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / (
        race_year_median_deg.abs().clip(lower=eps)
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / (
        race_year_median_tyre_life.clip(lower=1.0)
    )
    out["wear_pressure_vs_race_year_median"] = wear_pressure / (
        race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = out["best_stop_now_advantage"] - (
        race_year_median_stop_adv
    )

    remaining_after_1 = (remaining_laps_est - 1.0).clip(lower=0)
    soft_margin_now = 17.0 - remaining_laps_est
    medium_margin_now = 24.0 - remaining_laps_est
    hard_margin_now = 31.0 - remaining_laps_est

    soft_margin_next = 17.0 - remaining_after_1
    medium_margin_next = 24.0 - remaining_after_1
    hard_margin_next = 31.0 - remaining_after_1

    soft_newly_viable = (
        (soft_margin_now < 0) & (soft_margin_next >= 0) & dry_mask
    ).astype("int8")
    medium_newly_viable = (
        (medium_margin_now < 0) & (medium_margin_next >= 0) & dry_mask
    ).astype("int8")
    hard_newly_viable = (
        (hard_margin_now < 0) & (hard_margin_next >= 0) & dry_mask
    ).astype("int8")

    out["fresh_soft_newly_viable_next_lap"] = soft_newly_viable
    out["fresh_medium_newly_viable_next_lap"] = medium_newly_viable
    out["fresh_hard_newly_viable_next_lap"] = hard_newly_viable

    newly_viable_count = soft_newly_viable + medium_newly_viable + hard_newly_viable
    out["fresh_slick_newly_viable_next_lap_any"] = (newly_viable_count > 0).astype(
        "int8"
    )
    out["fresh_slick_newly_viable_next_lap_count"] = newly_viable_count.astype("int8")

    out["softest_newly_viable_slick"] = np.select(
        [soft_newly_viable == 1, medium_newly_viable == 1, hard_newly_viable == 1],
        ["SOFT", "MEDIUM", "HARD"],
        default="NONE",
    ).astype(object)

    boundary_margins_now = np.column_stack(
        [
            np.asarray(soft_margin_now, dtype="float32"),
            np.asarray(medium_margin_now, dtype="float32"),
            np.asarray(hard_margin_now, dtype="float32"),
        ]
    )
    nearest_idx = np.abs(boundary_margins_now).argmin(axis=1)
    out["fresh_slick_nearest_final_window_margin"] = boundary_margins_now[
        np.arange(len(out)), nearest_idx
    ]

    # Hypothesis 000443: preprocess-only adversarial-validation proxy using
    # raw covariate rarity across the expected shift axes.
    driver_race = (out["Driver"] + "_" + out["Race"]).astype("string")
    driver_year = (out["Driver"] + "_" + out["Year_str"]).astype("string")
    compound_race_year = (out["Compound"] + "_" + out["Race_Year"]).astype("string")

    race_year_count = out["Race_Year"].value_counts(dropna=False)
    driver_count = out["Driver"].value_counts(dropna=False)
    race_count = out["Race"].value_counts(dropna=False)
    driver_race_count = driver_race.value_counts(dropna=False)
    driver_year_count = driver_year.value_counts(dropna=False)
    compound_race_year_count = compound_race_year.value_counts(dropna=False)

    out["race_year_frequency"] = (
        out["Race_Year"].map(race_year_count).astype("float32").fillna(1.0)
    )
    out["driver_frequency"] = (
        out["Driver"].map(driver_count).astype("float32").fillna(1.0)
    )
    out["race_frequency"] = out["Race"].map(race_count).astype("float32").fillna(1.0)
    out["driver_race_frequency"] = (
        driver_race.map(driver_race_count).astype("float32").fillna(1.0)
    )
    out["driver_year_frequency"] = (
        driver_year.map(driver_year_count).astype("float32").fillna(1.0)
    )
    out["compound_race_year_frequency"] = (
        compound_race_year.map(compound_race_year_count).astype("float32").fillna(1.0)
    )

    race_year_rarity = 1.0 / np.sqrt(out["race_year_frequency"].clip(lower=1.0))
    driver_rarity = 1.0 / np.sqrt(out["driver_frequency"].clip(lower=1.0))
    race_rarity = 1.0 / np.sqrt(out["race_frequency"].clip(lower=1.0))
    driver_race_rarity = 1.0 / np.sqrt(out["driver_race_frequency"].clip(lower=1.0))
    driver_year_rarity = 1.0 / np.sqrt(out["driver_year_frequency"].clip(lower=1.0))
    compound_race_year_rarity = 1.0 / np.sqrt(
        out["compound_race_year_frequency"].clip(lower=1.0)
    )

    out["race_year_rarity"] = race_year_rarity.astype("float32")
    out["driver_race_rarity"] = driver_race_rarity.astype("float32")
    out["compound_race_year_rarity"] = compound_race_year_rarity.astype("float32")

    out["deployment_likeness_score"] = (
        0.24 * race_year_rarity
        + 0.12 * race_rarity
        + 0.16 * driver_rarity
        + 0.22 * driver_race_rarity
        + 0.14 * driver_year_rarity
        + 0.12 * compound_race_year_rarity
    ).astype("float32")

    out["deployment_likeness_x_testing"] = (
        out["deployment_likeness_score"] * out["is_testing"]
    ).astype("float32")
    out["deployment_likeness_x_wet"] = (
        out["deployment_likeness_score"] * out["is_wet_compound"]
    ).astype("float32")
    out["deployment_likeness_x_remaining_laps"] = (
        out["deployment_likeness_score"] * remaining_laps_est
    ).astype("float32")
    out["deployment_likeness_x_stop_advantage"] = (
        out["deployment_likeness_score"] * out["best_stop_now_advantage"]
    ).astype("float32")
    out["high_shift_context"] = (
        out["deployment_likeness_score"] >= out["deployment_likeness_score"].median()
    ).astype("int8")

    return out
```

## 4. `000015` - CV 0.952617822344, public -

- run: `2-delectable-curvy-dolphin`
- step: `403`
- timestamp: `20260519T063613`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260519T063613/solution.py`
- submission sha: `24f69e7eaafda684c6ce210dec50e2daa82c688b7e4cbfb68608203dcb411f8f`
- code sha: `7fb4c858864ac6c81bddff42fdeec691ba3793ac7ece19d82028e9c77101506a`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )

    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    if pd.isna(race_median_lap).any():
        race_median_lap = race_median_lap.fillna(
            out.groupby("Race")["LapTime (s)"].transform("median")
        )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask],
        [0.18, 0.28, 0.40],
        default=0.34,
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )

    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt

    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    eps = 1e-3

    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    if pd.isna(race_year_median_delta).any():
        race_year_median_delta = race_year_median_delta.fillna(
            out.groupby("Race")["LapTime_Delta"].transform("median")
        )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    if pd.isna(race_year_median_deg).any():
        race_year_median_deg = race_year_median_deg.fillna(
            out.groupby("Race")["Cumulative_Degradation"].transform("median")
        )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    if pd.isna(race_year_median_tyre_life).any():
        race_year_median_tyre_life = race_year_median_tyre_life.fillna(
            out.groupby("Race")["TyreLife"].transform("median")
        )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    if pd.isna(race_year_median_wear_pressure).any():
        race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
            pd.Series(wear_pressure, index=out.index)
            .groupby(out["Race"])
            .transform("median")
        )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    if pd.isna(race_year_median_stop_adv).any():
        race_year_median_stop_adv = race_year_median_stop_adv.fillna(
            out.groupby("Race")["best_stop_now_advantage"].transform("median")
        )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = lap_delta_pos / (
        race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / (
        race_year_median_deg.abs().clip(lower=eps)
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / (
        race_year_median_tyre_life.clip(lower=1.0)
    )
    out["wear_pressure_vs_race_year_median"] = wear_pressure / (
        race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = out["best_stop_now_advantage"] - (
        race_year_median_stop_adv
    )

    # Hypothesis 000015: detect the exact one-lap entry into the final-stop window
    # for fresh slick compounds that become feasible after waiting one more lap.
    remaining_after_1 = (remaining_laps_est - 1.0).clip(lower=0)
    soft_margin_now = 17.0 - remaining_laps_est
    medium_margin_now = 24.0 - remaining_laps_est
    hard_margin_now = 31.0 - remaining_laps_est

    soft_margin_next = 17.0 - remaining_after_1
    medium_margin_next = 24.0 - remaining_after_1
    hard_margin_next = 31.0 - remaining_after_1

    soft_newly_viable = (
        (soft_margin_now < 0) & (soft_margin_next >= 0) & dry_mask
    ).astype("int8")
    medium_newly_viable = (
        (medium_margin_now < 0) & (medium_margin_next >= 0) & dry_mask
    ).astype("int8")
    hard_newly_viable = (
        (hard_margin_now < 0) & (hard_margin_next >= 0) & dry_mask
    ).astype("int8")

    out["fresh_soft_newly_viable_next_lap"] = soft_newly_viable
    out["fresh_medium_newly_viable_next_lap"] = medium_newly_viable
    out["fresh_hard_newly_viable_next_lap"] = hard_newly_viable

    newly_viable_count = soft_newly_viable + medium_newly_viable + hard_newly_viable
    out["fresh_slick_newly_viable_next_lap_any"] = (newly_viable_count > 0).astype(
        "int8"
    )
    out["fresh_slick_newly_viable_next_lap_count"] = newly_viable_count.astype("int8")

    out["softest_newly_viable_slick"] = np.select(
        [soft_newly_viable == 1, medium_newly_viable == 1, hard_newly_viable == 1],
        ["SOFT", "MEDIUM", "HARD"],
        default="NONE",
    ).astype(object)

    boundary_margins_now = np.column_stack(
        [
            np.asarray(soft_margin_now, dtype="float32"),
            np.asarray(medium_margin_now, dtype="float32"),
            np.asarray(hard_margin_now, dtype="float32"),
        ]
    )
    nearest_idx = np.abs(boundary_margins_now).argmin(axis=1)
    out["fresh_slick_nearest_final_window_margin"] = boundary_margins_now[
        np.arange(len(out)), nearest_idx
    ]

    return out
```

## 5. `000979` - CV 0.952614874329, public 0.95063

- run: `2-delectable-curvy-dolphin`
- step: `727`
- timestamp: `20260521T123053`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260521T123053/solution.py`
- submission sha: `b59a1b8f8631baaddc351d51ef059acfa435fabf99aee5763df6d18d9f96849c`
- code sha: `c4b8d2788dedd55954eb014344314d00a68b78974b0bc57986c524d44fb2f4e3`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )
    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    race_median_lap = race_median_lap.fillna(
        out.groupby("Race")["LapTime (s)"].transform("median")
    )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask], [0.18, 0.28, 0.40], default=0.34
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )
    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt
    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    eps = 1e-3

    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    race_year_median_delta = race_year_median_delta.fillna(
        out.groupby("Race")["LapTime_Delta"].transform("median")
    )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    race_year_median_deg = race_year_median_deg.fillna(
        out.groupby("Race")["Cumulative_Degradation"].transform("median")
    )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(
        out.groupby("Race")["TyreLife"].transform("median")
    )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race"])
        .transform("median")
    )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out.groupby("Race")["best_stop_now_advantage"].transform("median")
    )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = (
        lap_delta_pos / race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / race_year_median_deg.abs().clip(
        lower=eps
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / race_year_median_tyre_life.clip(
        lower=1.0
    )
    out["wear_pressure_vs_race_year_median"] = (
        wear_pressure / race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = (
        out["best_stop_now_advantage"] - race_year_median_stop_adv
    )

    service_window_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 14.0,
                "MEDIUM": 20.0,
                "HARD": 26.0,
                "INTERMEDIATE": 18.0,
                "WET": 16.0,
            }
        )
        .astype("float32")
        .fillna(18.0)
    )
    service_window_template = np.minimum(
        service_window_template, conservative_compound_life - 2.0
    ).clip(lower=6.0)

    service_age_distance = tyre_life - service_window_template
    service_window_pressure = (
        service_age_distance / service_window_template.clip(lower=1.0)
    ).clip(lower=-2.0, upper=3.0)

    tyre_age_at_finish_est = current_tyre_age_to_finish
    finish_stop_pressure = (
        (-current_tyre_finish_margin_signed)
        / conservative_compound_life.clip(lower=1.0)
    ).clip(lower=0.0, upper=3.0)
    cannot_finish_on_current_tyre = current_tyre_cannot_finish

    laps_since_last_pit = tyre_life
    freshness_suppression = 1.0 / (1.0 + laps_since_last_pit)
    just_pitted_fresh_tyre = (laps_since_last_pit <= 2).astype("int8")

    out["Service_Age_Distance"] = service_age_distance
    out["Service_Window_Pressure"] = service_window_pressure
    out["Service_Window_Overage"] = service_age_distance.clip(lower=0.0)
    out["TyreAge_At_Finish_Est"] = tyre_age_at_finish_est
    out["Finish_Stop_Pressure"] = finish_stop_pressure
    out["Cannot_Finish_On_Current_Tyre"] = cannot_finish_on_current_tyre
    out["Laps_Since_Last_Pit"] = laps_since_last_pit
    out["Fresh_Tyre_Suppression"] = freshness_suppression
    out["Just_Pitted_Fresh_Tyre"] = just_pitted_fresh_tyre
    out["Monotone_Pit_Urgency_Proxy"] = (
        0.40 * service_window_pressure.clip(lower=0.0)
        + 0.85 * finish_stop_pressure
        + 0.25 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0.0)
        + 0.20 * out["remaining_dry_compound_debt"]
        - 0.75 * freshness_suppression
        - 0.35 * just_pitted_fresh_tyre
    )

    pit_window_gate = (
        (service_window_pressure >= -0.35)
        & (service_window_pressure <= 1.25)
        & (just_pitted_fresh_tyre == 0)
        & (remaining_laps_est > 0)
    ).astype("int8")
    late_stint_gate = (
        ((service_age_distance >= 0) | (finish_stop_pressure > 0))
        & (just_pitted_fresh_tyre == 0)
        & (remaining_laps_est > 0)
    ).astype("int8")
    early_race_gate = (race_progress < 0.35).astype("int8")

    out["Expert_Dry_Gate_000979"] = dry_mask.astype("int8")
    out["Expert_WetInter_Gate_000979"] = wet_mask.astype("int8")
    out["Expert_Testing_Gate_000979"] = test_mask.astype("int8")
    out["Expert_PitWindow_Gate_000979"] = pit_window_gate
    out["Expert_LateStint_Gate_000979"] = late_stint_gate
    out["Expert_EarlyRace_Gate_000979"] = early_race_gate
    out["Expert_Route_000979"] = np.select(
        [
            test_mask,
            wet_mask & late_stint_gate.astype(bool),
            wet_mask,
            dry_mask & late_stint_gate.astype(bool),
            dry_mask & pit_window_gate.astype(bool),
            dry_mask & early_race_gate.astype(bool),
            dry_mask,
        ],
        [
            "testing",
            "wet_late_stint",
            "wet_general",
            "dry_late_stint",
            "dry_pit_window",
            "dry_early",
            "dry_general",
        ],
        default="other",
    ).astype(object)
    out["Dry_Expert_Urgency_000979"] = (
        out["Monotone_Pit_Urgency_Proxy"] * out["Expert_Dry_Gate_000979"]
    )
    out["WetInter_Expert_Urgency_000979"] = (
        out["Monotone_Pit_Urgency_Proxy"] * out["Expert_WetInter_Gate_000979"]
    )
    out["LateStint_Expert_Urgency_000979"] = (
        out["Monotone_Pit_Urgency_Proxy"] * late_stint_gate
    )
    out["PitWindow_Expert_StopAdv_000979"] = (
        out["best_stop_now_advantage"] * pit_window_gate
    )
    out["DryPitWindow_Expert_Interaction_000979"] = (
        out["Expert_Dry_Gate_000979"] * pit_window_gate
    )
    out["WetLateStint_Expert_Interaction_000979"] = (
        out["Expert_WetInter_Gate_000979"] * late_stint_gate
    )

    driver_freq = out.groupby("Driver")["Driver"].transform("size").astype("float32")
    race_year_freq = (
        out.groupby("Race_Year")["Race_Year"].transform("size").astype("float32")
    )
    driver_race_year_freq = (
        out.groupby(["Driver", "Race_Year"])["Driver"]
        .transform("size")
        .astype("float32")
    )

    out["Driver_DataSupport"] = driver_freq
    out["RaceYear_DataSupport"] = race_year_freq
    out["DriverRaceYear_DataSupport"] = driver_race_year_freq
    out["Driver_Is_Rare"] = (driver_freq <= 4).astype("int8")
    out["RaceYear_Is_Rare"] = (race_year_freq <= 8).astype("int8")
    out["DriverRaceYear_Is_Rare"] = (driver_race_year_freq <= 2).astype("int8")

    out["Driver"] = out["Driver"].where(driver_freq >= 5, "RARE_DRIVER")
    out["Race_Year"] = out["Race_Year"].where(race_year_freq >= 10, "RARE_RACE_YEAR")
    out["Driver_Race_Year_Group"] = (
        out["Driver"].astype("string") + "__" + out["Race_Year"].astype("string")
    ).where(driver_race_year_freq >= 3, "RARE_DRIVER_RACE_YEAR")

    return out
```

## 6. `000017` - CV 0.952612763260, public 0.95070

- run: `2-delectable-curvy-dolphin`
- step: `726`
- timestamp: `20260521T121925`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260521T121925/solution.py`
- submission sha: `e0fdb4d7e31c0bc93e3af7af0c2ecd71bfed951d8ba04d8ac06972bb98471c2b`
- code sha: `1ce7e2975e42f84d538e4492b2776062b2f0082d2ce3a9e01e378d334637b779`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )
    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    race_median_lap = race_median_lap.fillna(
        out.groupby("Race")["LapTime (s)"].transform("median")
    )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask], [0.18, 0.28, 0.40], default=0.34
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )
    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt
    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    laps_after_next_lap_stop = (remaining_laps_est - 1.0).clip(lower=0.0)
    best_fresh_slick_life = np.float32(31.0)
    best_fresh_slick_finish_margin = (
        best_fresh_slick_life - laps_after_next_lap_stop
    ).clip(lower=-20.0, upper=45.0)
    fresh_slick_can_finish_next_lap = (
        dry_mask & (remaining_laps_est > 0) & (best_fresh_slick_finish_margin >= 0)
    ).astype("int8")
    current_fails_fresh_slick_can_finish = (
        (current_tyre_cannot_finish == 1) & (fresh_slick_can_finish_next_lap == 1)
    ).astype("int8")
    current_pressure_pos = current_tyre_finish_pressure.clip(lower=0.0)
    fresh_margin_pos = best_fresh_slick_finish_margin.clip(lower=0.0)
    late_stop_solve_window = (
        (race_progress >= 0.70) | (remaining_laps_est <= 20.0)
    ).astype("int8")

    out["laps_after_next_lap_stop_est"] = laps_after_next_lap_stop
    out["best_fresh_slick_finish_margin"] = best_fresh_slick_finish_margin
    out["fresh_slick_can_finish_next_lap"] = fresh_slick_can_finish_next_lap
    out["current_window_fails_and_fresh_slick_can_finish"] = (
        current_fails_fresh_slick_can_finish
    )
    out["current_pressure_x_fresh_slick_margin"] = (
        current_pressure_pos * fresh_margin_pos
    )
    out["current_fail_depth_x_fresh_slick_margin"] = (
        -current_tyre_finish_margin_signed
    ).clip(lower=0.0) * fresh_margin_pos
    out["late_current_fails_and_fresh_slick_can_finish"] = (
        current_fails_fresh_slick_can_finish * late_stop_solve_window
    ).astype("int8")
    out["late_pressure_x_fresh_slick_margin"] = (
        out["current_pressure_x_fresh_slick_margin"] * late_stop_solve_window
    )

    eps = 1e-3
    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    race_year_median_delta = race_year_median_delta.fillna(
        out.groupby("Race")["LapTime_Delta"].transform("median")
    )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    race_year_median_deg = race_year_median_deg.fillna(
        out.groupby("Race")["Cumulative_Degradation"].transform("median")
    )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(
        out.groupby("Race")["TyreLife"].transform("median")
    )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race"])
        .transform("median")
    )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out.groupby("Race")["best_stop_now_advantage"].transform("median")
    )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = (
        lap_delta_pos / race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / race_year_median_deg.abs().clip(
        lower=eps
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / race_year_median_tyre_life.clip(
        lower=1.0
    )
    out["wear_pressure_vs_race_year_median"] = (
        wear_pressure / race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = (
        out["best_stop_now_advantage"] - race_year_median_stop_adv
    )

    service_window_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 14.0,
                "MEDIUM": 20.0,
                "HARD": 26.0,
                "INTERMEDIATE": 18.0,
                "WET": 16.0,
            }
        )
        .astype("float32")
        .fillna(18.0)
    )
    service_window_template = np.minimum(
        service_window_template, conservative_compound_life - 2.0
    ).clip(lower=6.0)

    service_age_distance = tyre_life - service_window_template
    service_window_pressure = (
        service_age_distance / service_window_template.clip(lower=1.0)
    ).clip(lower=-2.0, upper=3.0)
    finish_stop_pressure = (
        (-current_tyre_finish_margin_signed)
        / conservative_compound_life.clip(lower=1.0)
    ).clip(lower=0.0, upper=3.0)

    laps_since_last_pit = tyre_life
    freshness_suppression = 1.0 / (1.0 + laps_since_last_pit)
    just_pitted_fresh_tyre = (laps_since_last_pit <= 2).astype("int8")

    out["Service_Age_Distance"] = service_age_distance
    out["Service_Window_Pressure"] = service_window_pressure
    out["Service_Window_Overage"] = service_age_distance.clip(lower=0.0)
    out["TyreAge_At_Finish_Est"] = current_tyre_age_to_finish
    out["Finish_Stop_Pressure"] = finish_stop_pressure
    out["Cannot_Finish_On_Current_Tyre"] = current_tyre_cannot_finish
    out["Laps_Since_Last_Pit"] = laps_since_last_pit
    out["Fresh_Tyre_Suppression"] = freshness_suppression
    out["Just_Pitted_Fresh_Tyre"] = just_pitted_fresh_tyre
    out["Monotone_Pit_Urgency_Proxy"] = (
        0.40 * service_window_pressure.clip(lower=0.0)
        + 0.85 * finish_stop_pressure
        + 0.25 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0.0)
        + 0.20 * out["remaining_dry_compound_debt"]
        - 0.75 * freshness_suppression
        - 0.35 * just_pitted_fresh_tyre
    )

    driver_freq = out.groupby("Driver")["Driver"].transform("size").astype("float32")
    race_year_freq = (
        out.groupby("Race_Year")["Race_Year"].transform("size").astype("float32")
    )
    driver_race_year_freq = (
        out.groupby(["Driver", "Race_Year"])["Driver"]
        .transform("size")
        .astype("float32")
    )

    out["Driver_DataSupport"] = driver_freq
    out["RaceYear_DataSupport"] = race_year_freq
    out["DriverRaceYear_DataSupport"] = driver_race_year_freq
    out["Driver_Is_Rare"] = (driver_freq <= 4).astype("int8")
    out["RaceYear_Is_Rare"] = (race_year_freq <= 8).astype("int8")
    out["DriverRaceYear_Is_Rare"] = (driver_race_year_freq <= 2).astype("int8")

    out["Driver"] = out["Driver"].where(driver_freq >= 5, "RARE_DRIVER")
    out["Race_Year"] = out["Race_Year"].where(race_year_freq >= 10, "RARE_RACE_YEAR")
    out["Driver_Race_Year_Group"] = (
        out["Driver"].astype("string") + "__" + out["Race_Year"].astype("string")
    ).where(driver_race_year_freq >= 3, "RARE_DRIVER_RACE_YEAR")

    return out
```

## 7. `000904` - CV 0.952597044718, public 0.95067

- run: `2-delectable-curvy-dolphin`
- step: `430`
- timestamp: `20260519T111046`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260519T111046/solution.py`
- submission sha: `8052cadeb74579dc9e0925de3369f7274fbf942a31989d2da9dd1f312bd71f2d`
- code sha: `8be0c1ee8591aed8f6bfe017d8624d732b981f7d30ed6f11157ffca28ba5deb4`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )

    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    if pd.isna(race_median_lap).any():
        race_median_lap = race_median_lap.fillna(
            out.groupby("Race")["LapTime (s)"].transform("median")
        )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask],
        [0.18, 0.28, 0.40],
        default=0.34,
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )

    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt

    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    eps = 1e-3

    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    if pd.isna(race_year_median_delta).any():
        race_year_median_delta = race_year_median_delta.fillna(
            out.groupby("Race")["LapTime_Delta"].transform("median")
        )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    if pd.isna(race_year_median_deg).any():
        race_year_median_deg = race_year_median_deg.fillna(
            out.groupby("Race")["Cumulative_Degradation"].transform("median")
        )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    if pd.isna(race_year_median_tyre_life).any():
        race_year_median_tyre_life = race_year_median_tyre_life.fillna(
            out.groupby("Race")["TyreLife"].transform("median")
        )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    if pd.isna(race_year_median_wear_pressure).any():
        race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
            pd.Series(wear_pressure, index=out.index)
            .groupby(out["Race"])
            .transform("median")
        )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    if pd.isna(race_year_median_stop_adv).any():
        race_year_median_stop_adv = race_year_median_stop_adv.fillna(
            out.groupby("Race")["best_stop_now_advantage"].transform("median")
        )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = lap_delta_pos / (
        race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / (
        race_year_median_deg.abs().clip(lower=eps)
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / (
        race_year_median_tyre_life.clip(lower=1.0)
    )
    out["wear_pressure_vs_race_year_median"] = wear_pressure / (
        race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = out["best_stop_now_advantage"] - (
        race_year_median_stop_adv
    )

    # Hypothesis 000904: treat Pre-Season Testing as a separate domain by
    # normalizing key signals within Year x testing-domain rather than pooling
    # them with race laps.
    out["testing_domain"] = out["is_testing"]
    year_domain_key = out["Year"].astype(str) + "_" + out["is_testing"].astype(str)

    year_median_lap_delta = out.groupby("Year")["LapTime_Delta"].transform("median")
    year_median_lap_delta = year_median_lap_delta.fillna(lap_delta_pos.median())

    year_median_lap_time = out.groupby("Year")["LapTime (s)"].transform("median")
    year_median_lap_time = year_median_lap_time.fillna(lap_time.median())

    year_median_cum_deg = out.groupby("Year")["Cumulative_Degradation"].transform(
        "median"
    )
    year_median_cum_deg = year_median_cum_deg.fillna(cum_deg.median())

    year_median_tyre_life = out.groupby("Year")["TyreLife"].transform("median")
    year_median_tyre_life = year_median_tyre_life.fillna(tyre_life.median())

    wear_pressure_s = pd.Series(wear_pressure, index=out.index)
    year_median_wear_pressure = wear_pressure_s.groupby(out["Year"]).transform("median")
    year_median_wear_pressure = year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    year_median_stop_adv = out.groupby("Year")["best_stop_now_advantage"].transform(
        "median"
    )
    year_median_stop_adv = year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    yd_median_lap_delta = out.groupby(year_domain_key)["LapTime_Delta"].transform(
        "median"
    )
    yd_median_lap_delta = yd_median_lap_delta.fillna(year_median_lap_delta)

    yd_median_lap_time = out.groupby(year_domain_key)["LapTime (s)"].transform("median")
    yd_median_lap_time = yd_median_lap_time.fillna(year_median_lap_time)

    yd_median_cum_deg = out.groupby(year_domain_key)[
        "Cumulative_Degradation"
    ].transform("median")
    yd_median_cum_deg = yd_median_cum_deg.fillna(year_median_cum_deg)

    yd_median_tyre_life = out.groupby(year_domain_key)["TyreLife"].transform("median")
    yd_median_tyre_life = yd_median_tyre_life.fillna(year_median_tyre_life)

    yd_median_wear_pressure = wear_pressure_s.groupby(year_domain_key).transform(
        "median"
    )
    yd_median_wear_pressure = yd_median_wear_pressure.fillna(year_median_wear_pressure)

    yd_median_stop_adv = out.groupby(year_domain_key)[
        "best_stop_now_advantage"
    ].transform("median")
    yd_median_stop_adv = yd_median_stop_adv.fillna(year_median_stop_adv)

    out["lapdelta_vs_year_domain_median"] = lap_delta_pos / (
        yd_median_lap_delta.abs().clip(lower=eps)
    )
    out["laptime_vs_year_domain_median"] = lap_time / yd_median_lap_time.clip(lower=eps)
    out["cumdeg_vs_year_domain_median"] = cum_deg / (
        yd_median_cum_deg.abs().clip(lower=eps)
    )
    out["tyrelife_vs_year_domain_median"] = tyre_life / (
        yd_median_tyre_life.clip(lower=1.0)
    )
    out["wear_pressure_vs_year_domain_median"] = wear_pressure / (
        yd_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_year_domain_median"] = (
        out["best_stop_now_advantage"] - yd_median_stop_adv
    )

    out["testing_lapdelta_domain_gap"] = (
        yd_median_lap_delta - year_median_lap_delta
    ) * out["is_testing"]
    out["testing_laptime_domain_gap"] = (
        yd_median_lap_time - year_median_lap_time
    ) * out["is_testing"]
    out["testing_wear_pressure_domain_gap"] = (
        yd_median_wear_pressure - year_median_wear_pressure
    ) * out["is_testing"]
    out["testing_stop_advantage_domain_gap"] = (
        yd_median_stop_adv - year_median_stop_adv
    ) * out["is_testing"]

    return out
```

## 8. `000316` - CV 0.952593343941, public -

- run: `2-delectable-curvy-dolphin`
- step: `418`
- timestamp: `20260519T090536`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260519T090536/solution.py`
- submission sha: `a868e30f7b4fb13bc9d45020216343ea405398ef393ce7838242e3ccf88c517d`
- code sha: `948679c85da34f986b437e0b1124f7654e48a632c383c4a746c6fe821fdeb437`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )

    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    if pd.isna(race_median_lap).any():
        race_median_lap = race_median_lap.fillna(
            out.groupby("Race")["LapTime (s)"].transform("median")
        )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask],
        [0.18, 0.28, 0.40],
        default=0.34,
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )

    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt

    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    eps = 1e-3

    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    if pd.isna(race_year_median_delta).any():
        race_year_median_delta = race_year_median_delta.fillna(
            out.groupby("Race")["LapTime_Delta"].transform("median")
        )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    if pd.isna(race_year_median_deg).any():
        race_year_median_deg = race_year_median_deg.fillna(
            out.groupby("Race")["Cumulative_Degradation"].transform("median")
        )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    if pd.isna(race_year_median_tyre_life).any():
        race_year_median_tyre_life = race_year_median_tyre_life.fillna(
            out.groupby("Race")["TyreLife"].transform("median")
        )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    if pd.isna(race_year_median_wear_pressure).any():
        race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
            pd.Series(wear_pressure, index=out.index)
            .groupby(out["Race"])
            .transform("median")
        )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    if pd.isna(race_year_median_stop_adv).any():
        race_year_median_stop_adv = race_year_median_stop_adv.fillna(
            out.groupby("Race")["best_stop_now_advantage"].transform("median")
        )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = lap_delta_pos / (
        race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / (
        race_year_median_deg.abs().clip(lower=eps)
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / (
        race_year_median_tyre_life.clip(lower=1.0)
    )
    out["wear_pressure_vs_race_year_median"] = wear_pressure / (
        race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = out["best_stop_now_advantage"] - (
        race_year_median_stop_adv
    )

    # Hypothesis 000316: online dry-tyre legality state per driver-race.
    state_order = (
        pd.DataFrame(
            {
                "Driver": out["Driver"],
                "Race": out["Race"],
                "Year": out["Year"],
                "_lap": lap_number.to_numpy(),
                "_row": np.arange(len(out)),
            },
            index=out.index,
        )
        .sort_values(["Driver", "Race", "Year", "_lap", "_row"], kind="mergesort")
        .index
    )

    state = out.loc[state_order, ["Driver", "Race", "Year", "Compound"]].copy()
    grp = [state["Driver"], state["Race"], state["Year"]]

    is_soft = (state["Compound"] == "SOFT").astype("int8")
    is_medium = (state["Compound"] == "MEDIUM").astype("int8")
    is_hard = (state["Compound"] == "HARD").astype("int8")
    is_dry_now = (is_soft | is_medium | is_hard).astype("int8")
    is_wet_now = state["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")

    soft_seen_to_date = (is_soft.groupby(grp).cumsum() > 0).astype("int8")
    medium_seen_to_date = (is_medium.groupby(grp).cumsum() > 0).astype("int8")
    hard_seen_to_date = (is_hard.groupby(grp).cumsum() > 0).astype("int8")
    wet_seen_to_date = (is_wet_now.groupby(grp).cumsum() > 0).astype("int8")

    prior_soft = is_soft.groupby(grp).cumsum() - is_soft
    prior_medium = is_medium.groupby(grp).cumsum() - is_medium
    prior_hard = is_hard.groupby(grp).cumsum() - is_hard

    current_compound_already_used = (
        ((is_soft == 1) & (prior_soft > 0))
        | ((is_medium == 1) & (prior_medium > 0))
        | ((is_hard == 1) & (prior_hard > 0))
    ).astype("int8")

    distinct_dry_used = (
        soft_seen_to_date + medium_seen_to_date + hard_seen_to_date
    ).astype("int8")
    dry_set_code = (
        soft_seen_to_date + 2 * medium_seen_to_date + 4 * hard_seen_to_date
    ).astype("int8")
    used_dry_set = dry_set_code.map(
        {
            0: "none",
            1: "soft",
            2: "medium",
            3: "soft_medium",
            4: "hard",
            5: "soft_hard",
            6: "medium_hard",
            7: "all_dry",
        }
    ).astype("string")

    wet_exemption_flag = wet_seen_to_date.astype("int8")
    remaining_required_dry_count = np.where(
        wet_exemption_flag == 1,
        0,
        np.clip(2 - distinct_dry_used, 0, 2),
    ).astype("int8")
    stop_debt_if_dry = (
        (is_dry_now == 1)
        & (wet_exemption_flag == 0)
        & (remaining_required_dry_count > 0)
        & (remaining_laps_est.loc[state.index].to_numpy() > 0)
    ).astype("int8")
    laps_remaining_when_still_illegal = (
        remaining_laps_est.loc[state.index].to_numpy() * stop_debt_if_dry
    )

    out["used_dry_set"] = used_dry_set.reindex(out.index)
    out["used_dry_compound_count"] = distinct_dry_used.reindex(out.index).astype("int8")
    out["remaining_required_dry_count"] = (
        pd.Series(remaining_required_dry_count, index=state.index)
        .reindex(out.index)
        .astype("int8")
    )
    out["current_compound_already_used"] = current_compound_already_used.reindex(
        out.index
    ).astype("int8")
    out["stop_debt_if_dry"] = (
        pd.Series(stop_debt_if_dry, index=state.index).reindex(out.index).astype("int8")
    )
    out["wet_exemption_flag"] = wet_exemption_flag.reindex(out.index).astype("int8")
    out["laps_remaining_when_still_illegal"] = (
        pd.Series(laps_remaining_when_still_illegal, index=state.index)
        .reindex(out.index)
        .astype("float32")
    )

    out["dry_legality_pressure"] = (
        out["stop_debt_if_dry"]
        * out["remaining_laps_est"]
        / (out["remaining_required_dry_count"] + 1.0)
    )
    out["illegal_on_reused_dry"] = (
        out["stop_debt_if_dry"] * out["current_compound_already_used"]
    ).astype("int8")
    out["dry_legality_x_finishable"] = (
        out["stop_debt_if_dry"] * out["can_finish_current_tyre"]
    ).astype("int8")
    out["dry_legality_x_wet_regime"] = (
        out["remaining_required_dry_count"] * (1 - out["is_wet_compound"])
    ).astype("int8")

    return out
```

## 9. `000257` - CV 0.952566795543, public -

- run: `2-delectable-curvy-dolphin`
- step: `390`
- timestamp: `20260519T042240`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260519T042240/solution.py`
- submission sha: `99ce4a444e437e72a879cb9c44b0f2870bda08a65273017667e7ad8c32796343`
- code sha: `06117aec2d39f3df3f2bf24ccb590cbc0b517caa93521c30caf7d5da711a4607`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )

    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    if pd.isna(race_median_lap).any():
        race_median_lap = race_median_lap.fillna(
            out.groupby("Race")["LapTime (s)"].transform("median")
        )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask],
        [0.18, 0.28, 0.40],
        default=0.34,
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )

    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt

    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    eps = 1e-3

    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    if pd.isna(race_year_median_delta).any():
        race_year_median_delta = race_year_median_delta.fillna(
            out.groupby("Race")["LapTime_Delta"].transform("median")
        )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    if pd.isna(race_year_median_deg).any():
        race_year_median_deg = race_year_median_deg.fillna(
            out.groupby("Race")["Cumulative_Degradation"].transform("median")
        )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    if pd.isna(race_year_median_tyre_life).any():
        race_year_median_tyre_life = race_year_median_tyre_life.fillna(
            out.groupby("Race")["TyreLife"].transform("median")
        )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    if pd.isna(race_year_median_wear_pressure).any():
        race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
            pd.Series(wear_pressure, index=out.index)
            .groupby(out["Race"])
            .transform("median")
        )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    if pd.isna(race_year_median_stop_adv).any():
        race_year_median_stop_adv = race_year_median_stop_adv.fillna(
            out.groupby("Race")["best_stop_now_advantage"].transform("median")
        )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = lap_delta_pos / (
        race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / (
        race_year_median_deg.abs().clip(lower=eps)
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / (
        race_year_median_tyre_life.clip(lower=1.0)
    )
    out["wear_pressure_vs_race_year_median"] = wear_pressure / (
        race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = out["best_stop_now_advantage"] - (
        race_year_median_stop_adv
    )

    # Hypothesis 000257: infer whether a Race_Year is undercut-friendly or overcut-friendly
    # from observable tyre warm-up and degradation behavior, then gate pit-urgency features.
    recent_stop_mask = out["Stint"].fillna(1).clip(lower=1) >= 2
    fresh_mask = recent_stop_mask & (tyre_life <= 1)
    warmed_mask = recent_stop_mask & tyre_life.between(3, 5)
    early_mask = tyre_life.between(1, 2)
    later_mask = tyre_life.between(5, 8)

    fresh_outlap_delta = out["LapTime_Delta"].where(fresh_mask)
    warmed_delta = out["LapTime_Delta"].where(warmed_mask)
    early_delta = out["LapTime_Delta"].where(early_mask)
    later_delta = out["LapTime_Delta"].where(later_mask)
    deg_per_tyre_lap_series = out["deg_per_tyre_lap"].astype("float32")

    race_year_fresh_delta = fresh_outlap_delta.groupby(out["Race_Year"]).transform(
        "median"
    )
    if pd.isna(race_year_fresh_delta).any():
        race_year_fresh_delta = race_year_fresh_delta.fillna(
            fresh_outlap_delta.groupby(out["Race"]).transform("median")
        )
    race_year_fresh_delta = race_year_fresh_delta.fillna(lap_delta_pos.median())

    race_year_warmed_delta = warmed_delta.groupby(out["Race_Year"]).transform("median")
    if pd.isna(race_year_warmed_delta).any():
        race_year_warmed_delta = race_year_warmed_delta.fillna(
            warmed_delta.groupby(out["Race"]).transform("median")
        )
    race_year_warmed_delta = race_year_warmed_delta.fillna(lap_delta_pos.median())

    race_year_early_delta = early_delta.groupby(out["Race_Year"]).transform("median")
    if pd.isna(race_year_early_delta).any():
        race_year_early_delta = race_year_early_delta.fillna(
            early_delta.groupby(out["Race"]).transform("median")
        )
    race_year_early_delta = race_year_early_delta.fillna(lap_delta_pos.median())

    race_year_later_delta = later_delta.groupby(out["Race_Year"]).transform("median")
    if pd.isna(race_year_later_delta).any():
        race_year_later_delta = race_year_later_delta.fillna(
            later_delta.groupby(out["Race"]).transform("median")
        )
    race_year_later_delta = race_year_later_delta.fillna(lap_delta_pos.median())

    race_year_deg_spread = deg_per_tyre_lap_series.groupby(out["Race_Year"]).transform(
        "std"
    )
    if pd.isna(race_year_deg_spread).any():
        race_year_deg_spread = race_year_deg_spread.fillna(
            deg_per_tyre_lap_series.groupby(out["Race"]).transform("std")
        )
    race_year_deg_spread = race_year_deg_spread.fillna(
        float(np.nanstd(deg_per_tyre_lap_series))
    )

    race_year_fresh_var = fresh_outlap_delta.groupby(out["Race_Year"]).transform("std")
    if pd.isna(race_year_fresh_var).any():
        race_year_fresh_var = race_year_fresh_var.fillna(
            fresh_outlap_delta.groupby(out["Race"]).transform("std")
        )
    race_year_fresh_var = race_year_fresh_var.fillna(float(np.nanstd(lap_delta_pos)))

    warmup_penalty = (race_year_fresh_delta - race_year_warmed_delta).clip(-10.0, 10.0)
    early_stint_deg_ramp = (race_year_later_delta - race_year_early_delta).clip(
        -10.0, 10.0
    )

    def robust_z(x: pd.Series) -> pd.Series:
        med = x.median()
        mad = (x - med).abs().median()
        scale = max(float(mad) * 1.4826, 1e-3)
        return ((x - med) / scale).clip(-6.0, 6.0)

    undercut_score = (
        1.15 * robust_z(early_stint_deg_ramp.astype("float32"))
        + 0.85 * robust_z(race_year_deg_spread.astype("float32"))
        - 1.00 * robust_z(warmup_penalty.astype("float32"))
        - 0.55 * robust_z(race_year_fresh_var.astype("float32"))
    )

    undercut_regime_prob = 1.0 / (1.0 + np.exp(-undercut_score))
    overcut_regime_prob = 1.0 - undercut_regime_prob

    out["race_warmup_penalty"] = warmup_penalty
    out["race_early_stint_deg_ramp"] = early_stint_deg_ramp
    out["race_deg_spread"] = race_year_deg_spread
    out["race_fresh_outlap_var"] = race_year_fresh_var
    out["undercut_regime_score"] = undercut_score
    out["undercut_regime_prob"] = undercut_regime_prob
    out["overcut_regime_prob"] = overcut_regime_prob

    out["undercut_x_stop_advantage"] = (
        undercut_regime_prob * out["best_stop_now_advantage"]
    )
    out["undercut_x_wear_pressure"] = undercut_regime_prob * wear_pressure
    out["overcut_x_wait_regret"] = overcut_regime_prob * out["regret_if_wait_1lap"]
    out["overcut_x_yellow_option"] = (
        overcut_regime_prob * out["yellow_flag_option_value_proxy"]
    )
    out["undercut_x_finish_pressure"] = (
        undercut_regime_prob * out["current_tyre_finish_pressure"]
    )

    return out
```

## 10. `000052` - CV 0.952566612538, public 0.95051

- run: `2-delectable-curvy-dolphin`
- step: `320`
- timestamp: `20260518T164418`
- solution: `logs/2-delectable-curvy-dolphin/artifacts/20260518T164418/solution.py`
- submission sha: `eae711286cfbf71479483223a62cfb9a843f038d4275d44fa1d4cb19b916e70b`
- code sha: `91a2fd2fa0a8aee90c9e291dde0b638e7c65d40ade51f5a61f4f195e4b38d346`

```python
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    cat_cols = ["Compound", "Driver", "Race"]
    for c in cat_cols:
        out[c] = out[c].astype("string").fillna("UNK")

    out["is_testing"] = (out["Race"] == "Pre-Season Testing").astype("int8")
    out["is_wet_compound"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    out["is_dry_compound"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    )

    out["Race_Year"] = out["Race"] + "_" + out["Year"].astype(str)
    out["Year_str"] = out["Year"].astype(str)

    lap_number = out["LapNumber"].clip(lower=1)
    tyre_life = out["TyreLife"].clip(lower=0)
    race_progress = out["RaceProgress"].clip(lower=1e-3, upper=0.999)
    cum_deg = out["Cumulative_Degradation"].fillna(0)
    lap_delta_pos = out["LapTime_Delta"].fillna(0).clip(lower=0)
    lap_time = out["LapTime (s)"].fillna(out["LapTime (s)"].median())

    out["lap_progress_remaining"] = 1.0 - out["RaceProgress"]
    out["tyre_life_ratio"] = tyre_life / lap_number
    out["stint_progress_ratio"] = tyre_life / (lap_number + 1.0)
    out["deg_per_tyre_lap"] = cum_deg / tyre_life.clip(lower=1)
    out["deg_per_race_lap"] = cum_deg / lap_number

    out["pace_x_deg"] = out["LapTime_Delta"] * out["deg_per_tyre_lap"]
    out["pace_x_tyrelife"] = out["LapTime_Delta"] * out["TyreLife"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]

    wet_mask = out["is_wet_compound"].astype(bool)
    test_mask = out["is_testing"].astype(bool)
    dry_mask = out["is_dry_compound"].astype(bool) & (~test_mask)

    for col in [
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "TyreLife",
        "deg_per_tyre_lap",
    ]:
        base = out[col]
        out[f"{col}_dry_only"] = base.where(dry_mask, 0)
        out[f"{col}_wet_only"] = base.where(wet_mask, 0)
        out[f"{col}_test_only"] = base.where(test_mask, 0)

    out["regime"] = np.select(
        [test_mask, wet_mask, dry_mask], ["testing", "wet", "dry"], default="other"
    ).astype(object)

    total_laps_est = (lap_number / race_progress).clip(lower=lap_number, upper=120)
    remaining_laps_est = (total_laps_est - lap_number).clip(lower=0)
    out["total_laps_est"] = total_laps_est
    out["remaining_laps_est"] = remaining_laps_est

    compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 1.00,
                "MEDIUM": 0.72,
                "HARD": 0.50,
                "INTERMEDIATE": 0.82,
                "WET": 0.92,
            }
        )
        .astype("float32")
        .fillna(0.65)
    )

    next_compound_deg_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 0.72,
                "MEDIUM": 0.50,
                "HARD": 0.50,
                "INTERMEDIATE": 0.70,
                "WET": 0.82,
            }
        )
        .astype("float32")
        .fillna(0.55)
    )

    race_median_lap = out.groupby("Race_Year")["LapTime (s)"].transform("median")
    if pd.isna(race_median_lap).any():
        race_median_lap = race_median_lap.fillna(
            out.groupby("Race")["LapTime (s)"].transform("median")
        )
    race_median_lap = race_median_lap.fillna(lap_time.median())

    pit_loss_factor = np.select(
        [test_mask, wet_mask, dry_mask],
        [0.18, 0.28, 0.40],
        default=0.34,
    )
    pit_loss_proxy = race_median_lap * pit_loss_factor

    wear_pressure = out["deg_per_tyre_lap"].fillna(0).clip(lower=0) * (
        1.0 + compound_deg_template * tyre_life / (remaining_laps_est + 1.0)
    )
    finish_pressure = (
        compound_deg_template * out["lap_progress_remaining"].clip(lower=0) * tyre_life
    )
    next_compound_margin = (
        (compound_deg_template - next_compound_deg_template).clip(lower=0)
        * race_median_lap
        * 0.12
    )

    yellow_flag_option_value_proxy = (
        pit_loss_proxy
        * out["lap_progress_remaining"].clip(lower=0)
        * (0.05 + 0.30 * out["is_wet_compound"] + 0.12 * out["is_testing"])
    )

    amortized_pit_loss = pit_loss_proxy / np.sqrt(remaining_laps_est + 1.0)
    wait_1lap_cost = lap_delta_pos + wear_pressure + 0.15 * finish_pressure
    wait_2lap_cost = 2.0 * lap_delta_pos + 3.0 * wear_pressure + 0.50 * finish_pressure

    out["best_next_compound_margin"] = next_compound_margin
    out["yellow_flag_option_value_proxy"] = yellow_flag_option_value_proxy
    out["regret_if_wait_1lap"] = (
        wait_1lap_cost + 0.35 * wait_2lap_cost - yellow_flag_option_value_proxy
    )
    out["best_stop_now_advantage"] = (
        wait_2lap_cost
        + next_compound_margin
        - amortized_pit_loss
        - yellow_flag_option_value_proxy
    )

    base_life_template = (
        out["Compound"]
        .map(
            {
                "SOFT": 18.0,
                "MEDIUM": 26.0,
                "HARD": 34.0,
                "INTERMEDIATE": 24.0,
                "WET": 22.0,
            }
        )
        .astype("float32")
        .fillna(24.0)
    )
    deg_penalty = 1.0 + 0.55 * out["deg_per_tyre_lap"].fillna(0).clip(lower=0)
    est_total_tyre_life = (base_life_template / deg_penalty).clip(lower=8.0, upper=45.0)
    tyre_laps_left_est = (est_total_tyre_life - tyre_life).clip(lower=-5.0, upper=45.0)
    finish_margin_current_tyre = tyre_laps_left_est - remaining_laps_est

    dry_race_mask = dry_mask.astype("int8")
    exempt_mask = (wet_mask | test_mask).astype("int8")
    observed_stop_debt = (
        (out["Stint"].fillna(1).clip(lower=1) < 2) & (remaining_laps_est > 0)
    ).astype("int8")
    remaining_dry_compound_debt = (
        dry_race_mask.astype(bool)
        & (observed_stop_debt == 1)
        & (remaining_laps_est > 0)
    ).astype("int8")
    can_finish_current_tyre = (
        (finish_margin_current_tyre >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["finish_margin_current_tyre"] = finish_margin_current_tyre
    out["can_finish_current_tyre"] = can_finish_current_tyre
    out["rule_exempt_wet_or_testing"] = exempt_mask
    out["observed_stop_debt"] = observed_stop_debt
    out["remaining_dry_compound_debt"] = remaining_dry_compound_debt

    out["can_finish_but_owes_stop"] = (
        (can_finish_current_tyre == 1) & (observed_stop_debt == 1) & (exempt_mask == 0)
    ).astype("int8")
    out["can_finish_but_owes_dry_compound"] = (
        (can_finish_current_tyre == 1)
        & (remaining_dry_compound_debt == 1)
        & (exempt_mask == 0)
    ).astype("int8")

    late_race_phase = (race_progress >= 0.70).astype("int8")
    out["late_race_phase"] = late_race_phase
    out["late_race_legal_pressure"] = (
        out["can_finish_but_owes_dry_compound"]
        * race_progress
        * (1.0 + 0.75 * (race_progress >= 0.85).astype("float32"))
    )
    out["finishable_stop_debt_margin"] = finish_margin_current_tyre * observed_stop_debt
    out["finishable_dry_debt_margin"] = (
        finish_margin_current_tyre * remaining_dry_compound_debt
    )

    conservative_compound_life = (
        out["Compound"]
        .map(
            {
                "SOFT": 17.0,
                "MEDIUM": 24.0,
                "HARD": 31.0,
                "INTERMEDIATE": 22.0,
                "WET": 20.0,
            }
        )
        .astype("float32")
        .fillna(22.0)
    )
    current_tyre_age_to_finish = tyre_life + remaining_laps_est
    current_tyre_finish_margin_signed = (
        conservative_compound_life - current_tyre_age_to_finish
    )
    current_tyre_finish_pressure = (
        (current_tyre_age_to_finish / conservative_compound_life.clip(lower=1.0)) - 1.0
    ).clip(lower=-2.0, upper=3.0)
    current_tyre_cannot_finish = (
        (current_tyre_finish_margin_signed < 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["current_tyre_age_to_finish"] = current_tyre_age_to_finish
    out["current_tyre_finish_margin_signed"] = current_tyre_finish_margin_signed
    out["current_tyre_finish_pressure"] = current_tyre_finish_pressure
    out["current_tyre_cannot_finish"] = current_tyre_cannot_finish

    eps = 1e-3

    race_year_median_delta = out.groupby("Race_Year")["LapTime_Delta"].transform(
        "median"
    )
    if pd.isna(race_year_median_delta).any():
        race_year_median_delta = race_year_median_delta.fillna(
            out.groupby("Race")["LapTime_Delta"].transform("median")
        )
    race_year_median_delta = race_year_median_delta.fillna(lap_delta_pos.median())

    race_year_median_deg = out.groupby("Race_Year")["Cumulative_Degradation"].transform(
        "median"
    )
    if pd.isna(race_year_median_deg).any():
        race_year_median_deg = race_year_median_deg.fillna(
            out.groupby("Race")["Cumulative_Degradation"].transform("median")
        )
    race_year_median_deg = race_year_median_deg.fillna(cum_deg.median())

    race_year_median_tyre_life = out.groupby("Race_Year")["TyreLife"].transform(
        "median"
    )
    if pd.isna(race_year_median_tyre_life).any():
        race_year_median_tyre_life = race_year_median_tyre_life.fillna(
            out.groupby("Race")["TyreLife"].transform("median")
        )
    race_year_median_tyre_life = race_year_median_tyre_life.fillna(tyre_life.median())

    race_year_median_wear_pressure = (
        pd.Series(wear_pressure, index=out.index)
        .groupby(out["Race_Year"])
        .transform("median")
    )
    if pd.isna(race_year_median_wear_pressure).any():
        race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
            pd.Series(wear_pressure, index=out.index)
            .groupby(out["Race"])
            .transform("median")
        )
    race_year_median_wear_pressure = race_year_median_wear_pressure.fillna(
        float(np.nanmedian(wear_pressure))
    )

    race_year_median_stop_adv = out.groupby("Race_Year")[
        "best_stop_now_advantage"
    ].transform("median")
    if pd.isna(race_year_median_stop_adv).any():
        race_year_median_stop_adv = race_year_median_stop_adv.fillna(
            out.groupby("Race")["best_stop_now_advantage"].transform("median")
        )
    race_year_median_stop_adv = race_year_median_stop_adv.fillna(
        out["best_stop_now_advantage"].median()
    )

    out["laptime_vs_race_year_median"] = lap_time / race_median_lap.clip(lower=eps)
    out["lapdelta_vs_race_year_median"] = lap_delta_pos / (
        race_year_median_delta.abs().clip(lower=eps)
    )
    out["cumdeg_vs_race_year_median"] = cum_deg / (
        race_year_median_deg.abs().clip(lower=eps)
    )
    out["tyrelife_vs_race_year_median"] = tyre_life / (
        race_year_median_tyre_life.clip(lower=1.0)
    )
    out["wear_pressure_vs_race_year_median"] = wear_pressure / (
        race_year_median_wear_pressure.abs().clip(lower=eps)
    )
    out["stop_advantage_vs_race_year_median"] = out["best_stop_now_advantage"] - (
        race_year_median_stop_adv
    )

    for col, rare_thresh in [("Driver", 8), ("Race_Year", 12), ("Race", 12)]:
        counts = out[col].value_counts(dropna=False)
        freq = out[col].map(counts).astype("float32")
        out[f"{col}_freq"] = freq
        out[f"{col}_log_freq"] = np.log1p(freq).astype("float32")
        out[f"{col}_is_rare"] = (freq <= rare_thresh).astype("int8")
        out[col] = out[col].where(freq > rare_thresh, f"RARE_{col}")

    driver_raceyear = (
        out["Driver"].astype("string") + "__" + out["Race_Year"].astype("string")
    )
    driver_raceyear_counts = driver_raceyear.value_counts(dropna=False)
    driver_raceyear_freq = driver_raceyear.map(driver_raceyear_counts).astype("float32")
    out["Driver_RaceYear_freq"] = driver_raceyear_freq
    out["Driver_RaceYear_log_freq"] = np.log1p(driver_raceyear_freq).astype("float32")
    out["Driver_RaceYear_is_rare"] = (driver_raceyear_freq <= 3).astype("int8")

    out["Is_Slick"] = out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    out["Is_WetWeather"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")

    fresh_slick_life = 31.0
    fresh_wetweather_life = 22.0

    slick_next_stop_margin = fresh_slick_life - remaining_laps_est
    wetweather_next_stop_margin = fresh_wetweather_life - remaining_laps_est

    out["slick_next_stop_finish_margin"] = slick_next_stop_margin
    out["wetweather_next_stop_finish_margin"] = wetweather_next_stop_margin

    out["class_aware_next_stop_finish_margin"] = np.select(
        [out["Is_Slick"] == 1, out["Is_WetWeather"] == 1],
        [slick_next_stop_margin, wetweather_next_stop_margin],
        default=np.maximum(slick_next_stop_margin, wetweather_next_stop_margin),
    ).astype("float32")

    out["class_aware_next_stop_can_finish"] = (
        (out["class_aware_next_stop_finish_margin"] >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    out["mixed_class_switch_to_wetweather"] = (
        (out["Is_Slick"] == 1)
        & (wetweather_next_stop_margin >= 0)
        & (remaining_laps_est > 0)
    ).astype("int8")

    out["mixed_class_switch_to_slick"] = (
        (out["Is_WetWeather"] == 1)
        & (slick_next_stop_margin >= 0)
        & (remaining_laps_est > 0)
    ).astype("int8")

    out["mixed_class_switch_margin"] = np.select(
        [out["Is_Slick"] == 1, out["Is_WetWeather"] == 1],
        [wetweather_next_stop_margin, slick_next_stop_margin],
        default=0.0,
    ).astype("float32")

    out["current_class_next_stop_margin"] = np.select(
        [out["Is_Slick"] == 1, out["Is_WetWeather"] == 1],
        [slick_next_stop_margin, wetweather_next_stop_margin],
        default=0.0,
    ).astype("float32")

    out["current_class_next_stop_can_finish"] = (
        (out["current_class_next_stop_margin"] >= 0) & (remaining_laps_est > 0)
    ).astype("int8")

    return out
```
