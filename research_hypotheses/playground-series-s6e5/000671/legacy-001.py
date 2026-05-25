import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORKING = Path("./working")
WORKING.mkdir(exist_ok=True)

TRAIN_PATH = INPUT / "train.csv.gz"
TEST_PATH = INPUT / "test.csv.gz"
SAMPLE_PATH = INPUT / "sample_submission.csv.gz"
TARGET = "PitNextLap"
ID = "id"
RANDOM_STATE = 42

print('{"research_hypotheses_llm_claimed_used":["000671"]}')

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

full = pd.concat(
    [train.drop(columns=[TARGET]).assign(_is_train=1), test.assign(_is_train=0)],
    ignore_index=True,
)


def normalize_event_name(name):
    name = str(name)
    aliases = {
        "Mexico City Grand Prix": "Mexico City Grand Prix",
        "Sao Paulo Grand Prix": "São Paulo Grand Prix",
        "Pre-Season Testing": None,
    }
    return aliases.get(name, name)


def to_seconds(x):
    if pd.isna(x):
        return np.nan
    if hasattr(x, "total_seconds"):
        return x.total_seconds()
    try:
        return pd.to_timedelta(x).total_seconds()
    except Exception:
        return np.nan


def fastf1_features_for_session(year, race):
    event_name = normalize_event_name(race)
    cols = [
        "Year",
        "Race",
        "Driver",
        "LapNumber",
        "ff1_sc_active",
        "ff1_vsc_active",
        "ff1_yellow_active",
        "ff1_neutral_just_ended",
        "ff1_laps_since_neutral",
        "ff1_gap_to_leader",
        "ff1_interval_ahead",
        "ff1_prior_pit_count",
        "ff1_pit_in_time",
        "ff1_pit_out_time",
        "ff1_air_temp",
        "ff1_track_temp",
        "ff1_rainfall",
        "ff1_track_temp_delta",
        "ff1_cheap_stop_window",
        "ff1_rejoin_traffic_risk",
        "ff1_stream_gap_flag",
    ]
    if event_name is None:
        return pd.DataFrame(columns=cols)

    try:
        import fastf1

        cache_dir = WORKING / "fastf1_cache"
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))

        session = fastf1.get_session(int(year), event_name, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=True)

        laps = session.laps.copy()
        if laps.empty:
            return pd.DataFrame(columns=cols)

        laps["LapNumber"] = pd.to_numeric(laps["LapNumber"], errors="coerce").astype(
            "Int64"
        )
        laps["Driver"] = laps["Driver"].astype(str)
        laps["lap_time_s"] = laps["LapTime"].apply(to_seconds)
        laps["lap_start_s"] = (
            laps["LapStartTime"].apply(to_seconds) if "LapStartTime" in laps else np.nan
        )
        laps["pit_in_s"] = (
            laps["PitInTime"].apply(to_seconds) if "PitInTime" in laps else np.nan
        )
        laps["pit_out_s"] = (
            laps["PitOutTime"].apply(to_seconds) if "PitOutTime" in laps else np.nan
        )

        if "Position" in laps:
            pos = pd.to_numeric(laps["Position"], errors="coerce")
            leader_time = laps.assign(_pos=pos).loc[
                lambda d: d["_pos"] == 1, ["LapNumber", "lap_start_s"]
            ]
            leader_time = leader_time.rename(
                columns={"lap_start_s": "leader_lap_start_s"}
            )
            laps = laps.merge(leader_time, on="LapNumber", how="left")
            laps["ff1_gap_to_leader"] = laps["lap_start_s"] - laps["leader_lap_start_s"]
            tmp = laps[["LapNumber", "Position", "lap_start_s"]].copy()
            tmp["Position"] = pd.to_numeric(tmp["Position"], errors="coerce") + 1
            tmp = tmp.rename(columns={"lap_start_s": "ahead_lap_start_s"})
            laps = laps.merge(tmp, on=["LapNumber", "Position"], how="left")
            laps["ff1_interval_ahead"] = laps["lap_start_s"] - laps["ahead_lap_start_s"]
        else:
            laps["ff1_gap_to_leader"] = np.nan
            laps["ff1_interval_ahead"] = np.nan

        laps = laps.sort_values(["Driver", "LapNumber"])
        pit_marker = laps["pit_in_s"].notna().astype(int)
        laps["ff1_prior_pit_count"] = (
            pit_marker.groupby(laps["Driver"]).cumsum() - pit_marker
        )
        laps["ff1_pit_in_time"] = laps["pit_in_s"]
        laps["ff1_pit_out_time"] = laps["pit_out_s"]

        status_laps = laps[["LapNumber"]].drop_duplicates().sort_values("LapNumber")
        for c in ["ff1_sc_active", "ff1_vsc_active", "ff1_yellow_active"]:
            status_laps[c] = 0

        if (
            hasattr(session, "track_status")
            and session.track_status is not None
            and len(session.track_status)
        ):
            ts = session.track_status.copy()
            ts["time_s"] = ts["Time"].apply(to_seconds)
            lap_starts = (
                laps.groupby("LapNumber")["lap_start_s"].median().dropna().sort_index()
            )
            for _, row in ts.iterrows():
                code = str(row.get("Status", ""))
                t = row.get("time_s", np.nan)
                if pd.isna(t) or lap_starts.empty:
                    continue
                lap_no = (
                    int(
                        lap_starts.index[
                            np.searchsorted(lap_starts.values, t, side="right") - 1
                        ]
                    )
                    if t >= lap_starts.iloc[0]
                    else int(lap_starts.index[0])
                )
                mask = status_laps["LapNumber"] >= lap_no
                if "4" in code:
                    status_laps.loc[mask, "ff1_sc_active"] = 1
                if "6" in code or "7" in code:
                    status_laps.loc[mask, "ff1_vsc_active"] = 1
                if "2" in code:
                    status_laps.loc[mask, "ff1_yellow_active"] = 1
                if "1" in code:
                    status_laps.loc[
                        mask, ["ff1_sc_active", "ff1_vsc_active", "ff1_yellow_active"]
                    ] = 0

        neutral = (
            status_laps["ff1_sc_active"]
            | status_laps["ff1_vsc_active"]
            | status_laps["ff1_yellow_active"]
        ).astype(int)
        prev_neutral = neutral.shift(1).fillna(0).astype(int)
        status_laps["ff1_neutral_just_ended"] = (
            (prev_neutral == 1) & (neutral == 0)
        ).astype(int)
        last_neutral_lap = np.where(neutral == 1, status_laps["LapNumber"], np.nan)
        last_neutral_lap = pd.Series(last_neutral_lap).ffill().to_numpy()
        status_laps["ff1_laps_since_neutral"] = (
            status_laps["LapNumber"].to_numpy() - last_neutral_lap
        )
        status_laps["ff1_laps_since_neutral"] = status_laps[
            "ff1_laps_since_neutral"
        ].fillna(99)

        laps = laps.merge(status_laps, on="LapNumber", how="left")

        weather_cols = ["AirTemp", "TrackTemp", "Rainfall"]
        if (
            hasattr(session, "weather_data")
            and session.weather_data is not None
            and len(session.weather_data)
        ):
            weather = session.weather_data.copy()
            weather["weather_time_s"] = weather["Time"].apply(to_seconds)
            weather = weather.sort_values("weather_time_s")
            lap_times = (
                laps[["LapNumber", "lap_start_s"]]
                .drop_duplicates()
                .sort_values("lap_start_s")
            )
            weather = pd.merge_asof(
                lap_times,
                weather[
                    ["weather_time_s"]
                    + [c for c in weather_cols if c in weather.columns]
                ],
                left_on="lap_start_s",
                right_on="weather_time_s",
                direction="nearest",
                tolerance=600,
            )
            weather = weather.rename(
                columns={
                    "AirTemp": "ff1_air_temp",
                    "TrackTemp": "ff1_track_temp",
                    "Rainfall": "ff1_rainfall",
                }
            )
            weather["ff1_track_temp_delta"] = weather["ff1_track_temp"].diff().fillna(0)
            laps = laps.merge(
                weather[
                    [
                        "LapNumber",
                        "ff1_air_temp",
                        "ff1_track_temp",
                        "ff1_rainfall",
                        "ff1_track_temp_delta",
                    ]
                ],
                on="LapNumber",
                how="left",
            )
        else:
            laps["ff1_air_temp"] = np.nan
            laps["ff1_track_temp"] = np.nan
            laps["ff1_rainfall"] = np.nan
            laps["ff1_track_temp_delta"] = np.nan

        laps["ff1_cheap_stop_window"] = (
            (
                laps["ff1_sc_active"].fillna(0).astype(int)
                | laps["ff1_vsc_active"].fillna(0).astype(int)
                | laps["ff1_neutral_just_ended"].fillna(0).astype(int)
            )
        ).astype(int)
        laps["ff1_rejoin_traffic_risk"] = (
            (laps["ff1_interval_ahead"].fillna(99).abs() < 2.0)
            | (laps["ff1_gap_to_leader"].fillna(99).abs() < 4.0)
        ).astype(int)
        laps["ff1_stream_gap_flag"] = (
            laps["ff1_gap_to_leader"].isna() | laps["ff1_interval_ahead"].isna()
        ).astype(int)

        out = laps.rename(columns={"LapNumber": "LapNumber"})[
            [
                "Driver",
                "LapNumber",
                "ff1_sc_active",
                "ff1_vsc_active",
                "ff1_yellow_active",
                "ff1_neutral_just_ended",
                "ff1_laps_since_neutral",
                "ff1_gap_to_leader",
                "ff1_interval_ahead",
                "ff1_prior_pit_count",
                "ff1_pit_in_time",
                "ff1_pit_out_time",
                "ff1_air_temp",
                "ff1_track_temp",
                "ff1_rainfall",
                "ff1_track_temp_delta",
                "ff1_cheap_stop_window",
                "ff1_rejoin_traffic_risk",
                "ff1_stream_gap_flag",
            ]
        ].copy()
        out["Year"] = int(year)
        out["Race"] = race
        return out[cols]
    except Exception as e:
        print(
            f"FastF1 load skipped for {year} {race}: {type(e).__name__}: {str(e)[:120]}"
        )
        return pd.DataFrame(columns=cols)


ff1_parts = []
for (year, race), _ in full[["Year", "Race"]].drop_duplicates().iterrows():
    ff1_parts.append(fastf1_features_for_session(year, race))

if ff1_parts:
    ff1 = pd.concat(ff1_parts, ignore_index=True)
else:
    ff1 = pd.DataFrame()

if len(ff1):
    ff1["LapNumber"] = pd.to_numeric(ff1["LapNumber"], errors="coerce")
    full = full.merge(ff1, on=["Year", "Race", "Driver", "LapNumber"], how="left")
else:
    for c in [
        "ff1_sc_active",
        "ff1_vsc_active",
        "ff1_yellow_active",
        "ff1_neutral_just_ended",
        "ff1_laps_since_neutral",
        "ff1_gap_to_leader",
        "ff1_interval_ahead",
        "ff1_prior_pit_count",
        "ff1_pit_in_time",
        "ff1_pit_out_time",
        "ff1_air_temp",
        "ff1_track_temp",
        "ff1_rainfall",
        "ff1_track_temp_delta",
        "ff1_cheap_stop_window",
        "ff1_rejoin_traffic_risk",
        "ff1_stream_gap_flag",
    ]:
        full[c] = np.nan

for c in full.columns:
    if c.startswith("ff1_"):
        if c in [
            "ff1_stream_gap_flag",
            "ff1_sc_active",
            "ff1_vsc_active",
            "ff1_yellow_active",
            "ff1_neutral_just_ended",
            "ff1_cheap_stop_window",
            "ff1_rejoin_traffic_risk",
        ]:
            full[c] = full[c].fillna(0).astype(np.int8)
        else:
            full[c] = full[c].fillna(-999.0)

full["race_driver"] = full["Race"].astype(str) + "_" + full["Driver"].astype(str)
full["compound_stint"] = full["Compound"].astype(str) + "_" + full["Stint"].astype(str)
full["lap_x_tyre"] = full["LapNumber"] * full["TyreLife"]
full["race_progress_left"] = 1.0 - full["RaceProgress"]
full["tyre_life_ratio"] = full["TyreLife"] / np.maximum(full["LapNumber"], 1)
full["degradation_per_tyre_lap"] = full["Cumulative_Degradation"] / np.maximum(
    full["TyreLife"], 1
)
full["pitstop_prev_lap_proxy"] = (
    full.groupby(["_is_train", "Year", "Race", "Driver"])["PitStop"].shift(1).fillna(0)
)
full["pitstop_cum_proxy"] = (
    full.groupby(["_is_train", "Year", "Race", "Driver"])["PitStop"].cumsum()
    - full["PitStop"]
)

feature_cols = [c for c in full.columns if c not in [ID, "_is_train"]]
X_all = full.loc[full["_is_train"] == 1, feature_cols].reset_index(drop=True)
X_test = full.loc[full["_is_train"] == 0, feature_cols].reset_index(drop=True)
y = train[TARGET].astype(int).reset_index(drop=True)

cat_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
for c in cat_cols:
    X_all[c] = X_all[c].astype(str).fillna("missing")
    X_test[c] = X_test[c].astype(str).fillna("missing")

num_cols = [c for c in feature_cols if c not in cat_cols]
for c in num_cols:
    X_all[c] = pd.to_numeric(X_all[c], errors="coerce").fillna(-999)
    X_test[c] = pd.to_numeric(X_test[c], errors="coerce").fillna(-999)

try:
    from catboost import CatBoostClassifier, Pool

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_all), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y), 1):
        train_pool = Pool(X_all.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
        valid_pool = Pool(X_all.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=900,
            learning_rate=0.045,
            depth=7,
            l2_leaf_reg=8,
            random_seed=RANDOM_STATE + fold,
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=80,
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        oof[va_idx] = model.predict_proba(valid_pool)[:, 1]
        test_pred += (
            model.predict_proba(Pool(X_test, cat_features=cat_cols))[:, 1]
            / skf.n_splits
        )
        auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        aucs.append(auc)
        print(f"fold {fold} roc_auc={auc:.6f}")

except Exception as e:
    print(
        f"CatBoost unavailable, using HistGradientBoosting fallback: {type(e).__name__}: {e}"
    )
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder

    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                cat_cols,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X_all), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y), 1):
        clf = make_pipeline(
            pre,
            HistGradientBoostingClassifier(
                learning_rate=0.055,
                max_leaf_nodes=31,
                max_iter=350,
                l2_regularization=0.05,
                random_state=RANDOM_STATE + fold,
            ),
        )
        clf.fit(X_all.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = clf.predict_proba(X_all.iloc[va_idx])[:, 1]
        test_pred += clf.predict_proba(X_test)[:, 1] / skf.n_splits
        auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        aucs.append(auc)
        print(f"fold {fold} roc_auc={auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"mean_fold_roc_auc={np.mean(aucs):.6f}")
print(f"oof_roc_auc={cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(WORKING / "oof_predictions.csv.gz", index=False, compression="gzip")

test_predictions = sample[[ID]].copy()
test_predictions[TARGET] = np.clip(test_pred, 0, 1)
test_predictions.to_csv(
    WORKING / "test_predictions.csv.gz", index=False, compression="gzip"
)
test_predictions.to_csv(WORKING / "submission.csv", index=False)
print(f"saved_submission={WORKING / 'submission.csv'}")
