import os
import signal
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORKING = Path("./working")
WORKING.mkdir(parents=True, exist_ok=True)

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
    sort=False,
)
full["Race"] = full["Race"].astype(str)
full["Driver"] = full["Driver"].astype(str)

KEY_COLS = ["Year", "Race", "Driver", "LapNumber"]
FF1_FEATURES = [
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
FF1_BINARY = {
    "ff1_sc_active",
    "ff1_vsc_active",
    "ff1_yellow_active",
    "ff1_neutral_just_ended",
    "ff1_cheap_stop_window",
    "ff1_rejoin_traffic_risk",
    "ff1_stream_gap_flag",
}
FF1_COLS = KEY_COLS + FF1_FEATURES


def empty_ff1_frame():
    return pd.DataFrame(columns=FF1_COLS)


def normalize_event_name(name):
    name = str(name)
    aliases = {
        "Sao Paulo Grand Prix": "S\u00e3o Paulo Grand Prix",
        "Mexico City Grand Prix": "Mexico City Grand Prix",
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


@contextmanager
def soft_timeout(seconds):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"FastF1 session load exceeded {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


try:
    import fastf1

    cache_dir = WORKING / "fastf1_cache"
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    try:
        fastf1.set_log_level("WARNING")
    except Exception:
        pass
except Exception as e:
    fastf1 = None
    print(f"FastF1 unavailable, using default ff1 features: {type(e).__name__}: {e}")


def fastf1_features_for_session(year, race):
    event_name = normalize_event_name(race)
    if fastf1 is None or event_name is None:
        return empty_ff1_frame()

    try:
        with soft_timeout(int(os.getenv("FASTF1_PER_SESSION_TIMEOUT_SECONDS", "45"))):
            session = fastf1.get_session(int(year), event_name, "R")
            session.load(laps=True, telemetry=False, weather=True, messages=True)

        laps = session.laps.copy()
        if (
            laps.empty
            or "Driver" not in laps.columns
            or "LapNumber" not in laps.columns
        ):
            return empty_ff1_frame()

        laps["LapNumber"] = pd.to_numeric(laps["LapNumber"], errors="coerce")
        laps = laps.dropna(subset=["LapNumber"]).copy()
        if laps.empty:
            return empty_ff1_frame()

        laps["LapNumber"] = laps["LapNumber"].astype(int)
        laps["Driver"] = laps["Driver"].astype(str)
        laps["lap_start_s"] = (
            laps["LapStartTime"].apply(to_seconds)
            if "LapStartTime" in laps.columns
            else np.nan
        )
        laps["pit_in_s"] = (
            laps["PitInTime"].apply(to_seconds)
            if "PitInTime" in laps.columns
            else np.nan
        )
        laps["pit_out_s"] = (
            laps["PitOutTime"].apply(to_seconds)
            if "PitOutTime" in laps.columns
            else np.nan
        )

        if "Position" in laps.columns and pd.Series(laps["lap_start_s"]).notna().any():
            laps["PositionNum"] = pd.to_numeric(laps["Position"], errors="coerce")
            leader = (
                laps.loc[laps["PositionNum"].eq(1)]
                .groupby("LapNumber", as_index=False)["lap_start_s"]
                .min()
                .rename(columns={"lap_start_s": "leader_lap_start_s"})
            )
            laps = laps.merge(leader, on="LapNumber", how="left")
            laps["ff1_gap_to_leader"] = laps["lap_start_s"] - laps["leader_lap_start_s"]

            ahead = laps[["LapNumber", "PositionNum", "lap_start_s"]].dropna().copy()
            ahead["PositionNum"] = ahead["PositionNum"] + 1
            ahead = (
                ahead.groupby(["LapNumber", "PositionNum"], as_index=False)[
                    "lap_start_s"
                ]
                .min()
                .rename(columns={"lap_start_s": "ahead_lap_start_s"})
            )
            laps = laps.merge(ahead, on=["LapNumber", "PositionNum"], how="left")
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

        status_laps = pd.DataFrame(
            {"LapNumber": np.sort(laps["LapNumber"].dropna().unique())}
        )
        for c in ["ff1_sc_active", "ff1_vsc_active", "ff1_yellow_active"]:
            status_laps[c] = 0

        if (
            hasattr(session, "track_status")
            and session.track_status is not None
            and len(session.track_status)
        ):
            ts = session.track_status.copy()
            if "Time" in ts.columns and "Status" in ts.columns:
                ts["time_s"] = ts["Time"].apply(to_seconds)
                lap_starts = (
                    laps.groupby("LapNumber")["lap_start_s"]
                    .median()
                    .dropna()
                    .sort_index()
                )
                for _, row in ts.iterrows():
                    if lap_starts.empty:
                        break
                    t = row.get("time_s", np.nan)
                    if pd.isna(t):
                        continue
                    pos = np.searchsorted(lap_starts.values, t, side="right") - 1
                    pos = max(0, min(pos, len(lap_starts.index) - 1))
                    lap_no = int(lap_starts.index[pos])
                    mask = status_laps["LapNumber"] >= lap_no
                    code = str(row.get("Status", ""))
                    if "1" in code:
                        status_laps.loc[
                            mask,
                            ["ff1_sc_active", "ff1_vsc_active", "ff1_yellow_active"],
                        ] = 0
                    if "4" in code:
                        status_laps.loc[mask, "ff1_sc_active"] = 1
                    if "6" in code or "7" in code:
                        status_laps.loc[mask, "ff1_vsc_active"] = 1
                    if "2" in code:
                        status_laps.loc[mask, "ff1_yellow_active"] = 1

        neutral = status_laps[
            ["ff1_sc_active", "ff1_vsc_active", "ff1_yellow_active"]
        ].max(axis=1)
        prev_neutral = neutral.shift(1).fillna(0).astype(int)
        status_laps["ff1_neutral_just_ended"] = (
            (prev_neutral == 1) & (neutral == 0)
        ).astype(int)
        last_neutral_lap = pd.Series(
            np.where(neutral.eq(1), status_laps["LapNumber"], np.nan)
        ).ffill()
        status_laps["ff1_laps_since_neutral"] = (
            status_laps["LapNumber"].to_numpy() - last_neutral_lap.to_numpy()
        )
        status_laps["ff1_laps_since_neutral"] = status_laps[
            "ff1_laps_since_neutral"
        ].fillna(99)

        laps = laps.merge(status_laps, on="LapNumber", how="left")

        if (
            hasattr(session, "weather_data")
            and session.weather_data is not None
            and len(session.weather_data)
        ):
            weather = session.weather_data.copy()
            if "Time" in weather.columns:
                weather["weather_time_s"] = weather["Time"].apply(to_seconds)
                weather = (
                    weather.dropna(subset=["weather_time_s"])
                    .sort_values("weather_time_s")
                    .copy()
                )
                lap_times = (
                    laps.groupby("LapNumber", as_index=False)["lap_start_s"]
                    .median()
                    .dropna()
                    .sort_values("lap_start_s")
                )
                present = [
                    c
                    for c in ["AirTemp", "TrackTemp", "Rainfall"]
                    if c in weather.columns
                ]
                if not weather.empty and not lap_times.empty:
                    aligned = pd.merge_asof(
                        lap_times,
                        weather[["weather_time_s"] + present],
                        left_on="lap_start_s",
                        right_on="weather_time_s",
                        direction="nearest",
                        tolerance=600,
                    ).rename(
                        columns={
                            "AirTemp": "ff1_air_temp",
                            "TrackTemp": "ff1_track_temp",
                            "Rainfall": "ff1_rainfall",
                        }
                    )
                    for c in ["ff1_air_temp", "ff1_track_temp", "ff1_rainfall"]:
                        if c not in aligned.columns:
                            aligned[c] = np.nan
                        aligned[c] = pd.to_numeric(aligned[c], errors="coerce")
                    aligned["ff1_track_temp_delta"] = (
                        aligned["ff1_track_temp"].diff().fillna(0)
                    )
                    laps = laps.merge(
                        aligned[
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
                    for c in [
                        "ff1_air_temp",
                        "ff1_track_temp",
                        "ff1_rainfall",
                        "ff1_track_temp_delta",
                    ]:
                        laps[c] = np.nan
            else:
                for c in [
                    "ff1_air_temp",
                    "ff1_track_temp",
                    "ff1_rainfall",
                    "ff1_track_temp_delta",
                ]:
                    laps[c] = np.nan
        else:
            for c in [
                "ff1_air_temp",
                "ff1_track_temp",
                "ff1_rainfall",
                "ff1_track_temp_delta",
            ]:
                laps[c] = np.nan

        laps["ff1_cheap_stop_window"] = (
            laps["ff1_sc_active"].fillna(0).astype(int)
            | laps["ff1_vsc_active"].fillna(0).astype(int)
            | laps["ff1_neutral_just_ended"].fillna(0).astype(int)
        ).astype(int)
        laps["ff1_rejoin_traffic_risk"] = (
            (pd.to_numeric(laps["ff1_interval_ahead"], errors="coerce").abs() < 2.0)
            | (pd.to_numeric(laps["ff1_gap_to_leader"], errors="coerce").abs() < 4.0)
        ).astype(int)
        laps["ff1_stream_gap_flag"] = (
            laps["ff1_gap_to_leader"].isna() | laps["ff1_interval_ahead"].isna()
        ).astype(int)

        out = laps[["Driver", "LapNumber"] + FF1_FEATURES].copy()
        out["Year"] = int(year)
        out["Race"] = str(race)
        out = out[FF1_COLS].drop_duplicates(subset=KEY_COLS, keep="first")
        return out

    except Exception as e:
        print(
            f"FastF1 load skipped for {year} {race}: {type(e).__name__}: {str(e)[:120]}"
        )
        return empty_ff1_frame()


ff1_parts = []
if fastf1 is not None:
    start = time.time()
    budget = float(os.getenv("FASTF1_TOTAL_BUDGET_SECONDS", "900"))
    sessions = full[["Year", "Race"]].drop_duplicates().sort_values(["Year", "Race"])
    for sess in sessions.itertuples(index=False):
        if time.time() - start > budget:
            print(
                f"FastF1 total budget reached after {len(ff1_parts)} sessions; using defaults for remaining sessions"
            )
            break
        ff1_parts.append(fastf1_features_for_session(sess.Year, sess.Race))

ff1_nonempty = [p for p in ff1_parts if len(p)]
if ff1_nonempty:
    ff1 = pd.concat(ff1_nonempty, ignore_index=True)
    ff1 = ff1.dropna(subset=KEY_COLS).copy()
    ff1["Year"] = pd.to_numeric(ff1["Year"], errors="coerce").astype(int)
    ff1["LapNumber"] = pd.to_numeric(ff1["LapNumber"], errors="coerce").astype(int)
    ff1["Race"] = ff1["Race"].astype(str)
    ff1["Driver"] = ff1["Driver"].astype(str)
    ff1 = ff1.drop_duplicates(subset=KEY_COLS, keep="first")
    before_rows = len(full)
    full = full.merge(ff1, on=KEY_COLS, how="left")
    print(f"fastf1_feature_rows={len(ff1)}")
    if len(full) != before_rows:
        print(f"warning_full_rows_changed_after_ff1_merge={before_rows}->{len(full)}")
else:
    for c in FF1_FEATURES:
        full[c] = np.nan
    print("fastf1_feature_rows=0")

for c in FF1_FEATURES:
    if c not in full.columns:
        full[c] = np.nan
    if c in FF1_BINARY:
        full[c] = pd.to_numeric(full[c], errors="coerce").fillna(0).astype(np.int8)
    else:
        full[c] = (
            pd.to_numeric(full[c], errors="coerce").fillna(-999.0).astype(np.float32)
        )

full["race_driver"] = full["Race"].astype(str) + "_" + full["Driver"].astype(str)
full["compound_stint"] = full["Compound"].astype(str) + "_" + full["Stint"].astype(str)
full["lap_x_tyre"] = full["LapNumber"] * full["TyreLife"]
full["race_progress_left"] = 1.0 - full["RaceProgress"]
full["tyre_life_ratio"] = full["TyreLife"] / np.maximum(full["LapNumber"], 1)
full["degradation_per_tyre_lap"] = full["Cumulative_Degradation"] / np.maximum(
    full["TyreLife"], 1
)

ordered = full.sort_values(["_is_train", "Year", "Race", "Driver", "LapNumber"]).copy()
grp_cols = ["_is_train", "Year", "Race", "Driver"]
full["pitstop_prev_lap_proxy"] = 0.0
full["pitstop_cum_proxy"] = 0.0
full.loc[ordered.index, "pitstop_prev_lap_proxy"] = (
    ordered.groupby(grp_cols)["PitStop"].shift(1).fillna(0).to_numpy()
)
full.loc[ordered.index, "pitstop_cum_proxy"] = (
    ordered.groupby(grp_cols)["PitStop"].cumsum().to_numpy()
    - ordered["PitStop"].to_numpy()
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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(X_all), dtype=float)
test_pred = np.zeros(len(X_test), dtype=float)
aucs = []

try:
    from catboost import CatBoostClassifier, Pool

    test_pool = Pool(X_test, cat_features=cat_cols)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y), 1):
        train_pool = Pool(X_all.iloc[tr_idx], y.iloc[tr_idx], cat_features=cat_cols)
        valid_pool = Pool(X_all.iloc[va_idx], y.iloc[va_idx], cat_features=cat_cols)
        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=800,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=8,
            random_seed=RANDOM_STATE + fold,
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=80,
            thread_count=max(1, os.cpu_count() or 1),
        )
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        oof[va_idx] = model.predict_proba(valid_pool)[:, 1]
        test_pred += model.predict_proba(test_pool)[:, 1] / skf.n_splits
        auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        aucs.append(auc)
        print(f"fold {fold} roc_auc={auc:.6f}")

except Exception as e:
    print(
        f"CatBoost unavailable or failed, using HistGradientBoosting fallback: {type(e).__name__}: {e}"
    )

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OrdinalEncoder

    def make_fallback(seed):
        pre = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    cat_cols,
                ),
                ("num", SimpleImputer(strategy="median"), num_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return make_pipeline(
            pre,
            HistGradientBoostingClassifier(
                learning_rate=0.055,
                max_leaf_nodes=31,
                max_iter=350,
                l2_regularization=0.05,
                random_state=seed,
            ),
        )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_all, y), 1):
        clf = make_fallback(RANDOM_STATE + fold)
        y_tr = y.iloc[tr_idx]
        pos_rate = max(y_tr.mean(), 1e-6)
        weights = np.where(y_tr.to_numpy() == 1, 0.5 / pos_rate, 0.5 / (1.0 - pos_rate))
        try:
            clf.fit(
                X_all.iloc[tr_idx],
                y_tr,
                histgradientboostingclassifier__sample_weight=weights,
            )
        except TypeError:
            clf.fit(X_all.iloc[tr_idx], y_tr)
        oof[va_idx] = clf.predict_proba(X_all.iloc[va_idx])[:, 1]
        test_pred += clf.predict_proba(X_test)[:, 1] / skf.n_splits
        auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        aucs.append(auc)
        print(f"fold {fold} roc_auc={auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"mean_fold_roc_auc={np.mean(aucs):.6f}")
print(f"oof_roc_auc={cv_auc:.6f}")

pd.DataFrame(
    {"row": np.arange(len(train)), "target": y, "prediction": np.clip(oof, 0, 1)}
).to_csv(WORKING / "oof_predictions.csv.gz", index=False, compression="gzip")

test_predictions = sample[[ID]].copy()
test_predictions[TARGET] = np.clip(test_pred, 0, 1)
test_predictions.to_csv(
    WORKING / "test_predictions.csv.gz", index=False, compression="gzip"
)
test_predictions.to_csv(WORKING / "submission.csv", index=False)

print(f"saved_submission={WORKING / 'submission.csv'}")
