import os
import re
import json
import time
import signal
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

RANDOM_STATE = 42
DRY = {"SOFT", "MEDIUM", "HARD"}
WET = {"INTERMEDIATE", "WET"}


@contextmanager
def time_limit(seconds):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("operation timed out")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def race_candidates(race):
    race = str(race)
    if "Pre-Season" in race:
        return []
    base = race.replace(" Grand Prix", "").strip()
    special = {
        "Mexico City Grand Prix": ["Mexico City", "Mexico"],
        "São Paulo Grand Prix": ["São Paulo", "Sao Paulo", "Brazil"],
        "Emilia Romagna Grand Prix": ["Emilia Romagna", "Imola"],
        "British Grand Prix": ["Great Britain", "Silverstone"],
        "United States Grand Prix": ["United States", "Austin"],
    }
    out = [race, base] + special.get(race, [])
    return list(dict.fromkeys([x for x in out if x]))


def status_neutral_flags(status_series):
    s = status_series.fillna("1").astype(str)
    sc = s.str.contains("4", regex=False).astype(int)
    vsc = s.str.contains("6|7", regex=True).astype(int)
    return sc, vsc, ((sc + vsc) > 0).astype(int)


def fetch_fastf1_one(year, race):
    import fastf1

    last_error = None
    for gp in race_candidates(race):
        try:
            sess = fastf1.get_session(int(year), gp, "R")
            try:
                sess.load(laps=True, weather=True, messages=True, telemetry=False)
            except TypeError:
                sess.load()
            laps = sess.laps.copy()
            if laps.empty or "LapNumber" not in laps or "LapStartTime" not in laps:
                continue

            lap_starts = laps[["LapNumber", "LapStartTime"]].dropna().copy()
            lap_starts["LapNumber"] = lap_starts["LapNumber"].astype(int)
            lap_starts["lap_time_ns"] = pd.to_timedelta(
                lap_starts["LapStartTime"]
            ).astype("int64")
            feat = lap_starts.groupby("LapNumber", as_index=False)[
                "lap_time_ns"
            ].median()
            feat = feat.sort_values("lap_time_ns").reset_index(drop=True)
            feat["ff_join_available"] = 1

            weather = getattr(sess, "weather_data", pd.DataFrame()).copy()
            if not weather.empty and "Time" in weather.columns:
                weather = weather.dropna(subset=["Time"]).copy()
                weather["time_ns"] = pd.to_timedelta(weather["Time"]).astype("int64")
                keep = [
                    c
                    for c in [
                        "Rainfall",
                        "TrackTemp",
                        "AirTemp",
                        "WindSpeed",
                        "Humidity",
                    ]
                    if c in weather.columns
                ]
                w = weather[["time_ns"] + keep].sort_values("time_ns")
                feat = pd.merge_asof(
                    feat.sort_values("lap_time_ns"),
                    w,
                    left_on="lap_time_ns",
                    right_on="time_ns",
                    direction="backward",
                )
                rename = {
                    "Rainfall": "ff_rainfall",
                    "TrackTemp": "ff_track_temp",
                    "AirTemp": "ff_air_temp",
                    "WindSpeed": "ff_wind_speed",
                    "Humidity": "ff_humidity",
                }
                feat = feat.rename(columns=rename).drop(
                    columns=[c for c in ["time_ns"] if c in feat.columns]
                )
            for c in [
                "ff_rainfall",
                "ff_track_temp",
                "ff_air_temp",
                "ff_wind_speed",
                "ff_humidity",
            ]:
                if c not in feat.columns:
                    feat[c] = np.nan

            ts = getattr(sess, "track_status", pd.DataFrame()).copy()
            if not ts.empty and {"Time", "Status"}.issubset(ts.columns):
                ts = ts.dropna(subset=["Time"]).copy()
                ts["time_ns"] = pd.to_timedelta(ts["Time"]).astype("int64")
                t = ts[["time_ns", "Status"]].sort_values("time_ns")
                merged = pd.merge_asof(
                    feat[["LapNumber", "lap_time_ns"]].sort_values("lap_time_ns"),
                    t,
                    left_on="lap_time_ns",
                    right_on="time_ns",
                    direction="backward",
                )
                feat["ff_track_status_raw"] = (
                    merged["Status"].fillna("1").astype(str).values
                )
            else:
                feat["ff_track_status_raw"] = "1"

            feat["ff_sc"], feat["ff_vsc"], feat["ff_neutralized"] = (
                status_neutral_flags(feat["ff_track_status_raw"])
            )

            rcm = getattr(sess, "race_control_messages", pd.DataFrame()).copy()
            if not rcm.empty and {"Lap", "Message"}.issubset(rcm.columns):
                msg = rcm["Message"].fillna("").astype(str).str.upper()
                neutral_msg = msg.str.contains("SAFETY CAR|VSC", regex=True)
                lap_flags = rcm.loc[neutral_msg, "Lap"].dropna().astype(int).unique()
                feat.loc[feat["LapNumber"].isin(lap_flags), "ff_neutralized"] = 1

            feat["Year"] = int(year)
            feat["Race"] = race
            return feat.drop(columns=["lap_time_ns"], errors="ignore")
        except Exception as e:
            last_error = e
    return None


def load_fastf1_features(base):
    cache_path = os.path.join(WORK_DIR, "fastf1_lap_features.csv.gz")
    expected_cols = [
        "Year",
        "Race",
        "LapNumber",
        "ff_join_available",
        "ff_rainfall",
        "ff_track_temp",
        "ff_air_temp",
        "ff_wind_speed",
        "ff_humidity",
        "ff_track_status_raw",
        "ff_sc",
        "ff_vsc",
        "ff_neutralized",
    ]
    if os.path.exists(cache_path):
        ff = pd.read_csv(cache_path)
        return ff[[c for c in expected_cols if c in ff.columns]]

    if os.environ.get("USE_FASTF1", "1").lower() not in {"1", "true", "yes"}:
        return pd.DataFrame(columns=expected_cols)

    try:
        import fastf1

        cache_dir = os.path.join(WORK_DIR, "fastf1_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            fastf1.Cache.enable_cache(cache_dir)
        except Exception:
            pass
    except Exception:
        return pd.DataFrame(columns=expected_cols)

    sessions = base[["Year", "Race"]].drop_duplicates().sort_values(["Year", "Race"])
    max_sessions = int(os.environ.get("FASTF1_MAX_SESSIONS", "999"))
    budget = float(os.environ.get("FASTF1_TIME_BUDGET_SEC", "360"))
    per_session = float(os.environ.get("FASTF1_SESSION_TIMEOUT_SEC", "25"))
    deadline = time.time() + budget

    frames = []
    for _, row in sessions.head(max_sessions).iterrows():
        if time.time() > deadline:
            break
        try:
            with time_limit(per_session):
                one = fetch_fastf1_one(row["Year"], row["Race"])
            if one is not None and not one.empty:
                frames.append(one)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=expected_cols)

    ff = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["Year", "Race", "LapNumber"]
    )
    ff.to_csv(cache_path, index=False, compression="gzip")
    return ff[[c for c in expected_cols if c in ff.columns]]


def add_expert_features(train, test):
    train = train.copy()
    test = test.copy()
    train["_is_train"] = 1
    test["_is_train"] = 0
    test["PitNextLap"] = np.nan

    df = pd.concat([train, test], ignore_index=True, sort=False)
    df["Compound"] = df["Compound"].astype(str)
    df["Driver"] = df["Driver"].astype(str)
    df["Race"] = df["Race"].astype(str)

    rp = df["RaceProgress"].clip(0.005, 1.0)
    row_total = (df["LapNumber"] / rp).replace([np.inf, -np.inf], np.nan)
    row_total = np.maximum(
        row_total.fillna(df["LapNumber"]).values, df["LapNumber"].values
    )
    df["_row_total_est"] = np.clip(row_total, df["LapNumber"].values, 120)
    session_total = (
        df.groupby(["Year", "Race"])["_row_total_est"].transform("median").round()
    )
    session_max = df.groupby(["Year", "Race"])["LapNumber"].transform("max")
    df["EstimatedTotalLaps"] = np.maximum(
        session_total.fillna(session_max), session_max
    )
    df["LapsRemaining"] = (df["EstimatedTotalLaps"] - df["LapNumber"]).clip(lower=0)
    df["LapFraction"] = df["LapNumber"] / df["EstimatedTotalLaps"].clip(lower=1)

    ff = load_fastf1_features(df)
    if not ff.empty:
        df = df.merge(ff, on=["Year", "Race", "LapNumber"], how="left")

    for c in ["ff_join_available", "ff_sc", "ff_vsc", "ff_neutralized"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)
    for c in [
        "ff_rainfall",
        "ff_track_temp",
        "ff_air_temp",
        "ff_wind_speed",
        "ff_humidity",
    ]:
        if c not in df.columns:
            df[c] = np.nan
    if "ff_track_status_raw" not in df.columns:
        df["ff_track_status_raw"] = "unknown"
    df["ff_track_status_raw"] = df["ff_track_status_raw"].fillna("unknown").astype(str)

    rain = df["ff_rainfall"]
    if rain.dtype == bool:
        rain = rain.astype(float)
    else:
        rain = rain.replace(
            {True: 1, False: 0, "True": 1, "False": 0, "TRUE": 1, "FALSE": 0}
        )
        rain = pd.to_numeric(rain, errors="coerce")
    df["ff_rainfall"] = rain

    df["is_dry_compound"] = df["Compound"].isin(DRY).astype(int)
    df["is_wet_compound"] = df["Compound"].isin(WET).astype(int)
    df["rain_or_wet"] = (
        (df["ff_rainfall"].fillna(0) > 0) | (df["is_wet_compound"] == 1)
    ).astype(int)

    race_lap = df.groupby(["Year", "Race", "LapNumber"], as_index=False).agg(
        race_lap_delta_median=("LapTime_Delta", "median"),
        race_lap_time_median=("LapTime (s)", "median"),
        race_pit_rate=("PitStop", "mean"),
    )
    race_lap["delta_rank"] = race_lap.groupby(["Year", "Race"])[
        "race_lap_delta_median"
    ].rank(pct=True)
    race_lap["proxy_neutralized"] = (
        ((race_lap["race_lap_delta_median"] > 8) & (race_lap["delta_rank"] > 0.90))
        | (race_lap["race_lap_delta_median"] > 20)
    ).astype(int)
    df = df.merge(race_lap, on=["Year", "Race", "LapNumber"], how="left")
    df["track_neutralized"] = (
        (df["ff_neutralized"] == 1) | (df["proxy_neutralized"] == 1)
    ).astype(int)

    lap_state = (
        df.groupby(["Year", "Race", "LapNumber"], as_index=False)
        .agg(
            track_neutralized=("track_neutralized", "max"),
            ff_sc=("ff_sc", "max"),
            ff_vsc=("ff_vsc", "max"),
            rain_or_wet=("rain_or_wet", "max"),
        )
        .sort_values(["Year", "Race", "LapNumber"])
    )

    since = np.full(len(lap_state), 999, dtype=np.int16)
    for _, idx in lap_state.groupby(["Year", "Race"], sort=False).indices.items():
        last_onset = None
        prev = 0
        laps = lap_state.iloc[idx]["LapNumber"].values
        neut = lap_state.iloc[idx]["track_neutralized"].values
        for j, pos in enumerate(idx):
            if neut[j] == 1 and prev == 0:
                last_onset = laps[j]
            if last_onset is not None:
                since[pos] = int(laps[j] - last_onset)
            prev = int(neut[j])
    lap_state["laps_since_sc_vsc_onset"] = since
    lap_state["cheap_stop_window"] = (
        (lap_state["track_neutralized"] == 1)
        & (lap_state["laps_since_sc_vsc_onset"] <= 6)
    ).astype(int)
    lap_state["post_neutral_window"] = (
        (lap_state["track_neutralized"] == 0)
        & (lap_state["laps_since_sc_vsc_onset"] <= 2)
    ).astype(int)

    df = df.drop(
        columns=["ff_sc", "ff_vsc", "track_neutralized", "rain_or_wet"], errors="ignore"
    )
    df = df.merge(
        lap_state[
            [
                "Year",
                "Race",
                "LapNumber",
                "track_neutralized",
                "ff_sc",
                "ff_vsc",
                "rain_or_wet",
                "laps_since_sc_vsc_onset",
                "cheap_stop_window",
                "post_neutral_window",
            ]
        ],
        on=["Year", "Race", "LapNumber"],
        how="left",
    )

    df["MonacoRuleActive"] = (
        df["Race"].str.contains("Monaco", case=False, na=False) & (df["Year"] >= 2025)
    ).astype(int)

    df["_order"] = np.arange(len(df))
    df = df.sort_values(["Year", "Race", "Driver", "LapNumber", "id"]).reset_index(
        drop=True
    )

    n = len(df)
    dry_count = np.zeros(n, dtype=np.int8)
    wet_seen = np.zeros(n, dtype=np.int8)
    obligation_open = np.zeros(n, dtype=np.int8)
    pit_count = np.zeros(n, dtype=np.int8)
    laps_since_pit = np.zeros(n, dtype=np.float32)
    monaco_remaining = np.zeros(n, dtype=np.int8)
    mandatory_pressure = np.zeros(n, dtype=np.float32)
    cheap_obligation_gate = np.zeros(n, dtype=np.float32)

    compounds = df["Compound"].values
    pitstop = df["PitStop"].fillna(0).astype(int).values
    laps = df["LapNumber"].values
    laps_rem = df["LapsRemaining"].fillna(0).values
    rain_wet = df["rain_or_wet"].fillna(0).astype(int).values
    cheap = df["cheap_stop_window"].fillna(0).astype(int).values
    monaco = df["MonacoRuleActive"].values

    for _, idx in df.groupby(["Year", "Race", "Driver"], sort=False).indices.items():
        used_dry = set()
        seen_wet = False
        stops = 0
        last_pit = None
        for pos in idx:
            comp = compounds[pos]
            if comp in WET or rain_wet[pos] == 1:
                seen_wet = True
            if comp in DRY:
                used_dry.add(comp)
            if pitstop[pos] == 1:
                stops += 1
                last_pit = laps[pos]

            dry_count[pos] = len(used_dry)
            wet_seen[pos] = int(seen_wet)
            pit_count[pos] = stops
            laps_since_pit[pos] = (
                float(laps[pos] - last_pit)
                if last_pit is not None
                else float(laps[pos])
            )

            open_rule = int((not seen_wet) and (len(used_dry) < 2))
            obligation_open[pos] = open_rule
            monaco_remaining[pos] = max(0, 2 - stops) if monaco[pos] else 0

            pressure = max(0.0, 1.0 - min(float(laps_rem[pos]), 12.0) / 12.0)
            mandatory_pressure[pos] = pressure * float(
                open_rule + monaco_remaining[pos]
            )
            cheap_obligation_gate[pos] = float(cheap[pos]) * float(
                open_rule + monaco_remaining[pos] > 0
            )

    df["DryCompoundsUsedSoFar"] = dry_count
    df["WetExemptionSeen"] = wet_seen
    df["DryCompoundObligationOpen"] = obligation_open
    df["PitCountSoFar"] = pit_count
    df["LapsSincePit"] = laps_since_pit
    df["MonacoStopsRemaining"] = monaco_remaining
    df["MandatoryStopPressure"] = mandatory_pressure
    df["CheapStopObligationGate"] = cheap_obligation_gate

    df = (
        df.sort_values("_order")
        .drop(columns=["_order", "_row_total_est"])
        .reset_index(drop=True)
    )

    df["DegradationPerTyreLap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(
        lower=1
    )
    df["TyreLifeFrac"] = df["TyreLife"] / df["EstimatedTotalLaps"].clip(lower=1)
    df["StintProgress"] = df["TyreLife"] / df["LapNumber"].clip(lower=1)
    df["AbsLapTimeDelta"] = df["LapTime_Delta"].abs()
    df["PositionProgress"] = df["Position"] * df["RaceProgress"]
    df["SessionKey"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)
    df["YearCat"] = df["Year"].astype(str)

    return df


def make_split(train_df, y):
    groups = train_df["SessionKey"].astype(str).values
    for seed in [42, 7, 123, 2026]:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_idx, va_idx = next(splitter.split(train_df, y, groups))
        if y.iloc[va_idx].nunique() == 2:
            return tr_idx, va_idx
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    return next(splitter.split(train_df, y))


def fit_lgbm(X_tr, y_tr, X_va, y_va, cat_cols):
    import lightgbm as lgb

    params = dict(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=5.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
    )
    pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
    best_iter = int(model.best_iteration_ or params["n_estimators"])

    final_params = params.copy()
    final_params["n_estimators"] = best_iter
    final_model = lgb.LGBMClassifier(**final_params)
    return model, final_model, pred


def fit_catboost(X_tr, y_tr, X_va, y_va, cat_cols):
    from catboost import CatBoostClassifier, Pool

    for c in cat_cols:
        X_tr[c] = X_tr[c].astype(str)
        X_va[c] = X_va[c].astype(str)

    model = CatBoostClassifier(
        iterations=900,
        learning_rate=0.045,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        l2_leaf_reg=5,
        allow_writing_files=False,
        thread_count=max(1, min(8, os.cpu_count() or 1)),
        verbose=100,
    )
    model.fit(
        Pool(X_tr, y_tr, cat_features=cat_cols),
        eval_set=Pool(X_va, y_va, cat_features=cat_cols),
        early_stopping_rounds=100,
    )
    pred = model.predict_proba(Pool(X_va, cat_features=cat_cols))[:, 1]

    final_model = CatBoostClassifier(**model.get_params())
    final_model.set_params(
        iterations=int(model.get_best_iteration() or 600), verbose=False
    )
    return model, final_model, pred


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

full = add_expert_features(train, test)

cat_cols = [
    "Compound",
    "Driver",
    "Race",
    "SessionKey",
    "YearCat",
    "ff_track_status_raw",
]
for c in cat_cols:
    full[c] = full[c].fillna("missing").astype(str).astype("category")

drop_cols = {"id", "PitNextLap", "_is_train"}
feature_cols = [c for c in full.columns if c not in drop_cols]
for c in feature_cols:
    if full[c].dtype == "object" and c not in cat_cols:
        full[c] = full[c].fillna("missing").astype(str).astype("category")
        cat_cols.append(c)

train_fe = full[full["_is_train"] == 1].copy()
test_fe = full[full["_is_train"] == 0].copy()
y = train_fe["PitNextLap"].astype(int)

X = train_fe[feature_cols].copy()
X_test = test_fe[feature_cols].copy()
cat_cols = [c for c in cat_cols if c in feature_cols]

tr_idx, va_idx = make_split(train_fe, y)
X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

try:
    _, final_model, val_pred = fit_lgbm(X_tr, y_tr, X_va, y_va, cat_cols)
    final_model.fit(X, y, categorical_feature=cat_cols)
    test_pred = final_model.predict_proba(X_test)[:, 1]
    model_name = "lightgbm"
except Exception:
    _, final_model, val_pred = fit_catboost(X_tr, y_tr, X_va, y_va, cat_cols)
    for c in cat_cols:
        X[c] = X[c].astype(str)
        X_test[c] = X_test[c].astype(str)
    final_model.fit(X, y, cat_features=cat_cols)
    test_pred = final_model.predict_proba(X_test)[:, 1]
    model_name = "catboost"

val_pred = np.clip(val_pred, 1e-7, 1 - 1e-7)
test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)
auc = roc_auc_score(y_va, val_pred)

pd.DataFrame(
    {
        "row": train_fe.iloc[va_idx]["id"].values,
        "target": y_va.values,
        "prediction": val_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "validation_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

target_col = [c for c in sample.columns if c != "id"][0]
submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

fastf1_rows = (
    int(full["ff_join_available"].fillna(0).sum())
    if "ff_join_available" in full.columns
    else 0
)
print(f"holdout_roc_auc={auc:.6f}")
print(
    json.dumps(
        {
            "validation_metric": "roc_auc",
            "validation_roc_auc": float(auc),
            "model": model_name,
            "fastf1_joined_rows": fastf1_rows,
            "research_hypotheses_llm_claimed_used": ["000814"],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        },
        sort_keys=True,
    )
)
