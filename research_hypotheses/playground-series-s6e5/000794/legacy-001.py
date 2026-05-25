import os
import re
import json
import difflib
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
KEY_COLS = ["Year", "Race", "Driver", "LapNumber"]

WEATHER_RAW_COLS = [
    "AirTemp",
    "Humidity",
    "Pressure",
    "Rainfall",
    "TrackTemp",
    "WindDirection",
    "WindSpeed",
]
EXT_CAT_COLS = [
    "ff1_track_status_end",
    "ff1_track_status_transition",
    "ff1_session_status_end",
    "ff1_session_status_transition",
]
EXT_NUM_COLS = [
    "ff1_status_has_clear",
    "ff1_status_has_yellow",
    "ff1_status_has_sc",
    "ff1_status_has_vsc",
    "ff1_status_has_red",
    "ff1_status_interrupted",
    "ff1_track_status_changed",
    "ff1_session_status_changed",
    "ff1_laps_since_sc_vsc_start",
    "ff1_laps_since_sc_vsc_end",
    "ff1_rc_message_count",
    "ff1_rc_incident_count",
    "ff1_rc_sc_vsc_count",
    "ff1_rc_yellow_count",
    "ff1_rc_red_count",
    "ff1_rc_cum_message_count",
    "ff1_rc_cum_incident_count",
    "ff1_rc_cum_sc_vsc_count",
    "ff1_rc_cum_yellow_count",
    "ff1_rc_cum_red_count",
] + [f"ff1_weather_{c}" for c in WEATHER_RAW_COLS]


def read_data():
    train = pd.read_csv(INPUT / "train.csv.gz")
    test = pd.read_csv(INPUT / "test.csv.gz")
    sample = pd.read_csv(INPUT / "sample_submission.csv.gz")
    return train, test, sample


def norm_text(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", "and")
    s = re.sub(r"\b(grand prix|gp|formula 1)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def seconds_series(s):
    if s is None:
        return pd.Series(dtype="float64")
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    return pd.to_timedelta(s, errors="coerce").dt.total_seconds()


def timeline_value_at(lap_times, timeline, value_col, default_value):
    out = np.array([default_value] * len(lap_times), dtype=object)
    if (
        timeline is None
        or len(timeline) == 0
        or "Time" not in timeline.columns
        or value_col not in timeline.columns
    ):
        return pd.Series(out)

    tt = seconds_series(timeline["Time"])
    valid = tt.notna()
    if valid.sum() == 0:
        return pd.Series(out)

    t = tt[valid].to_numpy()
    vals = timeline.loc[valid, value_col].astype(str).to_numpy()
    order = np.argsort(t)
    t = t[order]
    vals = vals[order]

    lt = seconds_series(lap_times).to_numpy()
    idx = np.searchsorted(t, lt, side="right") - 1
    mask = (idx >= 0) & np.isfinite(lt)
    out[mask] = vals[idx[mask]]
    return pd.Series(out)


def status_has(status, codes):
    ss = str(status)
    return int(any(code in ss for code in codes))


def laps_since(laps, events):
    last = None
    out = []
    for lap, event in zip(laps, events):
        if bool(event):
            last = lap
        out.append(999.0 if last is None else float(max(0, lap - last)))
    return out


def latest_weather_by_time(lap_times, weather):
    result = {
        c: np.full(len(lap_times), np.nan, dtype="float64") for c in WEATHER_RAW_COLS
    }
    if weather is None or len(weather) == 0 or "Time" not in weather.columns:
        return result

    wt = seconds_series(weather["Time"])
    valid = wt.notna()
    if valid.sum() == 0:
        return result

    t = wt[valid].to_numpy()
    order = np.argsort(t)
    t = t[order]
    lt = seconds_series(lap_times).to_numpy()
    idx = np.searchsorted(t, lt, side="right") - 1
    mask = (idx >= 0) & np.isfinite(lt)

    for col in WEATHER_RAW_COLS:
        if col not in weather.columns:
            continue
        raw = weather.loc[valid, col]
        if col == "Rainfall":
            vals = (
                raw.astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
                .astype(float)
                .to_numpy()
            )
        else:
            vals = pd.to_numeric(raw, errors="coerce").to_numpy()
        vals = vals[order]
        result[col][mask] = vals[idx[mask]]
    return result


def race_control_lap_counts(session, max_lap):
    cols = [
        "ff1_rc_message_count",
        "ff1_rc_incident_count",
        "ff1_rc_sc_vsc_count",
        "ff1_rc_yellow_count",
        "ff1_rc_red_count",
    ]
    base = pd.DataFrame({"LapNumber": np.arange(1, max_lap + 1, dtype=int)})
    for c in cols:
        base[c] = 0.0

    rc = getattr(session, "race_control_messages", None)
    if rc is None or len(rc) == 0 or "Lap" not in rc.columns:
        for c in cols:
            base[c.replace("ff1_rc_", "ff1_rc_cum_")] = base[c].cumsum()
        return base

    tmp = rc.copy()
    tmp["LapNumber"] = pd.to_numeric(tmp["Lap"], errors="coerce")
    tmp = tmp[tmp["LapNumber"].notna()].copy()
    if len(tmp) == 0:
        for c in cols:
            base[c.replace("ff1_rc_", "ff1_rc_cum_")] = base[c].cumsum()
        return base

    tmp["LapNumber"] = tmp["LapNumber"].astype(int)
    text = tmp.astype(str).agg(" ".join, axis=1).str.lower()
    tmp["ff1_rc_message_count"] = 1.0
    tmp["ff1_rc_incident_count"] = text.str.contains(
        r"incident|investigat|noted|collision|crash|stopped|debris|summon", regex=True
    ).astype(float)
    tmp["ff1_rc_sc_vsc_count"] = text.str.contains(
        r"safety car|virtual safety car|\bvsc\b", regex=True
    ).astype(float)
    tmp["ff1_rc_yellow_count"] = text.str.contains("yellow", regex=False).astype(float)
    tmp["ff1_rc_red_count"] = text.str.contains("red flag", regex=False).astype(float)

    agg = tmp.groupby("LapNumber", as_index=False)[cols].sum()
    base = base.drop(columns=cols).merge(agg, on="LapNumber", how="left")
    base[cols] = base[cols].fillna(0.0)
    for c in cols:
        base[c.replace("ff1_rc_", "ff1_rc_cum_")] = base[c].cumsum()
    return base


_schedule_cache = {}


def resolve_fastf1_round(fastf1, year, race):
    if "testing" in str(race).lower():
        return None

    year = int(year)
    if year not in _schedule_cache:
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except TypeError:
            sched = fastf1.get_event_schedule(year)
        _schedule_cache[year] = sched

    sched = _schedule_cache[year]
    target = norm_text(race)
    best_score, best_round = -1.0, None

    for _, row in sched.iterrows():
        names = []
        for col in ["EventName", "OfficialEventName", "Location", "Country"]:
            if col in sched.columns and pd.notna(row.get(col)):
                names.append(str(row[col]))

        for name in names:
            cand = norm_text(name)
            if not cand:
                continue
            score = difflib.SequenceMatcher(None, target, cand).ratio()
            if target in cand or cand in target:
                score = max(score, 0.95)
            if score > best_score:
                best_score = score
                best_round = int(row["RoundNumber"])

    return best_round if best_score >= 0.55 else None


def load_fastf1_session(session):
    signatures = [
        dict(laps=True, telemetry=False, weather=True, messages=True),
        dict(telemetry=False, weather=True, messages=True),
        dict(telemetry=False, weather=True),
        dict(telemetry=False),
    ]
    for kwargs in signatures:
        try:
            session.load(**kwargs)
            return
        except TypeError:
            continue
    session.load()


def extract_session_features(session, year, race):
    laps = session.laps.copy()
    if (
        laps is None
        or len(laps) == 0
        or "Driver" not in laps.columns
        or "LapNumber" not in laps.columns
    ):
        return pd.DataFrame()

    laps = laps[laps["Driver"].notna() & laps["LapNumber"].notna()].copy()
    if len(laps) == 0:
        return pd.DataFrame()

    laps["Driver"] = laps["Driver"].astype(str)
    laps["LapNumber"] = (
        pd.to_numeric(laps["LapNumber"], errors="coerce").round().astype(int)
    )
    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    feat = laps[["Driver", "LapNumber"]].copy()
    feat["Year"] = int(year)
    feat["Race"] = str(race)

    lap_times = (
        laps["Time"] if "Time" in laps.columns else pd.Series([np.nan] * len(laps))
    )
    fallback_track = (
        laps["TrackStatus"].astype(str).reset_index(drop=True)
        if "TrackStatus" in laps.columns
        else pd.Series(["1"] * len(laps))
    )
    track_timeline = getattr(session, "track_status", None)
    session_timeline = getattr(session, "session_status", None)

    track_status = timeline_value_at(
        lap_times, track_timeline, "Status", "Unknown"
    ).reset_index(drop=True)
    track_status = track_status.mask(
        track_status.isin(["Unknown", "nan", "NaT"]), fallback_track
    )
    feat["ff1_track_status_end"] = track_status.astype(str)

    sess_status = timeline_value_at(
        lap_times, session_timeline, "Status", "Unknown"
    ).reset_index(drop=True)
    feat["ff1_session_status_end"] = sess_status.astype(str).replace(
        {"nan": "Unknown", "NaT": "Unknown"}
    )

    weather_added = False
    try:
        w = laps.get_weather_data().reset_index(drop=True)
        if len(w) == len(laps):
            for col in WEATHER_RAW_COLS:
                if col in w.columns:
                    if col == "Rainfall":
                        feat[f"ff1_weather_{col}"] = (
                            w[col]
                            .astype(str)
                            .str.lower()
                            .isin(["true", "1", "yes"])
                            .astype(float)
                        )
                    else:
                        feat[f"ff1_weather_{col}"] = pd.to_numeric(
                            w[col], errors="coerce"
                        )
                else:
                    feat[f"ff1_weather_{col}"] = np.nan
            weather_added = True
    except Exception:
        weather_added = False

    if not weather_added:
        weather = getattr(session, "weather_data", None)
        latest = latest_weather_by_time(lap_times, weather)
        for col in WEATHER_RAW_COLS:
            feat[f"ff1_weather_{col}"] = latest[col]

    feat["ff1_status_has_clear"] = feat["ff1_track_status_end"].map(
        lambda x: status_has(x, ["1"])
    )
    feat["ff1_status_has_yellow"] = feat["ff1_track_status_end"].map(
        lambda x: status_has(x, ["2", "3"])
    )
    feat["ff1_status_has_sc"] = feat["ff1_track_status_end"].map(
        lambda x: status_has(x, ["4"])
    )
    feat["ff1_status_has_vsc"] = feat["ff1_track_status_end"].map(
        lambda x: status_has(x, ["6", "7"])
    )
    feat["ff1_status_has_red"] = feat["ff1_track_status_end"].map(
        lambda x: status_has(x, ["5"])
    )
    feat["ff1_status_interrupted"] = (
        feat["ff1_status_has_sc"]
        | feat["ff1_status_has_vsc"]
        | feat["ff1_status_has_red"]
    ).astype(int)

    max_lap = int(feat["LapNumber"].max())
    rc_counts = race_control_lap_counts(session, max_lap)
    feat = feat.merge(rc_counts, on="LapNumber", how="left")

    feat = feat.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)
    prev_track = (
        feat.groupby("Driver")["ff1_track_status_end"]
        .shift(1)
        .fillna(feat["ff1_track_status_end"])
    )
    prev_sess = (
        feat.groupby("Driver")["ff1_session_status_end"]
        .shift(1)
        .fillna(feat["ff1_session_status_end"])
    )

    feat["ff1_track_status_transition"] = (
        prev_track.astype(str) + "->" + feat["ff1_track_status_end"].astype(str)
    )
    feat["ff1_session_status_transition"] = (
        prev_sess.astype(str) + "->" + feat["ff1_session_status_end"].astype(str)
    )
    feat["ff1_track_status_changed"] = (
        prev_track.astype(str) != feat["ff1_track_status_end"].astype(str)
    ).astype(int)
    feat["ff1_session_status_changed"] = (
        prev_sess.astype(str) != feat["ff1_session_status_end"].astype(str)
    ).astype(int)

    def add_lags(g):
        interrupted = (
            g["ff1_status_has_sc"].eq(1) | g["ff1_status_has_vsc"].eq(1)
        ).to_numpy()
        prev = np.r_[False, interrupted[:-1]]
        starts = interrupted & ~prev
        ends = (~interrupted) & prev
        lap_arr = g["LapNumber"].to_numpy()
        g["ff1_laps_since_sc_vsc_start"] = laps_since(lap_arr, starts)
        g["ff1_laps_since_sc_vsc_end"] = laps_since(lap_arr, ends)
        return g

    feat = feat.groupby("Driver", group_keys=False).apply(add_lags)
    for c in EXT_NUM_COLS:
        if c not in feat.columns:
            feat[c] = np.nan
    for c in EXT_CAT_COLS:
        if c not in feat.columns:
            feat[c] = "Unknown"

    feat = feat[KEY_COLS + EXT_NUM_COLS + EXT_CAT_COLS]
    feat = feat.drop_duplicates(KEY_COLS, keep="last")
    return feat


def build_fastf1_features(train, test):
    cache_path = WORK / "fastf1_causal_features.csv.gz"
    info = {
        "fastf1_sessions_loaded": 0,
        "fastf1_sessions_failed": 0,
        "fastf1_cache_used": False,
    }

    if cache_path.exists():
        try:
            ext = pd.read_csv(cache_path)
            info["fastf1_cache_used"] = True
            return ext, info
        except Exception:
            pass

    try:
        import fastf1

        try:
            fastf1.set_log_level("WARNING")
        except Exception:
            pass
        try:
            fastf1.Cache.enable_cache(str(WORK / "fastf1_cache"))
        except Exception:
            pass
    except Exception as e:
        print(
            f"FastF1 unavailable, external features will be missing: {type(e).__name__}"
        )
        return pd.DataFrame(columns=KEY_COLS + EXT_NUM_COLS + EXT_CAT_COLS), info

    sessions = pd.concat(
        [train[["Year", "Race"]], test[["Year", "Race"]]], ignore_index=True
    ).drop_duplicates()
    frames = []

    for _, row in sessions.iterrows():
        year, race = int(row["Year"]), str(row["Race"])
        try:
            rnd = resolve_fastf1_round(fastf1, year, race)
            if rnd is None:
                info["fastf1_sessions_failed"] += 1
                continue
            session = fastf1.get_session(year, rnd, "R")
            load_fastf1_session(session)
            sf = extract_session_features(session, year, race)
            if len(sf):
                frames.append(sf)
                info["fastf1_sessions_loaded"] += 1
            else:
                info["fastf1_sessions_failed"] += 1
        except Exception:
            info["fastf1_sessions_failed"] += 1

    if frames:
        ext = pd.concat(frames, ignore_index=True).drop_duplicates(
            KEY_COLS, keep="last"
        )
    else:
        ext = pd.DataFrame(columns=KEY_COLS + EXT_NUM_COLS + EXT_CAT_COLS)

    try:
        ext.to_csv(cache_path, index=False, compression="gzip")
    except Exception:
        pass

    return ext, info


def attach_external_features(train, test, ext):
    if ext is None or len(ext) == 0:
        ext = pd.DataFrame(columns=KEY_COLS + EXT_NUM_COLS + EXT_CAT_COLS)

    for c in EXT_NUM_COLS:
        if c not in ext.columns:
            ext[c] = np.nan
    for c in EXT_CAT_COLS:
        if c not in ext.columns:
            ext[c] = "Unknown"

    train = train.merge(
        ext[KEY_COLS + EXT_NUM_COLS + EXT_CAT_COLS], on=KEY_COLS, how="left"
    )
    test = test.merge(
        ext[KEY_COLS + EXT_NUM_COLS + EXT_CAT_COLS], on=KEY_COLS, how="left"
    )

    for df in [train, test]:
        for c in EXT_NUM_COLS:
            df[c] = (
                pd.to_numeric(df[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(-1.0)
            )
        for c in EXT_CAT_COLS:
            df[c] = df[c].fillna("Unknown").astype(str)
    return train, test


def prepare_matrix(train, test):
    features = [c for c in train.columns if c not in [TARGET, ID_COL]]
    cat_cols = [
        c for c in features if train[c].dtype == "object" or test[c].dtype == "object"
    ]
    for c in ["Race", "Driver", "Compound"] + EXT_CAT_COLS:
        if c in features and c not in cat_cols:
            cat_cols.append(c)

    num_cols = [c for c in features if c not in cat_cols]
    for c in num_cols:
        train[c] = (
            pd.to_numeric(train[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(-1.0)
        )
        test[c] = (
            pd.to_numeric(test[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(-1.0)
        )

    for c in cat_cols:
        tr = train[c].fillna("__missing__").astype(str)
        te = test[c].fillna("__missing__").astype(str)
        cats = pd.Index(pd.concat([tr, te], ignore_index=True).unique())
        train[c] = pd.Categorical(tr, categories=cats)
        test[c] = pd.Categorical(te, categories=cats)

    return train[features], test[features], features, cat_cols


def main():
    train, test, sample = read_data()
    y = train[TARGET].astype(int).to_numpy()

    ext, ext_info = build_fastf1_features(train, test)
    train, test = attach_external_features(train, test, ext)
    X, X_test, features, cat_cols = prepare_matrix(train, test)

    groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(X, y, groups))

    oof = np.zeros(len(train), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        model = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=1800,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=60,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=2026 + fold,
            n_jobs=max(1, os.cpu_count() or 1),
            verbosity=-1,
            force_col_wise=True,
        )

        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / len(splits)
        fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
        fold_scores.append(float(fold_auc))
        print(f"fold {fold} roc_auc={fold_auc:.6f}")

    auc = roc_auc_score(y, oof)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    submission = sample.copy()
    submission[TARGET] = test_pred
    submission.to_csv(WORK / "submission.csv", index=False)

    submission.to_csv(WORK / "test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"row": np.arange(len(train)), "target": y, "prediction": oof}).to_csv(
        WORK / "oof_predictions.csv.gz", index=False, compression="gzip"
    )

    result = {
        "metric": "roc_auc",
        "validation_auc": float(auc),
        "fold_auc": fold_scores,
        "research_hypotheses_llm_claimed_used": ["000794"],
        "causal_alignment_audit": "FastF1 track/session/weather timelines use Time <= lap end; race-control cumulative counts use Lap <= current LapNumber.",
        **ext_info,
    }
    print(f"OOF ROC AUC: {auc:.6f}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
