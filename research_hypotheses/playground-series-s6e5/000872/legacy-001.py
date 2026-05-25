import os
import re
import json
import warnings
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

SEED = 42
INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"

STATUS_COLS = [
    "fastf1_status_available",
    "track_status_code",
    "track_is_sc_or_vsc",
    "track_is_yellow",
    "track_is_red_flag",
    "track_event_sc_or_vsc",
    "track_event_red_flag",
    "laps_since_status_change",
    "laps_since_sc_or_vsc",
    "first_green_lap_after_sc",
    "red_flag_restart_window",
    "rc_sc_message",
    "rc_vsc_message",
    "rc_red_flag_message",
    "rc_wet_message",
    "recent_wet_message",
    "laps_since_wet_message",
    "session_pause_message",
    "recent_session_pause",
]


def norm_ascii(s):
    return (
        unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    )


def fastf1_event_candidates(race):
    race = str(race)
    if "testing" in race.lower():
        return []

    aliases = {
        "British Grand Prix": ["British Grand Prix", "Great Britain", "Silverstone"],
        "Emilia Romagna Grand Prix": [
            "Emilia Romagna Grand Prix",
            "Emilia Romagna",
            "Imola",
        ],
        "Sao Paulo Grand Prix": ["Sao Paulo Grand Prix", "Brazil", "Sao Paulo"],
        "São Paulo Grand Prix": ["Sao Paulo Grand Prix", "Brazil", "Sao Paulo"],
        "Mexico City Grand Prix": ["Mexico City Grand Prix", "Mexico", "Mexico City"],
        "United States Grand Prix": [
            "United States Grand Prix",
            "United States",
            "Austin",
        ],
        "Austrian Grand Prix": ["Austrian Grand Prix", "Austria", "Spielberg"],
        "Belgian Grand Prix": ["Belgian Grand Prix", "Belgium", "Spa"],
        "Dutch Grand Prix": ["Dutch Grand Prix", "Netherlands", "Zandvoort"],
        "Hungarian Grand Prix": ["Hungarian Grand Prix", "Hungary", "Budapest"],
        "Italian Grand Prix": ["Italian Grand Prix", "Italy", "Monza"],
        "Saudi Arabian Grand Prix": [
            "Saudi Arabian Grand Prix",
            "Saudi Arabia",
            "Jeddah",
        ],
        "Bahrain Grand Prix": ["Bahrain Grand Prix", "Bahrain"],
        "Australian Grand Prix": ["Australian Grand Prix", "Australia", "Melbourne"],
        "Miami Grand Prix": ["Miami Grand Prix", "Miami"],
        "Spanish Grand Prix": ["Spanish Grand Prix", "Spain", "Barcelona"],
        "Canadian Grand Prix": ["Canadian Grand Prix", "Canada", "Montreal"],
        "Monaco Grand Prix": ["Monaco Grand Prix", "Monaco"],
        "Azerbaijan Grand Prix": ["Azerbaijan Grand Prix", "Azerbaijan", "Baku"],
        "Singapore Grand Prix": ["Singapore Grand Prix", "Singapore"],
        "Japanese Grand Prix": ["Japanese Grand Prix", "Japan", "Suzuka"],
        "Qatar Grand Prix": ["Qatar Grand Prix", "Qatar", "Lusail"],
        "Las Vegas Grand Prix": ["Las Vegas Grand Prix", "Las Vegas"],
        "Abu Dhabi Grand Prix": ["Abu Dhabi Grand Prix", "Abu Dhabi"],
        "Chinese Grand Prix": ["Chinese Grand Prix", "China", "Shanghai"],
    }

    clean = norm_ascii(race)
    base = re.sub(r"\s+Grand Prix$", "", clean).strip()
    candidates = [race, clean] + aliases.get(race, []) + aliases.get(clean, []) + [base]
    out = []
    seen = set()
    for c in candidates:
        c = str(c).strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def to_seconds(values):
    s = pd.Series(values)
    if len(s) == 0:
        return np.array([], dtype=float)
    if pd.api.types.is_timedelta64_dtype(s):
        return s.dt.total_seconds().to_numpy(dtype=float)
    if pd.api.types.is_numeric_dtype(s):
        return s.to_numpy(dtype=float)
    td = pd.to_timedelta(s, errors="coerce")
    return td.dt.total_seconds().to_numpy(dtype=float)


def status_num(x):
    digits = [int(ch) for ch in str(x) if ch.isdigit()]
    return max(digits) if digits else 0


def has_code(x, codes):
    txt = str(x)
    return any(str(c) in txt for c in codes)


def map_times_to_laps(times_sec, lap_nums, lap_secs, max_lap):
    times_sec = np.asarray(times_sec, dtype=float)
    out = np.full(len(times_sec), np.nan)
    ok = np.isfinite(times_sec)
    if not ok.any() or len(lap_secs) == 0:
        return out
    idx = np.searchsorted(lap_secs, times_sec[ok], side="right") - 1
    idx = np.clip(idx, 0, len(lap_nums) - 1)
    out[ok] = np.clip(lap_nums[idx], 1, max_lap)
    return out


def lap_reference(session, max_lap):
    laps = getattr(session, "laps", pd.DataFrame())
    if (
        laps is None
        or len(laps) == 0
        or "LapNumber" not in laps
        or "LapStartTime" not in laps
    ):
        return np.array([1], dtype=int), np.array([0.0], dtype=float)

    ref = laps[["LapNumber", "LapStartTime"]].dropna().copy()
    if len(ref) == 0:
        return np.array([1], dtype=int), np.array([0.0], dtype=float)
    ref["LapNumber"] = ref["LapNumber"].round().astype(int)
    ref = ref[(ref["LapNumber"] >= 1) & (ref["LapNumber"] <= max_lap)]
    ref = ref.groupby("LapNumber", as_index=False)["LapStartTime"].min()
    secs = to_seconds(ref["LapStartTime"])
    ref = ref.assign(sec=secs).replace([np.inf, -np.inf], np.nan).dropna(subset=["sec"])
    if len(ref) == 0:
        return np.array([1], dtype=int), np.array([0.0], dtype=float)

    ref = ref.sort_values("sec")
    lap_nums = ref["LapNumber"].to_numpy(dtype=int)
    lap_secs = ref["sec"].to_numpy(dtype=float)
    if lap_nums[0] != 1:
        lap_nums = np.r_[1, lap_nums]
        lap_secs = np.r_[0.0, lap_secs]
    return lap_nums, lap_secs


def default_status_frame(year, race, max_lap):
    frame = pd.DataFrame(
        {"Year": year, "Race": race, "LapNumber": np.arange(1, max_lap + 1)}
    )
    for col in STATUS_COLS:
        frame[col] = 0
    frame["laps_since_status_change"] = 99
    frame["laps_since_sc_or_vsc"] = 99
    frame["laps_since_wet_message"] = 99
    return frame


def session_status_features(session, year, race, max_lap):
    feat = default_status_frame(year, race, max_lap)
    feat["track_status_code"] = 1
    feat["fastf1_status_available"] = 1

    lap_nums, lap_secs = lap_reference(session, max_lap)

    ts = getattr(session, "track_status", pd.DataFrame())
    if ts is not None and len(ts) and "Status" in ts:
        ts = ts.copy()
        tcol = "Time" if "Time" in ts else None
        if tcol:
            ts["_sec"] = to_seconds(ts[tcol])
            ts["_lap"] = map_times_to_laps(ts["_sec"], lap_nums, lap_secs, max_lap)
        else:
            ts["_lap"] = np.nan
        ts = ts.dropna(subset=["_lap"]).sort_values(
            ["_lap", "_sec"] if "_sec" in ts else ["_lap"]
        )
        events = {}
        for _, row in ts.iterrows():
            lap = int(row["_lap"])
            code = str(row["Status"])
            events.setdefault(lap, []).append(code)

        cur_code = "1"
        last_change = 1
        last_sc = -10_000
        prev_sc_or_red = False
        last_red_end = -10_000

        for lap in range(1, max_lap + 1):
            event_sc = 0
            event_red = 0
            if lap in events:
                for code in events[lap]:
                    event_sc = max(event_sc, int(has_code(code, ["4", "6", "7"])))
                    event_red = max(event_red, int(has_code(code, ["5"])))
                    if code != cur_code:
                        cur_code = code
                        last_change = lap

            sc = int(has_code(cur_code, ["4", "6", "7"]))
            red = int(has_code(cur_code, ["5"]))
            yellow = int(has_code(cur_code, ["2", "3"]))
            green_after = int((sc == 0 and red == 0 and yellow == 0) and prev_sc_or_red)
            if (
                red == 0
                and prev_sc_or_red
                and feat.loc[lap - 1, "track_is_red_flag"] == 1
            ):
                last_red_end = lap
            restart = int(0 <= lap - last_red_end <= 2)

            if sc or event_sc:
                last_sc = lap

            loc = lap - 1
            feat.loc[loc, "track_status_code"] = status_num(cur_code)
            feat.loc[loc, "track_is_sc_or_vsc"] = sc
            feat.loc[loc, "track_is_red_flag"] = red
            feat.loc[loc, "track_is_yellow"] = yellow
            feat.loc[loc, "track_event_sc_or_vsc"] = event_sc
            feat.loc[loc, "track_event_red_flag"] = event_red
            feat.loc[loc, "laps_since_status_change"] = lap - last_change
            feat.loc[loc, "laps_since_sc_or_vsc"] = min(99, lap - last_sc)
            feat.loc[loc, "first_green_lap_after_sc"] = green_after
            feat.loc[loc, "red_flag_restart_window"] = restart

            prev_sc_or_red = bool(sc or red or event_sc or event_red)

    rc = getattr(session, "race_control_messages", pd.DataFrame())
    if rc is not None and len(rc):
        rc = rc.copy()
        msg_col = "Message" if "Message" in rc else None
        if msg_col:
            if "Lap" in rc:
                rc["_lap"] = pd.to_numeric(rc["Lap"], errors="coerce")
            elif "Time" in rc:
                rc["_sec"] = to_seconds(rc["Time"])
                rc["_lap"] = map_times_to_laps(rc["_sec"], lap_nums, lap_secs, max_lap)
            else:
                rc["_lap"] = np.nan

            for _, row in rc.dropna(subset=["_lap"]).iterrows():
                lap = int(np.clip(round(row["_lap"]), 1, max_lap))
                txt = str(row[msg_col]).upper()
                loc = lap - 1
                is_vsc = ("VSC" in txt) or ("VIRTUAL SAFETY CAR" in txt)
                is_sc = ("SAFETY CAR" in txt) and not is_vsc
                is_red = "RED FLAG" in txt
                is_wet = any(
                    k in txt for k in ["WET", "RAIN", "DRY", "WEATHER", "INTERMEDIATE"]
                )
                feat.loc[loc, "rc_sc_message"] = max(
                    feat.loc[loc, "rc_sc_message"], int(is_sc)
                )
                feat.loc[loc, "rc_vsc_message"] = max(
                    feat.loc[loc, "rc_vsc_message"], int(is_vsc)
                )
                feat.loc[loc, "rc_red_flag_message"] = max(
                    feat.loc[loc, "rc_red_flag_message"], int(is_red)
                )
                feat.loc[loc, "rc_wet_message"] = max(
                    feat.loc[loc, "rc_wet_message"], int(is_wet)
                )

    ss = getattr(session, "session_status", pd.DataFrame())
    if ss is not None and len(ss) and "Status" in ss:
        ss = ss.copy()
        if "Time" in ss:
            ss["_sec"] = to_seconds(ss["Time"])
            ss["_lap"] = map_times_to_laps(ss["_sec"], lap_nums, lap_secs, max_lap)
            for _, row in ss.dropna(subset=["_lap"]).iterrows():
                lap = int(np.clip(round(row["_lap"]), 1, max_lap))
                txt = str(row["Status"]).upper()
                pause = any(
                    k in txt for k in ["ABORT", "SUSPEND", "STOP", "INTERRUPT", "RED"]
                )
                feat.loc[lap - 1, "session_pause_message"] = max(
                    feat.loc[lap - 1, "session_pause_message"], int(pause)
                )

    wet_laps = feat["rc_wet_message"].to_numpy()
    pause_laps = feat["session_pause_message"].to_numpy()
    last_wet = -10_000
    for i in range(max_lap):
        lo = max(0, i - 2)
        feat.loc[i, "recent_wet_message"] = int(wet_laps[lo : i + 1].max() > 0)
        feat.loc[i, "recent_session_pause"] = int(pause_laps[lo : i + 1].max() > 0)
        if wet_laps[i] > 0:
            last_wet = i + 1
        feat.loc[i, "laps_since_wet_message"] = min(99, i + 1 - last_wet)

    return feat


def load_fastf1_session(fastf1, year, race):
    last_error = None
    for event_name in fastf1_event_candidates(race):
        try:
            session = fastf1.get_session(int(year), event_name, "R")
            try:
                session.load(laps=True, telemetry=False, weather=False, messages=True)
            except TypeError:
                session.load(laps=True, telemetry=False, weather=False)
            return session, event_name, None
        except Exception as exc:
            last_error = repr(exc)
    return None, None, last_error


def build_fastf1_status_table(all_df):
    meta = {
        "fastf1_imported": False,
        "attempted_pairs": 0,
        "loaded_pairs": 0,
        "failed_pairs": [],
    }
    empty = pd.DataFrame(columns=["Year", "Race", "LapNumber"] + STATUS_COLS)

    try:
        import fastf1
    except Exception as exc:
        meta["failed_pairs"].append({"reason": f"fastf1 import failed: {repr(exc)}"})
        return empty, meta

    meta["fastf1_imported"] = True
    try:
        fastf1.Cache.enable_cache(str(WORK_DIR / "fastf1_cache"))
    except Exception:
        pass

    frames = []
    pair_info = all_df.groupby(["Year", "Race"], as_index=False)["LapNumber"].max()
    for _, row in pair_info.iterrows():
        year, race, max_lap = int(row["Year"]), row["Race"], int(row["LapNumber"])
        if not fastf1_event_candidates(race):
            frames.append(default_status_frame(year, race, max_lap))
            continue

        meta["attempted_pairs"] += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            session, event_name, err = load_fastf1_session(fastf1, year, race)

        if session is None:
            miss = default_status_frame(year, race, max_lap)
            frames.append(miss)
            meta["failed_pairs"].append({"year": year, "race": str(race), "error": err})
            continue

        try:
            frames.append(session_status_features(session, year, race, max_lap))
            meta["loaded_pairs"] += 1
        except Exception as exc:
            frames.append(default_status_frame(year, race, max_lap))
            meta["failed_pairs"].append(
                {"year": year, "race": str(race), "error": repr(exc)}
            )

    if not frames:
        return empty, meta
    return pd.concat(frames, ignore_index=True), meta


def add_domain_interactions(df):
    df = df.copy()
    df["is_wet_compound"] = df["Compound"].isin(["WET", "INTERMEDIATE"]).astype(np.int8)
    df["is_slick_compound"] = (
        df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(np.int8)
    )

    df["sc_or_vsc_signal"] = df[
        [
            "track_is_sc_or_vsc",
            "track_event_sc_or_vsc",
            "rc_sc_message",
            "rc_vsc_message",
        ]
    ].max(axis=1)
    df["red_flag_signal"] = df[
        [
            "track_is_red_flag",
            "track_event_red_flag",
            "rc_red_flag_message",
            "red_flag_restart_window",
        ]
    ].max(axis=1)
    df["wet_transition_signal"] = df[["recent_wet_message", "rc_wet_message"]].max(
        axis=1
    )

    df["sc_tyre_life"] = df["sc_or_vsc_signal"] * df["TyreLife"]
    df["sc_race_progress"] = df["sc_or_vsc_signal"] * df["RaceProgress"]
    df["green_after_sc_tyre_life"] = df["first_green_lap_after_sc"] * df["TyreLife"]
    df["red_restart_tyre_life"] = df["red_flag_signal"] * df["TyreLife"]
    df["wet_tyre_life"] = df["wet_transition_signal"] * df["TyreLife"]
    df["wet_on_slick"] = df["wet_transition_signal"] * df["is_slick_compound"]
    df["wet_on_wet_compound"] = df["wet_transition_signal"] * df["is_wet_compound"]

    for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        key = comp.lower()
        is_comp = (df["Compound"] == comp).astype(np.int8)
        df[f"sc_on_{key}"] = df["sc_or_vsc_signal"] * is_comp
        df[f"wet_msg_on_{key}"] = df["wet_transition_signal"] * is_comp

    return df


def safe_feature_names(cols):
    used = {}
    out = {}
    for col in cols:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_")
        if not base:
            base = "feature"
        name = base
        k = 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used[name] = 1
        out[col] = name
    return out


train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

train_base = train.drop(columns=[TARGET]).copy()
test_base = test.copy()
train_base["_is_train"] = 1
test_base["_is_train"] = 0
all_data = pd.concat([train_base, test_base], ignore_index=True)

status_table, status_meta = build_fastf1_status_table(all_data)
all_data = all_data.merge(status_table, on=["Year", "Race", "LapNumber"], how="left")

for col in STATUS_COLS:
    if col not in all_data:
        all_data[col] = np.nan
distance_cols = [
    "laps_since_status_change",
    "laps_since_sc_or_vsc",
    "laps_since_wet_message",
]
for col in distance_cols:
    all_data[col] = all_data[col].fillna(99)
for col in STATUS_COLS:
    if col not in distance_cols:
        all_data[col] = all_data[col].fillna(0)

all_data = add_domain_interactions(all_data)

feature_cols = [c for c in all_data.columns if c not in [ID_COL, "_is_train"]]
rename_map = safe_feature_names(feature_cols)
X_all = all_data[feature_cols].rename(columns=rename_map)

cat_cols = []
for original, safe in rename_map.items():
    if X_all[safe].dtype == "object":
        X_all[safe] = X_all[safe].fillna("__MISSING__").astype("category")
        cat_cols.append(safe)

for col in X_all.columns:
    if col not in cat_cols:
        X_all[col] = pd.to_numeric(X_all[col], errors="coerce").fillna(0)

n_train = len(train)
X = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

oof = np.zeros(n_train, dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

pos = max(1, int(y.sum()))
neg = max(1, len(y) - pos)
scale_pos_weight = neg / pos

params = dict(
    objective="binary",
    n_estimators=2500,
    learning_rate=0.035,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)

cv = GroupKFold(n_splits=5)
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups=groups), 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = va_pred
    fold_auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(float(fold_auc))
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    print(f"fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold race-held-out ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(WORK_DIR / "submission.csv", index=False)

pd.DataFrame({"row": np.arange(n_train), "target": y, "prediction": oof}).to_csv(
    WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip"
)

submission[[ID_COL, TARGET]].to_csv(
    WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
)

result = {
    "metric": "race_held_out_5fold_roc_auc",
    "cv_auc": float(cv_auc),
    "fold_auc": fold_scores,
    "research_hypotheses_llm_claimed_used": ["000872"],
    "fastf1_status_meta": status_meta,
}
with open(WORK_DIR / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, sort_keys=True))
