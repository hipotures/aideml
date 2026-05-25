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
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"


@contextmanager
def time_limit(seconds):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise TimeoutError("FastF1 session load timed out")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def status_chars(value):
    if pd.isna(value):
        return set()
    return {c for c in str(value) if c.isdigit()}


def combine_track_status(values):
    chars = set()
    for v in values:
        chars |= status_chars(v)
    return "".join(sorted(chars)) if chars else "1"


def make_status_features_from_codes(lap_df):
    out = lap_df.copy()
    codes = out["track_status_codes"].fillna("1").astype(str)

    out["flag_yellow"] = codes.apply(lambda s: int(("2" in s) or ("3" in s)))
    out["flag_sc"] = codes.apply(lambda s: int("4" in s))
    out["flag_red"] = codes.apply(lambda s: int("5" in s))
    out["flag_vsc"] = codes.apply(lambda s: int(("6" in s) or ("7" in s)))
    out["flag_vsc_ending"] = codes.apply(lambda s: int("7" in s))
    out["flag_caution"] = (
        (out["flag_yellow"] + out["flag_sc"] + out["flag_vsc"] + out["flag_red"]) > 0
    ).astype(int)

    out["flag_state"] = 0
    out.loc[out["flag_yellow"] == 1, "flag_state"] = 1
    out.loc[out["flag_vsc"] == 1, "flag_state"] = 2
    out.loc[out["flag_sc"] == 1, "flag_state"] = 3
    out.loc[out["flag_red"] == 1, "flag_state"] = 4
    out["flag_state_name"] = (
        out["flag_state"]
        .map({0: "CLEAR", 1: "YELLOW", 2: "VSC", 3: "SC", 4: "RED"})
        .fillna("CLEAR")
    )

    out = out.sort_values("LapNumber").reset_index(drop=True)
    out["status_changed_this_lap"] = (
        out["track_status_codes"].fillna("1").astype(str)
        != out["track_status_codes"].fillna("1").astype(str).shift(1)
    ).astype(int)
    if len(out):
        out.loc[out.index[0], "status_changed_this_lap"] = 0

    ages = []
    age = 99
    prev_active = False
    for active in out["flag_caution"].astype(bool).values:
        if active:
            age = 0 if not prev_active else age + 1
            prev_active = True
        else:
            age = 99
            prev_active = False
        ages.append(age)
    out["laps_since_sc_vsc_started"] = ages

    return out


def race_control_features(session, lap_numbers):
    base = pd.DataFrame(
        {"LapNumber": sorted(pd.Series(lap_numbers).dropna().astype(int).unique())}
    )
    for col in [
        "rc_events_this_lap",
        "rc_sc_vsc_events_this_lap",
        "rc_yellow_events_this_lap",
        "rc_pit_events_this_lap",
        "rc_vsc_ending_this_lap",
    ]:
        base[col] = 0

    rc = getattr(session, "race_control_messages", None)
    if rc is None or not isinstance(rc, pd.DataFrame) or rc.empty:
        return add_recent_rc_rollups(base)

    rc = rc.copy()
    lap_col = next((c for c in rc.columns if c.lower() == "lap"), None)
    if lap_col is not None:
        msg_lap = pd.to_numeric(rc[lap_col], errors="coerce")
    else:
        msg_lap = align_messages_to_lap_by_time(session, rc)

    if msg_lap is None:
        return add_recent_rc_rollups(base)

    rc["LapNumber"] = pd.to_numeric(msg_lap, errors="coerce")
    rc = rc.dropna(subset=["LapNumber"])
    if rc.empty:
        return add_recent_rc_rollups(base)
    rc["LapNumber"] = rc["LapNumber"].astype(int)

    text_cols = [
        c for c in ["Category", "Message", "Status", "Flag", "Scope"] if c in rc.columns
    ]
    if text_cols:
        txt = rc[text_cols].astype(str).agg(" ".join, axis=1).str.upper()
    else:
        txt = pd.Series([""] * len(rc), index=rc.index)

    rc["rc_events_this_lap"] = 1
    rc["rc_sc_vsc_events_this_lap"] = txt.str.contains(
        "SAFETY CAR|VSC|VIRTUAL SAFETY", regex=True
    ).astype(int)
    rc["rc_yellow_events_this_lap"] = txt.str.contains("YELLOW", regex=False).astype(
        int
    )
    rc["rc_pit_events_this_lap"] = txt.str.contains(
        "PIT LANE|PIT EXIT|PIT ENTRY", regex=True
    ).astype(int)
    rc["rc_vsc_ending_this_lap"] = txt.str.contains(
        "VSC ENDING|VIRTUAL SAFETY CAR ENDING", regex=True
    ).astype(int)

    grouped = (
        rc.groupby("LapNumber")[
            [
                "rc_events_this_lap",
                "rc_sc_vsc_events_this_lap",
                "rc_yellow_events_this_lap",
                "rc_pit_events_this_lap",
                "rc_vsc_ending_this_lap",
            ]
        ]
        .sum()
        .reset_index()
    )

    base = base.drop(
        columns=[c for c in grouped.columns if c != "LapNumber"], errors="ignore"
    ).merge(grouped, on="LapNumber", how="left")
    base = base.fillna(0)
    return add_recent_rc_rollups(base)


def align_messages_to_lap_by_time(session, rc):
    if "Time" not in rc.columns:
        return None
    laps = getattr(session, "laps", None)
    if laps is None or not isinstance(laps, pd.DataFrame) or laps.empty:
        return None
    if "LapNumber" not in laps.columns or "LapStartTime" not in laps.columns:
        return None

    starts = laps[["LapNumber", "LapStartTime"]].copy()
    starts["LapNumber"] = pd.to_numeric(starts["LapNumber"], errors="coerce")
    starts = (
        starts.dropna(subset=["LapNumber"])
        .groupby("LapNumber")["LapStartTime"]
        .min()
        .sort_index()
    )
    starts = pd.to_timedelta(starts, errors="coerce")
    if len(starts) == 0:
        return None
    if 1 in starts.index and pd.isna(starts.loc[1]):
        starts.loc[1] = pd.Timedelta(0)
    starts = starts.dropna()
    if len(starts) == 0:
        return None

    msg_time = pd.to_timedelta(rc["Time"], errors="coerce")
    valid = msg_time.notna()
    mapped = pd.Series(np.nan, index=rc.index, dtype=float)
    if valid.sum() == 0:
        return mapped

    start_ns = starts.values.astype("timedelta64[ns]").astype(np.int64)
    lap_vals = starts.index.astype(int).values
    msg_ns = msg_time[valid].values.astype("timedelta64[ns]").astype(np.int64)
    idx = np.searchsorted(start_ns, msg_ns, side="right") - 1
    idx = np.clip(idx, 0, len(lap_vals) - 1)
    mapped.loc[valid] = lap_vals[idx]
    return mapped


def add_recent_rc_rollups(df):
    df = df.sort_values("LapNumber").reset_index(drop=True)
    count_cols = [
        c for c in df.columns if c.startswith("rc_") and c.endswith("_this_lap")
    ]
    for col in count_cols:
        prefix = col.replace("_this_lap", "")
        df[f"{prefix}_recent3"] = df[col].rolling(3, min_periods=1).sum()
        df[f"{prefix}_recent5"] = df[col].rolling(5, min_periods=1).sum()
    return df


def neutral_fastf1_features(keys):
    out = keys[["Year", "Race", "LapNumber"]].drop_duplicates().copy()
    out["track_status_codes"] = "1"
    out = make_status_features_from_codes(out)
    for col in [
        "rc_events_this_lap",
        "rc_sc_vsc_events_this_lap",
        "rc_yellow_events_this_lap",
        "rc_pit_events_this_lap",
        "rc_vsc_ending_this_lap",
        "rc_events_recent3",
        "rc_sc_vsc_events_recent3",
        "rc_yellow_events_recent3",
        "rc_pit_events_recent3",
        "rc_vsc_ending_recent3",
        "rc_events_recent5",
        "rc_sc_vsc_events_recent5",
        "rc_yellow_events_recent5",
        "rc_pit_events_recent5",
        "rc_vsc_ending_recent5",
    ]:
        out[col] = 0
    out["fastf1_status_available"] = 0
    return out


def get_fastf1_session(fastf1, year, race):
    race_str = str(race)
    if "Pre-Season Testing" in race_str:
        try:
            return fastf1.get_testing_session(int(year), 1, 1)
        except Exception:
            return None
    try:
        return fastf1.get_session(int(year), race_str, "R")
    except Exception:
        return None


def load_one_fastf1_status(fastf1, year, race, lap_numbers, timeout_seconds):
    session = get_fastf1_session(fastf1, year, race)
    if session is None:
        return None

    with time_limit(timeout_seconds):
        session.load(laps=True, telemetry=False, weather=False, messages=True)

    laps = getattr(session, "laps", None)
    if laps is None or not isinstance(laps, pd.DataFrame) or laps.empty:
        return None
    if "LapNumber" not in laps.columns or "TrackStatus" not in laps.columns:
        return None

    lap_status = laps[["LapNumber", "TrackStatus"]].copy()
    lap_status["LapNumber"] = pd.to_numeric(lap_status["LapNumber"], errors="coerce")
    lap_status = lap_status.dropna(subset=["LapNumber"])
    if lap_status.empty:
        return None
    lap_status["LapNumber"] = lap_status["LapNumber"].astype(int)

    lap_status = (
        lap_status.groupby("LapNumber")["TrackStatus"]
        .agg(combine_track_status)
        .reset_index()
        .rename(columns={"TrackStatus": "track_status_codes"})
    )

    needed = pd.DataFrame(
        {"LapNumber": sorted(pd.Series(lap_numbers).dropna().astype(int).unique())}
    )
    lap_status = needed.merge(lap_status, on="LapNumber", how="left")
    lap_status["track_status_codes"] = lap_status["track_status_codes"].fillna("1")
    lap_status = make_status_features_from_codes(lap_status)

    rc = race_control_features(session, needed["LapNumber"])
    out = lap_status.merge(rc, on="LapNumber", how="left").fillna(0)
    out["Year"] = int(year)
    out["Race"] = str(race)
    out["fastf1_status_available"] = 1
    return out


def build_fastf1_features(all_data):
    keys = all_data[["Year", "Race", "LapNumber"]].drop_duplicates().copy()
    neutral = neutral_fastf1_features(keys)

    try:
        import fastf1
    except Exception as e:
        print(f"FastF1 unavailable, using neutral status features: {e}")
        return neutral

    try:
        cache_dir = os.path.join(WORKING_DIR, "fastf1_cache")
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        try:
            fastf1.set_log_level("ERROR")
        except Exception:
            pass
    except Exception:
        pass

    timeout_seconds = int(os.environ.get("FASTF1_SESSION_TIMEOUT", "8"))
    max_failures = int(os.environ.get("FASTF1_MAX_FAILURES", "8"))

    pieces = []
    failures = 0
    sessions = keys.groupby(["Year", "Race"])["LapNumber"].apply(list).reset_index()
    print(f"Attempting FastF1 enrichment for {len(sessions)} race-year sessions")

    for i, row in sessions.iterrows():
        year, race, laps = int(row["Year"]), row["Race"], row["LapNumber"]
        try:
            part = load_one_fastf1_status(fastf1, year, race, laps, timeout_seconds)
            if part is None:
                failures += 1
            else:
                pieces.append(part)
                print(f"FastF1 loaded {year} {race} ({len(part)} laps)")
        except Exception as e:
            failures += 1
            print(f"FastF1 skipped {year} {race}: {type(e).__name__}: {e}")

        if failures >= max_failures and len(pieces) == 0:
            print(
                "Stopping FastF1 attempts after repeated failures; using neutral status features"
            )
            return neutral

    if not pieces:
        return neutral

    loaded = pd.concat(pieces, ignore_index=True)
    status_cols = [c for c in loaded.columns if c not in ["Year", "Race", "LapNumber"]]
    merged = keys.merge(loaded, on=["Year", "Race", "LapNumber"], how="left")
    neutral_small = neutral[["Year", "Race", "LapNumber"] + status_cols]
    merged = merged.merge(
        neutral_small,
        on=["Year", "Race", "LapNumber"],
        how="left",
        suffixes=("", "_neutral"),
    )

    for col in status_cols:
        merged[col] = merged[col].where(merged[col].notna(), merged[f"{col}_neutral"])
        merged = merged.drop(columns=[f"{col}_neutral"])
    return merged


def add_model_features(train, test):
    all_data = pd.concat(
        [train.drop(columns=[TARGET]), test],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    race_max_lap = all_data.groupby(["Year", "Race"])["LapNumber"].transform("max")
    all_data["race_total_laps"] = race_max_lap
    all_data["laps_remaining"] = all_data["race_total_laps"] - all_data["LapNumber"]
    all_data["lap_progress_from_number"] = all_data["LapNumber"] / all_data[
        "race_total_laps"
    ].replace(0, np.nan)
    all_data["tyrelife_lap_ratio"] = all_data["TyreLife"] / all_data["LapNumber"].clip(
        lower=1
    )
    all_data["tyrelife_remaining_ratio"] = all_data["TyreLife"] / all_data[
        "laps_remaining"
    ].clip(lower=1)
    all_data["degradation_per_tyre_lap"] = all_data[
        "Cumulative_Degradation"
    ] / all_data["TyreLife"].clip(lower=1)
    all_data["compound_stint"] = (
        all_data["Compound"].astype(str) + "_S" + all_data["Stint"].astype(str)
    )

    ff = build_fastf1_features(all_data)
    all_data = all_data.merge(ff, on=["Year", "Race", "LapNumber"], how="left")

    fill_zero = [
        c
        for c in all_data.columns
        if c.startswith("flag_")
        or c.startswith("rc_")
        or c
        in [
            "status_changed_this_lap",
            "laps_since_sc_vsc_started",
            "fastf1_status_available",
        ]
    ]
    for col in fill_zero:
        if col == "flag_state_name":
            all_data[col] = all_data[col].fillna("CLEAR")
        elif col == "track_status_codes":
            all_data[col] = all_data[col].fillna("1")
        else:
            all_data[col] = pd.to_numeric(all_data[col], errors="coerce").fillna(0)

    all_data["tyrelife_x_caution"] = all_data["TyreLife"] * all_data["flag_caution"]
    all_data["tyrelife_x_sc"] = all_data["TyreLife"] * all_data["flag_sc"]
    all_data["tyrelife_x_vsc"] = all_data["TyreLife"] * all_data["flag_vsc"]
    all_data["laps_remaining_x_caution"] = (
        all_data["laps_remaining"] * all_data["flag_caution"]
    )
    all_data["laps_remaining_x_vsc_ending"] = (
        all_data["laps_remaining"] * all_data["flag_vsc_ending"]
    )
    all_data["degradation_x_caution"] = (
        all_data["Cumulative_Degradation"] * all_data["flag_caution"]
    )
    all_data["compound_flag_state"] = (
        all_data["Compound"].astype(str) + "_" + all_data["flag_state_name"].astype(str)
    )
    all_data["compound_caution"] = (
        all_data["Compound"].astype(str)
        + "_C"
        + all_data["flag_caution"].astype(int).astype(str)
    )
    all_data["race_driver"] = (
        all_data["Race"].astype(str) + "_" + all_data["Driver"].astype(str)
    )

    train_fe = all_data.iloc[: len(train)].copy()
    test_fe = all_data.iloc[len(train) :].copy()
    train_fe[TARGET] = train[TARGET].values
    return train_fe, test_fe


def train_lgbm_cv(train_fe, test_fe, sample):
    import lightgbm as lgb

    drop_cols = [TARGET, ID_COL]
    feature_cols = [c for c in train_fe.columns if c not in drop_cols]
    y = train_fe[TARGET].astype(int).values

    X = train_fe[feature_cols].copy()
    X_test = test_fe[feature_cols].copy()

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        combined = pd.concat([X[col], X_test[col]], axis=0).astype(str).fillna("__NA__")
        cats = pd.Categorical(combined).categories
        X[col] = pd.Categorical(X[col].astype(str).fillna("__NA__"), categories=cats)
        X_test[col] = pd.Categorical(
            X_test[col].astype(str).fillna("__NA__"), categories=cats
        )

    for col in X.columns:
        if col not in categorical_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

    groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)
    if groups.nunique() >= 5:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X, y))

    oof = np.zeros(len(X), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_aucs = []

    params = dict(
        objective="binary",
        learning_rate=0.035,
        n_estimators=4000,
        num_leaves=96,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=3.0,
        random_state=SEED,
        n_jobs=max(1, min(16, os.cpu_count() or 1)),
        importance_type="gain",
        verbose=-1,
    )

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            categorical_feature=categorical_cols,
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        va_pred = model.predict_proba(
            X.iloc[va_idx], num_iteration=model.best_iteration_
        )[:, 1]
        te_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
        oof[va_idx] = va_pred
        test_pred += te_pred / len(splits)

        fold_auc = roc_auc_score(y[va_idx], va_pred)
        fold_aucs.append(float(fold_auc))
        print(
            f"fold {fold} roc_auc={fold_auc:.6f} best_iteration={model.best_iteration_}"
        )

    cv_auc = float(roc_auc_score(y, oof))
    print(f"OOF ROC AUC: {cv_auc:.6f}")

    pd.DataFrame(
        {"row": np.arange(len(train_fe)), "target": y, "prediction": oof}
    ).to_csv(
        os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    test_predictions = sample[[ID_COL]].copy()
    test_predictions[TARGET] = test_pred
    test_predictions.to_csv(
        os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    submission = sample[[ID_COL]].copy()
    submission[TARGET] = test_pred
    submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

    return cv_auc, fold_aucs


def main():
    start = time.time()

    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
    sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

    train_fe, test_fe = add_model_features(train, test)
    cv_auc, fold_aucs = train_lgbm_cv(train_fe, test_fe, sample)

    result = {
        "research_hypotheses_llm_claimed_used": ["000425"],
        "metric": "roc_auc",
        "cv_auc": cv_auc,
        "fold_auc": fold_aucs,
        "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
