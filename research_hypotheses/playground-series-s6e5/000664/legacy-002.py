import os
import re
import sys
import json
import time
import warnings
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5


def normalize_name(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    x = re.sub(r"[^a-z0-9]+", " ", x.lower())
    return re.sub(r"\s+", " ", x).strip()


def status_flags_from_text(values):
    text = " ".join(str(v) for v in values if pd.notna(v)).upper()
    digits = set(ch for ch in text if ch.isdigit())
    return {
        "ff1_track_yellow": int(("2" in digits) or ("YELLOW" in text)),
        "ff1_track_sc": int(
            ("4" in digits) or ("SAFETY CAR" in text and "VIRTUAL" not in text)
        ),
        "ff1_track_vsc": int(
            ("6" in digits)
            or ("7" in digits)
            or ("VIRTUAL SAFETY CAR" in text)
            or (" VSC" in text)
        ),
        "ff1_track_red": int(("5" in digits) or ("RED FLAG" in text)),
    }


def ensure_fastf1():
    meta = {"install_attempted": False}
    try:
        import fastf1

        return fastf1, meta
    except Exception as e:
        meta["initial_import_error"] = f"{type(e).__name__}: {e}"

    pkg_dir = WORK_DIR / "fastf1_pkg"
    if pkg_dir.exists():
        sys.path.insert(0, str(pkg_dir))
        try:
            import fastf1

            meta["loaded_from_working_target"] = True
            return fastf1, meta
        except Exception as e:
            meta["working_target_import_error"] = f"{type(e).__name__}: {e}"

    meta["install_attempted"] = True
    try:
        env = os.environ.copy()
        env.setdefault("PIP_DEFAULT_TIMEOUT", "20")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--target",
                str(pkg_dir),
                "fastf1",
            ],
            check=True,
            timeout=180,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        sys.path.insert(0, str(pkg_dir))
        import fastf1

        meta["installed_to_working"] = True
        return fastf1, meta
    except Exception as e:
        meta["install_error"] = f"{type(e).__name__}: {e}"
        return None, meta


def load_session_fastf1(fastf1, year, race):
    attempts = [str(race)]
    try:
        schedule = fastf1.get_event_schedule(int(year), include_testing=False)
        rn = normalize_name(race)
        best_name, best_score = None, -1
        for _, row in schedule.iterrows():
            for col in ["EventName", "OfficialEventName", "Location", "Country"]:
                if col in row and pd.notna(row[col]):
                    cand = str(row[col])
                    cn = normalize_name(cand)
                    score = len(set(rn.split()) & set(cn.split()))
                    if rn == cn:
                        score += 10
                    if score > best_score:
                        best_name, best_score = cand, score
        if best_name and best_name not in attempts:
            attempts.append(best_name)
    except Exception:
        pass

    last_error = None
    for event_name in attempts:
        try:
            session = fastf1.get_session(int(year), event_name, "R")
            try:
                session.load(laps=True, telemetry=False, weather=True, messages=True)
            except TypeError:
                session.load(laps=True, telemetry=False, weather=True)
            return session
        except Exception as e:
            last_error = e
    raise last_error


def build_context_from_fastf1(base_df):
    context_path = WORK_DIR / "fastf1_lap_context.csv.gz"
    if context_path.exists():
        return pd.read_csv(context_path), {
            "enabled": True,
            "loaded_from_cache_file": True,
            "attempted": 0,
            "loaded": 0,
            "failed": 0,
        }

    meta = {"attempted": 0, "loaded": 0, "failed": 0, "enabled": True}
    fastf1, import_meta = ensure_fastf1()
    meta.update(import_meta)
    if fastf1 is None:
        meta["enabled"] = False
        return pd.DataFrame(), meta

    try:
        fastf1.Cache.enable_cache(str(WORK_DIR / "fastf1_cache"))
    except Exception as e:
        meta["cache_warning"] = f"{type(e).__name__}: {e}"

    budget_seconds = int(os.environ.get("FASTF1_CONTEXT_BUDGET_SECONDS", "720"))
    start_time = time.time()
    context_parts = []

    events = (
        base_df[["Year", "Race"]]
        .drop_duplicates()
        .sort_values(["Year", "Race"])
        .itertuples(index=False, name=None)
    )

    for year, race in events:
        if "testing" in normalize_name(race):
            continue
        if time.time() - start_time > budget_seconds:
            meta["stopped_reason"] = "fastf1_context_time_budget"
            break

        meta["attempted"] += 1
        try:
            session = load_session_fastf1(fastf1, year, race)
            max_lap = int(
                base_df.loc[
                    (base_df["Year"] == year) & (base_df["Race"] == race), "LapNumber"
                ].max()
            )
            lap_ctx = pd.DataFrame({"LapNumber": np.arange(1, max_lap + 1, dtype=int)})
            lap_ctx["Year"] = int(year)
            lap_ctx["Race"] = str(race)

            laps = getattr(session, "laps", None)
            lap_times = None
            if laps is not None and len(laps):
                ldf = pd.DataFrame(laps).copy()
                if "LapNumber" in ldf.columns:
                    ldf["LapNumber"] = pd.to_numeric(
                        ldf["LapNumber"], errors="coerce"
                    ).astype("Int64")

                    if "TrackStatus" in ldf.columns:
                        rows = []
                        for lap, g in ldf.dropna(subset=["LapNumber"]).groupby(
                            "LapNumber"
                        ):
                            flags = status_flags_from_text(g["TrackStatus"].tolist())
                            flags["LapNumber"] = int(lap)
                            rows.append(flags)
                        if rows:
                            lap_ctx = lap_ctx.merge(
                                pd.DataFrame(rows), on="LapNumber", how="left"
                            )

                    pit_cols = [
                        c for c in ["PitInTime", "PitOutTime"] if c in ldf.columns
                    ]
                    if pit_cols:
                        pit_any = ldf[pit_cols].notna().any(axis=1).astype(int)
                        pit_counts = (
                            pd.DataFrame(
                                {
                                    "LapNumber": ldf["LapNumber"],
                                    "ff1_current_lap_pit_count": pit_any,
                                }
                            )
                            .dropna(subset=["LapNumber"])
                            .groupby("LapNumber", as_index=False)[
                                "ff1_current_lap_pit_count"
                            ]
                            .sum()
                        )
                        pit_counts["LapNumber"] = pit_counts["LapNumber"].astype(int)
                        lap_ctx = lap_ctx.merge(pit_counts, on="LapNumber", how="left")

                    if "Time" in ldf.columns:
                        lap_times = (
                            ldf.dropna(subset=["LapNumber", "Time"])
                            .groupby("LapNumber", as_index=False)["Time"]
                            .max()
                            .rename(columns={"Time": "lap_end_time"})
                            .sort_values("lap_end_time")
                        )
                        lap_times["LapNumber"] = lap_times["LapNumber"].astype(int)

            rcm = getattr(session, "race_control_messages", None)
            if rcm is not None and len(rcm):
                mdf = pd.DataFrame(rcm).copy()
                msg_col = "Message" if "Message" in mdf.columns else None
                if "Lap" in mdf.columns:
                    mdf["LapNumber"] = pd.to_numeric(mdf["Lap"], errors="coerce")
                elif "LapNumber" in mdf.columns:
                    mdf["LapNumber"] = pd.to_numeric(mdf["LapNumber"], errors="coerce")
                elif lap_times is not None and "Time" in mdf.columns:
                    tmp = mdf.dropna(subset=["Time"]).sort_values("Time")
                    mdf = pd.merge_asof(
                        tmp,
                        lap_times,
                        left_on="Time",
                        right_on="lap_end_time",
                        direction="forward",
                    )
                else:
                    mdf["LapNumber"] = np.nan

                if msg_col and "LapNumber" in mdf.columns:
                    rows = []
                    for lap, g in mdf.dropna(subset=["LapNumber"]).groupby(
                        mdf["LapNumber"].dropna().astype(int)
                    ):
                        text = " ".join(g[msg_col].astype(str)).upper()
                        rows.append(
                            {
                                "LapNumber": int(lap),
                                "ff1_msg_count": len(g),
                                "ff1_msg_sc": int(
                                    "SAFETY CAR" in text and "VIRTUAL" not in text
                                ),
                                "ff1_msg_vsc": int(
                                    "VIRTUAL SAFETY CAR" in text or " VSC" in text
                                ),
                                "ff1_msg_yellow": int("YELLOW" in text),
                                "ff1_msg_red": int("RED FLAG" in text),
                                "ff1_msg_rain": int(
                                    ("RAIN" in text) or ("WET" in text)
                                ),
                                "ff1_msg_incident": int(
                                    ("INCIDENT" in text)
                                    or ("STOPPED" in text)
                                    or ("DEBRIS" in text)
                                ),
                            }
                        )
                    if rows:
                        lap_ctx = lap_ctx.merge(
                            pd.DataFrame(rows), on="LapNumber", how="left"
                        )

            weather = getattr(session, "weather_data", None)
            if weather is not None and len(weather) and lap_times is not None:
                wdf = pd.DataFrame(weather).copy()
                weather_cols = [
                    c
                    for c in [
                        "AirTemp",
                        "Humidity",
                        "Pressure",
                        "Rainfall",
                        "TrackTemp",
                        "WindDirection",
                        "WindSpeed",
                    ]
                    if c in wdf.columns
                ]
                if "Time" in wdf.columns and weather_cols:
                    right = (
                        wdf[["Time"] + weather_cols]
                        .dropna(subset=["Time"])
                        .sort_values("Time")
                    )
                    if len(right):
                        merged = pd.merge_asof(
                            lap_times[["LapNumber", "lap_end_time"]].sort_values(
                                "lap_end_time"
                            ),
                            right,
                            left_on="lap_end_time",
                            right_on="Time",
                            direction="backward",
                        )
                        rename = {
                            c: "ff1_weather_" + re.sub(r"[^A-Za-z0-9]+", "_", c).lower()
                            for c in weather_cols
                        }
                        lap_ctx = lap_ctx.merge(
                            merged[["LapNumber"] + weather_cols].rename(columns=rename),
                            on="LapNumber",
                            how="left",
                        )

            core_cols = [
                "ff1_track_yellow",
                "ff1_track_sc",
                "ff1_track_vsc",
                "ff1_track_red",
                "ff1_current_lap_pit_count",
                "ff1_msg_count",
                "ff1_msg_sc",
                "ff1_msg_vsc",
                "ff1_msg_yellow",
                "ff1_msg_red",
                "ff1_msg_rain",
                "ff1_msg_incident",
            ]
            for c in core_cols:
                if c not in lap_ctx.columns:
                    lap_ctx[c] = 0
                lap_ctx[c] = pd.to_numeric(lap_ctx[c], errors="coerce").fillna(0)

            lap_ctx = lap_ctx.sort_values("LapNumber")
            lap_ctx["ff1_prev_lap_pit_count"] = (
                lap_ctx["ff1_current_lap_pit_count"].shift(1).fillna(0)
            )
            lap_ctx["ff1_pit_count_recent3"] = (
                lap_ctx["ff1_current_lap_pit_count"]
                .rolling(3, min_periods=1)
                .sum()
                .fillna(0)
            )
            lap_ctx["ff1_any_neutralized"] = lap_ctx[
                ["ff1_track_sc", "ff1_track_vsc", "ff1_track_red"]
            ].max(axis=1)

            if "ff1_weather_rainfall" not in lap_ctx.columns:
                lap_ctx["ff1_weather_rainfall"] = 0.0
            lap_ctx["ff1_wet_or_rain"] = (
                (lap_ctx["ff1_weather_rainfall"].fillna(0) > 0)
                | (lap_ctx["ff1_msg_rain"] > 0)
            ).astype(int)

            event_cols = [
                c
                for c in lap_ctx.columns
                if c.startswith("ff1_track_")
                or c.startswith("ff1_msg_")
                or c in ["ff1_any_neutralized", "ff1_wet_or_rain"]
            ]
            for c in event_cols:
                lap_ctx[c + "_recent3"] = (
                    lap_ctx[c].rolling(3, min_periods=1).max().fillna(0)
                )
                lap_ctx[c + "_started"] = (
                    (lap_ctx[c].fillna(0) > 0)
                    & (lap_ctx[c].fillna(0).shift(1).fillna(0) == 0)
                ).astype(int)

            context_parts.append(lap_ctx)
            meta["loaded"] += 1
        except Exception:
            meta["failed"] += 1

    if not context_parts:
        return pd.DataFrame(), meta

    context = pd.concat(context_parts, ignore_index=True).drop_duplicates(
        ["Year", "Race", "LapNumber"]
    )
    context.to_csv(context_path, index=False, compression="gzip")
    return context, meta


def add_base_features(df):
    out = df.copy()
    out["race_key"] = out["Year"].astype(str) + "_" + out["Race"].astype(str)
    out["driver_race_key"] = out["race_key"] + "_" + out["Driver"].astype(str)
    out["estimated_total_laps"] = np.where(
        out["RaceProgress"] > 0, out["LapNumber"] / out["RaceProgress"], np.nan
    )
    out["laps_remaining_est"] = out["estimated_total_laps"] - out["LapNumber"]
    out["tyre_life_share"] = out["TyreLife"] / np.maximum(
        out["estimated_total_laps"], 1
    )
    out["degradation_per_tyre_lap"] = out["Cumulative_Degradation"] / np.maximum(
        out["TyreLife"], 1
    )
    out["lap_progress_x_tyre"] = out["RaceProgress"] * out["TyreLife"]
    out["is_wet_compound"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    return out


def add_observed_lap_context(train, test):
    both = pd.concat([train, test], ignore_index=True, sort=False)
    lap = (
        both.groupby(["Year", "Race", "LapNumber"], as_index=False)
        .agg(
            obs_current_lap_pit_count=("PitStop", "sum"),
            obs_current_lap_pit_rate=("PitStop", "mean"),
            obs_driver_count=("PitStop", "size"),
            obs_median_lap_delta=("LapTime_Delta", "median"),
        )
        .sort_values(["Year", "Race", "LapNumber"])
    )
    g = lap.groupby(["Year", "Race"], sort=False)
    lap["obs_prev_lap_pit_count"] = g["obs_current_lap_pit_count"].shift(1).fillna(0)
    lap["obs_pit_count_recent3"] = (
        g["obs_current_lap_pit_count"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
    )
    train = train.merge(lap, on=["Year", "Race", "LapNumber"], how="left")
    test = test.merge(lap, on=["Year", "Race", "LapNumber"], how="left")
    for df in (train, test):
        df["obs_other_current_lap_pit_count"] = (
            df["obs_current_lap_pit_count"] - df["PitStop"]
        ).clip(lower=0)
    return train, test


def add_fastf1_context(train, test):
    base = pd.concat(
        [train[["Year", "Race", "LapNumber"]], test[["Year", "Race", "LapNumber"]]],
        ignore_index=True,
    ).drop_duplicates()
    context, meta = build_context_from_fastf1(base)

    default_cols = [
        "ff1_context_available",
        "ff1_track_yellow",
        "ff1_track_sc",
        "ff1_track_vsc",
        "ff1_track_red",
        "ff1_any_neutralized",
        "ff1_any_neutralized_recent3",
        "ff1_wet_or_rain",
        "ff1_weather_rainfall",
        "ff1_current_lap_pit_count",
        "ff1_prev_lap_pit_count",
        "ff1_pit_count_recent3",
    ]

    if context.empty:
        for df in (train, test):
            for c in default_cols:
                df[c] = 0.0
        return train, test, meta

    train = train.merge(context, on=["Year", "Race", "LapNumber"], how="left")
    test = test.merge(context, on=["Year", "Race", "LapNumber"], how="left")
    ff_cols = [c for c in context.columns if c not in ["Year", "Race", "LapNumber"]]

    for df in (train, test):
        df["ff1_context_available"] = df[ff_cols].notna().any(axis=1).astype(int)
        for c in ff_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        for c in default_cols:
            if c not in df.columns:
                df[c] = 0.0

    return train, test, meta


def prepare_features(train, test):
    drop_cols = [ID_COL, TARGET]
    features = [c for c in train.columns if c not in drop_cols and c in test.columns]

    combined = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
    cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in cat_cols:
        cats = pd.Categorical(combined[c].astype(str).fillna("missing")).categories
        train[c] = pd.Categorical(
            train[c].astype(str).fillna("missing"), categories=cats
        )
        test[c] = pd.Categorical(test[c].astype(str).fillna("missing"), categories=cats)

    for c in features:
        if c not in cat_cols:
            med = (
                pd.to_numeric(train[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .median()
            )
            if not np.isfinite(med):
                med = 0.0
            train[c] = (
                pd.to_numeric(train[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(med)
            )
            test[c] = (
                pd.to_numeric(test[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(med)
            )

    return train, test, features, cat_cols


def make_model(y_train, cat_cols):
    try:
        from lightgbm import LGBMClassifier

        pos = max(float(np.sum(y_train)), 1.0)
        neg = max(float(len(y_train) - np.sum(y_train)), 1.0)
        return (
            LGBMClassifier(
                objective="binary",
                n_estimators=900,
                learning_rate=0.035,
                num_leaves=63,
                min_child_samples=80,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.05,
                reg_lambda=0.2,
                scale_pos_weight=neg / pos,
                random_state=RANDOM_STATE,
                n_jobs=max(1, os.cpu_count() or 1),
                verbosity=-1,
                force_col_wise=True,
            ),
            "lightgbm",
        )
    except Exception:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder

        return (
            Pipeline(
                [
                    (
                        "pre",
                        ColumnTransformer(
                            [
                                (
                                    "cat",
                                    OrdinalEncoder(
                                        handle_unknown="use_encoded_value",
                                        unknown_value=-1,
                                    ),
                                    cat_cols,
                                ),
                                (
                                    "num",
                                    SimpleImputer(strategy="median"),
                                    [
                                        c
                                        for c in train_features_global
                                        if c not in cat_cols
                                    ],
                                ),
                            ],
                            remainder="drop",
                        ),
                    ),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_iter=380,
                            max_leaf_nodes=31,
                            l2_regularization=0.05,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "sklearn_hgb",
        )


def fit_model(model, model_kind, X_tr, y_tr, X_va, y_va, cat_cols):
    if model_kind == "lightgbm":
        try:
            import lightgbm as lgb

            callbacks = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                categorical_feature=cat_cols,
                callbacks=callbacks,
            )
        except Exception:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                categorical_feature=cat_cols,
            )
    else:
        model.fit(X_tr, y_tr)
    return model


def predict_positive(model, X):
    pred = model.predict_proba(X)[:, 1]
    return np.clip(pred, 0.0, 1.0)


train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

train = add_base_features(train)
test = add_base_features(test)
train, test = add_observed_lap_context(train, test)
train, test, ff1_meta = add_fastf1_context(train, test)
train, test, features, cat_cols = prepare_features(train, test)

train_features_global = features
y = train[TARGET].astype(int).values
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_aucs = []

splitter = GroupKFold(n_splits=N_SPLITS)
for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, y, groups=groups), 1):
    X_tr, y_tr = train.iloc[tr_idx][features], y[tr_idx]
    X_va, y_va = train.iloc[va_idx][features], y[va_idx]

    model, model_kind = make_model(y_tr, cat_cols)
    model = fit_model(model, model_kind, X_tr, y_tr, X_va, y_va, cat_cols)

    va_pred = predict_positive(model, X_va)
    oof[va_idx] = va_pred
    fold_auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(float(fold_auc))

    test_pred += predict_positive(model, test[features]) / N_SPLITS
    print(
        json.dumps(
            {"fold": fold, "validation_roc_auc": float(fold_auc)}, sort_keys=True
        )
    )

cv_auc = roc_auc_score(y, oof)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

submission = sample[[ID_COL]].copy()
submission[TARGET] = np.clip(test_pred, 0.0, 1.0)
submission.to_csv(WORK_DIR / "submission.csv", index=False)
submission.to_csv(WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip")

result = {
    "validation_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_aucs,
    "research_hypotheses_llm_claimed_used": ["000664"],
    "fastf1_context": ff1_meta,
    "n_features": int(len(features)),
    "n_train": int(len(train)),
    "n_test": int(len(test)),
    "n_splits": int(N_SPLITS),
}
print(json.dumps(result, sort_keys=True))
