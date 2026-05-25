import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42


def normalize_name(x):
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def status_flags_from_text(values):
    text = " ".join([str(v) for v in values if pd.notna(v)]).upper()
    return {
        "ff1_track_yellow": int(("2" in text) or ("YELLOW" in text)),
        "ff1_track_sc": int(
            ("4" in text) or ("SAFETY CAR" in text and "VIRTUAL" not in text)
        ),
        "ff1_track_vsc": int(
            ("6" in text)
            or ("7" in text)
            or ("VIRTUAL SAFETY CAR" in text)
            or (" VSC" in text)
        ),
        "ff1_track_red": int(("5" in text) or ("RED FLAG" in text)),
    }


def build_context_from_fastf1(base_df):
    unique_events = (
        base_df[["Year", "Race"]]
        .drop_duplicates()
        .sort_values(["Year", "Race"])
        .itertuples(index=False, name=None)
    )

    context_parts = []
    meta = {"attempted": 0, "loaded": 0, "failed": 0, "enabled": True}

    try:
        import fastf1

        fastf1.Cache.enable_cache(str(WORK_DIR / "fastf1_cache"))
    except Exception as e:
        meta.update(
            {"enabled": False, "reason": f"fastf1 unavailable: {type(e).__name__}: {e}"}
        )
        return pd.DataFrame(), meta

    for year, race in unique_events:
        if "testing" in normalize_name(race):
            continue

        meta["attempted"] += 1
        try:
            session = fastf1.get_session(int(year), str(race), "R")
            session.load(laps=True, telemetry=False, weather=True, messages=True)

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
                        )
                        lap_times["LapNumber"] = lap_times["LapNumber"].astype(int)
                        lap_times = lap_times.sort_values("lap_end_time")

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
                    tmp = pd.merge_asof(
                        tmp,
                        lap_times,
                        left_on="Time",
                        right_on="lap_end_time",
                        direction="forward",
                    )
                    mdf = tmp
                else:
                    mdf["LapNumber"] = np.nan

                if msg_col and "LapNumber" in mdf.columns:
                    rows = []
                    for lap, g in mdf.dropna(subset=["LapNumber"]).groupby(
                        mdf["LapNumber"].astype(int)
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
                    left = lap_times[["LapNumber", "lap_end_time"]].sort_values(
                        "lap_end_time"
                    )
                    right = (
                        wdf[["Time"] + weather_cols]
                        .dropna(subset=["Time"])
                        .sort_values("Time")
                    )
                    if len(right):
                        merged = pd.merge_asof(
                            left,
                            right,
                            left_on="lap_end_time",
                            right_on="Time",
                            direction="backward",
                        )
                        rename = {
                            c: "ff1_weather_" + re.sub(r"[^A-Za-z0-9]+", "_", c).lower()
                            for c in weather_cols
                        }
                        merged = merged[["LapNumber"] + weather_cols].rename(
                            columns=rename
                        )
                        lap_ctx = lap_ctx.merge(merged, on="LapNumber", how="left")

            for c in [
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
            ]:
                if c not in lap_ctx.columns:
                    lap_ctx[c] = 0
                lap_ctx[c] = lap_ctx[c].fillna(0)

            event_cols = [
                c
                for c in lap_ctx.columns
                if c.startswith("ff1_track_") or c.startswith("ff1_msg_")
            ]
            for c in event_cols:
                lap_ctx[c + "_recent3"] = (
                    lap_ctx[c].rolling(3, min_periods=1).max().fillna(0)
                )

            context_parts.append(lap_ctx)
            meta["loaded"] += 1

        except Exception:
            meta["failed"] += 1
            continue

    if not context_parts:
        return pd.DataFrame(), meta

    context = pd.concat(context_parts, ignore_index=True)
    context = context.drop_duplicates(["Year", "Race", "LapNumber"])
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


def add_fastf1_context(train, test):
    base = pd.concat(
        [train[["Year", "Race", "LapNumber"]], test[["Year", "Race", "LapNumber"]]],
        ignore_index=True,
    ).drop_duplicates()

    context, meta = build_context_from_fastf1(base)

    if context.empty:
        for df in (train, test):
            df["ff1_context_available"] = 0
            df["ff1_track_yellow"] = 0
            df["ff1_track_sc"] = 0
            df["ff1_track_vsc"] = 0
            df["ff1_track_red"] = 0
            df["ff1_any_neutralized"] = 0
            df["ff1_any_neutralized_recent3"] = 0
            df["ff1_weather_rainfall"] = 0.0
        return train, test, meta

    train = train.merge(context, on=["Year", "Race", "LapNumber"], how="left")
    test = test.merge(context, on=["Year", "Race", "LapNumber"], how="left")

    ff_cols = [c for c in context.columns if c not in ["Year", "Race", "LapNumber"]]
    for df in (train, test):
        df["ff1_context_available"] = df[ff_cols].notna().any(axis=1).astype(int)
        for c in ff_cols:
            df[c] = df[c].fillna(0)
        for c in ["ff1_track_sc", "ff1_track_vsc", "ff1_track_red", "ff1_track_yellow"]:
            if c not in df.columns:
                df[c] = 0
        df["ff1_any_neutralized"] = df[
            ["ff1_track_sc", "ff1_track_vsc", "ff1_track_red"]
        ].max(axis=1)
        recent_cols = [
            c
            for c in [
                "ff1_track_sc_recent3",
                "ff1_track_vsc_recent3",
                "ff1_track_red_recent3",
            ]
            if c in df.columns
        ]
        df["ff1_any_neutralized_recent3"] = (
            df[recent_cols].max(axis=1) if recent_cols else 0
        )

    return train, test, meta


def train_model(X_train, y_train, X_valid, y_valid, cat_cols):
    try:
        from lightgbm import LGBMClassifier

        pos = max(float(y_train.sum()), 1.0)
        neg = max(float(len(y_train) - y_train.sum()), 1.0)
        model = LGBMClassifier(
            objective="binary",
            n_estimators=1200,
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
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            categorical_feature=cat_cols,
        )
        return model

    except Exception:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder

        num_cols = [c for c in X_train.columns if c not in cat_cols]
        pre = ColumnTransformer(
            [
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
        )
        model = Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=450,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        return model


train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

train = add_base_features(train)
test = add_base_features(test)
train, test, ff1_meta = add_fastf1_context(train, test)

drop_cols = [ID_COL, TARGET]
features = [c for c in train.columns if c not in drop_cols and c in test.columns]

combined = pd.concat([train[features], test[features]], axis=0, ignore_index=True)
cat_cols = combined.select_dtypes(include=["object", "category"]).columns.tolist()

for c in cat_cols:
    cats = pd.Categorical(combined[c].astype(str)).categories
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

for c in features:
    if c not in cat_cols:
        med = pd.to_numeric(train[c], errors="coerce").median()
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

y = train[TARGET].astype(int)
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
tr_idx, va_idx = next(splitter.split(train, y, groups=groups))

X_tr = train.iloc[tr_idx][features]
y_tr = y.iloc[tr_idx]
X_va = train.iloc[va_idx][features]
y_va = y.iloc[va_idx]

model = train_model(X_tr, y_tr, X_va, y_va, cat_cols)
valid_pred = model.predict_proba(X_va)[:, 1]
auc = roc_auc_score(y_va, valid_pred)

pd.DataFrame(
    {
        "row": va_idx,
        "target": y_va.values,
        "prediction": valid_pred,
    }
).to_csv(WORK_DIR / "validation_predictions.csv.gz", index=False, compression="gzip")

final_model = train_model(train[features], y, X_va, y_va, cat_cols)
test_pred = final_model.predict_proba(test[features])[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(WORK_DIR / "submission.csv", index=False)
submission.to_csv(WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip")

result = {
    "validation_roc_auc": float(auc),
    "research_hypotheses_llm_claimed_used": ["000664"],
    "fastf1_context": ff1_meta,
    "n_features": len(features),
    "n_train": int(len(train)),
    "n_valid": int(len(va_idx)),
    "n_test": int(len(test)),
}
print(json.dumps(result, sort_keys=True))
