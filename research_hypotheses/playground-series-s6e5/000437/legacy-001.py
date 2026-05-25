import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

SEED = 2026
INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "id"
TARGET = "PitNextLap"

SUMMARY_COLS = [
    "RaceMeta_median_green_stop_total",
    "RaceMeta_mean_green_stop_total",
    "RaceMeta_total_green_stops",
    "RaceMeta_one_stop_rate",
    "RaceMeta_two_stop_rate",
    "RaceMeta_threeplus_stop_rate",
    "RaceMeta_median_pit_loss",
    "RaceMeta_pit_loss_iqr",
]

DEFAULT_PRIOR = {
    "RaceMeta_median_green_stop_total": 1.0,
    "RaceMeta_mean_green_stop_total": 1.35,
    "RaceMeta_total_green_stops": 27.0,
    "RaceMeta_one_stop_rate": 0.55,
    "RaceMeta_two_stop_rate": 0.35,
    "RaceMeta_threeplus_stop_rate": 0.05,
    "RaceMeta_median_pit_loss": 23.0,
    "RaceMeta_pit_loss_iqr": 5.0,
}

COMPOUND_HARDNESS = {
    "SOFT": 0.0,
    "MEDIUM": 1.0,
    "HARD": 2.0,
    "INTERMEDIATE": 1.25,
    "WET": 1.5,
}


def has_any(name, words):
    name = str(name).lower()
    return any(w in name for w in words)


def race_archetype(race):
    r = str(race).lower()
    if "testing" in r:
        return "testing"
    if has_any(r, ["monaco", "singapore", "las vegas", "azerbaijan"]):
        return "street_low_overtake"
    if has_any(r, ["bahrain", "spanish", "qatar", "hungarian", "japanese", "british"]):
        return "high_degradation"
    if has_any(r, ["italian", "belgian", "saudi", "austrian"]):
        return "high_speed"
    return "balanced"


def pirelli_hardness_bucket(race):
    r = str(race).lower()
    if "testing" in r:
        return 1.0
    if has_any(
        r,
        [
            "monaco",
            "singapore",
            "las vegas",
            "azerbaijan",
            "canadian",
            "australian",
            "italian",
            "mexico",
            "miami",
        ],
    ):
        return 0.0
    if has_any(
        r,
        [
            "bahrain",
            "spanish",
            "british",
            "japanese",
            "qatar",
            "saudi",
            "hungarian",
            "dutch",
        ],
    ):
        return 2.0
    return 1.0


def make_race_year_summaries(df):
    d = df.copy()
    d["Race"] = d["Race"].astype(str)
    d["Driver"] = d["Driver"].astype(str)
    d["RaceArchetype"] = d["Race"].map(race_archetype)

    is_event = ~d["Race"].str.lower().str.contains("testing", na=False)
    pit = pd.to_numeric(d["PitStop"], errors="coerce").fillna(0).astype(int).eq(1)
    lap = pd.to_numeric(d["LapNumber"], errors="coerce")
    progress = pd.to_numeric(d["RaceProgress"], errors="coerce")
    delta = pd.to_numeric(d["LapTime_Delta"], errors="coerce")

    monaco_anomaly = d["Race"].str.lower().str.contains("monaco", na=False) & (
        (lap <= 2) | (delta > 60)
    )
    green_stop = (
        pit & is_event & (lap > 1) & progress.between(0.03, 0.98) & ~monaco_anomaly
    )

    event_drivers = d.loc[is_event, ["Year", "Race", "Driver"]].drop_duplicates()
    stop_counts = (
        d.loc[green_stop]
        .groupby(["Year", "Race", "Driver"])
        .size()
        .rename("green_stop_count")
        .reset_index()
    )
    driver_stops = event_drivers.merge(
        stop_counts, on=["Year", "Race", "Driver"], how="left"
    )
    driver_stops["green_stop_count"] = driver_stops["green_stop_count"].fillna(0)

    race_stop = driver_stops.groupby(["Year", "Race"], as_index=False).agg(
        RaceMeta_median_green_stop_total=("green_stop_count", "median"),
        RaceMeta_mean_green_stop_total=("green_stop_count", "mean"),
        RaceMeta_total_green_stops=("green_stop_count", "sum"),
        RaceMeta_one_stop_rate=("green_stop_count", lambda s: float(np.mean(s == 1))),
        RaceMeta_two_stop_rate=("green_stop_count", lambda s: float(np.mean(s == 2))),
        RaceMeta_threeplus_stop_rate=(
            "green_stop_count",
            lambda s: float(np.mean(s >= 3)),
        ),
    )

    loss_mask = green_stop & delta.between(2, 90)
    if "LapTime_s" in d.columns:
        loss_mask &= pd.to_numeric(d["LapTime_s"], errors="coerce").lt(600)

    loss_rows = d.loc[loss_mask, ["Year", "Race", "LapTime_Delta"]].copy()
    if loss_rows.empty:
        loss_stats = pd.DataFrame(
            columns=[
                "Year",
                "Race",
                "RaceMeta_median_pit_loss",
                "RaceMeta_pit_loss_iqr",
            ]
        )
    else:
        q = (
            loss_rows.groupby(["Year", "Race"])["LapTime_Delta"]
            .quantile([0.25, 0.5, 0.75])
            .unstack()
        )
        q = q.rename(
            columns={0.25: "q25", 0.5: "RaceMeta_median_pit_loss", 0.75: "q75"}
        ).reset_index()
        q["RaceMeta_pit_loss_iqr"] = q["q75"] - q["q25"]
        loss_stats = q[
            ["Year", "Race", "RaceMeta_median_pit_loss", "RaceMeta_pit_loss_iqr"]
        ]

    out = race_stop.merge(loss_stats, on=["Year", "Race"], how="left")
    out["RaceArchetype"] = out["Race"].map(race_archetype)
    return out


def aggregate_history(hist, by):
    columns = [by] + SUMMARY_COLS + ["RaceMeta_prior_seasons"]
    if hist.empty:
        return pd.DataFrame(columns=columns)

    agg = {}
    for c in SUMMARY_COLS:
        agg[c] = "mean" if ("rate" in c or "mean" in c) else "median"

    out = hist.groupby(by, dropna=False).agg(agg)
    out = out.join(
        hist.groupby(by, dropna=False)["Year"]
        .nunique()
        .rename("RaceMeta_prior_seasons")
    )
    return out.reset_index()


def aggregate_global(hist):
    out = DEFAULT_PRIOR.copy()
    if hist.empty:
        out["RaceMeta_prior_seasons"] = 0.0
        return out

    for c in SUMMARY_COLS:
        vals = hist[c].dropna()
        if not vals.empty:
            out[c] = float(
                vals.mean() if ("rate" in c or "mean" in c) else vals.median()
            )
    out["RaceMeta_prior_seasons"] = float(hist["Year"].nunique())
    return out


def build_prior_features(df, summaries):
    base = df[["Race", "Year", "RaceArchetype"]].reset_index(drop=True).copy()
    base["_row_idx"] = np.arange(len(base))
    parts = []

    for year, subset in base.groupby("Year", sort=True):
        hist = summaries[summaries["Year"] < year]
        race_prior = aggregate_history(hist, "Race")
        arch_prior = aggregate_history(hist, "RaceArchetype")
        global_prior = aggregate_global(hist)

        part = subset.merge(race_prior, on="Race", how="left")
        race_has = part["RaceMeta_prior_seasons"].notna().to_numpy()
        part = part.merge(
            arch_prior, on="RaceArchetype", how="left", suffixes=("", "_arch")
        )
        arch_col = "RaceMeta_prior_seasons_arch"
        arch_has = (
            part[arch_col].notna().to_numpy()
            if arch_col in part.columns
            else np.zeros(len(part), dtype=bool)
        )

        for c in SUMMARY_COLS + ["RaceMeta_prior_seasons"]:
            if c not in part.columns:
                part[c] = np.nan
            ac = c + "_arch"
            if ac in part.columns:
                part[c] = part[c].fillna(part[ac])
            part[c] = part[c].fillna(global_prior.get(c, DEFAULT_PRIOR.get(c, 0.0)))

        has_global = global_prior["RaceMeta_prior_seasons"] > 0
        part["RaceMeta_prior_source"] = np.select(
            [race_has, arch_has, np.full(len(part), has_global)],
            [3, 2, 1],
            default=0,
        ).astype(np.int8)

        keep = (
            ["_row_idx"]
            + SUMMARY_COLS
            + ["RaceMeta_prior_seasons", "RaceMeta_prior_source"]
        )
        parts.append(part[keep])

    return (
        pd.concat(parts, axis=0)
        .sort_values("_row_idx")
        .drop(columns="_row_idx")
        .reset_index(drop=True)
    )


def add_features(df, summaries):
    out = df.copy()
    out["Race"] = out["Race"].astype(str)
    out["Driver"] = out["Driver"].astype(str)
    out["Compound"] = out["Compound"].astype(str).str.upper()
    out["RaceArchetype"] = out["Race"].map(race_archetype)
    out["PirelliHardnessBucket"] = (
        out["Race"].map(pirelli_hardness_bucket).astype(float)
    )
    out["CompoundHardness"] = (
        out["Compound"].map(COMPOUND_HARDNESS).fillna(1.0).astype(float)
    )

    lap = pd.to_numeric(out["LapNumber"], errors="coerce")
    progress = pd.to_numeric(out["RaceProgress"], errors="coerce").clip(
        lower=0.01, upper=1.0
    )
    total_laps = (
        (lap / progress).replace([np.inf, -np.inf], np.nan).clip(lower=30, upper=100)
    )
    out["EstimatedTotalLaps"] = total_laps
    out["LapsRemaining_Est"] = (total_laps - lap).clip(lower=0, upper=100)
    out["TyreLifeShare_Est"] = out["TyreLife"] / (
        out["TyreLife"] + out["LapsRemaining_Est"] + 1.0
    )

    priors = build_prior_features(out, summaries)
    out = pd.concat([out.reset_index(drop=True), priors], axis=1)

    out["RaceMeta_stop_balance"] = (
        out["RaceMeta_two_stop_rate"] - out["RaceMeta_one_stop_rate"]
    )
    out["PitLoss_x_LapsRemaining"] = (
        out["RaceMeta_median_pit_loss"] * out["LapsRemaining_Est"]
    )
    out["TwoStopRate_x_TyreLife"] = out["RaceMeta_two_stop_rate"] * out["TyreLife"]
    out["OneStopRate_x_LapsRemaining"] = (
        out["RaceMeta_one_stop_rate"] * out["LapsRemaining_Est"]
    )
    out["MedianStops_x_TyreLife"] = (
        out["RaceMeta_median_green_stop_total"] * out["TyreLife"]
    )
    out["PirelliBucket_x_TyreLife"] = out["PirelliHardnessBucket"] * out["TyreLife"]
    out["CompoundHardness_x_TyreLife"] = out["CompoundHardness"] * out["TyreLife"]
    out["PitLoss_x_CompoundHardness"] = (
        out["RaceMeta_median_pit_loss"] * out["CompoundHardness"]
    )
    out["StopBalance_x_TyreLife"] = out["RaceMeta_stop_balance"] * out["TyreLife"]
    return out.replace([np.inf, -np.inf], np.nan)


def make_model(y, n_estimators):
    pos = max(float(np.sum(y)), 1.0)
    neg = max(float(len(y) - np.sum(y)), 1.0)
    return LGBMClassifier(
        objective="binary",
        n_estimators=int(n_estimators),
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        scale_pos_weight=neg / pos,
        random_state=SEED,
        n_jobs=min(16, os.cpu_count() or 1),
        verbosity=-1,
    )


train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

train = train.rename(columns={"LapTime (s)": "LapTime_s"})
test = test.rename(columns={"LapTime (s)": "LapTime_s"})

y = train[TARGET].astype(int).to_numpy()
train_base = train.drop(columns=[TARGET])

summaries = make_race_year_summaries(train_base)
train_feat = add_features(train_base, summaries)
test_feat = add_features(test, summaries)

feature_cols = [c for c in train_feat.columns if c != ID_COL]
cat_cols = [c for c in feature_cols if train_feat[c].dtype == "object"]

for c in cat_cols:
    all_vals = (
        pd.concat([train_feat[c], test_feat[c]], axis=0)
        .astype("string")
        .fillna("__MISSING__")
        .astype(str)
    )
    cats = pd.Index(all_vals.unique())
    train_feat[c] = pd.Categorical(
        train_feat[c].astype("string").fillna("__MISSING__").astype(str),
        categories=cats,
    )
    test_feat[c] = pd.Categorical(
        test_feat[c].astype("string").fillna("__MISSING__").astype(str), categories=cats
    )

latest_year = int(train_feat["Year"].max())
valid_mask = train_feat["Year"].eq(latest_year).to_numpy()
train_mask = ~valid_mask

if (
    y[valid_mask].sum() == 0
    or y[valid_mask].sum() == valid_mask.sum()
    or y[train_mask].sum() == 0
):
    groups = train_feat["Race"].astype(str) + "_" + train_feat["Year"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(train_feat, y, groups))
    split_name = "group_holdout"
else:
    tr_idx = np.flatnonzero(train_mask)
    va_idx = np.flatnonzero(valid_mask)
    split_name = f"year_{latest_year}_holdout"

X_tr = train_feat.iloc[tr_idx][feature_cols]
X_va = train_feat.iloc[va_idx][feature_cols]
y_tr = y[tr_idx]
y_va = y[va_idx]

model = make_model(y_tr, n_estimators=1600)
model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="auc",
    categorical_feature=cat_cols,
    callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
)

valid_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
auc = roc_auc_score(y_va, valid_pred)
print(f"{split_name} ROC AUC: {auc:.6f}")

pd.DataFrame(
    {
        "row": va_idx,
        "target": y_va.astype(int),
        "prediction": valid_pred,
    }
).to_csv(WORK_DIR / "validation_predictions.csv.gz", index=False, compression="gzip")

best_iter = max(100, int(getattr(model, "best_iteration_", 0) or 800))
final_model = make_model(y, n_estimators=best_iter)
final_model.fit(
    train_feat[feature_cols],
    y,
    categorical_feature=cat_cols,
)

test_pred = np.clip(
    final_model.predict_proba(test_feat[feature_cols])[:, 1], 1e-6, 1 - 1e-6
)

submission = sample[[ID_COL, TARGET]].copy()
submission[TARGET] = test_pred
submission.to_csv(WORK_DIR / "submission.csv", index=False)
submission.to_csv(WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip")

result = {
    "validation_metric": "roc_auc",
    "validation_score": float(auc),
    "validation_split": split_name,
    "research_hypotheses_llm_claimed_used": ["000437"],
}
with open(WORK_DIR / "result_review.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, sort_keys=True))
