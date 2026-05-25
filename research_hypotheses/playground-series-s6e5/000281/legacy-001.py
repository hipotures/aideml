import gc
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

ID_COL = "id"
TARGET = "PitNextLap"
RANDOM_STATE = 2026


def clean_name(name):
    if name in (ID_COL, TARGET):
        return name
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def clean_columns(df):
    return df.rename(columns={c: clean_name(c) for c in df.columns})


train = clean_columns(pd.read_csv(INPUT / "train.csv.gz"))
test = clean_columns(pd.read_csv(INPUT / "test.csv.gz"))
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")


def add_sequence_state(df):
    df = df.copy()
    for c in ["Race", "Driver", "Compound"]:
        df[c] = df[c].astype(str)

    df["_row_order"] = np.arange(len(df))
    d = df.sort_values(["Year", "Race", "Driver", "LapNumber", ID_COL]).copy()
    gcols = ["Year", "Race", "Driver"]
    g = d.groupby(gcols, sort=False)

    d["prior_pit_stop"] = g["PitStop"].shift(1).fillna(0).astype(np.int8)
    d["pit_stop_2laps_ago"] = g["PitStop"].shift(2).fillna(0).astype(np.int8)
    d["pit_count_before_lap"] = (
        d.groupby(gcols, sort=False)["prior_pit_stop"].cumsum().astype(np.int16)
    )
    d["seq_stint_no"] = (d["pit_count_before_lap"] + 1).astype(np.int16)
    d["seq_stint_lap"] = (
        d.groupby(gcols + ["seq_stint_no"], sort=False)
        .cumcount()
        .add(1)
        .astype(np.int16)
    )

    stop_lap = d["LapNumber"].where(d["PitStop"].eq(1))
    last_stop_prior = stop_lap.groupby([d[c] for c in gcols], sort=False).ffill()
    last_stop_prior = (
        last_stop_prior.groupby([d[c] for c in gcols], sort=False).shift(1).fillna(0)
    )
    d["laps_since_last_stop_seq"] = (
        (d["LapNumber"] - last_stop_prior).clip(lower=0).astype(np.float32)
    )

    progress = d["RaceProgress"].replace(0, np.nan).astype(float)
    total_est = (
        (d["LapNumber"] / progress)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(d["LapNumber"])
    )
    total_est = np.clip(
        np.maximum(total_est.to_numpy(), d["LapNumber"].to_numpy()), 1, 120
    )
    d["race_total_est"] = total_est.astype(np.float32)
    d["race_laps_remaining_est"] = (
        (d["race_total_est"] - d["LapNumber"]).clip(lower=0).astype(np.float32)
    )

    age = np.maximum(
        d["TyreLife"].astype(float).to_numpy(),
        d["seq_stint_lap"].astype(float).to_numpy(),
    )
    d["stint_age_pct_in_race"] = np.clip(age / np.maximum(total_est, 1), 0, 2).astype(
        np.float32
    )
    d["tyrelife_pct_in_race"] = np.clip(
        d["TyreLife"].to_numpy() / np.maximum(total_est, 1), 0, 2
    ).astype(np.float32)
    d["degradation_per_tyre_lap"] = (
        d["Cumulative_Degradation"] / np.maximum(d["TyreLife"], 1)
    ).astype(np.float32)
    d["lap_delta_per_tyre_lap"] = (
        d["LapTime_Delta"] / np.maximum(d["TyreLife"], 1)
    ).astype(np.float32)
    d["late_stint_pressure"] = (
        d["RaceProgress"].astype(float).to_numpy() * np.log1p(np.maximum(age, 0))
    ).astype(np.float32)

    d["is_wet"] = d["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    d["stint_age_bin"] = np.digitize(
        age, np.array([3, 6, 9, 12, 15, 18, 22, 26, 32, 40, 60, 100]), right=True
    ).astype(np.int16)
    d["lap_bin"] = np.digitize(
        d["LapNumber"].to_numpy(), np.arange(5, 86, 5), right=True
    ).astype(np.int16)
    d["race_progress_bin"] = np.minimum(
        (d["RaceProgress"].to_numpy() * 10).astype(int), 9
    ).astype(np.int16)

    d["lapbin_compound"] = d["lap_bin"].astype(str) + "_" + d["Compound"]
    d["lapbin_wet"] = d["lap_bin"].astype(str) + "_" + d["is_wet"].astype(str)
    d["agebin_compound"] = d["stint_age_bin"].astype(str) + "_" + d["Compound"]
    d["race_year"] = d["Year"].astype(str) + "_" + d["Race"]

    d = d.sort_values("_row_order").drop(columns=["_row_order"]).reset_index(drop=True)
    return d


train_base = add_sequence_state(train)
test_base = add_sequence_state(test)

RATE_SPECS = [
    (["stint_age_bin", "Compound"], "hazard_age_compound", 40.0),
    (["lap_bin", "Compound", "is_wet"], "hazard_lap_compound_wet", 60.0),
    (["race_progress_bin", "stint_age_bin"], "hazard_progress_age", 60.0),
]
PRIOR_COLS = [
    "hazard_age_compound",
    "hazard_lap_compound_wet",
    "hazard_progress_age",
    "survival_age_compound",
    "neg_log_survival_age_compound",
]


def add_rate_feature(out, fit_df, pred_df, keys, col, alpha, training_mode):
    prior = float(fit_df[TARGET].mean())
    agg = (
        fit_df.groupby(keys, observed=True)[TARGET].agg(["sum", "count"]).reset_index()
    )

    if training_mode:
        stats = pred_df[keys + [TARGET]].merge(agg, on=keys, how="left", sort=False)
        sums = stats["sum"].fillna(0).astype(float).to_numpy()
        counts = stats["count"].fillna(0).astype(float).to_numpy()
        y = stats[TARGET].astype(float).to_numpy()
        out[col] = ((sums - y) + alpha * prior) / (np.maximum(counts - 1, 0) + alpha)
    else:
        agg[col] = (agg["sum"] + alpha * prior) / (agg["count"] + alpha)
        out = out.merge(agg[keys + [col]], on=keys, how="left", sort=False)
        out[col] = out[col].fillna(prior)

    out[col] = out[col].astype(np.float32)
    return out


def add_survival_feature(out, fit_df, pred_df):
    prior = float(fit_df[TARGET].mean())
    alpha = 50.0
    compounds = pd.Index(
        pd.concat([fit_df["Compound"], pred_df["Compound"]], ignore_index=True)
        .astype(str)
        .unique()
    )
    age_bins = np.sort(
        pd.concat(
            [fit_df["stint_age_bin"], pred_df["stint_age_bin"]], ignore_index=True
        )
        .dropna()
        .unique()
    )

    grid = pd.MultiIndex.from_product(
        [compounds, age_bins], names=["Compound", "stint_age_bin"]
    ).to_frame(index=False)
    agg = (
        fit_df.groupby(["Compound", "stint_age_bin"], observed=True)[TARGET]
        .agg(["sum", "count"])
        .reset_index()
    )
    table = grid.merge(agg, on=["Compound", "stint_age_bin"], how="left", sort=False)
    table["hazard"] = (table["sum"].fillna(0) + alpha * prior) / (
        table["count"].fillna(0) + alpha
    )
    table = table.sort_values(["Compound", "stint_age_bin"])
    table["survival_age_compound"] = table.groupby("Compound")["hazard"].transform(
        lambda s: np.cumprod(1.0 - np.clip(s.to_numpy(dtype=float), 1e-5, 0.95))
    )

    out = out.merge(
        table[["Compound", "stint_age_bin", "survival_age_compound"]],
        on=["Compound", "stint_age_bin"],
        how="left",
        sort=False,
    )
    default_survival = np.power(
        max(1.0 - prior, 1e-5), out["stint_age_bin"].astype(float) + 1.0
    )
    out["survival_age_compound"] = out["survival_age_compound"].fillna(
        pd.Series(default_survival, index=out.index)
    )
    out["survival_age_compound"] = (
        out["survival_age_compound"].clip(1e-6, 1.0).astype(np.float32)
    )
    out["neg_log_survival_age_compound"] = (
        -np.log(out["survival_age_compound"])
    ).astype(np.float32)
    return out


def add_prior_features(fit_df, pred_df, training_mode=False):
    out = pred_df.copy()
    for keys, col, alpha in RATE_SPECS:
        out = add_rate_feature(out, fit_df, pred_df, keys, col, alpha, training_mode)
    out = add_survival_feature(out, fit_df, pred_df)
    return out


CAT_COLS = [
    "Compound",
    "Driver",
    "Race",
    "Year",
    "lap_bin",
    "stint_age_bin",
    "race_progress_bin",
    "lapbin_compound",
    "lapbin_wet",
    "agebin_compound",
]
category_values = {}
category_is_string = {}
for col in CAT_COLS:
    vals = pd.concat([train_base[col], test_base[col]], ignore_index=True)
    category_is_string[col] = not pd.api.types.is_numeric_dtype(vals)
    if category_is_string[col]:
        category_values[col] = pd.Index(vals.astype(str).dropna().unique())
    else:
        category_values[col] = np.sort(vals.dropna().unique())


def cast_categories(df):
    out = df.copy()
    for col in CAT_COLS:
        if category_is_string[col]:
            out[col] = pd.Categorical(
                out[col].astype(str), categories=category_values[col]
            )
        else:
            out[col] = pd.Categorical(out[col], categories=category_values[col])
    return out


EXCLUDE = {ID_COL, TARGET, "race_year"}
FEATURES = [c for c in train_base.columns if c not in EXCLUDE] + PRIOR_COLS
CAT_FEATURES = [c for c in CAT_COLS if c in FEATURES]


def make_model(n_estimators=900):
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        random_state=RANDOM_STATE,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )


y = train_base[TARGET].astype(int).to_numpy()
groups = train_base["race_year"].astype(str).to_numpy()
oof = np.zeros(len(train_base), dtype=np.float32)
fold_scores = []
best_iterations = []

gkf = GroupKFold(n_splits=5)
for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_base, y, groups), 1):
    fit_part = train_base.iloc[tr_idx].reset_index(drop=True)
    valid_part = train_base.iloc[va_idx].reset_index(drop=True)

    x_train = cast_categories(
        add_prior_features(fit_part, fit_part, training_mode=True)
    )
    x_valid = cast_categories(
        add_prior_features(fit_part, valid_part, training_mode=False)
    )

    model = make_model()
    model.fit(
        x_train[FEATURES],
        fit_part[TARGET].astype(int),
        eval_set=[(x_valid[FEATURES], valid_part[TARGET].astype(int))],
        eval_metric="auc",
        categorical_feature=CAT_FEATURES,
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )

    pred = model.predict_proba(x_valid[FEATURES])[:, 1]
    oof[va_idx] = pred.astype(np.float32)
    score = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(score))
    best_iterations.append(int(model.best_iteration_ or model.n_estimators))
    print(f"fold {fold} ROC AUC: {score:.6f}")

    del fit_part, valid_part, x_train, x_valid, model, pred
    gc.collect()

cv_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {"row": np.arange(len(train_base)), "target": y, "prediction": oof}
).to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

final_estimators = int(np.median(best_iterations)) if best_iterations else 700
final_estimators = max(100, final_estimators)
print(f"Training final model with {final_estimators} trees")

x_all = cast_categories(add_prior_features(train_base, train_base, training_mode=True))
x_test = cast_categories(add_prior_features(train_base, test_base, training_mode=False))

final_model = make_model(n_estimators=final_estimators)
final_model.fit(
    x_all[FEATURES],
    y,
    categorical_feature=CAT_FEATURES,
)

test_pred = np.clip(final_model.predict_proba(x_test[FEATURES])[:, 1], 0, 1)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(WORK / "submission.csv", index=False)

test_predictions = sample[[ID_COL]].copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    WORK / "test_predictions.csv.gz", index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000281"],
    "metric": "roc_auc",
    "cv_score": float(cv_auc),
    "fold_scores": fold_scores,
    "n_features": len(FEATURES),
    "n_folds": 5,
}
with open(WORK / "result_review.json", "w") as f:
    json.dump(review, f, indent=2)

print(f"Saved submission to {WORK / 'submission.csv'}")
