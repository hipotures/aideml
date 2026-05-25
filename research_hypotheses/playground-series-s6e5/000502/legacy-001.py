import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
base_features = [c for c in test.columns if c != ID_COL]


def robust_median_map(df, keys, value, min_count=8):
    stats = df.groupby(keys)[value].agg(["median", "count"]).reset_index()
    stats.loc[stats["count"] < min_count, "median"] = np.nan
    return stats[keys + ["median"]].rename(columns={"median": value + "_map"})


def add_pit_economics_features(frame, ref):
    out = frame.copy()
    ref = ref.copy()

    global_lap_median = ref["LapTime (s)"].median()
    global_pit_excess = (
        ref.loc[ref["PitStop"] == 1, "LapTime (s)"] - global_lap_median
    ).median()
    if not np.isfinite(global_pit_excess):
        global_pit_excess = 22.0

    ref["race_year_median_lap"] = ref.groupby(["Race", "Year"])[
        "LapTime (s)"
    ].transform("median")
    ref["race_median_lap"] = ref.groupby("Race")["LapTime (s)"].transform("median")
    ref["lap_excess"] = ref["LapTime (s)"] - ref["race_year_median_lap"]

    pit_rows = ref[ref["PitStop"] == 1].copy()
    ry_pit = robust_median_map(pit_rows, ["Race", "Year"], "lap_excess", 5)
    r_pit = robust_median_map(pit_rows, ["Race"], "lap_excess", 10).rename(
        columns={"lap_excess_map": "race_pit_excess"}
    )
    out = out.merge(
        ry_pit.rename(columns={"lap_excess_map": "race_year_pit_excess"}),
        on=["Race", "Year"],
        how="left",
    )
    out = out.merge(r_pit, on="Race", how="left")
    out["estimated_pit_loss"] = (
        out["race_year_pit_excess"]
        .fillna(out["race_pit_excess"])
        .fillna(global_pit_excess)
    )

    warm_ref = ref[ref["TyreLife"].between(1, 3)].copy()
    warm_ref["warm_excess"] = warm_ref["LapTime (s)"] - warm_ref["race_year_median_lap"]
    global_warm = warm_ref.groupby("TyreLife")["warm_excess"].median().to_dict()
    for tl in [1, 2, 3]:
        col = f"warmup_penalty_lap{tl}"
        tmp = warm_ref[warm_ref["TyreLife"].round().astype(int) == tl]
        ry = robust_median_map(tmp, ["Race", "Year"], "warm_excess", 5).rename(
            columns={"warm_excess_map": col}
        )
        rr = robust_median_map(tmp, ["Race"], "warm_excess", 10).rename(
            columns={"warm_excess_map": col + "_race"}
        )
        out = out.merge(ry, on=["Race", "Year"], how="left")
        out = out.merge(rr, on="Race", how="left")
        out[col] = (
            out[col].fillna(out[col + "_race"]).fillna(global_warm.get(float(tl), 0.0))
        )
        out.drop(columns=[col + "_race"], inplace=True)

    ref["deg_per_lap"] = ref["Cumulative_Degradation"] / ref["TyreLife"].clip(lower=1)
    global_deg = ref["deg_per_lap"].replace([np.inf, -np.inf], np.nan).median()
    if not np.isfinite(global_deg):
        global_deg = 0.0

    deg_ry = robust_median_map(
        ref, ["Race", "Year", "Compound"], "deg_per_lap", 20
    ).rename(columns={"deg_per_lap_map": "deg_slope_ry_compound"})
    deg_r = robust_median_map(ref, ["Race", "Compound"], "deg_per_lap", 40).rename(
        columns={"deg_per_lap_map": "deg_slope_r_compound"}
    )
    out = out.merge(deg_ry, on=["Race", "Year", "Compound"], how="left")
    out = out.merge(deg_r, on=["Race", "Compound"], how="left")
    out["current_stint_degradation_slope"] = (
        out["deg_slope_ry_compound"]
        .fillna(out["deg_slope_r_compound"])
        .fillna(global_deg)
    )

    fresh_warm_total = out[[f"warmup_penalty_lap{i}" for i in [1, 2, 3]]].sum(axis=1)
    current_next3_degradation = out["current_stint_degradation_slope"] * (
        out["TyreLife"] + 1 + out["TyreLife"] + 2 + out["TyreLife"] + 3
    )
    fresh_next3_degradation = out["current_stint_degradation_slope"] * (1 + 2 + 3)
    out["expected_next_3_lap_gain_minus_pit_loss"] = (
        current_next3_degradation
        - fresh_next3_degradation
        - fresh_warm_total
        - out["estimated_pit_loss"]
    )

    per_lap_gain = (out["current_stint_degradation_slope"] * out["TyreLife"]).clip(
        lower=0.05
    )
    out["break_even_laps_to_recover_stop"] = (
        (out["estimated_pit_loss"] + fresh_warm_total.clip(lower=0)) / per_lap_gain
    ).clip(0, 200)

    out["pit_loss_vs_progress"] = out["estimated_pit_loss"] * (
        1.0 - out["RaceProgress"]
    )
    out["utility_per_lap_remaining"] = out[
        "expected_next_3_lap_gain_minus_pit_loss"
    ] / ((1.0 - out["RaceProgress"]).clip(lower=0.01) * out["LapNumber"].clip(lower=1))

    drop_cols = [
        "race_year_pit_excess",
        "race_pit_excess",
        "deg_slope_ry_compound",
        "deg_slope_r_compound",
    ]
    out.drop(columns=[c for c in drop_cols if c in out.columns], inplace=True)
    return out


oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
fold_scores = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=502)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
    tr_raw = train.iloc[tr_idx].reset_index(drop=True)
    va_raw = train.iloc[va_idx].reset_index(drop=True)
    te_raw = test.copy()

    tr_feat = add_pit_economics_features(tr_raw[base_features], tr_raw)
    va_feat = add_pit_economics_features(va_raw[base_features], tr_raw)
    te_feat = add_pit_economics_features(te_raw[base_features], tr_raw)

    for c in CAT_COLS:
        tr_feat[c] = tr_feat[c].astype("category")
        va_feat[c] = pd.Categorical(va_feat[c], categories=tr_feat[c].cat.categories)
        te_feat[c] = pd.Categorical(te_feat[c], categories=tr_feat[c].cat.categories)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=64,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=502 + fold,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        tr_feat,
        y[tr_idx],
        eval_set=[(va_feat, y[va_idx])],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[],
    )

    va_pred = model.predict_proba(va_feat)[:, 1]
    te_pred = model.predict_proba(te_feat)[:, 1]
    oof[va_idx] = va_pred
    test_pred += te_pred / skf.n_splits

    auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(auc)
    print(f"fold {fold} roc_auc: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"mean_fold_roc_auc: {np.mean(fold_scores):.6f}")
print(f"oof_roc_auc: {cv_auc:.6f}")

pd.DataFrame({"row": np.arange(len(train)), "target": y, "prediction": oof}).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_predictions = sample[[ID_COL]].copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
test_predictions.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

with open(os.path.join(WORKING_DIR, "result_review.json"), "w") as f:
    json.dump(
        {
            "research_hypotheses_llm_claimed_used": ["000502"],
            "metric": "roc_auc",
            "oof_roc_auc": float(cv_auc),
            "mean_fold_roc_auc": float(np.mean(fold_scores)),
        },
        f,
        indent=2,
    )
