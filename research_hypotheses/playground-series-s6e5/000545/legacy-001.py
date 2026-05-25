import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 545
N_FOLDS = 5

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))
pred_col = (
    TARGET
    if TARGET in sample.columns
    else [c for c in sample.columns if c != ID_COL][0]
)


def clean_series(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_median(s, default):
    s = clean_series(s).dropna()
    return float(s.median()) if len(s) else float(default)


def estimate_strategy_params(df):
    valid = df[
        df["LapTime (s)"].between(60, 180)
        & df["TyreLife"].between(1, 90)
        & df["Cumulative_Degradation"].notna()
    ].copy()

    age = valid["TyreLife"].clip(lower=1)
    triangular_age = 0.5 * age * (age + 1)
    raw_slope = clean_series(valid["Cumulative_Degradation"]) / triangular_age
    raw_slope = raw_slope.where(raw_slope > 0)

    lo, hi = raw_slope.quantile([0.05, 0.95])
    slope_frame = valid.assign(_deg_slope=raw_slope)
    slope_frame = slope_frame[slope_frame["_deg_slope"].between(max(0.001, lo), hi)]
    global_slope = safe_median(slope_frame["_deg_slope"], 0.05)
    compound_slope = (
        slope_frame.groupby("Compound")["_deg_slope"]
        .median()
        .clip(0.005, 2.0)
        .to_dict()
    )

    delta = clean_series(df["LapTime_Delta"])
    base_delta = safe_median(delta[(df["PitStop"] == 0) & delta.between(-10, 20)], 0.0)
    pit_delta = safe_median(
        delta[(df["PitStop"] == 1) & delta.between(-5, 90)], base_delta + 22.0
    )
    global_pit_loss = float(np.clip(pit_delta - base_delta, 12.0, 35.0))

    tmp = df.assign(_delta=delta)
    pit_by_race = {}
    for race, g in tmp.groupby("Race"):
        g_base = safe_median(
            g.loc[(g["PitStop"] == 0) & g["_delta"].between(-10, 20), "_delta"],
            base_delta,
        )
        g_pit = safe_median(
            g.loc[(g["PitStop"] == 1) & g["_delta"].between(-5, 90), "_delta"],
            g_base + global_pit_loss,
        )
        pit_by_race[race] = float(np.clip(g_pit - g_base, 12.0, 35.0))

    fresh = valid[(valid["PitStop"] == 0) & (valid["TyreLife"] <= 3)]
    global_fresh_pace = safe_median(
        fresh["LapTime (s)"], safe_median(valid["LapTime (s)"], 90.0)
    )
    fresh_pace_by_compound = fresh.groupby("Compound")["LapTime (s)"].median().to_dict()

    life_by_compound = (
        df.groupby("Compound")["TyreLife"].quantile(0.90).clip(5, 90).to_dict()
    )
    global_life = float(np.clip(df["TyreLife"].quantile(0.90), 5, 90))
    best_life = (
        float(max(life_by_compound.values())) if life_by_compound else global_life
    )

    row_total = (df["LapNumber"] / df["RaceProgress"].clip(lower=0.01)).clip(20, 100)
    total_df = df.assign(
        _event_key=df["Year"].astype(str) + "||" + df["Race"].astype(str),
        _race_total=row_total,
    )
    event_total = (
        total_df.groupby("_event_key")["_race_total"].median().clip(20, 100).to_dict()
    )
    race_total = (
        total_df.groupby("Race")["_race_total"].median().clip(20, 100).to_dict()
    )

    return {
        "global_slope": global_slope,
        "compound_slope": compound_slope,
        "global_pit_loss": global_pit_loss,
        "pit_by_race": pit_by_race,
        "global_fresh_pace": global_fresh_pace,
        "fresh_pace_by_compound": fresh_pace_by_compound,
        "life_by_compound": life_by_compound,
        "global_life": global_life,
        "best_life": best_life,
        "event_total": event_total,
        "race_total": race_total,
    }


def add_strategy_features(df, params):
    out = df.copy()

    event_key = out["Year"].astype(str) + "||" + out["Race"].astype(str)
    row_total = (out["LapNumber"] / out["RaceProgress"].clip(lower=0.01)).clip(20, 100)
    event_total = event_key.map(params["event_total"])
    race_total = out["Race"].map(params["race_total"])
    total_laps = (
        event_total.fillna(race_total)
        .fillna(row_total)
        .clip(lower=out["LapNumber"], upper=100)
    )

    laps_remaining = (total_laps - out["LapNumber"]).clip(0, 100)
    age = out["TyreLife"].clip(1, 100)
    compound = out["Compound"]

    d = (
        compound.map(params["compound_slope"])
        .fillna(params["global_slope"])
        .astype(float)
        .clip(0.005, 2.0)
    )
    pit_loss = (
        out["Race"]
        .map(params["pit_by_race"])
        .fillna(params["global_pit_loss"])
        .astype(float)
        .clip(12, 35)
    )
    max_life = (
        compound.map(params["life_by_compound"])
        .fillna(params["global_life"])
        .astype(float)
    )
    fresh_pace_compound = (
        compound.map(params["fresh_pace_by_compound"])
        .fillna(params["global_fresh_pace"])
        .astype(float)
    )

    triangular_remaining_fresh = 0.5 * laps_remaining * (laps_remaining + 1)
    triangular_remaining_worn = age * laps_remaining + triangular_remaining_fresh

    current_lap_time = out["LapTime (s)"].clip(50, 300)
    local_fresh_pace = current_lap_time - d * np.maximum(age - 1, 0)
    local_fresh_pace = local_fresh_pace.clip(50, 180)

    out["strategy_race_total_est"] = total_laps
    out["strategy_laps_remaining"] = laps_remaining
    out["strategy_deg_slope"] = d
    out["strategy_pit_loss_est"] = pit_loss
    out["strategy_optimal_stint_len"] = np.sqrt((2.0 * pit_loss) / d).clip(1, 100)
    out["strategy_stint_minus_optimal"] = age - out["strategy_optimal_stint_len"]
    out["strategy_stint_over_optimal"] = age / out[
        "strategy_optimal_stint_len"
    ].replace(0, np.nan)
    out["strategy_expected_loss_next_lap"] = d * (age + 1)
    out["strategy_marginal_stop_now_gain"] = d * age
    out["strategy_pit_loss_per_remaining_lap"] = pit_loss / np.maximum(
        laps_remaining, 1
    )
    out["strategy_stay_to_finish_deg_cost"] = d * triangular_remaining_worn
    out["strategy_stop_now_finish_cost"] = pit_loss + d * triangular_remaining_fresh
    out["strategy_stop_now_value"] = (
        out["strategy_stay_to_finish_deg_cost"] - out["strategy_stop_now_finish_cost"]
    )
    out["strategy_stop_now_value_per_lap"] = out[
        "strategy_stop_now_value"
    ] / np.maximum(laps_remaining, 1)
    out["strategy_fresh_finish_margin_current_compound"] = max_life - laps_remaining
    out["strategy_fresh_finish_margin_best_compound"] = (
        params["best_life"] - laps_remaining
    )
    out["strategy_finishable_current_compound"] = (
        out["strategy_fresh_finish_margin_current_compound"] >= 0
    ).astype(np.int8)
    out["strategy_finishable_best_compound"] = (
        out["strategy_fresh_finish_margin_best_compound"] >= 0
    ).astype(np.int8)
    out["strategy_local_fresh_pace"] = local_fresh_pace
    out["strategy_fresh_pace_compound"] = fresh_pace_compound
    out["strategy_current_vs_fresh_pace"] = current_lap_time - local_fresh_pace
    out["strategy_fresh_pace_gap_to_compound"] = local_fresh_pace - fresh_pace_compound
    out["strategy_undercut_margin"] = (
        d * age - out["strategy_pit_loss_per_remaining_lap"]
    )
    out["strategy_undercut_feasible"] = (out["strategy_undercut_margin"] > 0).astype(
        np.int8
    )
    out["strategy_life_remaining_ratio"] = age / np.maximum(laps_remaining, 1)
    out["strategy_stop_pressure"] = out["strategy_stop_now_value"] / (pit_loss + 1.0)

    return out


def unique_clean_names(cols):
    used = {}
    mapping = {}
    for c in cols:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", c).strip("_")
        if not base:
            base = "feature"
        name = base
        i = 1
        while name in used:
            i += 1
            name = f"{base}_{i}"
        used[name] = True
        mapping[c] = name
    return mapping


params = estimate_strategy_params(train)
train_fe = add_strategy_features(train, params)
test_fe = add_strategy_features(test, params)

feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
cat_cols = [
    c
    for c in feature_cols
    if train_fe[c].dtype == "object" or test_fe[c].dtype == "object"
]

for c in cat_cols:
    combined = pd.concat([train_fe[c], test_fe[c]], axis=0).astype(str)
    categories = pd.Categorical(combined).categories
    train_fe[c] = pd.Categorical(train_fe[c].astype(str), categories=categories)
    test_fe[c] = pd.Categorical(test_fe[c].astype(str), categories=categories)

name_map = unique_clean_names(feature_cols)
X = train_fe[feature_cols].rename(columns=name_map)
X_test = test_fe[feature_cols].rename(columns=name_map)
cat_cols = [name_map[c] for c in cat_cols]
num_cols = [c for c in X.columns if c not in cat_cols]

X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
X_test[num_cols] = (
    X_test[num_cols].replace([np.inf, -np.inf], np.nan).astype(np.float32)
)

y = train[TARGET].astype(int).values
pos = max(float(y.sum()), 1.0)
scale_pos_weight = min((len(y) - pos) / pos, 50.0)

base_params = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    learning_rate=0.035,
    n_estimators=1800,
    num_leaves=63,
    min_child_samples=120,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1,
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(train), dtype=np.float32)
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = lgb.LGBMClassifier(**base_params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    fold_auc = roc_auc_score(y[va_idx], pred)
    best_iter = (
        model.best_iteration_ if model.best_iteration_ else base_params["n_estimators"]
    )
    best_iterations.append(best_iter)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f} | best_iteration={best_iter}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold CV ROC AUC: {cv_auc:.6f}")

final_iterations = int(
    np.clip(np.median(best_iterations), 100, base_params["n_estimators"])
)
final_params = dict(base_params)
final_params["n_estimators"] = final_iterations

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_cols)
test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[pred_col] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample.copy()
test_pred_df[pred_col] = test_pred
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "research_hypotheses_llm_claimed_used": ["000545"],
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "n_folds": N_FOLDS,
    "model": "single_lightgbm_classifier",
    "final_iterations": final_iterations,
    "estimated_global_pit_loss_seconds": float(params["global_pit_loss"]),
    "estimated_global_degradation_slope": float(params["global_slope"]),
    "outputs": {
        "submission": os.path.join(WORKING_DIR, "submission.csv"),
        "oof_predictions": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
        "test_predictions": os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    },
}
with open(os.path.join(WORKING_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review, indent=2))
