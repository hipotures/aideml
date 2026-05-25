import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["is_test"] = 0
test["is_test"] = 1
test[TARGET] = np.nan
all_df = pd.concat([train, test], axis=0, ignore_index=True)

all_df["Race_Year"] = all_df["Race"].astype(str) + "_" + all_df["Year"].astype(str)
sort_cols = ["Year", "Race", "LapNumber", "Position", "Driver", "id"]
all_df = all_df.sort_values(sort_cols).reset_index(drop=True)


def add_past_only_strategy_features(df):
    df = df.copy()

    # Prior-lap rolling stint degradation and pace trends: shifted so current/future laps are excluded.
    keys = ["Race_Year", "Driver", "Stint"]
    g = df.groupby(keys, sort=False)
    shifted_deg = g["Cumulative_Degradation"].shift(1)
    shifted_lap = g["LapNumber"].shift(1)
    shifted_time = g["LapTime (s)"].shift(1)

    df["prior_deg_mean3"] = (
        shifted_deg.groupby([df[k] for k in keys], sort=False)
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )
    df["prior_laptime_mean3"] = (
        shifted_time.groupby([df[k] for k in keys], sort=False)
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )

    prior_deg_first = g["Cumulative_Degradation"].transform("first")
    prior_lap_first = g["LapNumber"].transform("first")
    elapsed_laps = (df["LapNumber"] - prior_lap_first).clip(lower=1)
    df["stint_deg_slope_prior"] = (
        ((shifted_deg - prior_deg_first) / elapsed_laps)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Race/lap-relative pace and traffic proxies use expanding history shifted within each race/lap bucket.
    lap_key = ["Race_Year", "LapNumber"]
    lap_mean_prior = df.groupby(lap_key, sort=False)["LapTime (s)"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    race_mean_prior = df.groupby("Race_Year", sort=False)["LapTime (s)"].transform(
        lambda s: s.shift(1).expanding(min_periods=10).mean()
    )
    global_prior_mean = df["LapTime (s)"].expanding(min_periods=10).mean().shift(1)

    pace_base = (
        lap_mean_prior.fillna(race_mean_prior)
        .fillna(global_prior_mean)
        .fillna(df["LapTime (s)"].median())
    )
    df["lap_relative_pace_prior"] = df["LapTime (s)"] - pace_base

    df["cars_ahead"] = (df["Position"] - 1).clip(lower=0)
    df["cars_behind"] = (20 - df["Position"]).clip(lower=0)
    df["traffic_pressure"] = (
        df["cars_ahead"] / 19.0
        + 0.15 * df["Position_Change"].clip(lower=0)
        + 0.10 * df["lap_relative_pace_prior"].clip(lower=0)
    )

    # Past-only pit penalty proxy: previous pit events in same race, with historical expanding fallback.
    df["prev_laptime_driver"] = df.groupby(["Race_Year", "Driver"], sort=False)[
        "LapTime (s)"
    ].shift(1)
    df["pit_event_penalty_raw"] = np.where(
        df["PitStop"].eq(1), df["LapTime (s)"] - df["prev_laptime_driver"], np.nan
    )
    df["pit_event_penalty_raw"] = df["pit_event_penalty_raw"].clip(lower=5, upper=90)

    same_race_prior_pit_loss = df.groupby("Race_Year", sort=False)[
        "pit_event_penalty_raw"
    ].transform(lambda s: s.shift(1).expanding(min_periods=1).median())
    historical_prior_pit_loss = (
        df["pit_event_penalty_raw"].expanding(min_periods=20).median().shift(1)
    )
    global_pit_loss = df["pit_event_penalty_raw"].median()
    if not np.isfinite(global_pit_loss):
        global_pit_loss = 22.0

    df["pit_loss_prior"] = same_race_prior_pit_loss.fillna(
        historical_prior_pit_loss
    ).fillna(global_pit_loss)

    # Counterfactual undercut-value features.
    compound_warmup = {
        "SOFT": 0.6,
        "MEDIUM": 1.0,
        "HARD": 1.4,
        "INTERMEDIATE": 1.8,
        "WET": 2.2,
    }
    df["warmup_penalty_proxy"] = df["Compound"].map(compound_warmup).fillna(1.2)

    df["stay_out_cost_1lap"] = (
        df["stint_deg_slope_prior"].clip(lower=-0.2, upper=8.0)
        + 0.18 * df["TyreLife"].clip(lower=0, upper=80)
        + 0.10 * df["traffic_pressure"].clip(lower=0, upper=20)
        + 0.20 * df["lap_relative_pace_prior"].clip(lower=0, upper=20)
    )

    df["expected_undercut_gain_now"] = (
        df["stay_out_cost_1lap"]
        + 0.25 * df["traffic_pressure"].clip(lower=0, upper=20)
        - df["warmup_penalty_proxy"]
    )

    df["pit_now_net_gain"] = (
        df["expected_undercut_gain_now"] - 0.06 * df["pit_loss_prior"]
    )
    df["pit_now_vs_next_lap_margin"] = (
        df["pit_now_net_gain"]
        + 0.12 * df["RaceProgress"].clip(0, 1) * df["TyreLife"].clip(0, 80)
        - 0.35 * (1.0 - df["RaceProgress"]).clip(0, 1)
    )

    df["fresh_tyre_finish_gap"] = (1.0 - df["RaceProgress"]) * df["LapNumber"].clip(
        lower=1
    ) - df["TyreLife"]
    df["late_race_window"] = (df["RaceProgress"] > 0.55).astype(int)
    df["mid_strategy_window"] = (
        (df["RaceProgress"] > 0.25) & (df["RaceProgress"] < 0.85)
    ).astype(int)

    drop_tmp = ["prev_laptime_driver", "pit_event_penalty_raw"]
    return df.drop(columns=drop_tmp)


all_df = add_past_only_strategy_features(all_df)
all_df = all_df.sort_values(ID_COL).reset_index(drop=True)

train_fe = all_df[all_df["is_test"].eq(0)].copy()
test_fe = all_df[all_df["is_test"].eq(1)].copy()

y = train_fe[TARGET].astype(int).values
drop_cols = [TARGET, ID_COL, "is_test"]
features = [c for c in train_fe.columns if c not in drop_cols]

for c in CAT_COLS + ["Race_Year"]:
    if c in features:
        combined = pd.concat([train_fe[c], test_fe[c]], axis=0).astype("category")
        train_fe[c] = pd.Categorical(train_fe[c], categories=combined.cat.categories)
        test_fe[c] = pd.Categorical(test_fe[c], categories=combined.cat.categories)

X = train_fe[features]
X_test = test_fe[features]
cat_features = [c for c in CAT_COLS + ["Race_Year"] if c in features]

groups = train_fe["Race_Year"].astype(str).values
if len(np.unique(groups)) >= 5:
    splitter = GroupKFold(n_splits=5)
    splits = splitter.split(X, y, groups)
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=359)
    splits = splitter.split(X, y)

oof = np.zeros(len(train_fe))
test_pred = np.zeros(len(test_fe))
fold_scores = []

params = dict(
    objective="binary",
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=2.0,
    class_weight="balanced",
    random_state=359,
    n_jobs=-1,
    verbosity=-1,
)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[],
    )
    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = va_pred
    fold_auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(fold_auc)
    test_pred += model.predict_proba(X_test)[:, 1] / 5.0
    print(f"fold {fold} roc_auc={fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame({"row": np.arange(len(train_fe)), "target": y, "prediction": oof}).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame({ID_COL: sample[ID_COL].values, TARGET: test_pred}).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": [float(x) for x in fold_scores],
            "research_hypotheses_llm_claimed_used": ["000359"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        },
        indent=2,
    )
)
