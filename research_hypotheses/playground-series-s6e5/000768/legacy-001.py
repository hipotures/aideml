import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.rename(columns={"LapTime (s)": "LapTime_s"})
test = test.rename(columns={"LapTime (s)": "LapTime_s"})

TARGET = "PitNextLap"
ID = "id"


def shrinkage_mean(df, keys, value, global_value, k):
    g = df.groupby(keys, dropna=False)[value].agg(["sum", "count"]).reset_index()
    g["prior"] = (g["sum"] + k * global_value) / (g["count"] + k)
    return g[keys + ["prior"]]


def add_strategy_features(train_df, test_df):
    train_part = train_df.drop(columns=[TARGET]).copy()
    test_part = test_df.copy()
    train_part["_is_train"] = 1
    test_part["_is_train"] = 0
    all_df = pd.concat([train_part, test_part], axis=0, ignore_index=True)

    pit_obs = train_df.loc[
        (train_df["PitStop"] == 1) & np.isfinite(train_df["LapTime_Delta"]),
        ["Year", "Race", "LapTime_Delta"],
    ].copy()
    pit_obs["PitLossObs"] = pit_obs["LapTime_Delta"].clip(5, 60)
    pit_obs = pit_obs[(pit_obs["PitLossObs"] >= 5) & (pit_obs["PitLossObs"] <= 60)]
    global_pit_loss = float(pit_obs["PitLossObs"].median()) if len(pit_obs) else 22.0
    if not np.isfinite(global_pit_loss):
        global_pit_loss = 22.0

    yr_race_prior = shrinkage_mean(
        pit_obs, ["Year", "Race"], "PitLossObs", global_pit_loss, 8
    )
    yr_race_prior = yr_race_prior.rename(columns={"prior": "PitLossPrior_YearRace"})
    race_prior = shrinkage_mean(pit_obs, ["Race"], "PitLossObs", global_pit_loss, 16)
    race_prior = race_prior.rename(columns={"prior": "PitLossPrior_Race"})

    all_df = all_df.merge(yr_race_prior, on=["Year", "Race"], how="left")
    all_df = all_df.merge(race_prior, on=["Race"], how="left")
    all_df["RacePitLossPrior"] = (
        all_df["PitLossPrior_YearRace"]
        .fillna(all_df["PitLossPrior_Race"])
        .fillna(global_pit_loss)
        .clip(5, 60)
    )

    race_max_lap = all_df.groupby(["Year", "Race"])["LapNumber"].transform("max")
    all_df["RaceMaxLap"] = race_max_lap
    all_df["RemainingLaps"] = (all_df["RaceMaxLap"] - all_df["LapNumber"]).clip(lower=0)
    all_df["LapFracOfRace"] = all_df["LapNumber"] / all_df["RaceMaxLap"].replace(
        0, np.nan
    )

    rate_train = (
        train_df["Cumulative_Degradation"].clip(lower=0)
        / train_df["TyreLife"].clip(lower=1)
    ).replace([np.inf, -np.inf], np.nan)
    global_d = float(rate_train.replace(0, np.nan).median())
    if not np.isfinite(global_d) or global_d <= 0:
        global_d = 0.25
    d_upper = float(rate_train.quantile(0.98))
    if not np.isfinite(d_upper):
        d_upper = 5.0
    d_upper = float(np.clip(d_upper, 0.75, 8.0))

    comp_rate = train_df[["Compound", "Cumulative_Degradation", "TyreLife"]].copy()
    comp_rate["CompoundDegRatePrior"] = (
        comp_rate["Cumulative_Degradation"].clip(lower=0)
        / comp_rate["TyreLife"].clip(lower=1)
    ).clip(0.03, d_upper)
    comp_prior = (
        comp_rate.groupby("Compound")["CompoundDegRatePrior"].median().to_dict()
    )
    all_df["CompoundDegRatePrior"] = all_df["Compound"].map(comp_prior).fillna(global_d)

    sort_cols = ["Year", "Race", "Driver", "Stint", "LapNumber", ID]
    sdf = all_df.sort_values(sort_cols).copy()
    grp = sdf.groupby(["Year", "Race", "Driver", "Stint"], sort=False)

    sdf["StintLapIndex"] = grp.cumcount() + 1
    sdf["PrevLapTime"] = grp["LapTime_s"].shift(1)
    sdf["LapTimeIncrease"] = (sdf["LapTime_s"] - sdf["PrevLapTime"]).clip(
        lower=0, upper=d_upper
    )
    sdf["StintBestLapSoFar"] = grp["LapTime_s"].cummin()
    sdf["LossFromBestRate"] = (
        (sdf["LapTime_s"] - sdf["StintBestLapSoFar"]).clip(lower=0)
        / sdf["StintLapIndex"].clip(lower=1)
    ).clip(0.03, d_upper)
    sdf["PosDeltaForRoll"] = sdf["LapTime_Delta"].clip(lower=0, upper=d_upper)
    sdf["RollingPosDelta4"] = grp["PosDeltaForRoll"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )

    causal_cols = [
        "StintLapIndex",
        "LapTimeIncrease",
        "StintBestLapSoFar",
        "LossFromBestRate",
        "RollingPosDelta4",
    ]
    all_df.loc[sdf.index, causal_cols] = sdf[causal_cols]

    cum_rate = (
        all_df["Cumulative_Degradation"].clip(lower=0)
        / all_df["TyreLife"].clip(lower=1)
    ).clip(0.03, d_upper)
    d_components = pd.concat(
        [
            cum_rate.rename("CumDegRate"),
            all_df["RollingPosDelta4"].clip(0.03, d_upper),
            all_df["LossFromBestRate"].clip(0.03, d_upper),
            all_df["CompoundDegRatePrior"].clip(0.03, d_upper),
        ],
        axis=1,
    )
    all_df["LocalDegRate"] = (
        d_components.mean(axis=1).fillna(global_d).clip(0.03, d_upper)
    )

    p = all_df["RacePitLossPrior"].clip(5, 60)
    d = all_df["LocalDegRate"].clip(0.03, d_upper)
    optimal = np.sqrt((2.0 * p) / d).clip(2, 90)

    all_df["Optimal_Stint_Length"] = optimal
    all_df["TyreLife_minus_Optimal"] = all_df["TyreLife"] - optimal
    all_df["RemainingLaps_minus_OptimalFresh"] = all_df["RemainingLaps"] - optimal
    all_df["OldTyreNextLapLoss"] = d * (all_df["TyreLife"] + 1.0)
    all_df["FreshTyreRecoveryNextLap"] = d * all_df["TyreLife"]
    all_df["Undercut_Advantage"] = all_df["FreshTyreRecoveryNextLap"] - p
    horizon = np.minimum(all_df["RemainingLaps"].clip(lower=0), optimal)
    all_df["Undercut_Advantage_Horizon"] = d * all_df["TyreLife"] * horizon - p
    all_df["PitLoss_to_DegRate"] = p / d
    all_df["TyreLife_to_OptimalRatio"] = all_df["TyreLife"] / (optimal + 1e-6)
    all_df["RemainingLaps_to_OptimalRatio"] = all_df["RemainingLaps"] / (optimal + 1e-6)

    all_df = all_df.drop(
        columns=["PitLossPrior_YearRace", "PitLossPrior_Race"], errors="ignore"
    )
    train_feat = (
        all_df.loc[all_df["_is_train"] == 1]
        .drop(columns=["_is_train"])
        .reset_index(drop=True)
    )
    test_feat = (
        all_df.loc[all_df["_is_train"] == 0]
        .drop(columns=["_is_train"])
        .reset_index(drop=True)
    )
    return train_feat, test_feat


train_feat, test_feat = add_strategy_features(train, test)
y = train[TARGET].astype(int).values

drop_cols = [ID]
X = train_feat.drop(columns=drop_cols)
X_test = test_feat.drop(columns=drop_cols)

cat_cols = [c for c in X.columns if X[c].dtype == "object"]
cat_cols += [
    c
    for c in ["Year", "Compound", "Race", "Driver"]
    if c in X.columns and c not in cat_cols
]
cat_cols = list(dict.fromkeys(cat_cols))

combined = pd.concat([X, X_test], axis=0, ignore_index=True)
for c in cat_cols:
    combined[c] = combined[c].astype("category")

for c in combined.columns:
    if c not in cat_cols:
        combined[c] = pd.to_numeric(combined[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )


def clean_feature_names(cols):
    used = {}
    out = []
    for c in cols:
        name = re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_")
        name = name or "feature"
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 0
        out.append(name)
    return out


old_cols = list(combined.columns)
new_cols = clean_feature_names(old_cols)
rename_map = dict(zip(old_cols, new_cols))
combined.columns = new_cols
cat_cols = [rename_map[c] for c in cat_cols]

X = combined.iloc[: len(train)].reset_index(drop=True)
X_test = combined.iloc[len(train) :].reset_index(drop=True)

groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

pos = y.sum()
neg = len(y) - pos
scale_pos_weight = float(neg / max(pos, 1))

params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=2500,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=60,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_alpha=0.05,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )
    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = va_pred
    fold_auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(fold_auc)
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold grouped ROC AUC: {cv_auc:.6f}")

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "5fold_stratified_group_roc_auc",
    "score": float(cv_auc),
    "fold_scores": [float(x) for x in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000768"],
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
