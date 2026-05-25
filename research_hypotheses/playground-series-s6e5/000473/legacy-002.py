import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
LAPTIME_COL = "LapTime (s)"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["_is_train"] = 1
test["_is_train"] = 0
test[TARGET] = np.nan
df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
df["_row_order"] = np.arange(len(df))


def map_group_stat(frame, cols, value):
    stat = frame.groupby(cols, sort=False)[value].median()
    if isinstance(cols, str):
        return frame[cols].map(stat)
    keys = pd.MultiIndex.from_frame(frame[cols])
    return pd.Series(keys.map(stat), index=frame.index)


def add_hypothesis_000473_features(frame):
    frame = frame.copy()

    frame["RaceYear"] = frame["Race"].astype(str) + "_" + frame["Year"].astype(str)
    frame["RaceYearCompound"] = frame["RaceYear"] + "_" + frame["Compound"].astype(str)
    frame["RaceYearCompoundStint"] = (
        frame["RaceYearCompound"] + "_" + frame["Stint"].astype(str)
    )

    lo, hi = frame[LAPTIME_COL].quantile([0.005, 0.995])
    frame["LapTime_winsor"] = frame[LAPTIME_COL].clip(lower=lo, upper=hi)

    frame["race_lap_median_pace"] = frame.groupby(
        ["RaceYear", "LapNumber"], sort=False
    )["LapTime_winsor"].transform("median")
    frame["race_median_pace"] = frame.groupby("RaceYear", sort=False)[
        "LapTime_winsor"
    ].transform("median")

    base_pace = frame["race_lap_median_pace"].fillna(frame["race_median_pace"])
    frame["pace_resid"] = frame["LapTime_winsor"] - base_pace

    delta_cut = frame["LapTime_Delta"].abs().quantile(0.98)
    clean = (
        frame["PitStop"].fillna(0).eq(0)
        & frame["TyreLife"].notna()
        & frame["pace_resid"].notna()
        & frame["LapTime_Delta"].abs().le(delta_cut)
    )

    group_cols = ["RaceYear", "Compound", "Stint"]
    order = frame.sort_values(group_cols + ["LapNumber", "_row_order"]).index
    tmp = frame.loc[order, group_cols + ["TyreLife", "pace_resid"]].copy()
    valid = clean.loc[order].astype(float)

    x = tmp["TyreLife"].astype(float).where(valid.astype(bool), 0.0)
    y_resid = tmp["pace_resid"].astype(float).where(valid.astype(bool), 0.0)

    tmp["_n"] = valid.values
    tmp["_x"] = x.values
    tmp["_y"] = y_resid.values
    tmp["_xx"] = (x * x).values
    tmp["_xy"] = (x * y_resid).values

    gb = tmp.groupby(group_cols, sort=False)
    n = gb["_n"].cumsum()
    sx = gb["_x"].cumsum()
    sy = gb["_y"].cumsum()
    sxx = gb["_xx"].cumsum()
    sxy = gb["_xy"].cumsum()

    n_safe = n.replace(0, np.nan)
    denom = sxx - (sx * sx / n_safe)
    numer = sxy - (sx * sy / n_safe)
    slope = numer / denom
    slope[(n < 6) | (denom <= 1e-8) | (slope <= 0)] = np.nan

    frame["causal_deg_slope_raw"] = np.nan
    frame.loc[order, "causal_deg_slope_raw"] = slope.values
    frame["causal_deg_slope_raw"] = (
        frame["causal_deg_slope_raw"]
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=0.005, upper=2.5)
    )

    global_slope = frame["causal_deg_slope_raw"].median()
    if not np.isfinite(global_slope):
        global_slope = 0.06

    frame["deg_slope_smooth"] = frame["causal_deg_slope_raw"]
    for cols in [["Race", "Compound"], "Compound", "Race"]:
        frame["deg_slope_smooth"] = frame["deg_slope_smooth"].fillna(
            map_group_stat(frame, cols, "causal_deg_slope_raw")
        )
    frame["deg_slope_smooth"] = (
        frame["deg_slope_smooth"].fillna(global_slope).clip(lower=0.005, upper=2.5)
    )

    pit_excess = (frame["LapTime_winsor"] - base_pace).where(
        frame["PitStop"].fillna(0).eq(1)
    )
    positive_pit = pit_excess[(pit_excess > 0) & pit_excess.notna()]
    if len(positive_pit):
        _, p_hi = positive_pit.quantile([0.05, 0.95])
        pit_excess = pit_excess.clip(upper=max(float(p_hi), 10.0))
        global_pit_loss = float(positive_pit.clip(lower=8, upper=80).median())
    else:
        global_pit_loss = 22.0

    pit_order = frame.sort_values(["RaceYear", "LapNumber", "_row_order"]).index
    pit_tmp = frame.loc[pit_order, ["RaceYear"]].copy()
    pit_valid = frame.loc[pit_order, "PitStop"].fillna(0).eq(1) & pit_excess.loc[
        pit_order
    ].gt(0)
    pit_tmp["_cnt"] = pit_valid.astype(float).values
    pit_tmp["_sum"] = pit_excess.loc[pit_order].where(pit_valid, 0.0).values

    pit_gb = pit_tmp.groupby("RaceYear", sort=False)
    pit_cnt = pit_gb["_cnt"].cumsum()
    pit_sum = pit_gb["_sum"].cumsum()

    frame["pit_loss_causal"] = np.nan
    frame.loc[pit_order, "pit_loss_causal"] = (
        pit_sum / pit_cnt.replace(0, np.nan)
    ).values

    frame["pit_loss_est"] = frame["pit_loss_causal"]
    pit_source = frame.assign(_pit_excess=pit_excess.where(pit_excess > 0))
    for cols in ["RaceYear", "Race", "Compound"]:
        frame["pit_loss_est"] = frame["pit_loss_est"].fillna(
            map_group_stat(pit_source, cols, "_pit_excess")
        )
    frame["pit_loss_est"] = (
        frame["pit_loss_est"].fillna(global_pit_loss).clip(lower=8, upper=80)
    )

    frame["optimal_stint_len"] = np.sqrt(
        2.0 * frame["pit_loss_est"] / frame["deg_slope_smooth"]
    ).clip(lower=3, upper=90)
    frame["distance_to_optimal_stint_length"] = (
        frame["TyreLife"] - frame["optimal_stint_len"]
    )
    frame["abs_distance_to_optimal_stint_length"] = frame[
        "distance_to_optimal_stint_length"
    ].abs()
    frame["rel_distance_to_optimal_stint_length"] = frame[
        "distance_to_optimal_stint_length"
    ] / (frame["optimal_stint_len"] + 1e-6)
    frame["in_optimal_window_2laps"] = (
        frame["abs_distance_to_optimal_stint_length"].le(2).astype(np.int8)
    )
    frame["in_optimal_window_4laps"] = (
        frame["abs_distance_to_optimal_stint_length"].le(4).astype(np.int8)
    )
    frame["past_optimal_window"] = (
        frame["distance_to_optimal_stint_length"].gt(0).astype(np.int8)
    )
    frame["stop_window_pressure"] = 1.0 / (
        1.0
        + np.exp(
            -frame["distance_to_optimal_stint_length"].clip(lower=-30, upper=30) / 2.5
        )
    )
    frame["pit_loss_to_degradation_load"] = frame["pit_loss_est"] / (
        frame["deg_slope_smooth"] * frame["TyreLife"].clip(lower=1) + 1e-6
    )

    return frame


df = add_hypothesis_000473_features(df)

cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "RaceYear",
    "RaceYearCompound",
    "RaceYearCompoundStint",
]
cat_cols = [c for c in cat_cols if c in df.columns]
for c in cat_cols:
    df[c] = df[c].astype("category")

drop_cols = {TARGET, ID_COL, "_is_train", "_row_order"}
feature_cols = [c for c in df.columns if c not in drop_cols]

X_all = df[feature_cols].copy()
num_cols = X_all.select_dtypes(include=[np.number]).columns
X_all[num_cols] = X_all[num_cols].replace([np.inf, -np.inf], np.nan)

train_mask = df["_is_train"].eq(1).values
X = X_all.loc[train_mask].reset_index(drop=True)
X_test = X_all.loc[~train_mask].reset_index(drop=True)
y = df.loc[train_mask, TARGET].astype(int).reset_index(drop=True)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 3.0,
    "verbosity": -1,
    "seed": 2026,
    "num_threads": max(1, os.cpu_count() or 1),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
oof = np.zeros(len(X), dtype=float)
test_pred = np.zeros(len(X_test), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
    dtrain = lgb.Dataset(
        X.iloc[tr_idx],
        label=y.iloc[tr_idx],
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X.iloc[va_idx],
        label=y.iloc[va_idx],
        categorical_feature=cat_cols,
        reference=dtrain,
        free_raw_data=False,
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2500,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=120, verbose=False),
            lgb.log_evaluation(period=250),
        ],
    )

    val_pred = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
    oof[va_idx] = val_pred
    fold_auc = roc_auc_score(y.iloc[va_idx], val_pred)
    fold_scores.append(float(fold_auc))
    test_pred += (
        model.predict(X_test, num_iteration=model.best_iteration) / skf.n_splits
    )
    print(f"fold {fold} roc_auc: {fold_auc:.6f}")

oof_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {oof_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y.values,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

target_col = [c for c in sample.columns if c != ID_COL][0]
pred_frame = pd.DataFrame(
    {ID_COL: test[ID_COL].values, target_col: np.clip(test_pred, 0, 1)}
)

if len(sample) == len(pred_frame) and np.array_equal(
    sample[ID_COL].values, pred_frame[ID_COL].values
):
    submission = sample.copy()
    submission[target_col] = pred_frame[target_col].values
else:
    submission = sample[[ID_COL]].merge(pred_frame, on=ID_COL, how="left")
    submission[target_col] = submission[target_col].fillna(float(np.mean(test_pred)))

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "oof_roc_auc": float(oof_auc),
            "fold_roc_auc": fold_scores,
            "research_hypotheses_llm_claimed_used": ["000473"],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
            "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
            "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
        },
        indent=2,
    )
)
