import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["_is_train"] = 1
test["_is_train"] = 0
train["_row"] = np.arange(len(train))
test["_row"] = np.arange(len(test))

if TARGET not in test.columns:
    test[TARGET] = np.nan

df = pd.concat([train, test], axis=0, ignore_index=True)
df["RaceEvent"] = df["Year"].astype(str) + "__" + df["Race"].astype(str)

event_cols = ["Year", "Race"]
lap_cols = event_cols + ["LapNumber"]
comp_lap_cols = lap_cols + ["Compound"]

pit = df["PitStop"].astype(float)
tyre = df["TyreLife"].astype(float)

# Hypothesis 000929: field pit-wave same-lap aggregates, excluding the row's own PitStop where possible.
lap_count = df.groupby(lap_cols, sort=False)["PitStop"].transform("count").astype(float)
lap_pit_sum = df.groupby(lap_cols, sort=False)["PitStop"].transform("sum").astype(float)
other_count = (lap_count - 1).clip(lower=0)

df["field_size_lap"] = lap_count
df["field_pit_count_excl_self"] = (lap_pit_sum - pit).clip(lower=0)
df["field_pit_rate_excl_self"] = np.where(
    other_count > 0, df["field_pit_count_excl_self"] / other_count, 0.0
)

pitted_tyre = tyre.where(df["PitStop"].eq(1))
pitted_count = lap_pit_sum
df["pitted_tyrelife_median_lap"] = pitted_tyre.groupby(
    [df[c] for c in lap_cols], sort=False
).transform("median")
df.loc[(df["PitStop"].eq(1)) & (pitted_count <= 1), "pitted_tyrelife_median_lap"] = (
    np.nan
)

nonpit = df["PitStop"].eq(0)
nonpit_count = (
    nonpit.astype(float).groupby([df[c] for c in lap_cols], sort=False).transform("sum")
)
nonpit_tyre_sum = (
    tyre.where(nonpit, 0.0)
    .groupby([df[c] for c in lap_cols], sort=False)
    .transform("sum")
)
own_nonpit = nonpit.astype(float)
den = nonpit_count - own_nonpit
num = nonpit_tyre_sum - tyre * own_nonpit
df["nonpitter_tyrelife_mean_excl_self"] = np.where(den > 0, num / den, np.nan)

nonpit_small = df.loc[nonpit, lap_cols + ["TyreLife"]].copy()
min1 = nonpit_small.groupby(lap_cols, sort=False)["TyreLife"].transform("min")
nonpit_small["_min1"] = min1
min_freq = (
    nonpit_small.loc[nonpit_small["TyreLife"].eq(nonpit_small["_min1"])]
    .groupby(lap_cols, sort=False)
    .size()
    .rename("_min_freq")
)
second_min = (
    nonpit_small.loc[nonpit_small["TyreLife"].gt(nonpit_small["_min1"])]
    .groupby(lap_cols, sort=False)["TyreLife"]
    .min()
    .rename("_second_min")
)
min_table = (
    nonpit_small.groupby(lap_cols, sort=False)["_min1"]
    .first()
    .to_frame()
    .join(min_freq)
    .join(second_min)
    .reset_index()
)
df = df.merge(min_table, on=lap_cols, how="left")
own_unique_min = nonpit & df["TyreLife"].eq(df["_min1"]) & df["_min_freq"].eq(1)
df["nonpitter_tyrelife_min_excl_self"] = np.where(
    own_unique_min, df["_second_min"], df["_min1"]
)
df.drop(columns=["_min1", "_min_freq", "_second_min"], inplace=True)

# Compound-specific same-lap pit rates.
comp_count = (
    df.groupby(comp_lap_cols, sort=False)["PitStop"].transform("count").astype(float)
)
comp_pit_sum = (
    df.groupby(comp_lap_cols, sort=False)["PitStop"].transform("sum").astype(float)
)
comp_other_count = (comp_count - 1).clip(lower=0)
df["compound_lap_size"] = comp_count
df["compound_pit_count_excl_self"] = (comp_pit_sum - pit).clip(lower=0)
df["compound_pit_rate_excl_self"] = np.where(
    comp_other_count > 0, df["compound_pit_count_excl_self"] / comp_other_count, 0.0
)

lap_comp = (
    df.groupby(comp_lap_cols, sort=False)["PitStop"].agg(["sum", "count"]).reset_index()
)
lap_comp["rate"] = lap_comp["sum"] / lap_comp["count"].clip(lower=1)
rate_wide = lap_comp.pivot_table(
    index=lap_cols, columns="Compound", values="rate", fill_value=0.0
)
rate_wide.columns = [f"field_pit_rate_compound_{str(c)}" for c in rate_wide.columns]
rate_wide = rate_wide.reset_index()
df = df.merge(rate_wide, on=lap_cols, how="left")

# Lagged field pit counts for previous 1-3 laps by race event, plus cumulative stops by race and compound.
lap_table = (
    df.groupby(lap_cols, sort=False)["PitStop"]
    .agg(field_pit_count_total="sum", field_driver_count_total="count")
    .reset_index()
    .sort_values(event_cols + ["LapNumber"])
)
lap_table["field_pit_rate_total"] = lap_table["field_pit_count_total"] / lap_table[
    "field_driver_count_total"
].clip(lower=1)
lap_table["race_cum_pitstops_prior"] = (
    lap_table.groupby(event_cols, sort=False)["field_pit_count_total"]
    .cumsum()
    .groupby([lap_table[c] for c in event_cols], sort=False)
    .shift(1)
    .fillna(0.0)
)
lap_table["race_cum_pitstops_incl_lap"] = lap_table.groupby(event_cols, sort=False)[
    "field_pit_count_total"
].cumsum()

for k in (1, 2, 3):
    lag = lap_table[lap_cols + ["field_pit_count_total"]].copy()
    lag["LapNumber"] = lag["LapNumber"] + k
    lag.rename(
        columns={"field_pit_count_total": f"field_pit_count_lag{k}"}, inplace=True
    )
    lap_table = lap_table.merge(lag, on=lap_cols, how="left")

df = df.merge(lap_table, on=lap_cols, how="left")
df["race_cum_pitstops_through_lap_excl_self"] = (
    df["race_cum_pitstops_prior"] + df["field_pit_count_excl_self"]
)

comp_table = (
    df.groupby(comp_lap_cols, sort=False)["PitStop"]
    .sum()
    .rename("compound_pit_count_total")
    .reset_index()
    .sort_values(event_cols + ["Compound", "LapNumber"])
)
comp_table["compound_cum_pitstops_prior"] = (
    comp_table.groupby(event_cols + ["Compound"], sort=False)[
        "compound_pit_count_total"
    ]
    .cumsum()
    .groupby([comp_table[c] for c in event_cols + ["Compound"]], sort=False)
    .shift(1)
    .fillna(0.0)
)
comp_table["compound_cum_pitstops_incl_lap"] = comp_table.groupby(
    event_cols + ["Compound"], sort=False
)["compound_pit_count_total"].cumsum()

df = df.merge(comp_table, on=comp_lap_cols, how="left")
df["compound_cum_pitstops_through_lap_excl_self"] = (
    df["compound_cum_pitstops_prior"] + df["compound_pit_count_excl_self"]
)

for c in [c for c in df.columns if c.startswith("field_pit_count_lag")]:
    df[c] = df[c].fillna(0.0)
for c in [
    "race_cum_pitstops_prior",
    "race_cum_pitstops_incl_lap",
    "compound_cum_pitstops_prior",
    "compound_cum_pitstops_incl_lap",
]:
    df[c] = df[c].fillna(0.0)

train_fe = df[df["_is_train"].eq(1)].sort_values("_row").reset_index(drop=True)
test_fe = df[df["_is_train"].eq(0)].sort_values("_row").reset_index(drop=True)

drop_cols = {TARGET, ID_COL, "_is_train", "_row"}
features = [c for c in train_fe.columns if c not in drop_cols and not c.startswith("_")]

cat_features = []
for c in features:
    if train_fe[c].dtype == "object" or test_fe[c].dtype == "object":
        cats = pd.Index(
            pd.concat([train_fe[c], test_fe[c]], axis=0).astype(str).unique()
        )
        train_fe[c] = pd.Categorical(train_fe[c].astype(str), categories=cats)
        test_fe[c] = pd.Categorical(test_fe[c].astype(str), categories=cats)
        cat_features.append(c)


def make_safe_names(cols):
    used = {}
    out = {}
    for c in cols:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_")
        if not base:
            base = "feature"
        name = base
        i = 1
        while name in used:
            i += 1
            name = f"{base}_{i}"
        used[name] = True
        out[c] = name
    return out


rename_map = make_safe_names(features)
X = train_fe[features].rename(columns=rename_map)
X_test = test_fe[features].rename(columns=rename_map)
cat_features_safe = [rename_map[c] for c in cat_features]

y = train_fe[TARGET].astype(int).values
groups = train_fe["RaceEvent"].astype(str).values

try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("lightgbm is required for this solution") from e

try:
    from sklearn.model_selection import StratifiedGroupKFold

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y, groups))
except Exception:
    from sklearn.model_selection import GroupKFold

    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(X, y, groups))

base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=2.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=max(1, os.cpu_count() or 1),
    verbosity=-1,
)

oof = np.zeros(len(train_fe), dtype=float)
fold_scores = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**base_params, n_estimators=2000)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features_safe,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))
    best_iters.append(int(model.best_iteration_ or model.n_estimators))
    print(f"Fold {fold} ROC AUC: {auc:.6f} best_iteration={best_iters[-1]}")

cv_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_estimators = int(np.clip(np.median(best_iters), 100, 2000))
final_model = lgb.LGBMClassifier(**base_params, n_estimators=final_estimators)
final_model.fit(X, y, categorical_feature=cat_features_safe)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

test_predictions = pd.DataFrame(
    {
        ID_COL: test_fe[ID_COL].values,
        TARGET: test_pred,
    }
)
test_predictions.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

submission = sample[[ID_COL]].merge(test_predictions, on=ID_COL, how="left")
submission[TARGET] = submission[TARGET].fillna(float(np.mean(oof)))
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

review = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
    "research_hypotheses_llm_claimed_used": ["000929"],
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
    "oof_path": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
}
print(json.dumps(review, sort_keys=True))
