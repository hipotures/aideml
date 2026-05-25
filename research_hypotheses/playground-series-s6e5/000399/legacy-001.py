import os
import re
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGKF = True
except Exception:
    from sklearn.model_selection import GroupKFold

    HAS_SGKF = False

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
RANDOM_STATE = 42
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["_row"] = np.arange(len(train))
y = train[TARGET].astype(int).values

train_feat = train.drop(columns=[TARGET]).copy()
test_feat = test.copy()
train_feat["_is_train"] = 1
test_feat["_is_train"] = 0
all_df = pd.concat([train_feat, test_feat], ignore_index=True, sort=False)

lt_lo, lt_hi = train["LapTime (s)"].quantile([0.005, 0.995])
delta_hi = train["LapTime_Delta"].abs().quantile(0.995)
deg_lo, deg_hi = train["Cumulative_Degradation"].quantile([0.002, 0.998])

all_df["_wear_good"] = (
    all_df["PitStop"].eq(0)
    & all_df["TyreLife"].gt(0)
    & all_df["LapTime (s)"].between(lt_lo, lt_hi)
    & all_df["LapTime_Delta"].abs().le(delta_hi)
    & all_df["Cumulative_Degradation"].between(deg_lo, deg_hi)
)


def robust_prior_stats(g):
    g = g[g["_wear_good"]]
    x = g["TyreLife"].astype(float).to_numpy()
    z = g["Cumulative_Degradation"].astype(float).to_numpy()

    if len(z) >= 20:
        lo, hi = np.quantile(z, [0.05, 0.95])
        keep = (z >= lo) & (z <= hi)
        x, z = x[keep], z[keep]

    n = len(z)
    slope = intercept = sigma = np.nan
    if n >= 5 and np.nanvar(x) > 1e-9:
        xm, zm = np.nanmean(x), np.nanmean(z)
        denom = np.nansum((x - xm) ** 2)
        slope = np.nansum((x - xm) * (z - zm)) / max(denom, 1e-9)
        intercept = zm - slope * xm
        resid = z - (intercept + slope * x)
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(resid)

    return pd.Series(
        {
            "wear_prior_slope": slope,
            "wear_prior_intercept": intercept,
            "wear_prior_sigma": sigma,
            "wear_prior_cap": np.nanquantile(z, 0.90) if n else np.nan,
            "wear_prior_life_p85": np.nanquantile(x, 0.85) if n else np.nan,
            "wear_prior_n": n,
        }
    )


prior_base = (
    all_df[all_df["_is_train"].eq(1)]
    .groupby(["Race", "Year", "Compound"], observed=True)
    .apply(robust_prior_stats)
    .reset_index()
)

stat_cols = [
    "wear_prior_slope",
    "wear_prior_intercept",
    "wear_prior_sigma",
    "wear_prior_cap",
    "wear_prior_life_p85",
    "wear_prior_n",
]
global_stats = prior_base[stat_cols].median(numeric_only=True)
fallbacks = {
    "wear_prior_slope": 0.0,
    "wear_prior_intercept": float(train["Cumulative_Degradation"].median()),
    "wear_prior_sigma": float(train["Cumulative_Degradation"].std()),
    "wear_prior_cap": float(train["Cumulative_Degradation"].quantile(0.90)),
    "wear_prior_life_p85": float(train["TyreLife"].quantile(0.85)),
    "wear_prior_n": 0.0,
}
for c, v in fallbacks.items():
    if c not in global_stats or not np.isfinite(global_stats[c]):
        global_stats[c] = v

prev = prior_base[["Race", "Year", "Compound"] + stat_cols].copy()
prev["Year"] = prev["Year"] + 1
all_df = all_df.merge(prev, on=["Race", "Year", "Compound"], how="left", sort=False)

race_comp = prior_base.groupby(["Race", "Compound"], as_index=False)[stat_cols].median()
race_comp = race_comp.rename(columns={c: f"{c}_race_comp" for c in stat_cols})
all_df = all_df.merge(race_comp, on=["Race", "Compound"], how="left", sort=False)

comp = prior_base.groupby(["Compound"], as_index=False)[stat_cols].median()
comp = comp.rename(columns={c: f"{c}_compound" for c in stat_cols})
all_df = all_df.merge(comp, on=["Compound"], how="left", sort=False)

for c in stat_cols:
    all_df[c] = (
        all_df[c]
        .fillna(all_df[f"{c}_race_comp"])
        .fillna(all_df[f"{c}_compound"])
        .fillna(global_stats[c])
    )

all_df = all_df.drop(
    columns=[
        c for c in all_df.columns if c.endswith("_race_comp") or c.endswith("_compound")
    ]
)

slope_vals = prior_base["wear_prior_slope"].replace([np.inf, -np.inf], np.nan).dropna()
if len(slope_vals):
    slope_lo, slope_hi = np.quantile(slope_vals, [0.01, 0.99])
else:
    slope_lo, slope_hi = -10.0, 50.0
if not np.isfinite(slope_lo) or not np.isfinite(slope_hi) or slope_lo >= slope_hi:
    slope_lo, slope_hi = -10.0, 50.0


def add_rolling_slope_features(df, keys, prefix, window, min_good):
    sort_cols = keys + ["LapNumber", "id"]
    order = df.sort_values(sort_cols, kind="mergesort").index

    good = df["_wear_good"].fillna(False).to_numpy()
    x = df["TyreLife"].astype(float).to_numpy()
    z = df["Cumulative_Degradation"].astype(float).to_numpy()

    tmp = pd.DataFrame(
        {
            "_n": good.astype(float),
            "_sx": np.where(good, x, 0.0),
            "_sz": np.where(good, z, 0.0),
            "_sxx": np.where(good, x * x, 0.0),
            "_sxz": np.where(good, x * z, 0.0),
            "_szz": np.where(good, z * z, 0.0),
        },
        index=df.index,
    )
    tmp = pd.concat([df[keys], tmp], axis=1).loc[order]

    sum_cols = ["_n", "_sx", "_sz", "_sxx", "_sxz", "_szz"]
    rolled = (
        tmp.groupby(keys, sort=False, observed=True)[sum_cols]
        .rolling(window=window, min_periods=1)
        .sum()
    )
    rolled.index = rolled.index.get_level_values(-1)
    rolled = rolled.reindex(df.index)

    n = rolled["_n"].to_numpy()
    sx = rolled["_sx"].to_numpy()
    sz = rolled["_sz"].to_numpy()
    sxx = rolled["_sxx"].to_numpy()
    sxz = rolled["_sxz"].to_numpy()
    szz = rolled["_szz"].to_numpy()

    denom = n * sxx - sx * sx
    slope = np.full(len(df), np.nan)
    intercept = np.full(len(df), np.nan)
    mask = (n >= min_good) & (np.abs(denom) > 1e-9)
    slope[mask] = (n[mask] * sxz[mask] - sx[mask] * sz[mask]) / denom[mask]
    intercept[mask] = (sz[mask] - slope[mask] * sx[mask]) / np.maximum(n[mask], 1)

    sxx_centered = sxx - sx * sx / np.maximum(n, 1)
    var_z = (szz - sz * sz / np.maximum(n, 1)) / np.maximum(n - 1, 1)
    unc = np.sqrt(np.maximum(var_z, 0) / np.maximum(sxx_centered, 1e-3))

    return pd.DataFrame(
        {
            f"{prefix}_obs_n": n,
            f"{prefix}_obs_slope": slope,
            f"{prefix}_obs_intercept": intercept,
            f"{prefix}_obs_unc": unc,
        },
        index=df.index,
    )


all_df = pd.concat(
    [
        all_df,
        add_rolling_slope_features(
            all_df, ["Race", "Year", "Compound"], "circuit", window=180, min_good=8
        ),
        add_rolling_slope_features(
            all_df,
            ["Race", "Year", "Driver", "Stint", "Compound"],
            "stint",
            window=35,
            min_good=4,
        ),
    ],
    axis=1,
)

prior_strength = 30.0
c_n = all_df["circuit_obs_n"].astype(float).clip(lower=0)
c_slope = all_df["circuit_obs_slope"].clip(slope_lo, slope_hi)
p_slope = all_df["wear_prior_slope"].clip(slope_lo, slope_hi)

all_df["wear_posterior_slope"] = np.where(
    c_slope.notna(),
    (c_n * c_slope.fillna(0) + prior_strength * p_slope) / (c_n + prior_strength),
    p_slope,
)
all_df["wear_posterior_slope"] = all_df["wear_posterior_slope"].clip(slope_lo, slope_hi)

disagreement = (c_slope - p_slope).abs().fillna(0)
all_df["wear_posterior_uncertainty"] = (
    disagreement
    + all_df["wear_prior_sigma"].abs().fillna(global_stats["wear_prior_sigma"])
) / np.sqrt(c_n + prior_strength + 1.0)

stint_strength = 8.0
s_n = all_df["stint_obs_n"].astype(float).clip(lower=0)
s_slope = all_df["stint_obs_slope"].clip(slope_lo, slope_hi)
all_df["stint_wear_posterior_slope"] = np.where(
    s_slope.notna(),
    (s_n * s_slope.fillna(0) + stint_strength * all_df["wear_posterior_slope"])
    / (s_n + stint_strength),
    all_df["wear_posterior_slope"],
).clip(slope_lo, slope_hi)

race_progress = all_df["RaceProgress"].clip(lower=0.01)
all_df["estimated_total_laps"] = (all_df["LapNumber"] / race_progress).clip(1, 120)
all_df["laps_left_est"] = (all_df["estimated_total_laps"] - all_df["LapNumber"]).clip(
    0, 120
)
all_df["race_lap_frac"] = all_df["LapNumber"] / all_df["estimated_total_laps"].clip(
    lower=1
)
all_df["tyre_frac_of_race"] = all_df["TyreLife"] / all_df["estimated_total_laps"].clip(
    lower=1
)

effective_slope = (
    all_df["stint_wear_posterior_slope"]
    .fillna(all_df["wear_posterior_slope"])
    .clip(lower=0.02)
)
all_df["posterior_wear_load"] = all_df["TyreLife"] * effective_slope
all_df["wear_degradation_to_finish"] = (
    all_df["Cumulative_Degradation"] + effective_slope * all_df["laps_left_est"]
)
all_df["wear_finish_margin"] = (
    all_df["wear_prior_cap"] - all_df["wear_degradation_to_finish"]
)
all_df["wear_laps_to_prior_cap"] = (
    all_df["wear_prior_cap"] - all_df["Cumulative_Degradation"]
) / effective_slope
denom_laps = all_df["wear_laps_to_prior_cap"].where(
    all_df["wear_laps_to_prior_cap"] > 0.1, 0.1
)
all_df["wear_need_stop_ratio"] = all_df["laps_left_est"] / denom_laps
all_df["wear_prior_gap"] = (
    all_df["stint_wear_posterior_slope"] - all_df["wear_prior_slope"]
)
all_df["tyrelife_minus_prior_life85"] = (
    all_df["TyreLife"] - all_df["wear_prior_life_p85"]
)
all_df["degradation_per_tyre_lap"] = all_df["Cumulative_Degradation"] / all_df[
    "TyreLife"
].clip(lower=1)

pit_order = all_df.sort_values(
    ["Race", "Year", "Driver", "LapNumber", "id"], kind="mergesort"
).index
all_df["driver_pitstops_so_far"] = 0.0
all_df.loc[pit_order, "driver_pitstops_so_far"] = (
    all_df.loc[pit_order]
    .groupby(["Race", "Year", "Driver"], observed=True)["PitStop"]
    .cumsum()
    .to_numpy()
)

train_features = all_df.iloc[: len(train)].copy()
test_features = all_df.iloc[len(train) :].copy()

drop_cols = {"id", "_row", "_is_train", "_wear_good"}
feature_cols = [c for c in train_features.columns if c not in drop_cols]

cat_cols = [c for c in feature_cols if train_features[c].dtype == "object"]
for c in cat_cols:
    both = (
        pd.concat([train_features[c], test_features[c]], ignore_index=True)
        .astype("string")
        .fillna("__NA__")
    )
    cats = pd.Index(both.unique())
    train_features[c] = pd.Categorical(
        train_features[c].astype("string").fillna("__NA__"), categories=cats
    )
    test_features[c] = pd.Categorical(
        test_features[c].astype("string").fillna("__NA__"), categories=cats
    )


def clean_feature_names(cols):
    out, seen = {}, {}
    for c in cols:
        name = re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_")
        name = name if name else "feature"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out[c] = name
    return out


name_map = clean_feature_names(feature_cols)
X = (
    train_features[feature_cols]
    .rename(columns=name_map)
    .replace([np.inf, -np.inf], np.nan)
)
X_test = (
    test_features[feature_cols]
    .rename(columns=name_map)
    .replace([np.inf, -np.inf], np.nan)
)
cat_features = [name_map[c] for c in cat_cols]

groups = train["Race"].astype(str) + "_" + train["Year"].astype(str)
pos = max(int(y.sum()), 1)
neg = len(y) - pos
scale_pos_weight = max(1.0, min(30.0, neg / pos))

params = dict(
    objective="binary",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=4.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

if HAS_SGKF:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = cv.split(X, y, groups)
else:
    cv = GroupKFold(n_splits=5)
    splits = cv.split(X, y, groups)

oof = np.zeros(len(X), dtype=float)
fold_scores, best_iters = [], []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[early_stopping(120, verbose=False), log_evaluation(period=0)],
    )
    best_iter = model.best_iteration_ or params["n_estimators"]
    pred = model.predict_proba(X.iloc[va_idx], num_iteration=best_iter)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))
    best_iters.append(int(best_iter))
    print(f"Fold {fold} ROC AUC: {auc:.6f} best_iter={best_iter}")

oof_auc = roc_auc_score(y, oof)
mean_auc = float(np.mean(fold_scores))
print(f"OOF ROC AUC: {oof_auc:.6f}")
print(f"Mean fold ROC AUC: {mean_auc:.6f}")

final_iter = int(np.median(best_iters)) if best_iters else 800
final_params = params.copy()
final_params["n_estimators"] = max(50, final_iter)

final_model = LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_features)
test_pred = np.clip(final_model.predict_proba(X_test)[:, 1], 0, 1)

oof_df = pd.DataFrame(
    {
        "row": train["_row"].astype(int).values,
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = pd.DataFrame(
    {
        "id": test["id"].values,
        TARGET: test_pred,
    }
)
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

if np.array_equal(sample["id"].values, test["id"].values):
    submission = pd.DataFrame({"id": sample["id"].values, TARGET: test_pred})
else:
    submission = sample[["id"]].merge(test_pred_df, on="id", how="left")
    if submission[TARGET].isna().any():
        raise ValueError("Sample submission ids do not align with test ids.")

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

result = {
    "metric": "roc_auc",
    "oof_roc_auc": float(oof_auc),
    "mean_fold_roc_auc": mean_auc,
    "fold_roc_auc": fold_scores,
    "research_hypotheses_llm_claimed_used": ["000399"],
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
