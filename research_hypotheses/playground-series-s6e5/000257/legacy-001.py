import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan
all_df = pd.concat([train, test], axis=0, ignore_index=True)

all_df["race_year"] = all_df["Race"].astype(str) + "_" + all_df["Year"].astype(str)
all_df = all_df.sort_values(
    ["Year", "Race", "Driver", "LapNumber", ID_COL]
).reset_index(drop=True)

grp = all_df.groupby(["Year", "Race", "Driver"], sort=False)
all_df["prev_pitstop"] = grp["PitStop"].shift(1).fillna(0)
all_df["laps_since_pit"] = grp["PitStop"].cumsum()
all_df["stint_lap_frac"] = all_df["TyreLife"] / (
    all_df.groupby(["Year", "Race"])["LapNumber"].transform("max") + 1e-6
)
all_df["tyre_x_progress"] = all_df["TyreLife"] * all_df["RaceProgress"]
all_df["degradation_x_tyre"] = all_df["Cumulative_Degradation"] * all_df["TyreLife"]
all_df["lap_delta_x_tyre"] = all_df["LapTime_Delta"] * all_df["TyreLife"]
all_df["pitstop_recent_signal"] = (
    (all_df["PitStop"] == 1) | (all_df["prev_pitstop"] == 1)
).astype(int)

# Hypothesis 000257: infer undercut/overcut race regimes from observable race characteristics.
obs = all_df.copy()
early = obs[(obs["TyreLife"].between(2, 12)) & (obs["is_train"].notna())].copy()
early["x"] = early["TyreLife"]
early["xy"] = early["TyreLife"] * early["LapTime_Delta"]
early["x2"] = early["TyreLife"] ** 2
s = early.groupby("race_year").agg(
    n=("LapTime_Delta", "size"),
    sx=("x", "sum"),
    sy=("LapTime_Delta", "sum"),
    sxy=("xy", "sum"),
    sx2=("x2", "sum"),
)
s["early_delta_slope"] = (s["n"] * s["sxy"] - s["sx"] * s["sy"]) / (
    s["n"] * s["sx2"] - s["sx"] ** 2 + 1e-6
)

race_stats = obs.groupby("race_year").agg(
    deg_spread=("Cumulative_Degradation", "std"),
    fresh_outlap_var=("LapTime_Delta", lambda x: np.nan),
)
fresh = obs[(obs["TyreLife"].between(1, 3)) | (obs["prev_pitstop"] == 1)]
fresh_var = fresh.groupby("race_year")["LapTime_Delta"].var().rename("fresh_outlap_var")
race_stats = race_stats.drop(columns=["fresh_outlap_var"]).join(fresh_var, how="left")

warm = obs[obs["TyreLife"].between(1, 4)].copy()
warm_comp = warm.groupby(["race_year", "Compound"])["LapTime_Delta"].mean().unstack()
warm_comp["warmup_range"] = warm_comp.max(axis=1) - warm_comp.min(axis=1)
warm_comp["warmup_mean"] = warm_comp.mean(axis=1)

regime = race_stats.join(s[["early_delta_slope"]], how="left").join(
    warm_comp[["warmup_range", "warmup_mean"]], how="left"
)
for c in regime.columns:
    regime[c] = regime[c].fillna(regime[c].median())


def zscore(v):
    return (v - v.mean()) / (v.std(ddof=0) + 1e-6)


# High degradation and worsening early stints favor undercut; difficult warm-up and volatile out-laps favor overcut.
regime["undercut_score"] = zscore(regime["deg_spread"]) + zscore(
    regime["early_delta_slope"]
)
regime["overcut_score"] = (
    zscore(regime["warmup_range"])
    + zscore(regime["fresh_outlap_var"])
    - 0.5 * zscore(regime["deg_spread"])
)
regime["regime_prob_undercut"] = 1.0 / (
    1.0 + np.exp(-(regime["undercut_score"] - regime["overcut_score"]))
)
regime["regime_bucket"] = pd.qcut(
    regime["regime_prob_undercut"].rank(method="first"),
    q=3,
    labels=False,
    duplicates="drop",
).astype(int)

all_df = all_df.merge(
    regime[
        [
            "undercut_score",
            "overcut_score",
            "regime_prob_undercut",
            "regime_bucket",
            "early_delta_slope",
            "deg_spread",
            "warmup_range",
            "fresh_outlap_var",
        ]
    ],
    left_on="race_year",
    right_index=True,
    how="left",
)

urgency_cols = [
    "TyreLife",
    "RaceProgress",
    "Cumulative_Degradation",
    "LapTime_Delta",
    "Position_Change",
]
for c in urgency_cols:
    all_df[f"{c}_x_undercut"] = all_df[c] * all_df["regime_prob_undercut"]
    all_df[f"{c}_x_overcut"] = all_df[c] * (1.0 - all_df["regime_prob_undercut"])

all_df["driver_race_lap_rank"] = all_df.groupby(["race_year", "LapNumber"])[
    "LapTime (s)"
].rank(pct=True)
all_df["position_x_progress"] = all_df["Position"] * all_df["RaceProgress"]
all_df["stint_x_regime_bucket"] = all_df["Stint"] * (all_df["regime_bucket"] + 1)

for c in CAT_COLS:
    all_df[c] = all_df[c].astype("category")
all_df["regime_bucket"] = all_df["regime_bucket"].astype("category")
all_df["Year"] = all_df["Year"].astype("category")

drop_cols = [TARGET, ID_COL, "is_train", "race_year"]
features = [c for c in all_df.columns if c not in drop_cols]

trn = all_df[all_df["is_train"] == 1].copy()
tst = all_df[all_df["is_train"] == 0].copy()
X = trn[features]
y = trn[TARGET].astype(int).values
X_test = tst[features]
groups = trn["race_year"].values

cat_features = [
    c for c in ["Driver", "Race", "Compound", "Year", "regime_bucket"] if c in features
]

oof = np.zeros(len(trn), dtype=float)
test_pred = np.zeros(len(tst), dtype=float)

cv = GroupKFold(n_splits=5)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    model = LGBMClassifier(
        objective="binary",
        n_estimators=1600,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=80,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=20260524 + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[],
    )
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    score = roc_auc_score(y[va_idx], oof[va_idx])
    fold_scores.append(score)
    print(f"fold {fold} auc: {score:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"mean fold auc: {np.mean(fold_scores):.6f}")
print(f"overall oof roc_auc: {cv_auc:.6f}")

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(trn)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: sample[ID_COL].values,
        TARGET: test_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "overall_oof_roc_auc": float(cv_auc),
    "mean_fold_roc_auc": float(np.mean(fold_scores)),
    "research_hypotheses_llm_claimed_used": ["000257"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print(json.dumps(result, indent=2))
