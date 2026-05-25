import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

TARGET = "PitNextLap"
IDCOL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
FIXED_LIFE = {
    "SOFT": 18.0,
    "MEDIUM": 28.0,
    "HARD": 38.0,
    "INTERMEDIATE": 22.0,
    "WET": 26.0,
}

train = pd.read_csv("./input/train.csv.gz")
test = pd.read_csv("./input/test.csv.gz")
sample = pd.read_csv("./input/sample_submission.csv.gz")


def learn_life_priors(df):
    # Use observed stint endpoints only, train-fold data only. PitNextLap is never used.
    endpoints = df[df["PitStop"].eq(1)].copy()
    if len(endpoints) < 100:
        endpoints = df.copy()

    qs = (
        endpoints.groupby("Compound")["TyreLife"].quantile([0.60, 0.75, 0.90]).unstack()
    )
    qs.columns = ["life_q60", "life_q75", "life_q90"]

    global_q = endpoints["TyreLife"].quantile([0.60, 0.75, 0.90]).values
    counts = endpoints.groupby("Compound")["TyreLife"].size()
    alpha = 50.0
    for c in qs.index:
        w = counts.loc[c] / (counts.loc[c] + alpha)
        qs.loc[c] = w * qs.loc[c].values + (1.0 - w) * global_q

    return qs, dict(zip(["life_q60", "life_q75", "life_q90"], global_q))


def add_features(df, priors, global_prior):
    out = df.copy()
    out["fixed_expected_life"] = (
        out["Compound"].map(FIXED_LIFE).fillna(np.mean(list(FIXED_LIFE.values())))
    )

    for col in ["life_q60", "life_q75", "life_q90"]:
        out[col] = out["Compound"].map(priors[col]).fillna(global_prior[col])

    est_total_laps = (out["LapNumber"] / out["RaceProgress"].clip(0.01, 1.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    out["est_total_laps"] = est_total_laps.fillna(out["LapNumber"])
    out["laps_remaining"] = (out["est_total_laps"] - out["LapNumber"]).clip(lower=0)

    for col in ["life_q60", "life_q75", "life_q90"]:
        out[f"tyrelife_minus_{col}"] = out["TyreLife"] - out[col]
        out[f"laps_to_{col}"] = out[col] - out["TyreLife"]
        out[f"finish_margin_{col}"] = out[col] - (
            out["TyreLife"] + out["laps_remaining"]
        )
        out[f"nextlap_margin_{col}"] = out[col] - (out["TyreLife"] + 1.0)

    out["tyrelife_minus_fixed"] = out["TyreLife"] - out["fixed_expected_life"]
    out["finish_margin_fixed"] = out["fixed_expected_life"] - (
        out["TyreLife"] + out["laps_remaining"]
    )
    out["nextlap_margin_fixed"] = out["fixed_expected_life"] - (out["TyreLife"] + 1.0)
    out["degradation_per_life"] = out["Cumulative_Degradation"] / out["TyreLife"].clip(
        lower=1
    )
    out["lap_progress_x_tyre"] = out["RaceProgress"] * out["TyreLife"]
    out["stint_progress"] = out["TyreLife"] / out["est_total_laps"].clip(lower=1)
    return out


y = train[TARGET].astype(int)
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
features = [c for c in train.columns if c not in [TARGET, IDCOL]]

oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
fold_scores = []

cv = GroupKFold(n_splits=5)
for fold, (tr_idx, va_idx) in enumerate(cv.split(train, y, groups), 1):
    tr_raw = train.iloc[tr_idx].reset_index(drop=True)
    va_raw = train.iloc[va_idx].reset_index(drop=True)

    priors, global_prior = learn_life_priors(tr_raw)
    tr_x = add_features(tr_raw[features], priors, global_prior)
    va_x = add_features(va_raw[features], priors, global_prior)
    te_x = add_features(test[features], priors, global_prior)

    for c in CAT_COLS:
        tr_x[c] = tr_x[c].astype("category")
        va_x[c] = pd.Categorical(va_x[c], categories=tr_x[c].cat.categories)
        te_x[c] = pd.Categorical(te_x[c], categories=tr_x[c].cat.categories)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        min_child_samples=80,
        random_state=2026 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        tr_x,
        y.iloc[tr_idx],
        eval_set=[(va_x, y.iloc[va_idx])],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
    )

    oof[va_idx] = model.predict_proba(va_x)[:, 1]
    test_pred += model.predict_proba(te_x)[:, 1] / cv.n_splits
    auc = roc_auc_score(y.iloc[va_idx], oof[va_idx])
    fold_scores.append(auc)
    print(f"fold {fold} auc: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold grouped CV ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y.values,
        "prediction": oof,
    }
).to_csv("./working/oof_predictions.csv.gz", index=False, compression="gzip")

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv("./working/submission.csv", index=False)
submission.to_csv("./working/test_predictions.csv.gz", index=False, compression="gzip")

print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000004"],
            "metric": "roc_auc",
            "cv_auc": float(cv_auc),
            "fold_auc": [float(x) for x in fold_scores],
            "submission_path": "./working/submission.csv",
        }
    )
)
