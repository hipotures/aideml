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
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]
SLICK_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan
all_df = pd.concat([train, test], axis=0, ignore_index=True)

all_df["race_driver_key"] = (
    all_df["Year"].astype(str)
    + "_"
    + all_df["Race"].astype(str)
    + "_"
    + all_df["Driver"].astype(str)
)
all_df = all_df.sort_values(["race_driver_key", "LapNumber", ID_COL]).reset_index(
    drop=True
)

all_df["LapsRemaining"] = (
    all_df["LapNumber"] / all_df["RaceProgress"].clip(0.01, None) - all_df["LapNumber"]
).clip(0, 120)
all_df["is_wet_tyre"] = all_df["Compound"].isin(WET_COMPOUNDS).astype(np.int8)
all_df["is_slick"] = all_df["Compound"].isin(SLICK_COMPOUNDS).astype(np.int8)

g = all_df.groupby("race_driver_key", sort=False)

all_df["prior_pit_count"] = g["PitStop"].cumsum() - all_df["PitStop"]
all_df["wet_exempt"] = (g["is_wet_tyre"].cumsum() > 0).astype(np.int8)

for comp in SLICK_COMPOUNDS:
    used = (all_df["Compound"] == comp).astype(np.int8)
    all_df[f"used_{comp.lower()}_ever"] = (
        used.groupby(all_df["race_driver_key"]).cumsum() > 0
    ).astype(np.int8)

slick_used_cols = [f"used_{c.lower()}_ever" for c in SLICK_COMPOUNDS]
all_df["slick_compounds_used_count"] = (
    all_df[slick_used_cols].sum(axis=1).astype(np.int8)
)
all_df["dry_noncompliant_one_slick_used"] = (
    (all_df["wet_exempt"] == 0) & (all_df["slick_compounds_used_count"] == 1)
).astype(np.int8)
all_df["dry_compliant"] = (
    (all_df["wet_exempt"] == 1) | (all_df["slick_compounds_used_count"] >= 2)
).astype(np.int8)

all_df["current_tyre_can_finish_proxy"] = (
    (all_df["TyreLife"] <= all_df["LapsRemaining"] + 3)
    & (all_df["LapsRemaining"] <= 18)
).astype(np.int8)
all_df["current_tyre_can_finish_but_owes_compound_change"] = (
    (all_df["wet_exempt"] == 0)
    & (all_df["slick_compounds_used_count"] == 1)
    & (all_df["is_slick"] == 1)
    & (all_df["current_tyre_can_finish_proxy"] == 1)
).astype(np.int8)

state_cols = [
    "wet_exempt",
    "dry_noncompliant_one_slick_used",
    "dry_compliant",
    "current_tyre_can_finish_but_owes_compound_change",
]
base_interact_cols = ["LapsRemaining", "TyreLife", "Stint", "prior_pit_count"]
for s in state_cols:
    for c in base_interact_cols:
        all_df[f"{s}_x_{c}"] = all_df[s] * all_df[c]

all_df["tyre_life_frac_remaining"] = all_df["TyreLife"] / (all_df["LapsRemaining"] + 1)
all_df["late_race_owes_change"] = all_df["dry_noncompliant_one_slick_used"] * (
    all_df["LapsRemaining"] <= 8
).astype(np.int8)
all_df["pit_now_after_prior_pits"] = all_df["PitStop"] * all_df["prior_pit_count"]

all_df = all_df.sort_values(ID_COL).reset_index(drop=True)
train_fe = all_df[all_df["is_train"] == 1].copy()
test_fe = all_df[all_df["is_train"] == 0].copy()

drop_cols = [TARGET, ID_COL, "is_train", "race_driver_key"]
features = [c for c in train_fe.columns if c not in drop_cols]

for c in CAT_COLS:
    train_fe[c] = train_fe[c].astype("category")
    test_fe[c] = pd.Categorical(test_fe[c], categories=train_fe[c].cat.categories)

X = train_fe[features]
y = train_fe[TARGET].astype(int).values
X_test = test_fe[features]
groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)

oof = np.zeros(len(train_fe), dtype=float)
test_pred = np.zeros(len(test_fe), dtype=float)

cv = GroupKFold(n_splits=5)
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    model = LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=420 + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=CAT_COLS,
        callbacks=[],
    )
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    print(f"fold {fold} auc: {roc_auc_score(y[va_idx], oof[va_idx]):.6f}")

auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
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

with open(os.path.join(WORKING_DIR, "result.json"), "w") as f:
    json.dump(
        {
            "metric": "roc_auc",
            "cv_score": float(auc),
            "research_hypotheses_llm_claimed_used": ["000420"],
        },
        f,
        indent=2,
    )
