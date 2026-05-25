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

TRAIN_PATH = "./input/train.csv.gz"
TEST_PATH = "./input/test.csv.gz"
SAMPLE_PATH = "./input/sample_submission.csv.gz"
TARGET = "PitNextLap"
ID = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
SORT_COLS = ["Year", "Race", "Driver", "LapNumber", "id"]

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

for df in (train, test):
    df["_is_test"] = ID
    df["race_driver"] = (
        df["Year"].astype(str)
        + "_"
        + df["Race"].astype(str)
        + "_"
        + df["Driver"].astype(str)
    )
    df["stint_frac_life"] = df["TyreLife"] / df.groupby(
        ["Year", "Race", "Driver", "Stint"]
    )["TyreLife"].transform("max").clip(lower=1)
    df["lap_frac_race"] = df["LapNumber"] / df.groupby(["Year", "Race"])[
        "LapNumber"
    ].transform("max").clip(lower=1)
    df["degradation_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(
        lower=1
    )
    df["is_wet_compound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    df["is_dry_compound"] = df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(int)

all_data = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
for c in CAT_COLS:
    codes, uniques = pd.factorize(all_data[c].astype(str), sort=True)
    all_data[c + "_enc"] = codes.astype("int32")
enc_cols = [c + "_enc" for c in CAT_COLS]

train_enc = all_data.iloc[: len(train)].copy()
test_enc = all_data.iloc[len(train) :].copy()
train_enc[TARGET] = train[TARGET].astype(int).values

base_features = [
    "Year",
    "LapNumber",
    "LapTime (s)",
    "LapTime_Delta",
    "PitStop",
    "Position",
    "Position_Change",
    "RaceProgress",
    "Stint",
    "TyreLife",
    "Cumulative_Degradation",
    "stint_frac_life",
    "lap_frac_race",
    "degradation_per_lap",
    "is_wet_compound",
    "is_dry_compound",
] + enc_cols

for c in base_features:
    train_enc[c] = train_enc[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    test_enc[c] = test_enc[c].replace([np.inf, -np.inf], np.nan).fillna(0)


def add_horizon_labels(df, max_h=3):
    out = df.sort_values(SORT_COLS).copy()
    g = out.groupby(["Year", "Race", "Driver"], sort=False)["PitStop"]
    for h in range(1, max_h + 1):
        out[f"pit_in_{h}"] = g.shift(-h).fillna(0).astype(int)
    return out.sort_index()


train_h = add_horizon_labels(train_enc, 3)


def make_hazard_frame(df, indices, training=True):
    pieces = []
    for h in (1, 2, 3):
        part = df.loc[indices, base_features].copy()
        part["horizon"] = h
        if training:
            part[TARGET] = df.loc[indices, f"pit_in_{h}"].values
        pieces.append(part)
    return pd.concat(pieces, axis=0, ignore_index=True)


groups = train_enc["race_driver"].astype(str).values
y = train_enc[TARGET].astype(int).values
folds = GroupKFold(n_splits=5)

oof_main = np.zeros(len(train_enc))
oof_hazard = np.zeros(len(train_enc))
test_main = np.zeros(len(test_enc))
test_hazard = np.zeros(len(test_enc))
fold_scores = []

main_params = dict(
    n_estimators=900,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=80,
    reg_alpha=0.2,
    reg_lambda=1.5,
    objective="binary",
    random_state=267,
    n_jobs=-1,
    verbosity=-1,
)

hazard_params = dict(
    n_estimators=700,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=120,
    reg_alpha=0.2,
    reg_lambda=2.0,
    objective="binary",
    random_state=1267,
    n_jobs=-1,
    verbosity=-1,
)

X_test_main = test_enc[base_features]
X_test_h1 = test_enc[base_features].copy()
X_test_h1["horizon"] = 1

for fold, (tr_idx, va_idx) in enumerate(folds.split(train_enc, y, groups), 1):
    X_tr, X_va = (
        train_enc.iloc[tr_idx][base_features],
        train_enc.iloc[va_idx][base_features],
    )
    y_tr, y_va = y[tr_idx], y[va_idx]

    main = LGBMClassifier(**main_params)
    main.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[],
    )
    pred_main = main.predict_proba(X_va)[:, 1]
    oof_main[va_idx] = pred_main
    test_main += main.predict_proba(X_test_main)[:, 1] / folds.n_splits

    haz_tr = make_hazard_frame(train_h, tr_idx, training=True)
    haz_va_h1 = train_h.iloc[va_idx][base_features].copy()
    haz_va_h1["horizon"] = 1

    hazard = LGBMClassifier(**hazard_params)
    hazard.fit(haz_tr[base_features + ["horizon"]], haz_tr[TARGET])
    pred_hazard = hazard.predict_proba(haz_va_h1[base_features + ["horizon"]])[:, 1]
    oof_hazard[va_idx] = pred_hazard
    test_hazard += (
        hazard.predict_proba(X_test_h1[base_features + ["horizon"]])[:, 1]
        / folds.n_splits
    )

    blend = 0.70 * pred_main + 0.30 * pred_hazard
    score = roc_auc_score(y_va, blend)
    fold_scores.append(score)
    print(f"Fold {fold} ROC AUC: {score:.6f}")

oof_blend = 0.70 * oof_main + 0.30 * oof_hazard
cv_auc = roc_auc_score(y, oof_blend)
print(f"OOF ROC AUC: {cv_auc:.6f}")

test_pred = np.clip(0.70 * test_main + 0.30 * test_hazard, 0, 1)

submission = sample[[ID]].copy()
submission[TARGET] = test_pred
submission.to_csv("./working/submission.csv", index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train_enc)),
        "target": y,
        "prediction": oof_blend,
    }
).to_csv("./working/oof_predictions.csv.gz", index=False, compression="gzip")

test_predictions = sample[[ID]].copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    "./working/test_predictions.csv.gz", index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_auc": float(cv_auc),
            "fold_auc": [float(x) for x in fold_scores],
            "research_hypotheses_llm_claimed_used": ["000267"],
            "submission_path": "./working/submission.csv",
            "oof_path": "./working/oof_predictions.csv.gz",
            "test_predictions_path": "./working/test_predictions.csv.gz",
        },
        indent=2,
    )
)
