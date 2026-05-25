import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    from sklearn.model_selection import GroupKFold

    StratifiedGroupKFold = None

import lightgbm as lgb

INPUT_DIR = Path("./input")
WORKING_DIR = Path("./working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5


def add_same_lap_pit_wave_features(df):
    df = df.copy()
    group_cols = ["Year", "Race", "LapNumber"]

    pit = df["PitStop"].astype("float32")
    grouped = df.groupby(group_cols, observed=False)["PitStop"]
    lap_cars = grouped.transform("size").astype("float32")
    lap_pits = grouped.transform("sum").astype("float32")

    peer_cars = (lap_cars - 1).clip(lower=0)
    peer_pits = (lap_pits - pit).clip(lower=0)
    peer_rate = np.divide(
        peer_pits,
        peer_cars,
        out=np.zeros(len(df), dtype="float32"),
        where=peer_cars.to_numpy() > 0,
    )

    df["same_lap_peer_cars"] = peer_cars.astype("float32")
    df["same_lap_peer_pit_count"] = peer_pits.astype("float32")
    df["same_lap_peer_pit_rate"] = peer_rate.astype("float32")
    df["same_lap_any_peer_pit"] = (peer_pits > 0).astype("int8")
    df["same_lap_multi_peer_pit"] = (peer_pits >= 2).astype("int8")
    df["same_lap_pit_wave_strength"] = (np.log1p(peer_pits) * (1.0 + peer_rate)).astype(
        "float32"
    )

    tyre_life = df["TyreLife"].astype("float32").clip(lower=1)
    lap_number = df["LapNumber"].astype("float32").clip(lower=1)
    race_progress = df["RaceProgress"].astype("float32").clip(lower=0.001)

    df["degradation_per_tyre_lap"] = (
        df["Cumulative_Degradation"].astype("float32") / tyre_life
    ).clip(-100, 100)
    df["tyre_pressure_proxy"] = (
        np.log1p(tyre_life)
        * np.sign(df["Cumulative_Degradation"].astype("float32"))
        * np.log1p(np.abs(df["Cumulative_Degradation"].astype("float32")))
    ).astype("float32")
    df["tyre_life_fraction_of_race"] = (tyre_life / lap_number).clip(0, 5)

    estimated_total_laps = lap_number / race_progress
    laps_remaining = (estimated_total_laps - lap_number).clip(lower=0, upper=100)

    df["estimated_laps_remaining"] = laps_remaining.astype("float32")
    df["late_race"] = (df["RaceProgress"] >= 0.70).astype("int8")
    df["very_late_race"] = (df["RaceProgress"] >= 0.85).astype("int8")
    df["last_10_laps_proxy"] = (laps_remaining <= 10).astype("int8")

    compound = df["Compound"].astype(str).str.upper()
    is_slick = compound.isin(["SOFT", "MEDIUM", "HARD"])
    different_fresh_slick_feasible = (
        is_slick & (df["RaceProgress"] < 0.98) & (df["Stint"] < 7)
    )

    df["is_slick_compound"] = is_slick.astype("int8")
    df["different_fresh_slick_feasible"] = different_fresh_slick_feasible.astype("int8")
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        df[f"fresh_{comp.lower()}_different_feasible"] = (
            is_slick
            & (compound != comp)
            & (df["RaceProgress"] < 0.98)
            & (df["Stint"] < 7)
        ).astype("int8")

    wave_cols = [
        "same_lap_peer_pit_count",
        "same_lap_peer_pit_rate",
        "same_lap_pit_wave_strength",
    ]
    context_cols = [
        "TyreLife",
        "Cumulative_Degradation",
        "degradation_per_tyre_lap",
        "tyre_pressure_proxy",
        "late_race",
        "very_late_race",
        "last_10_laps_proxy",
        "different_fresh_slick_feasible",
    ]

    for w in wave_cols:
        for c in context_cols:
            df[f"{w}_x_{c}"] = (
                df[w].astype("float32") * df[c].astype("float32")
            ).astype("float32")

    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    return df


train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

train = add_same_lap_pit_wave_features(train)
test = add_same_lap_pit_wave_features(test)

y = train[TARGET].astype(int).to_numpy()
feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
cat_cols = [
    c for c in feature_cols if train[c].dtype == "object" or test[c].dtype == "object"
]

for c in cat_cols:
    tr = train[c].astype(str).fillna("__NA__")
    te = test[c].astype(str).fillna("__NA__")
    cats = pd.Index(pd.concat([tr, te], axis=0).unique())
    train[c] = pd.Categorical(tr, categories=cats)
    test[c] = pd.Categorical(te, categories=cats)

X = train[feature_cols]
X_test = test[feature_cols]
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

if StratifiedGroupKFold is not None:
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = cv.split(X, y, groups)
else:
    cv = GroupKFold(n_splits=N_SPLITS)
    splits = cv.split(X, y, groups)

params = dict(
    objective="binary",
    n_estimators=4000,
    learning_rate=0.025,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    n_jobs=max(1, (os.cpu_count() or 2) - 1),
    verbosity=-1,
)

oof = np.zeros(len(train), dtype=np.float32)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
    )

    best_iter = int(getattr(model, "best_iteration_", 0) or params["n_estimators"])
    best_iterations.append(best_iter)

    pred = model.predict_proba(X.iloc[va_idx], num_iteration=best_iter)[:, 1]
    oof[va_idx] = pred.astype(np.float32)

    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))
    print(f"Fold {fold} ROC AUC: {auc:.6f}  best_iteration={best_iter}")

cv_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int32),
        "target": y,
        "prediction": oof,
    }
).to_csv(WORKING_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

final_params = params.copy()
final_params["n_estimators"] = int(np.median(best_iterations))
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0, 1)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(WORKING_DIR / "submission.csv", index=False)
submission.to_csv(
    WORKING_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": fold_scores,
            "research_hypotheses_llm_claimed_used": ["000040"],
            "files_written": [
                str(WORKING_DIR / "submission.csv"),
                str(WORKING_DIR / "oof_predictions.csv.gz"),
                str(WORKING_DIR / "test_predictions.csv.gz"),
            ],
        },
        indent=2,
    )
)
