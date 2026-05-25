import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SLICKS = {"SOFT", "MEDIUM", "HARD"}
WET_TYRES = {"INTERMEDIATE", "WET"}

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_strategy_features(df):
    df = df.copy()
    df["_orig_order"] = np.arange(len(df))
    df["race_driver"] = (
        df["Year"].astype(str)
        + "_"
        + df["Race"].astype(str)
        + "_"
        + df["Driver"].astype(str)
    )
    df["event_group"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)

    df["is_slick"] = df["Compound"].isin(SLICKS).astype(np.int8)
    df["is_wet_tyre"] = df["Compound"].isin(WET_TYRES).astype(np.int8)
    df["is_soft"] = (df["Compound"] == "SOFT").astype(np.int8)
    df["is_medium"] = (df["Compound"] == "MEDIUM").astype(np.int8)
    df["is_hard"] = (df["Compound"] == "HARD").astype(np.int8)
    df["is_intermediate"] = (df["Compound"] == "INTERMEDIATE").astype(np.int8)
    df["is_full_wet"] = (df["Compound"] == "WET").astype(np.int8)

    sort_cols = ["race_driver", "LapNumber", ID_COL]
    df = df.sort_values(sort_cols)

    g = df.groupby("race_driver", sort=False)
    df["race_total_laps_so_far"] = g["LapNumber"].transform("max")
    df["laps_remaining_proxy"] = (df["race_total_laps_so_far"] - df["LapNumber"]).clip(
        lower=0
    )
    df["wet_seen_so_far"] = g["is_wet_tyre"].cumsum().clip(upper=1).astype(np.int8)
    df["dry_legal_regime"] = (
        (df["wet_seen_so_far"] == 0) & (df["is_slick"] == 1)
    ).astype(np.int8)
    df["wet_regime"] = (1 - df["dry_legal_regime"]).astype(np.int8)

    for comp in ["SOFT", "MEDIUM", "HARD"]:
        seen = g["Compound"].transform(
            lambda s, c=comp: (s.eq(c)).cummax().astype(np.int8)
        )
        df[f"seen_{comp.lower()}_so_far"] = seen.astype(np.int8)

    df["unique_slicks_so_far"] = (
        df["seen_soft_so_far"] + df["seen_medium_so_far"] + df["seen_hard_so_far"]
    ).astype(np.int8)
    df["currently_first_unique_slick"] = (
        (df["dry_legal_regime"] == 1) & (df["unique_slicks_so_far"] == 1)
    ).astype(np.int8)
    df["currently_second_or_more_unique_slick"] = (
        (df["dry_legal_regime"] == 1) & (df["unique_slicks_so_far"] >= 2)
    ).astype(np.int8)
    df["mandatory_second_dry_needed"] = (
        (df["dry_legal_regime"] == 1) & (df["unique_slicks_so_far"] < 2)
    ).astype(np.int8)
    df["different_slick_options"] = (
        3 - df["unique_slicks_so_far"].clip(lower=0, upper=3)
    ).astype(np.int8)
    df["laps_remaining_after_hypothetical_stop"] = (
        df["laps_remaining_proxy"] - 1
    ).clip(lower=0)
    df["dry_stop_pressure"] = df["mandatory_second_dry_needed"] * (
        1.0 / (df["laps_remaining_after_hypothetical_stop"] + 1.0)
    )
    df["dry_legal_tyre_life"] = df["dry_legal_regime"] * df["TyreLife"]
    df["dry_legal_progress"] = df["dry_legal_regime"] * df["RaceProgress"]
    df["wet_warmup_phase"] = ((df["wet_regime"] == 1) & (df["TyreLife"] <= 3)).astype(
        np.int8
    )
    df["wet_tyre_life"] = df["wet_regime"] * df["TyreLife"]
    df["wet_degradation"] = df["wet_regime"] * df["Cumulative_Degradation"]
    df["wet_laptime_delta"] = df["wet_regime"] * df["LapTime_Delta"]
    df["wet_progress"] = df["wet_regime"] * df["RaceProgress"]

    df["tyre_life_x_progress"] = df["TyreLife"] * df["RaceProgress"]
    df["degradation_per_tyre_lap"] = df["Cumulative_Degradation"] / (
        df["TyreLife"] + 1.0
    )
    df["lap_delta_abs"] = df["LapTime_Delta"].abs()
    df["position_loss"] = df["Position_Change"].clip(lower=0)
    df["position_gain"] = (-df["Position_Change"]).clip(lower=0)

    return df.sort_values("_orig_order").drop(columns=["_orig_order"])


train_fe = add_strategy_features(train)
test_fe = add_strategy_features(test)

features = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
cat_cols = ["Compound", "Driver", "Race", "race_driver", "event_group"]
for c in cat_cols:
    all_vals = pd.concat([train_fe[c], test_fe[c]], axis=0).astype("category")
    cats = all_vals.cat.categories
    train_fe[c] = pd.Categorical(train_fe[c], categories=cats)
    test_fe[c] = pd.Categorical(test_fe[c], categories=cats)

X = train_fe[features]
y = train_fe[TARGET].astype(int)
X_test = test_fe[features]
groups = train_fe["event_group"].astype(str)

params = dict(
    objective="binary",
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    random_state=284,
    n_jobs=-1,
    verbose=-1,
)

oof = np.zeros(len(train_fe), dtype=float)
test_pred = np.zeros(len(test_fe), dtype=float)
fold_scores = []

cv = GroupKFold(n_splits=5)
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    model = LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        categorical_feature=cat_cols,
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        eval_metric="auc",
    )
    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = va_pred
    fold_auc = roc_auc_score(y.iloc[va_idx], va_pred)
    fold_scores.append(fold_auc)
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    print(f"fold_{fold}_auc={fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"cv_roc_auc={cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y.values,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_predictions = sample[[ID_COL]].copy()
test_predictions[TARGET] = np.clip(test_pred, 0, 1)
test_predictions.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
test_predictions.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": [float(x) for x in fold_scores],
            "research_hypotheses_llm_claimed_used": ["000284"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        }
    )
)
