import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target = "PitNextLap"
id_col = "id"

slick_compounds = {"SOFT", "MEDIUM", "HARD"}
wet_compounds = {"INTERMEDIATE", "WET"}


def add_weather_class_features(df):
    df = df.copy()

    compound = df["Compound"].astype(str)
    df["Is_Slick"] = compound.isin(slick_compounds).astype(np.int8)
    df["Is_WetWeather"] = compound.isin(wet_compounds).astype(np.int8)

    est_total_laps = (df["LapNumber"] / df["RaceProgress"].clip(0.01, 1.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    est_total_laps = est_total_laps.fillna(df["LapNumber"]).clip(df["LapNumber"], 90)
    remaining_after_next_lap_stop = (est_total_laps - df["LapNumber"] - 1).clip(lower=0)

    slick_capacity = {
        "SOFT": 20.0,
        "MEDIUM": 32.0,
        "HARD": 45.0,
    }
    wet_capacity = {
        "INTERMEDIATE": 32.0,
        "WET": 36.0,
    }

    slick_best = max(slick_capacity.values())
    slick_mean = float(np.mean(list(slick_capacity.values())))
    wet_best = max(wet_capacity.values())
    wet_mean = float(np.mean(list(wet_capacity.values())))

    df["RemainingLaps_AfterNextStop"] = remaining_after_next_lap_stop
    df["Slick_Best_Margin"] = slick_best - remaining_after_next_lap_stop
    df["Slick_Mean_Margin"] = slick_mean - remaining_after_next_lap_stop
    df["Wet_Best_Margin"] = wet_best - remaining_after_next_lap_stop
    df["Wet_Mean_Margin"] = wet_mean - remaining_after_next_lap_stop

    df["DryRow_SlickCanFinish"] = (
        (df["Is_Slick"] == 1) & (df["Slick_Best_Margin"] >= 0)
    ).astype(np.int8)
    df["WetRow_WetCanFinish"] = (
        (df["Is_WetWeather"] == 1) & (df["Wet_Best_Margin"] >= 0)
    ).astype(np.int8)

    df["ClassMatched_Best_Margin"] = np.where(
        df["Is_Slick"] == 1, df["Slick_Best_Margin"], df["Wet_Best_Margin"]
    )
    df["ClassMatched_Mean_Margin"] = np.where(
        df["Is_Slick"] == 1, df["Slick_Mean_Margin"], df["Wet_Mean_Margin"]
    )
    df["MixedClass_Best_Margin"] = np.where(
        df["Is_Slick"] == 1, df["Wet_Best_Margin"], df["Slick_Best_Margin"]
    )

    df["CanFinish_On_ClassMatchedFreshTyre"] = (
        df["ClassMatched_Best_Margin"] >= 0
    ).astype(np.int8)
    df["CanFinish_On_MixedClassFreshTyre"] = (df["MixedClass_Best_Margin"] >= 0).astype(
        np.int8
    )
    df["MixedClass_Switch_Feasible"] = (
        (df["CanFinish_On_MixedClassFreshTyre"] == 1)
        & (df["CanFinish_On_ClassMatchedFreshTyre"] == 0)
    ).astype(np.int8)

    df["TyreLife_To_ClassMatchedMargin"] = df["TyreLife"] / (
        df["ClassMatched_Best_Margin"].abs() + 1.0
    )
    df["TyreLife_To_MixedClassMargin"] = df["TyreLife"] / (
        df["MixedClass_Best_Margin"].abs() + 1.0
    )

    return df


train_fe = add_weather_class_features(train)
test_fe = add_weather_class_features(test)

features = [c for c in train_fe.columns if c not in [target, id_col]]
cat_cols = [c for c in features if train_fe[c].dtype == "object"]

for c in cat_cols:
    train_fe[c] = train_fe[c].astype("category")
    test_fe[c] = pd.Categorical(test_fe[c], categories=train_fe[c].cat.categories)

X = train_fe[features]
y = train_fe[target].astype(int)
X_test = test_fe[features]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=52)
oof = np.zeros(len(train_fe), dtype=np.float32)
test_pred = np.zeros(len(test_fe), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=5200 + fold,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
    )

    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = va_pred
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits

    score = roc_auc_score(y.iloc[va_idx], va_pred)
    fold_scores.append(score)
    print(f"Fold {fold} ROC AUC: {score:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

submission = sample[[id_col]].copy()
submission[target] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y.values,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        id_col: sample[id_col].values,
        target: np.clip(test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": [float(s) for s in fold_scores],
            "research_hypotheses_llm_claimed_used": ["000052"],
        },
        f,
        indent=2,
    )
