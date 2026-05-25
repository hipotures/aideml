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

target_col = "PitNextLap"
id_col = "id"


def add_features(df):
    df = df.copy()

    eps = 1e-6
    df["FreshTyre"] = (df["TyreLife"] <= 2).astype(np.int8)
    df["VeryFreshTyre"] = (df["TyreLife"] <= 1).astype(np.int8)
    df["OldTyre"] = (df["TyreLife"] >= 20).astype(np.int8)

    est_total_laps = df["LapNumber"] / np.clip(df["RaceProgress"], eps, None)
    est_total_laps = est_total_laps.replace([np.inf, -np.inf], np.nan)
    df["EstimatedTotalLaps"] = est_total_laps.fillna(df["LapNumber"])
    df["LapsRemaining_Est"] = np.maximum(df["EstimatedTotalLaps"] - df["LapNumber"], 0)
    df["LateRace"] = (df["RaceProgress"] >= 0.75).astype(np.int8)
    df["MidRace"] = ((df["RaceProgress"] >= 0.35) & (df["RaceProgress"] < 0.75)).astype(
        np.int8
    )

    df["TyreFinishPressure"] = df["TyreLife"] / (df["LapsRemaining_Est"] + 1.0)
    df["CurrentTyreFinishPressure"] = df["TyreFinishPressure"] * (1 + df["OldTyre"])

    # Hypothesis 000053: current pit-stop interaction block, using only current-row state.
    df["PitStop_x_TyreLife"] = df["PitStop"] * df["TyreLife"]
    df["PitStop_x_FreshTyre"] = df["PitStop"] * df["FreshTyre"]
    df["PitStop_x_VeryFreshTyre"] = df["PitStop"] * df["VeryFreshTyre"]
    df["PitStop_x_Stint"] = df["PitStop"] * df["Stint"]
    df["PitStop_x_RaceProgress"] = df["PitStop"] * df["RaceProgress"]
    df["PitStop_x_LapsRemaining_Est"] = df["PitStop"] * df["LapsRemaining_Est"]
    df["PitStop_x_TyreFinishPressure"] = df["PitStop"] * df["CurrentTyreFinishPressure"]
    df["PitStop_x_LateRace"] = df["PitStop"] * df["LateRace"]
    df["PitStop_x_MidRace"] = df["PitStop"] * df["MidRace"]

    for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        df[f"PitStop_x_Compound_{comp}"] = df["PitStop"] * (
            df["Compound"].astype(str) == comp
        ).astype(np.int8)

    df["TyreLife_x_RaceProgress"] = df["TyreLife"] * df["RaceProgress"]
    df["Stint_x_RaceProgress"] = df["Stint"] * df["RaceProgress"]
    df["Position_x_RaceProgress"] = df["Position"] * df["RaceProgress"]

    return df


train_fe = add_features(train)
test_fe = add_features(test)

y = train_fe[target_col].astype(int)
drop_cols = [target_col, id_col]
features = [c for c in train_fe.columns if c not in drop_cols]

categorical_cols = [c for c in features if train_fe[c].dtype == "object"]
for col in categorical_cols:
    train_fe[col] = train_fe[col].astype("category")
    test_fe[col] = pd.Categorical(test_fe[col], categories=train_fe[col].cat.categories)

X = train_fe[features]
X_test = test_fe[features]

params = dict(
    objective="binary",
    n_estimators=1400,
    learning_rate=0.035,
    num_leaves=96,
    max_depth=-1,
    min_child_samples=90,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    random_state=53,
    n_jobs=-1,
    verbose=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=53)
oof = np.zeros(len(train_fe), dtype=np.float32)
test_pred = np.zeros(len(test_fe), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[],
    )

    val_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = val_pred
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits

    auc = roc_auc_score(y.iloc[va_idx], val_pred)
    fold_scores.append(float(auc))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[target_col] = np.clip(test_pred, 0, 1)
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

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "oof_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
    "research_hypotheses_llm_claimed_used": ["000053"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print(json.dumps(result, indent=2))
