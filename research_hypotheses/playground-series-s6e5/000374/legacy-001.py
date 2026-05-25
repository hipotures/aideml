import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

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


def race_metadata(race):
    r = str(race).lower()

    street_terms = [
        "monaco",
        "singapore",
        "jeddah",
        "las vegas",
        "azerbaijan",
        "miami",
        "saudi",
        "baku",
    ]
    high_stress_terms = [
        "bahrain",
        "suzuka",
        "qatar",
        "silverstone",
        "spa",
        "hungarian",
        "japanese",
    ]
    low_stress_terms = [
        "monaco",
        "las vegas",
        "canadian",
        "mexico",
        "mexico city",
        "baku",
        "azerbaijan",
    ]
    undercut_terms = [
        "bahrain",
        "spain",
        "hungarian",
        "japanese",
        "qatar",
        "dutch",
        "abu dhabi",
        "austrian",
    ]
    overtake_hard_terms = [
        "monaco",
        "singapore",
        "hungarian",
        "dutch",
        "emilia",
        "imola",
        "japanese",
    ]
    sc_risk_terms = [
        "monaco",
        "singapore",
        "jeddah",
        "saudi",
        "azerbaijan",
        "baku",
        "las vegas",
        "miami",
        "canadian",
    ]

    is_street = int(any(t in r for t in street_terms))
    high_stress = int(any(t in r for t in high_stress_terms))
    low_stress = int(any(t in r for t in low_stress_terms))
    tyre_stress = 2 if high_stress else (0 if low_stress else 1)

    return pd.Series(
        {
            "circuit_street": is_street,
            "circuit_tyre_stress": tyre_stress,
            "circuit_undercut_friendly": int(any(t in r for t in undercut_terms)),
            "circuit_overtaking_hard": int(any(t in r for t in overtake_hard_terms)),
            "circuit_sc_risk": int(any(t in r for t in sc_risk_terms)),
        }
    )


def add_features(df):
    df = df.copy()

    meta = df["Race"].apply(race_metadata)
    df = pd.concat([df, meta], axis=1)

    race_year = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    max_lap_by_event = df.groupby(race_year)["LapNumber"].transform("max")
    df["LapsRemaining_Est"] = (max_lap_by_event - df["LapNumber"]).clip(lower=0)

    df["TyreLife_x_tyre_stress"] = df["TyreLife"] * df["circuit_tyre_stress"]
    df["TyreLife_x_undercut"] = df["TyreLife"] * df["circuit_undercut_friendly"]
    df["TyreLife_x_overtaking_hard"] = df["TyreLife"] * df["circuit_overtaking_hard"]
    df["LapDelta_x_tyre_stress"] = df["LapTime_Delta"] * df["circuit_tyre_stress"]
    df["LapDelta_x_undercut"] = df["LapTime_Delta"] * df["circuit_undercut_friendly"]
    df["Position_x_overtaking_hard"] = df["Position"] * df["circuit_overtaking_hard"]
    df["Position_x_undercut"] = df["Position"] * df["circuit_undercut_friendly"]
    df["LapsRemain_x_tyre_stress"] = df["LapsRemaining_Est"] * df["circuit_tyre_stress"]
    df["LapsRemain_x_sc_risk"] = df["LapsRemaining_Est"] * df["circuit_sc_risk"]
    df["RaceProgress_x_street"] = df["RaceProgress"] * df["circuit_street"]

    for col in ["Compound", "Driver", "Race"]:
        df[col] = df[col].astype("category")

    return df


train_fe = add_features(train)
test_fe = add_features(test)

features = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
X = train_fe[features]
y = train_fe[TARGET].astype(int)
X_test = test_fe[features]

groups = train_fe["Race"].astype(str) + "_" + train_fe["Year"].astype(str)
cv = GroupKFold(n_splits=5)

oof = np.zeros(len(train_fe), dtype=float)
test_pred = np.zeros(len(test_fe), dtype=float)
fold_scores = []

cat_features = [c for c in ["Compound", "Driver", "Race"] if c in features]

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        learning_rate=0.035,
        n_estimators=2500,
        num_leaves=64,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    va_pred = model.predict_proba(X_va)[:, 1]
    oof[va_idx] = va_pred
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits

    auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(auc)
    print(f"Fold {fold} Race_Year grouped ROC AUC: {auc:.6f}")

mean_auc = roc_auc_score(y, oof)
print(f"Mean OOF ROC AUC: {mean_auc:.6f}")

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
            "cv": "5-fold GroupKFold by Race_Year",
            "fold_auc": [float(x) for x in fold_scores],
            "mean_oof_auc": float(mean_auc),
            "research_hypotheses_llm_claimed_used": ["000374"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        },
        indent=2,
    )
)
