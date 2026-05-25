import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESES = ["000490"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["is_train"] = 1
test["is_train"] = 0
test[TARGET] = np.nan
df = pd.concat([train, test], ignore_index=True)

df["Race_Year"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)
df["Race_Year_Driver"] = df["Race_Year"].astype(str) + "_" + df["Driver"].astype(str)

sort_cols = ["Race_Year", "Driver", "LapNumber", ID_COL]
df = df.sort_values(sort_cols).reset_index(drop=True)

dry_compounds = ["SOFT", "MEDIUM", "HARD"]
wet_compounds = ["INTERMEDIATE", "WET"]

df["IsWetCompound"] = df["Compound"].isin(wet_compounds).astype(np.int8)
df["IsDryCompound"] = df["Compound"].isin(dry_compounds).astype(np.int8)
df["WetSeenSoFar"] = (
    df.groupby("Race_Year_Driver")["IsWetCompound"].cummax().astype(np.int8)
)

for comp in dry_compounds:
    c = f"Used_{comp}_SoFar"
    df[c] = (df["Compound"].eq(comp)).astype(np.int8)
    df[c] = df.groupby("Race_Year_Driver")[c].cummax().astype(np.int8)

df["DistinctDryCompoundsSoFar"] = (
    df[[f"Used_{c}_SoFar" for c in dry_compounds]].sum(axis=1).astype(np.int8)
)
df["DryCompoundDebt"] = (
    (df["WetSeenSoFar"] == 0) & (df["DistinctDryCompoundsSoFar"] < 2)
).astype(np.int8)
df["DryCompoundsStillNeeded"] = np.where(
    df["WetSeenSoFar"] == 0, np.maximum(0, 2 - df["DistinctDryCompoundsSoFar"]), 0
).astype(np.int8)

race_max_lap = df.groupby("Race_Year")["LapNumber"].transform("max")
df["LapsRemaining_Est"] = (
    (race_max_lap - df["LapNumber"]).clip(lower=0).astype(np.float32)
)

reach_proxy = (
    df.groupby(["Race_Year", "Compound"])["TyreLife"]
    .transform(lambda s: s.quantile(0.85))
    .fillna(df["TyreLife"].median())
)
df["CurrentTyreCanReachFinish"] = (
    (df["TyreLife"] + df["LapsRemaining_Est"]) <= reach_proxy
).astype(np.int8)

df["Debt_x_LapsRemaining"] = df["DryCompoundDebt"] * df["LapsRemaining_Est"]
df["Debt_x_CannotReachFinish"] = df["DryCompoundDebt"] * (
    1 - df["CurrentTyreCanReachFinish"]
)
df["Debt_x_LateRace"] = df["DryCompoundDebt"] * (df["RaceProgress"] > 0.70).astype(
    np.int8
)
df["WetSeen_x_LapsRemaining"] = df["WetSeenSoFar"] * df["LapsRemaining_Est"]

for col in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "TyreLife"]:
    df[f"{col}_drv_prev"] = df.groupby("Race_Year_Driver")[col].shift(1)
    df[f"{col}_drv_delta"] = df[col] - df[f"{col}_drv_prev"]
    df[f"{col}_drv_prev"] = df[f"{col}_drv_prev"].fillna(df[col])
    df[f"{col}_drv_delta"] = df[f"{col}_drv_delta"].fillna(0)

df = df.sort_values(ID_COL).reset_index(drop=True)
train_fe = df[df["is_train"] == 1].copy()
test_fe = df[df["is_train"] == 0].copy()

drop_cols = [TARGET, "is_train", ID_COL, "Race_Year_Driver"]
features = [c for c in train_fe.columns if c not in drop_cols]

cat_cols = ["Compound", "Driver", "Race", "Race_Year"]
for col in cat_cols:
    both = pd.concat([train_fe[col], test_fe[col]], axis=0).astype("category")
    train_fe[col] = pd.Categorical(train_fe[col], categories=both.cat.categories)
    test_fe[col] = pd.Categorical(test_fe[col], categories=both.cat.categories)

X = train_fe[features]
y = train_fe[TARGET].astype(int).values
X_test = test_fe[features]
groups = train_fe["Race_Year"].values

try:
    from lightgbm import LGBMClassifier

    model_params = dict(
        objective="binary",
        n_estimators=1400,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.25,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbosity=-1,
    )

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train_fe), dtype=np.float32)
    test_pred = np.zeros(len(test_fe), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
        model = LGBMClassifier(**model_params)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            categorical_feature=cat_cols,
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
        print(f"fold {fold} auc: {roc_auc_score(y[va_idx], oof[va_idx]):.6f}")

except Exception:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    num_cols = [c for c in features if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat_cols),
        ]
    )
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(train_fe), dtype=np.float32)
    test_pred = np.zeros(len(test_fe), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
        model = Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", solver="saga", n_jobs=-1
                    ),
                ),
            ]
        )
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
        print(f"fold {fold} auc: {roc_auc_score(y[va_idx], oof[va_idx]):.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold grouped CV ROC AUC: {cv_auc:.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": HYPOTHESES,
            "cv_roc_auc": float(cv_auc),
        }
    )
)

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
test_predictions[TARGET] = np.clip(test_pred, 0, 1)
test_predictions.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
test_predictions.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
