import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("lightgbm is required for this solution") from e


INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]
REL_COLS = ["TyreLife", "Cumulative_Degradation", "LapTime_Delta", "Position"]


def add_relative_context_features(df):
    df = df.copy()
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["StintBucket"] = pd.cut(
        df["Stint"],
        bins=[0, 1, 2, 3, 99],
        labels=["s1", "s2", "s3", "s4p"],
        include_lowest=True,
    ).astype(str)

    lap_group = ["Race_Year", "LapNumber"]
    comp_group = ["Race_Year", "LapNumber", "Compound"]

    for col in REL_COLS:
        g = df.groupby(lap_group, observed=True)[col]
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        df[f"{col}_lap_pct"] = g.rank(pct=True).astype("float32")
        df[f"{col}_lap_z"] = ((df[col] - mean) / std).fillna(0).astype("float32")
        df[f"{col}_lap_median_diff"] = (df[col] - g.transform("median")).astype(
            "float32"
        )

        gc = df.groupby(comp_group, observed=True)[col]
        cmean = gc.transform("mean")
        cstd = gc.transform("std").replace(0, np.nan)
        df[f"{col}_comp_lap_pct"] = gc.rank(pct=True).astype("float32")
        df[f"{col}_comp_lap_z"] = ((df[col] - cmean) / cstd).fillna(0).astype("float32")
        df[f"{col}_comp_lap_median_diff"] = (df[col] - gc.transform("median")).astype(
            "float32"
        )

    lap_size = (
        df.groupby(lap_group, observed=True)[ID_COL]
        .transform("count")
        .astype("float32")
    )
    tyre_rank = (
        df.groupby(lap_group, observed=True)["TyreLife"]
        .rank(method="average")
        .astype("float32")
    )
    df["older_tyre_rival_share"] = (
        ((lap_size - tyre_rank) / (lap_size - 1).replace(0, np.nan))
        .fillna(0)
        .astype("float32")
    )

    comp_size = (
        df.groupby(comp_group, observed=True)[ID_COL]
        .transform("count")
        .astype("float32")
    )
    comp_tyre_rank = (
        df.groupby(comp_group, observed=True)["TyreLife"]
        .rank(method="average")
        .astype("float32")
    )
    df["older_same_compound_share"] = (
        ((comp_size - comp_tyre_rank) / (comp_size - 1).replace(0, np.nan))
        .fillna(0)
        .astype("float32")
    )

    df["alternate_compound_share"] = (
        ((lap_size - comp_size) / (lap_size - 1).replace(0, np.nan))
        .fillna(0)
        .astype("float32")
    )
    df["field_size"] = lap_size.astype("float32")
    df["same_compound_count"] = comp_size.astype("float32")
    df["compound_field_share"] = (comp_size / lap_size).astype("float32")

    pit_lap = (
        df.groupby(["Race_Year", "LapNumber", "Compound"], observed=True)["PitStop"]
        .sum()
        .rename("lag1_pit_count_compound")
        .reset_index()
    )
    pit_lap["LapNumber"] += 1
    df = df.merge(pit_lap, on=["Race_Year", "LapNumber", "Compound"], how="left")

    pit_stint = (
        df.groupby(
            ["Race_Year", "LapNumber", "Compound", "StintBucket"], observed=True
        )["PitStop"]
        .sum()
        .rename("lag1_pit_count_compound_stint")
        .reset_index()
    )
    pit_stint["LapNumber"] += 1
    df = df.merge(
        pit_stint, on=["Race_Year", "LapNumber", "Compound", "StintBucket"], how="left"
    )

    pit_any = (
        df.groupby(["Race_Year", "LapNumber"], observed=True)["PitStop"]
        .sum()
        .rename("lag1_pit_count_field")
        .reset_index()
    )
    pit_any["LapNumber"] += 1
    df = df.merge(pit_any, on=["Race_Year", "LapNumber"], how="left")

    for col in [
        "lag1_pit_count_compound",
        "lag1_pit_count_compound_stint",
        "lag1_pit_count_field",
    ]:
        df[col] = df[col].fillna(0).astype("float32")

    df["lag1_pit_share_field"] = (
        df["lag1_pit_count_field"] / df["field_size"].clip(lower=1)
    ).astype("float32")
    df["lag1_pit_share_compound"] = (
        df["lag1_pit_count_compound"] / df["same_compound_count"].clip(lower=1)
    ).astype("float32")

    return df


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train_len = len(train)
y = train[TARGET].astype(int).values

combined = pd.concat(
    [train.drop(columns=[TARGET]), test],
    axis=0,
    ignore_index=True,
)
combined = add_relative_context_features(combined)

for col in CAT_COLS + ["Race_Year", "StintBucket"]:
    combined[col] = combined[col].astype("category")

feature_cols = [c for c in combined.columns if c != ID_COL]
X = combined.iloc[:train_len][feature_cols].copy()
X_test = combined.iloc[train_len:][feature_cols].copy()

cat_features = [c for c in X.columns if str(X[c].dtype) == "category"]

params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.035,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=120,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=2.0,
    n_estimators=2500,
    random_state=42,
    n_jobs=max(1, os.cpu_count() or 1),
    verbosity=-1,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(train_len, dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    te_pred = model.predict_proba(X_test)[:, 1]

    oof[va_idx] = va_pred.astype(np.float32)
    test_pred += te_pred.astype(np.float32) / skf.n_splits

    auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(float(auc))
    print(f"fold {fold} roc_auc: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold cv roc_auc: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(train_len),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample.copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
test_pred_df.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_score": float(cv_auc),
            "fold_scores": fold_scores,
            "research_hypotheses_llm_claimed_used": ["000235"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
            "oof_path": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
            "test_predictions_path": os.path.join(
                WORKING_DIR, "test_predictions.csv.gz"
            ),
        },
        indent=2,
    )
)
