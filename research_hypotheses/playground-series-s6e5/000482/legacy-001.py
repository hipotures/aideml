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
CAT_COLS = ["Driver", "Race", "Compound"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_features = train.drop(columns=[TARGET]).copy()
test_features = test.copy()

train_features["_is_train"] = 1
test_features["_is_train"] = 0
all_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)

all_df["Race_Year"] = all_df["Year"].astype(str) + "_" + all_df["Race"].astype(str)
lap_group = ["Race_Year", "LapNumber"]


def add_same_lap_context(df):
    g = df.groupby(lap_group, sort=False)

    for col in ["TyreLife", "Cumulative_Degradation", "Position"]:
        df[f"{col}_lap_pct"] = g[col].rank(pct=True, method="average").astype("float32")

    lap_mean = g["LapTime_Delta"].transform("mean")
    lap_std = g["LapTime_Delta"].transform("std").replace(0, np.nan)
    df["LapTime_Delta_lap_z"] = (
        ((df["LapTime_Delta"] - lap_mean) / lap_std).fillna(0).astype("float32")
    )

    lap_size = g[ID_COL].transform("count").astype("float32")
    for compound in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        is_compound = (df["Compound"] == compound).astype("float32")
        cnt = (
            is_compound.groupby([df["Race_Year"], df["LapNumber"]])
            .transform("sum")
            .astype("float32")
        )
        df[f"lap_count_compound_{compound}"] = cnt
        df[f"lap_share_compound_{compound}"] = (cnt / lap_size).astype("float32")

    df["lap_pitstop_count"] = g["PitStop"].transform("sum").astype("float32")
    df["lap_pitstop_share"] = (df["lap_pitstop_count"] / lap_size).astype("float32")

    cs_group = ["Race_Year", "LapNumber", "Compound", "Stint"]
    csg = df.groupby(cs_group, sort=False)
    tyre_med = csg["TyreLife"].transform("median")
    tyre_std = csg["TyreLife"].transform("std").replace(0, np.nan)
    deg_med = csg["Cumulative_Degradation"].transform("median")
    deg_std = csg["Cumulative_Degradation"].transform("std").replace(0, np.nan)

    df["tyrelife_compound_stint_z"] = (
        ((df["TyreLife"] - tyre_med) / tyre_std).fillna(0).astype("float32")
    )
    df["degradation_compound_stint_z"] = (
        ((df["Cumulative_Degradation"] - deg_med) / deg_std).fillna(0).astype("float32")
    )
    df["is_early_tyre_outlier"] = (df["tyrelife_compound_stint_z"] <= -1.0).astype(
        "int8"
    )
    df["is_late_tyre_outlier"] = (df["tyrelife_compound_stint_z"] >= 1.0).astype("int8")

    df["tyrelife_x_raceprogress"] = (df["TyreLife"] * df["RaceProgress"]).astype(
        "float32"
    )
    df["degradation_per_tyre_lap"] = (
        (df["Cumulative_Degradation"] / df["TyreLife"].replace(0, np.nan))
        .fillna(0)
        .astype("float32")
    )
    return df


all_df = add_same_lap_context(all_df)

for col in CAT_COLS + ["Race_Year"]:
    all_df[col] = all_df[col].astype("category")

train_x = all_df.loc[all_df["_is_train"] == 1].drop(columns=["_is_train"])
test_x = all_df.loc[all_df["_is_train"] == 0].drop(columns=["_is_train"])
test_x = test_x[train_x.columns]

feature_cols = [c for c in train_x.columns if c != ID_COL]
X = train_x[feature_cols].copy()
X_test = test_x[feature_cols].copy()

cat_features = [c for c in CAT_COLS + ["Race_Year"] if c in X.columns]
groups = train_x["Race_Year"].astype(str).values
unique_groups = np.unique(groups)
n_splits = min(5, len(unique_groups))

params = dict(
    objective="binary",
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=60,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

oof = np.zeros(len(X), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)
fold_scores = []

cv = GroupKFold(n_splits=n_splits)
for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), 1):
    model = LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[],
    )

    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = va_pred
    test_pred += model.predict_proba(X_test)[:, 1] / n_splits

    fold_auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(fold_auc)
    print(f"fold_{fold}_roc_auc={fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"cv_roc_auc={cv_auc:.6f}")

submission = sample[[ID_COL]].copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample[[ID_COL]].copy()
test_pred_df[TARGET] = np.clip(test_pred, 0, 1)
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(v) for v in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000482"],
    "files_written": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
    ],
}
print(json.dumps(result, indent=2))
