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
GROUP_COLS = ["Year", "Race", "LapNumber"]
CAT_COLS = ["Driver", "Race", "Compound"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_len = len(train)

all_df = pd.concat(
    [train.drop(columns=[TARGET]), test],
    axis=0,
    ignore_index=True,
)
all_df["_is_train"] = np.r_[
    np.ones(train_len, dtype=np.int8), np.zeros(len(test), dtype=np.int8)
]


def add_context_features(df):
    df = df.copy()

    grp = df.groupby(GROUP_COLS, sort=False)
    lap_count = grp[ID_COL].transform("count").astype(float)
    lap_pits = grp["PitStop"].transform("sum").astype(float)

    df["lap_field_size"] = lap_count
    df["lap_pit_count"] = lap_pits
    df["lap_pit_frac"] = lap_pits / lap_count.clip(lower=1)
    df["lap_other_pit_count"] = lap_pits - df["PitStop"]
    df["lap_other_pit_frac"] = df["lap_other_pit_count"] / (lap_count - 1).clip(lower=1)

    for col in [
        "LapTime (s)",
        "LapTime_Delta",
        "TyreLife",
        "Cumulative_Degradation",
        "Position_Change",
    ]:
        mean = grp[col].transform("mean")
        std = grp[col].transform("std").fillna(0)
        df[f"{col}_lap_z"] = (df[col] - mean) / std.replace(0, np.nan)
        df[f"{col}_lap_z"] = df[f"{col}_lap_z"].fillna(0)

    df["field_zone"] = pd.cut(
        df["Position"],
        bins=[0, 6, 14, 25],
        labels=["front", "mid", "back"],
        include_lowest=True,
    ).astype(str)

    zone_grp = df.groupby(GROUP_COLS + ["field_zone"], sort=False)
    zone_count = zone_grp[ID_COL].transform("count").astype(float)
    zone_pits = zone_grp["PitStop"].transform("sum").astype(float)
    df["zone_pit_frac"] = zone_pits / zone_count.clip(lower=1)
    df["zone_other_pit_frac"] = (zone_pits - df["PitStop"]) / (zone_count - 1).clip(
        lower=1
    )

    ordered = df.sort_values(GROUP_COLS + ["Position", ID_COL]).copy()
    pos_grp = ordered.groupby(GROUP_COLS, sort=False)

    ordered["ahead_pit"] = pos_grp["PitStop"].shift(1).fillna(0)
    ordered["behind_pit"] = pos_grp["PitStop"].shift(-1).fillna(0)
    ordered["ahead_tyre_life"] = pos_grp["TyreLife"].shift(1)
    ordered["behind_tyre_life"] = pos_grp["TyreLife"].shift(-1)
    ordered["ahead_laptime"] = pos_grp["LapTime (s)"].shift(1)
    ordered["behind_laptime"] = pos_grp["LapTime (s)"].shift(-1)

    ordered["neighbor_pit_count"] = ordered["ahead_pit"] + ordered["behind_pit"]
    ordered["neighbor_any_pit"] = (ordered["neighbor_pit_count"] > 0).astype(int)
    ordered["ahead_tyre_delta"] = ordered["TyreLife"] - ordered["ahead_tyre_life"]
    ordered["behind_tyre_delta"] = ordered["TyreLife"] - ordered["behind_tyre_life"]
    ordered["ahead_pace_delta"] = ordered["LapTime (s)"] - ordered["ahead_laptime"]
    ordered["behind_pace_delta"] = ordered["LapTime (s)"] - ordered["behind_laptime"]

    neighbor_cols = [
        "ahead_pit",
        "behind_pit",
        "neighbor_pit_count",
        "neighbor_any_pit",
        "ahead_tyre_delta",
        "behind_tyre_delta",
        "ahead_pace_delta",
        "behind_pace_delta",
    ]
    df = ordered.sort_index()
    for c in neighbor_cols:
        df[c] = df[c].fillna(0)

    df["pit_wave_x_tyre_life"] = df["lap_other_pit_frac"] * df["TyreLife"]
    df["pit_wave_x_race_progress"] = df["lap_other_pit_frac"] * df["RaceProgress"]
    df["pit_wave_x_stint"] = df["lap_other_pit_frac"] * df["Stint"]
    df["neighbor_pit_x_position"] = df["neighbor_pit_count"] * df["Position"]
    df["zone_pit_x_tyre_life"] = df["zone_other_pit_frac"] * df["TyreLife"]

    df["compound_stint"] = df["Compound"].astype(str) + "_s" + df["Stint"].astype(str)
    df["compound_phase"] = (
        df["Compound"].astype(str)
        + "_p"
        + pd.cut(
            df["RaceProgress"],
            bins=[0, 0.33, 0.66, 1.01],
            labels=["early", "mid", "late"],
            include_lowest=True,
        ).astype(str)
    )

    return df


all_df = add_context_features(all_df)

for col in CAT_COLS + ["field_zone", "compound_stint", "compound_phase"]:
    all_df[col] = all_df[col].astype("category")

drop_cols = [ID_COL, "_is_train"]
features = [c for c in all_df.columns if c not in drop_cols]

X = all_df.iloc[:train_len][features].copy()
X_test = all_df.iloc[train_len:][features].copy()
test_ids = sample[ID_COL].values

groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
gkf = GroupKFold(n_splits=5)

oof = np.zeros(train_len)
test_pred = np.zeros(len(test))

params = dict(
    objective="binary",
    n_estimators=1400,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    verbose=-1,
)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
    model = LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature="auto",
        callbacks=[],
    )
    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / gkf.n_splits
    fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold GroupKFold ROC AUC: {cv_auc:.6f}")

submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred})
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(train_len),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_strategy": "5-fold GroupKFold by Year-Race",
            "validation_roc_auc": float(cv_auc),
            "research_hypotheses_llm_claimed_used": ["001030"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
            "oof_path": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
            "test_predictions_path": os.path.join(
                WORKING_DIR, "test_predictions.csv.gz"
            ),
        },
        indent=2,
    )
)
