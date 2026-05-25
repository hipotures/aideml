import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
SORT_COLS = ["Year", "Race", "Driver", "LapNumber", ID_COL]
GROUP_COLS = ["Year", "Race", "Driver"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["_is_train"] = 1
test["_is_train"] = 0
test[TARGET] = np.nan
all_df = pd.concat([train, test], axis=0, ignore_index=True)
all_df = all_df.sort_values(SORT_COLS).reset_index(drop=True)

for c in CAT_COLS:
    all_df[c] = all_df[c].astype("category")

g = all_df.groupby(GROUP_COLS, sort=False, observed=True)

all_df["RaceDriverRowsBefore"] = g.cumcount()
all_df["PrevPitStop"] = g["PitStop"].shift(1).fillna(0)
all_df["PitStopsBefore"] = g["PitStop"].cumsum() - all_df["PitStop"]
all_df["PrevTyreLife"] = g["TyreLife"].shift(1)
all_df["TyreLifeDelta"] = all_df["TyreLife"] - all_df["PrevTyreLife"]
all_df["PrevLapTime"] = g["LapTime (s)"].shift(1)
all_df["PrevLapDelta"] = g["LapTime_Delta"].shift(1)
all_df["PrevPosition"] = g["Position"].shift(1)
all_df["PositionDeltaFromPrev"] = all_df["Position"] - all_df["PrevPosition"]

pit_lap = all_df["LapNumber"].where(all_df["PitStop"].eq(1))
all_df["LastPitLap"] = pit_lap.groupby(
    [all_df[c] for c in GROUP_COLS], sort=False, observed=True
).ffill()
all_df["LastPitLapBefore"] = g["LastPitLap"].shift(1)
all_df["LapsSincePit"] = all_df["LapNumber"] - all_df["LastPitLapBefore"]
all_df["LapsSincePit"] = all_df["LapsSincePit"].fillna(all_df["TyreLife"])

slick = all_df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(np.int8)
all_df["PrevSlick"] = (
    slick.groupby([all_df[c] for c in GROUP_COLS], sort=False, observed=True)
    .shift(1)
    .fillna(0)
)
all_df["SlickLapsBefore"] = (
    slick.groupby([all_df[c] for c in GROUP_COLS], sort=False, observed=True).cumsum()
    - slick
)

for col in [
    "LapTime (s)",
    "LapTime_Delta",
    "TyreLife",
    "Position_Change",
    "Cumulative_Degradation",
]:
    shifted = g[col].shift(1)
    for w in [3, 5]:
        all_df[f"{col}_roll{w}_mean"] = (
            shifted.groupby([all_df[c] for c in GROUP_COLS], sort=False, observed=True)
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=list(range(len(GROUP_COLS))), drop=True)
        )
    all_df[f"{col}_exp_mean"] = (
        shifted.groupby([all_df[c] for c in GROUP_COLS], sort=False, observed=True)
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=list(range(len(GROUP_COLS))), drop=True)
    )

all_df["TyreLife_x_Progress"] = all_df["TyreLife"] * all_df["RaceProgress"]
all_df["Stint_x_TyreLife"] = all_df["Stint"] * all_df["TyreLife"]
all_df["ProgressRemaining"] = 1.0 - all_df["RaceProgress"]

drop_cols = [TARGET, "_is_train", ID_COL, "LastPitLap", "LastPitLapBefore"]
feature_cols = [c for c in all_df.columns if c not in drop_cols]

for c in feature_cols:
    if all_df[c].dtype.name == "category":
        continue
    all_df[c] = all_df[c].replace([np.inf, -np.inf], np.nan).fillna(-1)

train_fe = all_df[all_df["_is_train"].eq(1)].copy()
test_fe = all_df[all_df["_is_train"].eq(0)].copy()

cut = train_fe[ID_COL].quantile(0.8)
tr_idx = train_fe[ID_COL] <= cut
va_idx = train_fe[ID_COL] > cut

X_tr = train_fe.loc[tr_idx, feature_cols]
y_tr = train_fe.loc[tr_idx, TARGET].astype(int)
X_va = train_fe.loc[va_idx, feature_cols]
y_va = train_fe.loc[va_idx, TARGET].astype(int)

cat_features = [c for c in CAT_COLS if c in feature_cols]

model = LGBMClassifier(
    objective="binary",
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
    verbosity=-1,
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="auc",
    categorical_feature=cat_features,
    callbacks=[],
)

val_pred = model.predict_proba(X_va)[:, 1]
val_auc = roc_auc_score(y_va, val_pred)
print(f"holdout_roc_auc={val_auc:.6f}")

pd.DataFrame(
    {
        "row": X_va.index,
        "target": y_va.values,
        "prediction": val_pred,
    }
).to_csv(
    os.path.join(WORKING_DIR, "validation_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_model = LGBMClassifier(**model.get_params())
final_model.fit(
    train_fe[feature_cols],
    train_fe[TARGET].astype(int),
    categorical_feature=cat_features,
)

test_pred = final_model.predict_proba(test_fe[feature_cols])[:, 1]
test_pred = np.clip(test_pred, 0, 1)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "validation_metric": "roc_auc",
    "validation_score": float(val_auc),
    "research_hypotheses_llm_claimed_used": ["000497"],
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
with open(os.path.join(WORKING_DIR, "review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review, indent=2))
