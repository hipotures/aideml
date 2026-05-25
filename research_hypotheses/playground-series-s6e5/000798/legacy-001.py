import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

TARGET = "PitNextLap"
ID_COL = "id"


def add_features(df):
    df = df.copy()

    # Basic race-state features.
    df["Laps_Remaining_Est"] = np.maximum(
        0.0, df["LapNumber"] * (1.0 / np.clip(df["RaceProgress"], 1e-3, 1.0) - 1.0)
    )
    df["Race_Total_Laps_Est"] = df["LapNumber"] + df["Laps_Remaining_Est"]
    df["TyreLife_Ratio"] = df["TyreLife"] / np.maximum(df["Race_Total_Laps_Est"], 1.0)
    df["LapNumber_Ratio"] = df["LapNumber"] / np.maximum(df["Race_Total_Laps_Est"], 1.0)
    df["Deg_Per_Tyre_Lap"] = df["Cumulative_Degradation"] / np.maximum(
        df["TyreLife"], 1.0
    )
    df["Abs_Position_Change"] = df["Position_Change"].abs()
    df["LapTime_x_Progress"] = df["LapTime (s)"] * df["RaceProgress"]
    df["TyreLife_x_Deg"] = df["TyreLife"] * df["Deg_Per_Tyre_Lap"]
    df["Is_Recent_Pit"] = (df["PitStop"] == 1).astype(int)

    # Directional hypothesis features.
    df["Expected_Stint_Length"] = df["Race_Total_Laps_Est"] / np.maximum(
        df["Stint"] + 1.0, 1.0
    )
    df["Service_Age_Distance"] = df["TyreLife"] - df["Expected_Stint_Length"]
    df["Service_Window_Pressure"] = df["Service_Age_Distance"] / np.maximum(
        df["Expected_Stint_Length"], 1.0
    )
    df["TyreAge_At_Finish_Est"] = df["TyreLife"] + df["Laps_Remaining_Est"]
    df["Finish_Stop_Pressure"] = df["TyreAge_At_Finish_Est"] - np.maximum(
        1.35 * df["Expected_Stint_Length"], 1.0
    )
    df["Cannot_Finish_On_Current_Tyre"] = (
        df["TyreAge_At_Finish_Est"]
        > np.maximum(1.5 * df["Expected_Stint_Length"], 18.0)
    ).astype(int)

    # A freshness-style proxy: if the current row is a pit-stop row or tyre life is tiny,
    # next-lap pit probability should usually be suppressed.
    df["Laps_Since_Last_Pit"] = df["TyreLife"].astype(float)
    df["Fresh_Tyre_Suppression"] = np.maximum(0.0, 5.0 - df["Laps_Since_Last_Pit"])

    # Interactions that help the unconstrained model but are not used for monotone constraints.
    df["Compound_Race"] = df["Compound"].astype(str) + "_" + df["Race"].astype(str)
    df["Driver_Race"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)
    df["Year_Race"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)

    return df


full = pd.concat(
    [train.drop(columns=[TARGET]), test],
    axis=0,
    ignore_index=True,
)
full_fe = add_features(full)

cat_cols = ["Compound", "Driver", "Race", "Compound_Race", "Driver_Race", "Year_Race"]
for c in cat_cols:
    full_fe[c] = full_fe[c].astype("category")

train_fe = full_fe.iloc[: len(train)].reset_index(drop=True)
test_fe = full_fe.iloc[len(train) :].reset_index(drop=True)
y = train[TARGET].astype(int).values

drop_cols = [ID_COL]
features = [c for c in train_fe.columns if c not in drop_cols]
mono_features = [
    "Service_Window_Pressure",
    "Service_Age_Distance",
    "Finish_Stop_Pressure",
    "Cannot_Finish_On_Current_Tyre",
    "TyreAge_At_Finish_Est",
    "Laps_Since_Last_Pit",
    "Fresh_Tyre_Suppression",
    "TyreLife",
    "RaceProgress",
    "Laps_Remaining_Est",
    "Compound",
    "Race",
    "Year",
    "Stint",
    "Position",
    "PitStop",
]

mono_constraints = []
for c in mono_features:
    if c in [
        "Service_Window_Pressure",
        "Service_Age_Distance",
        "Finish_Stop_Pressure",
        "Cannot_Finish_On_Current_Tyre",
        "TyreAge_At_Finish_Est",
        "TyreLife",
        "RaceProgress",
    ]:
        mono_constraints.append(1)
    elif c in ["Laps_Since_Last_Pit", "Fresh_Tyre_Suppression", "Laps_Remaining_Est"]:
        mono_constraints.append(-1)
    else:
        mono_constraints.append(0)

main_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.045,
    num_leaves=96,
    max_depth=-1,
    min_data_in_leaf=80,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=1,
    lambda_l1=0.1,
    lambda_l2=1.5,
    verbosity=-1,
    n_jobs=-1,
    seed=42,
)

mono_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.055,
    num_leaves=31,
    max_depth=6,
    min_data_in_leaf=120,
    feature_fraction=0.95,
    bagging_fraction=0.90,
    bagging_freq=1,
    lambda_l1=0.2,
    lambda_l2=3.0,
    monotone_constraints=mono_constraints,
    monotone_constraints_method="advanced",
    verbosity=-1,
    n_jobs=-1,
    seed=2026,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
oof_main = np.zeros(len(train_fe))
oof_mono = np.zeros(len(train_fe))
test_main = np.zeros(len(test_fe))
test_mono = np.zeros(len(test_fe))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_fe, y), 1):
    X_tr, X_va = train_fe.iloc[tr_idx][features], train_fe.iloc[va_idx][features]
    y_tr, y_va = y[tr_idx], y[va_idx]

    main_model = lgb.LGBMClassifier(**main_params, n_estimators=4000)
    main_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=[c for c in cat_cols if c in features],
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    oof_main[va_idx] = main_model.predict_proba(X_va)[:, 1]
    test_main += main_model.predict_proba(test_fe[features])[:, 1] / skf.n_splits

    mono_cat = [c for c in ["Compound", "Race"] if c in mono_features]
    mono_model = lgb.LGBMClassifier(**mono_params, n_estimators=2500)
    mono_model.fit(
        train_fe.iloc[tr_idx][mono_features],
        y_tr,
        eval_set=[(train_fe.iloc[va_idx][mono_features], y_va)],
        eval_metric="auc",
        categorical_feature=mono_cat,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    oof_mono[va_idx] = mono_model.predict_proba(train_fe.iloc[va_idx][mono_features])[
        :, 1
    ]
    test_mono += mono_model.predict_proba(test_fe[mono_features])[:, 1] / skf.n_splits

    main_auc = roc_auc_score(y_va, oof_main[va_idx])
    mono_auc = roc_auc_score(y_va, oof_mono[va_idx])
    fold_scores.append((main_auc, mono_auc))
    print(f"fold={fold} main_auc={main_auc:.6f} monotone_auc={mono_auc:.6f}")

rank_main = rankdata(oof_main) / len(oof_main)
rank_mono = rankdata(oof_mono) / len(oof_mono)
test_rank_main = rankdata(test_main) / len(test_main)
test_rank_mono = rankdata(test_mono) / len(test_mono)

weights = np.linspace(0.0, 0.35, 36)
best_w, best_auc = 0.0, -1.0
for w in weights:
    pred = (1.0 - w) * rank_main + w * rank_mono
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_auc, best_w = auc, w

oof_blend = (1.0 - best_w) * rank_main + best_w * rank_mono
test_blend = (1.0 - best_w) * test_rank_main + best_w * test_rank_mono

cv_main = roc_auc_score(y, oof_main)
cv_mono = roc_auc_score(y, oof_mono)
cv_blend = roc_auc_score(y, oof_blend)

submission = sample.copy()
submission[TARGET] = test_blend
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_blend,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: sample[ID_COL].values,
        TARGET: test_blend,
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_main_auc": float(cv_main),
    "cv_monotone_auc": float(cv_mono),
    "cv_blend_auc": float(cv_blend),
    "best_monotone_rank_blend_weight": float(best_w),
    "fold_main_auc": [float(x[0]) for x in fold_scores],
    "fold_monotone_auc": [float(x[1]) for x in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000798"],
    "files_written": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
    ],
}
print(json.dumps(result, indent=2))
