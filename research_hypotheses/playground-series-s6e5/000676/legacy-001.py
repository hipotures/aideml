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
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 2026

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.rename(columns={"LapTime (s)": "LapTime_s"})
test = test.rename(columns={"LapTime (s)": "LapTime_s"})


def add_features(df):
    df = df.copy()
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["Driver_Race"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)

    wet = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    df["IsWetCompound"] = wet
    df["IsSlickCompound"] = 1 - wet

    abs_delta = df["LapTime_Delta"].abs()
    df["CautionRainProxy"] = (
        (df["IsWetCompound"] == 1) | (abs_delta > 12.0) | (df["LapTime_s"] > 180.0)
    ).astype(np.int8)

    progress = df["RaceProgress"].clip(lower=0.01)
    tyre_life = df["TyreLife"].clip(lower=1)
    lap_num = df["LapNumber"].clip(lower=1)

    df["Progress_Left"] = 1.0 - df["RaceProgress"]
    df["Estimated_Total_Laps"] = df["LapNumber"] / progress
    df["TyreLife_to_Lap"] = df["TyreLife"] / lap_num
    df["Deg_per_TyreLap"] = df["Cumulative_Degradation"] / tyre_life
    df["TyreLife_x_Progress"] = df["TyreLife"] * df["RaceProgress"]
    df["Stint_x_TyreLife"] = df["Stint"] * df["TyreLife"]
    df["PitStop_x_TyreLife"] = df["PitStop"] * df["TyreLife"]
    df["Abs_LapTime_Delta"] = abs_delta
    return df


all_df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
all_df = add_features(all_df)

cat_cols = ["Compound", "Driver", "Race", "Race_Year", "Driver_Race"]
for c in cat_cols:
    all_df[c] = all_df[c].astype("category")

train_x = all_df.iloc[: len(train)].reset_index(drop=True)
test_x = all_df.iloc[len(train) :].reset_index(drop=True)
y = train[TARGET].astype(int).to_numpy()
groups = train_x["Race_Year"].astype(str).to_numpy()

features = [c for c in train_x.columns if c != ID_COL]
cat_features = [c for c in cat_cols if c in features]


def fit_expert(x_fit, y_fit, seed):
    prior = float(np.mean(y_fit)) if len(y_fit) else 0.0
    if len(y_fit) < 300 or np.unique(y_fit).size < 2:
        return None, prior

    pos = float(np.sum(y_fit))
    neg = float(len(y_fit) - pos)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.045,
        num_leaves=48,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        scale_pos_weight=max(1.0, neg / max(pos, 1.0)),
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    model.fit(x_fit[features], y_fit, categorical_feature=cat_features)
    return model, prior


def predict_expert(model, prior, x):
    if model is None:
        return np.full(len(x), prior, dtype=float)
    return model.predict_proba(x[features])[:, 1]


gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(train), dtype=float)
test_fold_preds = []
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_x, y, groups), 1):
    x_tr = train_x.iloc[tr_idx]
    y_tr = y[tr_idx]
    x_va = train_x.iloc[va_idx]
    y_va = y[va_idx]

    expert_masks = {
        "slick": x_tr["IsWetCompound"].to_numpy() == 0,
        "wet": x_tr["IsWetCompound"].to_numpy() == 1,
        "green": x_tr["CautionRainProxy"].to_numpy() == 0,
        "caution": x_tr["CautionRainProxy"].to_numpy() == 1,
    }

    val_preds = {}
    tst_preds = {}

    for name, mask in expert_masks.items():
        model, prior = fit_expert(
            x_tr.loc[mask],
            y_tr[mask],
            SEED + fold * 10 + len(name),
        )
        val_preds[name] = predict_expert(model, prior, x_va)
        tst_preds[name] = predict_expert(model, prior, test_x)

    va_wet = x_va["IsWetCompound"].to_numpy() == 1
    va_caution = x_va["CautionRainProxy"].to_numpy() == 1
    te_wet = test_x["IsWetCompound"].to_numpy() == 1
    te_caution = test_x["CautionRainProxy"].to_numpy() == 1

    va_compound = np.where(va_wet, val_preds["wet"], val_preds["slick"])
    va_status = np.where(va_caution, val_preds["caution"], val_preds["green"])
    te_compound = np.where(te_wet, tst_preds["wet"], tst_preds["slick"])
    te_status = np.where(te_caution, tst_preds["caution"], tst_preds["green"])

    va_pred = 0.5 * va_compound + 0.5 * va_status
    te_pred = 0.5 * te_compound + 0.5 * te_status

    oof[va_idx] = va_pred
    test_fold_preds.append(te_pred)

    fold_auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} Race_Year GroupKFold ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
test_pred = np.mean(test_fold_preds, axis=0)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

print(f"OOF Race_Year GroupKFold ROC AUC: {cv_auc:.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000676"],
            "metric": "roc_auc",
            "cv_strategy": "5-fold GroupKFold by Race_Year",
            "fold_auc": [float(x) for x in fold_scores],
            "cv_auc": float(cv_auc),
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        }
    )
)
