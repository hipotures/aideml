import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values


def add_features(df):
    out = df.copy()
    total_laps_by_race_year = out.groupby(["Year", "Race"])["LapNumber"].transform(
        "max"
    )
    out["EstimatedTotalLaps"] = np.maximum(total_laps_by_race_year, out["LapNumber"])
    out["RemainingLaps"] = (out["EstimatedTotalLaps"] - out["LapNumber"]).clip(lower=0)
    out["Positive_LapTime_Delta"] = out["LapTime_Delta"].clip(lower=0)
    out["Negative_LapTime_Delta"] = (-out["LapTime_Delta"]).clip(lower=0)
    out["CurrentTyreFinishExcess"] = (out["TyreLife"] - out["RemainingLaps"]).clip(
        lower=0
    )
    out["TyreLife_To_Remaining"] = out["TyreLife"] / (out["RemainingLaps"] + 1.0)
    out["FinishDeficit"] = 1.0 - out["RaceProgress"]
    out["IsWetCompound"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    out["IsDryCompound"] = 1 - out["IsWetCompound"]
    out["DryCompoundUsed"] = (
        out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(int)
    )
    out["StopDebtAfterCurrent"] = (
        (out["IsDryCompound"] == 1) & (out["Stint"] <= 1) & (out["RaceProgress"] > 0.25)
    ).astype(int)
    out["TyreLife_x_Dry"] = out["TyreLife"] * out["IsDryCompound"]
    out["PositiveDelta_x_Dry"] = out["Positive_LapTime_Delta"] * out["IsDryCompound"]
    out["Excess_x_Dry"] = out["CurrentTyreFinishExcess"] * out["IsDryCompound"]
    out["LateRace"] = (out["RaceProgress"] > 0.75).astype(int)
    out["EarlyRace"] = (out["RaceProgress"] < 0.25).astype(int)
    out["LapTime_per_TyreLife"] = out["LapTime (s)"] / (out["TyreLife"] + 1.0)
    out["Degradation_per_TyreLife"] = out["Cumulative_Degradation"] / (
        out["TyreLife"] + 1.0
    )
    return out


full = pd.concat(
    [train.drop(columns=[TARGET]), test],
    axis=0,
    ignore_index=True,
)
full = add_features(full)

for col in CAT_COLS:
    full[col] = full[col].astype(str)
    counts = full[col].map(full[col].value_counts())
    full[f"{col}_count"] = counts.astype(np.float32)

train_fe = full.iloc[: len(train)].reset_index(drop=True)
test_fe = full.iloc[len(train) :].reset_index(drop=True)

drop_cols = [ID_COL]
base_features = [
    c for c in train_fe.columns if c not in drop_cols and c not in CAT_COLS
]
features = base_features + [f"{c}_te" for c in CAT_COLS]

monotone_positive = {
    "TyreLife",
    "Positive_LapTime_Delta",
    "CurrentTyreFinishExcess",
    "StopDebtAfterCurrent",
    "TyreLife_To_Remaining",
    "TyreLife_x_Dry",
    "PositiveDelta_x_Dry",
    "Excess_x_Dry",
}
monotone_constraints = tuple(1 if f in monotone_positive else 0 for f in features)


def add_fold_target_encoding(tr_part, va_part, te_part, tr_y, cols, smooth=30.0):
    tr_part = tr_part.copy()
    va_part = va_part.copy()
    te_part = te_part.copy()
    prior = float(np.mean(tr_y))
    tmp = tr_part.copy()
    tmp["_target_"] = tr_y

    for col in cols:
        stats = tmp.groupby(col)["_target_"].agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + prior * smooth) / (
            stats["count"] + smooth
        )
        tr_part[f"{col}_te"] = tr_part[col].map(enc).fillna(prior).astype(np.float32)
        va_part[f"{col}_te"] = va_part[col].map(enc).fillna(prior).astype(np.float32)
        te_part[f"{col}_te"] = te_part[col].map(enc).fillna(prior).astype(np.float32)

    return tr_part, va_part, te_part


groups = (train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)).values
if len(np.unique(groups)) >= 5:
    splitter = GroupKFold(n_splits=5)
    folds = splitter.split(train_fe, y, groups)
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=341)
    folds = splitter.split(train_fe, y)

oof = np.zeros(len(train_fe), dtype=np.float32)
test_pred = np.zeros(len(test_fe), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    tr_df, va_df, te_df = add_fold_target_encoding(
        train_fe.iloc[tr_idx],
        train_fe.iloc[va_idx],
        test_fe,
        y[tr_idx],
        CAT_COLS,
    )

    X_tr = tr_df[features].replace([np.inf, -np.inf], np.nan)
    X_va = va_df[features].replace([np.inf, -np.inf], np.nan)
    X_te = te_df[features].replace([np.inf, -np.inf], np.nan)

    scale_pos_weight = max(
        1.0, (len(tr_idx) - y[tr_idx].sum()) / max(1, y[tr_idx].sum())
    )

    model = XGBClassifier(
        n_estimators=900,
        max_depth=5,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=12,
        reg_lambda=8.0,
        reg_alpha=0.2,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_bin=256,
        random_state=341 + fold,
        n_jobs=max(1, os.cpu_count() or 1),
        scale_pos_weight=scale_pos_weight,
        monotone_constraints=monotone_constraints,
        early_stopping_rounds=80,
    )

    model.fit(X_tr, y[tr_idx], eval_set=[(X_va, y[va_idx])], verbose=False)
    oof[va_idx] = model.predict_proba(X_va)[:, 1]
    test_pred += model.predict_proba(X_te)[:, 1] / 5.0

    fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
    fold_scores.append(fold_auc)
    print(f"fold {fold} roc_auc: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"cv roc_auc: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: np.clip(test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": [float(x) for x in fold_scores],
            "research_hypotheses_llm_claimed_used": ["000341"],
        },
        f,
        indent=2,
    )
