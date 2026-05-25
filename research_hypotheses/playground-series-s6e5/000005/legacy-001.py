import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SLICKS = ("SOFT", "MEDIUM", "HARD")

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_race_year(df):
    df = df.copy()
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    return df


def remaining_laps(df):
    lap = df["LapNumber"].astype(float)
    progress = df["RaceProgress"].astype(float).clip(lower=1e-5)
    total_laps = (lap / progress).clip(lower=lap, upper=120)
    return (total_laps - lap).clip(lower=0, upper=120)


train = add_race_year(train)
test = add_race_year(test)

finish_threshold = train.groupby("Compound")["TyreLife"].quantile(0.95).to_dict()
default_finish_threshold = float(train["TyreLife"].quantile(0.95))


def add_stop_debt_block(df):
    df = df.copy()
    df["EstimatedRemainingLaps"] = remaining_laps(df)

    if "PitStop" not in df.columns:
        return df

    work = df[
        [
            "Race_Year",
            "Driver",
            "LapNumber",
            ID_COL,
            "PitStop",
            "Compound",
            "TyreLife",
            "EstimatedRemainingLaps",
        ]
    ].copy()
    work["_pos"] = np.arange(len(work))
    work = work.sort_values(
        ["Race_Year", "Driver", "LapNumber", ID_COL], kind="mergesort"
    )

    keys = [work["Race_Year"], work["Driver"]]
    pit = work["PitStop"].fillna(0).astype(int).clip(0, 1)
    prior_pits = pit.groupby(keys, sort=False).cumsum() - pit

    compound = work["Compound"].astype(str).str.upper()
    prior_div = np.zeros(len(work), dtype=np.int16)
    current_div = np.zeros(len(work), dtype=np.int16)

    for comp in SLICKS:
        flag = (compound == comp).astype(np.int16)
        seen_count = flag.groupby(keys, sort=False).cumsum()
        current_div += (seen_count > 0).astype(np.int16)
        prior_div += ((seen_count - flag) > 0).astype(np.int16)

    current_is_slick = compound.isin(SLICKS).to_numpy()
    strategic_debt_before = (prior_pits.to_numpy() < 1).astype(np.int16)
    strategic_debt_after = ((prior_pits.to_numpy() + pit.to_numpy()) < 1).astype(
        np.int16
    )
    regulatory_debt_before = (current_is_slick & (prior_div < 2)).astype(np.int16)
    regulatory_debt_after = (current_is_slick & (current_div < 2)).astype(np.int16)

    debt_before = np.maximum(strategic_debt_before, regulatory_debt_before)
    debt_after = np.maximum(strategic_debt_after, regulatory_debt_after)
    rem = work["EstimatedRemainingLaps"].astype(float).to_numpy()

    threshold = (
        compound.map(finish_threshold)
        .fillna(default_finish_threshold)
        .astype(float)
        .to_numpy()
    )
    can_finish = (work["TyreLife"].astype(float).to_numpy() + rem) <= threshold

    work["PriorPitStops"] = prior_pits.to_numpy()
    work["CurrentPitStop"] = pit.to_numpy()
    work["PriorSlickCompoundDiversity"] = prior_div
    work["CurrentSlickCompoundDiversity"] = current_div
    work["StopDebtBeforeCurrentLap"] = debt_before
    work["StopDebtAfterCurrentLap"] = debt_after
    work["LapsPerRemainingStopDebt"] = np.where(
        debt_after > 0, rem / debt_after, rem + 1.0
    )
    work["CanFinishButStillOwesStop"] = (can_finish & (debt_after > 0)).astype(np.int16)
    work["CurrentPitRelievesStopDebt"] = (
        (pit.to_numpy() == 1) & (debt_before > debt_after)
    ).astype(np.int16)

    new_cols = [
        "PriorPitStops",
        "CurrentPitStop",
        "PriorSlickCompoundDiversity",
        "CurrentSlickCompoundDiversity",
        "StopDebtBeforeCurrentLap",
        "StopDebtAfterCurrentLap",
        "LapsPerRemainingStopDebt",
        "CanFinishButStillOwesStop",
        "CurrentPitRelievesStopDebt",
    ]
    work = work.sort_values("_pos")
    for col in new_cols:
        df[col] = work[col].to_numpy()

    return df


train = add_stop_debt_block(train)
test = add_stop_debt_block(test)

feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
cat_cols = [
    c
    for c in feature_cols
    if train[c].dtype == "object" or str(train[c].dtype).startswith("category")
]

for c in cat_cols:
    all_values = pd.concat([train[c], test[c]], axis=0).astype(str).fillna("__NA__")
    categories = pd.Index(all_values.unique())
    train[c] = pd.Categorical(
        train[c].astype(str).fillna("__NA__"), categories=categories
    )
    test[c] = pd.Categorical(
        test[c].astype(str).fillna("__NA__"), categories=categories
    )


def safe_names(cols):
    used = {}
    mapping = {}
    for col in cols:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", col).strip("_")
        if not base:
            base = "feature"
        name = base
        k = 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used[name] = True
        mapping[col] = name
    return mapping


name_map = safe_names(feature_cols)
X = train[feature_cols].rename(columns=name_map)
X_test = test[feature_cols].rename(columns=name_map)
cat_cols_safe = [name_map[c] for c in cat_cols]
y = train[TARGET].astype(int).to_numpy()
groups = train["Race_Year"].astype(str).to_numpy()

oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

cv = GroupKFold(n_splits=5)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups), start=1):
    model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=2500,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=20260524 + fold,
        n_jobs=min(8, os.cpu_count() or 1),
        verbosity=-1,
    )

    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols_safe,
        callbacks=[early_stopping(100), log_evaluation(0)],
    )

    val_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = val_pred
    fold_auc = roc_auc_score(y[va_idx], val_pred)
    fold_scores.append(fold_auc)

    test_pred += model.predict_proba(X_test)[:, 1] / cv.get_n_splits()

cv_auc = roc_auc_score(y, oof)
print(f"Fold AUCs: {[round(x, 6) for x in fold_scores]}")
print(f"Mean fold AUC: {np.mean(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame({"row": np.arange(len(train)), "target": y, "prediction": oof}).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000005"],
            "metric": "roc_auc",
            "oof_roc_auc": float(cv_auc),
            "mean_fold_auc": float(np.mean(fold_scores)),
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        }
    )
)
