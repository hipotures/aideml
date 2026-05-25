import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
GROUP_COLS = ["Year", "Race", "Driver", "Stint"]
CAT_COLS = ["Compound", "Driver", "Race"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_hazard_labels(df):
    df = df.copy()
    df["_orig_order"] = np.arange(len(df))
    df = df.sort_values(GROUP_COLS + ["LapNumber", ID_COL]).reset_index(drop=True)

    time_to_next = np.full(len(df), np.nan, dtype=np.float32)
    uncensored = np.zeros(len(df), dtype=np.int8)

    for _, idx in df.groupby(GROUP_COLS, sort=False).groups.items():
        idx = np.asarray(idx)
        pit_flags = df.loc[idx, "PitStop"].to_numpy()
        laps = df.loc[idx, "LapNumber"].to_numpy(dtype=np.float32)
        pit_pos = np.flatnonzero(pit_flags == 1)

        if len(pit_pos) == 0:
            continue

        for local_i in range(len(idx)):
            future = pit_pos[pit_pos > local_i]
            if len(future):
                t = laps[future[0]] - laps[local_i]
                if t >= 1:
                    time_to_next[idx[local_i]] = t
                    uncensored[idx[local_i]] = 1

    df["time_to_next_pit"] = time_to_next
    df["event_observed"] = uncensored
    df = (
        df.sort_values("_orig_order").drop(columns="_orig_order").reset_index(drop=True)
    )
    return df


train = add_hazard_labels(train)

uncensored_times = train.loc[train["event_observed"].eq(1), "time_to_next_pit"].dropna()
if len(uncensored_times) >= 100:
    q = np.quantile(uncensored_times, [0.20, 0.40, 0.60, 0.80])
    bin_edges = np.unique(np.rint(q).astype(int))
    bin_edges = bin_edges[(bin_edges >= 1) & (bin_edges <= int(uncensored_times.max()))]
else:
    bin_edges = np.array([1, 2, 4, 8], dtype=int)

if len(bin_edges) < 2 or 1 not in bin_edges:
    bin_edges = np.unique(np.r_[1, bin_edges, [2, 4, 8]])


def make_hazard_class(df):
    cls = np.full(
        len(df), len(bin_edges), dtype=np.int32
    )  # final class = censored/no observed future pit
    observed = df["event_observed"].eq(1).to_numpy()
    t = df["time_to_next_pit"].fillna(10**6).to_numpy()
    cls[observed] = np.searchsorted(bin_edges, t[observed], side="left")
    return cls


train["hazard_class"] = make_hazard_class(train)


def add_features(df):
    df = df.copy()
    df["lap_frac_tyre"] = df["LapNumber"] / (df["TyreLife"] + 1.0)
    df["degradation_per_lap"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1.0)
    df["stint_progress"] = df["TyreLife"] / (df["LapNumber"] + 1.0)
    df["race_lap_remaining"] = (
        (1.0 - df["RaceProgress"])
        * df["LapNumber"]
        / np.maximum(df["RaceProgress"], 0.01)
    )
    df["is_wet_compound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    df["pitstop_x_tyre"] = df["PitStop"] * df["TyreLife"]
    return df


train = add_features(train)
test = add_features(test)

feature_cols = [c for c in test.columns if c != ID_COL]
all_data = pd.concat(
    [train[feature_cols], test[feature_cols]], axis=0, ignore_index=True
)

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
all_data[CAT_COLS] = encoder.fit_transform(all_data[CAT_COLS].astype(str)).astype(
    np.int32
)

for c in feature_cols:
    if all_data[c].dtype == "object":
        all_data[c] = pd.to_numeric(all_data[c], errors="coerce")
    if all_data[c].isna().any():
        all_data[c] = all_data[c].fillna(all_data[c].median())

X = all_data.iloc[: len(train)].reset_index(drop=True)
X_test = all_data.iloc[len(train) :].reset_index(drop=True)
y = train[TARGET].astype(int).to_numpy()
hazard_y = train["hazard_class"].to_numpy()
groups = (train["Year"].astype(str) + "_" + train["Race"].astype(str)).to_numpy()

try:
    from lightgbm import LGBMClassifier
except Exception as e:
    raise RuntimeError("lightgbm is required for this solution") from e

n_classes = int(hazard_y.max() + 1)
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(train), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
    model = LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.5,
        random_state=2026 + fold,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
        class_weight="balanced",
    )
    model.fit(
        X.iloc[tr_idx],
        hazard_y[tr_idx],
        eval_set=[(X.iloc[va_idx], hazard_y[va_idx])],
        eval_metric="multi_logloss",
        callbacks=[],
    )

    va_proba = model.predict_proba(X.iloc[va_idx])
    te_proba = model.predict_proba(X_test)

    first_bin = va_proba[:, 0]
    test_first_bin = te_proba[:, 0]

    oof[va_idx] = first_bin
    test_pred += test_first_bin / gkf.n_splits

    fold_auc = roc_auc_score(y[va_idx], first_bin)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

submission = sample[[ID_COL]].copy()
submission[TARGET] = np.clip(test_pred, 0.0, 1.0)
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

test_predictions = sample[[ID_COL]].copy()
test_predictions[TARGET] = np.clip(test_pred, 0.0, 1.0)
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000148"],
    "metric": "roc_auc",
    "oof_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(x) for x in fold_scores],
    "hazard_bin_edges": [int(x) for x in bin_edges],
}
with open(os.path.join(WORK_DIR, "review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review))
