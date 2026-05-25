import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

warnings.filterwarnings("ignore", category=FutureWarning)

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

ID_COL = "id"
TARGET = "PitNextLap"
RANDOM_STATE = 2026
RENAMES = {"LapTime (s)": "LapTime_s"}


def safe_name(col):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in col).strip("_")


def add_nearby_position_features(df, keys, window=2):
    df = df.reset_index(drop=True)
    n = len(df)
    pos = df["Position"].to_numpy()
    tyre = df["TyreLife"].to_numpy(dtype=np.float32)
    large_delta = df["ctx_laptime_delta_z"].to_numpy(dtype=np.float32) > 1.0

    total = np.zeros(n, dtype=np.float32)
    older = np.zeros(n, dtype=np.float32)
    large = np.zeros(n, dtype=np.float32)
    older_or_large = np.zeros(n, dtype=np.float32)
    older_ahead = np.zeros(n, dtype=np.float32)
    older_behind = np.zeros(n, dtype=np.float32)

    for idx in df.groupby(keys, sort=False).indices.values():
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size <= 1:
            continue
        order = idx[np.argsort(pos[idx], kind="mergesort")]
        m = len(order)
        for j, row in enumerate(order):
            lo, hi = max(0, j - window), min(m, j + window + 1)
            neigh = np.concatenate((order[lo:j], order[j + 1 : hi]))
            if neigh.size == 0:
                continue

            old_mask = tyre[neigh] > tyre[row]
            large_mask = large_delta[neigh]
            total[row] = neigh.size
            older[row] = old_mask.sum()
            large[row] = large_mask.sum()
            older_or_large[row] = np.logical_or(old_mask, large_mask).sum()

            ahead = order[lo:j]
            behind = order[j + 1 : hi]
            if ahead.size:
                older_ahead[row] = (tyre[ahead] > tyre[row]).sum()
            if behind.size:
                older_behind[row] = (tyre[behind] > tyre[row]).sum()

    denom = np.maximum(total, 1.0)
    df["ctx_nearby_position_count"] = total
    df["ctx_nearby_older_tyre_count"] = older
    df["ctx_nearby_large_delta_count"] = large
    df["ctx_nearby_older_or_large_delta_count"] = older_or_large
    df["ctx_nearby_older_tyre_frac"] = older / denom
    df["ctx_nearby_large_delta_frac"] = large / denom
    df["ctx_older_tyre_ahead_count"] = older_ahead
    df["ctx_older_tyre_behind_count"] = older_behind
    return df


def add_context_features(df):
    df = df.copy()
    keys = ["Year", "Race", "LapNumber"]
    g = df.groupby(keys, sort=False)

    df["ctx_field_count"] = g[ID_COL].transform("count").astype("float32")

    for col in [
        "TyreLife",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "LapTime_s",
        "Position_Change",
    ]:
        if col not in df.columns:
            continue
        base = safe_name(col)
        mean = g[col].transform("mean")
        std = g[col].transform("std").replace(0, np.nan)
        z = (
            ((df[col] - mean) / std)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(-10, 10)
        )
        df[f"ctx_{base}_z"] = z.astype("float32")
        df[f"ctx_{base}_mean"] = mean.astype("float32")
        df[f"ctx_{base}_spread"] = (
            g[col].transform("max") - g[col].transform("min")
        ).astype("float32")

    df["ctx_tyre_life_pct"] = g["TyreLife"].rank(pct=True).astype("float32")
    df["ctx_lap_delta_pct"] = g["LapTime_Delta"].rank(pct=True).astype("float32")
    df["ctx_degradation_rank_pct"] = (
        g["Cumulative_Degradation"].rank(pct=True).astype("float32")
    )
    df["ctx_position_pct"] = g["Position"].rank(pct=True).astype("float32")

    comp = (
        df.groupby(keys + ["Compound"], sort=False)
        .size()
        .rename("compound_count")
        .reset_index()
    )
    comp["snapshot_total"] = comp.groupby(keys, sort=False)["compound_count"].transform(
        "sum"
    )
    comp["ctx_compound_share"] = (
        comp["compound_count"] / comp["snapshot_total"]
    ).astype("float32")
    p = comp["ctx_compound_share"].clip(1e-12, 1.0)
    comp["entropy_piece"] = -p * np.log(p)
    entropy = (
        comp.groupby(keys, sort=False)["entropy_piece"]
        .sum()
        .rename("ctx_compound_entropy")
        .reset_index()
    )
    nunique = (
        comp.groupby(keys, sort=False)["Compound"]
        .size()
        .rename("ctx_compound_nunique")
        .reset_index()
    )

    df = df.merge(entropy, on=keys, how="left")
    df = df.merge(nunique, on=keys, how="left")
    df = df.merge(
        comp[keys + ["Compound", "ctx_compound_share"]],
        on=keys + ["Compound"],
        how="left",
    )
    df["ctx_compound_rarity"] = 1.0 - df["ctx_compound_share"]

    stint = (
        df.groupby(keys + ["Stint"], sort=False)
        .size()
        .rename("stint_count")
        .reset_index()
    )
    stint = stint.sort_values(
        keys + ["Stint"], ascending=[True, True, True, False]
    ).reset_index(drop=True)
    stint["snapshot_total"] = stint.groupby(keys, sort=False)["stint_count"].transform(
        "sum"
    )
    stint["later_count"] = (
        stint.groupby(keys, sort=False)["stint_count"].cumsum() - stint["stint_count"]
    )
    stint["ctx_later_stint_frac"] = (
        stint["later_count"] / stint["snapshot_total"]
    ).astype("float32")
    stint["ctx_same_stint_frac"] = (
        stint["stint_count"] / stint["snapshot_total"]
    ).astype("float32")
    df = df.merge(
        stint[keys + ["Stint", "ctx_later_stint_frac", "ctx_same_stint_frac"]],
        on=keys + ["Stint"],
        how="left",
    )

    by_snapshot = [df[k] for k in keys]
    slow_flag = (df["ctx_laptime_delta_z"] > 1.0).astype("float32")
    df["ctx_field_large_delta_frac"] = (
        slow_flag.groupby(by_snapshot, sort=False).transform("mean").astype("float32")
    )
    df["ctx_field_current_pit_frac"] = (
        df["PitStop"]
        .astype("float32")
        .groupby(by_snapshot, sort=False)
        .transform("mean")
        .astype("float32")
    )

    df = add_nearby_position_features(df, keys, window=2)

    for col in [c for c in df.columns if c.startswith("ctx_")]:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype("float32")
        )
    return df


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz")).rename(columns=RENAMES)
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz")).rename(columns=RENAMES)
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
combined = pd.concat(
    [train.drop(columns=[TARGET]), test], ignore_index=True, sort=False
)
combined = add_context_features(combined)

cat_cols = [c for c in ["Compound", "Driver", "Race"] if c in combined.columns]
for col in cat_cols:
    combined[col] = combined[col].astype("category")

for col in combined.columns:
    if col == ID_COL or col in cat_cols:
        continue
    if pd.api.types.is_float_dtype(combined[col]):
        combined[col] = combined[col].astype("float32")

X = combined.iloc[: len(train)].drop(columns=[ID_COL])
X_test = combined.iloc[len(train) :].drop(columns=[ID_COL])

groups = train["Year"].astype(str) + "|" + train["Race"].astype(str)
if StratifiedGroupKFold is not None:
    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X, y, groups))
        if any(len(np.unique(y[val_idx])) < 2 for _, val_idx in splits):
            raise ValueError("A validation fold has only one class.")
        cv_name = "StratifiedGroupKFold(Year,Race)"
    except Exception:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = list(cv.split(X, y))
        cv_name = "StratifiedKFold"
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X, y))
    cv_name = "StratifiedKFold"

pos = max(float(y.sum()), 1.0)
neg = float(len(y) - y.sum())
params = dict(
    objective="binary",
    n_estimators=1600,
    learning_rate=0.04,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    scale_pos_weight=max(1.0, neg / pos),
    random_state=RANDOM_STATE,
    n_jobs=max(1, min(8, os.cpu_count() or 1)),
    verbosity=-1,
)

print(f"Validation scheme: {cv_name}")
oof = np.zeros(len(X), dtype=np.float32)
fold_aucs = []
best_iterations = []

for fold, (tr_idx, val_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[val_idx], y[val_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(100, first_metric_only=True, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    pred = model.predict_proba(X.iloc[val_idx], num_iteration=model.best_iteration_)[
        :, 1
    ]
    oof[val_idx] = pred.astype(np.float32)
    auc = roc_auc_score(y[val_idx], pred)
    fold_aucs.append(float(auc))
    best_iterations.append(int(model.best_iteration_ or params["n_estimators"]))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y.astype(int),
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_estimators = int(np.clip(np.median(best_iterations), 100, params["n_estimators"]))
final_params = params.copy()
final_params["n_estimators"] = final_estimators
final_params["random_state"] = RANDOM_STATE + 1

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_cols)
test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

sample_target = [c for c in sample.columns if c != ID_COL][0]
submission = sample[[ID_COL]].copy()
if sample[ID_COL].equals(test[ID_COL]):
    submission[sample_target] = test_pred
else:
    pred_map = pd.Series(test_pred, index=test[ID_COL].to_numpy())
    submission[sample_target] = (
        sample[ID_COL].map(pred_map).fillna(float(np.mean(test_pred))).to_numpy()
    )

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000696"],
    "metric": "roc_auc",
    "validation_scheme": cv_name,
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_aucs,
    "final_n_estimators": final_estimators,
}
for name in ["result.json", "review.json"]:
    with open(os.path.join(WORK_DIR, name), "w") as f:
        json.dump(review, f, indent=2)

print(f"Final model trees: {final_estimators}")
print(f"Saved submission to {os.path.join(WORK_DIR, 'submission.csv')}")
