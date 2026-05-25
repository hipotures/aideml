import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier, LGBMRanker

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
SEED = 404

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["is_test"] = 0
test["is_test"] = 1
test[TARGET] = np.nan
all_df = pd.concat([train, test], ignore_index=True)

all_df["Race_Year"] = all_df["Race"].astype(str) + "_" + all_df["Year"].astype(str)
snapshot = [all_df["Race_Year"], all_df["LapNumber"]]

for col in [
    "LapTime_Delta",
    "TyreLife",
    "LapTime (s)",
    "Cumulative_Degradation",
    "Position",
    "RaceProgress",
]:
    all_df[f"{col}_snap_rank_pct"] = all_df.groupby(snapshot)[col].rank(
        pct=True, method="average"
    )
    all_df[f"{col}_snap_z"] = (
        all_df[col] - all_df.groupby(snapshot)[col].transform("mean")
    ) / (all_df.groupby(snapshot)[col].transform("std").replace(0, np.nan))

all_df["gap_to_same_lap_median_pace"] = all_df["LapTime (s)"] - all_df.groupby(
    snapshot
)["LapTime (s)"].transform("median")
all_df["delta_gap_to_same_lap_median"] = all_df["LapTime_Delta"] - all_df.groupby(
    snapshot
)["LapTime_Delta"].transform("median")
all_df["tyrelife_gap_to_same_lap_median"] = all_df["TyreLife"] - all_df.groupby(
    snapshot
)["TyreLife"].transform("median")
all_df["compound_count_in_snapshot"] = all_df.groupby(
    ["Race_Year", "LapNumber", "Compound"]
)[ID_COL].transform("count")
all_df["snapshot_size"] = all_df.groupby(snapshot)[ID_COL].transform("count")
all_df["compound_rarity_in_snapshot"] = 1.0 - all_df[
    "compound_count_in_snapshot"
] / all_df["snapshot_size"].clip(lower=1)
all_df["driver_order_in_snapshot"] = (
    all_df.groupby(snapshot)["Driver"].rank(method="dense").astype(float)
)
all_df["driver_position_rank_in_snapshot"] = all_df.groupby(snapshot)["Position"].rank(
    pct=True, method="average"
)
all_df["stint_tyre_ratio"] = all_df["TyreLife"] / all_df["Stint"].clip(lower=1)
all_df["laps_remaining_est"] = (
    all_df["LapNumber"] / all_df["RaceProgress"].clip(lower=1e-4)
) - all_df["LapNumber"]

for col in CAT_COLS + ["Race_Year"]:
    all_df[col] = all_df[col].astype(str).fillna("missing")

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
all_df[CAT_COLS + ["Race_Year"]] = enc.fit_transform(all_df[CAT_COLS + ["Race_Year"]])

all_df = all_df.replace([np.inf, -np.inf], np.nan)
feature_cols = [c for c in all_df.columns if c not in [TARGET, ID_COL, "is_test"]]
all_df[feature_cols] = all_df[feature_cols].fillna(0)

trn = all_df[all_df["is_test"] == 0].copy()
tst = all_df[all_df["is_test"] == 1].copy()
y = trn[TARGET].astype(int).values
groups_block = train["Race"].astype(str) + "_" + train["Year"].astype(str)

rank_features = feature_cols
base_features = feature_cols + ["rank_oof_score"]

oof_rank = np.zeros(len(trn))
oof_cls = np.zeros(len(trn))
test_rank_folds = []
test_cls_folds = []
fold_scores = []

gkf = GroupKFold(n_splits=N_SPLITS)


def sort_for_rank(df, X, yvals=None):
    tmp = df[["Race_Year", "LapNumber"]].copy()
    tmp["_ord"] = np.arange(len(df))
    tmp = tmp.sort_values(["Race_Year", "LapNumber", "_ord"])
    idx = tmp["_ord"].values
    q = df.iloc[idx].groupby(["Race_Year", "LapNumber"], sort=False).size().values
    if yvals is None:
        return X.iloc[idx], q, idx
    return X.iloc[idx], yvals[idx], q, idx


for fold, (tr_idx, va_idx) in enumerate(gkf.split(trn, y, groups_block), 1):
    tr_fold = trn.iloc[tr_idx].copy()
    va_fold = trn.iloc[va_idx].copy()

    Xr_tr, yr_tr, q_tr, _ = sort_for_rank(tr_fold, tr_fold[rank_features], y[tr_idx])
    Xr_va, q_va, va_order = sort_for_rank(va_fold, va_fold[rank_features])

    ranker = LGBMRanker(
        objective="lambdarank",
        metric="auc",
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=35,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1,
    )
    ranker.fit(Xr_tr, yr_tr, group=q_tr)

    va_rank_sorted = ranker.predict(Xr_va)
    va_rank = np.zeros(len(va_idx))
    va_rank[va_order] = va_rank_sorted
    oof_rank[va_idx] = va_rank

    test_rank = ranker.predict(tst[rank_features])
    test_rank_folds.append(test_rank)

    tr_aug = tr_fold.copy()
    va_aug = va_fold.copy()
    tst_aug = tst.copy()
    tr_aug["rank_oof_score"] = ranker.predict(tr_aug[rank_features])
    va_aug["rank_oof_score"] = va_rank
    tst_aug["rank_oof_score"] = test_rank

    clf = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=95,
        max_depth=-1,
        min_child_samples=45,
        subsample=0.88,
        colsample_bytree=0.9,
        reg_lambda=3.0,
        random_state=SEED + 100 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(tr_aug[base_features], y[tr_idx])
    va_cls = clf.predict_proba(va_aug[base_features])[:, 1]
    oof_cls[va_idx] = va_cls
    test_cls_folds.append(clf.predict_proba(tst_aug[base_features])[:, 1])

    rank_auc = roc_auc_score(y[va_idx], va_rank)
    cls_auc = roc_auc_score(y[va_idx], va_cls)
    fold_scores.append(cls_auc)
    print(f"fold {fold}: rank_auc={rank_auc:.6f}, classifier_auc={cls_auc:.6f}")

rank_min, rank_max = np.min(oof_rank), np.max(oof_rank)
oof_rank_scaled = (oof_rank - rank_min) / (rank_max - rank_min + 1e-12)

best_w, best_auc = 0.0, -1.0
for w in np.linspace(0, 1, 51):
    blend = (1 - w) * oof_cls + w * oof_rank_scaled
    auc = roc_auc_score(y, blend)
    if auc > best_auc:
        best_auc, best_w = auc, w

test_cls = np.mean(test_cls_folds, axis=0)
test_rank = np.mean(test_rank_folds, axis=0)
test_rank_scaled = (test_rank - rank_min) / (rank_max - rank_min + 1e-12)
test_pred = np.clip((1 - best_w) * test_cls + best_w * test_rank_scaled, 0, 1)
oof_pred = np.clip((1 - best_w) * oof_cls + best_w * oof_rank_scaled, 0, 1)

print(f"mean_fold_classifier_auc={np.mean(fold_scores):.6f}")
print(f"oof_blended_auc={best_auc:.6f}")
print(f"best_rank_blend_weight={best_w:.3f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample[[ID_COL]].copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
test_pred_df.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(
        {
            "metric": "roc_auc",
            "oof_blended_auc": float(best_auc),
            "mean_fold_classifier_auc": float(np.mean(fold_scores)),
            "best_rank_blend_weight": float(best_w),
            "research_hypotheses_llm_claimed_used": ["000404"],
        },
        f,
        indent=2,
    )
