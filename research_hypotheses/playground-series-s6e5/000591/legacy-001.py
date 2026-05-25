import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, LGBMRanker

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]
QUERY_COLS = ["Year", "Race", "LapNumber"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values


def add_features(df):
    df = df.copy()
    df["TyreLife_x_Progress"] = df["TyreLife"] * df["RaceProgress"]
    df["TyreLife_frac_lap"] = df["TyreLife"] / (df["LapNumber"] + 1.0)
    df["Deg_per_TyreLife"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1.0)
    df["LapTime_per_progress"] = df["LapTime (s)"] / (df["RaceProgress"] + 0.01)
    df["LateRace"] = (df["RaceProgress"] > 0.70).astype(int)
    df["EarlyStint"] = (df["TyreLife"] <= 3).astype(int)
    df["WetOrInter"] = df["Compound"].isin(["WET", "INTERMEDIATE"]).astype(int)

    q = df.groupby(QUERY_COLS, sort=False)
    for col in [
        "TyreLife",
        "LapTime (s)",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "Position",
    ]:
        mean = q[col].transform("mean")
        std = q[col].transform("std").replace(0, np.nan)
        df[f"{col}_lap_mean"] = mean
        df[f"{col}_lap_diff"] = df[col] - mean
        df[f"{col}_lap_z"] = ((df[col] - mean) / std).fillna(0)

    df["lap_field_size"] = q[ID_COL].transform("count")
    return df


full = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
full = add_features(full)

for col in CAT_COLS:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col].astype(str))

train_x = full.iloc[: len(train)].copy()
test_x = full.iloc[len(train) :].copy()

drop_cols = [ID_COL]
feature_cols = [c for c in train_x.columns if c not in drop_cols]

race_groups = train_x["Year"].astype(str) + "_" + train_x["Race"].astype(str)
gkf = GroupKFold(n_splits=5)

oof_cls = np.zeros(len(train))
oof_rank = np.zeros(len(train))
test_cls = np.zeros(len(test))
test_rank = np.zeros(len(test))


def sorted_rank_data(x, labels=None):
    sort_cols = QUERY_COLS + [ID_COL]
    order = np.lexsort(tuple(x[c].values for c in reversed(sort_cols)))
    xs = x.iloc[order]
    group_sizes = xs.groupby(QUERY_COLS, sort=False).size().values
    if labels is None:
        return xs, group_sizes, order
    return xs, labels[order], group_sizes, order


for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_x, y, groups=race_groups), 1):
    x_tr, x_va = train_x.iloc[tr_idx], train_x.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=80,
        reg_alpha=0.05,
        reg_lambda=1.5,
        random_state=1000 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        x_tr[feature_cols],
        y_tr,
        eval_set=[(x_va[feature_cols], y_va)],
        eval_metric="auc",
        callbacks=[],
    )

    oof_cls[va_idx] = clf.predict_proba(x_va[feature_cols])[:, 1]
    test_cls += clf.predict_proba(test_x[feature_cols])[:, 1] / gkf.n_splits

    xr_tr, yr_tr, rank_groups, _ = sorted_rank_data(x_tr, y_tr)
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="auc",
        n_estimators=650,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        subsample=0.90,
        colsample_bytree=0.90,
        min_child_samples=40,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=2000 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    ranker.fit(xr_tr[feature_cols], yr_tr, group=rank_groups)

    va_rank_raw = ranker.predict(x_va[feature_cols])
    te_rank_raw = ranker.predict(test_x[feature_cols])

    va_rank_prob = pd.Series(va_rank_raw).rank(pct=True).values
    te_rank_prob = pd.Series(te_rank_raw).rank(pct=True).values

    oof_rank[va_idx] = va_rank_prob
    test_rank += te_rank_prob / gkf.n_splits

    fold_pred = 0.75 * oof_cls[va_idx] + 0.25 * oof_rank[va_idx]
    print(f"fold {fold} auc: {roc_auc_score(y_va, fold_pred):.6f}")

oof_pred = 0.75 * oof_cls + 0.25 * oof_rank
test_pred = np.clip(0.75 * test_cls + 0.25 * test_rank, 0, 1)

auc = roc_auc_score(y, oof_pred)
print(f"CV ROC AUC: {auc:.6f}")
print(
    json.dumps(
        {"research_hypotheses_llm_claimed_used": ["000591"], "cv_roc_auc": float(auc)}
    )
)

submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred})
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {"row": np.arange(len(train)), "target": y, "prediction": oof_pred}
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame({ID_COL: test_ids, TARGET: test_pred}).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
