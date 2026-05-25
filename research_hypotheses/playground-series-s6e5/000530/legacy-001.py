import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

warnings.filterwarnings("ignore")

SEED = 530
TARGET = "PitNextLap"
ID_COL = "id"
QUERY_COLS = ["Year", "Race", "LapNumber"]
INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)
N_JOBS = min(16, os.cpu_count() or 1)


def clean_columns(df):
    used, mapping = set(), {}
    for col in df.columns:
        new = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") or "col"
        base, k = new, 1
        while new in used:
            k += 1
            new = f"{base}_{k}"
        mapping[col] = new
        used.add(new)
    return df.rename(columns=mapping)


def add_base_features(df):
    df = df.copy()
    eps = 1e-3
    progress = df["RaceProgress"].clip(lower=eps)

    df["LapTime_Delta_abs"] = df["LapTime_Delta"].abs().astype("float32")
    df["Position_Change_abs"] = df["Position_Change"].abs().astype("float32")
    df["EstimatedRaceLaps"] = (
        (df["LapNumber"] / progress).clip(upper=120).astype("float32")
    )
    df["LapsRemaining"] = (
        (df["EstimatedRaceLaps"] - df["LapNumber"]).clip(lower=0).astype("float32")
    )
    df["Deg_per_TyreLife"] = (
        df["Cumulative_Degradation"] / (df["TyreLife"] + 1.0)
    ).astype("float32")
    df["TyreLife_x_Progress"] = (df["TyreLife"] * df["RaceProgress"]).astype("float32")
    df["Stint_TyreLife"] = (df["Stint"] * df["TyreLife"]).astype("float32")
    df["PostPitEarlyStint"] = ((df["Stint"] > 1) & (df["TyreLife"] <= 3)).astype("int8")
    return df


def add_within_query_features(df):
    df = df.copy()
    g = df.groupby(QUERY_COLS, sort=False, observed=True)
    df["CarsInSnapshot"] = g[ID_COL].transform("size").astype("int16")

    rel_cols = [
        "TyreLife",
        "Cumulative_Degradation",
        "LapTime_s",
        "LapTime_Delta",
        "Position",
        "Position_Change",
        "RaceProgress",
    ]
    for col in rel_cols:
        if col not in df.columns:
            continue
        gc = g[col]
        mean = gc.transform("mean")
        median = gc.transform("median")
        std = gc.transform("std").replace(0, np.nan)
        df[f"{col}_SnapshotDev"] = (df[col] - median).astype("float32")
        df[f"{col}_SnapshotZ"] = ((df[col] - mean) / std).fillna(0).astype("float32")
        df[f"{col}_SnapshotRank"] = gc.rank(method="average", pct=True).astype(
            "float32"
        )
    return df


def make_group_key(df):
    return (
        df["Year"].astype(str)
        + "|"
        + df["Race"].astype(str)
        + "|"
        + df["LapNumber"].astype(str)
    )


def make_ranker_frame(df_part, y_all, feature_cols):
    ordered = df_part.sort_values(QUERY_COLS, kind="mergesort")
    group_sizes = (
        ordered.groupby(QUERY_COLS, sort=False, observed=True).size().to_numpy()
    )
    return ordered[feature_cols], y_all.loc[ordered.index].astype(int), group_sizes


def add_sidecar_features(df_part, raw_scores, feature_cols):
    out = df_part[feature_cols].copy()
    tmp = pd.DataFrame(
        {"RankerScore": np.asarray(raw_scores, dtype=np.float32)}, index=df_part.index
    )
    for col in QUERY_COLS:
        tmp[col] = df_part[col].values

    grp = tmp.groupby(QUERY_COLS, sort=False, observed=True)["RankerScore"]
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0, np.nan)

    out["RankerScore"] = tmp["RankerScore"].astype("float32")
    out["RankerSnapshotPct"] = (
        grp.rank(method="average", pct=True).astype("float32").to_numpy()
    )
    out["RankerSnapshotZ"] = (
        ((tmp["RankerScore"] - mean) / std).fillna(0).astype("float32").to_numpy()
    )
    return out


train = clean_columns(pd.read_csv(INPUT_DIR / "train.csv.gz"))
test = clean_columns(pd.read_csv(INPUT_DIR / "test.csv.gz"))
sample = clean_columns(pd.read_csv(INPUT_DIR / "sample_submission.csv.gz"))

train = add_within_query_features(add_base_features(train))
test = add_within_query_features(add_base_features(test))

cat_cols = [c for c in ["Compound", "Driver", "Race"] if c in train.columns]
for col in cat_cols:
    cats = (
        pd.concat([train[col], test[col]], ignore_index=True)
        .astype("category")
        .cat.categories
    )
    train[col] = pd.Categorical(train[col], categories=cats)
    test[col] = pd.Categorical(test[col], categories=cats)

feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
missing = [c for c in feature_cols if c not in test.columns]
if missing:
    raise ValueError(f"Missing test feature columns: {missing}")

y = train[TARGET].astype(int)
groups = make_group_key(train)

ranker_params = dict(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=320,
    learning_rate=0.045,
    num_leaves=31,
    min_child_samples=25,
    subsample=0.90,
    subsample_freq=1,
    colsample_bytree=0.90,
    reg_alpha=0.05,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=N_JOBS,
    verbosity=-1,
)

clf_params = dict(
    objective="binary",
    metric="auc",
    n_estimators=650,
    learning_rate=0.035,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=N_JOBS,
    verbosity=-1,
)

if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
else:
    splitter = GroupKFold(n_splits=5)

oof = np.zeros(len(train), dtype=np.float32)
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(splitter.split(train, y, groups), 1):
    tr_df = train.iloc[tr_idx]
    va_df = train.iloc[va_idx]
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    X_rank, y_rank, rank_groups = make_ranker_frame(tr_df, y, feature_cols)
    ranker = lgb.LGBMRanker(**{**ranker_params, "random_state": SEED + fold})
    ranker.fit(X_rank, y_rank, group=rank_groups, categorical_feature=cat_cols)

    tr_rank = ranker.predict(tr_df[feature_cols])
    va_rank = ranker.predict(va_df[feature_cols])

    X_tr = add_sidecar_features(tr_df, tr_rank, feature_cols)
    X_va = add_sidecar_features(va_df, va_rank, feature_cols)

    clf = lgb.LGBMClassifier(**{**clf_params, "random_state": SEED + 100 + fold})
    clf.fit(X_tr, y_tr, categorical_feature=cat_cols)

    va_pred = clf.predict_proba(X_va)[:, 1]
    oof[va_idx] = va_pred.astype(np.float32)

    auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(float(auc))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = float(roc_auc_score(y, oof))
print(f"OOF ROC AUC: {cv_auc:.6f}")
print(f"Mean fold ROC AUC: {np.mean(fold_aucs):.6f} +/- {np.std(fold_aucs):.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y.to_numpy(),
        "prediction": oof,
    }
).to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

X_rank_full, y_rank_full, full_groups = make_ranker_frame(train, y, feature_cols)
final_ranker = lgb.LGBMRanker(**ranker_params)
final_ranker.fit(
    X_rank_full, y_rank_full, group=full_groups, categorical_feature=cat_cols
)

train_rank_full = final_ranker.predict(train[feature_cols])
test_rank_full = final_ranker.predict(test[feature_cols])

X_train_full = add_sidecar_features(train, train_rank_full, feature_cols)
X_test_full = add_sidecar_features(test, test_rank_full, feature_cols)

final_clf = lgb.LGBMClassifier(**clf_params)
final_clf.fit(X_train_full, y, categorical_feature=cat_cols)

test_pred = np.clip(final_clf.predict_proba(X_test_full)[:, 1], 0, 1)

submission = sample[[ID_COL, TARGET]].copy()
submission[TARGET] = test_pred
submission.to_csv(WORK_DIR / "submission.csv", index=False)
submission.to_csv(WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip")

result = {
    "metric": "roc_auc",
    "oof_roc_auc": cv_auc,
    "fold_roc_auc": fold_aucs,
    "research_hypotheses_llm_claimed_used": ["000530"],
    "submission_path": str(WORK_DIR / "submission.csv"),
    "oof_predictions_path": str(WORK_DIR / "oof_predictions.csv.gz"),
    "test_predictions_path": str(WORK_DIR / "test_predictions.csv.gz"),
}
print(json.dumps(result, indent=2))
