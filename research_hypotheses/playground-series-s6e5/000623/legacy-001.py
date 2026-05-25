import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 2026


def clean_col(name):
    name = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return name or "col"


def sanitize_columns(df):
    mapping = {}
    used = set()
    for c in df.columns:
        base = clean_col(c)
        new = base
        k = 1
        while new in used:
            k += 1
            new = f"{base}_{k}"
        mapping[c] = new
        used.add(new)
    return df.rename(columns=mapping)


train = sanitize_columns(pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz")))
test = sanitize_columns(pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz")))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

if "LapTime_s" not in train.columns:
    raise RuntimeError(
        "Expected sanitized column LapTime_s from original 'LapTime (s)'."
    )

y = train[TARGET].astype(int).values


def add_features(df):
    df = df.copy()
    eps = 1e-6

    df["RankRaceKey"] = df["Year"].astype(str) + "__" + df["Race"].astype(str)
    df["RankRaceLapKey"] = df["RankRaceKey"] + "__lap_" + df["LapNumber"].astype(str)

    df["LapsRemainingEst"] = (
        df["LapNumber"] / df["RaceProgress"].clip(lower=eps)
    ) - df["LapNumber"]
    df["TyreLifeOverLap"] = df["TyreLife"] / df["LapNumber"].clip(lower=1)
    df["TyreLifeOverRaceProgress"] = df["TyreLife"] / df["RaceProgress"].clip(lower=eps)
    df["DegradationPerTyreLap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(
        lower=1
    )
    df["LapDeltaPerTyreLap"] = df["LapTime_Delta"] / df["TyreLife"].clip(lower=1)
    df["AbsPositionChange"] = df["Position_Change"].abs()
    df["IsWetCompound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    df["IsDrySoft"] = (df["Compound"] == "SOFT").astype(np.int8)
    df["IsFreshTyre"] = (df["TyreLife"] <= 2).astype(np.int8)
    df["LateRaceOldTyre"] = df["RaceProgress"] * df["TyreLife"]

    rank_cols = [
        "TyreLife",
        "Cumulative_Degradation",
        "LapTime_s",
        "LapTime_Delta",
        "Position",
        "Position_Change",
    ]

    g = df.groupby("RankRaceLapKey", sort=False)
    for col in rank_cols:
        df[f"{col}_RaceLapPctRank"] = (
            g[col].rank(method="average", pct=True).astype(np.float32)
        )
        mean = g[col].transform("mean")
        std = g[col].transform("std").replace(0, np.nan)
        df[f"{col}_RaceLapZ"] = ((df[col] - mean) / std).fillna(0).astype(np.float32)

    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)

    return df


train = add_features(train)
test = add_features(test)

drop_cols = {ID_COL, TARGET, "RankRaceKey", "RankRaceLapKey"}
features = [c for c in train.columns if c not in drop_cols]

cat_cols = [
    c
    for c in features
    if train[c].dtype == "object" or str(train[c].dtype).startswith("category")
]

for c in cat_cols:
    cats = pd.Index(pd.concat([train[c], test[c]], axis=0).astype(str).unique())
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

X = train[features]
X_test = test[features]

race_groups = train["RankRaceKey"].astype(str).values
rank_groups = train["RankRaceLapKey"].astype(str).values
test_rank_groups = test["RankRaceLapKey"].astype(str).values

if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y, race_groups))
else:
    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(X, y, race_groups))


def rank_normalize(scores, groups=None):
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    global_rank = s.rank(method="average", pct=True).values

    if groups is None:
        return global_rank.astype(np.float32)

    tmp = pd.DataFrame({"score": s.values, "group": np.asarray(groups)})
    local_rank = (
        tmp.groupby("group", sort=False)["score"]
        .rank(method="average", pct=True)
        .values
    )
    return (0.5 * global_rank + 0.5 * local_rank).astype(np.float32)


def make_rank_data(row_idx):
    local_groups = rank_groups[row_idx]
    order = np.argsort(local_groups, kind="mergesort")
    sorted_idx = row_idx[order]
    sorted_groups = local_groups[order]
    _, counts = np.unique(sorted_groups, return_counts=True)
    return X.iloc[sorted_idx], y[sorted_idx], counts


oof_clf = np.zeros(len(train), dtype=np.float32)
oof_rank = np.zeros(len(train), dtype=np.float32)
test_clf_preds = []
test_rank_preds = []
fold_rows = []

base_clf_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=31,
    max_depth=6,
    min_child_samples=160,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.15,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)

ranker_params = dict(
    objective="lambdarank",
    boosting_type="gbdt",
    n_estimators=900,
    learning_rate=0.04,
    num_leaves=31,
    max_depth=6,
    min_child_samples=120,
    subsample=0.90,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.10,
    reg_lambda=1.5,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
    metric="ndcg",
)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    pos = max(1, int(y_tr.sum()))
    neg = max(1, len(y_tr) - pos)

    clf_params = base_clf_params.copy()
    clf_params["random_state"] = SEED + fold
    clf_params["scale_pos_weight"] = neg / pos

    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    val_clf = clf.predict_proba(X_va)[:, 1]
    test_clf = clf.predict_proba(X_test)[:, 1]
    oof_clf[va_idx] = val_clf.astype(np.float32)
    test_clf_preds.append(test_clf.astype(np.float32))

    X_rank_tr, y_rank_tr, rank_tr_counts = make_rank_data(tr_idx)
    X_rank_va, y_rank_va, rank_va_counts = make_rank_data(va_idx)

    rparams = ranker_params.copy()
    rparams["random_state"] = SEED + 100 + fold

    ranker = lgb.LGBMRanker(**rparams)
    ranker.fit(
        X_rank_tr,
        y_rank_tr,
        group=rank_tr_counts,
        eval_set=[(X_rank_va, y_rank_va)],
        eval_group=[rank_va_counts],
        eval_at=[1, 3, 5, 10],
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(80, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    val_rank_raw = ranker.predict(X_va)
    test_rank_raw = ranker.predict(X_test)

    val_rank = rank_normalize(val_rank_raw, train["RankRaceLapKey"].iloc[va_idx].values)
    test_rank = rank_normalize(test_rank_raw, test_rank_groups)

    oof_rank[va_idx] = val_rank
    test_rank_preds.append(test_rank)

    fold_clf_auc = roc_auc_score(y_va, val_clf)
    fold_rank_auc = roc_auc_score(y_va, val_rank)
    fold_rows.append(
        {
            "fold": fold,
            "classifier_auc": float(fold_clf_auc),
            "ranker_auc": float(fold_rank_auc),
            "valid_rows": int(len(va_idx)),
            "valid_positive_rate": float(y_va.mean()),
        }
    )

clf_auc = roc_auc_score(y, oof_clf)
rank_auc = roc_auc_score(y, oof_rank)

candidate_weights = np.linspace(0.0, 1.0, 21)
blend_scores = []
for w in candidate_weights:
    pred = w * oof_clf + (1.0 - w) * oof_rank
    blend_scores.append(roc_auc_score(y, pred))

best_i = int(np.argmax(blend_scores))
best_clf_weight = float(candidate_weights[best_i])
best_auc = float(blend_scores[best_i])

test_clf_mean = np.mean(np.vstack(test_clf_preds), axis=0)
test_rank_mean = np.mean(np.vstack(test_rank_preds), axis=0)

oof_blend = best_clf_weight * oof_clf + (1.0 - best_clf_weight) * oof_rank
test_pred = best_clf_weight * test_clf_mean + (1.0 - best_clf_weight) * test_rank_mean
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y.astype(int),
        "prediction": oof_blend.astype(np.float32),
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

test_pred_df = sample.copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

result = {
    "research_hypotheses_llm_claimed_used": ["000623"],
    "metric": "roc_auc",
    "cv_roc_auc_blend": best_auc,
    "cv_roc_auc_classifier": float(clf_auc),
    "cv_roc_auc_ranker": float(rank_auc),
    "classifier_blend_weight": best_clf_weight,
    "ranker_blend_weight": float(1.0 - best_clf_weight),
    "folds": fold_rows,
    "saved_files": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
    ],
}

print(json.dumps(result, indent=2))
