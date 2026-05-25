import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRanker

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values
test_ids = sample[ID_COL].values

train_features = train.drop(columns=[TARGET])
test_features = test.copy()
n_train = len(train_features)

all_df = pd.concat([train_features, test_features], axis=0, ignore_index=True)

cat_cols = ["Compound", "Driver", "Race"]
num_cols = [c for c in all_df.columns if c not in cat_cols + [ID_COL]]

lap_key = ["Year", "Race", "LapNumber"]
for col in [
    "TyreLife",
    "Cumulative_Degradation",
    "LapTime (s)",
    "LapTime_Delta",
    "Position",
    "RaceProgress",
    "Stint",
]:
    grp = all_df.groupby(lap_key, sort=False)[col]
    all_df[f"{col}_lap_mean"] = grp.transform("mean")
    all_df[f"{col}_lap_std"] = grp.transform("std").fillna(0)
    all_df[f"{col}_vs_lap_mean"] = all_df[col] - all_df[f"{col}_lap_mean"]
    all_df[f"{col}_lap_rank_pct"] = grp.rank(method="average", pct=True)

all_df["tyrelife_x_progress"] = all_df["TyreLife"] * all_df["RaceProgress"]
all_df["degradation_per_tyre_lap"] = all_df["Cumulative_Degradation"] / (
    all_df["TyreLife"] + 1.0
)
all_df["lap_delta_abs"] = all_df["LapTime_Delta"].abs()
all_df["is_wet_compound"] = all_df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)

for c in cat_cols:
    all_df[c] = all_df[c].astype("category")

feature_cols = [c for c in all_df.columns if c != ID_COL]
X_all = all_df[feature_cols]
X = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)

cat_feature_names = [c for c in cat_cols if c in X.columns]
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_cls = np.zeros(n_train)
oof_rank = np.zeros(n_train)
test_cls_folds = []
test_rank_folds = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    clf = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_feature_names,
        callbacks=[],
    )
    oof_cls[va_idx] = clf.predict_proba(X_va)[:, 1]
    test_cls_folds.append(clf.predict_proba(X_test)[:, 1])

    rank_train = X_tr.copy()
    rank_train["_target"] = y_tr
    rank_train["_orig_order"] = np.arange(len(rank_train))
    rank_train = rank_train.sort_values(lap_key + ["_orig_order"]).reset_index(
        drop=True
    )
    rank_groups = rank_train.groupby(lap_key, sort=False).size().values
    y_rank = rank_train["_target"].values
    X_rank = rank_train[feature_cols]

    ranker = LGBMRanker(
        objective="rank_xendcg",
        metric="ndcg",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE + 100 + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    ranker.fit(
        X_rank,
        y_rank,
        group=rank_groups,
        categorical_feature=cat_feature_names,
    )

    va_scores = ranker.predict(X_va)
    te_scores = ranker.predict(X_test)

    va_ranks = pd.Series(va_scores).rank(pct=True).values
    te_ranks = pd.Series(te_scores).rank(pct=True).values
    oof_rank[va_idx] = va_ranks
    test_rank_folds.append(te_ranks)

    fold_stack_auc = roc_auc_score(
        y_va, 0.65 * oof_cls[va_idx] + 0.35 * oof_rank[va_idx]
    )
    print(f"Fold {fold} simple blend ROC AUC: {fold_stack_auc:.6f}")

test_cls = np.mean(test_cls_folds, axis=0)
test_rank = np.mean(test_rank_folds, axis=0)

stack_X = np.column_stack([oof_cls, oof_rank])
stacker = LogisticRegression(
    C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
)
stacker.fit(stack_X, y)

oof_pred = stacker.predict_proba(stack_X)[:, 1]
test_pred = stacker.predict_proba(np.column_stack([test_cls, test_rank]))[:, 1]

auc = roc_auc_score(y, oof_pred)
print(f"5-fold OOF ROC AUC: {auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: np.clip(test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

with open(os.path.join(WORKING_DIR, "result_review.json"), "w") as f:
    json.dump(
        {
            "research_hypotheses_llm_claimed_used": ["001012"],
            "validation_metric": "roc_auc",
            "validation_score": float(auc),
            "cv": f"{N_SPLITS}-fold StratifiedKFold",
            "blend": "LogisticRegression stacker over LightGBM binary classifier and Year-Race-LapNumber LightGBM rank_xendcg specialist",
        },
        f,
        indent=2,
    )
