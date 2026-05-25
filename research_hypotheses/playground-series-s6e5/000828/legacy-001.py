import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = [
    "Driver",
    "Race",
    "Compound",
    "Race_Year",
    "Driver_Compound",
    "Driver_Race",
    "Race_Compound",
]


def add_hierarchical_cats(df):
    df = df.copy()
    for c in ["Driver", "Race", "Compound"]:
        df[c] = df[c].astype("string").fillna("NA").astype(str)

    year = df["Year"].astype(str)
    df["Race_Year"] = df["Race"] + "_" + year
    df["Driver_Compound"] = df["Driver"] + "_" + df["Compound"]
    df["Driver_Race"] = df["Driver"] + "_" + df["Race"]
    df["Race_Compound"] = df["Race"] + "_" + df["Compound"]

    for c in CAT_COLS:
        df[c] = df[c].astype("string").fillna("NA").astype(str)
    return df


def make_chrono_embargo_folds(df, group_col="Race_Year", n_splits=5, embargo=1):
    seg = (
        df.reset_index()
        .groupby(group_col, sort=False)
        .agg(first_row=("index", "min"), year=("Year", "min"))
        .reset_index()
        .sort_values(["year", "first_row"])
        .reset_index(drop=True)
    )
    groups = seg[group_col].to_numpy()
    n_groups = len(groups)
    n_splits = min(n_splits, n_groups)
    bounds = np.linspace(0, n_groups, n_splits + 1, dtype=int)

    folds = []
    for i in range(n_splits):
        start, end = bounds[i], bounds[i + 1]
        emb_start = max(0, start - embargo)
        emb_end = min(n_groups, end + embargo)

        valid_groups = set(groups[start:end])
        blocked_groups = set(groups[emb_start:emb_end])
        train_groups = set(groups) - blocked_groups

        valid_mask = df[group_col].isin(valid_groups).to_numpy()
        train_mask = df[group_col].isin(train_groups).to_numpy()
        folds.append(
            (np.flatnonzero(train_mask), np.flatnonzero(valid_mask), len(valid_groups))
        )
    return folds


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["row"] = np.arange(len(train))
train = train.sort_values(ID_COL).reset_index(drop=True)

test = test.set_index(ID_COL).loc[sample[ID_COL]].reset_index()

train = add_hierarchical_cats(train)
test = add_hierarchical_cats(test)

y = train[TARGET].astype(int)
features = [c for c in train.columns if c not in {TARGET, ID_COL, "row"}]

folds = make_chrono_embargo_folds(train, group_col="Race_Year", n_splits=5, embargo=1)

oof = np.full(len(train), np.nan, dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

test_pool = Pool(test[features], cat_features=CAT_COLS)
threads = max(1, os.cpu_count() or 1)

for fold, (tr_idx, va_idx, n_valid_groups) in enumerate(folds, 1):
    train_pool = Pool(
        train.iloc[tr_idx][features],
        y.iloc[tr_idx],
        cat_features=CAT_COLS,
    )
    valid_pool = Pool(
        train.iloc[va_idx][features],
        y.iloc[va_idx],
        cat_features=CAT_COLS,
    )

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        learning_rate=0.055,
        depth=6,
        l2_leaf_reg=8.0,
        random_strength=0.8,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        boosting_type="Ordered",
        has_time=True,
        max_ctr_complexity=2,
        od_type="Iter",
        od_wait=80,
        random_seed=2026 + fold,
        thread_count=threads,
        allow_writing_files=False,
        verbose=200,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    val_pred = model.predict_proba(valid_pool)[:, 1]
    oof[va_idx] = val_pred
    test_pred += model.predict_proba(test_pool)[:, 1] / len(folds)

    fold_auc = roc_auc_score(y.iloc[va_idx], val_pred)
    fold_scores.append(float(fold_auc))
    print(
        f"fold={fold} valid_rows={len(va_idx)} valid_groups={n_valid_groups} auc={fold_auc:.6f}"
    )

valid_oof = ~np.isnan(oof)
cv_auc = roc_auc_score(y.iloc[valid_oof], oof[valid_oof])
print(f"OOF ROC AUC: {cv_auc:.6f}")

oof_df = pd.DataFrame(
    {
        "row": train.loc[valid_oof, "row"].to_numpy(),
        "target": y.iloc[valid_oof].to_numpy(),
        "prediction": oof[valid_oof],
    }
).sort_values("row")
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

target_col = sample.columns[1]
submission = sample.copy()
submission[target_col] = np.clip(test_pred, 0.0, 1.0)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": fold_scores,
            "research_hypotheses_llm_claimed_used": ["000828"],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
            "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
            "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
        }
    )
)
