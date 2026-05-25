import os
import re
import gc
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def safe_name(col):
    if col in [ID_COL, TARGET]:
        return col
    return re.sub(r"[^A-Za-z0-9_]+", "_", col).strip("_")


def add_features(df):
    df = df.copy()
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["Race_Year_Driver"] = df["Race_Year"] + "_" + df["Driver"].astype(str)
    df["Race_Year_Driver_Stint"] = (
        df["Race_Year_Driver"] + "_S" + df["Stint"].astype(str)
    )
    df["Compound_Stint"] = df["Compound"].astype(str) + "_S" + df["Stint"].astype(str)
    df["WetCompound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    df["ProgressLeft"] = 1.0 - df["RaceProgress"]
    df["EstimatedRaceLaps"] = df["LapNumber"] / np.clip(df["RaceProgress"], 1e-4, None)
    df["EstimatedLapsLeft"] = df["EstimatedRaceLaps"] - df["LapNumber"]
    df["TyreLifeFraction"] = df["TyreLife"] / np.clip(
        df["EstimatedRaceLaps"], 1.0, None
    )
    df["DegPerTyreLife"] = df["Cumulative_Degradation"] / (df["TyreLife"].abs() + 1.0)
    df["LapDeltaPerTyreLife"] = df["LapTime_Delta"] / (df["TyreLife"].abs() + 1.0)
    df["TyreLife_x_Progress"] = df["TyreLife"] * df["RaceProgress"]
    return df


def purge_train_indices(train_idx, val_idx, seq_codes, laps, embargo=1):
    val_seq = seq_codes[val_idx]
    val_lap = laps[val_idx]
    near = pd.concat(
        [
            pd.DataFrame({"seq": val_seq, "lap": val_lap + delta})
            for delta in range(-embargo, embargo + 1)
        ],
        ignore_index=True,
    ).drop_duplicates()
    train_pairs = pd.DataFrame(
        {"idx": train_idx, "seq": seq_codes[train_idx], "lap": laps[train_idx]}
    )
    train_pairs = train_pairs.merge(
        near.assign(_purge=1), on=["seq", "lap"], how="left"
    )
    keep = train_pairs["_purge"].isna().to_numpy()
    return train_pairs.loc[keep, "idx"].to_numpy(dtype=np.int64), int((~keep).sum())


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

rename_map = {c: safe_name(c) for c in train.columns}
train = train.rename(columns=rename_map)
test = test.rename(columns={c: safe_name(c) for c in test.columns})

train = add_features(train)
test = add_features(test)

cat_cols = [
    "Race",
    "Driver",
    "Compound",
    "Race_Year",
    "Race_Year_Driver",
    "Race_Year_Driver_Stint",
    "Compound_Stint",
]
cat_cols = [c for c in cat_cols if c in train.columns]

for col in cat_cols:
    all_vals = (
        pd.concat([train[col], test[col]], ignore_index=True)
        .astype("string")
        .fillna("__NA__")
    )
    cats = pd.Index(all_vals.unique())
    train[col] = pd.Categorical(
        train[col].astype("string").fillna("__NA__"), categories=cats
    )
    test[col] = pd.Categorical(
        test[col].astype("string").fillna("__NA__"), categories=cats
    )

feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
X = train[feature_cols]
X_test = test[feature_cols]
y = train[TARGET].astype(int).to_numpy()

pos = max(float(y.sum()), 1.0)
scale_pos_weight = float((len(y) - pos) / pos)

fold_groups = train["Race_Year_Driver_Stint"].astype(str).to_numpy()
seq_codes = train["Race_Year_Driver"].cat.codes.to_numpy()
laps = train["LapNumber"].astype(int).to_numpy()


def make_model(seed, n_estimators=900):
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


oof_plain = np.zeros(len(train), dtype=np.float32)
oof_purged = np.zeros(len(train), dtype=np.float32)
fold_stats = []
best_iters = []

gkf = GroupKFold(n_splits=N_SPLITS)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=fold_groups), 1):
    plain_model = make_model(SEED + fold)
    plain_model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
    )
    oof_plain[va_idx] = plain_model.predict_proba(X.iloc[va_idx])[:, 1]
    plain_auc = roc_auc_score(y[va_idx], oof_plain[va_idx])
    del plain_model
    gc.collect()

    purged_idx, purged_rows = purge_train_indices(
        tr_idx, va_idx, seq_codes, laps, embargo=1
    )
    purged_model = make_model(SEED + 100 + fold)
    purged_model.fit(
        X.iloc[purged_idx],
        y[purged_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
    )
    oof_purged[va_idx] = purged_model.predict_proba(X.iloc[va_idx])[:, 1]
    purged_auc = roc_auc_score(y[va_idx], oof_purged[va_idx])
    best_iters.append(int(purged_model.best_iteration_ or purged_model.n_estimators))
    fold_stats.append(
        {
            "fold": fold,
            "ordinary_auc": float(plain_auc),
            "purged_auc": float(purged_auc),
            "purged_rows": purged_rows,
            "train_rows_after_purge": int(len(purged_idx)),
            "valid_rows": int(len(va_idx)),
        }
    )
    print(
        f"fold {fold}: ordinary_auc={plain_auc:.6f} "
        f"purged_auc={purged_auc:.6f} purged_rows={purged_rows}"
    )
    del purged_model
    gc.collect()

ordinary_auc = roc_auc_score(y, oof_plain)
purged_auc = roc_auc_score(y, oof_purged)
score_drop = ordinary_auc - purged_auc

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y,
        "prediction": oof_purged,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_n_estimators = int(max(100, round(np.median(best_iters) * 1.05)))
final_model = make_model(SEED + 999, n_estimators=final_n_estimators)
final_model.fit(X, y, categorical_feature=cat_cols)
test_pred = final_model.predict_proba(X_test)[:, 1]

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000217"],
    "metric": "roc_auc",
    "validation_scheme": "5-fold GroupKFold on Race_Year_Driver_Stint with 1-lap same-race-year-driver purge",
    "ordinary_group_cv_roc_auc": float(ordinary_auc),
    "purged_1lap_group_cv_roc_auc": float(purged_auc),
    "score_drop_vs_ordinary_group_cv": float(score_drop),
    "folds": fold_stats,
    "final_n_estimators": final_n_estimators,
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"ordinary_group_cv_roc_auc: {ordinary_auc:.6f}")
print(f"purged_1lap_group_cv_roc_auc: {purged_auc:.6f}")
print(f"score_drop_vs_ordinary_group_cv: {score_drop:.6f}")
print(json.dumps(result, indent=2))
