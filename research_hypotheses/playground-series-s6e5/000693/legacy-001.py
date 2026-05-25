import os
import json
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

RANDOM_STATE = 42
TARGET = "PitNextLap"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.rename(columns={"LapTime (s)": "LapTime_s"})
test = test.rename(columns={"LapTime (s)": "LapTime_s"})

train["__is_train"] = 1
test["__is_train"] = 0
train["__row"] = np.arange(len(train))
test["__row"] = np.arange(len(test))
test[TARGET] = np.nan

all_df = pd.concat([train, test], axis=0, ignore_index=True, sort=False)


def sanitize_columns(cols):
    used = {}
    out = []
    for c in cols:
        s = re.sub(r"[^0-9A-Za-z_]+", "_", c).strip("_")
        if not s:
            s = "feature"
        base = s
        k = used.get(base, 0)
        if k:
            s = f"{base}_{k}"
        used[base] = k + 1
        out.append(s)
    return dict(zip(cols, out))


def rolling_slope(values):
    mask = np.isfinite(values)
    y = values[mask]
    n = len(y)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=np.float32)
    x -= x.mean()
    denom = float(np.sum(x * x))
    if denom == 0:
        return np.nan
    return float(np.sum(x * (y - y.mean())) / denom)


def past_lap_z(s):
    prior_mean = s.expanding(min_periods=2).mean().shift(1)
    prior_std = s.expanding(min_periods=2).std().shift(1)
    return (s - prior_mean) / prior_std.replace(0, np.nan)


def add_sequence_features(df):
    df = df.copy()
    sort_cols = ["Year", "Race", "Driver", "Stint", "LapNumber", "id"]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    base_keys = ["Year", "Race", "Driver", "Stint"]
    df["__pit_reset"] = (
        df.groupby(base_keys, sort=False)["PitStop"]
        .transform(lambda s: s.shift(1).fillna(0).cumsum())
        .astype("int16")
    )
    seq_keys = base_keys + ["__pit_reset"]
    g = df.groupby(seq_keys, sort=False)

    df["incremental_degradation"] = g["Cumulative_Degradation"].diff()
    df["lap_time_z_past"] = g["LapTime_s"].transform(past_lap_z)

    seq_vars = [
        "LapTime_Delta",
        "lap_time_z_past",
        "incremental_degradation",
        "Position_Change",
    ]

    for col in seq_vars:
        for lag in (1, 2, 3):
            df[f"{col}_lag{lag}"] = g[col].shift(lag)

        df[f"{col}_roll3_vol"] = g[col].transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).std()
        )
        df[f"{col}_roll3_slope"] = g[col].transform(
            lambda s: s.shift(1)
            .rolling(3, min_periods=2)
            .apply(rolling_slope, raw=True)
        )

    df = df.sort_values(
        ["__is_train", "__row"], ascending=[False, True], kind="mergesort"
    )
    return df.reset_index(drop=True)


all_df = add_sequence_features(all_df)

drop_cols = {TARGET, "id", "__is_train", "__row", "__pit_reset"}
feature_cols = [c for c in all_df.columns if c not in drop_cols]

cat_cols = [
    c
    for c in feature_cols
    if all_df[c].dtype == "object" or str(all_df[c].dtype).startswith("category")
]
for c in cat_cols:
    all_df[c] = all_df[c].astype("category")

for c in feature_cols:
    if c not in cat_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if all_df[c].dtype == "float64":
            all_df[c] = all_df[c].astype("float32")

trn = all_df[all_df["__is_train"] == 1].sort_values("__row").reset_index(drop=True)
tst = all_df[all_df["__is_train"] == 0].sort_values("__row").reset_index(drop=True)

rename_map = sanitize_columns(feature_cols)
X = trn[feature_cols].rename(columns=rename_map)
X_test = tst[feature_cols].rename(columns=rename_map)
cat_cols = [rename_map[c] for c in cat_cols]
y = trn[TARGET].astype(int)

groups = (
    trn["Year"].astype(str)
    + "_"
    + trn["Race"].astype(str)
    + "_"
    + trn["Driver"].astype(str)
)


def make_splits():
    if StratifiedGroupKFold is not None:
        try:
            cv = StratifiedGroupKFold(
                n_splits=5, shuffle=True, random_state=RANDOM_STATE
            )
            splits = list(cv.split(X, y, groups))
            if all(y.iloc[val].nunique() == 2 for _, val in splits):
                return splits, "StratifiedGroupKFold"
        except Exception:
            pass
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(X, y)), "StratifiedKFold"


splits, split_name = make_splits()

pos = int(y.sum())
neg = int(len(y) - pos)
scale_pos_weight = neg / max(pos, 1)

params = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_estimators=1600,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=90,
    subsample=0.90,
    subsample_freq=1,
    colsample_bytree=0.90,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=max(1, os.cpu_count() or 1),
    verbose=-1,
)

oof = np.zeros(len(X), dtype=np.float32)
fold_aucs = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[
            lgb.early_stopping(100, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred.astype(np.float32)
    auc = roc_auc_score(y.iloc[va_idx], pred)
    fold_aucs.append(float(auc))
    best_iters.append(int(model.best_iteration_ or params["n_estimators"]))
    print(f"fold {fold} {split_name} roc_auc={auc:.6f} best_iter={best_iters[-1]}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(y)),
        "target": y.values,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_params = params.copy()
final_params["n_estimators"] = int(
    np.clip(np.mean(best_iters), 100, params["n_estimators"])
)
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_auc": float(cv_auc),
            "fold_aucs": fold_aucs,
            "cv_split": split_name,
            "research_hypotheses_llm_claimed_used": ["000693"],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        }
    )
)
