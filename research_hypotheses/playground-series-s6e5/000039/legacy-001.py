import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT = "./input"
WORKING = "./working"
os.makedirs(WORKING, exist_ok=True)


def clean_columns(df):
    out = df.copy()
    seen = {}
    new_cols = []
    for c in out.columns:
        nc = re.sub(r"[^0-9A-Za-z_]+", "_", c).strip("_")
        if nc in seen:
            seen[nc] += 1
            nc = f"{nc}_{seen[nc]}"
        else:
            seen[nc] = 0
        new_cols.append(nc)
    out.columns = new_cols
    return out


train = clean_columns(pd.read_csv(f"{INPUT}/train.csv.gz"))
test = clean_columns(pd.read_csv(f"{INPUT}/test.csv.gz"))
sample = pd.read_csv(f"{INPUT}/sample_submission.csv.gz")

target = "PitNextLap"
id_col = "id"
y = train[target].astype(int).values
n_train = len(train)

all_df = pd.concat([train.drop(columns=[target]), test], axis=0, ignore_index=True)


def add_peer_pressure_features(df, n_train):
    df = df.copy()
    base_train = df.iloc[:n_train]

    pressure_cols = [
        "TyreLife",
        "Cumulative_Degradation",
        "LapTime_Delta",
        "RaceProgress",
        "Stint",
    ]
    weights = {
        "TyreLife": 0.35,
        "Cumulative_Degradation": 0.25,
        "LapTime_Delta": 0.25,
        "RaceProgress": 0.10,
        "Stint": 0.05,
    }

    own_pressure = np.zeros(len(df), dtype=np.float32)
    for col in pressure_cols:
        q1, q3 = base_train[col].quantile([0.25, 0.75])
        scale = max(float(q3 - q1), 1e-6)
        med = float(base_train[col].median())
        z = ((df[col].astype(float) - med) / scale).clip(-5, 5).astype(np.float32)
        df[f"pressure_z_{col}"] = z
        own_pressure += weights[col] * z

    df["own_pressure"] = own_pressure
    pressure_cut = float(pd.Series(own_pressure[:n_train]).quantile(0.75))
    df["own_high_pressure"] = (df["own_pressure"] >= pressure_cut).astype(np.int8)

    tyre_cut = float(base_train["TyreLife"].quantile(0.75))
    deg_cut = float(base_train["Cumulative_Degradation"].quantile(0.75))
    delta_cut = float(base_train["LapTime_Delta"].quantile(0.75))
    df["old_tyre_flag"] = (df["TyreLife"] >= tyre_cut).astype(np.int8)
    df["high_degradation_flag"] = (df["Cumulative_Degradation"] >= deg_cut).astype(
        np.int8
    )
    df["slow_lap_flag"] = (df["LapTime_Delta"] >= delta_cut).astype(np.int8)
    df["position_loss_flag"] = (df["Position_Change"] > 0).astype(np.int8)

    group_cols = ["Year", "Race", "LapNumber"]
    peer_cols = [
        "own_pressure",
        "own_high_pressure",
        "old_tyre_flag",
        "high_degradation_flag",
        "slow_lap_flag",
        "position_loss_flag",
        "PitStop",
        "TyreLife",
        "Cumulative_Degradation",
        "LapTime_Delta",
        "RaceProgress",
        "Stint",
        "Position_Change",
    ]

    group = df.groupby(group_cols, sort=False, observed=True)
    counts = group[id_col].transform("count").astype(np.float32)
    denom = (counts - 1).replace(0, np.nan)
    sums = group[peer_cols].transform("sum")
    peer_means = sums.sub(df[peer_cols], axis=0).div(denom, axis=0)
    defaults = df.iloc[:n_train][peer_cols].mean()
    peer_means = peer_means.fillna(defaults)

    df["race_lap_peer_count"] = (counts - 1).clip(lower=0).astype(np.float32)
    for col in peer_cols:
        if col == "own_pressure":
            peer_name = "peer_mean_pressure"
        elif col == "own_high_pressure":
            peer_name = "peer_high_pressure_rate"
        else:
            peer_name = f"peer_loo_mean_{col}"
        df[peer_name] = peer_means[col].astype(np.float32)
        df[f"own_minus_peer_{col}"] = df[col].astype(np.float32) - peer_means[
            col
        ].astype(np.float32)

    return df


all_df = add_peer_pressure_features(all_df, n_train)

for col in all_df.select_dtypes(include=["object"]).columns:
    all_df[col] = all_df[col].astype("category")

for col in all_df.select_dtypes(include=["float64"]).columns:
    all_df[col] = all_df[col].astype("float32")

X = all_df.iloc[:n_train].drop(columns=[id_col])
X_test = all_df.iloc[n_train:].drop(columns=[id_col])
cat_cols = X.select_dtypes(include=["category"]).columns.tolist()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
oof = np.zeros(n_train, dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    y_tr, y_va = y[tr_idx], y[va_idx]
    pos = max(int(y_tr.sum()), 1)
    neg = len(y_tr) - pos

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=2500,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        scale_pos_weight=neg / pos,
        random_state=2026 + fold,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X.iloc[tr_idx],
        y_tr,
        eval_set=[(X.iloc[va_idx], y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / skf.n_splits
    fold_auc = roc_auc_score(y_va, oof[va_idx])
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold CV ROC AUC: {cv_auc:.6f}")

pd.DataFrame({"row": np.arange(n_train), "target": y, "prediction": oof}).to_csv(
    f"{WORKING}/oof_predictions.csv.gz", index=False, compression="gzip"
)

sub = sample.copy()
pred_col = [c for c in sub.columns if c != id_col][0]
sub[pred_col] = np.clip(test_pred, 0, 1)
sub.to_csv(f"{WORKING}/submission.csv", index=False)
sub.to_csv(f"{WORKING}/test_predictions.csv.gz", index=False, compression="gzip")

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "research_hypotheses_llm_claimed_used": ["000039"],
        }
    )
)
