import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    from sklearn.model_selection import GroupKFold

    StratifiedGroupKFold = None

import lightgbm as lgb

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
GROUP_COLS = ["Year", "Race", "Driver", "Stint"]
RANDOM_STATE = 483


def sanitize_columns(cols):
    seen = {}
    out = []
    for c in cols:
        s = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
        if not s:
            s = "feature"
        base = s
        k = seen.get(base, 0)
        while s in seen:
            k += 1
            s = f"{base}_{k}"
        seen[base] = k
        seen[s] = 0
        out.append(s)
    return dict(zip(cols, out))


def add_lag_state_features(all_df):
    df = all_df.copy()
    df["_order"] = np.arange(len(df))
    df["RaceYear"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)

    sort_cols = GROUP_COLS + ["LapNumber", ID_COL]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    gb = df.groupby(GROUP_COLS, sort=False, observed=True)

    prior_pace_mean = gb["LapTime (s)"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["_pace_resid"] = df["LapTime (s)"] - prior_pace_mean

    df["_deg_diff"] = gb["Cumulative_Degradation"].diff()
    df["_deg_accel"] = gb["_deg_diff"].diff()
    df["_pace_diff"] = gb["LapTime (s)"].diff()
    df["_delta_diff"] = gb["LapTime_Delta"].diff()

    state_defs = {
        "pace_resid": "_pace_resid",
        "deg_slope": "_deg_diff",
        "deg_accel": "_deg_accel",
        "pace_slope": "_pace_diff",
        "delta_slope": "_delta_diff",
    }

    for name, col in state_defs.items():
        for span in (3, 5):
            df[f"state_{name}_ema{span}"] = df.groupby(
                GROUP_COLS, sort=False, observed=True
            )[col].transform(
                lambda s, sp=span: s.shift(1)
                .ewm(span=sp, adjust=False, min_periods=1)
                .mean()
            )

    for name, col in [
        ("pace_resid", "_pace_resid"),
        ("lap_delta", "LapTime_Delta"),
        ("deg_slope", "_deg_diff"),
    ]:
        for window in (3, 5):
            df[f"state_{name}_vol{window}"] = df.groupby(
                GROUP_COLS, sort=False, observed=True
            )[col].transform(
                lambda s, w=window: s.shift(1).rolling(w, min_periods=2).std()
            )

    df["state_tyre_life_lag1"] = gb["TyreLife"].shift(1)
    df["state_laptime_lag1"] = gb["LapTime (s)"].shift(1)
    df["state_lapdelta_lag1"] = gb["LapTime_Delta"].shift(1)
    df["state_cumdeg_lag1"] = gb["Cumulative_Degradation"].shift(1)
    df["state_pitstop_lag1"] = gb["PitStop"].shift(1)
    df["state_stint_lap_index"] = gb.cumcount().astype("float32")

    resid_lag1 = gb["_pace_resid"].shift(1)
    resid_hist_mean = gb["_pace_resid"].transform(
        lambda s: s.shift(2).rolling(5, min_periods=2).mean()
    )
    resid_hist_std = gb["_pace_resid"].transform(
        lambda s: s.shift(2).rolling(5, min_periods=2).std()
    )
    df["state_slow_shock_prev"] = (
        (resid_hist_std > 0) & (resid_lag1 > resid_hist_mean + 2.0 * resid_hist_std)
    ).astype("int8")
    df["state_slow_shock_recent3"] = (
        df.groupby(GROUP_COLS, sort=False, observed=True)["state_slow_shock_prev"]
        .transform(lambda s: s.rolling(3, min_periods=1).max())
        .astype("int8")
    )

    df["state_resid_ema_gap_3_5"] = (
        df["state_pace_resid_ema3"] - df["state_pace_resid_ema5"]
    )
    df["state_deg_slope_ema_gap_3_5"] = (
        df["state_deg_slope_ema3"] - df["state_deg_slope_ema5"]
    )

    drop_tmp = [c for c in df.columns if c.startswith("_") and c != "_order"]
    df = df.drop(columns=drop_tmp)
    df = (
        df.sort_values("_order", kind="mergesort")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    state_cols = [c for c in df.columns if c.startswith("state_")]
    df[state_cols] = df[state_cols].astype("float32")
    return df


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["_is_train"] = 1
test["_is_train"] = 0
test[TARGET] = np.nan

all_df = pd.concat([train, test], axis=0, ignore_index=True)
all_df = add_lag_state_features(all_df)

cat_cols = ["Compound", "Race", "Driver", "RaceYear"]
for c in cat_cols:
    all_df[c] = all_df[c].astype("category")

drop_cols = [ID_COL, TARGET, "_is_train"]
feature_cols = [c for c in all_df.columns if c not in drop_cols]
rename_map = sanitize_columns(feature_cols)

X_all = all_df[feature_cols].rename(columns=rename_map)
cat_features = [rename_map[c] for c in cat_cols if c in rename_map]

train_mask = all_df["_is_train"].values == 1
X = X_all.loc[train_mask].reset_index(drop=True)
X_test = X_all.loc[~train_mask].reset_index(drop=True)
y = all_df.loc[train_mask, TARGET].astype(int).reset_index(drop=True)

groups = (
    all_df.loc[train_mask, "Year"].astype(str).values
    + "_"
    + all_df.loc[train_mask, "Race"].astype(str).values
)

pos = float(y.sum())
neg = float(len(y) - y.sum())
scale_pos_weight = np.sqrt(neg / max(pos, 1.0))

base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    learning_rate=0.035,
    n_estimators=1800,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=5.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=max(1, min(16, os.cpu_count() or 1)),
    verbosity=-1,
)

if StratifiedGroupKFold is not None:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = cv.split(X, y, groups)
else:
    cv = GroupKFold(n_splits=5)
    splits = cv.split(X, y, groups)

oof = np.zeros(len(X), dtype=np.float32)
fold_aucs = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**base_params)
    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y.iloc[va_idx], pred)
    fold_aucs.append(float(auc))
    best_iters.append(int(model.best_iteration_ or base_params["n_estimators"]))
    print(f"Fold {fold} ROC AUC: {auc:.6f} best_iteration={best_iters[-1]}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold grouped ROC AUC: {cv_auc:.6f}")

final_iters = int(np.median(best_iters)) if best_iters else 600
final_params = dict(base_params)
final_params["n_estimators"] = max(50, final_iters)

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_features)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(y), dtype=np.int64),
        "target": y.values,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission[[ID_COL, TARGET]].to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "research_hypotheses_llm_claimed_used": ["000483"],
    "metric": "roc_auc",
    "validation_scheme": "5-fold StratifiedGroupKFold by Year/Race",
    "validation_auc": float(cv_auc),
    "fold_aucs": fold_aucs,
    "final_model_iterations": final_params["n_estimators"],
}
with open(os.path.join(WORK_DIR, "review.json"), "w", encoding="utf-8") as f:
    json.dump(review, f, indent=2)

print(f"Saved ./working/submission.csv")
print(json.dumps(review, indent=2))
