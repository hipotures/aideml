import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
GROUP_COLS = ["Year", "Race"]
LAP_KEY = ["Year", "Race", "LapNumber"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_n = len(train)

all_df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)


def add_neutralization_features(df):
    df = df.copy()
    grp = df.groupby(LAP_KEY, sort=False)

    agg = grp.agg(
        lap_time_med=("LapTime (s)", "median"),
        lap_time_std=("LapTime (s)", "std"),
        delta_med=("LapTime_Delta", "median"),
        delta_std=("LapTime_Delta", "std"),
        pit_share=("PitStop", "mean"),
        n_cars=(ID_COL, "size"),
    ).reset_index()

    q_lap_hi = df["LapTime (s)"].quantile(0.95)
    q_delta_hi = df["LapTime_Delta"].quantile(0.95)
    slow = df.assign(
        extreme_slow=(
            (df["LapTime (s)"] >= q_lap_hi) | (df["LapTime_Delta"] >= q_delta_hi)
        ).astype(float)
    )
    slow_agg = (
        slow.groupby(LAP_KEY, sort=False)["extreme_slow"]
        .mean()
        .reset_index(name="extreme_slow_share")
    )
    agg = agg.merge(slow_agg, on=LAP_KEY, how="left")

    lag_cols = [
        "lap_time_med",
        "lap_time_std",
        "delta_med",
        "delta_std",
        "pit_share",
        "extreme_slow_share",
        "n_cars",
    ]
    agg = agg.sort_values(["Year", "Race", "LapNumber"])
    for c in lag_cols:
        agg[f"lag1_{c}"] = agg.groupby(["Year", "Race"], sort=False)[c].shift(1)

    df = df.merge(
        agg[LAP_KEY + [f"lag1_{c}" for c in lag_cols]], on=LAP_KEY, how="left"
    )

    for c in [f"lag1_{x}" for x in lag_cols]:
        df[c] = df[c].fillna(df[c].median())

    def zscore(s):
        sd = s.std()
        if sd == 0 or np.isnan(sd):
            return s * 0.0
        return (s - s.mean()) / sd

    z_delta = zscore(df["lag1_delta_med"])
    z_lap = zscore(df["lag1_lap_time_med"])
    z_pit = zscore(df["lag1_pit_share"])
    z_slow = zscore(df["lag1_extreme_slow_share"])
    raw_score = 0.35 * z_delta + 0.25 * z_lap + 0.25 * z_slow + 0.15 * z_pit
    df["neutralization_score"] = 1.0 / (1.0 + np.exp(-raw_score.clip(-8, 8)))

    df["neutral_x_tyre"] = df["neutralization_score"] * df["TyreLife"]
    df["neutral_x_progress"] = df["neutralization_score"] * df["RaceProgress"]
    df["neutral_x_stint"] = df["neutralization_score"] * df["Stint"]
    df["tyre_progress"] = df["TyreLife"] * df["RaceProgress"]
    df["lap_progress_gap"] = df["LapNumber"] - (
        df["RaceProgress"] * df.groupby(["Year", "Race"])["LapNumber"].transform("max")
    )
    return df


all_df = add_neutralization_features(all_df)

for c in CAT_COLS:
    all_df[c] = all_df[c].astype("category")

features = [c for c in all_df.columns if c != ID_COL]
X = all_df.iloc[:train_n][features].copy()
X_test = all_df.iloc[train_n:][features].copy()
groups = train[GROUP_COLS].astype(str).agg("_".join, axis=1)

try:
    from lightgbm import LGBMClassifier

    model_name = "lightgbm"
    base_params = dict(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=80,
        reg_alpha=0.1,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=max(1, os.cpu_count() or 1),
        verbose=-1,
    )

    def make_model(seed):
        p = base_params.copy()
        p["random_state"] = seed
        return LGBMClassifier(**p)

except Exception:
    from catboost import CatBoostClassifier

    model_name = "catboost"
    cat_idx = [features.index(c) for c in CAT_COLS if c in features]

    def make_model(seed):
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=900,
            learning_rate=0.045,
            depth=7,
            l2_leaf_reg=5.0,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=max(1, os.cpu_count() or 1),
            cat_features=cat_idx,
        )


gkf = GroupKFold(n_splits=5)
oof = np.zeros(train_n, dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
    clf = make_model(41 + fold)
    if model_name == "lightgbm":
        clf.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            categorical_feature=[c for c in CAT_COLS if c in features],
        )
    else:
        clf.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=(X.iloc[va_idx], y[va_idx]),
            use_best_model=True,
        )

    oof[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += clf.predict_proba(X_test)[:, 1] / gkf.n_splits
    score = roc_auc_score(y[va_idx], oof[va_idx])
    fold_scores.append(score)
    print(f"fold {fold} roc_auc: {score:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"overall_grouped_cv_roc_auc: {cv_auc:.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000369"],
            "metric": "roc_auc",
            "cv_auc": float(cv_auc),
            "fold_auc": [float(s) for s in fold_scores],
        }
    )
)

submission = sample[[ID_COL]].copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(train_n),
        "target": y,
        "prediction": np.clip(oof, 0, 1),
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_predictions = sample[[ID_COL]].copy()
test_predictions[TARGET] = np.clip(test_pred, 0, 1)
test_predictions.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
