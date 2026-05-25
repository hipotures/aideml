import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

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

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
y = train[target_col].astype(int).reset_index(drop=True)


def add_current_pit_features(df, keys, prefix):
    g = df.groupby(keys, observed=True)["PitStop"]
    cnt = g.transform("count").astype("float32")
    sm = g.transform("sum").astype("float32")
    own = df["PitStop"].astype("float32")

    df[f"{prefix}_cur_pit_count"] = sm
    df[f"{prefix}_cur_group_size"] = cnt
    df[f"{prefix}_cur_pit_share"] = sm / np.maximum(cnt, 1.0)
    df[f"{prefix}_cur_pit_loo_share"] = np.where(
        cnt > 1, (sm - own) / np.maximum(cnt - 1.0, 1.0), 0.0
    ).astype("float32")
    return df


def add_prev_pit_features(df, keys, prefix):
    group_cols = list(keys) + ["LapNumber"]
    agg = (
        df.groupby(group_cols, observed=True)["PitStop"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": f"{prefix}_prev_pit_count",
                "count": f"{prefix}_prev_group_size",
            }
        )
    )
    agg["LapNumber"] = agg["LapNumber"] + 1
    agg[f"{prefix}_prev_pit_share"] = agg[f"{prefix}_prev_pit_count"] / np.maximum(
        agg[f"{prefix}_prev_group_size"], 1
    )
    keep = group_cols + [
        f"{prefix}_prev_pit_count",
        f"{prefix}_prev_group_size",
        f"{prefix}_prev_pit_share",
    ]
    return df.merge(agg[keep], on=group_cols, how="left", sort=False)


def build_features(train_df, test_df):
    base_train = train_df.drop(columns=[target_col])
    df = pd.concat([base_train, test_df], axis=0, ignore_index=True, sort=False)

    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["Position_Band"] = (
        ((df["Position"].astype(int) - 1) // 5).clip(0, 3).astype("int8")
    )

    rp = df["RaceProgress"].astype("float32").clip(lower=1e-3)
    tyre = df["TyreLife"].astype("float32").clip(lower=1.0)
    lap = df["LapNumber"].astype("float32").clip(lower=1.0)

    df["EstimatedRaceLaps"] = lap / rp
    df["EstimatedLapsRemaining"] = df["EstimatedRaceLaps"] - lap
    df["TyreLife_to_LapNumber"] = tyre / lap
    df["TyreLife_to_EstRaceLaps"] = tyre / df["EstimatedRaceLaps"].clip(lower=1.0)
    df["Degradation_per_TyreLap"] = (
        df["Cumulative_Degradation"].astype("float32") / tyre
    )
    df["Abs_LapTime_Delta"] = df["LapTime_Delta"].abs().astype("float32")
    df["LapDelta_per_TyreLap"] = df["LapTime_Delta"].astype("float32") / tyre
    df["PositionLoss"] = (df["Position_Change"] > 0).astype("int8")
    df["PositionGain"] = (df["Position_Change"] < 0).astype("int8")
    df["TyreLife_x_RaceProgress"] = tyre * rp
    df["Stint_x_TyreLife"] = df["Stint"].astype("float32") * tyre

    compound_center = {
        "SOFT": 17.0,
        "MEDIUM": 24.0,
        "HARD": 31.0,
        "INTERMEDIATE": 13.0,
        "WET": 10.0,
    }
    center = (
        df["Compound"].astype(str).map(compound_center).fillna(20.0).astype("float32")
    )
    df["CompoundWindowCenter"] = center
    df["TyreWindowExcess"] = tyre - center
    df["TyreWindowAbsDistance"] = (tyre - center).abs()
    df["IsSlick"] = df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    df["IsWetWeatherCompound"] = (
        df["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")
    )

    df = add_current_pit_features(df, ["Race_Year", "LapNumber"], "lap")
    df = add_current_pit_features(
        df, ["Race_Year", "LapNumber", "Compound"], "compound"
    )
    df = add_current_pit_features(
        df, ["Race_Year", "LapNumber", "Position_Band"], "posband"
    )

    df = add_prev_pit_features(df, ["Race_Year"], "lap")
    df = add_prev_pit_features(df, ["Race_Year", "Compound"], "compound")
    df = add_prev_pit_features(df, ["Race_Year", "Position_Band"], "posband")

    rank_cols = ["TyreLife", "Cumulative_Degradation", "LapTime_Delta"]
    for col in rank_cols:
        df[f"{col}_lap_rank_pct"] = (
            df.groupby(["Race_Year", "LapNumber"], observed=True)[col]
            .rank(method="average", pct=True)
            .astype("float32")
        )
        df[f"{col}_posband_rank_pct"] = (
            df.groupby(["Race_Year", "LapNumber", "Position_Band"], observed=True)[col]
            .rank(method="average", pct=True)
            .astype("float32")
        )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in ["Race", "Driver", "Compound", "Race_Year", "Position_Band"]:
        df[c] = df[c].astype("category")

    train_fe = df.iloc[: len(train_df)].reset_index(drop=True)
    test_fe = df.iloc[len(train_df) :].reset_index(drop=True)
    return train_fe, test_fe


X_train, X_test = build_features(train, test)

all_features = [c for c in X_train.columns if c != "id"]
current_pit_features = [c for c in all_features if "_cur_" in c]
lag_only_drop = set(current_pit_features + ["PitStop"])

feature_sets = {
    "lag_only": [c for c in all_features if c not in lag_only_drop],
    "current_and_lag": all_features,
}

groups = X_train["Race_Year"].astype(str).values
if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(X_train, y, groups=groups))
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(X_train, y))

if any(y.iloc[val_idx].nunique() < 2 for _, val_idx in splits):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(splitter.split(X_train, y))

pos = max(int(y.sum()), 1)
neg = max(int(len(y) - y.sum()), 1)
scale_pos_weight = neg / pos

base_params = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_estimators=900,
    learning_rate=0.035,
    num_leaves=64,
    min_child_samples=90,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=3.0,
    max_bin=255,
    scale_pos_weight=scale_pos_weight,
    random_state=2026,
    n_jobs=-1,
    verbose=-1,
)


def run_cv(name, features):
    cat_features = [c for c in features if str(X_train[c].dtype) == "category"]
    oof = np.zeros(len(X_train), dtype=np.float32)
    fold_scores = []
    best_iters = []

    for fold, (tr_idx, val_idx) in enumerate(splits, 1):
        model = lgb.LGBMClassifier(**base_params)
        model.fit(
            X_train.iloc[tr_idx][features],
            y.iloc[tr_idx],
            eval_set=[(X_train.iloc[val_idx][features], y.iloc[val_idx])],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
        )
        pred = model.predict_proba(X_train.iloc[val_idx][features])[:, 1]
        oof[val_idx] = pred.astype(np.float32)
        auc = roc_auc_score(y.iloc[val_idx], pred)
        fold_scores.append(float(auc))
        best_iters.append(int(model.best_iteration_ or base_params["n_estimators"]))
        print(f"{name} fold {fold} ROC AUC: {auc:.6f}")

    overall = roc_auc_score(y, oof)
    print(f"{name} OOF ROC AUC: {overall:.6f}")
    return {
        "name": name,
        "features": features,
        "oof": oof,
        "fold_scores": fold_scores,
        "auc": float(overall),
        "best_iteration": int(np.median(best_iters)),
    }


results = [run_cv(name, feats) for name, feats in feature_sets.items()]
chosen = max(results, key=lambda r: r["auc"])

final_params = dict(base_params)
final_params["n_estimators"] = max(100, chosen["best_iteration"])
final_params["random_state"] = 9001
cat_features = [c for c in chosen["features"] if str(X_train[c].dtype) == "category"]

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(
    X_train[chosen["features"]],
    y,
    categorical_feature=cat_features,
)

test_pred = final_model.predict_proba(X_test[chosen["features"]])[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission["PitNextLap"] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        "row": np.arange(len(y), dtype=np.int32),
        "target": y.astype(int),
        "prediction": chosen["oof"],
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

report = {
    "metric": "roc_auc",
    "cv_auc": chosen["auc"],
    "chosen_variant": chosen["name"],
    "variant_auc": {r["name"]: r["auc"] for r in results},
    "fold_scores": {r["name"]: r["fold_scores"] for r in results},
    "research_hypotheses_llm_claimed_used": ["000356"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "oof_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
}

with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(report, f, indent=2)

print(f"Selected variant: {chosen['name']}")
print(f"Validation ROC AUC: {chosen['auc']:.6f}")
print(json.dumps(report, sort_keys=True))
