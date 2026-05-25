import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESES = ["000371"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values
test_ids = sample[ID_COL].values


def add_traffic_features(df):
    df = df.copy()
    group_cols = ["Year", "Race", "LapNumber"]
    sort_cols = group_cols + ["Position"]

    df["_orig_order"] = np.arange(len(df))
    df = df.sort_values(sort_cols).reset_index(drop=True)

    g = df.groupby(group_cols, sort=False)
    lap_time = df["LapTime (s)"].astype(float)
    pos_change = df["Position_Change"].astype(float)

    df["lap_size"] = g["Position"].transform("count").astype(float)
    df["pos_pct"] = (df["Position"].astype(float) - 1.0) / np.maximum(
        df["lap_size"] - 1.0, 1.0
    )
    df["lap_time_rank_pct"] = g["LapTime (s)"].rank(pct=True, method="average")
    df["pos_change_rank_pct"] = g["Position_Change"].rank(pct=True, method="average")

    lap_median = g["LapTime (s)"].transform("median")
    lap_q75 = g["LapTime (s)"].transform(lambda s: s.quantile(0.75))
    lap_q25 = g["LapTime (s)"].transform(lambda s: s.quantile(0.25))
    lap_iqr = (lap_q75 - lap_q25).replace(0, np.nan)
    df["pace_loss_iqr"] = (
        ((lap_time - lap_median) / lap_iqr).replace([np.inf, -np.inf], 0).fillna(0)
    )

    for k in (1, 2):
        df[f"ahead{k}_lap_rank"] = g["lap_time_rank_pct"].shift(k)
        df[f"behind{k}_lap_rank"] = g["lap_time_rank_pct"].shift(-k)
        df[f"ahead{k}_poschg_rank"] = g["pos_change_rank_pct"].shift(k)
        df[f"behind{k}_poschg_rank"] = g["pos_change_rank_pct"].shift(-k)
        df[f"ahead{k}_pace_delta"] = df[f"ahead{k}_lap_rank"] - df["lap_time_rank_pct"]
        df[f"behind{k}_pace_delta"] = (
            df[f"behind{k}_lap_rank"] - df["lap_time_rank_pct"]
        )

    neigh_rank_cols = [
        "ahead1_lap_rank",
        "ahead2_lap_rank",
        "behind1_lap_rank",
        "behind2_lap_rank",
    ]
    neigh_chg_cols = [
        "ahead1_poschg_rank",
        "ahead2_poschg_rank",
        "behind1_poschg_rank",
        "behind2_poschg_rank",
    ]

    df["neighborhood_slow_rank_mean"] = df[neigh_rank_cols].mean(axis=1)
    df["neighborhood_slow_rank_max"] = df[neigh_rank_cols].max(axis=1)
    df["neighborhood_poschg_volatility"] = df[neigh_chg_cols].std(axis=1).fillna(0)

    df["ahead_pressure"] = (
        (df["ahead1_lap_rank"].fillna(0.5) > df["lap_time_rank_pct"]).astype(float)
        + 0.5
        * (df["ahead2_lap_rank"].fillna(0.5) > df["lap_time_rank_pct"]).astype(float)
        + np.maximum(df["ahead1_pace_delta"].fillna(0), 0)
    )
    df["behind_pressure"] = (
        (df["behind1_lap_rank"].fillna(0.5) < df["lap_time_rank_pct"]).astype(float)
        + 0.5
        * (df["behind2_lap_rank"].fillna(0.5) < df["lap_time_rank_pct"]).astype(float)
        + np.maximum(-df["behind1_pace_delta"].fillna(0), 0)
    )

    df["local_traffic_density_3"] = (
        df["ahead1_lap_rank"].sub(df["lap_time_rank_pct"]).abs() < 0.15
    ).astype(float) + (
        df["behind1_lap_rank"].sub(df["lap_time_rank_pct"]).abs() < 0.15
    ).astype(
        float
    )
    df["local_traffic_density_5"] = sum(
        (df[c].sub(df["lap_time_rank_pct"]).abs() < 0.20).astype(float)
        for c in neigh_rank_cols
    )

    df["free_air_ahead"] = (
        df["ahead1_lap_rank"].isna()
        | (df["ahead1_lap_rank"] < df["lap_time_rank_pct"] - 0.25)
        | (df["ahead1_poschg_rank"].fillna(0.5) < 0.35)
    ).astype(int)
    df["free_air_behind"] = (
        df["behind1_lap_rank"].isna()
        | (df["behind1_lap_rank"] > df["lap_time_rank_pct"] + 0.25)
        | (df["behind1_poschg_rank"].fillna(0.5) > 0.65)
    ).astype(int)
    df["free_air_window"] = (
        (df["free_air_ahead"] == 1) & (df["local_traffic_density_5"] <= 1)
    ).astype(int)

    race_max_lap = df.groupby(["Year", "Race"], sort=False)["LapNumber"].transform(
        "max"
    )
    df["laps_remaining"] = (race_max_lap - df["LapNumber"]).clip(lower=0)
    df["race_progress_x_free_air"] = df["RaceProgress"] * df["free_air_window"]
    df["tyrelife_x_traffic"] = df["TyreLife"] * df["local_traffic_density_5"]
    df["tyrelife_x_free_air"] = df["TyreLife"] * df["free_air_window"]
    df["pace_loss_x_ahead_pressure"] = df["pace_loss_iqr"] * df["ahead_pressure"]
    df["pace_loss_x_free_air"] = df["pace_loss_iqr"] * df["free_air_window"]
    df["laps_remaining_x_traffic"] = (
        df["laps_remaining"] * df["local_traffic_density_5"]
    )

    for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        is_comp = (df["Compound"] == comp).astype(int)
        df[f"{comp.lower()}_traffic"] = is_comp * df["local_traffic_density_5"]
        df[f"{comp.lower()}_free_air"] = is_comp * df["free_air_window"]
        df[f"{comp.lower()}_pace_loss"] = is_comp * df["pace_loss_iqr"]

    df = (
        df.sort_values("_orig_order")
        .drop(columns=["_orig_order"])
        .reset_index(drop=True)
    )
    return df


all_df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
all_df = add_traffic_features(all_df)

cat_cols = all_df.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in all_df.columns if c not in cat_cols + [ID_COL]]

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
)
all_df[cat_cols] = encoder.fit_transform(all_df[cat_cols].astype(str))

feature_cols = cat_cols + num_cols
X = all_df.iloc[: len(train)][feature_cols].replace([np.inf, -np.inf], np.nan)
X_test = all_df.iloc[len(train) :][feature_cols].replace([np.inf, -np.inf], np.nan)

try:
    from lightgbm import LGBMClassifier

    model_factory = lambda seed: LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    fit_kwargs = {}
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model_factory = lambda seed: HistGradientBoostingClassifier(
        max_iter=450,
        learning_rate=0.045,
        max_leaf_nodes=48,
        l2_regularization=0.05,
        random_state=seed,
    )
    fit_kwargs = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=371)
oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    model = model_factory(371 + fold)
    model.fit(X.iloc[tr_idx], y[tr_idx], **fit_kwargs)

    va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    te_pred = model.predict_proba(X_test)[:, 1]

    oof[va_idx] = va_pred
    test_pred += te_pred / skf.n_splits

    score = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(score)
    print(f"fold {fold} roc_auc: {score:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"CV ROC AUC: {cv_auc:.6f}")

pd.DataFrame({"row": np.arange(len(train)), "target": y, "prediction": oof}).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_predictions = sample.copy()
test_predictions[TARGET] = np.clip(test_pred, 0, 1)
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
test_predictions.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(s) for s in fold_scores],
    "research_hypotheses_llm_claimed_used": HYPOTHESES,
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
