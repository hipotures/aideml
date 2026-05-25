import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    from sklearn.model_selection import GroupKFold

    HAS_SGK = False

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RACE_COLS = ["Year", "Race"]
LAGS = (1, 2, 3)
DECAY_WEIGHTS = {1: 1.0, 2: 0.55, 3: 0.30}


def make_unique_safe_names(cols):
    out, seen = {}, {}
    for c in cols:
        s = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
        if not s:
            s = "feature"
        if s[0].isdigit():
            s = "f_" + s
        base = s
        k = seen.get(base, 0)
        while s in seen:
            k += 1
            s = f"{base}_{k}"
        seen[base] = k
        seen[s] = 0
        out[c] = s
    return out


def add_shifted_pit_counts(df, group_cols, prefix, denom_col):
    keys = RACE_COLS + ["LapNumber"] + group_cols
    agg = df.groupby(keys, observed=True)["PitStop"].sum().reset_index(name="pit_count")

    for lag in LAGS:
        tmp = agg.copy()
        tmp["LapNumber"] = tmp["LapNumber"] + lag
        count_col = f"{prefix}_pits_lag{lag}"
        tmp = tmp.rename(columns={"pit_count": count_col})
        df = df.merge(tmp, on=keys, how="left")
        df[count_col] = df[count_col].fillna(0).astype("float32")
        rate_col = f"{prefix}_pit_rate_lag{lag}"
        df[rate_col] = (df[count_col] / df[denom_col].clip(lower=1)).astype("float32")

    df[f"{prefix}_count_intensity"] = sum(
        DECAY_WEIGHTS[lag] * df[f"{prefix}_pits_lag{lag}"] for lag in LAGS
    ).astype("float32")
    df[f"{prefix}_intensity"] = sum(
        DECAY_WEIGHTS[lag] * df[f"{prefix}_pit_rate_lag{lag}"] for lag in LAGS
    ).astype("float32")
    return df


def add_local_position_counts(df):
    keys = RACE_COLS + ["LapNumber", "Position"]
    pos_agg = (
        df.groupby(keys, observed=True)["PitStop"].sum().reset_index(name="pit_count")
    )

    for lag in LAGS:
        shifted = pos_agg.copy()
        shifted["LapNumber"] = shifted["LapNumber"] + lag
        frames = []
        for offset in range(-2, 3):
            tmp = shifted.copy()
            tmp["Position"] = tmp["Position"] + offset
            frames.append(tmp)
        neigh = (
            pd.concat(frames, ignore_index=True)
            .groupby(keys, observed=True)["pit_count"]
            .sum()
            .reset_index()
        )
        count_col = f"local_pos_pits_lag{lag}"
        neigh = neigh.rename(columns={"pit_count": count_col})
        df = df.merge(neigh, on=keys, how="left")
        df[count_col] = df[count_col].fillna(0).astype("float32")
        rate_col = f"local_pos_pit_rate_lag{lag}"
        df[rate_col] = (
            df[count_col] / df["nearby_position_denom"].clip(lower=1)
        ).astype("float32")

    df["local_pos_count_intensity"] = sum(
        DECAY_WEIGHTS[lag] * df[f"local_pos_pits_lag{lag}"] for lag in LAGS
    ).astype("float32")
    df["local_pos_intensity"] = sum(
        DECAY_WEIGHTS[lag] * df[f"local_pos_pit_rate_lag{lag}"] for lag in LAGS
    ).astype("float32")
    return df


def add_driver_cooldown(df):
    df = df.sort_values(RACE_COLS + ["Driver", "LapNumber", ID_COL]).copy()
    driver_keys = RACE_COLS + ["Driver"]

    df["_pit_lap_marker"] = np.where(df["PitStop"].eq(1), df["LapNumber"], np.nan)
    df["_prior_pit_lap_marker"] = df.groupby(driver_keys, observed=True)[
        "_pit_lap_marker"
    ].shift(1)
    df["_last_prior_pit_lap"] = df.groupby(driver_keys, observed=True)[
        "_prior_pit_lap_marker"
    ].ffill()

    df["laps_since_prior_pit"] = (
        (df["LapNumber"] - df["_last_prior_pit_lap"])
        .fillna(99)
        .clip(0, 99)
        .astype("float32")
    )

    for lag in LAGS:
        df[f"own_pit_lag{lag}"] = (
            df.groupby(driver_keys, observed=True)["PitStop"]
            .shift(lag)
            .fillna(0)
            .astype("float32")
        )

    df["own_prior_pit_intensity"] = sum(
        DECAY_WEIGHTS[lag] * df[f"own_pit_lag{lag}"] for lag in LAGS
    ).astype("float32")
    df["cooldown_shrink"] = (1.0 / (1.0 + df["laps_since_prior_pit"])).astype("float32")

    df = df.drop(
        columns=["_pit_lap_marker", "_prior_pit_lap_marker", "_last_prior_pit_lap"]
    )
    return df.sort_index()


def engineer_features(df):
    df = df.copy()
    df["PitStop"] = df["PitStop"].astype("int8")
    df["LapNumber"] = df["LapNumber"].astype("int16")
    df["Position"] = df["Position"].astype("int16")
    df["Year"] = df["Year"].astype("int16")

    df["race_max_lap"] = (
        df.groupby(RACE_COLS, observed=True)["LapNumber"]
        .transform("max")
        .astype("float32")
    )
    df["laps_remaining"] = (
        (df["race_max_lap"] - df["LapNumber"]).clip(lower=0).astype("float32")
    )
    df["lap_frac"] = (df["LapNumber"] / df["race_max_lap"].clip(lower=1)).astype(
        "float32"
    )
    df["tyre_life_frac"] = (df["TyreLife"] / df["race_max_lap"].clip(lower=1)).astype(
        "float32"
    )
    df["degradation_per_tyre_lap"] = (
        df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1)
    ).astype("float32")
    df["lap_delta_per_tyre_lap"] = (
        df["LapTime (s)"].sub(
            df["LapTime (s)"]
            .groupby([df["Year"], df["Race"], df["LapNumber"]], observed=True)
            .transform("median")
        )
        / df["TyreLife"].clip(lower=1)
    ).astype("float32")

    df["field_size"] = (
        df.groupby(RACE_COLS + ["LapNumber"], observed=True)["Driver"]
        .transform("nunique")
        .astype("float32")
    )
    df["compound_peer_count"] = (
        df.groupby(RACE_COLS + ["LapNumber", "Compound"], observed=True)["Driver"]
        .transform("nunique")
        .astype("float32")
    )
    df["strategy_peer_count"] = (
        df.groupby(RACE_COLS + ["LapNumber", "Compound", "Stint"], observed=True)[
            "Driver"
        ]
        .transform("nunique")
        .astype("float32")
    )
    df["nearby_position_denom"] = np.minimum(df["field_size"], 5).astype("float32")

    df = add_driver_cooldown(df)
    df = add_shifted_pit_counts(df, [], "race", "field_size")
    df = add_shifted_pit_counts(
        df, ["Compound"], "same_compound", "compound_peer_count"
    )
    df = add_shifted_pit_counts(
        df, ["Compound", "Stint"], "same_strategy", "strategy_peer_count"
    )
    df = add_local_position_counts(df)

    intensity_cols = [
        "race_intensity",
        "local_pos_intensity",
        "same_compound_intensity",
        "same_strategy_intensity",
    ]
    df["combined_pit_wave_intensity"] = df[intensity_cols].sum(axis=1).astype("float32")
    df["max_pit_wave_intensity"] = df[intensity_cols].max(axis=1).astype("float32")

    for col in intensity_cols + ["combined_pit_wave_intensity"]:
        df[f"{col}_x_tyre_life"] = (df[col] * df["TyreLife"]).astype("float32")
        df[f"{col}_x_laps_remaining"] = (df[col] * df["laps_remaining"]).astype(
            "float32"
        )
        df[f"{col}_x_prior_cooldown"] = (df[col] * df["cooldown_shrink"]).astype(
            "float32"
        )
        df[f"{col}_x_own_recent_pit"] = (
            df[col] * df["own_prior_pit_intensity"]
        ).astype("float32")

    df["pit_wave_pressure"] = (
        df["combined_pit_wave_intensity"]
        * (df["TyreLife"] + 1.0)
        / (df["laps_remaining"] + 1.0)
    ).astype("float32")
    df["local_cover_pressure"] = (
        df["local_pos_intensity"] * (21.0 - df["Position"]) / 20.0
    ).astype("float32")

    for c in ["Compound", "Driver", "Race"]:
        df[c] = df[c].astype("category")

    return df.replace([np.inf, -np.inf], np.nan)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train["__is_train"] = 1
test["__is_train"] = 0
test[TARGET] = np.nan

all_df = pd.concat([train, test], ignore_index=True, sort=False)
all_df = engineer_features(all_df)

train_fe = all_df[all_df["__is_train"].eq(1)].copy()
test_fe = all_df[all_df["__is_train"].eq(0)].copy()

exclude = {ID_COL, TARGET, "__is_train"}
feature_cols = [c for c in train_fe.columns if c not in exclude]
rename_map = make_unique_safe_names(feature_cols)

X_all = all_df[feature_cols].rename(columns=rename_map)
cat_features = [
    rename_map[c] for c in ["Compound", "Driver", "Race"] if c in rename_map
]

X_train = X_all.iloc[: len(train_fe)].copy()
X_test = X_all.iloc[len(train_fe) :].copy()
y = train_fe[TARGET].astype(int).values
groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)

for c in cat_features:
    X_train[c] = X_train[c].astype("category")
    X_test[c] = X_test[c].astype("category")

pos = max(float(y.sum()), 1.0)
neg = float(len(y) - y.sum())
scale_pos_weight = min(30.0, max(1.0, neg / pos))

base_params = dict(
    objective="binary",
    learning_rate=0.035,
    n_estimators=2200,
    num_leaves=96,
    min_child_samples=90,
    subsample=0.88,
    colsample_bytree=0.86,
    reg_alpha=0.05,
    reg_lambda=3.0,
    scale_pos_weight=scale_pos_weight,
    random_state=592,
    n_jobs=max(1, os.cpu_count() or 1),
    verbosity=-1,
    deterministic=True,
    force_col_wise=True,
)

if HAS_SGK:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=592)
    splits = list(cv.split(X_train, y, groups))
else:
    cv = GroupKFold(n_splits=5)
    splits = list(cv.split(X_train, y, groups))

oof = np.zeros(len(X_train), dtype=np.float32)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = LGBMClassifier(**base_params)
    model.fit(
        X_train.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X_train.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )
    pred = model.predict_proba(X_train.iloc[va_idx])[:, 1]
    oof[va_idx] = pred.astype(np.float32)
    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))
    best_iterations.append(
        int(getattr(model, "best_iteration_", None) or base_params["n_estimators"])
    )
    print(f"fold {fold} roc_auc: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold CV ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(y), dtype=np.int32),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_estimators = int(
    np.clip(np.mean(best_iterations) * 1.10, 200, base_params["n_estimators"])
)
final_params = dict(base_params)
final_params["n_estimators"] = final_estimators

final_model = LGBMClassifier(**final_params)
final_model.fit(
    X_train,
    y,
    categorical_feature=cat_features,
)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

pred_by_id = pd.Series(test_pred, index=test_fe[ID_COL].values)
submission = sample[[ID_COL]].copy()
submission[TARGET] = (
    pred_by_id.reindex(submission[ID_COL]).fillna(float(np.mean(oof))).values
)

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": fold_scores,
            "final_n_estimators": final_estimators,
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
            "research_hypotheses_llm_claimed_used": ["000592"],
        }
    )
)
