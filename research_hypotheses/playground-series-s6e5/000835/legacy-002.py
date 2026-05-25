import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 2026
N_SPLITS = 5
EMBARGO_LAPS = 2
SURVIVAL_MARGIN = 1e-5


def clean_col(c):
    c = re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_")
    return c if c else "col"


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.rename(columns={c: clean_col(c) for c in train.columns})
test = test.rename(columns={c: clean_col(c) for c in test.columns})
sample = sample.rename(columns={c: clean_col(c) for c in sample.columns})

TARGET = clean_col(TARGET)
ID_COL = clean_col(ID_COL)

y = train[TARGET].astype(int).reset_index(drop=True)
n_train = len(train)

base_train = train.drop(columns=[TARGET])
all_df = pd.concat([base_train, test], axis=0, ignore_index=True)


def add_features(df):
    df = df.copy()

    df["EstimatedRaceLaps"] = df["LapNumber"] / df["RaceProgress"].clip(lower=0.01)
    df["LapsRemainingEst"] = df["EstimatedRaceLaps"] - df["LapNumber"]
    df["TyreLifeRaceFrac"] = df["TyreLife"] / df["EstimatedRaceLaps"].clip(lower=1.0)
    df["DegradationPerTyreLap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(
        lower=1.0
    )
    df["LapDeltaAbs"] = df["LapTime_Delta"].abs()
    df["PositionChangeAbs"] = df["Position_Change"].abs()
    df["TyreLife_x_Progress"] = df["TyreLife"] * df["RaceProgress"]
    df["IsWetOrIntermediate"] = (
        df["Compound"].isin(["WET", "INTERMEDIATE"]).astype(np.int8)
    )

    year = df["Year"].astype(str)
    race = df["Race"].astype(str)
    driver = df["Driver"].astype(str)
    compound = df["Compound"].astype(str)

    df["YearRace"] = year + "_" + race
    df["YearRaceDriver"] = year + "_" + race + "_" + driver
    df["RaceCompound"] = race + "_" + compound
    df["DriverCompound"] = driver + "_" + compound

    sort_cols = ["Year", "Race", "Driver", "LapNumber", ID_COL]
    ordered = df.sort_values(sort_cols).copy()
    g = ordered.groupby(["Year", "Race", "Driver"], sort=False)

    for col in [
        "LapTime_s",
        "LapTime_Delta",
        "Position",
        "Position_Change",
        "TyreLife",
        "Cumulative_Degradation",
        "PitStop",
    ]:
        prev = g[col].shift(1)
        ordered[f"prev_{col}"] = prev
        ordered[f"diff_prev_{col}"] = ordered[col] - prev

    ordered["prev3_laptime_mean"] = g["LapTime_s"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    ordered["prev3_delta_mean"] = g["LapTime_Delta"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    ordered["prev3_position_mean"] = g["Position"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    return ordered.sort_index()


all_df = add_features(all_df)

cat_cols_all = []
cat_names = {
    "Year",
    "Compound",
    "Race",
    "Driver",
    "YearRace",
    "YearRaceDriver",
    "RaceCompound",
    "DriverCompound",
}

for c in all_df.columns:
    if all_df[c].dtype == "object" or c in cat_names:
        all_df[c] = all_df[c].astype("string").fillna("__NA__").astype("category")
        cat_cols_all.append(c)

all_df = all_df.replace([np.inf, -np.inf], np.nan)

train_X = all_df.iloc[:n_train].reset_index(drop=True)
test_X = all_df.iloc[n_train:].reset_index(drop=True)

raw_features = [c for c in base_train.columns if c != ID_COL]
extra_features = [c for c in train_X.columns if c not in set(raw_features + [ID_COL])]

feature_sets = {
    "raw": raw_features,
    "timeline_candidate": raw_features + extra_features,
}

meta = pd.DataFrame(
    {
        "group": train_X["Year"].astype(str) + "__" + train_X["Race"].astype(str),
        "timeline": train_X["Year"].astype(str)
        + "__"
        + train_X["Race"].astype(str)
        + "__"
        + train_X["Driver"].astype(str),
        "lap": train_X["LapNumber"].astype(float),
    }
)


def year_race_group_splits(meta_df, y_values, n_splits=5):
    groups = meta_df["group"]
    splitter = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    idx = np.arange(len(meta_df))
    for fold, (tr_idx, va_idx) in enumerate(splitter.split(idx, y_values, groups), 1):
        yield fold, tr_idx, va_idx, 0


def timeline_block_embargo_splits(meta_df, n_splits=5, embargo_laps=2):
    idx = np.arange(len(meta_df))
    fold_id = np.full(len(meta_df), -1, dtype=np.int16)

    ordered = meta_df[["timeline", "lap"]].copy()
    ordered["_row"] = idx
    ordered = ordered.sort_values(["timeline", "lap", "_row"])

    for _, part in ordered.groupby("timeline", sort=False):
        laps = np.sort(part["lap"].unique())
        for fold, chunk in enumerate(np.array_split(laps, n_splits)):
            if len(chunk) == 0:
                continue
            rows = part.loc[part["lap"].isin(chunk), "_row"].to_numpy()
            fold_id[rows] = fold

    if (fold_id < 0).any():
        raise RuntimeError("Some rows were not assigned to an embargo validation fold.")

    for fold in range(n_splits):
        va_idx = idx[fold_id == fold]
        train_before = np.ones(len(meta_df), dtype=bool)
        train_before[va_idx] = False

        val_ranges = meta_df.iloc[va_idx].groupby("timeline")["lap"].agg(["min", "max"])
        joined = meta_df[["timeline", "lap"]].join(val_ranges, on="timeline")

        near_validation_block = (
            joined["min"].notna()
            & (joined["lap"] >= joined["min"] - embargo_laps)
            & (joined["lap"] <= joined["max"] + embargo_laps)
        ).to_numpy()

        train_mask = train_before & ~near_validation_block
        tr_idx = idx[train_mask]
        purged_count = int((train_before & near_validation_block).sum())
        yield fold + 1, tr_idx, va_idx, purged_count


splits_by_scheme = {
    "race_group": list(year_race_group_splits(meta, y, N_SPLITS)),
    "timeline_embargo": list(
        timeline_block_embargo_splits(meta, N_SPLITS, EMBARGO_LAPS)
    ),
}

pos = int(y.sum())
neg = int(len(y) - pos)
scale_pos_weight = float(neg / max(pos, 1))

base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=1200,
    learning_rate=0.04,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    n_jobs=max(1, (os.cpu_count() or 2) - 1),
    verbosity=-1,
)


def evaluate_feature_set(scheme_name, splits, feature_name, features):
    oof = np.zeros(len(y), dtype=np.float32)
    aucs, best_iters, purged_counts = [], [], []
    cat_features = [c for c in features if c in cat_cols_all]

    for fold, tr_idx, va_idx, purged_count in splits:
        model = LGBMClassifier(
            **{**base_params, "random_state": SEED + 100 * fold + len(features)}
        )
        model.fit(
            train_X.loc[tr_idx, features],
            y.iloc[tr_idx],
            eval_set=[(train_X.loc[va_idx, features], y.iloc[va_idx])],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
        )

        pred = model.predict_proba(train_X.loc[va_idx, features])[:, 1]
        oof[va_idx] = pred.astype(np.float32)

        auc = roc_auc_score(y.iloc[va_idx], pred)
        best_iter = int(model.best_iteration_ or base_params["n_estimators"])

        aucs.append(float(auc))
        best_iters.append(best_iter)
        purged_counts.append(int(purged_count))

        print(
            f"{scheme_name}/{feature_name} fold {fold}: "
            f"auc={auc:.6f}, purged_rows={purged_count}, best_iter={best_iter}"
        )

    mean_auc = roc_auc_score(y, oof)
    print(f"{scheme_name}/{feature_name} OOF ROC AUC: {mean_auc:.6f}")

    return {
        "oof": oof,
        "fold_auc": aucs,
        "mean_auc": float(mean_auc),
        "best_iters": best_iters,
        "purged_rows": purged_counts,
    }


results = {}
for scheme_name, splits in splits_by_scheme.items():
    results[scheme_name] = {}
    for feature_name, features in feature_sets.items():
        results[scheme_name][feature_name] = evaluate_feature_set(
            scheme_name, splits, feature_name, features
        )

race_gain = (
    results["race_group"]["timeline_candidate"]["mean_auc"]
    - results["race_group"]["raw"]["mean_auc"]
)
embargo_gain = (
    results["timeline_embargo"]["timeline_candidate"]["mean_auc"]
    - results["timeline_embargo"]["raw"]["mean_auc"]
)

selected_name = (
    "timeline_candidate"
    if race_gain > SURVIVAL_MARGIN and embargo_gain > SURVIVAL_MARGIN
    else "raw"
)

selected_race = results["race_group"][selected_name]
selected_embargo = results["timeline_embargo"][selected_name]
features = feature_sets[selected_name]
cat_features = [c for c in features if c in cat_cols_all]

pd.DataFrame(
    {
        "row": np.arange(len(y)),
        "target": y.astype(int),
        "prediction": selected_race["oof"],
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_iters = int(
    np.clip(
        np.mean(selected_race["best_iters"]) * 1.10, 100, base_params["n_estimators"]
    )
)

final_model = LGBMClassifier(
    **{**base_params, "n_estimators": final_iters, "random_state": SEED + 999}
)
final_model.fit(
    train_X[features],
    y,
    categorical_feature=cat_features,
)

test_pred = final_model.predict_proba(test_X[features])[:, 1]
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = sample.copy()
target_col = [c for c in submission.columns if c != ID_COL][0]
submission[target_col] = test_pred

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "research_hypotheses_llm_claimed_used": ["000835"],
    "validation_metric": "roc_auc",
    "cv_strategy": (
        f"{N_SPLITS}-fold Year+Race GroupKFold benchmark plus "
        f"{N_SPLITS}-fold contiguous Year+Race+Driver lap-block split "
        f"with {EMBARGO_LAPS}-lap embargo"
    ),
    "race_group_raw_auc": float(results["race_group"]["raw"]["mean_auc"]),
    "race_group_timeline_candidate_auc": float(
        results["race_group"]["timeline_candidate"]["mean_auc"]
    ),
    "timeline_embargo_raw_auc": float(results["timeline_embargo"]["raw"]["mean_auc"]),
    "timeline_embargo_timeline_candidate_auc": float(
        results["timeline_embargo"]["timeline_candidate"]["mean_auc"]
    ),
    "race_group_timeline_gain": float(race_gain),
    "timeline_embargo_timeline_gain": float(embargo_gain),
    "selected_feature_set": selected_name,
    "selected_race_group_auc": float(selected_race["mean_auc"]),
    "selected_timeline_embargo_auc": float(selected_embargo["mean_auc"]),
    "timeline_embargo_mean_purged_rows": float(
        np.mean(results["timeline_embargo"][selected_name]["purged_rows"])
    ),
    "final_n_estimators": int(final_iters),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}

print(f"Selected race_group OOF ROC AUC: {selected_race['mean_auc']:.6f}")
print(f"Selected timeline_embargo OOF ROC AUC: {selected_embargo['mean_auc']:.6f}")
print(json.dumps(review, indent=2, sort_keys=True))
