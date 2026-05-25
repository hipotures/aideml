import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

SEED = 42
TARGET = "PitNextLap"
ID_COL = "id"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def rec(
    a_stops, a_first, a_second, a_trans, b_stops, b_first, b_second, b_trans, source
):
    return {
        "plan_a_stop_count": a_stops,
        "plan_a_first": a_first,
        "plan_a_second": a_second,
        "plan_a_transition": a_trans,
        "plan_b_stop_count": b_stops,
        "plan_b_first": b_first,
        "plan_b_second": b_second,
        "plan_b_transition": b_trans,
        "strategy_source_tag": source,
    }


RACE_PROFILES = {
    "Bahrain Grand Prix": rec(
        2,
        (13, 19),
        (34, 42),
        "SOFT>HARD>HARD",
        1,
        (24, 32),
        None,
        "MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Saudi Arabian Grand Prix": rec(
        1,
        (18, 25),
        None,
        "MEDIUM>HARD",
        2,
        (12, 18),
        (34, 42),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Australian Grand Prix": rec(
        1,
        (17, 25),
        None,
        "MEDIUM>HARD",
        2,
        (11, 18),
        (34, 43),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Azerbaijan Grand Prix": rec(
        1,
        (14, 22),
        None,
        "MEDIUM>HARD",
        2,
        (10, 16),
        (30, 38),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Miami Grand Prix": rec(
        1,
        (20, 29),
        None,
        "MEDIUM>HARD",
        2,
        (12, 18),
        (36, 45),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Emilia Romagna Grand Prix": rec(
        1,
        (24, 30),
        None,
        "MEDIUM>HARD",
        2,
        (14, 20),
        (38, 45),
        "SOFT>MEDIUM>HARD",
        "autohebdo_2025_imola",
    ),
    "Monaco Grand Prix": rec(
        1,
        (18, 34),
        None,
        "MEDIUM>HARD",
        1,
        (1, 12),
        None,
        "HARD>MEDIUM",
        "public_pirelli_profile",
    ),
    "Spanish Grand Prix": rec(
        2,
        (13, 20),
        (38, 47),
        "SOFT>MEDIUM>HARD",
        1,
        (25, 34),
        None,
        "MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Canadian Grand Prix": rec(
        1,
        (19, 29),
        None,
        "MEDIUM>HARD",
        2,
        (12, 18),
        (36, 45),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Austrian Grand Prix": rec(
        2,
        (14, 21),
        (38, 47),
        "MEDIUM>HARD>HARD",
        1,
        (24, 33),
        None,
        "MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "British Grand Prix": rec(
        1,
        (19, 27),
        None,
        "MEDIUM>HARD",
        2,
        (13, 20),
        (36, 45),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Hungarian Grand Prix": rec(
        2,
        (16, 23),
        (42, 51),
        "MEDIUM>HARD>HARD",
        1,
        (28, 37),
        None,
        "MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Belgian Grand Prix": rec(
        1,
        (16, 24),
        None,
        "MEDIUM>HARD",
        2,
        (11, 17),
        (31, 39),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Dutch Grand Prix": rec(
        1,
        (22, 31),
        None,
        "MEDIUM>HARD",
        2,
        (13, 19),
        (39, 48),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Italian Grand Prix": rec(
        1,
        (19, 26),
        None,
        "MEDIUM>HARD",
        2,
        (12, 18),
        (36, 44),
        "SOFT>MEDIUM>HARD",
        "si_2025_italy",
    ),
    "Singapore Grand Prix": rec(
        1,
        (18, 29),
        None,
        "MEDIUM>HARD",
        2,
        (13, 20),
        (37, 46),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Japanese Grand Prix": rec(
        2,
        (14, 21),
        (35, 44),
        "MEDIUM>HARD>HARD",
        1,
        (23, 32),
        None,
        "MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Qatar Grand Prix": rec(
        2,
        (16, 23),
        (36, 45),
        "MEDIUM>HARD>HARD",
        3,
        (13, 18),
        (31, 37),
        "SOFT>MEDIUM>HARD>HARD",
        "public_pirelli_profile",
    ),
    "United States Grand Prix": rec(
        2,
        (13, 19),
        (35, 43),
        "MEDIUM>HARD>HARD",
        1,
        (25, 34),
        None,
        "MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Mexico City Grand Prix": rec(
        1,
        (28, 35),
        None,
        "MEDIUM>HARD",
        2,
        (18, 24),
        (45, 52),
        "SOFT>MEDIUM>HARD",
        "news24_2023_mexico",
    ),
    "Sao Paulo Grand Prix": rec(
        2,
        (16, 23),
        (42, 51),
        "SOFT>MEDIUM>HARD",
        1,
        (26, 35),
        None,
        "MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Las Vegas Grand Prix": rec(
        1,
        (18, 27),
        None,
        "MEDIUM>HARD",
        2,
        (10, 16),
        (32, 41),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Abu Dhabi Grand Prix": rec(
        1,
        (18, 25),
        None,
        "MEDIUM>HARD",
        2,
        (14, 20),
        (37, 46),
        "SOFT>MEDIUM>HARD",
        "public_pirelli_profile",
    ),
    "Chinese Grand Prix": rec(
        2,
        (12, 18),
        (33, 42),
        "MEDIUM>HARD>HARD",
        1,
        (23, 31),
        None,
        "MEDIUM>HARD",
        "public_pirelli_profile",
    ),
}

SPECIAL_OVERRIDES = {
    (2025, "Italian Grand Prix"): rec(
        1,
        (19, 25),
        None,
        "MEDIUM>HARD",
        2,
        (12, 18),
        (36, 44),
        "SOFT>MEDIUM>HARD",
        "si_2025_italy",
    ),
    (2025, "Emilia Romagna Grand Prix"): rec(
        1,
        (24, 30),
        None,
        "MEDIUM>HARD",
        2,
        (14, 20),
        (38, 45),
        "SOFT>MEDIUM>HARD",
        "autohebdo_2025_imola",
    ),
    (2023, "Mexico City Grand Prix"): rec(
        1,
        (28, 34),
        None,
        "MEDIUM>HARD",
        2,
        (18, 24),
        (45, 52),
        "SOFT>MEDIUM>HARD",
        "news24_2023_mexico",
    ),
}


def split_window(prefix, window):
    if window is None:
        return {f"{prefix}_start": np.nan, f"{prefix}_end": np.nan}
    return {f"{prefix}_start": float(window[0]), f"{prefix}_end": float(window[1])}


def generic_prior(total_laps):
    total_laps = int(np.clip(round(total_laps), 20, 90))
    return rec(
        1,
        (max(2, round(0.34 * total_laps)), max(3, round(0.46 * total_laps))),
        None,
        "MEDIUM>HARD",
        2,
        (max(2, round(0.22 * total_laps)), max(3, round(0.32 * total_laps))),
        (max(4, round(0.58 * total_laps)), max(5, round(0.70 * total_laps))),
        "SOFT>MEDIUM>HARD",
        "generic_public_strategy_window",
    )


def make_prior_table(all_data):
    tmp = all_data[[ID_COL, "Year", "Race", "LapNumber", "RaceProgress"]].copy()
    tmp["_total_lap_est"] = tmp["LapNumber"] / tmp["RaceProgress"].replace(0, np.nan)
    total_laps = (
        tmp.groupby(["Year", "Race"])["_total_lap_est"]
        .median()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(tmp["LapNumber"].max())
        .clip(20, 90)
    )

    rows = []
    pairs = all_data[["Year", "Race"]].drop_duplicates()
    for _, pair in pairs.iterrows():
        year = int(pair["Year"])
        race = pair["Race"]
        tlaps = (
            total_laps.loc[(year, race)]
            if (year, race) in total_laps.index
            else all_data["LapNumber"].max()
        )
        base = SPECIAL_OVERRIDES.get(
            (year, race), RACE_PROFILES.get(race, generic_prior(tlaps))
        ).copy()

        row = {"Year": year, "Race": race}
        for k in [
            "plan_a_stop_count",
            "plan_b_stop_count",
            "plan_a_transition",
            "plan_b_transition",
            "strategy_source_tag",
        ]:
            row[k] = base[k]
        row.update(split_window("plan_a_first", base["plan_a_first"]))
        row.update(split_window("plan_a_second", base["plan_a_second"]))
        row.update(split_window("plan_b_first", base["plan_b_first"]))
        row.update(split_window("plan_b_second", base["plan_b_second"]))
        rows.append(row)

    return pd.DataFrame(rows)


def add_strategy_features(train_df, test_df):
    train_part = train_df.copy()
    test_part = test_df.copy()
    test_part[TARGET] = np.nan
    all_data = pd.concat([train_part, test_part], axis=0, ignore_index=True)

    priors = make_prior_table(all_data)
    all_data = all_data.merge(priors, on=["Year", "Race"], how="left")

    lap = all_data["LapNumber"].astype(float)

    def dist_to_window(start_col, end_col):
        start = all_data[start_col].astype(float)
        end = all_data[end_col].astype(float)
        valid = start.notna() & end.notna()
        return np.where(
            valid,
            np.where(lap < start, start - lap, np.where(lap > end, lap - end, 0.0)),
            99.0,
        )

    distances = np.vstack(
        [
            dist_to_window("plan_a_first_start", "plan_a_first_end"),
            dist_to_window("plan_a_second_start", "plan_a_second_end"),
            dist_to_window("plan_b_first_start", "plan_b_first_end"),
            dist_to_window("plan_b_second_start", "plan_b_second_end"),
        ]
    )
    all_data["distance_to_nearest_strategy_window"] = np.min(distances, axis=0)

    inside_a_first = (lap >= all_data["plan_a_first_start"]) & (
        lap <= all_data["plan_a_first_end"]
    )
    inside_a_second = (lap >= all_data["plan_a_second_start"]) & (
        lap <= all_data["plan_a_second_end"]
    )
    inside_b_first = (lap >= all_data["plan_b_first_start"]) & (
        lap <= all_data["plan_b_first_end"]
    )
    inside_b_second = (lap >= all_data["plan_b_second_start"]) & (
        lap <= all_data["plan_b_second_end"]
    )

    all_data["inside_plan_a_window"] = (inside_a_first | inside_a_second).astype(
        np.int8
    )
    all_data["inside_plan_b_window"] = (inside_b_first | inside_b_second).astype(
        np.int8
    )

    completed_stops = np.maximum(all_data["Stint"].astype(float) - 1.0, 0.0)
    plan_a_due = (lap >= all_data["plan_a_first_start"].fillna(999)).astype(float) + (
        lap >= all_data["plan_a_second_start"].fillna(999)
    ).astype(float)
    plan_a_due = np.minimum(plan_a_due, all_data["plan_a_stop_count"].astype(float))
    remaining_plan_a = np.maximum(
        all_data["plan_a_stop_count"].astype(float) - completed_stops, 0.0
    )
    all_data["plan_stop_count_pressure"] = np.maximum(
        plan_a_due - completed_stops, 0.0
    ) + remaining_plan_a * all_data["RaceProgress"].astype(float)

    all_data["planned_stops_remaining"] = remaining_plan_a
    all_data["completed_stops_from_stint"] = completed_stops
    all_data["compound_matches_plan_a_start"] = (
        all_data["Compound"].astype(str)
        == all_data["plan_a_transition"].astype(str).str.split(">").str[0]
    ).astype(np.int8)
    all_data["compound_matches_plan_b_start"] = (
        all_data["Compound"].astype(str)
        == all_data["plan_b_transition"].astype(str).str.split(">").str[0]
    ).astype(np.int8)

    return all_data.iloc[: len(train_df)].copy(), all_data.iloc[len(train_df) :].copy()


train_fe, test_fe = add_strategy_features(train, test)

feature_cols = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]


def safe_name(name):
    out = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    return out if out else "feature"


name_map = {}
used = set()
for col in feature_cols:
    base = safe_name(col)
    new = base
    i = 1
    while new in used:
        i += 1
        new = f"{base}_{i}"
    used.add(new)
    name_map[col] = new

X_all = pd.concat(
    [train_fe[feature_cols], test_fe[feature_cols]], axis=0, ignore_index=True
).rename(columns=name_map)
cat_cols = [
    name_map[c]
    for c in feature_cols
    if train_fe[c].dtype == "object" or str(train_fe[c].dtype) == "category"
]

for col in cat_cols:
    X_all[col] = X_all[col].astype("category")

X_all = X_all.replace([np.inf, -np.inf], np.nan)
X = X_all.iloc[: len(train)].copy()
X_test = X_all.iloc[len(train) :].copy()
y = train[TARGET].astype(int).values
groups = train["Year"].astype(str) + "|" + train["Race"].astype(str)

try:
    import lightgbm as lgb
except Exception as exc:
    raise RuntimeError(
        "This solution requires lightgbm, which is listed as an installed package."
    ) from exc

try:
    from sklearn.model_selection import StratifiedGroupKFold

    if groups.nunique() >= 5:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X, y, groups))
        split_name = "StratifiedGroupKFold"
    else:
        raise ValueError("Not enough groups")
except Exception:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y))
    split_name = "StratifiedKFold"

pos = max(int(y.sum()), 1)
neg = len(y) - pos
params = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 900,
    "learning_rate": 0.035,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_lambda": 4.0,
    "scale_pos_weight": neg / pos,
    "random_state": SEED,
    "n_jobs": min(8, os.cpu_count() or 1),
    "verbosity": -1,
}

oof = np.zeros(len(train), dtype=float)
fold_scores = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    pred = model.predict_proba(X.iloc[va_idx])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(auc)
    best_iterations.append(
        model.best_iteration_ if model.best_iteration_ else params["n_estimators"]
    )
    print(f"Fold {fold} {split_name} ROC AUC: {auc:.6f}")

oof_auc = roc_auc_score(y, oof)
mean_auc = float(np.mean(fold_scores))
std_auc = float(np.std(fold_scores))
print(f"OOF ROC AUC: {oof_auc:.6f}")
print(f"Mean fold ROC AUC: {mean_auc:.6f} +/- {std_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_params = params.copy()
final_params["n_estimators"] = int(
    np.clip(round(np.mean(best_iterations)), 100, params["n_estimators"])
)
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "research_hypotheses_llm_claimed_used": ["000504"],
    "validation": split_name,
    "metric": "roc_auc",
    "oof_roc_auc": float(oof_auc),
    "mean_fold_roc_auc": mean_auc,
    "std_fold_roc_auc": std_auc,
    "n_folds": len(splits),
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
with open(os.path.join(WORKING_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review, sort_keys=True))
