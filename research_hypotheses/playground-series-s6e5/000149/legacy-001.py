import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.rename(columns={"LapTime (s)": "LapTime_s"})
test = test.rename(columns={"LapTime (s)": "LapTime_s"})
y = train["PitNextLap"].astype(int).to_numpy()


def build_base_features(train_df, test_df):
    all_df = pd.concat(
        [
            train_df.drop(columns=["PitNextLap"]).assign(_dataset="train"),
            test_df.assign(_dataset="test"),
        ],
        ignore_index=True,
    )
    all_df["_orig_order"] = np.arange(len(all_df))
    all_df["RaceYear"] = all_df["Year"].astype(str) + "_" + all_df["Race"].astype(str)

    compound_rank = {
        "SOFT": 0.0,
        "MEDIUM": 1.0,
        "HARD": 2.0,
        "INTERMEDIATE": 1.0,
        "WET": 1.0,
    }
    all_df["compound_hardness"] = (
        all_df["Compound"].map(compound_rank).fillna(1.0).astype("float32")
    )

    lap_keys = ["Year", "Race", "LapNumber"]
    all_df["lap_delta_pct"] = (
        all_df.groupby(lap_keys)["LapTime_Delta"].rank(pct=True).astype("float32")
    )
    all_df["lap_degradation_pct"] = (
        all_df.groupby(lap_keys)["Cumulative_Degradation"]
        .rank(pct=True)
        .astype("float32")
    )
    all_df["lap_tyrelife_pct"] = (
        all_df.groupby(lap_keys)["TyreLife"].rank(pct=True).astype("float32")
    )
    all_df["position_pct_lap"] = (
        all_df.groupby(lap_keys)["Position"].rank(pct=True).astype("float32")
    )

    s = all_df.sort_values(lap_keys + ["Position", "id"]).copy()
    g = s.groupby(lap_keys, sort=False)

    for c in [
        "_near_count",
        "_local_pressure_sum",
        "_defend_behind_sum",
        "_attack_ahead_sum",
        "_nearby_older",
        "_nearby_newer",
        "_nearby_harder",
        "_nearby_softer",
    ]:
        s[c] = 0.0

    own_delta_pct = s["lap_delta_pct"].astype(float)
    own_tyre = s["TyreLife"].astype(float)
    own_hard = s["compound_hardness"].astype(float)

    for offset in [1, 2, -1, -2]:
        n_delta_pct = g["lap_delta_pct"].shift(offset).astype(float)
        n_tyre = g["TyreLife"].shift(offset).astype(float)
        n_hard = g["compound_hardness"].shift(offset).astype(float)
        valid = n_delta_pct.notna()

        s["_near_count"] += valid.astype("float32")
        s["_local_pressure_sum"] += (
            (own_delta_pct - n_delta_pct).clip(lower=0).fillna(0).astype("float32")
        )

        if offset < 0:
            s["_defend_behind_sum"] += (
                (own_delta_pct - n_delta_pct).clip(lower=0).fillna(0).astype("float32")
            )
        else:
            s["_attack_ahead_sum"] += (
                (n_delta_pct - own_delta_pct).clip(lower=0).fillna(0).astype("float32")
            )

        s["_nearby_older"] += ((n_tyre > own_tyre + 1.0) & valid).astype("float32")
        s["_nearby_newer"] += ((n_tyre + 1.0 < own_tyre) & valid).astype("float32")
        s["_nearby_harder"] += ((n_hard > own_hard + 0.1) & valid).astype("float32")
        s["_nearby_softer"] += ((n_hard + 0.1 < own_hard) & valid).astype("float32")

    denom = s["_near_count"].replace(0, np.nan)
    s["local_pressure"] = (s["_local_pressure_sum"] / denom).fillna(0).astype("float32")
    s["defend_behind_pressure"] = (
        (s["_defend_behind_sum"] / 2.0).fillna(0).clip(0, 1).astype("float32")
    )
    s["attack_ahead_pressure"] = (
        (s["_attack_ahead_sum"] / 2.0).fillna(0).clip(0, 1).astype("float32")
    )
    s["nearby_older_tyre_count"] = s["_nearby_older"].astype("float32")
    s["nearby_newer_tyre_count"] = s["_nearby_newer"].astype("float32")
    s["nearby_harder_compound_count"] = s["_nearby_harder"].astype("float32")
    s["nearby_softer_compound_count"] = s["_nearby_softer"].astype("float32")
    s["nearby_older_tyre_frac"] = (
        (s["_nearby_older"] / denom).fillna(0).astype("float32")
    )
    s["nearby_newer_tyre_frac"] = (
        (s["_nearby_newer"] / denom).fillna(0).astype("float32")
    )

    drop_tmp = [
        c
        for c in s.columns
        if c.startswith("_") and c not in ["_dataset", "_orig_order"]
    ]
    all_df = s.sort_values("_orig_order").drop(columns=drop_tmp).reset_index(drop=True)

    d = all_df.sort_values(["Year", "Race", "Driver", "LapNumber", "id"]).copy()
    dg = d.groupby(["Year", "Race", "Driver"], sort=False)
    d["prev_pitstop"] = dg["PitStop"].shift(1).fillna(0).astype("float32")
    d["prev_lap_delta_driver"] = dg["LapTime_Delta"].shift(1).astype("float32")
    d["post_stop_pace_response"] = (
        d["LapTime_Delta"] - d["prev_lap_delta_driver"]
    ).astype("float32")
    d["post_stop_warmup_flag"] = (
        (d["prev_pitstop"] > 0) | ((d["TyreLife"] <= 2) & (d["Stint"] > 1))
    ).astype("int8")
    all_df = d.sort_values("_orig_order").reset_index(drop=True)

    for c in ["Compound", "Driver", "Race", "RaceYear"]:
        all_df[c] = all_df[c].astype("category")

    n_train = len(train_df)
    return all_df.iloc[:n_train].reset_index(drop=True), all_df.iloc[
        n_train:
    ].reset_index(drop=True)


def q_bounds(s, lo=0.05, hi=0.95):
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        return (0.0, 1.0)
    a, b = np.quantile(vals, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        a, b = float(vals.median() - 1.0), float(vals.median() + 1.0)
    return float(a), float(b)


def scale_series(s, bounds):
    a, b = bounds
    return ((s.astype(float) - a) / max(b - a, 1e-6)).clip(0, 1).astype("float32")


def fit_strategy_tables(df):
    abs_move = df["Position_Change"].abs()
    race_move = abs_move.groupby(df["Race"], observed=True).median()
    race_difficulty = (1.0 - race_move.rank(pct=True)).clip(0, 1).astype("float32")

    warm_src = df.loc[
        df["post_stop_warmup_flag"].eq(1) & df["post_stop_pace_response"].notna(),
        ["Year", "Race", "Compound", "post_stop_pace_response"],
    ].copy()
    if warm_src.empty:
        warm_src = df.loc[
            df["post_stop_pace_response"].notna(),
            ["Year", "Race", "Compound", "post_stop_pace_response"],
        ].copy()
    if warm_src.empty:
        warm_src = pd.DataFrame(
            {
                "Year": [0],
                "Race": ["__missing__"],
                "Compound": ["MEDIUM"],
                "post_stop_pace_response": [0.0],
            }
        )

    return {
        "race_difficulty": race_difficulty,
        "race_difficulty_default": (
            float(race_difficulty.median()) if len(race_difficulty) else 0.5
        ),
        "deg_bounds": q_bounds(df["Cumulative_Degradation"]),
        "tyre_bounds": q_bounds(df["TyreLife"]),
        "warm_rc": warm_src.groupby(["Year", "Race", "Compound"], observed=True)[
            "post_stop_pace_response"
        ].median(),
        "warm_yc": warm_src.groupby(["Year", "Compound"], observed=True)[
            "post_stop_pace_response"
        ].median(),
        "warm_c": warm_src.groupby(["Compound"], observed=True)[
            "post_stop_pace_response"
        ].median(),
        "warm_global": float(warm_src["post_stop_pace_response"].median()),
        "warm_bounds": q_bounds(warm_src["post_stop_pace_response"], 0.10, 0.90),
    }


def add_strategy_scores(df, tables):
    out = df.copy()

    out = out.join(
        tables["race_difficulty"].rename("race_overtake_difficulty"), on="Race"
    )
    out["race_overtake_difficulty"] = (
        out["race_overtake_difficulty"]
        .fillna(tables["race_difficulty_default"])
        .astype("float32")
    )

    out["degradation_high"] = scale_series(
        out["Cumulative_Degradation"], tables["deg_bounds"]
    )
    out["stint_age_high"] = scale_series(out["TyreLife"], tables["tyre_bounds"])
    out["stint_opportunity_window"] = (
        (
            0.45 * out["stint_age_high"]
            + 0.35 * out["degradation_high"]
            + 0.20 * out["RaceProgress"].clip(0, 1).astype(float)
        )
        .clip(0, 1)
        .astype("float32")
    )

    pressure = (
        0.50 * out["defend_behind_pressure"]
        + 0.30 * out["attack_ahead_pressure"]
        + 0.20 * out["local_pressure"]
    ).clip(0, 1)

    out["undercut_opportunity"] = (
        (
            out["degradation_high"]
            * (0.5 + 0.5 * out["race_overtake_difficulty"])
            * (0.65 * pressure + 0.35 * out["nearby_newer_tyre_frac"])
            * out["stint_opportunity_window"]
        )
        .clip(0, 1)
        .astype("float32")
    )

    out = out.join(
        tables["warm_rc"].rename("_warm_rc"), on=["Year", "Race", "Compound"]
    )
    out = out.join(tables["warm_yc"].rename("_warm_yc"), on=["Year", "Compound"])
    out = out.join(tables["warm_c"].rename("_warm_c"), on=["Compound"])

    out["warmup_response_median"] = (
        out["_warm_rc"]
        .fillna(out["_warm_yc"])
        .fillna(out["_warm_c"])
        .fillna(tables["warm_global"])
        .astype("float32")
    )
    out["warmup_response_norm"] = scale_series(
        out["warmup_response_median"], tables["warm_bounds"]
    )

    out["overcut_regime_score"] = (
        (
            out["warmup_response_norm"]
            * (1.0 - out["lap_delta_pct"].fillna(0.5))
            * out["race_overtake_difficulty"]
            * (0.5 + 0.5 * out["nearby_older_tyre_frac"])
            * out["stint_opportunity_window"]
        )
        .clip(0, 1)
        .astype("float32")
    )

    out["warmup_adjusted_undercut"] = (
        (out["undercut_opportunity"] * (1.0 - out["warmup_response_norm"]))
        .clip(0, 1)
        .astype("float32")
    )

    return out.drop(columns=["_warm_rc", "_warm_yc", "_warm_c"])


train_base, test_base = build_base_features(train, test)

strategy_cols = [
    "race_overtake_difficulty",
    "degradation_high",
    "stint_age_high",
    "stint_opportunity_window",
    "undercut_opportunity",
    "warmup_response_median",
    "warmup_response_norm",
    "overcut_regime_score",
    "warmup_adjusted_undercut",
]

exclude = {"id", "_dataset", "_orig_order"}
base_feature_cols = [c for c in train_base.columns if c not in exclude]
feature_cols = base_feature_cols + strategy_cols
cat_features = [
    c for c in ["Compound", "Driver", "Race", "RaceYear"] if c in feature_cols
]

groups = train_base["RaceYear"].astype(str).to_numpy()
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(train_base), dtype=np.float32)
fold_scores = []
best_iterations = []

base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=700,
    learning_rate=0.055,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    n_jobs=max(1, os.cpu_count() or 1),
    force_col_wise=True,
    verbosity=-1,
)

for fold, (tr_idx, va_idx) in enumerate(cv.split(train_base, y, groups), 1):
    fold_tables = fit_strategy_tables(train_base.iloc[tr_idx])
    tr_df = add_strategy_scores(train_base.iloc[tr_idx], fold_tables)
    va_df = add_strategy_scores(train_base.iloc[va_idx], fold_tables)

    y_tr, y_va = y[tr_idx], y[va_idx]
    params = base_params.copy()
    params["scale_pos_weight"] = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))

    model = LGBMClassifier(**params)
    model.fit(
        tr_df[feature_cols],
        y_tr,
        eval_set=[(va_df[feature_cols], y_va)],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[early_stopping(75, verbose=False), log_evaluation(0)],
    )

    pred = model.predict_proba(va_df[feature_cols])[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y_va, pred)
    fold_scores.append(float(auc))
    best_iterations.append(int(model.best_iteration_ or base_params["n_estimators"]))
    print(f"fold {fold} roc_auc={auc:.6f} best_iteration={best_iterations[-1]}")

cv_auc = roc_auc_score(y, oof)
pd.DataFrame({"row": np.arange(len(y)), "target": y, "prediction": oof}).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

full_tables = fit_strategy_tables(train_base)
full_train = add_strategy_scores(train_base, full_tables)
full_test = add_strategy_scores(test_base, full_tables)

final_params = base_params.copy()
final_params["n_estimators"] = int(
    np.clip(round(np.mean(best_iterations)), 100, base_params["n_estimators"])
)
final_params["scale_pos_weight"] = float((len(y) - y.sum()) / max(y.sum(), 1))

final_model = LGBMClassifier(**final_params)
final_model.fit(
    full_train[feature_cols],
    y,
    categorical_feature=cat_features,
)

test_pred = final_model.predict_proba(full_test[feature_cols])[:, 1].clip(0, 1)
target_col = [c for c in sample.columns if c != "id"][0]
submission = sample.copy()
submission[target_col] = test_pred

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
    "final_model_iterations": final_params["n_estimators"],
    "research_hypotheses_llm_claimed_used": ["000149"],
    "artifacts": {
        "submission": os.path.join(WORK_DIR, "submission.csv"),
        "oof_predictions": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
        "test_predictions": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    },
}
print(json.dumps(result, indent=2))
