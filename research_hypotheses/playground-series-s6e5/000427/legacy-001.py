import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORKING = Path("./working")
WORKING.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
IDCOL = "id"
RANDOM_STATE = 2026
N_SPLITS = 5

train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

y = train[TARGET].astype(int)
test_ids = sample[IDCOL].copy()

TEAM_BY_YEAR = {
    2022: {
        "HAM": "Mercedes",
        "RUS": "Mercedes",
        "VER": "Red Bull",
        "PER": "Red Bull",
        "LEC": "Ferrari",
        "SAI": "Ferrari",
        "NOR": "McLaren",
        "RIC": "McLaren",
        "ALO": "Alpine",
        "OCO": "Alpine",
        "GAS": "AlphaTauri",
        "TSU": "AlphaTauri",
        "VET": "Aston Martin",
        "STR": "Aston Martin",
        "ALB": "Williams",
        "LAT": "Williams",
        "BOT": "Alfa Romeo",
        "ZHO": "Alfa Romeo",
        "MAG": "Haas",
        "MSC": "Haas",
        "HUL": "Aston Martin",
        "DEV": "Williams",
    },
    2023: {
        "HAM": "Mercedes",
        "RUS": "Mercedes",
        "VER": "Red Bull",
        "PER": "Red Bull",
        "LEC": "Ferrari",
        "SAI": "Ferrari",
        "NOR": "McLaren",
        "PIA": "McLaren",
        "OCO": "Alpine",
        "GAS": "Alpine",
        "ALO": "Aston Martin",
        "STR": "Aston Martin",
        "ALB": "Williams",
        "SAR": "Williams",
        "TSU": "AlphaTauri",
        "DEV": "AlphaTauri",
        "RIC": "AlphaTauri",
        "LAW": "AlphaTauri",
        "BOT": "Alfa Romeo",
        "ZHO": "Alfa Romeo",
        "MAG": "Haas",
        "HUL": "Haas",
    },
    2024: {
        "HAM": "Mercedes",
        "RUS": "Mercedes",
        "VER": "Red Bull",
        "PER": "Red Bull",
        "LEC": "Ferrari",
        "SAI": "Ferrari",
        "NOR": "McLaren",
        "PIA": "McLaren",
        "OCO": "Alpine",
        "GAS": "Alpine",
        "ALO": "Aston Martin",
        "STR": "Aston Martin",
        "ALB": "Williams",
        "SAR": "Williams",
        "COL": "Williams",
        "TSU": "RB",
        "RIC": "RB",
        "LAW": "RB",
        "BOT": "Kick Sauber",
        "ZHO": "Kick Sauber",
        "MAG": "Haas",
        "HUL": "Haas",
        "BEA": "Haas",
    },
    2025: {
        "RUS": "Mercedes",
        "ANT": "Mercedes",
        "VER": "Red Bull",
        "PER": "Red Bull",
        "LAW": "Racing Bulls",
        "TSU": "Red Bull",
        "LEC": "Ferrari",
        "HAM": "Ferrari",
        "NOR": "McLaren",
        "PIA": "McLaren",
        "GAS": "Alpine",
        "DOO": "Alpine",
        "COL": "Alpine",
        "ALO": "Aston Martin",
        "STR": "Aston Martin",
        "ALB": "Williams",
        "SAI": "Williams",
        "HUL": "Kick Sauber",
        "BOR": "Kick Sauber",
        "OCO": "Haas",
        "BEA": "Haas",
        "HAD": "Racing Bulls",
    },
}

HISTORIC_TEAM_FALLBACK = {
    "MAS": "Williams",
    "WEB": "Red Bull",
    "GLO": "Marussia",
    "BUT": "McLaren",
    "RAI": "Alfa Romeo",
    "BAR": "Brawn",
    "VIL": "Williams",
    "COU": "McLaren",
    "MON": "McLaren",
    "TRU": "Toyota",
    "FIS": "Force India",
    "KOV": "McLaren",
    "HEI": "BMW Sauber",
    "KUB": "BMW Sauber",
    "ROS": "Mercedes",
    "MSC": "Ferrari",
    "SCH": "Ferrari",
    "ALO": "Aston Martin",
    "VET": "Aston Martin",
    "DIR": "Force India",
    "SUT": "Force India",
    "ALG": "Toro Rosso",
    "BUE": "Toro Rosso",
    "KOB": "Sauber",
    "PET": "Renault",
    "MAL": "Lotus",
    "GRO": "Haas",
    "VER": "Red Bull",
    "KVY": "Toro Rosso",
    "ERI": "Sauber",
    "NAS": "Sauber",
    "PAL": "Renault",
    "VAN": "McLaren",
    "HAR": "Toro Rosso",
    "SIR": "Williams",
    "GIO": "Alfa Romeo",
    "MAZ": "Haas",
    "LAT": "Williams",
    "RIC": "RB",
    "MAG": "Haas",
    "HUL": "Haas",
    "BOT": "Kick Sauber",
    "ZHO": "Kick Sauber",
    "STR": "Aston Martin",
}

RACE_OVERRIDES = {
    (2024, "Saudi Arabian Grand Prix", "BEA"): "Ferrari",
}


def map_team(row):
    year = int(row["Year"])
    race = str(row["Race"])
    drv = str(row["Driver"]).upper()
    if (year, race, drv) in RACE_OVERRIDES:
        return RACE_OVERRIDES[(year, race, drv)]
    return TEAM_BY_YEAR.get(year, {}).get(
        drv, HISTORIC_TEAM_FALLBACK.get(drv, "UnknownTeam")
    )


def add_base_features(train_df, test_df):
    tr = train_df.drop(columns=[TARGET]).copy()
    te = test_df.copy()

    for df in (tr, te):
        df["TeamName"] = df.apply(map_team, axis=1)
        df["WetRace"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")
        df["IsFirstStint"] = (df["Stint"] == 1).astype("int8")
        df["LateRace"] = (df["RaceProgress"] >= 0.70).astype("int8")
        df["TeamCompound"] = (
            df["TeamName"].astype(str) + "__" + df["Compound"].astype(str)
        )
        df["TeamRace"] = df["TeamName"].astype(str) + "__" + df["Race"].astype(str)

    grp = tr.groupby(["Year", "Race"], observed=True)["LapTime_Delta"].quantile(0.90)
    fallback_delta = float(tr["LapTime_Delta"].quantile(0.90))

    def add_caution_proxy(df):
        key = pd.MultiIndex.from_frame(df[["Year", "Race"]])
        thresholds = pd.Series(key.map(grp), index=df.index).fillna(fallback_delta)
        df["CautionProxy"] = (df["LapTime_Delta"].values >= thresholds.values).astype(
            "int8"
        )
        df["TeamCaution"] = (
            df["TeamName"].astype(str) + "__" + df["CautionProxy"].astype(str)
        )
        df["TeamWet"] = df["TeamName"].astype(str) + "__" + df["WetRace"].astype(str)
        return df

    tr = add_caution_proxy(tr)
    te = add_caution_proxy(te)
    return tr, te


base_train, base_test = add_base_features(train, test)

PRIOR_FEATURES = [
    "prior_team_compound_pit_rate",
    "prior_team_caution_stop_tendency",
    "prior_team_wet_aggressiveness",
    "prior_team_race_pit_rate",
    "prior_team_race_first_stop_progress",
    "prior_team_race_first_stop_tyre_life",
    "prior_dist_to_team_race_first_stop",
    "prior_tyre_delta_to_team_race_first_stop",
]


def smoothed_target_rate(fit_df, apply_df, y_fit, keys, name, alpha=50.0):
    tmp = fit_df[list(keys)].copy()
    tmp["_y"] = np.asarray(y_fit)
    global_mean = float(np.mean(y_fit))
    stats = (
        tmp.groupby(list(keys), observed=True)["_y"].agg(["sum", "count"]).reset_index()
    )
    stats[name] = (stats["sum"] + alpha * global_mean) / (stats["count"] + alpha)
    out = apply_df[list(keys)].merge(
        stats[list(keys) + [name]], on=list(keys), how="left"
    )[name]
    return out.fillna(global_mean).astype("float32").to_numpy()


def first_stop_prior(fit_df, apply_df, y_fit):
    tmp = fit_df[["TeamName", "Race", "RaceProgress", "TyreLife", "Stint"]].copy()
    tmp["_y"] = np.asarray(y_fit)
    pos = tmp[(tmp["Stint"] == 1) & (tmp["_y"] == 1)]

    if len(pos) == 0:
        gp = float(fit_df["RaceProgress"].median())
        gt = float(fit_df["TyreLife"].median())
        progress = np.full(len(apply_df), gp, dtype="float32")
        tyre = np.full(len(apply_df), gt, dtype="float32")
    else:
        gp = float(pos["RaceProgress"].median())
        gt = float(pos["TyreLife"].median())

        def stat_table(keys, col, global_value, smooth=12.0):
            s = (
                pos.groupby(keys, observed=True)[col]
                .agg(["mean", "count"])
                .reset_index()
            )
            s[f"_{col}"] = (s["mean"] * s["count"] + smooth * global_value) / (
                s["count"] + smooth
            )
            return s[keys + [f"_{col}"]]

        pair_p = stat_table(["TeamName", "Race"], "RaceProgress", gp)
        pair_t = stat_table(["TeamName", "Race"], "TyreLife", gt)
        team_p = stat_table(["TeamName"], "RaceProgress", gp)
        team_t = stat_table(["TeamName"], "TyreLife", gt)
        race_p = stat_table(["Race"], "RaceProgress", gp)
        race_t = stat_table(["Race"], "TyreLife", gt)

        m = apply_df[["TeamName", "Race", "RaceProgress", "TyreLife"]].copy()
        m = m.merge(pair_p, on=["TeamName", "Race"], how="left")
        m = m.merge(pair_t, on=["TeamName", "Race"], how="left")
        m = m.merge(team_p, on=["TeamName"], how="left", suffixes=("", "_team"))
        m = m.merge(team_t, on=["TeamName"], how="left", suffixes=("", "_team"))
        m = m.merge(race_p, on=["Race"], how="left", suffixes=("", "_race"))
        m = m.merge(race_t, on=["Race"], how="left", suffixes=("", "_race"))

        progress = (
            m["_RaceProgress"]
            .fillna(m["_RaceProgress_team"])
            .fillna(m["_RaceProgress_race"])
            .fillna(gp)
            .astype("float32")
            .to_numpy()
        )
        tyre = (
            m["_TyreLife"]
            .fillna(m["_TyreLife_team"])
            .fillna(m["_TyreLife_race"])
            .fillna(gt)
            .astype("float32")
            .to_numpy()
        )

    return progress, tyre


def build_priors(fit_df, apply_df, y_fit):
    pri = pd.DataFrame(index=apply_df.index)
    pri["prior_team_compound_pit_rate"] = smoothed_target_rate(
        fit_df,
        apply_df,
        y_fit,
        ["TeamName", "Compound"],
        "prior_team_compound_pit_rate",
    )
    pri["prior_team_caution_stop_tendency"] = smoothed_target_rate(
        fit_df,
        apply_df,
        y_fit,
        ["TeamName", "CautionProxy"],
        "prior_team_caution_stop_tendency",
    )
    pri["prior_team_wet_aggressiveness"] = smoothed_target_rate(
        fit_df,
        apply_df,
        y_fit,
        ["TeamName", "WetRace"],
        "prior_team_wet_aggressiveness",
    )
    pri["prior_team_race_pit_rate"] = smoothed_target_rate(
        fit_df,
        apply_df,
        y_fit,
        ["TeamName", "Race"],
        "prior_team_race_pit_rate",
        alpha=80.0,
    )
    prog, tyre = first_stop_prior(fit_df, apply_df, y_fit)
    pri["prior_team_race_first_stop_progress"] = prog
    pri["prior_team_race_first_stop_tyre_life"] = tyre
    pri["prior_dist_to_team_race_first_stop"] = np.abs(
        apply_df["RaceProgress"].to_numpy() - prog
    ).astype("float32")
    pri["prior_tyre_delta_to_team_race_first_stop"] = (
        apply_df["TyreLife"].to_numpy() - tyre
    ).astype("float32")
    return pri[PRIOR_FEATURES].astype("float32")


def sanitize_columns(cols):
    mapping, seen = {}, {}
    for c in cols:
        s = re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_")
        if not s:
            s = "f"
        if s[0].isdigit():
            s = "f_" + s
        base = s
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            s = f"{base}_{seen[base]}"
        mapping[c] = s
    return mapping


feature_cols = [c for c in base_train.columns if c != IDCOL]
cat_cols = [
    c
    for c in feature_cols
    if base_train[c].dtype == "object"
    or c in ["Year", "CautionProxy", "WetRace", "IsFirstStint", "LateRace"]
]

for c in cat_cols:
    vals = (
        pd.concat([base_train[c], base_test[c]], axis=0)
        .astype(str)
        .fillna("__NA__")
        .unique()
    )
    dtype = pd.CategoricalDtype(categories=sorted(vals), ordered=False)
    base_train[c] = base_train[c].astype(str).fillna("__NA__").astype(dtype)
    base_test[c] = base_test[c].astype(str).fillna("__NA__").astype(dtype)

for c in feature_cols:
    if c not in cat_cols:
        base_train[c] = pd.to_numeric(base_train[c], errors="coerce").astype("float32")
        base_test[c] = pd.to_numeric(base_test[c], errors="coerce").astype("float32")

rename_map = sanitize_columns(feature_cols + PRIOR_FEATURES)
model_cat_cols = [rename_map[c] for c in cat_cols if c in feature_cols]


def make_matrix(base_part, prior_part):
    x = pd.concat([base_part[feature_cols], prior_part[PRIOR_FEATURES]], axis=1)
    return x.rename(columns=rename_map)


outer_cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(base_train), dtype="float32")
test_pred = np.zeros(len(base_test), dtype="float64")
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(base_train, y), 1):
    print(f"Fold {fold}/{N_SPLITS}")
    tr_base = base_train.iloc[tr_idx]
    va_base = base_train.iloc[va_idx]
    y_tr = y.iloc[tr_idx]
    y_va = y.iloc[va_idx]

    inner_prior = pd.DataFrame(
        index=tr_base.index, columns=PRIOR_FEATURES, dtype="float32"
    )
    inner_cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + fold
    )

    tr_positions = np.arange(len(tr_base))
    for in_fit_pos, in_apply_pos in inner_cv.split(tr_positions, y_tr):
        fit_idx = tr_base.index[in_fit_pos]
        apply_idx = tr_base.index[in_apply_pos]
        inner_prior.loc[apply_idx, PRIOR_FEATURES] = build_priors(
            base_train.loc[fit_idx], base_train.loc[apply_idx], y.loc[fit_idx]
        ).values

    va_prior = build_priors(tr_base, va_base, y_tr)
    test_prior = build_priors(tr_base, base_test, y_tr)

    X_tr = make_matrix(tr_base, inner_prior.loc[tr_base.index])
    X_va = make_matrix(va_base, va_prior)
    X_te = make_matrix(base_test, test_prior)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=2500,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=3.0,
        random_state=RANDOM_STATE + fold,
        n_jobs=min(8, os.cpu_count() or 1),
        verbosity=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=model_cat_cols,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    va_pred = model.predict_proba(X_va)[:, 1]
    oof[va_idx] = va_pred.astype("float32")
    test_pred += model.predict_proba(X_te)[:, 1] / N_SPLITS

    auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(float(auc))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(WORKING / "submission.csv", index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y.to_numpy(),
        "prediction": oof,
    }
).to_csv(WORKING / "oof_predictions.csv.gz", index=False, compression="gzip")

pd.DataFrame(
    {
        IDCOL: test_ids,
        TARGET: np.clip(test_pred, 0, 1),
    }
).to_csv(WORKING / "test_predictions.csv.gz", index=False, compression="gzip")

print(
    json.dumps(
        {
            "cv_roc_auc": float(cv_auc),
            "fold_roc_auc": fold_scores,
            "research_hypotheses_llm_claimed_used": ["000427"],
            "submission_path": str(WORKING / "submission.csv"),
            "oof_path": str(WORKING / "oof_predictions.csv.gz"),
            "test_predictions_path": str(WORKING / "test_predictions.csv.gz"),
        }
    )
)
