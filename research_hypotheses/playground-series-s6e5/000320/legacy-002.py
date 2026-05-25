import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESIS_ID = "000320"
RANDOM_STATE = 42

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values

CAT_COLS = ["Driver", "Race", "Compound", "Race_Year"]


def add_basic_features(df):
    out = df.copy()
    out["Race_Year"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)
    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["Deg_per_TyreLife"] = out["Cumulative_Degradation"] / (out["TyreLife"] + 1.0)
    out["LapTime_per_Progress"] = out["LapTime (s)"] / (out["RaceProgress"] + 0.01)
    out["IsWetCompound"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    out["IsSlick"] = (1 - out["IsWetCompound"]).astype(int)
    return out


def make_prior_maps(fit_df, alpha=40.0):
    d = add_basic_features(fit_df)
    d["late_stint_stop"] = ((d[TARGET] == 1) & (d["RaceProgress"] >= 0.45)).astype(int)

    global_hazard = float(d[TARGET].mean())
    global_stint = float(d["TyreLife"].median())
    global_deg = float(d["Deg_per_TyreLife"].median())
    global_late = float(d["late_stint_stop"].mean())

    def shrunk_mean(group_cols, value_col, global_value):
        g = (
            d.groupby(group_cols, observed=True)[value_col]
            .agg(["mean", "count"])
            .reset_index()
        )
        g["value"] = (g["mean"] * g["count"] + global_value * alpha) / (
            g["count"] + alpha
        )
        return g[group_cols + ["value"]]

    def shrunk_median(group_cols, value_col, global_value):
        g = (
            d.groupby(group_cols, observed=True)[value_col]
            .agg(["median", "count"])
            .reset_index()
        )
        g["value"] = (g["median"] * g["count"] + global_value * alpha) / (
            g["count"] + alpha
        )
        return g[group_cols + ["value"]]

    maps = {
        "race_hazard": shrunk_mean(["Race"], TARGET, global_hazard),
        "race_year_hazard": shrunk_mean(["Race_Year"], TARGET, global_hazard),
        "race_compound_hazard": shrunk_mean(
            ["Race", "Compound"], TARGET, global_hazard
        ),
        "race_stint_median": shrunk_median(["Race"], "TyreLife", global_stint),
        "race_year_stint_median": shrunk_median(
            ["Race_Year"], "TyreLife", global_stint
        ),
        "race_deg_regime": shrunk_median(["Race"], "Deg_per_TyreLife", global_deg),
        "race_late_stop_proxy": shrunk_mean(["Race"], "late_stint_stop", global_late),
        "compound_hazard": shrunk_mean(["Compound"], TARGET, global_hazard),
        "wet_hazard": shrunk_mean(["IsWetCompound"], TARGET, global_hazard),
    }
    globals_ = {
        "global_hazard": global_hazard,
        "global_stint": global_stint,
        "global_deg": global_deg,
        "global_late": global_late,
    }
    return maps, globals_


def apply_prior_maps(df, maps, globals_):
    out = add_basic_features(df)

    specs = [
        ("race_hazard", ["Race"], "circuit_stop_prior", globals_["global_hazard"]),
        (
            "race_year_hazard",
            ["Race_Year"],
            "race_year_stop_prior",
            globals_["global_hazard"],
        ),
        (
            "race_compound_hazard",
            ["Race", "Compound"],
            "compound_circuit_hazard",
            globals_["global_hazard"],
        ),
        (
            "race_stint_median",
            ["Race"],
            "circuit_median_stint",
            globals_["global_stint"],
        ),
        (
            "race_year_stint_median",
            ["Race_Year"],
            "race_year_median_stint",
            globals_["global_stint"],
        ),
        ("race_deg_regime", ["Race"], "circuit_deg_regime", globals_["global_deg"]),
        (
            "race_late_stop_proxy",
            ["Race"],
            "overcut_favorability_proxy",
            globals_["global_late"],
        ),
        (
            "compound_hazard",
            ["Compound"],
            "compound_stop_prior",
            globals_["global_hazard"],
        ),
        (
            "wet_hazard",
            ["IsWetCompound"],
            "wet_regime_stop_prior",
            globals_["global_hazard"],
        ),
    ]

    for map_name, keys, new_col, fill_value in specs:
        m = maps[map_name].rename(columns={"value": new_col})
        out = out.merge(m, on=keys, how="left")
        out[new_col] = out[new_col].fillna(fill_value)

    out["tyrelife_vs_circuit_median"] = out["TyreLife"] - out["circuit_median_stint"]
    out["tyrelife_ratio_circuit_median"] = out["TyreLife"] / (
        out["circuit_median_stint"] + 1.0
    )
    out["race_year_stint_delta"] = out["TyreLife"] - out["race_year_median_stint"]
    out["deg_vs_circuit_regime"] = out["Deg_per_TyreLife"] - out["circuit_deg_regime"]
    return out


def prepare_model_frame(df, fit_categories=None):
    out = df.copy()
    for c in CAT_COLS:
        if c in out.columns:
            out[c] = out[c].astype("category")
            if fit_categories is not None:
                out[c] = out[c].cat.set_categories(fit_categories[c])
    return out


def make_model(seed):
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
    tr_raw = train.iloc[tr_idx].reset_index(drop=True)
    va_raw = train.iloc[va_idx].reset_index(drop=True)

    maps, globals_ = make_prior_maps(tr_raw)
    tr_feat = apply_prior_maps(tr_raw.drop(columns=[TARGET]), maps, globals_)
    va_feat = apply_prior_maps(va_raw.drop(columns=[TARGET]), maps, globals_)

    tr_feat = prepare_model_frame(tr_feat)
    categories = {c: tr_feat[c].cat.categories for c in CAT_COLS}
    va_feat = prepare_model_frame(va_feat, categories)

    feature_cols = [c for c in tr_feat.columns if c != ID_COL]
    categorical_feature = [c for c in CAT_COLS if c in feature_cols]

    model = make_model(RANDOM_STATE + fold)
    model.fit(
        tr_feat[feature_cols],
        tr_raw[TARGET].astype(int),
        categorical_feature=categorical_feature,
        eval_set=[(va_feat[feature_cols], va_raw[TARGET].astype(int))],
        eval_metric="auc",
    )

    va_pred = model.predict_proba(va_feat[feature_cols])[:, 1]
    oof[va_idx] = va_pred
    fold_auc = roc_auc_score(va_raw[TARGET].astype(int), va_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

    test_fold = apply_prior_maps(test, maps, globals_)
    test_fold = prepare_model_frame(test_fold, categories)
    test_pred += model.predict_proba(test_fold[feature_cols])[:, 1] / skf.n_splits

cv_auc = roc_auc_score(y, oof)
print(f"5-fold CV ROC AUC: {cv_auc:.6f}")
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")

submission = sample[[ID_COL]].copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(oof, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample[[ID_COL]].copy()
test_pred_df[TARGET] = np.clip(test_pred, 0, 1)
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(x) for x in fold_scores],
    "research_hypotheses_llm_claimed_used": [HYPOTHESIS_ID],
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
