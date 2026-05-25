import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
WEIGHTS = {"finish": 0.35, "fresh": 0.45, "debt": 0.20}
TARGET = "PitNextLap"
ID_COL = "id"

os.makedirs("./working", exist_ok=True)

train = pd.read_csv("./input/train.csv.gz")
test = pd.read_csv("./input/test.csv.gz")
sample = pd.read_csv("./input/sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
n_train = len(train)


def uniq(seq):
    return list(dict.fromkeys(seq))


def rank01(x):
    return pd.Series(np.asarray(x)).rank(method="average").to_numpy() / (len(x) + 1.0)


def safe_auc(y_true, pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, pred)


def build_features(train_df, test_df):
    train_x = train_df.drop(columns=[TARGET])
    all_df = pd.concat([train_x, test_df], axis=0, ignore_index=True)

    for c in ["Compound", "Race", "Driver"]:
        all_df[c] = all_df[c].astype(str).fillna("missing")

    eps = 1e-6
    slicks = ["SOFT", "MEDIUM", "HARD"]

    all_df["RaceProgress_clip"] = all_df["RaceProgress"].clip(lower=0.01, upper=1.0)
    all_df["race_total_est"] = (all_df["LapNumber"] / all_df["RaceProgress_clip"]).clip(
        1, 120
    )
    all_df["laps_remaining"] = (all_df["race_total_est"] - all_df["LapNumber"]).clip(
        0, 120
    )
    all_df["laps_remaining_log1p"] = np.log1p(all_df["laps_remaining"])

    speed_map = {
        "HARD": 1.0,
        "MEDIUM": 2.0,
        "SOFT": 3.0,
        "INTERMEDIATE": 0.5,
        "WET": 0.25,
    }
    wear_map = {
        "HARD": 1.0,
        "MEDIUM": 2.0,
        "SOFT": 3.0,
        "INTERMEDIATE": 2.2,
        "WET": 1.8,
    }

    all_df["compound_speed_rank"] = (
        all_df["Compound"].map(speed_map).fillna(0).astype(float)
    )
    all_df["compound_wear_rank"] = (
        all_df["Compound"].map(wear_map).fillna(0).astype(float)
    )

    for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        all_df[f"is_{comp.lower()}"] = (all_df["Compound"] == comp).astype("int8")
    all_df["is_slick"] = all_df["Compound"].isin(slicks).astype("int8")

    all_df["tyre_life_race_frac"] = all_df["TyreLife"] / (
        all_df["race_total_est"] + eps
    )
    all_df["tyre_life_remaining_ratio"] = all_df["TyreLife"] / (
        all_df["laps_remaining"] + 1.0
    )
    all_df["remaining_tyre_interaction"] = all_df["TyreLife"] * all_df["laps_remaining"]
    all_df["degradation_per_tyre_lap"] = all_df["Cumulative_Degradation"] / all_df[
        "TyreLife"
    ].clip(lower=1)
    all_df["lapdelta_per_tyre_lap"] = all_df["LapTime_Delta"] / all_df["TyreLife"].clip(
        lower=1
    )
    all_df["lap_time_per_progress"] = (
        all_df["LapTime (s)"] / all_df["RaceProgress_clip"]
    )
    all_df["abs_lap_time_delta"] = all_df["LapTime_Delta"].abs()
    all_df["degradation_projected_finish"] = (
        all_df["Cumulative_Degradation"]
        + all_df["degradation_per_tyre_lap"] * all_df["laps_remaining"]
    )
    all_df["late_race"] = (all_df["RaceProgress"] >= 0.75).astype("int8")
    all_df["early_race"] = (all_df["RaceProgress"] <= 0.25).astype("int8")
    all_df["position_x_progress"] = all_df["Position"] * all_df["RaceProgress"]
    all_df["tyrelife_x_progress"] = all_df["TyreLife"] * all_df["RaceProgress"]
    all_df["stint_x_progress"] = all_df["Stint"] * all_df["RaceProgress"]

    train_part = all_df.iloc[:n_train].copy()
    global_q = {
        q: float(train_part["TyreLife"].quantile(q)) for q in [0.80, 0.90, 0.95]
    }
    comp_q = {
        q: train_part.groupby("Compound")["TyreLife"].quantile(q).to_dict()
        for q in [0.80, 0.90, 0.95]
    }

    for q, suffix in [(0.80, "p80"), (0.90, "p90"), (0.95, "p95")]:
        all_df[f"compound_life_{suffix}"] = (
            all_df["Compound"].map(comp_q[q]).fillna(global_q[q])
        )
        all_df[f"current_life_margin_{suffix}"] = (
            all_df[f"compound_life_{suffix}"] - all_df["TyreLife"]
        )
        all_df[f"finish_margin_current_{suffix}"] = (
            all_df[f"current_life_margin_{suffix}"] - all_df["laps_remaining"]
        )
        all_df[f"current_can_finish_{suffix}"] = (
            all_df[f"finish_margin_current_{suffix}"] >= 0
        ).astype("int8")

    all_df["finish_ratio_current_p90"] = all_df["current_life_margin_p90"] / (
        all_df["laps_remaining"] + 1.0
    )

    slick_life = {c: comp_q[0.90].get(c, global_q[0.90]) for c in slicks}
    margin_cols = []
    for comp in slicks:
        name = comp.lower()
        all_df[f"fresh_{name}_finish_margin"] = (
            slick_life[comp] - all_df["laps_remaining"]
        )
        all_df[f"fresh_{name}_finish_ok"] = (
            all_df[f"fresh_{name}_finish_margin"] >= 0
        ).astype("int8")
        all_df[f"fresh_{name}_late_ok"] = (
            all_df[f"fresh_{name}_finish_ok"] * all_df["late_race"]
        )
        margin_cols.append(f"fresh_{name}_finish_margin")

    margins = all_df[margin_cols].to_numpy()
    sorted_margins = np.sort(margins, axis=1)
    ok = margins >= 0

    all_df["fresh_slick_option_count"] = ok.sum(axis=1).astype("int8")
    all_df["fresh_slick_margin_best"] = sorted_margins[:, -1]
    all_df["fresh_slick_margin_second"] = sorted_margins[:, -2]
    all_df["fresh_slick_min_shortfall"] = np.maximum(
        -all_df["fresh_slick_margin_best"], 0
    )
    all_df["fresh_slick_all_too_short"] = (
        all_df["fresh_slick_option_count"] == 0
    ).astype("int8")

    current_speed = all_df["compound_speed_rank"].to_numpy()
    option_speeds = np.array([3.0, 2.0, 1.0])
    all_df["fresh_faster_or_equal_count"] = (
        (ok) & (option_speeds[None, :] >= current_speed[:, None])
    ).sum(axis=1)
    all_df["fresh_slower_safe_count"] = (
        (ok) & (option_speeds[None, :] < current_speed[:, None])
    ).sum(axis=1)
    all_df["fresh_best_speed_finishing"] = np.where(ok, option_speeds[None, :], 0).max(
        axis=1
    )
    all_df["fresh_option_richness_late"] = all_df["fresh_slick_option_count"] / (
        all_df["laps_remaining"] + 1.0
    )
    all_df["fresh_option_balance"] = (
        all_df["fresh_hard_finish_ok"] - all_df["fresh_soft_finish_ok"]
    )

    all_df["completed_stops_proxy"] = np.maximum(
        (all_df["Stint"] - 1).clip(lower=0), all_df["PitStop"]
    )
    all_df["no_stop_yet"] = (all_df["completed_stops_proxy"] <= 0).astype("int8")
    all_df["owes_stop_strict"] = (
        (all_df["Stint"] <= 1)
        & (all_df["PitStop"] == 0)
        & (all_df["is_slick"] == 1)
        & (all_df["RaceProgress"] < 0.995)
    ).astype("int8")
    all_df["stop_debt_pressure"] = all_df["owes_stop_strict"] / (
        all_df["laps_remaining"] + 1.0
    )

    for w in [3, 5, 10, 15]:
        all_df[f"debt_window_{w}"] = (
            (all_df["owes_stop_strict"] == 1) & (all_df["laps_remaining"] <= w)
        ).astype("int8")

    all_df["late_no_stop_slick"] = all_df["owes_stop_strict"] * all_df["RaceProgress"]
    all_df["debt_tyre_age_pressure"] = (
        all_df["owes_stop_strict"] * all_df["tyre_life_remaining_ratio"]
    )
    all_df["debt_and_current_can_finish"] = (
        all_df["owes_stop_strict"] * all_df["current_can_finish_p90"]
    )
    all_df["debt_with_no_fresh_options"] = all_df["owes_stop_strict"] * (
        all_df["fresh_slick_option_count"] == 0
    ).astype("int8")
    all_df["debt_with_fresh_options"] = all_df["owes_stop_strict"] * (
        all_df["fresh_slick_option_count"] > 0
    ).astype("int8")
    all_df["debt_fresh_option_count"] = (
        all_df["owes_stop_strict"] * all_df["fresh_slick_option_count"]
    )
    all_df["debt_best_fresh_margin"] = (
        all_df["owes_stop_strict"] * all_df["fresh_slick_margin_best"]
    )
    all_df["mandatory_last_laps"] = all_df["owes_stop_strict"] * (
        all_df["laps_remaining"] <= 2
    ).astype("int8")
    all_df["stint_stop_deficit"] = (
        np.maximum(1 - all_df["completed_stops_proxy"], 0) * all_df["is_slick"]
    )
    all_df["rule_state_pressure"] = all_df["stint_stop_deficit"] / (
        all_df["laps_remaining"] + 1.0
    )

    for c in ["Compound", "Race", "Driver"]:
        cats = pd.Index(all_df[c].astype(str).unique())
        all_df[c] = pd.Categorical(all_df[c].astype(str), categories=cats)

    all_df = all_df.drop(columns=["RaceProgress_clip"])
    return all_df.iloc[:n_train].reset_index(drop=True), all_df.iloc[
        n_train:
    ].reset_index(drop=True)


train_feat, test_feat = build_features(train, test)

raw_core = [
    "Year",
    "LapNumber",
    "LapTime (s)",
    "LapTime_Delta",
    "PitStop",
    "Position",
    "Position_Change",
    "RaceProgress",
    "Stint",
    "TyreLife",
    "Cumulative_Degradation",
]

finish_features = uniq(
    ["Compound", "Race", "Driver"]
    + raw_core
    + [
        "race_total_est",
        "laps_remaining",
        "laps_remaining_log1p",
        "tyre_life_race_frac",
        "tyre_life_remaining_ratio",
        "remaining_tyre_interaction",
        "degradation_per_tyre_lap",
        "lapdelta_per_tyre_lap",
        "lap_time_per_progress",
        "abs_lap_time_delta",
        "compound_wear_rank",
        "compound_speed_rank",
        "is_slick",
        "is_soft",
        "is_medium",
        "is_hard",
        "is_intermediate",
        "is_wet",
        "compound_life_p80",
        "compound_life_p90",
        "compound_life_p95",
        "current_life_margin_p80",
        "current_life_margin_p90",
        "current_life_margin_p95",
        "finish_margin_current_p80",
        "finish_margin_current_p90",
        "finish_margin_current_p95",
        "finish_ratio_current_p90",
        "current_can_finish_p80",
        "current_can_finish_p90",
        "current_can_finish_p95",
        "degradation_projected_finish",
        "late_race",
        "early_race",
        "position_x_progress",
        "tyrelife_x_progress",
        "stint_x_progress",
    ]
)

fresh_features = uniq(
    ["Compound", "Race"]
    + [
        "Year",
        "LapNumber",
        "RaceProgress",
        "Stint",
        "TyreLife",
        "Position",
        "PitStop",
        "LapTime_Delta",
        "Cumulative_Degradation",
        "race_total_est",
        "laps_remaining",
        "laps_remaining_log1p",
        "compound_speed_rank",
        "compound_wear_rank",
        "is_slick",
        "is_soft",
        "is_medium",
        "is_hard",
        "is_intermediate",
        "is_wet",
        "fresh_soft_finish_margin",
        "fresh_medium_finish_margin",
        "fresh_hard_finish_margin",
        "fresh_soft_finish_ok",
        "fresh_medium_finish_ok",
        "fresh_hard_finish_ok",
        "fresh_slick_option_count",
        "fresh_slick_margin_best",
        "fresh_slick_margin_second",
        "fresh_slick_min_shortfall",
        "fresh_slick_all_too_short",
        "fresh_faster_or_equal_count",
        "fresh_slower_safe_count",
        "fresh_best_speed_finishing",
        "fresh_option_richness_late",
        "fresh_option_balance",
        "fresh_soft_late_ok",
        "fresh_medium_late_ok",
        "fresh_hard_late_ok",
        "position_x_progress",
        "stint_x_progress",
        "tyre_life_remaining_ratio",
        "abs_lap_time_delta",
        "degradation_per_tyre_lap",
    ]
)

debt_features = uniq(
    fresh_features
    + [
        "completed_stops_proxy",
        "no_stop_yet",
        "owes_stop_strict",
        "stop_debt_pressure",
        "debt_window_3",
        "debt_window_5",
        "debt_window_10",
        "debt_window_15",
        "late_no_stop_slick",
        "debt_tyre_age_pressure",
        "debt_and_current_can_finish",
        "debt_with_no_fresh_options",
        "debt_with_fresh_options",
        "debt_fresh_option_count",
        "debt_best_fresh_margin",
        "mandatory_last_laps",
        "stint_stop_deficit",
        "rule_state_pressure",
    ]
)

feature_sets = {
    "finish": finish_features,
    "fresh": fresh_features,
    "debt": debt_features,
}

groups = (
    train["Year"].astype(str)
    + "_"
    + train["Race"].astype(str)
    + "_"
    + train["Driver"].astype(str)
)

if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(train_feat, y, groups))
else:
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(train_feat, y, groups))

pos = y.sum()
scale_pos_weight = float(np.clip((len(y) - pos) / max(pos, 1), 1.0, 30.0))
n_jobs = max(1, min(8, os.cpu_count() or 1))


def make_model(name, fold):
    base = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.04,
        subsample=0.86,
        colsample_bytree=0.86,
        reg_alpha=0.08,
        reg_lambda=1.4,
        min_child_samples=80,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED + 101 * fold + len(name),
        n_jobs=n_jobs,
        verbosity=-1,
        force_col_wise=True,
    )
    if name == "finish":
        base.update(num_leaves=63, min_child_samples=70, colsample_bytree=0.82)
    elif name == "fresh":
        base.update(num_leaves=47, min_child_samples=95, reg_lambda=1.7)
    else:
        base.update(num_leaves=55, min_child_samples=85, reg_alpha=0.12)
    return LGBMClassifier(**base)


oof_blend = np.zeros(n_train, dtype=float)
test_blend = np.zeros(len(test_feat), dtype=float)
component_oof_rank = {name: np.zeros(n_train, dtype=float) for name in feature_sets}
component_test_rank = {
    name: np.zeros(len(test_feat), dtype=float) for name in feature_sets
}
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    fold_blend = np.zeros(len(va_idx), dtype=float)
    fold_component_preds = {}

    for name, feats in feature_sets.items():
        cat_feats = [c for c in ["Compound", "Race", "Driver"] if c in feats]
        model = make_model(name, fold)

        model.fit(
            train_feat.iloc[tr_idx][feats],
            y[tr_idx],
            eval_set=[(train_feat.iloc[va_idx][feats], y[va_idx])],
            eval_metric="auc",
            categorical_feature=cat_feats,
            callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
        )

        val_pred = model.predict_proba(train_feat.iloc[va_idx][feats])[:, 1]
        tst_pred = model.predict_proba(test_feat[feats])[:, 1]

        val_rank = rank01(val_pred)
        tst_rank = rank01(tst_pred)

        component_oof_rank[name][va_idx] = val_rank
        component_test_rank[name] += tst_rank / N_SPLITS
        fold_component_preds[name] = val_rank

        fold_blend += WEIGHTS[name] * val_rank
        test_blend += WEIGHTS[name] * tst_rank / N_SPLITS

    oof_blend[va_idx] = fold_blend
    fold_auc = safe_auc(y[va_idx], fold_blend)
    fold_aucs.append(fold_auc)
    print(f"fold {fold} rank_blend_roc_auc={fold_auc:.6f}")

component_auc = {
    name: float(safe_auc(y, component_oof_rank[name])) for name in feature_sets
}
cv_auc = float(roc_auc_score(y, oof_blend))

oof_df = pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": oof_blend,
    }
)
oof_df.to_csv("./working/oof_predictions.csv.gz", index=False, compression="gzip")

target_col = [c for c in sample.columns if c != ID_COL][0]
test_pred_by_id = pd.DataFrame(
    {
        ID_COL: test[ID_COL].to_numpy(),
        target_col: np.clip(test_blend, 0, 1),
    }
)
submission = sample[[ID_COL]].merge(test_pred_by_id, on=ID_COL, how="left")
if submission[target_col].isna().any():
    raise RuntimeError("Submission merge produced missing predictions.")

submission.to_csv("./working/submission.csv", index=False)
submission.to_csv("./working/test_predictions.csv.gz", index=False, compression="gzip")

print(f"OOF rank-average ROC AUC: {cv_auc:.6f}")
print("Component OOF ROC AUC:", json.dumps(component_auc, sort_keys=True))

review = {
    "metric": "roc_auc",
    "cv_roc_auc": cv_auc,
    "fold_roc_auc": [None if np.isnan(v) else float(v) for v in fold_aucs],
    "component_roc_auc": component_auc,
    "research_hypotheses_llm_claimed_used": ["000008"],
    "submission_path": "./working/submission.csv",
    "oof_path": "./working/oof_predictions.csv.gz",
    "test_predictions_path": "./working/test_predictions.csv.gz",
}
print(json.dumps(review, sort_keys=True))
