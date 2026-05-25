import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
LAPTIME = "LapTime (s)"
LAPDELTA = "LapTime_Delta"
CAT_COLS = ["Race", "Driver", "Compound"]
RANDOM_STATE = 2026
N_SPLITS = 5


def safe_divide(a, b):
    return np.where(np.asarray(b) > 0, np.asarray(a) / np.asarray(b), np.nan)


def set_categories(df, cat_levels):
    df = df.copy()
    for c in CAT_COLS:
        df[c] = pd.Categorical(df[c].astype(str), categories=cat_levels[c])
    return df


def add_online_priors(df):
    out = df.copy()
    ev = out.loc[
        out["PitStop"].eq(1),
        [
            "Year",
            "Race",
            "LapNumber",
            "driver_pos_gain_prev",
            "Position_Change",
            LAPDELTA,
        ],
    ].copy()
    ev["pit_pos_gain_event"] = (
        ev["driver_pos_gain_prev"].fillna(-ev["Position_Change"]).clip(-20, 20)
    )
    ev["pit_loss_event"] = ev[LAPDELTA].replace([np.inf, -np.inf], np.nan)
    if len(ev):
        lo, hi = ev["pit_loss_event"].quantile([0.01, 0.99])
        ev["pit_loss_event"] = (
            ev["pit_loss_event"].clip(lo, hi).fillna(ev["pit_loss_event"].median())
        )
    else:
        ev["pit_loss_event"] = 0.0

    keys = out[["Year", "Race"]].drop_duplicates()
    laps = pd.DataFrame({"LapNumber": np.arange(1, int(out["LapNumber"].max()) + 1)})
    grid = keys.merge(laps, how="cross")

    agg = (
        ev.groupby(["Year", "Race", "LapNumber"], observed=True)
        .agg(
            online_event_count=("pit_loss_event", "size"),
            online_pos_sum=("pit_pos_gain_event", "sum"),
            online_loss_sum=("pit_loss_event", "sum"),
        )
        .reset_index()
    )

    grid = grid.merge(agg, on=["Year", "Race", "LapNumber"], how="left")
    for c in ["online_event_count", "online_pos_sum", "online_loss_sum"]:
        grid[c] = grid[c].fillna(0.0)

    grid = grid.sort_values(["Year", "Race", "LapNumber"])
    for raw, prior in [
        ("online_event_count", "online_pit_event_count_prior"),
        ("online_pos_sum", "online_pit_pos_sum_prior"),
        ("online_loss_sum", "online_pit_loss_sum_prior"),
    ]:
        grid[prior] = (
            grid.groupby(["Year", "Race"], observed=True)[raw].cumsum() - grid[raw]
        )

    grid = grid[
        [
            "Year",
            "Race",
            "LapNumber",
            "online_pit_event_count_prior",
            "online_pit_pos_sum_prior",
            "online_pit_loss_sum_prior",
        ]
    ]

    out = out.merge(grid, on=["Year", "Race", "LapNumber"], how="left")
    cnt = out["online_pit_event_count_prior"].fillna(0.0)
    out["online_pit_pos_gain_prior"] = safe_divide(out["online_pit_pos_sum_prior"], cnt)
    out["online_pit_loss_prior"] = safe_divide(out["online_pit_loss_sum_prior"], cnt)
    out["online_pit_event_count_log"] = np.log1p(cnt)
    out["online_undercut_score"] = (
        out["online_pit_pos_gain_prior"] - 0.03 * out["online_pit_loss_prior"]
    )
    return out.drop(columns=["online_pit_pos_sum_prior", "online_pit_loss_sum_prior"])


def add_base_features(df):
    out = df.copy()

    out["tyre_life_sq"] = out["TyreLife"] ** 2
    out["degradation_per_tyre_lap"] = out["Cumulative_Degradation"] / (
        out["TyreLife"] + 1.0
    )
    out["tyre_x_progress"] = out["TyreLife"] * out["RaceProgress"]
    out["position_x_progress"] = out["Position"] * out["RaceProgress"]
    out["lap_delta_x_tyre"] = out[LAPDELTA] * out["TyreLife"]

    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
    out["compound_ord"] = (
        out["Compound"].astype(str).map(compound_map).fillna(0).astype("int8")
    )

    s = out.sort_values(["Year", "Race", "Driver", "LapNumber", ID_COL]).copy()
    g = s.groupby(["Year", "Race", "Driver"], observed=True, sort=False)
    s["prev_position"] = g["Position"].shift(1)
    s["prev_laptime"] = g[LAPTIME].shift(1)
    s["prev_lap_delta"] = g[LAPDELTA].shift(1)
    s["prev_pitstop"] = g["PitStop"].shift(1).fillna(0)
    s["driver_pos_gain_prev"] = s["prev_position"] - s["Position"]
    s["lap_delta_roll3_prior"] = g[LAPDELTA].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    s["lap_time_roll3_prior"] = g[LAPTIME].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    seq_cols = [
        "prev_position",
        "prev_laptime",
        "prev_lap_delta",
        "prev_pitstop",
        "driver_pos_gain_prev",
        "lap_delta_roll3_prior",
        "lap_time_roll3_prior",
    ]
    out[seq_cols] = s[seq_cols]

    lap_g = out.groupby(["Year", "Race", "LapNumber"], observed=True)
    out["lap_laptime_median"] = lap_g[LAPTIME].transform("median")
    out["lap_delta_median"] = lap_g[LAPDELTA].transform("median")
    out["lap_driver_count"] = lap_g["Driver"].transform("count")
    out["lap_pit_count"] = lap_g["PitStop"].transform("sum")
    out["lap_laptime_rank"] = lap_g[LAPTIME].rank(method="average", ascending=True)
    out["lap_laptime_vs_median"] = out[LAPTIME] - out["lap_laptime_median"]
    out["lap_delta_vs_median"] = out[LAPDELTA] - out["lap_delta_median"]
    out["position_pct"] = (out["Position"] - 1) / (out["lap_driver_count"] - 1).replace(
        0, np.nan
    )

    r = out.sort_values(["Year", "Race", "LapNumber", "Position", ID_COL]).copy()
    rg = r.groupby(["Year", "Race", "LapNumber"], observed=True, sort=False)
    r["ahead_laptime"] = rg[LAPTIME].shift(1)
    r["behind_laptime"] = rg[LAPTIME].shift(-1)
    r["ahead_tyre_life"] = rg["TyreLife"].shift(1)
    r["behind_tyre_life"] = rg["TyreLife"].shift(-1)
    r["ahead_compound_ord"] = rg["compound_ord"].shift(1)
    r["behind_compound_ord"] = rg["compound_ord"].shift(-1)
    rival_cols = [
        "ahead_laptime",
        "behind_laptime",
        "ahead_tyre_life",
        "behind_tyre_life",
        "ahead_compound_ord",
        "behind_compound_ord",
    ]
    out[rival_cols] = r[rival_cols]

    out["attack_pressure"] = out["ahead_laptime"] - out[LAPTIME]
    out["cover_pressure"] = out[LAPTIME] - out["behind_laptime"]
    out["ahead_tyre_age_diff"] = out["TyreLife"] - out["ahead_tyre_life"]
    out["behind_tyre_age_diff"] = out["TyreLife"] - out["behind_tyre_life"]
    out["ahead_compound_diff"] = out["compound_ord"] - out["ahead_compound_ord"]
    out["behind_compound_diff"] = out["compound_ord"] - out["behind_compound_ord"]
    out["is_clean_air"] = (out["Position"] == 1).astype("int8")
    out["is_top3"] = (out["Position"] <= 3).astype("int8")

    out = add_online_priors(out)

    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = out[c].astype("float32")
    return out


def make_pit_events(df):
    ev = df.loc[
        df["PitStop"].eq(1),
        [
            "Year",
            "Race",
            "LapNumber",
            "driver_pos_gain_prev",
            "Position_Change",
            LAPDELTA,
            "TyreLife",
        ],
    ].copy()
    ev["pit_pos_gain_event"] = (
        ev["driver_pos_gain_prev"]
        .fillna(-ev["Position_Change"])
        .replace([np.inf, -np.inf], np.nan)
    )
    ev["pit_pos_gain_event"] = ev["pit_pos_gain_event"].clip(-20, 20)
    ev["pit_loss_event"] = ev[LAPDELTA].replace([np.inf, -np.inf], np.nan)
    if len(ev):
        lo, hi = ev["pit_loss_event"].quantile([0.01, 0.99])
        ev["pit_loss_event"] = ev["pit_loss_event"].clip(lo, hi)
        ev["pit_pos_gain_event"] = ev["pit_pos_gain_event"].fillna(
            ev["pit_pos_gain_event"].median()
        )
        ev["pit_loss_event"] = ev["pit_loss_event"].fillna(
            ev["pit_loss_event"].median()
        )
    else:
        ev["pit_pos_gain_event"] = 0.0
        ev["pit_loss_event"] = 0.0
    return ev


def cumulative_grid(events, key_cols, target_keys, max_lap, prefix):
    keys = target_keys[key_cols].drop_duplicates()
    laps = pd.DataFrame({"LapNumber": np.arange(1, max_lap + 1)})
    grid = keys.merge(laps, how="cross")

    if len(events):
        agg = (
            events.groupby(key_cols + ["LapNumber"], observed=True)
            .agg(
                event_count=("pit_loss_event", "size"),
                pos_sum=("pit_pos_gain_event", "sum"),
                loss_sum=("pit_loss_event", "sum"),
            )
            .reset_index()
        )
        grid = grid.merge(agg, on=key_cols + ["LapNumber"], how="left")
    else:
        grid["event_count"] = 0.0
        grid["pos_sum"] = 0.0
        grid["loss_sum"] = 0.0

    for c in ["event_count", "pos_sum", "loss_sum"]:
        grid[c] = grid[c].fillna(0.0)

    grid = grid.sort_values(key_cols + ["LapNumber"])
    for raw, name in [
        ("event_count", "count"),
        ("pos_sum", "pos_sum"),
        ("loss_sum", "loss_sum"),
    ]:
        grid[f"{prefix}_{name}"] = (
            grid.groupby(key_cols, observed=True)[raw].cumsum() - grid[raw]
        )

    return grid[
        key_cols
        + ["LapNumber", f"{prefix}_count", f"{prefix}_pos_sum", f"{prefix}_loss_sum"]
    ]


def add_track_undercut_priors(target_df, history_df, leave_current_group=False):
    out = target_df.copy()
    out["__order__"] = np.arange(len(out))

    ev = make_pit_events(history_df)
    global_pos = float(ev["pit_pos_gain_event"].mean()) if len(ev) else 0.0
    global_loss = float(ev["pit_loss_event"].mean()) if len(ev) else 0.0
    global_pos_std = float(ev["pit_pos_gain_event"].std()) if len(ev) > 1 else 1.0
    global_loss_std = float(ev["pit_loss_event"].std()) if len(ev) > 1 else 1.0
    global_pos_std = global_pos_std if global_pos_std > 1e-6 else 1.0
    global_loss_std = global_loss_std if global_loss_std > 1e-6 else 1.0

    race_stats = (
        ev.groupby("Race", observed=True)
        .agg(
            circuit_count=("pit_loss_event", "size"),
            circuit_pos_sum=("pit_pos_gain_event", "sum"),
            circuit_loss_sum=("pit_loss_event", "sum"),
        )
        .reset_index()
    )

    out = out.merge(race_stats, on="Race", how="left")
    for c in ["circuit_count", "circuit_pos_sum", "circuit_loss_sum"]:
        out[c] = out[c].fillna(0.0)

    if leave_current_group:
        same_stats = (
            ev.groupby(["Year", "Race"], observed=True)
            .agg(
                same_count=("pit_loss_event", "size"),
                same_pos_sum=("pit_pos_gain_event", "sum"),
                same_loss_sum=("pit_loss_event", "sum"),
            )
            .reset_index()
        )
        out = out.merge(same_stats, on=["Year", "Race"], how="left")
        for c in ["same_count", "same_pos_sum", "same_loss_sum"]:
            out[c] = out[c].fillna(0.0)
        out["circuit_count"] = (out["circuit_count"] - out["same_count"]).clip(lower=0)
        out["circuit_pos_sum"] = out["circuit_pos_sum"] - out["same_pos_sum"]
        out["circuit_loss_sum"] = out["circuit_loss_sum"] - out["same_loss_sum"]
        out = out.drop(columns=["same_count", "same_pos_sum", "same_loss_sum"])

    cnt = out["circuit_count"]
    out["hist_circuit_pit_pos_gain_prior"] = safe_divide(out["circuit_pos_sum"], cnt)
    out["hist_circuit_pit_loss_prior"] = safe_divide(out["circuit_loss_sum"], cnt)
    out["hist_circuit_pit_pos_gain_prior"] = out[
        "hist_circuit_pit_pos_gain_prior"
    ].fillna(global_pos)
    out["hist_circuit_pit_loss_prior"] = out["hist_circuit_pit_loss_prior"].fillna(
        global_loss
    )
    out["hist_circuit_event_count_log"] = np.log1p(cnt)
    out["hist_circuit_undercut_score"] = (
        out["hist_circuit_pit_pos_gain_prior"] - global_pos
    ) / global_pos_std - (
        out["hist_circuit_pit_loss_prior"] - global_loss
    ) / global_loss_std

    max_lap = int(
        max(
            out["LapNumber"].max(),
            ev["LapNumber"].max() if len(ev) else out["LapNumber"].max(),
        )
    )
    race_lap = cumulative_grid(ev, ["Race"], out[["Race"]], max_lap, "hist_lap")
    out = out.merge(race_lap, on=["Race", "LapNumber"], how="left")

    if leave_current_group:
        group_lap = cumulative_grid(
            ev, ["Year", "Race"], out[["Year", "Race"]], max_lap, "same_lap"
        )
        out = out.merge(group_lap, on=["Year", "Race", "LapNumber"], how="left")
        for c in ["same_lap_count", "same_lap_pos_sum", "same_lap_loss_sum"]:
            out[c] = out[c].fillna(0.0)
        out["hist_lap_count"] = (
            out["hist_lap_count"].fillna(0.0) - out["same_lap_count"]
        ).clip(lower=0)
        out["hist_lap_pos_sum"] = (
            out["hist_lap_pos_sum"].fillna(0.0) - out["same_lap_pos_sum"]
        )
        out["hist_lap_loss_sum"] = (
            out["hist_lap_loss_sum"].fillna(0.0) - out["same_lap_loss_sum"]
        )
        out = out.drop(
            columns=["same_lap_count", "same_lap_pos_sum", "same_lap_loss_sum"]
        )
    else:
        for c in ["hist_lap_count", "hist_lap_pos_sum", "hist_lap_loss_sum"]:
            out[c] = out[c].fillna(0.0)

    lcnt = out["hist_lap_count"]
    out["hist_lap_pit_pos_gain_prior"] = safe_divide(out["hist_lap_pos_sum"], lcnt)
    out["hist_lap_pit_loss_prior"] = safe_divide(out["hist_lap_loss_sum"], lcnt)
    out["hist_lap_pit_pos_gain_prior"] = out["hist_lap_pit_pos_gain_prior"].fillna(
        out["hist_circuit_pit_pos_gain_prior"]
    )
    out["hist_lap_pit_loss_prior"] = out["hist_lap_pit_loss_prior"].fillna(
        out["hist_circuit_pit_loss_prior"]
    )
    out["hist_lap_event_count_log"] = np.log1p(lcnt)
    out["hist_lap_undercut_score"] = (
        out["hist_lap_pit_pos_gain_prior"] - global_pos
    ) / global_pos_std - (
        out["hist_lap_pit_loss_prior"] - global_loss
    ) / global_loss_std

    for score_col in [
        "hist_circuit_undercut_score",
        "hist_lap_undercut_score",
        "online_undercut_score",
    ]:
        s = out[score_col].fillna(0.0)
        out[f"{score_col}_x_position"] = s * out["Position"]
        out[f"{score_col}_x_tyre_life"] = s * out["TyreLife"]
        out[f"{score_col}_x_clean_air"] = s * out["is_clean_air"]
        out[f"{score_col}_x_attack"] = s * out["attack_pressure"].fillna(0.0)
        out[f"{score_col}_x_cover"] = s * out["cover_pressure"].fillna(0.0)

    out = out.sort_values("__order__").drop(
        columns=[
            "__order__",
            "circuit_pos_sum",
            "circuit_loss_sum",
            "hist_lap_pos_sum",
            "hist_lap_loss_sum",
        ]
    )

    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = out[c].astype("float32")
    return out


def prepare_matrix(df, feature_cols, cat_levels):
    df = set_categories(df, cat_levels)
    x = df[feature_cols].copy()
    num_cols = [c for c in feature_cols if c not in CAT_COLS]
    x[num_cols] = x[num_cols].replace([np.inf, -np.inf], np.nan)
    return x


train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

cat_levels = {}
for c in CAT_COLS:
    cat_levels[c] = pd.Index(
        pd.concat([train[c], test[c]], ignore_index=True).astype(str).unique()
    )

train_base = set_categories(train.drop(columns=[TARGET]), cat_levels)
test_base = set_categories(test.copy(), cat_levels)

train_base = add_base_features(train_base)
test_base = add_base_features(test_base)

if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(train_base, y, groups))
else:
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(train_base, y, groups))

oof = np.zeros(len(train_base), dtype=np.float32)
test_pred = np.zeros(len(test_base), dtype=np.float64)
fold_scores = []
feature_cols = None
cat_features = None
n_jobs = min(16, os.cpu_count() or 1)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    hist = train_base.iloc[tr_idx].copy()

    tr_feat = add_track_undercut_priors(
        train_base.iloc[tr_idx].copy(), hist, leave_current_group=True
    )
    va_feat = add_track_undercut_priors(
        train_base.iloc[va_idx].copy(), hist, leave_current_group=False
    )
    te_feat = add_track_undercut_priors(
        test_base.copy(), hist, leave_current_group=False
    )

    if feature_cols is None:
        drop_cols = {ID_COL}
        feature_cols = [
            c for c in tr_feat.columns if c not in drop_cols and not c.startswith("__")
        ]
        cat_features = [c for c in CAT_COLS if c in feature_cols]

    X_tr = prepare_matrix(tr_feat, feature_cols, cat_levels)
    X_va = prepare_matrix(va_feat, feature_cols, cat_levels)
    X_te = prepare_matrix(te_feat, feature_cols, cat_levels)
    y_tr, y_va = y[tr_idx], y[va_idx]

    pos = max(float(y_tr.sum()), 1.0)
    neg = float(len(y_tr) - y_tr.sum())

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE + fold,
        n_jobs=n_jobs,
        force_col_wise=True,
        verbose=-1,
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    va_pred = model.predict_proba(X_va, num_iteration=model.best_iteration_)[:, 1]
    te_pred = model.predict_proba(X_te, num_iteration=model.best_iteration_)[:, 1]

    oof[va_idx] = va_pred.astype(np.float32)
    test_pred += te_pred / N_SPLITS

    fold_auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(WORK_DIR / "submission.csv", index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train_base)),
        "target": y,
        "prediction": oof,
    }
).to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

submission[[ID_COL, TARGET]].to_csv(
    WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_auc": float(cv_auc),
    "fold_auc": [float(x) for x in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000585"],
}
with open(WORK_DIR / "result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
