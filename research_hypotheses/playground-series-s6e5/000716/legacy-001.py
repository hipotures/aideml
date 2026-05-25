import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"

os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_peer_context(df):
    d = df.copy()
    keys = ["Year", "Race", "LapNumber"]
    nonpit = d[d["PitStop"].eq(0)]
    peer = (
        nonpit.groupby(keys, observed=True)["LapTime (s)"]
        .median()
        .rename("peer_lap_median")
    )
    lap_all = (
        d.groupby(keys, observed=True)["LapTime (s)"].median().rename("lap_median_all")
    )
    d = d.merge(peer.reset_index(), on=keys, how="left")
    d = d.merge(lap_all.reset_index(), on=keys, how="left")
    d["peer_lap_median"] = d["peer_lap_median"].fillna(d["lap_median_all"])
    race_base = d.groupby(["Year", "Race"], observed=True)["peer_lap_median"].transform(
        "median"
    )
    d["race_lap_base"] = race_base.replace(0, np.nan).fillna(
        d["peer_lap_median"].median()
    )
    d["lap_slow_ratio"] = d["peer_lap_median"] / d["race_lap_base"]
    d["sc_proxy"] = (
        (d["lap_slow_ratio"] > 1.10)
        | ((d["peer_lap_median"] - d["race_lap_base"]) > 8.0)
    ).astype(np.int8)
    return d


def shrunk_table(df, value, cols, global_value, shrink=20.0):
    z = df.dropna(subset=[value])
    if len(z) == 0:
        return pd.DataFrame(columns=cols + ["prior"])
    g = z.groupby(cols, observed=True)[value].agg(["median", "count"]).reset_index()
    w = g["count"] / (g["count"] + shrink)
    g["prior"] = w * g["median"] + (1.0 - w) * global_value
    return g[cols + ["prior"]]


def build_metric_prior(df, value, global_default):
    clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[value])
    global_value = float(clean[value].median()) if len(clean) else float(global_default)
    specs = [
        ["Year", "Race", "Compound", "sc_proxy"],
        ["Race", "Compound", "sc_proxy"],
        ["Race", "sc_proxy"],
        ["Race"],
        ["Compound", "sc_proxy"],
    ]
    return {
        "global": global_value,
        "tables": [
            (cols, shrunk_table(clean, value, cols, global_value)) for cols in specs
        ],
    }


def estimate_pit_economics(ref):
    d = add_peer_context(ref)
    d = d.sort_values(["Year", "Race", "Driver", "LapNumber"]).reset_index(drop=True)
    g = d.groupby(["Year", "Race", "Driver"], sort=False, observed=True)

    for col in ["LapNumber", "LapTime (s)", "peer_lap_median"]:
        d[f"n1_{col}"] = g[col].shift(-1)
        d[f"n2_{col}"] = g[col].shift(-2)
    d["p1_LapTime (s)"] = g["LapTime (s)"].shift(1)
    d["p1_TyreLife"] = g["TyreLife"].shift(1)
    d["p1_PitStop"] = g["PitStop"].shift(1)

    stops = d[d["PitStop"].eq(1)].copy()
    cont1 = stops["n1_LapNumber"].eq(stops["LapNumber"] + 1)
    cont2 = stops["n2_LapNumber"].eq(stops["LapNumber"] + 2)

    stops["pit_loss"] = (stops["LapTime (s)"] - stops["peer_lap_median"]).clip(0, 80)
    stops["warmup_penalty"] = np.where(
        cont1,
        stops["n1_LapTime (s)"] - stops["n1_peer_lap_median"],
        np.nan,
    )
    stops["warmup_penalty"] = pd.Series(stops["warmup_penalty"]).clip(-5, 25)
    stops["undercut_benefit"] = np.where(
        cont2,
        stops["n2_peer_lap_median"] - stops["n2_LapTime (s)"],
        np.nan,
    )
    stops["undercut_benefit"] = pd.Series(stops["undercut_benefit"]).clip(-10, 30)

    deg_mask = (
        d["PitStop"].eq(0)
        & d["p1_PitStop"].eq(0)
        & (d["TyreLife"] > 1)
        & ((d["TyreLife"] - d["p1_TyreLife"]).between(0.5, 1.5))
    )
    deg = d.loc[
        deg_mask,
        ["Year", "Race", "Compound", "sc_proxy", "LapTime (s)", "p1_LapTime (s)"],
    ].copy()
    deg["deg_slope"] = (deg["LapTime (s)"] - deg["p1_LapTime (s)"]).clip(-3, 5)

    return {
        "pit_loss": build_metric_prior(
            stops, "pit_loss", stops["pit_loss"].median() if len(stops) else 22.0
        ),
        "warmup_penalty": build_metric_prior(stops, "warmup_penalty", 0.0),
        "undercut_benefit": build_metric_prior(stops, "undercut_benefit", 0.0),
        "deg_slope": build_metric_prior(deg, "deg_slope", 0.0),
    }


def lookup_prior(frame, spec):
    out = np.full(len(frame), spec["global"], dtype=np.float32)
    missing = np.ones(len(frame), dtype=bool)
    for cols, table in spec["tables"]:
        if table.empty:
            continue
        vals = (
            frame[cols]
            .merge(table, on=cols, how="left")["prior"]
            .to_numpy(dtype=np.float32)
        )
        ok = missing & np.isfinite(vals)
        out[ok] = vals[ok]
        missing[ok] = False
    return out


def make_features(df, priors):
    f = add_peer_context(df)

    total_laps = (f["LapNumber"] / f["RaceProgress"].clip(lower=0.01)).replace(
        [np.inf, -np.inf], np.nan
    )
    total_laps = total_laps.fillna(total_laps.median()).clip(1, 120)
    f["race_total_laps_est"] = total_laps
    f["laps_remaining_est"] = (total_laps - f["LapNumber"]).clip(0, 120)
    f["tyre_life_frac"] = f["TyreLife"] / total_laps.clip(lower=1)
    f["stint_lap_ratio"] = f["TyreLife"] / f["LapNumber"].clip(lower=1)
    f["degradation_per_tyre_lap"] = f["Cumulative_Degradation"] / f["TyreLife"].clip(
        lower=1
    )
    f["lap_delta_abs"] = f["LapTime_Delta"].abs()
    f["position_loss_pressure"] = np.maximum(f["Position_Change"], 0)
    f["position_gain_pressure"] = np.maximum(-f["Position_Change"], 0)

    normal = f.copy()
    slow = f.copy()
    normal["sc_proxy"] = 0
    slow["sc_proxy"] = 1

    f["learned_pit_loss_normal"] = lookup_prior(normal, priors["pit_loss"])
    f["learned_pit_loss_sc"] = lookup_prior(slow, priors["pit_loss"])
    f["learned_pit_loss"] = lookup_prior(f, priors["pit_loss"])
    f["learned_warmup_penalty"] = lookup_prior(f, priors["warmup_penalty"])
    f["learned_undercut_benefit"] = lookup_prior(f, priors["undercut_benefit"])
    f["learned_deg_slope"] = lookup_prior(f, priors["deg_slope"])

    f["sc_adjusted_loss_saving"] = (
        f["learned_pit_loss_normal"] - f["learned_pit_loss_sc"]
    )
    f["learned_stop_now_cost"] = (
        f["learned_pit_loss"]
        + np.maximum(f["learned_warmup_penalty"], 0)
        - np.maximum(f["learned_undercut_benefit"], 0)
    )
    observed_deg = f["degradation_per_tyre_lap"].clip(-5, 8)
    f["learned_wait_one_lap_cost"] = (
        f["learned_deg_slope"]
        + 0.15 * observed_deg
        + 0.02 * f["TyreLife"] * np.maximum(f["learned_deg_slope"], 0)
    )
    f["learned_stop_vs_wait_value"] = (
        f["learned_wait_one_lap_cost"] - f["learned_stop_now_cost"]
    )
    f["learned_undercut_window_score"] = (
        f["learned_undercut_benefit"]
        + np.maximum(f["learned_deg_slope"], 0) * f["TyreLife"]
    ) / (f["learned_pit_loss"] + 1.0)
    f["sc_economic_stop_value"] = f["sc_proxy"] * f["sc_adjusted_loss_saving"]

    drop_cols = [ID_COL, TARGET, "lap_median_all", "peer_lap_median", "race_lap_base"]
    f = f.drop(columns=[c for c in drop_cols if c in f.columns])
    return f.replace([np.inf, -np.inf], np.nan)


def set_categories(train_x, valid_x=None, test_x=None):
    frames = [train_x]
    if valid_x is not None:
        frames.append(valid_x)
    if test_x is not None:
        frames.append(test_x)
    cat_cols = [c for c in train_x.columns if train_x[c].dtype == "object"]
    for c in cat_cols:
        cats = (
            pd.concat([z[c].astype("object") for z in frames], axis=0)
            .astype("category")
            .cat.categories
        )
        train_x[c] = pd.Categorical(train_x[c], categories=cats)
        if valid_x is not None:
            valid_x[c] = pd.Categorical(valid_x[c], categories=cats)
        if test_x is not None:
            test_x[c] = pd.Categorical(test_x[c], categories=cats)
    return cat_cols


try:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
    splits = list(splitter.split(train, train[TARGET].astype(int), groups))
except Exception:
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(splitter.split(train, train[TARGET].astype(int)))

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

y = train[TARGET].astype(int).to_numpy()
oof = np.zeros(len(train), dtype=np.float32)
fold_scores = []
best_iters = []


def make_model(n_estimators=3000):
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=64,
        min_child_samples=80,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=2.00,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    ref = train.iloc[tr_idx].reset_index(drop=True)
    priors = estimate_pit_economics(ref)

    x_tr = make_features(train.iloc[tr_idx].reset_index(drop=True), priors)
    x_va = make_features(train.iloc[va_idx].reset_index(drop=True), priors)
    cat_cols = set_categories(x_tr, x_va)

    model = make_model()
    model.fit(
        x_tr,
        y[tr_idx],
        eval_set=[(x_va, y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(150, verbose=False), log_evaluation(0)],
    )

    pred = model.predict_proba(x_va)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(y[va_idx], pred)
    fold_scores.append(float(auc))
    best_iters.append(int(model.best_iteration_ or model.n_estimators))
    print(f"fold {fold} roc_auc={auc:.6f} best_iteration={best_iters[-1]}")

    del ref, priors, x_tr, x_va, model
    gc.collect()

cv_auc = roc_auc_score(y, oof)
pd.DataFrame({"row": np.arange(len(train)), "target": y, "prediction": oof}).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_priors = estimate_pit_economics(train)
x_full = make_features(train, final_priors)
x_test = make_features(test, final_priors)
cat_cols = set_categories(x_full, test_x=x_test)

final_n_estimators = int(np.clip(np.mean(best_iters) * 1.05, 100, 3000))
final_model = make_model(n_estimators=final_n_estimators)
final_model.fit(
    x_full,
    y,
    categorical_feature=cat_cols,
)

test_pred = final_model.predict_proba(x_test)[:, 1].clip(0, 1)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
    "final_n_estimators": final_n_estimators,
    "research_hypotheses_llm_claimed_used": ["000716"],
    "saved_files": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
    ],
}
print(json.dumps(result, indent=2))
