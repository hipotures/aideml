import os
import re
import gc
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
RANDOM_STATE = 559
N_SPLITS = 5
N_REGIMES = 4
N_JOBS = min(8, os.cpu_count() or 1)


def make_safe_columns(columns):
    seen, out = {}, []
    for col in columns:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_") or "col"
        name = base
        if name in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        out.append(name)
    return out


def add_features(df):
    df = df.copy()
    for c in ["Compound", "Driver", "Race"]:
        df[c] = df[c].astype(str)

    df["Year_Race"] = df["Year"].astype(str) + "_" + df["Race"]
    df["Year_cat"] = df["Year"].astype(str)
    df["Driver_Race"] = df["Driver"] + "_" + df["Race"]
    df["Race_Compound"] = df["Race"] + "_" + df["Compound"]
    df["Compound_Stint"] = df["Compound"] + "_" + df["Stint"].astype(str)

    work = df.sort_values(["Year", "Race", "Driver", "LapNumber", "id"]).copy()
    grp = work.groupby(["Year", "Race", "Driver"], sort=False)

    work["prev_PitStop"] = grp["PitStop"].shift(1).fillna(0)
    work["prev_LapTime_s"] = grp["LapTime_s"].shift(1)
    work["prev_LapTime_Delta"] = grp["LapTime_Delta"].shift(1)
    work["prev_Position"] = grp["Position"].shift(1)
    work["prev_TyreLife"] = grp["TyreLife"].shift(1)
    work["prev_Compound"] = grp["Compound"].shift(1).fillna("__START__")
    work["lap_gap"] = (work["LapNumber"] - grp["LapNumber"].shift(1)).fillna(1)
    work["prior_pit_count"] = grp["PitStop"].cumsum() - work["PitStop"]
    work["driver_lap_count"] = grp.cumcount() + 1
    work["compound_changed"] = (work["Compound"] != work["prev_Compound"]).astype(
        "int8"
    )
    work["first_lap_after_pit"] = (work["prev_PitStop"] > 0.5).astype("int8")
    work = work.sort_index()

    tyre = np.maximum(work["TyreLife"].astype(float), 1.0)
    lap = np.maximum(work["LapNumber"].astype(float), 1.0)
    rp = np.maximum(work["RaceProgress"].astype(float), 0.01)

    work["is_soft"] = (work["Compound"] == "SOFT").astype("int8")
    work["is_medium"] = (work["Compound"] == "MEDIUM").astype("int8")
    work["is_hard"] = (work["Compound"] == "HARD").astype("int8")
    work["is_wet_or_inter"] = (
        work["Compound"].isin(["WET", "INTERMEDIATE"]).astype("int8")
    )
    work["compound_order"] = (
        work["Compound"]
        .map({"HARD": 1, "MEDIUM": 2, "SOFT": 3, "INTERMEDIATE": 0, "WET": 0})
        .fillna(0)
        .astype(float)
    )

    work["deg_per_tyre_life"] = work["Cumulative_Degradation"] / tyre
    work["deg_curvature_proxy"] = work["Cumulative_Degradation"] / (tyre * tyre)
    work["lap_delta_per_tyre_life"] = work["LapTime_Delta"] / tyre
    work["tyre_life_frac_of_race"] = work["TyreLife"] / lap
    work["race_progress_left"] = 1.0 - work["RaceProgress"]
    work["laps_remaining_est"] = (work["LapNumber"] / rp) - work["LapNumber"]
    work["position_x_progress"] = work["Position"] * work["RaceProgress"]
    work["stint_x_tyre_life"] = work["Stint"] * work["TyreLife"]
    work["degradation_x_progress"] = (
        work["Cumulative_Degradation"] * work["RaceProgress"]
    )
    work["delta_vs_prev_lap"] = work["LapTime_s"] - work["prev_LapTime_s"]
    work["position_vs_prev"] = work["Position"] - work["prev_Position"]
    work["warmup_lap"] = (
        (work["TyreLife"] <= 2) | (work["first_lap_after_pit"] == 1)
    ).astype("int8")

    return work


def build_regime_table(df):
    g = df.groupby("Year_Race", observed=True)

    base = g.agg(
        rows=("LapNumber", "size"),
        max_lap=("LapNumber", "max"),
        mean_lap=("LapNumber", "mean"),
        mean_progress=("RaceProgress", "mean"),
        mean_tyre_life=("TyreLife", "mean"),
        p75_tyre_life=("TyreLife", lambda s: s.quantile(0.75)),
        max_tyre_life=("TyreLife", "max"),
        mean_deg=("Cumulative_Degradation", "mean"),
        std_deg=("Cumulative_Degradation", "std"),
        mean_deg_per_life=("deg_per_tyre_life", "mean"),
        mean_deg_curve=("deg_curvature_proxy", "mean"),
        pitstop_rate=("PitStop", "mean"),
        wet_or_inter_share=("is_wet_or_inter", "mean"),
        mean_lap_delta=("LapTime_Delta", "mean"),
        std_lap_delta=("LapTime_Delta", "std"),
        mean_position_change=("Position_Change", "mean"),
    )

    warm = (
        df[df["warmup_lap"] == 1]
        .groupby("Year_Race", observed=True)
        .agg(
            warm_delta_mean=("LapTime_Delta", "mean"),
            warm_delta_std=("LapTime_Delta", "std"),
            warm_position_change=("Position_Change", "mean"),
        )
    )

    driver_stints = df.groupby(["Year_Race", "Driver"], observed=True)["Stint"].max()
    stint_level = driver_stints.groupby(level=0)
    one_stop = pd.DataFrame(
        {
            "driver_max_stint_mean": stint_level.mean(),
            "driver_max_stint_std": stint_level.std(),
            "one_stop_proxy": stint_level.apply(lambda s: float((s <= 2).mean())),
            "multi_stop_proxy": stint_level.apply(lambda s: float((s >= 3).mean())),
        }
    )

    stint_lengths = df.groupby(["Year_Race", "Driver", "Stint"], observed=True)[
        "TyreLife"
    ].max()
    stint_stats = stint_lengths.groupby(level=0).agg(["mean", "std", "median", "max"])
    stint_stats.columns = [f"stint_len_{c}" for c in stint_stats.columns]

    compound_counts = (
        df.groupby(["Year_Race", "Compound"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    compound_mix = compound_counts.div(compound_counts.sum(axis=1), axis=0).add_prefix(
        "compound_share_"
    )

    table = base.join([warm, one_stop, stint_stats, compound_mix], how="left")
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.fillna(table.median(numeric_only=True)).fillna(0)
    return table


def fit_regime_labels(train_full, seed):
    table = build_regime_table(train_full)
    if len(table) < 2:
        return np.zeros(len(train_full), dtype=int), 1

    k = min(N_REGIMES, len(table))
    z = StandardScaler().fit_transform(table.values)
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(z)
    by_race = pd.Series(labels, index=table.index)
    row_labels = train_full["Year_Race"].map(by_race).fillna(0).astype(int).to_numpy()
    return row_labels, k


def binary_lgb(seed, n_estimators, y):
    pos = float(np.sum(y))
    neg = float(len(y) - pos)
    scale = min(50.0, max(1.0, neg / max(pos, 1.0)))
    return LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=48,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        scale_pos_weight=scale,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_row_wise=True,
    )


def regime_lgb(seed, k):
    params = dict(
        n_estimators=160,
        learning_rate=0.06,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_row_wise=True,
    )
    if k > 2:
        params.update(objective="multiclass", num_class=k)
    else:
        params.update(objective="binary")
    return LGBMClassifier(**params)


def aligned_proba(model, X, k):
    if model is None or k == 1:
        return np.ones((len(X), 1), dtype=float)

    raw = np.asarray(model.predict_proba(X))
    if raw.ndim == 1:
        raw = np.vstack([1.0 - raw, raw]).T

    out = np.zeros((len(X), k), dtype=float)
    for j, cls in enumerate(model.classes_):
        cls = int(cls)
        if 0 <= cls < k:
            out[:, cls] = raw[:, j]

    sums = out.sum(axis=1)
    empty = sums <= 0
    out[empty, :] = 1.0 / k
    out[~empty, :] /= sums[~empty, None]
    return out


os.makedirs(WORKING_DIR, exist_ok=True)

train_raw = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test_raw = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train_raw["PitNextLap"].astype(int).to_numpy()
orig_features = [c for c in train_raw.columns if c != "PitNextLap"]
safe_features = make_safe_columns(orig_features)
rename_map = dict(zip(orig_features, safe_features))

train_x = train_raw[orig_features].rename(columns=rename_map)
test_x = test_raw.rename(columns=rename_map)

n_train = len(train_x)
all_x = pd.concat([train_x, test_x], axis=0, ignore_index=True)
all_x = add_features(all_x)

cat_cols = [
    c
    for c in all_x.columns
    if all_x[c].dtype == "object" or str(all_x[c].dtype).startswith("string")
]
for c in cat_cols:
    all_x[c] = all_x[c].astype("string").fillna("__MISSING__").astype("category")

model_features = [c for c in all_x.columns if c != "id"]
num_cols = [c for c in model_features if c not in cat_cols]
all_x[num_cols] = all_x[num_cols].replace([np.inf, -np.inf], np.nan)
medians = all_x.loc[: n_train - 1, num_cols].median(numeric_only=True)
all_x[num_cols] = all_x[num_cols].fillna(medians).fillna(0)

train_full = all_x.iloc[:n_train].reset_index(drop=True)
test_full = all_x.iloc[n_train:].reset_index(drop=True)
X = train_full[model_features]
X_test = test_full[model_features]

oof = np.zeros(n_train, dtype=float)
test_pred = np.zeros(len(test_full), dtype=float)
fold_aucs = []

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    full_tr = train_full.iloc[tr_idx]

    regime_labels, k = fit_regime_labels(full_tr, RANDOM_STATE + fold)

    reg_model = None
    if k > 1:
        reg_model = regime_lgb(RANDOM_STATE + 100 + fold, k)
        reg_model.fit(X_tr, regime_labels, categorical_feature=cat_cols)

    reg_va = aligned_proba(reg_model, X_va, k)
    reg_te = aligned_proba(reg_model, X_test, k)

    global_model = binary_lgb(RANDOM_STATE + 200 + fold, 420, y_tr)
    global_model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    global_va = global_model.predict_proba(X_va)[:, 1]
    global_te = global_model.predict_proba(X_test)[:, 1]

    expert_va_mix = np.zeros(len(va_idx), dtype=float)
    expert_te_mix = np.zeros(len(test_full), dtype=float)

    for r in range(k):
        mask = regime_labels == r
        yr = y_tr[mask]
        if mask.sum() >= 2500 and yr.sum() >= 20 and len(np.unique(yr)) == 2:
            expert = binary_lgb(RANDOM_STATE + 500 + fold * 10 + r, 280, yr)
            expert.fit(X_tr.iloc[mask], yr, categorical_feature=cat_cols)
            pv = expert.predict_proba(X_va)[:, 1]
            pt = expert.predict_proba(X_test)[:, 1]
        else:
            pv, pt = global_va, global_te

        expert_va_mix += reg_va[:, r] * pv
        expert_te_mix += reg_te[:, r] * pt

    fold_va = np.clip(0.35 * global_va + 0.65 * expert_va_mix, 1e-6, 1 - 1e-6)
    fold_te = np.clip(0.35 * global_te + 0.65 * expert_te_mix, 1e-6, 1 - 1e-6)

    oof[va_idx] = fold_va
    test_pred += fold_te / N_SPLITS

    auc = roc_auc_score(y_va, fold_va)
    fold_aucs.append(float(auc))
    print(f"Fold {fold} ROC AUC: {auc:.6f} using {k} regimes")

    del global_model, reg_model
    gc.collect()

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_aucs):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

target_col = [c for c in sample.columns if c != "id"][0]
submission = sample.copy()
submission[target_col] = np.clip(test_pred, 1e-6, 1 - 1e-6)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

oof_df = pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": np.clip(oof, 1e-6, 1 - 1e-6),
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_aucs,
    "research_hypotheses_llm_claimed_used": ["000559"],
}
with open(os.path.join(WORKING_DIR, "result_review.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result))
