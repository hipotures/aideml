import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 2026
INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
LAP_COLS = ["Race", "Year", "LapNumber"]
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.rename(columns={"LapTime (s)": "LapTime_s"})
test = test.rename(columns={"LapTime (s)": "LapTime_s"})


def add_features(df):
    df = df.copy()
    eps = 1e-6
    df["is_wet"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    df["degradation_per_tyre_lap"] = df["Cumulative_Degradation"] / (
        df["TyreLife"] + eps
    )
    df["lap_time_per_lap"] = df["LapTime_s"] / (df["LapNumber"] + eps)
    df["abs_laptime_delta"] = df["LapTime_Delta"].abs()
    df["progress_x_tyre"] = df["RaceProgress"] * df["TyreLife"]
    df["stint_x_tyre"] = df["Stint"] * df["TyreLife"]
    df["position_x_progress"] = df["Position"] * df["RaceProgress"]
    df["position_change_abs"] = df["Position_Change"].abs()

    progress_bins = [0.0, 0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 1.01]
    df["progress_bin"] = pd.cut(
        df["RaceProgress"], bins=progress_bins, labels=False, include_lowest=True
    ).astype("int16")

    df["tyre_life_bin"] = pd.cut(
        df["TyreLife"],
        bins=[0, 5, 10, 16, 24, 35, 100],
        labels=False,
        include_lowest=True,
    ).astype("int16")

    df["stint_bin"] = pd.cut(
        df["Stint"],
        bins=[0, 1, 2, 3, 99],
        labels=["1", "2", "3", "4+"],
        include_lowest=True,
    ).astype(str)

    df["wet_dry"] = np.where(df["is_wet"] == 1, "wet", "dry")
    return df


train_fe = add_features(train)
test_fe = add_features(test)
y = train_fe[TARGET].astype(int).to_numpy()

cat_cols = [
    "Compound",
    "Driver",
    "Race",
    "Year",
    "progress_bin",
    "tyre_life_bin",
    "stint_bin",
    "wet_dry",
]
for col in cat_cols:
    tr = train_fe[col].astype("string").fillna("missing")
    te = test_fe[col].astype("string").fillna("missing")
    cats = pd.Index(pd.concat([tr, te], ignore_index=True).unique())
    train_fe[col] = pd.Categorical(tr, categories=cats)
    test_fe[col] = pd.Categorical(te, categories=cats)

feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
cat_feature_cols = [c for c in cat_cols if c in feature_cols]


def mode_or_first(s):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else s.iloc[0]


def make_lap_context(df, target=None):
    tmp = df.copy()
    if target is not None:
        tmp["__target__"] = np.asarray(target)

    g = tmp.groupby(LAP_COLS, observed=True, sort=False)
    lap = g.agg(
        n_rows=("LapNumber", "size"),
        race_progress=("RaceProgress", "mean"),
        wet_share=("is_wet", "mean"),
        stint_median=("Stint", "median"),
        dominant_compound=("Compound", mode_or_first),
    ).reset_index()

    progress_bins = [0.0, 0.12, 0.25, 0.40, 0.55, 0.70, 0.85, 1.01]
    lap["progress_ctx"] = (
        pd.cut(
            lap["race_progress"], bins=progress_bins, labels=False, include_lowest=True
        )
        .astype("int16")
        .astype(str)
    )
    lap["wet_ctx"] = np.where(lap["wet_share"] > 0.20, "wet", "dry")
    lap["stint_ctx"] = pd.cut(
        lap["stint_median"],
        bins=[0, 1, 2, 3, 99],
        labels=["1", "2", "3", "4+"],
        include_lowest=True,
    ).astype(str)
    lap["compound_ctx"] = lap["dominant_compound"].astype(str)
    lap["race_ctx"] = lap["Race"].astype(str)

    if target is not None:
        counts = g["__target__"].sum().reset_index(name="target_count")
        lap = lap.merge(counts, on=LAP_COLS, how="left")
    return lap


CONTEXT_LEVELS = [
    ["race_ctx", "progress_ctx", "wet_ctx", "compound_ctx", "stint_ctx"],
    ["race_ctx", "progress_ctx", "wet_ctx", "stint_ctx"],
    ["progress_ctx", "wet_ctx", "compound_ctx", "stint_ctx"],
    ["progress_ctx", "wet_ctx", "stint_ctx"],
    ["progress_ctx", "wet_ctx"],
    ["progress_ctx"],
]


def fit_lap_prior(df, target):
    lap = make_lap_context(df, target)
    global_rate = lap["target_count"].sum() / max(lap["n_rows"].sum(), 1)
    maps = []
    smooth_rows = 80.0

    for cols in CONTEXT_LEVELS:
        agg = lap.groupby(cols, observed=True).agg(
            pitters=("target_count", "sum"),
            rows=("n_rows", "sum"),
        )
        rate = (agg["pitters"] + smooth_rows * global_rate) / (
            agg["rows"] + smooth_rows
        )
        maps.append((cols, rate.to_dict()))

    return {"global_rate": float(global_rate), "maps": maps}


def map_context_rates(lap, prior):
    rates = pd.Series(np.nan, index=lap.index, dtype=float)
    for cols, mapping in prior["maps"]:
        if len(cols) == 1:
            vals = lap[cols[0]].astype(str).map(mapping)
        else:
            keys = list(map(tuple, lap[cols].astype(str).to_numpy()))
            vals = pd.Series([mapping.get(k, np.nan) for k in keys], index=lap.index)
        rates = rates.fillna(vals)
    return rates.fillna(prior["global_rate"]).to_numpy()


def predict_lap_prior_counts(df, prior):
    lap = make_lap_context(df, None)
    rates = map_context_rates(lap, prior)
    lap["prior_count"] = np.clip(
        rates * lap["n_rows"].to_numpy(), 0.0, lap["n_rows"].to_numpy()
    )
    return lap[LAP_COLS + ["prior_count"]]


def normalize_with_lap_prior(pred, df, prior, alpha=0.40):
    tmp = df[LAP_COLS].copy()
    tmp["base_pred"] = np.asarray(pred, dtype=float)

    lap_prior = predict_lap_prior_counts(df, prior)
    tmp = tmp.merge(lap_prior, on=LAP_COLS, how="left")

    base_sum = (
        tmp.groupby(LAP_COLS, observed=True)["base_pred"].transform("sum").to_numpy()
    )
    prior_count = (
        tmp["prior_count"].fillna(pd.Series(base_sum, index=tmp.index)).to_numpy()
    )

    scale = np.ones_like(base_sum, dtype=float)
    mask = base_sum > 1e-12
    scale[mask] = prior_count[mask] / base_sum[mask]
    scale = np.clip(scale, 0.25, 4.0)

    scaled = np.clip(np.asarray(pred) * scale, 1e-6, 1 - 1e-6)
    blended = (1.0 - alpha) * np.asarray(pred) + alpha * scaled
    return np.clip(blended, 1e-6, 1 - 1e-6)


groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)
if HAS_SGKF:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(train_fe, y, groups))
else:
    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(train_fe, y, groups))

params = dict(
    objective="binary",
    n_estimators=2500,
    learning_rate=0.035,
    num_leaves=64,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=min(8, os.cpu_count() or 1),
    verbosity=-1,
)

oof = np.zeros(len(train_fe), dtype=float)
fold_aucs = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr = train_fe.iloc[tr_idx][feature_cols]
    X_va = train_fe.iloc[va_idx][feature_cols]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_feature_cols,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    base_va = model.predict_proba(X_va)[:, 1]
    prior = fit_lap_prior(train_fe.iloc[tr_idx], y_tr)
    pred_va = normalize_with_lap_prior(base_va, train_fe.iloc[va_idx], prior)

    oof[va_idx] = pred_va
    auc = roc_auc_score(y_va, pred_va)
    fold_aucs.append(float(auc))
    best_iters.append(model.best_iteration_ or params["n_estimators"])
    print(f"fold {fold} normalized_auc={auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"OOF normalized ROC AUC: {cv_auc:.6f}")

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_estimators = int(
    np.clip(np.median(best_iters) * 1.10, 100, params["n_estimators"])
)
final_params = params.copy()
final_params["n_estimators"] = final_estimators

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(
    train_fe[feature_cols],
    y,
    categorical_feature=cat_feature_cols,
)

base_test = final_model.predict_proba(test_fe[feature_cols])[:, 1]
full_prior = fit_lap_prior(train_fe, y)
test_pred = normalize_with_lap_prior(base_test, test_fe, full_prior)

target_col = [c for c in sample.columns if c != ID_COL][0]
submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "cv_auc": float(cv_auc),
    "fold_auc": fold_aucs,
    "research_hypotheses_llm_claimed_used": ["000194"],
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
print(json.dumps(result))
