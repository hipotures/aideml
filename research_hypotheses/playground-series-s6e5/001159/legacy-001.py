import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

RANDOM_STATE = 2026
TARGET = "PitNextLap"
ID_COL = "id"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_prior_bins(df):
    df = df.copy()
    df["TyreLifeBin"] = pd.cut(
        df["TyreLife"],
        bins=[-np.inf, 3, 7, 12, 18, 25, 35, np.inf],
        labels=[
            "tl_00_03",
            "tl_04_07",
            "tl_08_12",
            "tl_13_18",
            "tl_19_25",
            "tl_26_35",
            "tl_36p",
        ],
    ).astype(str)
    df["RacePhase"] = pd.cut(
        df["RaceProgress"],
        bins=[-np.inf, 0.15, 0.35, 0.55, 0.75, 0.90, np.inf],
        labels=[
            "phase_start",
            "phase_early",
            "phase_mid",
            "phase_late",
            "phase_end",
            "phase_finish",
        ],
    ).astype(str)
    return df


train = add_prior_bins(train)
test = add_prior_bins(test)
y = train[TARGET].astype(float).values

PRIOR_SPECS = [
    ("drv_cmp", ["Driver", "Compound"], ["Compound"], 35.0, 120.0),
    (
        "race_cmp_stint",
        ["Race", "Compound", "Stint"],
        ["Race", "Compound"],
        45.0,
        140.0,
    ),
    (
        "cmp_life_phase",
        ["Compound", "TyreLifeBin", "RacePhase"],
        ["Compound", "RacePhase"],
        55.0,
        160.0,
    ),
]


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def ordered_eb_features(df, target_values, specs, global_prior):
    n = len(df)
    out = pd.DataFrame(index=df.index)
    work = df.copy()
    work["_target"] = np.asarray(target_values, dtype=float)
    order_cols = [ID_COL] if ID_COL in work.columns else []
    if "Year" in work.columns:
        order_cols = ["Year"] + order_cols
    if order_cols:
        work = work.sort_values(order_cols, kind="mergesort")
    else:
        work = work.sort_index(kind="mergesort")

    for name, keys, parent_keys, cell_strength, parent_strength in specs:
        parent_cnt = work.groupby(parent_keys, observed=False).cumcount().astype(float)
        parent_sum = (
            work.groupby(parent_keys, observed=False)["_target"].cumsum()
            - work["_target"]
        )
        parent_mean = (parent_sum + parent_strength * global_prior) / (
            parent_cnt + parent_strength
        )

        cell_cnt = work.groupby(keys, observed=False).cumcount().astype(float)
        cell_sum = (
            work.groupby(keys, observed=False)["_target"].cumsum() - work["_target"]
        )
        cell_mean = (cell_sum + cell_strength * parent_mean) / (
            cell_cnt + cell_strength
        )

        tmp = pd.DataFrame(index=work.index)
        tmp[f"eb_{name}_mean"] = cell_mean.values
        tmp[f"eb_{name}_logit"] = safe_logit(cell_mean.values)
        tmp[f"eb_{name}_count"] = np.log1p(cell_cnt.values)
        tmp[f"eb_{name}_parent_mean"] = parent_mean.values
        tmp[f"eb_{name}_parent_count"] = np.log1p(parent_cnt.values)
        out = out.join(tmp)

    return out.loc[df.index].reset_index(drop=True)


def aggregate_eb_features(fit_df, fit_y, pred_df, specs, global_prior):
    out = pd.DataFrame(index=pred_df.index)
    fit = fit_df.copy()
    fit["_target"] = np.asarray(fit_y, dtype=float)

    for name, keys, parent_keys, cell_strength, parent_strength in specs:
        parent_stats = (
            fit.groupby(parent_keys, observed=False)["_target"]
            .agg(["sum", "count"])
            .reset_index()
        )
        parent_stats = parent_stats.rename(
            columns={"sum": "_parent_sum", "count": "_parent_count"}
        )
        cell_stats = (
            fit.groupby(keys, observed=False)["_target"]
            .agg(["sum", "count"])
            .reset_index()
        )
        cell_stats = cell_stats.rename(
            columns={"sum": "_cell_sum", "count": "_cell_count"}
        )

        tmp = pred_df[keys].copy()
        tmp["_row_order"] = np.arange(len(pred_df))
        tmp = tmp.merge(parent_stats, on=parent_keys, how="left")
        tmp = tmp.merge(cell_stats, on=keys, how="left")
        tmp = tmp.sort_values("_row_order")

        parent_count = tmp["_parent_count"].fillna(0).astype(float).values
        parent_sum = tmp["_parent_sum"].fillna(0).astype(float).values
        cell_count = tmp["_cell_count"].fillna(0).astype(float).values
        cell_sum = tmp["_cell_sum"].fillna(0).astype(float).values

        parent_mean = (parent_sum + parent_strength * global_prior) / (
            parent_count + parent_strength
        )
        cell_mean = (cell_sum + cell_strength * parent_mean) / (
            cell_count + cell_strength
        )

        out[f"eb_{name}_mean"] = cell_mean
        out[f"eb_{name}_logit"] = safe_logit(cell_mean)
        out[f"eb_{name}_count"] = np.log1p(cell_count)
        out[f"eb_{name}_parent_mean"] = parent_mean
        out[f"eb_{name}_parent_count"] = np.log1p(parent_count)

    return out.reset_index(drop=True)


feature_source_cols = [c for c in train.columns if c not in [TARGET]]
base_cols = [c for c in feature_source_cols if c != ID_COL]

cat_cols_raw = []
for c in base_cols:
    if train[c].dtype == "object" or c in [
        "Year",
        "Stint",
        "PitStop",
        "TyreLifeBin",
        "RacePhase",
    ]:
        cat_cols_raw.append(c)

all_for_levels = pd.concat(
    [train[base_cols], test[base_cols]], axis=0, ignore_index=True
)
cat_levels = {
    c: sorted(all_for_levels[c].astype(str).fillna("__NA__").unique().tolist())
    for c in cat_cols_raw
}


def sanitize_columns(cols):
    seen = {}
    safe = []
    for c in cols:
        s = re.sub(r"[^0-9A-Za-z_]+", "_", str(c)).strip("_")
        if not s:
            s = "feature"
        if s[0].isdigit():
            s = "f_" + s
        base = s
        k = seen.get(base, 0)
        if k:
            s = f"{base}_{k}"
        seen[base] = k + 1
        safe.append(s)
    return safe


safe_base_cols = sanitize_columns(base_cols)
raw_to_safe = dict(zip(base_cols, safe_base_cols))
cat_cols = [raw_to_safe[c] for c in cat_cols_raw]


def make_model_frame(df, prior_df):
    X = df[base_cols].copy()
    for c in cat_cols_raw:
        X[c] = pd.Categorical(
            X[c].astype(str).fillna("__NA__"), categories=cat_levels[c]
        )
    X = pd.concat([X.reset_index(drop=True), prior_df.reset_index(drop=True)], axis=1)
    X.columns = sanitize_columns(X.columns)
    return X


try:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(train, y, groups=train["Race"].astype(str)))
except Exception:
    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(train, y, groups=train["Race"].astype(str)))

import lightgbm as lgb

oof = np.zeros(len(train), dtype=float)
fold_aucs = []
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)
    tr_y = y[tr_idx]
    va_y = y[va_idx]
    gp = float(tr_y.mean())

    tr_prior = ordered_eb_features(tr_df, tr_y, PRIOR_SPECS, gp)
    va_prior = aggregate_eb_features(tr_df, tr_y, va_df, PRIOR_SPECS, gp)

    X_tr = make_model_frame(tr_df, tr_prior)
    X_va = make_model_frame(va_df, va_prior)

    pos = max(tr_y.sum(), 1.0)
    neg = max(len(tr_y) - tr_y.sum(), 1.0)

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=1800,
        learning_rate=0.025,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        scale_pos_weight=neg / pos,
        random_state=RANDOM_STATE + fold,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )
    model.fit(
        X_tr,
        tr_y,
        eval_set=[(X_va, va_y)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    pred = model.predict_proba(X_va)[:, 1]
    oof[va_idx] = pred
    auc = roc_auc_score(va_y, pred)
    fold_aucs.append(float(auc))
    best_iters.append(int(model.best_iteration_ or model.n_estimators))
    print(f"fold {fold} roc_auc={auc:.6f} best_iteration={best_iters[-1]}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold grouped CV ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

global_prior = float(y.mean())
full_prior = ordered_eb_features(
    train.reset_index(drop=True), y, PRIOR_SPECS, global_prior
)
test_prior = aggregate_eb_features(
    train.reset_index(drop=True),
    y,
    test.reset_index(drop=True),
    PRIOR_SPECS,
    global_prior,
)

X_full = make_model_frame(train.reset_index(drop=True), full_prior)
X_test = make_model_frame(test.reset_index(drop=True), test_prior)

final_iters = int(np.clip(np.mean(best_iters) * 1.05, 100, 2200))
pos = max(y.sum(), 1.0)
neg = max(len(y) - y.sum(), 1.0)

final_model = lgb.LGBMClassifier(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_estimators=final_iters,
    learning_rate=0.025,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=3.0,
    scale_pos_weight=neg / pos,
    random_state=RANDOM_STATE,
    n_jobs=max(1, os.cpu_count() or 1),
    verbosity=-1,
)
final_model.fit(X_full, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "validation": "5-fold StratifiedGroupKFold grouped by Race",
            "cv_auc": float(cv_auc),
            "fold_auc": fold_aucs,
            "final_n_estimators": final_iters,
            "research_hypotheses_llm_claimed_used": ["001159"],
        }
    )
)
