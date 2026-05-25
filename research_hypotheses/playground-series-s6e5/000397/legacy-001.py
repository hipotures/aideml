import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

SEED = 42
TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS_RAW = ["Race", "Driver", "Compound"]
GROUP_COLS = ["Year", "Race", "Driver"]
BLEND_W = 0.18

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def rank01(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float32)
    r = pd.Series(x).rank(method="average").to_numpy()
    return ((r - 1.0) / (len(x) - 1.0)).astype(np.float32)


def safe_div(a, b):
    return a / np.where(np.abs(b) < 1e-9, np.nan, b)


def add_features(df):
    out = df.copy()
    out["race_remaining"] = 1.0 - out["RaceProgress"]
    out["position_pct"] = (out["Position"] - 1.0) / 19.0
    out["tyrelife_x_progress"] = out["TyreLife"] * out["RaceProgress"]
    out["stint_x_tyrelife"] = out["Stint"] * out["TyreLife"]
    out["deg_per_tyrelife"] = safe_div(
        out["Cumulative_Degradation"], np.maximum(out["TyreLife"], 1.0)
    )
    out["delta_per_tyrelife"] = safe_div(
        out["LapTime_Delta"], np.maximum(out["TyreLife"], 1.0)
    )
    out["lap_time_per_progress"] = safe_div(
        out["LapTime (s)"], np.maximum(out["RaceProgress"], 0.01)
    )
    out["is_wet_or_inter"] = (
        out["Compound"].isin(["WET", "INTERMEDIATE"]).astype(np.int8)
    )
    out["is_slick"] = out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(np.int8)

    s = out.sort_values(GROUP_COLS + ["LapNumber", ID_COL])
    g = s.groupby(GROUP_COLS, sort=False)

    out.loc[s.index, "prev_pitstop"] = g["PitStop"].shift(1)
    out.loc[s.index, "prev_laptime"] = g["LapTime (s)"].shift(1)
    out.loc[s.index, "prev_laptime_delta"] = g["LapTime_Delta"].shift(1)
    out.loc[s.index, "prev_position"] = g["Position"].shift(1)
    out.loc[s.index, "prev_position_change"] = g["Position_Change"].shift(1)
    out.loc[s.index, "prev_degradation"] = g["Cumulative_Degradation"].shift(1)
    out.loc[s.index, "prev_tyrelife"] = g["TyreLife"].shift(1)

    out.loc[s.index, "delta_roll3_prev"] = g["LapTime_Delta"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    out.loc[s.index, "laptime_roll3_prev"] = g["LapTime (s)"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    out.loc[s.index, "position_roll3_prev"] = g["Position"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    out["lap_time_accel"] = out["LapTime_Delta"] - out["prev_laptime_delta"]
    out["lap_time_vs_prev"] = out["LapTime (s)"] - out["prev_laptime"]
    out["degradation_jump"] = out["Cumulative_Degradation"] - out["prev_degradation"]
    out["position_vs_prev"] = out["Position"] - out["prev_position"]
    out["tyrelife_jump"] = out["TyreLife"] - out["prev_tyrelife"]

    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)
    for c in num_cols:
        if c != ID_COL:
            out[c] = out[c].astype(np.float32)
    return out


def sanitize_columns(train_df, test_df):
    rename = {}
    used = set()
    for c in train_df.columns:
        base = re.sub(r"[^A-Za-z0-9_]+", "_", c).strip("_") or "feature"
        name = base
        k = 1
        while name in used:
            k += 1
            name = f"{base}_{k}"
        used.add(name)
        rename[c] = name
    return train_df.rename(columns=rename), test_df.rename(columns=rename), rename


def align_categories(train_df, test_df, cat_cols):
    for c in cat_cols:
        vals = (
            pd.concat([train_df[c], test_df[c]], axis=0)
            .astype("string")
            .fillna("__missing__")
        )
        cats = pd.Index(vals.unique())
        train_df[c] = pd.Categorical(
            train_df[c].astype("string").fillna("__missing__"), categories=cats
        )
        test_df[c] = pd.Categorical(
            test_df[c].astype("string").fillna("__missing__"), categories=cats
        )
    return train_df, test_df


def adjacent_stop_mask(raw_df, y_series):
    tmp = raw_df[[ID_COL, "Year", "Race", "Driver", "LapNumber", "PitStop"]].copy()
    tmp["_y"] = pd.Series(y_series, index=raw_df.index).astype(np.int8)
    tmp = tmp.sort_values(GROUP_COLS + ["LapNumber", ID_COL])
    grp = tmp.groupby(GROUP_COLS, sort=False)

    adj = np.zeros(len(tmp), dtype=bool)
    for k in [-3, -2, -1, 1, 2, 3]:
        adj |= grp["_y"].shift(k).fillna(0).to_numpy(dtype=np.int8) == 1
    for k in [-2, -1, 0, 1, 2]:
        adj |= grp["PitStop"].shift(k).fillna(0).to_numpy(dtype=np.int8) == 1

    out = pd.Series(False, index=raw_df.index)
    out.loc[tmp.index] = adj
    return out


def pct_score(s, lo_q=0.55, hi_q=0.95):
    lo, hi = s.quantile(lo_q), s.quantile(hi_q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1).fillna(0)


def build_hard_negatives(raw_df, feat_df, y_series):
    y_series = pd.Series(y_series, index=raw_df.index).astype(np.int8)
    neg = y_series == 0
    adj = adjacent_stop_mask(raw_df, y_series)

    late_stint = feat_df["TyreLife"] >= feat_df["TyreLife"].quantile(0.70)
    high_deg = feat_df["Cumulative_Degradation"] >= feat_df[
        "Cumulative_Degradation"
    ].quantile(0.72)
    high_deg_rate = feat_df["deg_per_tyrelife"] >= feat_df["deg_per_tyrelife"].quantile(
        0.72
    )
    big_drift = feat_df["LapTime_Delta"] >= feat_df["LapTime_Delta"].quantile(0.72)
    accel = feat_df["lap_time_accel"] >= feat_df["lap_time_accel"].quantile(0.72)

    signal_count = (
        late_stint.astype(int)
        + high_deg.astype(int)
        + high_deg_rate.astype(int)
        + big_drift.astype(int)
        + accel.astype(int)
        + adj.astype(int) * 2
    )

    soft_score = (
        pct_score(feat_df["TyreLife"])
        + pct_score(feat_df["Cumulative_Degradation"])
        + pct_score(feat_df["deg_per_tyrelife"])
        + pct_score(feat_df["LapTime_Delta"])
        + pct_score(feat_df["lap_time_accel"])
        + adj.astype(float) * 1.5
    )

    hard = neg & ((signal_count >= 2) | adj)
    min_neg = int(min(neg.sum(), max(5000, 3 * int((y_series == 1).sum()))))
    if hard.sum() < min_neg and min_neg > 0:
        hard.loc[soft_score[neg].nlargest(min_neg).index] = True

    weight_score = (1.0 + signal_count.astype(float) + soft_score).clip(1, 10)
    return hard, weight_score


train_raw = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test_raw = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train_raw[TARGET].astype(np.int8)
train_base = train_raw.drop(columns=[TARGET])

train_feat = add_features(train_base)
test_feat = add_features(test_raw)
train_feat, test_feat, rename_map = sanitize_columns(train_feat, test_feat)
cat_cols = [rename_map[c] for c in CAT_COLS_RAW]
train_feat, test_feat = align_categories(train_feat, test_feat, cat_cols)

feature_cols = [c for c in train_feat.columns if c != rename_map[ID_COL]]
X = train_feat[feature_cols]
X_test = test_feat[feature_cols]

groups = (
    train_base["Year"].astype(str)
    + "_"
    + train_base["Race"].astype(str)
    + "_"
    + train_base["Driver"].astype(str)
)

if HAS_SGK:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y, groups))
    cv_name = "StratifiedGroupKFold"
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(cv.split(X, y))
    cv_name = "StratifiedKFold"

oof = np.zeros(len(X), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float64)
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = neg / max(pos, 1)

    main = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=2200,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=5.0,
        scale_pos_weight=spw,
        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    main.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(120, verbose=False), log_evaluation(0)],
    )

    hard_mask, hard_score = build_hard_negatives(
        train_base.iloc[tr_idx], train_feat.iloc[tr_idx], y_tr
    )
    spec_rows = (y_tr == 1) | hard_mask.loc[X_tr.index]
    X_sp, y_sp = X_tr.loc[spec_rows], y_tr.loc[spec_rows]

    sw = np.ones(len(y_sp), dtype=np.float32)
    sp_pos = int(y_sp.sum())
    sp_neg = int(len(y_sp) - sp_pos)
    sw[y_sp.to_numpy() == 1] = max(1.0, sp_neg / max(sp_pos, 1))
    neg_w = hard_score.loc[X_sp.index].to_numpy(dtype=np.float32)
    sw[y_sp.to_numpy() == 0] = neg_w[y_sp.to_numpy() == 0]
    sw /= max(sw.mean(), 1e-6)

    specialist = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1600,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=35,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_alpha=0.05,
        reg_lambda=3.0,
        random_state=SEED + 100 + fold,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True,
    )
    specialist.fit(
        X_sp,
        y_sp,
        sample_weight=sw,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
    )

    main_va = main.predict_proba(X_va)[:, 1]
    spec_va = specialist.predict_proba(X_va)[:, 1]
    blend_va = (1.0 - BLEND_W) * rank01(main_va) + BLEND_W * rank01(spec_va)
    oof[va_idx] = blend_va

    main_te = main.predict_proba(X_test)[:, 1]
    spec_te = specialist.predict_proba(X_test)[:, 1]
    blend_te = (1.0 - BLEND_W) * rank01(main_te) + BLEND_W * rank01(spec_te)
    test_pred += blend_te / len(splits)

    fold_auc = roc_auc_score(y_va, blend_va)
    fold_aucs.append(float(fold_auc))
    print(
        f"fold {fold}: auc={fold_auc:.6f}, "
        f"hard_negatives={int(hard_mask.sum())}, specialist_rows={len(X_sp)}"
    )

cv_auc = roc_auc_score(y, oof)
test_pred = np.clip(test_pred, 0.0, 1.0)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train_raw), dtype=np.int64),
        "target": y.to_numpy(dtype=np.int8),
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

target_col = [c for c in sample.columns if c != ID_COL][0]
submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_strategy": cv_name,
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_aucs,
    "blend_weight_specialist": BLEND_W,
    "research_hypotheses_llm_claimed_used": ["000397"],
    "saved_files": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
    ],
}
print(json.dumps(result, indent=2))
