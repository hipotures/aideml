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

import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
WINDOW_HORIZON = 5
GATE_RECALL = 0.995
TARGET = "PitNextLap"
ID_COL = "id"
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def clean_columns(df):
    mapping = {}
    used = set()
    for c in df.columns:
        nc = re.sub(r"[^0-9A-Za-z_]+", "_", c).strip("_")
        if not nc:
            nc = "col"
        base = nc
        i = 1
        while nc in used:
            nc = f"{base}_{i}"
            i += 1
        used.add(nc)
        mapping[c] = nc
    return df.rename(columns=mapping), mapping


def add_features(df):
    df = df.copy()
    eps = 1e-6
    df["ProgressLeft"] = 1.0 - df["RaceProgress"]
    df["EstimatedRaceLaps"] = df["LapNumber"] / np.maximum(df["RaceProgress"], eps)
    df["LapsRemaining"] = df["EstimatedRaceLaps"] - df["LapNumber"]
    df["TyreLifeRaceFrac"] = df["TyreLife"] / np.maximum(df["EstimatedRaceLaps"], 1.0)
    df["TyreLifeLapFrac"] = df["TyreLife"] / np.maximum(df["LapNumber"], 1.0)
    df["DegPerTyreLap"] = df["Cumulative_Degradation"] / np.maximum(df["TyreLife"], 1.0)
    df["AbsLapTimeDelta"] = np.abs(df["LapTime_Delta"])
    df["OldTyreLateRace"] = df["TyreLife"] * df["RaceProgress"]
    df["DegXProgress"] = df["Cumulative_Degradation"] * df["RaceProgress"]
    comp = df["Compound"].astype(str)
    df["IsWetCompound"] = comp.isin(["WET", "INTERMEDIATE"]).astype(np.int8)
    df["IsSoftCompound"] = (comp == "SOFT").astype(np.int8)
    return df.replace([np.inf, -np.inf], np.nan)


def make_window_target(df, horizon=5):
    group_cols = ["Year", "Race", "Driver"]
    order_cols = group_cols + ["LapNumber", ID_COL]
    sdf = df.sort_values(order_cols)
    grouped = sdf.groupby(group_cols, sort=False)[TARGET]
    win = pd.Series(False, index=sdf.index)
    for k in range(horizon):
        win |= grouped.shift(-k).fillna(0).astype(float).gt(0)
    out = pd.Series(0, index=df.index, dtype=np.int8)
    out.loc[sdf.index] = win.astype(np.int8).values
    return out.values


def scale_pos_weight(y):
    y = np.asarray(y)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    return min(100.0, neg / pos)


def make_model(seed, spw, n_estimators=1200, stage="gate"):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.035 if stage == "gate" else 0.03,
        num_leaves=63 if stage == "gate" else 47,
        min_child_samples=90 if stage == "gate" else 55,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.5,
        scale_pos_weight=spw,
        random_state=seed,
        n_jobs=min(8, os.cpu_count() or 1),
        force_col_wise=True,
        verbosity=-1,
    )


def high_recall_threshold(scores, y, recall=0.995):
    scores = np.asarray(scores)
    y = np.asarray(y)
    pos_scores = scores[y == 1]
    if len(pos_scores) == 0:
        return 0.0
    idx = int(np.floor((1.0 - recall) * max(len(pos_scores) - 1, 0)))
    return float(np.sort(pos_scores)[max(0, min(idx, len(pos_scores) - 1))])


train_raw = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test_raw = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train, col_map = clean_columns(train_raw)
test = test_raw.rename(
    columns={
        c: col_map.get(c, re.sub(r"[^0-9A-Za-z_]+", "_", c).strip("_"))
        for c in test_raw.columns
    }
)
sample = sample.rename(columns={c: col_map.get(c, c) for c in sample.columns})

train = add_features(train)
test = add_features(test)

y = train[TARGET].astype(int).values
y_window = make_window_target(train, WINDOW_HORIZON)

drop_cols = [ID_COL, TARGET]
feature_cols = [c for c in train.columns if c not in drop_cols]
cat_cols = [c for c in feature_cols if train[c].dtype == "object"]

for c in cat_cols:
    cats = pd.Index(pd.concat([train[c], test[c]], axis=0).astype(str).unique())
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

for c in feature_cols:
    if c not in cat_cols:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

X = train[feature_cols]
X_test = test[feature_cols]
groups = (
    train["Year"].astype(str)
    + "_"
    + train["Race"].astype(str)
    + "_"
    + train["Driver"].astype(str)
)

if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y, groups))
else:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(X, y))

gate_oof = np.zeros(len(train), dtype=np.float32)
stage2_oof = np.zeros(len(train), dtype=np.float32)
final_oof = np.zeros(len(train), dtype=np.float32)
gate_best_iters = []
stage2_best_iters = []

print(f"Training hypothesis 000909 with {N_SPLITS}-fold CV")

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    gate = make_model(SEED + fold, scale_pos_weight(y_window[tr_idx]), stage="gate")
    gate.fit(
        X.iloc[tr_idx],
        y_window[tr_idx],
        eval_set=[(X.iloc[va_idx], y_window[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    gate_oof[va_idx] = gate.predict_proba(X.iloc[va_idx])[:, 1]
    gate_best_iters.append(gate.best_iteration_ or gate.n_estimators)

print(f"Gate OOF ROC AUC vs window target: {roc_auc_score(y_window, gate_oof):.6f}")
print(f"Gate OOF ROC AUC vs PitNextLap: {roc_auc_score(y, gate_oof):.6f}")

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    thr = high_recall_threshold(gate_oof[tr_idx], y[tr_idx], GATE_RECALL)
    cand_idx = tr_idx[gate_oof[tr_idx] >= thr]

    if (
        len(cand_idx) < 1000
        or y[cand_idx].sum() < 10
        or len(np.unique(y[cand_idx])) < 2
    ):
        cand_idx = tr_idx

    train_recall = y[cand_idx].sum() / max(1, y[tr_idx].sum())
    cand_frac = len(cand_idx) / len(tr_idx)

    stage2 = make_model(
        SEED + 100 + fold, scale_pos_weight(y[cand_idx]), stage="stage2"
    )
    stage2.fit(
        X.iloc[cand_idx],
        y[cand_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
    )
    stage2_oof[va_idx] = stage2.predict_proba(X.iloc[va_idx])[:, 1]
    final_oof[va_idx] = np.clip(gate_oof[va_idx] * stage2_oof[va_idx], 0.0, 1.0)
    stage2_best_iters.append(stage2.best_iteration_ or stage2.n_estimators)

    fold_auc = roc_auc_score(y[va_idx], final_oof[va_idx])
    print(
        f"Fold {fold}: ROC AUC={fold_auc:.6f}, "
        f"gate_threshold={thr:.6f}, candidate_frac={cand_frac:.3f}, "
        f"candidate_positive_recall={train_recall:.4f}"
    )

cv_auc = roc_auc_score(y, final_oof)
stage2_auc = roc_auc_score(y, stage2_oof)
print(f"Stage2-only OOF ROC AUC: {stage2_auc:.6f}")
print(f"Final gated-product OOF ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": final_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

global_thr = high_recall_threshold(gate_oof, y, GATE_RECALL)
full_cand_idx = np.where(gate_oof >= global_thr)[0]
if (
    len(full_cand_idx) < 1000
    or y[full_cand_idx].sum() < 10
    or len(np.unique(y[full_cand_idx])) < 2
):
    full_cand_idx = np.arange(len(train))

gate_iters = max(100, int(np.mean(gate_best_iters) * 1.08))
stage2_iters = max(100, int(np.mean(stage2_best_iters) * 1.08))

full_gate = make_model(
    SEED + 777, scale_pos_weight(y_window), n_estimators=gate_iters, stage="gate"
)
full_gate.fit(X, y_window, categorical_feature=cat_cols)

full_stage2 = make_model(
    SEED + 888,
    scale_pos_weight(y[full_cand_idx]),
    n_estimators=stage2_iters,
    stage="stage2",
)
full_stage2.fit(X.iloc[full_cand_idx], y[full_cand_idx], categorical_feature=cat_cols)

test_gate = full_gate.predict_proba(X_test)[:, 1]
test_stage2 = full_stage2.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_gate * test_stage2, 0.0, 1.0)

pred_df = pd.DataFrame({ID_COL: test[ID_COL].values, TARGET: test_pred})
if not np.array_equal(sample[ID_COL].values, pred_df[ID_COL].values):
    sub = sample[[ID_COL]].merge(pred_df, on=ID_COL, how="left")
else:
    sub = pred_df

sub[TARGET] = sub[TARGET].fillna(float(np.mean(final_oof))).clip(0.0, 1.0)
sub.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
sub.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000909"],
    "metric": "roc_auc",
    "cv_auc": float(cv_auc),
    "stage2_only_oof_auc": float(stage2_auc),
    "gate_window_oof_auc": float(roc_auc_score(y_window, gate_oof)),
    "gate_target_oof_auc": float(roc_auc_score(y, gate_oof)),
    "window_horizon": WINDOW_HORIZON,
    "gate_recall_target": GATE_RECALL,
    "global_gate_threshold": float(global_thr),
    "full_candidate_fraction": float(len(full_cand_idx) / len(train)),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
