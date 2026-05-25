import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_STRATIFIED_GROUP = True
except Exception:
    HAS_STRATIFIED_GROUP = False

import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

N_JOBS = max(1, os.cpu_count() or 1)

train_raw = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test_raw = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train_raw.pop(TARGET).astype(int).reset_index(drop=True)


def prepare_frame(df):
    df = df.copy()
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["WetFlag"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")
    df["TyreLifeBin"] = np.floor(df["TyreLife"].clip(0, 90)).astype("int16")
    df["RaceProgressBin"] = np.floor(df["RaceProgress"].clip(0, 0.999999) * 20).astype(
        "int8"
    )
    return df.reset_index(drop=True)


train = prepare_frame(train_raw)
test = prepare_frame(test_raw)

CATEGORICAL_COLS = [
    "Compound",
    "Driver",
    "Race",
    "Race_Year",
    "Year",
    "Stint",
    "PitStop",
    "TyreLifeBin",
    "RaceProgressBin",
    "WetFlag",
]

for col in CATEGORICAL_COLS:
    cats = pd.Index(
        pd.concat([train[col], test[col]], ignore_index=True).drop_duplicates()
    )
    train[col] = pd.Categorical(train[col], categories=cats)
    test[col] = pd.Categorical(test[col], categories=cats)

HAZARD_SPECS = [
    (
        "haz_comp_life_stint_prog_wet",
        ["Compound", "TyreLifeBin", "Stint", "RaceProgressBin", "WetFlag"],
        80.0,
    ),
    ("haz_comp_life_stint_wet", ["Compound", "TyreLifeBin", "Stint", "WetFlag"], 100.0),
    ("haz_comp_life_wet", ["Compound", "TyreLifeBin", "WetFlag"], 120.0),
    ("haz_life_prog_wet", ["TyreLifeBin", "RaceProgressBin", "WetFlag"], 120.0),
    (
        "haz_comp_stint_prog_wet",
        ["Compound", "Stint", "RaceProgressBin", "WetFlag"],
        140.0,
    ),
]
HAZARD_COLUMNS = [name + "_logit" for name, _, _ in HAZARD_SPECS]

BASE_FEATURES = [c for c in train.columns if c != ID_COL]
MODEL_CATEGORICAL = [c for c in CATEGORICAL_COLS if c in BASE_FEATURES]


def make_group_cv(n_splits, seed):
    if HAS_STRATIFIED_GROUP:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_splits)


def fit_hazard_maps(df, target):
    target_arr = np.asarray(target, dtype=np.float32)
    prior = float(np.clip(target_arr.mean(), 1e-6, 1 - 1e-6))
    maps = {}

    for name, cols, alpha in HAZARD_SPECS:
        tmp = df[cols].copy()
        tmp["__target__"] = target_arr
        stats = (
            tmp.groupby(cols, observed=True)["__target__"]
            .agg(["sum", "count"])
            .reset_index()
        )
        prob_col = name + "_prob"
        stats[prob_col] = (stats["sum"] + alpha * prior) / (stats["count"] + alpha)
        maps[name] = (cols, stats[cols + [prob_col]], prob_col, prior)

    return maps


def apply_hazard_maps(df, maps):
    out = pd.DataFrame(index=df.index)

    for name, (cols, table, prob_col, prior) in maps.items():
        left = df[cols].reset_index(drop=True)
        merged = left.merge(table, how="left", on=cols, sort=False)
        p = (
            merged[prob_col]
            .fillna(prior)
            .clip(1e-6, 1 - 1e-6)
            .to_numpy(dtype=np.float32)
        )
        out[name + "_logit"] = np.log(p / (1.0 - p)).astype(np.float32)

    return out[HAZARD_COLUMNS]


def build_oof_hazards(df, target, groups, seed):
    target_arr = np.asarray(target, dtype=np.int8)
    n_groups = pd.Series(groups).nunique()
    n_splits = min(N_SPLITS, n_groups)

    if n_splits < 2:
        return apply_hazard_maps(df, fit_hazard_maps(df, target_arr)).astype(np.float32)

    encoded = pd.DataFrame(index=df.index, columns=HAZARD_COLUMNS, dtype=np.float32)
    splitter = make_group_cv(n_splits, seed)

    for tr_idx, va_idx in splitter.split(df, target_arr, groups):
        maps = fit_hazard_maps(df.iloc[tr_idx], target_arr[tr_idx])
        encoded.iloc[va_idx, :] = apply_hazard_maps(df.iloc[va_idx], maps).to_numpy(
            dtype=np.float32
        )

    if encoded.isna().any().any():
        full_maps = fit_hazard_maps(df, target_arr)
        encoded = encoded.fillna(apply_hazard_maps(df, full_maps))

    return encoded.astype(np.float32)


def make_X(df, hazard_df):
    return pd.concat(
        [df[BASE_FEATURES].reset_index(drop=True), hazard_df.reset_index(drop=True)],
        axis=1,
    )


def make_model(seed, scale_pos_weight, n_estimators=2500):
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
    )


groups = train["Race_Year"].astype(str).to_numpy()
outer_splits = min(N_SPLITS, pd.Series(groups).nunique())
outer_cv = make_group_cv(outer_splits, SEED)

oof_pred = np.zeros(len(train), dtype=np.float32)
cv_hazards = pd.DataFrame(index=train.index, columns=HAZARD_COLUMNS, dtype=np.float32)
fold_aucs = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(train, y, groups), start=1):
    tr_df = train.iloc[tr_idx].reset_index(drop=True)
    va_df = train.iloc[va_idx].reset_index(drop=True)
    tr_y = y.iloc[tr_idx].reset_index(drop=True)
    va_y = y.iloc[va_idx].reset_index(drop=True)
    tr_groups = tr_df["Race_Year"].astype(str).to_numpy()

    tr_haz = build_oof_hazards(tr_df, tr_y, tr_groups, SEED + fold * 17)
    outer_maps = fit_hazard_maps(tr_df, tr_y)
    va_haz = apply_hazard_maps(va_df, outer_maps)

    cv_hazards.iloc[va_idx, :] = va_haz.to_numpy(dtype=np.float32)

    X_tr = make_X(tr_df, tr_haz)
    X_va = make_X(va_df, va_haz)

    pos = float(tr_y.sum())
    scale_pos_weight = (len(tr_y) - pos) / max(pos, 1.0)

    model = make_model(SEED + fold, scale_pos_weight)
    model.fit(
        X_tr,
        tr_y,
        eval_set=[(X_va, va_y)],
        eval_metric="auc",
        categorical_feature=MODEL_CATEGORICAL,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None and best_iter > 0:
        best_iterations.append(int(best_iter))
    else:
        best_iter = None

    pred = model.predict_proba(X_va, num_iteration=best_iter)[:, 1]
    oof_pred[va_idx] = pred.astype(np.float32)

    fold_auc = roc_auc_score(va_y, pred) if va_y.nunique() == 2 else np.nan
    fold_aucs.append(fold_auc)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

oof_auc = roc_auc_score(y, oof_pred)
print(f"OOF ROC AUC: {oof_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y.to_numpy(),
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

if cv_hazards.isna().any().any():
    cv_hazards = build_oof_hazards(train, y, groups, SEED + 999)

full_maps = fit_hazard_maps(train, y)
test_haz = apply_hazard_maps(test, full_maps)

X_full = make_X(train, cv_hazards.astype(np.float32))
X_test = make_X(test, test_haz)

if best_iterations:
    final_estimators = int(np.clip(np.mean(best_iterations) * 1.08, 200, 2500))
else:
    final_estimators = 1200

pos = float(y.sum())
final_scale_pos_weight = (len(y) - pos) / max(pos, 1.0)

final_model = make_model(
    SEED + 1000, final_scale_pos_weight, n_estimators=final_estimators
)
final_model.fit(
    X_full,
    y,
    categorical_feature=MODEL_CATEGORICAL,
)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample[[ID_COL, TARGET]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "oof_roc_auc": float(oof_auc),
            "fold_roc_auc": [None if pd.isna(v) else float(v) for v in fold_aucs],
            "research_hypotheses_llm_claimed_used": ["000118"],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        }
    )
)
