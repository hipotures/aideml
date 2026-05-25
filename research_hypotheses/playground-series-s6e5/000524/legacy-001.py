import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
cat_cols = ["Driver", "Race", "Compound"]
base_num_cols = [c for c in train.columns if c not in [id_col, target_col] + cat_cols]


def add_group_keys(df):
    out = df.copy()
    out["RaceYear"] = out["Year"].astype(str) + "_" + out["Race"].astype(str)
    return out


train = add_group_keys(train)
test = add_group_keys(test)


def make_temporal_group_folds(df, n_splits=5, purge=1):
    groups = (
        df.groupby("RaceYear", sort=False)
        .agg(year=("Year", "min"), min_id=(id_col, "min"))
        .reset_index()
        .sort_values(["year", "min_id"])
        .reset_index(drop=True)
    )
    group_names = groups["RaceYear"].to_numpy()
    fold_blocks = np.array_split(np.arange(len(group_names)), n_splits)
    folds = []
    all_idx = np.arange(len(df))
    group_to_pos = {g: i for i, g in enumerate(group_names)}
    positions = df["RaceYear"].map(group_to_pos).to_numpy()

    for block in fold_blocks:
        if len(block) == 0:
            continue
        val_pos = set(block.tolist())
        lo, hi = int(block.min()), int(block.max())
        purged_pos = set(
            range(max(0, lo - purge), min(len(group_names), hi + purge + 1))
        )
        train_mask = np.array([(p not in purged_pos) for p in positions])
        val_mask = np.array([(p in val_pos) for p in positions])
        tr_idx, va_idx = all_idx[train_mask], all_idx[val_mask]
        if len(tr_idx) and len(va_idx) and df.iloc[va_idx][target_col].nunique() == 2:
            folds.append((tr_idx, va_idx))
    return folds


def fit_feature_maps(df, y):
    maps = {}
    global_rate = float(np.mean(y))
    for col in ["Driver", "Race", "Compound", "RaceYear"]:
        tmp = pd.DataFrame({col: df[col].astype(str), "y": y})
        stats = tmp.groupby(col)["y"].agg(["mean", "count"])
        maps[col] = (stats["mean"].to_dict(), stats["count"].to_dict(), global_rate)
    return maps


def transform_features(df, maps=None, fit_y=None):
    x = df[base_num_cols].copy()

    x["lap_frac"] = df["LapNumber"] / (df["LapNumber"].max() + 1.0)
    x["tyre_life_progress"] = df["TyreLife"] / (df["LapNumber"].clip(lower=1))
    x["degradation_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(
        lower=1
    )
    x["position_x_progress"] = df["Position"] * df["RaceProgress"]
    x["stint_x_tyre"] = df["Stint"] * df["TyreLife"]
    x["is_wet_compound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)

    if maps is None:
        maps = fit_feature_maps(df, fit_y)

    for col in ["Driver", "Race", "Compound", "RaceYear"]:
        mean_map, count_map, global_rate = maps[col]
        vals = df[col].astype(str)
        x[f"{col}_target_rate"] = vals.map(mean_map).fillna(global_rate).astype(float)
        x[f"{col}_count"] = np.log1p(vals.map(count_map).fillna(0)).astype(float)

    for col in cat_cols:
        codes, _ = pd.factorize(df[col].astype(str), sort=True)
        x[f"{col}_code"] = codes.astype(float)

    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x, maps


folds = make_temporal_group_folds(train, n_splits=5, purge=1)
if len(folds) < 3:
    raise RuntimeError(f"Too few usable folds were created: {len(folds)}")

y = train[target_col].astype(int).to_numpy()
oof_lgb_like = np.full(len(train), np.nan)
oof_lr = np.full(len(train), np.nan)
test_lgb_like = np.zeros(len(test))
test_lr = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    x_tr, maps = transform_features(tr_df, fit_y=y_tr)
    x_va, _ = transform_features(va_df, maps=maps)
    x_te, _ = transform_features(test, maps=maps)

    gbm = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.055,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        min_samples_leaf=35,
        random_state=1000 + fold,
        early_stopping=True,
        validation_fraction=0.12,
        class_weight="balanced",
    )
    gbm.fit(x_tr, y_tr)
    oof_lgb_like[va_idx] = gbm.predict_proba(x_va)[:, 1]
    test_lgb_like += gbm.predict_proba(x_te)[:, 1] / len(folds)

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.75,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=2000 + fold,
        ),
    )
    lr.fit(x_tr, y_tr)
    oof_lr[va_idx] = lr.predict_proba(x_va)[:, 1]
    test_lr += lr.predict_proba(x_te)[:, 1] / len(folds)

    fold_auc = roc_auc_score(y_va, 0.5 * oof_lgb_like[va_idx] + 0.5 * oof_lr[va_idx])
    print(f"fold={fold} rows={len(va_idx)} blended_base_auc={fold_auc:.6f}")

valid_mask = np.isfinite(oof_lgb_like) & np.isfinite(oof_lr)
stack_x = np.column_stack([oof_lgb_like[valid_mask], oof_lr[valid_mask]])
stack_y = y[valid_mask]

stacker = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
stacker.fit(stack_x, stack_y)
oof_stack_raw = stacker.predict_proba(stack_x)[:, 1]

calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
calibrator.fit(oof_stack_raw, stack_y)
oof_stack_cal = calibrator.transform(oof_stack_raw)

auc_raw = roc_auc_score(stack_y, oof_stack_raw)
auc_cal = roc_auc_score(stack_y, oof_stack_cal)
print(f"OOF stacked ROC AUC raw={auc_raw:.6f}")
print(f"OOF stacked ROC AUC calibrated={auc_cal:.6f}")

test_stack_x = np.column_stack([test_lgb_like, test_lr])
test_raw = stacker.predict_proba(test_stack_x)[:, 1]
test_pred = calibrator.transform(test_raw)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample[[id_col]].copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

oof_out = pd.DataFrame(
    {
        "row": np.where(valid_mask)[0],
        "target": stack_y,
        "prediction": oof_stack_cal,
    }
)
oof_out.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_out = sample[[id_col]].copy()
test_out[target_col] = test_pred
test_out.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "validation_score": float(auc_cal),
    "raw_oof_score": float(auc_raw),
    "n_folds": len(folds),
    "research_hypotheses_llm_claimed_used": ["000524"],
    "files_written": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
    ],
}
print(json.dumps(result, indent=2))
