import gc
import json
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
EMBARGO_LAPS = 1
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

TARGET = "PitNextLap"
y = train[TARGET].astype(int).to_numpy()


def add_features(df):
    df = df.copy()
    for c in ["Driver", "Race", "Compound"]:
        df[c] = df[c].fillna("__NA__").astype(str)

    df["Race_Year"] = df["Race"] + "_" + df["Year"].astype(str)
    df["Driver_Race"] = df["Driver"] + "_" + df["Race"]
    df["Driver_Compound"] = df["Driver"] + "_" + df["Compound"]
    df["Race_Compound"] = df["Race"] + "_" + df["Compound"]

    tyre = df["TyreLife"].astype(float).clip(lower=1)
    lap = df["LapNumber"].astype(float).clip(lower=1)

    df["DegPerTyreLife"] = df["Cumulative_Degradation"] / tyre
    df["LapTimeLog"] = np.log1p(df["LapTime (s)"].astype(float).clip(lower=0))
    df["AbsLapTimeDelta"] = np.abs(df["LapTime_Delta"].astype(float))
    df["LapDeltaPerTyreLife"] = df["LapTime_Delta"].astype(float) / tyre
    df["TyreLifeFrac"] = tyre / lap
    df["ProgressRemaining"] = 1.0 - df["RaceProgress"].astype(float)
    df["StintTyre"] = df["Stint"].astype(float) * tyre
    df["PositionLoss"] = np.maximum(df["Position_Change"].astype(float), 0.0)
    df["PositionGain"] = np.maximum(-df["Position_Change"].astype(float), 0.0)

    return df.replace([np.inf, -np.inf], 0).fillna(0)


train_f = add_features(train.drop(columns=[TARGET]))
test_f = add_features(test)

CAT_COLS = [
    "Compound",
    "Driver",
    "Race",
    "Race_Year",
    "Driver_Race",
    "Driver_Compound",
    "Race_Compound",
]
CAT_CODE_COLS = [f"{c}_code" for c in CAT_COLS]

NUM_COLS = [
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
    "DegPerTyreLife",
    "LapTimeLog",
    "AbsLapTimeDelta",
    "LapDeltaPerTyreLife",
    "TyreLifeFrac",
    "ProgressRemaining",
    "StintTyre",
    "PositionLoss",
    "PositionGain",
]

for c in CAT_COLS:
    vals = (
        pd.concat([train_f[c], test_f[c]], ignore_index=True)
        .fillna("__NA__")
        .astype(str)
    )
    codes, _ = pd.factorize(vals, sort=True)
    train_f[f"{c}_code"] = codes[: len(train_f)].astype(np.int32)
    test_f[f"{c}_code"] = codes[len(train_f) :].astype(np.int32)


class SmoothTargetEncoder:
    def __init__(self, cols, smoothing=40.0):
        self.cols = cols
        self.smoothing = smoothing
        self.global_mean = None
        self.maps = {}
        self.count_maps = {}

    def fit(self, df, target):
        target = np.asarray(target, dtype=np.float64)
        self.global_mean = float(target.mean())
        for col in self.cols:
            tmp = pd.DataFrame(
                {
                    "key": df[col].fillna("__NA__").astype(str).to_numpy(),
                    "target": target,
                }
            )
            stats = tmp.groupby("key")["target"].agg(["sum", "count"])
            enc = (stats["sum"] + self.global_mean * self.smoothing) / (
                stats["count"] + self.smoothing
            )
            self.maps[col] = enc.astype(np.float32)
            self.count_maps[col] = np.log1p(stats["count"]).astype(np.float32)
        return self

    def transform(self, df):
        out = {}
        for col in self.cols:
            vals = df[col].fillna("__NA__").astype(str)
            out[f"{col}_te"] = (
                vals.map(self.maps[col])
                .fillna(self.global_mean)
                .astype(np.float32)
                .to_numpy()
            )
            out[f"{col}_cnt"] = (
                vals.map(self.count_maps[col]).fillna(0.0).astype(np.float32).to_numpy()
            )
        return pd.DataFrame(out, index=df.index)


def make_lgb_frame(df, encoder):
    x_num = df[NUM_COLS].astype(np.float32).reset_index(drop=True)
    x_cat = df[CAT_CODE_COLS].astype(np.int32).reset_index(drop=True)
    x_te = encoder.transform(df).astype(np.float32).reset_index(drop=True)
    return pd.concat([x_num, x_cat, x_te], axis=1)


def embargo_training_indices(df, train_idx, valid_idx, laps=1):
    train_idx = np.asarray(train_idx, dtype=np.int64)
    valid_idx = np.asarray(valid_idx, dtype=np.int64)
    val_keys = (
        df.iloc[valid_idx][["Driver", "Race", "LapNumber"]].drop_duplicates().copy()
    )

    blocks = []
    for delta in range(-laps, laps + 1):
        tmp = val_keys.copy()
        tmp["LapNumber"] = tmp["LapNumber"].astype(int) + delta
        blocks.append(tmp)
    blocked = pd.concat(blocks, ignore_index=True).drop_duplicates()

    train_keys = df.iloc[train_idx][["Driver", "Race", "LapNumber"]].copy()
    train_keys["LapNumber"] = train_keys["LapNumber"].astype(int)
    keep = ~pd.MultiIndex.from_frame(train_keys).isin(pd.MultiIndex.from_frame(blocked))
    return train_idx[np.asarray(keep, dtype=bool)]


def make_lgb_model(target):
    target = np.asarray(target)
    pos = max(float(target.sum()), 1.0)
    neg = max(float(len(target) - target.sum()), 1.0)
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=650,
        learning_rate=0.035,
        num_leaves=64,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=5.0,
        scale_pos_weight=neg / pos,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        deterministic=True,
        force_col_wise=True,
    )


def clipped_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def auc_or_nan(yt, pred):
    return roc_auc_score(yt, pred) if len(np.unique(yt)) == 2 else np.nan


groups = train_f["Race_Year"].to_numpy()
n_splits = min(N_SPLITS, pd.Series(groups).nunique())
folds = list(GroupKFold(n_splits=n_splits).split(train_f, y, groups))

oof_base = np.full(len(train_f), np.nan, dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    pure_idx = embargo_training_indices(train_f, tr_idx, va_idx, EMBARGO_LAPS)

    enc = SmoothTargetEncoder(CAT_COLS).fit(train_f.iloc[pure_idx], y[pure_idx])
    x_tr = make_lgb_frame(train_f.iloc[pure_idx], enc)
    x_va = make_lgb_frame(train_f.iloc[va_idx], enc)

    model = make_lgb_model(y[pure_idx])
    model.fit(x_tr, y[pure_idx], categorical_feature=CAT_CODE_COLS)
    oof_base[va_idx] = model.predict_proba(x_va)[:, 1].astype(np.float32)

    print(
        f"Fold {fold}: base AUC={auc_or_nan(y[va_idx], oof_base[va_idx]):.6f}, "
        f"train_rows_after_embargo={len(pure_idx)}"
    )

    del enc, x_tr, x_va, model
    gc.collect()

if np.isnan(oof_base).any():
    raise RuntimeError("Some base OOF predictions were not filled.")

META_COLS = [
    "TyreLife",
    "RaceProgress",
    "Stint",
    "PitStop",
    "LapNumber",
    "Position",
    "Position_Change",
    "Cumulative_Degradation",
    "DegPerTyreLife",
    "LapTime_Delta",
    "AbsLapTimeDelta",
    "ProgressRemaining",
]


def make_meta_frame(df, base_pred):
    x = df[META_COLS].astype(np.float32).reset_index(drop=True).copy()
    p = np.clip(np.asarray(base_pred, dtype=np.float64), 1e-6, 1 - 1e-6)
    x["base_pred"] = p.astype(np.float32)
    x["base_logit"] = clipped_logit(p).astype(np.float32)
    return x


def make_meta_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.7,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=SEED,
        ),
    )


meta_x = make_meta_frame(train_f, oof_base)
oof_stack = np.full(len(train_f), np.nan, dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    pure_idx = embargo_training_indices(train_f, tr_idx, va_idx, EMBARGO_LAPS)
    meta = make_meta_model()
    meta.fit(meta_x.iloc[pure_idx], y[pure_idx])
    oof_stack[va_idx] = meta.predict_proba(meta_x.iloc[va_idx])[:, 1].astype(np.float32)
    print(f"Fold {fold}: stacker AUC={auc_or_nan(y[va_idx], oof_stack[va_idx]):.6f}")

if np.isnan(oof_stack).any():
    raise RuntimeError("Some stacker OOF predictions were not filled.")


def make_calib_frame(pred):
    return pd.DataFrame({"stack_logit": clipped_logit(pred).astype(np.float32)})


oof_cal = np.full(len(train_f), np.nan, dtype=np.float32)
cal_x = make_calib_frame(oof_stack)

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    pure_idx = embargo_training_indices(train_f, tr_idx, va_idx, EMBARGO_LAPS)
    cal = LogisticRegression(C=1000.0, max_iter=1000, solver="lbfgs")
    cal.fit(cal_x.iloc[pure_idx], y[pure_idx])
    oof_cal[va_idx] = cal.predict_proba(cal_x.iloc[va_idx])[:, 1].astype(np.float32)

if np.isnan(oof_cal).any():
    raise RuntimeError("Some calibrated OOF predictions were not filled.")

base_auc = roc_auc_score(y, oof_base)
stack_auc = roc_auc_score(y, oof_stack)
final_auc = roc_auc_score(y, oof_cal)

full_encoder = SmoothTargetEncoder(CAT_COLS).fit(train_f, y)
x_full = make_lgb_frame(train_f, full_encoder)
x_test = make_lgb_frame(test_f, full_encoder)

full_model = make_lgb_model(y)
full_model.fit(x_full, y, categorical_feature=CAT_CODE_COLS)
base_test = full_model.predict_proba(x_test)[:, 1].astype(np.float32)

final_meta = make_meta_model()
final_meta.fit(meta_x, y)
test_meta_x = make_meta_frame(test_f, base_test)
stack_test = final_meta.predict_proba(test_meta_x)[:, 1].astype(np.float32)

final_cal = LogisticRegression(C=1000.0, max_iter=1000, solver="lbfgs")
final_cal.fit(make_calib_frame(oof_stack), y)
test_pred = final_cal.predict_proba(make_calib_frame(stack_test))[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample[["id"]].copy()
submission["PitNextLap"] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        "row": np.arange(len(train_f)),
        "target": y.astype(int),
        "prediction": oof_cal,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "race_year_blocked_lap_embargoed_5fold_oof_roc_auc",
    "metric_value": float(final_auc),
    "base_oof_roc_auc": float(base_auc),
    "stacker_oof_roc_auc": float(stack_auc),
    "n_splits": int(n_splits),
    "embargo_laps": int(EMBARGO_LAPS),
    "research_hypotheses_llm_claimed_used": ["000405"],
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"Race-year blocked + lap-embargoed OOF ROC AUC: {final_auc:.6f}")
print(json.dumps(result, sort_keys=True))
