import os
import re
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

RANDOM_STATE = 20260523
N_SPLITS = 5
EMBARGO_BLOCKS = 1
WET_COMPOUNDS = {"WET", "INTERMEDIATE"}


def clean_col(name):
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z_]+", "_", name)).strip("_")


def add_features(all_df):
    all_df = all_df.copy()
    all_df["RaceBlock"] = all_df["Year"].astype(str) + "__" + all_df["Race"].astype(str)
    all_df["IsWetCompound"] = (
        all_df["Compound"].astype(str).isin(WET_COMPOUNDS).astype("int8")
    )
    all_df["RaceRemaining"] = 1.0 - all_df["RaceProgress"]
    all_df["TyreLife_x_Progress"] = all_df["TyreLife"] * all_df["RaceProgress"]
    all_df["DegradationPerTyreLap"] = all_df["Cumulative_Degradation"] / np.maximum(
        all_df["TyreLife"], 1
    )
    all_df["AbsLapTimeDelta"] = all_df["LapTime_Delta"].abs()
    all_df["PitStop_x_TyreLife"] = all_df["PitStop"] * all_df["TyreLife"]
    all_df["LateRaceOldTyre"] = all_df["RaceProgress"] * np.log1p(all_df["TyreLife"])

    all_df["_orig_order"] = np.arange(len(all_df))
    sort_cols = ["Year", "Race", "Driver", "LapNumber", "id"]
    tmp = all_df.sort_values(sort_cols).copy()
    grp = tmp.groupby(["Year", "Race", "Driver"], sort=False)

    lag_cols = [
        "LapTime_s",
        "LapTime_Delta",
        "Position",
        "Position_Change",
        "TyreLife",
        "Cumulative_Degradation",
        "PitStop",
    ]
    for col in lag_cols:
        tmp[f"Prev_{col}"] = grp[col].shift(1)

    tmp["LapTimeMinusPrev"] = tmp["LapTime_s"] - tmp["Prev_LapTime_s"]
    tmp["PositionMinusPrev"] = tmp["Position"] - tmp["Prev_Position"]
    tmp["PrevPitStopMissing0"] = tmp["Prev_PitStop"].fillna(0)
    tmp["DriverRacePitsSoFar"] = grp["PitStop"].cumsum() - tmp["PitStop"]

    tmp = tmp.sort_values("_orig_order").drop(columns=["_orig_order"])
    num_cols = tmp.select_dtypes(include=[np.number]).columns
    tmp[num_cols] = tmp[num_cols].replace([np.inf, -np.inf], np.nan)
    return tmp


def make_chrono_embargo_folds(train_df):
    block_meta = (
        train_df.groupby("RaceBlock")
        .agg(year=("Year", "min"), first_id=("id", "min"))
        .sort_values(["year", "first_id"])
        .reset_index()
    )
    blocks = block_meta["RaceBlock"].tolist()
    row_blocks = train_df["RaceBlock"].to_numpy()
    chunks = np.array_split(np.arange(len(blocks)), N_SPLITS)

    folds = []
    for fold, valid_pos in enumerate(chunks):
        valid_blocks = {blocks[i] for i in valid_pos}
        left = max(0, int(valid_pos[0]) - EMBARGO_BLOCKS)
        right = min(len(blocks) - 1, int(valid_pos[-1]) + EMBARGO_BLOCKS)
        excluded_blocks = {blocks[i] for i in range(left, right + 1)}

        valid_idx = np.where(np.isin(row_blocks, list(valid_blocks)))[0]
        train_idx = np.where(~np.isin(row_blocks, list(excluded_blocks)))[0]

        assert (
            len(set(row_blocks[train_idx]).intersection(set(row_blocks[valid_idx])))
            == 0
        )
        folds.append((train_idx, valid_idx))
    return folds


def make_model(seed, n_estimators, scale_pos_weight):
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=int(n_estimators),
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=min(8, os.cpu_count() or 1),
        verbose=-1,
    )


def fit_lgbm(
    X_tr, y_tr, X_va=None, y_va=None, cat_cols=None, seed=RANDOM_STATE, n_estimators=900
):
    pos = float(np.sum(y_tr))
    neg = float(len(y_tr) - pos)
    spw = neg / max(pos, 1.0)
    model = make_model(seed, n_estimators, spw)

    if X_va is not None and len(np.unique(y_va)) == 2:
        try:
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="auc",
                categorical_feature=cat_cols,
                callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
            )
        except TypeError:
            model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    else:
        model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    return model


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_sigmoid(raw, y):
    y = np.asarray(y).astype(int)
    prior = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
    if len(y) < 50 or len(np.unique(y)) < 2:
        return lambda p: np.full(len(p), prior, dtype=float)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    lr.fit(logit(raw), y)
    return lambda p: lr.predict_proba(logit(p))[:, 1]


def fit_isotonic(raw, y):
    y = np.asarray(y).astype(int)
    if len(y) < 1000 or len(np.unique(y)) < 2:
        return fit_sigmoid(raw, y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(np.asarray(raw, dtype=float), y)
    return lambda p: iso.predict(np.asarray(p, dtype=float))


def calibrated_stack(fit_raw, fit_y, pred_raw, fit_compound, pred_compound):
    fit_compound = np.asarray(fit_compound).astype(str)
    pred_compound = np.asarray(pred_compound).astype(str)

    global_cal = fit_sigmoid(fit_raw, fit_y)
    global_pred = global_cal(pred_raw)

    fit_dry = ~np.isin(fit_compound, list(WET_COMPOUNDS))
    pred_dry = ~np.isin(pred_compound, list(WET_COMPOUNDS))

    segment_pred = global_pred.copy()
    dry_cal = fit_isotonic(np.asarray(fit_raw)[fit_dry], np.asarray(fit_y)[fit_dry])
    wet_cal = fit_sigmoid(np.asarray(fit_raw)[~fit_dry], np.asarray(fit_y)[~fit_dry])

    if np.any(pred_dry):
        segment_pred[pred_dry] = dry_cal(np.asarray(pred_raw)[pred_dry])
    if np.any(~pred_dry):
        segment_pred[~pred_dry] = wet_cal(np.asarray(pred_raw)[~pred_dry])

    return np.clip(0.5 * global_pred + 0.5 * segment_pred, 1e-6, 1 - 1e-6)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train = train.rename(columns={c: clean_col(c) for c in train.columns})
test = test.rename(columns={c: clean_col(c) for c in test.columns})

target_col = "PitNextLap"
id_col = "id"
target_out_col = [c for c in sample.columns if c != "id"][0]

y = train[target_col].astype(int).to_numpy()
n_train = len(train)

all_df = pd.concat([train.drop(columns=[target_col]), test], axis=0, ignore_index=True)
all_df = add_features(all_df)

cat_cols_all = ["Compound", "Driver", "Race"]
for col in cat_cols_all:
    all_df[col] = all_df[col].astype("category")

train_blocks_df = all_df.iloc[:n_train][["id", "Year", "RaceBlock"]].copy()
feature_cols = [c for c in all_df.columns if c not in [id_col, "RaceBlock"]]
cat_cols = [c for c in cat_cols_all if c in feature_cols]

X = all_df.iloc[:n_train][feature_cols].copy()
X_test = all_df.iloc[n_train:][feature_cols].copy()
train_compound = all_df.iloc[:n_train]["Compound"].astype(str).to_numpy()
test_compound = all_df.iloc[n_train:]["Compound"].astype(str).to_numpy()

folds = make_chrono_embargo_folds(train_blocks_df)
raw_oof = np.zeros(n_train, dtype=float)
best_iters = []

for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
    model = fit_lgbm(
        X.iloc[tr_idx],
        y[tr_idx],
        X.iloc[va_idx],
        y[va_idx],
        cat_cols=cat_cols,
        seed=RANDOM_STATE + fold,
        n_estimators=900,
    )
    raw_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None and best_iter > 0:
        best_iters.append(best_iter)
    print(
        f"Fold {fold}: raw AUC={roc_auc_score(y[va_idx], raw_oof[va_idx]):.6f}, rows={len(va_idx)}"
    )

cal_oof = np.zeros(n_train, dtype=float)
all_idx = np.arange(n_train)
for _, va_idx in folds:
    fit_idx = np.setdiff1d(all_idx, va_idx, assume_unique=False)
    cal_oof[va_idx] = calibrated_stack(
        raw_oof[fit_idx],
        y[fit_idx],
        raw_oof[va_idx],
        train_compound[fit_idx],
        train_compound[va_idx],
    )

raw_auc = roc_auc_score(y, raw_oof)
cal_auc = roc_auc_score(y, cal_oof)

final_estimators = int(
    np.clip(np.median(best_iters) * 1.05 if best_iters else 650, 250, 1000)
)
final_model = fit_lgbm(
    X,
    y,
    cat_cols=cat_cols,
    seed=RANDOM_STATE + 99,
    n_estimators=final_estimators,
)
raw_test = final_model.predict_proba(X_test)[:, 1]
test_pred = calibrated_stack(raw_oof, y, raw_test, train_compound, test_compound)

oof_df = pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": cal_oof,
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pred_by_id = pd.Series(test_pred, index=test[id_col].to_numpy())
submission = sample.copy()
submission[target_out_col] = sample["id"].map(pred_by_id).astype(float).to_numpy()
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

print(f"Raw chronological embargo CV ROC AUC: {raw_auc:.6f}")
print(f"Calibrated chronological embargo CV ROC AUC: {cal_auc:.6f}")
print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_auc": float(cal_auc),
            "raw_cv_auc": float(raw_auc),
            "n_splits": N_SPLITS,
            "embargo_race_blocks": EMBARGO_BLOCKS,
            "research_hypotheses_llm_claimed_used": ["000599"],
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
            "oof_path": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
            "test_predictions_path": os.path.join(
                WORKING_DIR, "test_predictions.csv.gz"
            ),
        },
        indent=2,
    )
)
