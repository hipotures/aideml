import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

import lightgbm as lgb

warnings.filterwarnings("ignore")

RANDOM_STATE = 968
N_FOLDS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Race", "Driver", "Compound"]


def make_features(df):
    df = df.copy()
    race_progress = df["RaceProgress"].clip(lower=0.01)
    tyre_life = df["TyreLife"].replace(0, np.nan)
    lap_number = df["LapNumber"].replace(0, np.nan)

    df["TotalLaps_Est"] = df["LapNumber"] / race_progress
    df["LapsRemaining_Est"] = df["TotalLaps_Est"] - df["LapNumber"]
    df["TyreLife_RaceFrac"] = df["TyreLife"] / df["TotalLaps_Est"].replace(0, np.nan)
    df["TyreLife_RemainRatio"] = df["TyreLife"] / (
        df["LapsRemaining_Est"].clip(lower=0) + 1.0
    )
    df["DegradationPerLap"] = df["Cumulative_Degradation"] / tyre_life
    df["DegradationPerRaceProgress"] = df["Cumulative_Degradation"] / race_progress
    df["LapTimePerProgress"] = df["LapTime (s)"] / race_progress
    df["LapDeltaAbs"] = df["LapTime_Delta"].abs()
    df["PositionChangeAbs"] = df["Position_Change"].abs()
    df["StintTyreLife"] = df["Stint"] * df["TyreLife"]
    df["LapInStintRatio"] = df["TyreLife"] / lap_number
    df["IsWetCompound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    df["IsLateRace"] = (df["RaceProgress"] >= 0.70).astype(np.int8)
    df["Is2025"] = (df["Year"] == 2025).astype(np.int8)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def set_categories(train_x, test_x):
    for col in CAT_COLS:
        cats = pd.Index(
            pd.concat([train_x[col], test_x[col]], axis=0).astype(str).unique()
        )
        dtype = pd.CategoricalDtype(categories=cats)
        train_x[col] = train_x[col].astype(str).astype(dtype)
        test_x[col] = test_x[col].astype(str).astype(dtype)
    return train_x, test_x


def make_model(seed, n_estimators=260, num_leaves=63):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=num_leaves,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=min(8, os.cpu_count() or 1),
        force_col_wise=True,
        verbosity=-1,
    )


def can_fit(y_subset, min_rows=800, min_pos=15):
    if len(y_subset) < min_rows:
        return False
    positives = int(np.sum(y_subset == 1))
    negatives = int(np.sum(y_subset == 0))
    return positives >= min_pos and negatives >= min_pos


def fit_predict_subset(
    train_x,
    y,
    test_x,
    train_idx,
    val_idx,
    train_mask,
    val_mask,
    test_mask,
    fallback_val,
    fallback_test,
    seed,
):
    val_pred = fallback_val.copy()
    test_pred = fallback_test.copy()
    sub_idx = train_idx[train_mask]

    if can_fit(y[sub_idx]):
        model = make_model(seed)
        model.fit(train_x.iloc[sub_idx], y[sub_idx], categorical_feature=CAT_COLS)
        if np.any(val_mask):
            val_pred[val_mask] = model.predict_proba(train_x.iloc[val_idx[val_mask]])[
                :, 1
            ]
        if np.any(test_mask):
            test_pred[test_mask] = model.predict_proba(test_x.iloc[test_mask])[:, 1]

    return val_pred, test_pred


def simplex_grid(k, step=0.05):
    units = int(round(1.0 / step))
    if k == 1:
        yield np.array([1.0])
    elif k == 2:
        for a in range(units + 1):
            yield np.array([a, units - a], dtype=float) / units
    elif k == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                yield np.array([a, b, units - a - b], dtype=float) / units
    else:
        raise ValueError("Only k<=3 is used in this simple blend search.")


os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

train_x = make_features(train.drop(columns=[TARGET]))
test_x = make_features(test)
train_x, test_x = set_categories(train_x, test_x)

feature_cols = [c for c in train_x.columns if c != ID_COL]
train_x = train_x[feature_cols]
test_x = test_x[feature_cols]

combined_x = pd.concat([train_x, test_x], axis=0, ignore_index=True)
domain_y = np.r_[np.zeros(len(train_x), dtype=int), np.ones(len(test_x), dtype=int)]
adv_oof = np.zeros(len(combined_x), dtype=np.float32)

adv_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr_idx, va_idx) in enumerate(adv_cv.split(combined_x, domain_y), 1):
    adv_model = make_model(RANDOM_STATE + 100 + fold, n_estimators=180, num_leaves=31)
    adv_model.fit(
        combined_x.iloc[tr_idx], domain_y[tr_idx], categorical_feature=CAT_COLS
    )
    adv_oof[va_idx] = adv_model.predict_proba(combined_x.iloc[va_idx])[:, 1]

adv_auc = roc_auc_score(domain_y, adv_oof)

train_x["AdvScore"] = adv_oof[: len(train_x)]
test_x["AdvScore"] = adv_oof[len(train_x) :]

stream_names = ["global", "compound_gated", "shift_gated"]
oof_streams = np.zeros((len(train_x), len(stream_names)), dtype=np.float32)
test_streams = np.zeros((len(test_x), len(stream_names)), dtype=np.float32)

if HAS_SGK:
    splitter = StratifiedGroupKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = splitter.split(train_x, y, groups)
else:
    splitter = GroupKFold(n_splits=N_FOLDS)
    splits = splitter.split(train_x, y, groups)

test_is_wet = test_x["Compound"].astype(str).isin(["INTERMEDIATE", "WET"]).to_numpy()
test_adv = test_x["AdvScore"].to_numpy()

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    x_tr = train_x.iloc[tr_idx]
    x_va = train_x.iloc[va_idx]
    y_tr = y[tr_idx]

    global_model = make_model(RANDOM_STATE + fold)
    global_model.fit(x_tr, y_tr, categorical_feature=CAT_COLS)

    global_val = global_model.predict_proba(x_va)[:, 1]
    global_test = global_model.predict_proba(test_x)[:, 1]

    oof_streams[va_idx, 0] = global_val
    test_streams[:, 0] += global_test / N_FOLDS

    tr_is_wet = x_tr["Compound"].astype(str).isin(["INTERMEDIATE", "WET"]).to_numpy()
    va_is_wet = x_va["Compound"].astype(str).isin(["INTERMEDIATE", "WET"]).to_numpy()

    compound_val = global_val.copy()
    compound_test = global_test.copy()

    dry_val, dry_test = fit_predict_subset(
        train_x,
        y,
        test_x,
        tr_idx,
        va_idx,
        train_mask=~tr_is_wet,
        val_mask=~va_is_wet,
        test_mask=~test_is_wet,
        fallback_val=compound_val,
        fallback_test=compound_test,
        seed=RANDOM_STATE + 200 + fold,
    )
    compound_val[~va_is_wet] = dry_val[~va_is_wet]
    compound_test[~test_is_wet] = dry_test[~test_is_wet]

    wet_val, wet_test = fit_predict_subset(
        train_x,
        y,
        test_x,
        tr_idx,
        va_idx,
        train_mask=tr_is_wet,
        val_mask=va_is_wet,
        test_mask=test_is_wet,
        fallback_val=compound_val,
        fallback_test=compound_test,
        seed=RANDOM_STATE + 300 + fold,
    )
    compound_val[va_is_wet] = wet_val[va_is_wet]
    compound_test[test_is_wet] = wet_test[test_is_wet]

    oof_streams[va_idx, 1] = compound_val
    test_streams[:, 1] += compound_test / N_FOLDS

    shift_cut = np.quantile(x_tr["AdvScore"].to_numpy(), 0.65)
    tr_high_shift = x_tr["AdvScore"].to_numpy() >= shift_cut
    va_high_shift = x_va["AdvScore"].to_numpy() >= shift_cut
    test_high_shift = test_adv >= shift_cut

    shift_val = global_val.copy()
    shift_test = global_test.copy()

    low_val, low_test = fit_predict_subset(
        train_x,
        y,
        test_x,
        tr_idx,
        va_idx,
        train_mask=~tr_high_shift,
        val_mask=~va_high_shift,
        test_mask=~test_high_shift,
        fallback_val=shift_val,
        fallback_test=shift_test,
        seed=RANDOM_STATE + 400 + fold,
    )
    shift_val[~va_high_shift] = low_val[~va_high_shift]
    shift_test[~test_high_shift] = low_test[~test_high_shift]

    high_val, high_test = fit_predict_subset(
        train_x,
        y,
        test_x,
        tr_idx,
        va_idx,
        train_mask=tr_high_shift,
        val_mask=va_high_shift,
        test_mask=test_high_shift,
        fallback_val=shift_val,
        fallback_test=shift_test,
        seed=RANDOM_STATE + 500 + fold,
    )
    shift_val[va_high_shift] = high_val[va_high_shift]
    shift_test[test_high_shift] = high_test[test_high_shift]

    oof_streams[va_idx, 2] = shift_val
    test_streams[:, 2] += shift_test / N_FOLDS

individual_auc = {
    name: float(roc_auc_score(y, oof_streams[:, i]))
    for i, name in enumerate(stream_names)
}

best_auc = -1.0
best_w = None
for w in simplex_grid(len(stream_names), step=0.05):
    pred = oof_streams @ w
    auc = roc_auc_score(y, pred)
    if auc > best_auc:
        best_auc = auc
        best_w = w

oof_pred = np.clip(oof_streams @ best_w, 0, 1)
test_pred = np.clip(test_streams @ best_w, 0, 1)

pd.DataFrame(
    {
        "row": np.arange(len(y)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = pd.DataFrame(
    {
        ID_COL: sample[ID_COL].to_numpy(),
        TARGET: test_pred,
    }
)
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
test_pred_df.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

print(
    json.dumps(
        {
            "cv_roc_auc": float(best_auc),
            "adversarial_train_test_auc": float(adv_auc),
            "individual_stream_auc": individual_auc,
            "blend_weights": {
                name: float(weight) for name, weight in zip(stream_names, best_w)
            },
            "research_hypotheses_llm_claimed_used": ["000968"],
            "files_written": [
                "./working/submission.csv",
                "./working/oof_predictions.csv.gz",
                "./working/test_predictions.csv.gz",
            ],
        },
        indent=2,
    )
)
