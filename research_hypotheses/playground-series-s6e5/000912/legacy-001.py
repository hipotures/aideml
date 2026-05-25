import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

ID_COL = "id"
TARGET = "PitNextLap"
GROUP_COLS = ["Year", "Race", "Driver"]
SEED = 912
N_SPLITS = 5
N_JOBS = min(16, os.cpu_count() or 1)
MAX_LAPS_AUX = 35

COMPOUND_BUCKET = {
    "HARD": "HARD",
    "MEDIUM": "MEDIUM",
    "SOFT": "SOFT",
    "INTERMEDIATE": "WET_TRACK",
    "WET": "WET_TRACK",
    "NONE": "NONE",
}


def clean_col(c):
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_")
    if not s:
        s = "col"
    if s[0].isdigit():
        s = "f_" + s
    return s


def sanitize_columns(df):
    out, seen = {}, {}
    for c in df.columns:
        base = clean_col(c)
        name = base
        k = 2
        while name in seen:
            name = f"{base}_{k}"
            k += 1
        seen[name] = True
        out[c] = name
    return df.rename(columns=out)


def safe_token(x):
    return clean_col(str(x)).lower()


def bucket_compound(x):
    return COMPOUND_BUCKET.get(str(x), "OTHER")


def signed_log1p(s):
    return np.sign(s) * np.log1p(np.abs(s))


def add_features(train_df, test_df):
    tr = train_df.drop(columns=[TARGET]).copy()
    te = test_df.copy()
    tr["_part"] = 0
    te["_part"] = 1
    full = pd.concat([tr, te], axis=0, ignore_index=True, sort=False)
    full["_orig_order"] = np.arange(len(full))

    for c in ["Race", "Driver", "Compound"]:
        full[c] = full[c].astype("string").fillna("Missing").astype(str)

    full["compound_bucket"] = full["Compound"].map(bucket_compound)
    full["is_wet_current"] = (full["compound_bucket"] == "WET_TRACK").astype(np.int8)
    full["is_slick_current"] = (
        full["compound_bucket"].isin(["HARD", "MEDIUM", "SOFT"]).astype(np.int8)
    )

    rp = full["RaceProgress"].clip(0.01, 1.0)
    tyre = full["TyreLife"].clip(lower=1)
    full["estimated_race_laps"] = (full["LapNumber"] / rp).clip(1, 120)
    full["laps_remaining_est"] = (full["estimated_race_laps"] - full["LapNumber"]).clip(
        0, 90
    )
    full["race_progress_left"] = (1.0 - full["RaceProgress"]).clip(0, 1)
    full["tyre_life_share_of_race"] = (
        full["TyreLife"] / full["estimated_race_laps"]
    ).clip(0, 2)
    full["degradation_per_tyre_lap"] = full["Cumulative_Degradation"] / tyre
    full["signed_log_degradation"] = signed_log1p(full["Cumulative_Degradation"])
    full["abs_laptime_delta"] = full["LapTime_Delta"].abs()
    full["position_x_progress"] = full["Position"] * full["RaceProgress"]
    full["tyre_x_progress"] = full["TyreLife"] * full["RaceProgress"]
    full["stint_x_tyre"] = full["Stint"] * full["TyreLife"]

    full = full.sort_values(GROUP_COLS + ["LapNumber", ID_COL]).copy()
    gb = full.groupby(GROUP_COLS, sort=False)

    hist_cols = [
        "LapTime_s",
        "LapTime_Delta",
        "Position",
        "TyreLife",
        "Cumulative_Degradation",
        "PitStop",
    ]
    for c in hist_cols:
        full[f"{c}_prev1"] = gb[c].shift(1)
        full[f"{c}_prev2"] = gb[c].shift(2)
        full[f"{c}_diff_prev1"] = full[c] - full[f"{c}_prev1"]

    full["prev_lap_gap"] = full["LapNumber"] - gb["LapNumber"].shift(1)
    full["pit_count_so_far"] = gb["PitStop"].cumsum()
    prev_pit_lap = (
        full["LapNumber"]
        .where(full["PitStop"].eq(1))
        .groupby([full[c] for c in GROUP_COLS], sort=False)
        .ffill()
    )
    full["laps_since_observed_pit"] = (
        (full["LapNumber"] - prev_pit_lap).fillna(full["TyreLife"]).clip(0, 100)
    )

    full = full.sort_values("_orig_order").reset_index(drop=True)
    drop_cols = [ID_COL, "_part", "_orig_order"]
    feature_cols = [c for c in full.columns if c not in drop_cols]

    X = full[feature_cols].copy()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("Missing").astype(str).astype("category")

    X_train = X.iloc[: len(train_df)].reset_index(drop=True)
    X_test = X.iloc[len(train_df) :].reset_index(drop=True)
    cat_cols = X_train.select_dtypes(include=["category"]).columns.tolist()
    return X_train, X_test, cat_cols


def make_aux_labels(df):
    laps_until = np.full(len(df), MAX_LAPS_AUX, dtype=np.float32)
    next_compound = np.full(len(df), "NONE", dtype=object)

    work = df[
        [ID_COL, "Year", "Race", "Driver", "LapNumber", "PitStop", "Compound"]
    ].copy()
    work["_row"] = np.arange(len(df))
    work = work.sort_values(GROUP_COLS + ["LapNumber", ID_COL])

    for _, g in work.groupby(GROUP_COLS, sort=False):
        row = g["_row"].to_numpy()
        lap = g["LapNumber"].to_numpy()
        pit = g["PitStop"].to_numpy().astype(int) == 1
        comp = g["Compound"].astype(str).to_numpy()
        n = len(g)
        pit_pos = np.flatnonzero(pit)
        if len(pit_pos) == 0:
            continue

        loc = np.searchsorted(pit_pos, np.arange(n) + 1)
        has_next = loc < len(pit_pos)
        next_pos = np.full(n, -1, dtype=int)
        next_pos[has_next] = pit_pos[loc[has_next]]

        cur_pos = np.arange(n)
        local_laps = np.full(n, MAX_LAPS_AUX, dtype=np.float32)
        local_laps[has_next] = lap[next_pos[has_next]] - lap[cur_pos[has_next]]
        laps_until[row] = np.clip(local_laps, 0, MAX_LAPS_AUX)

        after_pos = next_pos + 1
        has_after = has_next & (after_pos < n)
        local_comp = np.full(n, "NONE", dtype=object)
        local_comp[has_after] = comp[after_pos[has_after]]
        next_compound[row] = np.array(
            [bucket_compound(x) for x in local_comp], dtype=object
        )

    return laps_until, next_compound


def best_iter(model):
    bi = getattr(model, "best_iteration_", None)
    return bi if bi is not None and bi > 0 else None


def aligned_proba(model, X, n_classes):
    p = model.predict_proba(X, num_iteration=best_iter(model))
    out = np.zeros((len(X), n_classes), dtype=np.float32)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, j]
    return out


train = sanitize_columns(pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz")))
test = sanitize_columns(pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz")))
sample = sanitize_columns(
    pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))
)

y = train[TARGET].astype(int).to_numpy()
groups = train["Year"].astype(str) + "|" + train["Race"].astype(str)

X_base, X_test_base, cat_cols = add_features(train, test)
aux_laps, aux_comp = make_aux_labels(train)

le = LabelEncoder()
aux_comp_y = le.fit_transform(aux_comp)
aux_classes = list(le.classes_)
n_aux_classes = len(aux_classes)

folds = list(GroupKFold(n_splits=N_SPLITS).split(X_base, y, groups))

aux_laps_oof = np.zeros(len(train), dtype=np.float32)
aux_laps_test = np.zeros(len(test), dtype=np.float32)
aux_comp_oof = np.zeros((len(train), n_aux_classes), dtype=np.float32)
aux_comp_test = np.zeros((len(test), n_aux_classes), dtype=np.float32)

common_params = dict(
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=3.0,
    force_col_wise=True,
    verbosity=-1,
    n_jobs=N_JOBS,
)

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    reg = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=350,
        learning_rate=0.05,
        random_state=SEED + fold,
        **common_params,
    )
    reg.fit(
        X_base.iloc[tr_idx],
        aux_laps[tr_idx],
        eval_set=[(X_base.iloc[va_idx], aux_laps[va_idx])],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    aux_laps_oof[va_idx] = reg.predict(
        X_base.iloc[va_idx], num_iteration=best_iter(reg)
    )
    aux_laps_test += reg.predict(X_test_base, num_iteration=best_iter(reg)) / N_SPLITS

    clf_aux = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_aux_classes,
        metric="multi_logloss",
        n_estimators=320,
        learning_rate=0.05,
        random_state=SEED + 100 + fold,
        **common_params,
    )
    clf_aux.fit(
        X_base.iloc[tr_idx],
        aux_comp_y[tr_idx],
        eval_set=[(X_base.iloc[va_idx], aux_comp_y[va_idx])],
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    aux_comp_oof[va_idx] = aligned_proba(clf_aux, X_base.iloc[va_idx], n_aux_classes)
    aux_comp_test += aligned_proba(clf_aux, X_test_base, n_aux_classes) / N_SPLITS

aux_laps_oof = np.clip(aux_laps_oof, 0, MAX_LAPS_AUX)
aux_laps_test = np.clip(aux_laps_test, 0, MAX_LAPS_AUX)

X_main = X_base.copy()
X_test_main = X_test_base.copy()
X_main["aux_laps_to_next_pit_pred"] = aux_laps_oof
X_test_main["aux_laps_to_next_pit_pred"] = aux_laps_test
X_main["aux_near_pit_score"] = 1.0 / (1.0 + aux_laps_oof)
X_test_main["aux_near_pit_score"] = 1.0 / (1.0 + aux_laps_test)

for j, cls in enumerate(aux_classes):
    col = f"aux_next_comp_{safe_token(cls)}_prob"
    X_main[col] = aux_comp_oof[:, j]
    X_test_main[col] = aux_comp_test[:, j]

wet_idx = aux_classes.index("WET_TRACK") if "WET_TRACK" in aux_classes else None
if wet_idx is not None:
    X_main["aux_wet_track_next_prob"] = aux_comp_oof[:, wet_idx]
    X_test_main["aux_wet_track_next_prob"] = aux_comp_test[:, wet_idx]

main_cat_cols = X_main.select_dtypes(include=["category"]).columns.tolist()
main_oof = np.zeros(len(train), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    pos = y[tr_idx].sum()
    neg = len(tr_idx) - pos
    scale_pos_weight = float(np.sqrt(neg / max(pos, 1)))

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=90,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=4.0,
        scale_pos_weight=scale_pos_weight,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=N_JOBS,
        random_state=SEED + 200 + fold,
    )
    model.fit(
        X_main.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X_main.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=main_cat_cols,
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )

    va_pred = model.predict_proba(X_main.iloc[va_idx], num_iteration=best_iter(model))[
        :, 1
    ]
    main_oof[va_idx] = va_pred
    fold_auc = roc_auc_score(y[va_idx], va_pred)
    fold_scores.append(float(fold_auc))
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

    test_pred += (
        model.predict_proba(X_test_main, num_iteration=best_iter(model))[:, 1]
        / N_SPLITS
    )

cv_auc = roc_auc_score(y, main_oof)
test_pred = np.clip(
    np.nan_to_num(test_pred, nan=float(np.mean(main_oof))), 1e-6, 1 - 1e-6
)

oof_df = pd.DataFrame(
    {"row": np.arange(len(train)), "target": y, "prediction": main_oof}
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pred_by_id = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: test_pred})
submission = sample[[ID_COL]].merge(pred_by_id, on=ID_COL, how="left")
if submission[TARGET].isna().any():
    raise RuntimeError("Missing predictions for some sample_submission ids.")

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000912"],
    "metric": "grouped_5fold_roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": fold_scores,
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"CV ROC AUC: {cv_auc:.6f}")
print(json.dumps(result))
