import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
PURGE_LAPS = 2

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

train["_is_train"] = 1
test["_is_train"] = 0
test[target_col] = np.nan
all_df = pd.concat([train, test], axis=0, ignore_index=True)


def add_features(df):
    df = df.copy()
    eps = 1e-6

    df["RaceYear"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)
    df["DriverRaceYear"] = (
        df["Year"].astype(str)
        + "_"
        + df["Race"].astype(str)
        + "_"
        + df["Driver"].astype(str)
    )
    df["CompoundStint"] = df["Compound"].astype(str) + "_" + df["Stint"].astype(str)

    df["RaceLeft"] = 1.0 - df["RaceProgress"]
    df["TyreLifeOverLap"] = df["TyreLife"] / (df["LapNumber"] + eps)
    df["TyreLifeOverRaceProgress"] = df["TyreLife"] / (df["RaceProgress"] * 100.0 + eps)
    df["DegradationPerTyreLap"] = df["Cumulative_Degradation"] / (df["TyreLife"] + eps)
    df["LapDeltaPct"] = df["LapTime_Delta"] / (df["LapTime (s)"] + eps)
    df["AbsLapDelta"] = df["LapTime_Delta"].abs()
    df["AbsPositionChange"] = df["Position_Change"].abs()
    df["PositionNorm"] = df["Position"] / 20.0
    df["PitStopTyreLife"] = df["PitStop"] * df["TyreLife"]
    df["IsWetFamily"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)

    ordered = df.sort_values(["Year", "Race", "Driver", "LapNumber", id_col])
    grp = ordered.groupby(["Year", "Race", "Driver"], sort=False)

    lag_cols = [
        "PitStop",
        "LapTime (s)",
        "LapTime_Delta",
        "Position",
        "TyreLife",
        "Cumulative_Degradation",
    ]
    for col in lag_cols:
        ordered[f"Prev_{col}"] = grp[col].shift(1)

    ordered["LapGapInSequence"] = ordered["LapNumber"] - grp["LapNumber"].shift(1)
    ordered["TyreLifeDeltaInSequence"] = ordered["TyreLife"] - ordered["Prev_TyreLife"]
    ordered["LapTimeChangeInSequence"] = (
        ordered["LapTime (s)"] - ordered["Prev_LapTime (s)"]
    )

    lagged = ordered[
        [f"Prev_{c}" for c in lag_cols]
        + ["LapGapInSequence", "TyreLifeDeltaInSequence", "LapTimeChangeInSequence"]
    ]
    df = df.join(lagged)

    numeric_new = lagged.columns.tolist()
    for col in numeric_new:
        df[col] = df[col].fillna(0)

    return df


all_df = add_features(all_df)

cat_cols = ["Compound", "Driver", "Race", "RaceYear", "DriverRaceYear", "CompoundStint"]
for col in cat_cols:
    all_df[col] = all_df[col].astype("category")

exclude = {target_col, "_is_train", id_col}
feature_cols = [c for c in all_df.columns if c not in exclude]

train_fe = all_df[all_df["_is_train"] == 1].copy()
test_fe = all_df[all_df["_is_train"] == 0].copy()
y = train_fe[target_col].astype(int).to_numpy()


def import_lightgbm():
    try:
        import lightgbm as lgb

        return lgb
    except Exception as e:
        raise RuntimeError("lightgbm is required for this script") from e


lgb = import_lightgbm()


def make_model(
    n_estimators=700,
    learning_rate=0.04,
    random_state=RANDOM_STATE,
    scale_pos_weight=1.0,
):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=63,
        min_child_samples=60,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )


def pos_weight(y_part):
    positives = max(1, int(np.sum(y_part == 1)))
    negatives = max(1, int(np.sum(y_part == 0)))
    return min(100.0, negatives / positives)


def safe_auc(y_true, pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, pred)


def train_adversarial_classifier():
    adv_df = all_df[feature_cols].copy()
    adv_y = all_df["_is_train"].map({1: 0, 0: 1}).astype(int).to_numpy()

    max_rows = 300000
    if len(adv_df) > max_rows:
        idx = train_test_split(
            np.arange(len(adv_df)),
            train_size=max_rows,
            stratify=adv_y,
            random_state=RANDOM_STATE,
        )[0]
        adv_df = adv_df.iloc[idx]
        adv_y = adv_y[idx]

    tr_idx, va_idx = train_test_split(
        np.arange(len(adv_df)),
        test_size=0.25,
        stratify=adv_y,
        random_state=RANDOM_STATE,
    )

    model = make_model(n_estimators=350, learning_rate=0.05, random_state=RANDOM_STATE)
    model.fit(
        adv_df.iloc[tr_idx],
        adv_y[tr_idx],
        eval_set=[(adv_df.iloc[va_idx], adv_y[va_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    pred = model.predict_proba(adv_df.iloc[va_idx])[:, 1]
    auc = safe_auc(adv_y[va_idx], pred)
    imp = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    )
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return auc, imp


adv_auc, adv_imp = train_adversarial_classifier()
top_shift_features = adv_imp.head(12)["feature"].tolist()


def choose_split_key(top_features):
    race_shift = {
        "Year",
        "Race",
        "RaceYear",
        "DriverRaceYear",
        "Driver",
        "LapNumber",
        "RaceProgress",
    }
    if any(f in race_shift for f in top_features):
        return (
            "RaceYear",
            "race-year chronological groups selected from adversarial shift features",
        )
    return (
        "_IdBlock",
        "id-block chronological groups selected because adversarial shift was not race/year dominated",
    )


split_key, split_strategy = choose_split_key(top_shift_features)

if split_key == "_IdBlock":
    train_fe["_IdBlock"] = pd.qcut(
        train_fe[id_col].rank(method="first"),
        q=min(80, max(10, len(train_fe) // 5000)),
        labels=False,
        duplicates="drop",
    ).astype(str)
    test_fe["_IdBlock"] = "test"
    if "_IdBlock" not in feature_cols:
        feature_cols.append("_IdBlock")
    train_fe["_IdBlock"] = train_fe["_IdBlock"].astype("category")
    test_fe["_IdBlock"] = test_fe["_IdBlock"].astype("category")


def chronological_group_splits(df, group_col, n_splits=5):
    groups = (
        df.groupby(group_col, observed=False)[id_col]
        .min()
        .sort_values()
        .index.to_numpy()
    )
    if len(groups) <= n_splits:
        order = df.sort_values(id_col).index.to_numpy()
        tss = TimeSeriesSplit(n_splits=n_splits)
        for tr_pos, va_pos in tss.split(order):
            yield order[tr_pos], order[va_pos]
    else:
        tss = TimeSeriesSplit(n_splits=n_splits)
        for tr_gpos, va_gpos in tss.split(groups):
            tr_groups = set(groups[tr_gpos])
            va_groups = set(groups[va_gpos])
            tr_idx = df.index[df[group_col].isin(tr_groups)].to_numpy()
            va_idx = df.index[df[group_col].isin(va_groups)].to_numpy()
            yield tr_idx, va_idx


def purge_sequence_neighbors(df, train_idx, valid_idx, lap_gap=2):
    train_part = df.loc[train_idx, ["Year", "Race", "Driver", "LapNumber"]].copy()
    train_part["_idx"] = train_idx

    valid_ranges = (
        df.loc[valid_idx, ["Year", "Race", "Driver", "LapNumber"]]
        .groupby(["Year", "Race", "Driver"], observed=False)["LapNumber"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "valid_min_lap", "max": "valid_max_lap"})
    )

    merged = train_part.merge(valid_ranges, on=["Year", "Race", "Driver"], how="left")
    near_valid = (
        merged["valid_min_lap"].notna()
        & (merged["LapNumber"] >= merged["valid_min_lap"] - lap_gap)
        & (merged["LapNumber"] <= merged["valid_max_lap"] + lap_gap)
    )
    keep_idx = merged.loc[~near_valid, "_idx"].to_numpy()
    return keep_idx


def run_purged_sequence_cv():
    oof = np.full(len(train_fe), np.nan, dtype=float)
    fold_scores = []
    best_iterations = []

    index_to_pos = pd.Series(np.arange(len(train_fe)), index=train_fe.index)

    for fold, (tr_idx_labels, va_idx_labels) in enumerate(
        chronological_group_splits(train_fe, split_key, N_SPLITS), start=1
    ):
        tr_idx_labels = purge_sequence_neighbors(
            train_fe, tr_idx_labels, va_idx_labels, PURGE_LAPS
        )

        tr_pos = index_to_pos.loc[tr_idx_labels].to_numpy()
        va_pos = index_to_pos.loc[va_idx_labels].to_numpy()

        X_tr = train_fe.iloc[tr_pos][feature_cols]
        X_va = train_fe.iloc[va_pos][feature_cols]
        y_tr = y[tr_pos]
        y_va = y[va_pos]

        model = make_model(
            n_estimators=900,
            learning_rate=0.035,
            random_state=RANDOM_STATE + fold,
            scale_pos_weight=pos_weight(y_tr),
        )
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
        )

        pred = model.predict_proba(X_va)[:, 1]
        oof[va_pos] = pred
        score = safe_auc(y_va, pred)
        fold_scores.append(score)

        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None and best_iter > 0:
            best_iterations.append(best_iter)

        print(
            f"Purged sequence fold {fold}: auc={score:.6f}, "
            f"train_rows={len(tr_pos)}, valid_rows={len(va_pos)}"
        )

    mask = ~np.isnan(oof)
    overall = safe_auc(y[mask], oof[mask])
    return overall, fold_scores, best_iterations, oof, mask


def run_random_cv_comparison():
    max_rows = 220000
    if len(train_fe) > max_rows:
        idx = train_test_split(
            np.arange(len(train_fe)),
            train_size=max_rows,
            stratify=y,
            random_state=RANDOM_STATE,
        )[0]
    else:
        idx = np.arange(len(train_fe))

    X_cmp = train_fe.iloc[idx][feature_cols]
    y_cmp = y[idx]
    pred = np.zeros(len(idx), dtype=float)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, va) in enumerate(skf.split(X_cmp, y_cmp), start=1):
        model = make_model(
            n_estimators=450,
            learning_rate=0.04,
            random_state=RANDOM_STATE + 100 + fold,
            scale_pos_weight=pos_weight(y_cmp[tr]),
        )
        model.fit(
            X_cmp.iloc[tr],
            y_cmp[tr],
            eval_set=[(X_cmp.iloc[va], y_cmp[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        pred[va] = model.predict_proba(X_cmp.iloc[va])[:, 1]

    return safe_auc(y_cmp, pred)


random_cv_auc = run_random_cv_comparison()
purged_auc, purged_fold_scores, best_iterations, oof, oof_mask = (
    run_purged_sequence_cv()
)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train_fe))[oof_mask],
        "target": y[oof_mask],
        "prediction": oof[oof_mask],
    }
)
oof_df.to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

final_estimators = int(np.median(best_iterations)) if best_iterations else 650
final_estimators = max(100, final_estimators)

final_model = make_model(
    n_estimators=final_estimators,
    learning_rate=0.035,
    random_state=RANDOM_STATE + 999,
    scale_pos_weight=pos_weight(y),
)
final_model.fit(train_fe[feature_cols], y)

test_pred = final_model.predict_proba(test_fe[feature_cols])[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample[[id_col]].copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "research_hypotheses_llm_claimed_used": ["000486"],
    "metric": "roc_auc",
    "adversarial_train_test_auc": float(adv_auc),
    "adversarial_top_shift_features": top_shift_features,
    "split_strategy": split_strategy,
    "random_stratified_cv_auc_comparison": float(random_cv_auc),
    "purged_sequence_cv_auc": float(purged_auc),
    "purged_sequence_fold_auc": [
        None if np.isnan(v) else float(v) for v in purged_fold_scores
    ],
    "purge_laps": PURGE_LAPS,
    "oof_rows_scored": int(oof_mask.sum()),
    "final_model_estimators": int(final_estimators),
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
    "oof_path": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
}

print(f"Random stratified CV ROC AUC comparison: {random_cv_auc:.6f}")
print(f"Purged sequence 5-fold ROC AUC: {purged_auc:.6f}")
print(json.dumps(result, indent=2))
