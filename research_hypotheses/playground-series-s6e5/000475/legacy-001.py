import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
SEED = 42


def add_pit_cycle_features(df):
    df = df.copy()
    group_cols = ["Race", "Year", "Driver"]
    sort_cols = group_cols + ["LapNumber", ID_COL]

    df["_orig_order"] = np.arange(len(df))
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    df["PitStop"] = df["PitStop"].fillna(0).astype(np.int8)
    g = df.groupby(group_cols, sort=False, observed=True)

    df["previous_lap_pitstop"] = g["PitStop"].shift(1).fillna(0).astype(np.int8)
    df["pitstop_lag2"] = g["PitStop"].shift(2).fillna(0).astype(np.int8)

    pitstops_so_far = g["PitStop"].cumsum()
    df["_pit_lap_marker"] = np.where(df["PitStop"].eq(1), df["LapNumber"], np.nan)
    df["_last_pit_lap"] = df.groupby(group_cols, sort=False, observed=True)[
        "_pit_lap_marker"
    ].ffill()

    stint_elapsed = (df["TyreLife"].astype(float) - 1.0).clip(lower=0)
    df["laps_since_last_pit"] = (
        df["LapNumber"]
        .astype(float)
        .sub(df["_last_pit_lap"])
        .fillna(stint_elapsed)
        .clip(lower=0)
    ).astype(np.float32)

    df["laps_until_current_stint_start"] = stint_elapsed.astype(np.float32)

    df["current_is_outlap_proxy"] = (
        df["previous_lap_pitstop"].eq(1)
        | df["TyreLife"].le(1.5)
        | df["laps_since_last_pit"].between(0.5, 1.5)
    ).astype(np.int8)

    df["first_2_laps_after_stop"] = (
        ((pitstops_so_far > 0) & df["laps_since_last_pit"].le(2.0))
        | df["TyreLife"].le(2.0)
        | df["previous_lap_pitstop"].eq(1)
        | df["pitstop_lag2"].eq(1)
    ).astype(np.int8)

    df["long_stint_without_stop"] = (
        df["PitStop"].eq(0)
        & df["TyreLife"].ge(24.0)
        & df["laps_until_current_stint_start"].ge(23.0)
    ).astype(np.int8)

    df = df.sort_values("_orig_order", kind="mergesort").reset_index(drop=True)
    return df.drop(columns=["_orig_order", "_pit_lap_marker", "_last_pit_lap"])


def make_model(seed, n_estimators, scale_pos_weight):
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
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
        n_jobs=min(16, os.cpu_count() or 1),
        verbosity=-1,
    )


def main():
    os.makedirs(WORKING_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
    sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

    y = train[TARGET].astype(int).to_numpy()
    n_train = len(train)

    combined = pd.concat(
        [train.drop(columns=[TARGET]), test],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    combined = add_pit_cycle_features(combined)

    cat_cols = [c for c in ["Compound", "Driver", "Race"] if c in combined.columns]
    for col in cat_cols:
        combined[col] = combined[col].astype("category")

    train_fe = combined.iloc[:n_train].reset_index(drop=True)
    test_fe = combined.iloc[n_train:].reset_index(drop=True)

    feature_cols = [c for c in train_fe.columns if c != ID_COL]
    X = train_fe[feature_cols].copy()
    X_test = test_fe[feature_cols].copy()

    groups = train["Race"].astype(str) + "_" + train["Year"].astype(str)

    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=SEED
        )
        splits = list(splitter.split(X, y, groups=groups))
        cv_name = "StratifiedGroupKFold"
    else:
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X, y))
        cv_name = "StratifiedKFold"

    oof = np.zeros(len(train), dtype=np.float32)
    fold_aucs = []
    best_iterations = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        y_tr, y_va = y[tr_idx], y[va_idx]
        pos = float(y_tr.sum())
        neg = float(len(y_tr) - pos)
        scale_pos_weight = neg / max(pos, 1.0)

        model = make_model(SEED + fold, 2000, scale_pos_weight)
        model.fit(
            X.iloc[tr_idx],
            y_tr,
            eval_set=[(X.iloc[va_idx], y_va)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = pred.astype(np.float32)

        fold_auc = roc_auc_score(y_va, pred)
        fold_aucs.append(float(fold_auc))
        best_iterations.append(int(getattr(model, "best_iteration_", 0) or 2000))
        print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

    cv_auc = roc_auc_score(y, oof)
    print(f"OOF ROC AUC ({cv_name}): {cv_auc:.6f}")
    print(f"Mean fold ROC AUC: {np.mean(fold_aucs):.6f}")

    pd.DataFrame(
        {
            "row": np.arange(len(train), dtype=np.int32),
            "target": y.astype(np.int8),
            "prediction": oof,
        }
    ).to_csv(
        os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    final_n_estimators = int(np.clip(np.median(best_iterations) * 1.10, 200, 2000))
    full_pos = float(y.sum())
    full_neg = float(len(y) - full_pos)
    full_scale_pos_weight = full_neg / max(full_pos, 1.0)

    final_model = make_model(SEED, final_n_estimators, full_scale_pos_weight)
    final_model.fit(X, y, categorical_feature=cat_cols)

    test_pred = final_model.predict_proba(X_test)[:, 1]
    test_pred = np.clip(test_pred, 0.0, 1.0)

    target_col = [c for c in sample.columns if c != ID_COL][0]
    submission = sample.copy()
    submission[target_col] = test_pred

    submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
    submission.to_csv(
        os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    result = {
        "metric": "roc_auc",
        "cv_strategy": cv_name,
        "cv_roc_auc": float(cv_auc),
        "mean_fold_roc_auc": float(np.mean(fold_aucs)),
        "fold_roc_auc": fold_aucs,
        "research_hypotheses_llm_claimed_used": ["000475"],
        "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        "oof_path": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
        "test_predictions_path": os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
