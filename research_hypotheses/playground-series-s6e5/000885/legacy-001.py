import json
import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

warnings.filterwarnings("ignore")

SEED = 42
TARGET = "PitNextLap"
ID_COL = "id"
INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)
N_JOBS = min(16, os.cpu_count() or 1)


def add_features(df):
    out = df.copy()
    eps = 1e-6

    lap = out["LapNumber"].astype(float).to_numpy()
    progress = np.clip(out["RaceProgress"].astype(float).to_numpy(), eps, None)
    tyre = np.maximum(out["TyreLife"].astype(float).to_numpy(), 1.0)
    est_total = np.maximum(lap / progress, 1.0)
    compound = out["Compound"].astype(str)

    out["EstimatedTotalLaps"] = est_total
    out["LapsRemaining"] = np.maximum(est_total - lap, 0.0)
    out["TyreLifeFracOfRace"] = tyre / est_total
    out["TyreLifeToLap"] = tyre / np.maximum(lap, 1.0)
    out["DegradationPerTyreLap"] = (
        out["Cumulative_Degradation"].astype(float).to_numpy() / tyre
    )
    out["AbsLapTimeDelta"] = np.abs(out["LapTime_Delta"].astype(float).to_numpy())
    out["PositionRaceProgress"] = out["Position"].astype(float).to_numpy() * progress
    out["StintTyreLife"] = out["Stint"].astype(float).to_numpy() * tyre
    out["LateRace"] = (progress >= 0.75).astype(np.int8)
    out["FreshTyre"] = (tyre <= 2).astype(np.int8)
    out["WetOrIntermediate"] = compound.isin(["WET", "INTERMEDIATE"]).astype(np.int8)
    out["SoftCompound"] = (compound == "SOFT").astype(np.int8)
    out["HardCompound"] = (compound == "HARD").astype(np.int8)
    return out


def make_predictive_model(seed):
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1400,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=70,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def make_adversarial_model(seed):
    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=700,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=120,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )


def fit_lgb(
    model, X_tr, y_tr, X_va, y_va, sample_weight=None, cat_cols=None, stopping_rounds=75
):
    kwargs = {}
    if cat_cols:
        kwargs["categorical_feature"] = cat_cols

    model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
        **kwargs,
    )
    return model


def main():
    train_raw = pd.read_csv(INPUT_DIR / "train.csv.gz")
    test_raw = pd.read_csv(INPUT_DIR / "test.csv.gz")
    sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

    train = add_features(train_raw)
    test = add_features(test_raw)

    feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
    cat_cols = [c for c in feature_cols if train[c].dtype == "object"]

    for col in cat_cols:
        categories = pd.Index(
            pd.concat([train[col], test[col]], ignore_index=True).astype(str).unique()
        )
        train[col] = pd.Categorical(train[col].astype(str), categories=categories)
        test[col] = pd.Categorical(test[col].astype(str), categories=categories)

    X = train[feature_cols]
    X_test = test[feature_cols]
    y = train_raw[TARGET].astype(int).to_numpy()

    X_adv = pd.concat([X, X_test], axis=0, ignore_index=True)
    domain_y = np.r_[
        np.zeros(len(X), dtype=np.int8),
        np.ones(len(X_test), dtype=np.int8),
    ]

    adv_idx = np.arange(len(domain_y))
    adv_tr, adv_va = train_test_split(
        adv_idx,
        test_size=0.25,
        random_state=SEED,
        stratify=domain_y,
    )

    adv_model = make_adversarial_model(SEED)
    fit_lgb(
        adv_model,
        X_adv.iloc[adv_tr],
        domain_y[adv_tr],
        X_adv.iloc[adv_va],
        domain_y[adv_va],
        cat_cols=cat_cols,
        stopping_rounds=50,
    )

    adv_pred = adv_model.predict_proba(X_adv.iloc[adv_va])[:, 1]
    adv_auc = roc_auc_score(domain_y[adv_va], adv_pred)

    p_test_like = np.clip(adv_model.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
    density_ratio = (p_test_like / (1.0 - p_test_like)) * (len(X) / len(X_test))
    lo, hi = np.quantile(density_ratio, [0.01, 0.99])
    lo, hi = max(float(lo), 0.2), min(float(hi), 5.0)
    if hi <= lo:
        lo, hi = np.quantile(density_ratio, [0.005, 0.995])
    sample_weight = np.clip(density_ratio, lo, hi)
    if np.isfinite(sample_weight.mean()) and sample_weight.mean() > 0:
        sample_weight = sample_weight / sample_weight.mean()
    else:
        sample_weight = np.ones(len(X), dtype=float)

    group_key = train_raw["Year"].astype(str) + "__" + train_raw["Race"].astype(str)
    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    except TypeError:
        cv = StratifiedGroupKFold(n_splits=5)

    oof = np.zeros(len(train_raw), dtype=np.float32)
    test_pred = np.zeros(len(test_raw), dtype=np.float64)
    fold_aucs = []
    splits = list(cv.split(X, y, groups=group_key))

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        model = make_predictive_model(SEED + fold)
        fit_lgb(
            model,
            X.iloc[tr_idx],
            y[tr_idx],
            X.iloc[va_idx],
            y[va_idx],
            sample_weight=sample_weight[tr_idx],
            cat_cols=cat_cols,
        )

        va_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = va_pred.astype(np.float32)
        test_pred += model.predict_proba(X_test)[:, 1] / len(splits)

        fold_auc = roc_auc_score(y[va_idx], va_pred)
        fold_aucs.append(float(fold_auc))
        print(f"fold {fold} StratifiedGroupKFold ROC AUC: {fold_auc:.6f}")

    group_order = (
        pd.DataFrame(
            {
                "group": group_key.to_numpy(),
                "Year": train_raw["Year"].to_numpy(),
                ID_COL: train_raw[ID_COL].to_numpy(),
            }
        )
        .groupby("group", as_index=False)
        .agg(Year=("Year", "min"), first_id=(ID_COL, "min"))
        .sort_values(["Year", "first_id", "group"])
    )

    ordered_groups = group_order["group"].tolist()
    found_forward = False
    forward_auc = None
    forward_rows = np.array([], dtype=int)
    forward_pred = np.array([], dtype=float)

    for frac in [0.20, 0.25, 0.30, 0.35, 0.40]:
        n_valid_groups = max(1, int(round(len(ordered_groups) * frac)))
        valid_groups = set(ordered_groups[-n_valid_groups:])
        valid_mask = group_key.isin(valid_groups).to_numpy()
        train_mask = ~valid_mask

        if (
            valid_mask.sum() > 0
            and train_mask.sum() > 0
            and len(np.unique(y[valid_mask])) == 2
            and len(np.unique(y[train_mask])) == 2
        ):
            found_forward = True
            break

    if found_forward:
        tr_idx = np.flatnonzero(train_mask)
        va_idx = np.flatnonzero(valid_mask)
        forward_model = make_predictive_model(SEED + 777)
        fit_lgb(
            forward_model,
            X.iloc[tr_idx],
            y[tr_idx],
            X.iloc[va_idx],
            y[va_idx],
            sample_weight=sample_weight[tr_idx],
            cat_cols=cat_cols,
        )
        forward_rows = va_idx
        forward_pred = forward_model.predict_proba(X.iloc[va_idx])[:, 1]
        forward_auc = float(roc_auc_score(y[va_idx], forward_pred))
        print(f"forward Year/Race holdout ROC AUC: {forward_auc:.6f}")
    else:
        print("forward Year/Race holdout ROC AUC: unavailable")

    oof_auc = float(roc_auc_score(y, oof))
    fold_mean = float(np.mean(fold_aucs))
    fold_std = float(np.std(fold_aucs))
    mean_minus_std = fold_mean - fold_std
    worst_fold = float(np.min(fold_aucs))
    robust_candidates = [mean_minus_std, worst_fold]
    if forward_auc is not None:
        robust_candidates.append(forward_auc)
    robust_auc = float(min(robust_candidates))

    print(f"adversarial validation ROC AUC: {adv_auc:.6f}")
    print(f"OOF ROC AUC: {oof_auc:.6f}")
    print(f"fold mean ROC AUC: {fold_mean:.6f}")
    print(f"fold mean-minus-std ROC AUC: {mean_minus_std:.6f}")
    print(f"worst-fold ROC AUC: {worst_fold:.6f}")
    print(f"robust selected ROC AUC: {robust_auc:.6f}")

    target_col = [c for c in sample.columns if c != ID_COL][0]
    submission = sample.copy()
    submission[target_col] = np.clip(test_pred, 0, 1)
    submission.to_csv(WORK_DIR / "submission.csv", index=False)
    submission.to_csv(
        WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
    )

    pd.DataFrame(
        {
            "row": np.arange(len(train_raw)),
            "target": y,
            "prediction": oof,
        }
    ).to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

    pd.DataFrame(
        {
            "row": forward_rows,
            "target": y[forward_rows] if len(forward_rows) else [],
            "prediction": forward_pred,
        }
    ).to_csv(
        WORK_DIR / "validation_predictions.csv.gz", index=False, compression="gzip"
    )

    result = {
        "research_hypotheses_llm_claimed_used": ["000885"],
        "metric": "roc_auc",
        "adversarial_validation_auc": float(adv_auc),
        "stratified_group_oof_auc": oof_auc,
        "stratified_group_fold_mean_auc": fold_mean,
        "stratified_group_fold_std_auc": fold_std,
        "stratified_group_mean_minus_std_auc": float(mean_minus_std),
        "stratified_group_worst_fold_auc": worst_fold,
        "forward_year_race_holdout_auc": forward_auc,
        "robust_selected_auc": robust_auc,
        "submission_path": str(WORK_DIR / "submission.csv"),
        "oof_predictions_path": str(WORK_DIR / "oof_predictions.csv.gz"),
        "test_predictions_path": str(WORK_DIR / "test_predictions.csv.gz"),
    }

    with open(WORK_DIR / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
