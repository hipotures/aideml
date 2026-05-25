import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
BASE_ESTIMATORS = 400
EARLY_STOPPING = 40
EMBARGO_LAPS = 2
ID_COL = "id"
TARGET = "PitNextLap"
INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)


def safe_feature_names(cols):
    used = set()
    mapping = {}
    for i, col in enumerate(cols):
        base = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_")
        if not base:
            base = f"f_{i}"
        if base[0].isdigit():
            base = f"f_{base}"
        name = base
        k = 2
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        mapping[col] = name
    return mapping


def prepare_features(train, test):
    features = [c for c in train.columns if c not in (ID_COL, TARGET)]
    cat_cols = [
        c for c in features if train[c].dtype == "object" or test[c].dtype == "object"
    ]

    for c in cat_cols:
        tr = (
            train[c].astype("object").where(train[c].notna(), "__MISSING__").astype(str)
        )
        te = test[c].astype("object").where(test[c].notna(), "__MISSING__").astype(str)
        cats = sorted(pd.concat([tr, te], ignore_index=True).unique().tolist())
        train[c] = pd.Categorical(tr, categories=cats)
        test[c] = pd.Categorical(te, categories=cats)

    mapping = safe_feature_names(features)
    X = train[features].rename(columns=mapping).copy()
    X_test = test[features].rename(columns=mapping).copy()
    model_cat_cols = [mapping[c] for c in cat_cols]
    return X, X_test, model_cat_cols


def model_params(n_estimators, seed):
    return dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        n_estimators=int(n_estimators),
        learning_rate=0.045,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.00,
        random_state=int(seed),
        n_jobs=max(1, min(8, os.cpu_count() or 1)),
        verbosity=-1,
        force_col_wise=True,
    )


def fit_lgbm(
    X_tr, y_tr, cat_cols, X_va=None, y_va=None, n_estimators=BASE_ESTIMATORS, seed=SEED
):
    model = LGBMClassifier(**model_params(n_estimators, seed))
    fit_kwargs = {}
    if cat_cols:
        fit_kwargs["categorical_feature"] = cat_cols
    if X_va is not None:
        fit_kwargs.update(
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    model.fit(X_tr, y_tr, **fit_kwargs)
    return model


def predict_lgbm(model, X):
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None and best_iter > 0:
        return model.predict_proba(X, num_iteration=best_iter)[:, 1]
    return model.predict_proba(X)[:, 1]


def ordered_race_embargo_splits(
    train_meta, n_splits=N_SPLITS, embargo_laps=EMBARGO_LAPS
):
    meta = (
        train_meta[[ID_COL, "Year", "Race", "LapNumber"]].reset_index(drop=True).copy()
    )
    meta["_block"] = meta["Year"].astype(str) + "||" + meta["Race"].astype(str)

    block_info = (
        meta.groupby("_block", sort=False)
        .agg(year=("Year", "min"), first_id=(ID_COL, "min"), race=("Race", "first"))
        .reset_index()
        .sort_values(["year", "first_id"], kind="mergesort")
    )
    ordered_blocks = block_info["_block"].to_numpy()
    folds = [x for x in np.array_split(ordered_blocks, n_splits) if len(x) > 0]
    all_idx = np.arange(len(meta))

    for fold, val_blocks in enumerate(folds, start=1):
        val_set = set(val_blocks.tolist())
        val_mask = meta["_block"].isin(val_set).to_numpy()
        train_mask = ~val_mask

        before_embargo = int(train_mask.sum())
        for block in val_blocks:
            rows = meta.loc[meta["_block"].eq(block)]
            if rows.empty:
                continue
            year = rows["Year"].iloc[0]
            race = rows["Race"].iloc[0]
            lo = rows["LapNumber"].min() - embargo_laps
            hi = rows["LapNumber"].max() + embargo_laps
            same_event_window = (
                meta["Year"].eq(year)
                & meta["Race"].eq(race)
                & meta["LapNumber"].between(lo, hi)
            )
            train_mask &= ~same_event_window.to_numpy()

        extra_embargoed = before_embargo - int(train_mask.sum())
        yield fold, all_idx[train_mask], all_idx[val_mask], {
            "race_blocks": len(val_blocks),
            "extra_embargoed_rows": extra_embargoed,
        }


def run_cv(name, split_iter, X, y, cat_cols):
    oof = np.full(len(y), np.nan, dtype=np.float32)
    aucs, best_iters = [], []

    for fold, tr_idx, va_idx, info in split_iter:
        model = fit_lgbm(
            X.iloc[tr_idx],
            y[tr_idx],
            cat_cols,
            X.iloc[va_idx],
            y[va_idx],
            n_estimators=BASE_ESTIMATORS,
            seed=SEED + fold,
        )
        pred = predict_lgbm(model, X.iloc[va_idx])
        auc = roc_auc_score(y[va_idx], pred)
        best_iter = getattr(model, "best_iteration_", None) or BASE_ESTIMATORS

        oof[va_idx] = pred.astype(np.float32)
        aucs.append(float(auc))
        best_iters.append(int(best_iter))

        detail = " ".join(f"{k}={v}" for k, v in info.items())
        print(
            f"{name}_fold_{fold}_auc={auc:.6f} "
            f"best_iter={best_iter} train_rows={len(tr_idx)} valid_rows={len(va_idx)} {detail}"
        )

    filled = ~np.isnan(oof)
    overall = roc_auc_score(y[filled], oof[filled])
    print(f"{name}_overall_auc={overall:.6f}")
    return float(overall), aucs, best_iters, oof


def main():
    train = pd.read_csv(INPUT_DIR / "train.csv.gz")
    test = pd.read_csv(INPUT_DIR / "test.csv.gz")
    sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

    y = train[TARGET].astype(np.int8).to_numpy()
    X, X_test, cat_cols = prepare_features(train, test)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    standard_splits = (
        (fold, tr_idx, va_idx, {})
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1)
    )
    standard_auc, _, _, _ = run_cv(
        "standard_stratified", standard_splits, X, y, cat_cols
    )

    embargo_splits = ordered_race_embargo_splits(train, N_SPLITS, EMBARGO_LAPS)
    embargo_auc, embargo_fold_aucs, embargo_best_iters, embargo_oof = run_cv(
        "embargoed_racewise", embargo_splits, X, y, cat_cols
    )

    pd.DataFrame(
        {
            "row": np.arange(len(train), dtype=np.int64),
            "target": y.astype(int),
            "prediction": embargo_oof,
        }
    ).to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

    valid_iters = [b for b in embargo_best_iters if b and b > 0]
    final_estimators = (
        int(np.clip(np.median(valid_iters), 50, BASE_ESTIMATORS))
        if valid_iters
        else BASE_ESTIMATORS
    )

    final_model = fit_lgbm(
        X,
        y,
        cat_cols,
        n_estimators=final_estimators,
        seed=SEED + 999,
    )
    test_pred = np.clip(predict_lgbm(final_model, X_test), 0.0, 1.0)

    pred_by_id = pd.DataFrame({ID_COL: test[ID_COL].to_numpy(), TARGET: test_pred})
    submission = sample[[ID_COL]].merge(pred_by_id, on=ID_COL, how="left")
    if submission[TARGET].isna().any():
        raise RuntimeError(
            "Some sample_submission ids were not found in test predictions."
        )

    submission.to_csv(WORK_DIR / "submission.csv", index=False)
    submission.to_csv(
        WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip"
    )

    result = {
        "research_hypotheses_llm_claimed_used": ["000980"],
        "metric": "roc_auc",
        "standard_stratified_cv_auc": standard_auc,
        "embargoed_racewise_cv_auc": embargo_auc,
        "embargoed_fold_aucs": embargo_fold_aucs,
        "embargo_laps": EMBARGO_LAPS,
        "final_n_estimators": final_estimators,
        "submission_path": str(WORK_DIR / "submission.csv"),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
