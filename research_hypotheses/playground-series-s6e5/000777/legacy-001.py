import os
import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 42

train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
base_features = [c for c in train.columns if c not in [ID_COL, TARGET]]
cat_cols = [c for c in base_features if train[c].dtype == "object"]

for c in cat_cols:
    both = pd.concat([train[c], test[c]], axis=0).astype("string").fillna("__MISSING__")
    cats = pd.Index(both.unique())
    train[c] = pd.Categorical(
        train[c].astype("string").fillna("__MISSING__"), categories=cats
    )
    test[c] = pd.Categorical(
        test[c].astype("string").fillna("__MISSING__"), categories=cats
    )

for c in base_features:
    if c not in cat_cols:
        train[c] = pd.to_numeric(train[c], errors="coerce").astype("float32")
        test[c] = pd.to_numeric(test[c], errors="coerce").astype("float32")

year_race_groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
year_groups = train["Year"].astype(str)

common_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=450,
    learning_rate=0.055,
    random_state=SEED,
    n_jobs=max(1, os.cpu_count() or 1),
    verbose=-1,
    deterministic=True,
    force_col_wise=True,
)

candidates = [
    {
        "name": "all_features_regularized",
        "features": base_features,
        "params": dict(
            num_leaves=31,
            min_child_samples=120,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.90,
            reg_alpha=0.10,
            reg_lambda=2.0,
        ),
    },
    {
        "name": "all_features_conservative",
        "features": base_features,
        "params": dict(
            num_leaves=15,
            min_child_samples=260,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.30,
            reg_lambda=5.0,
        ),
    },
    {
        "name": "drop_shift_identifiers_conservative",
        "features": [c for c in base_features if c not in ["Driver", "Race", "Year"]],
        "params": dict(
            num_leaves=15,
            min_child_samples=260,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.90,
            reg_alpha=0.30,
            reg_lambda=5.0,
        ),
    },
]


def safe_auc(y_true, pred):
    if np.unique(y_true).size < 2:
        return np.nan
    return float(roc_auc_score(y_true, pred))


def cat_features_for(features):
    return [c for c in cat_cols if c in features]


def make_model(extra_params, n_estimators=None):
    params = common_params.copy()
    params.update(extra_params)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    return lgb.LGBMClassifier(**params)


stress_splits = []
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for fold, (tr_idx, va_idx) in enumerate(
    sgkf.split(train[base_features], y, groups=year_race_groups), 1
):
    stress_splits.append(("StratifiedGroupKFold_Year_Race", fold, tr_idx, va_idx))

gkf = GroupKFold(n_splits=year_groups.nunique())
for fold, (tr_idx, va_idx) in enumerate(
    gkf.split(train[base_features], y, groups=year_groups), 1
):
    stress_splits.append(("GroupKFold_Year", fold, tr_idx, va_idx))

selection_rows = []
candidate_details = {}

for spec in candidates:
    features = spec["features"]
    cat_features = cat_features_for(features)
    fold_scores = []
    best_iters = []

    for family, fold, tr_idx, va_idx in stress_splits:
        model = make_model(spec["params"])
        model.fit(
            train.iloc[tr_idx][features],
            y[tr_idx],
            eval_set=[(train.iloc[va_idx][features], y[va_idx])],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
        )
        best_iter = (
            getattr(model, "best_iteration_", None) or common_params["n_estimators"]
        )
        pred = model.predict_proba(
            train.iloc[va_idx][features], num_iteration=best_iter
        )[:, 1]
        auc = safe_auc(y[va_idx], pred)
        fold_scores.append({"family": family, "fold": fold, "auc": auc})
        best_iters.append(best_iter)
        del model
        gc.collect()

    aucs = np.array([r["auc"] for r in fold_scores], dtype=float)
    finite = aucs[np.isfinite(aucs)]
    summary = {
        "name": spec["name"],
        "worst_auc": float(np.min(finite)),
        "q20_auc": float(np.quantile(finite, 0.20)),
        "mean_auc": float(np.mean(finite)),
        "median_best_iteration": int(np.median(best_iters)),
    }
    selection_rows.append(summary)
    candidate_details[spec["name"]] = fold_scores
    print(
        f"{spec['name']}: stress worst ROC AUC={summary['worst_auc']:.6f}, "
        f"q20={summary['q20_auc']:.6f}, mean={summary['mean_auc']:.6f}"
    )

selected_summary = sorted(
    selection_rows,
    key=lambda r: (r["worst_auc"], r["q20_auc"], r["mean_auc"]),
    reverse=True,
)[0]
selected = next(c for c in candidates if c["name"] == selected_summary["name"])
features = selected["features"]
cat_features = cat_features_for(features)

print(f"Selected candidate: {selected['name']}")
print(f"Selected stress-suite worst-fold ROC AUC: {selected_summary['worst_auc']:.6f}")

oof = np.zeros(len(train), dtype=np.float32)
oof_fold_aucs = []
oof_best_iters = []

for fold, (tr_idx, va_idx) in enumerate(
    sgkf.split(train[features], y, groups=year_race_groups), 1
):
    model = make_model(selected["params"])
    model.fit(
        train.iloc[tr_idx][features],
        y[tr_idx],
        eval_set=[(train.iloc[va_idx][features], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    best_iter = (
        getattr(model, "best_iteration_", None)
        or selected_summary["median_best_iteration"]
    )
    pred = model.predict_proba(train.iloc[va_idx][features], num_iteration=best_iter)[
        :, 1
    ]
    oof[va_idx] = pred.astype(np.float32)
    fold_auc = safe_auc(y[va_idx], pred)
    oof_fold_aucs.append(fold_auc)
    oof_best_iters.append(best_iter)
    print(f"OOF fold {fold} Year_Race ROC AUC: {fold_auc:.6f}")
    del model
    gc.collect()

oof_auc = safe_auc(y, oof)
print(f"5-fold Year_Race OOF ROC AUC: {oof_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int64),
        "target": y,
        "prediction": oof,
    }
).to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

final_n_estimators = (
    int(np.median(oof_best_iters))
    if oof_best_iters
    else selected_summary["median_best_iteration"]
)
final_n_estimators = max(50, final_n_estimators)

final_model = make_model(selected["params"], n_estimators=final_n_estimators)
final_model.fit(
    train[features],
    y,
    categorical_feature=cat_features,
)

test_pred = final_model.predict_proba(test[features])[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

pred_col = [c for c in sample.columns if c != ID_COL][0]
submission = sample.copy()
submission[pred_col] = test_pred
submission.to_csv(WORK / "submission.csv", index=False)

test_predictions = sample[[ID_COL]].copy()
test_predictions[pred_col] = test_pred
test_predictions.to_csv(
    WORK / "test_predictions.csv.gz", index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000777"],
    "selection_metric": "worst-fold ROC AUC across StratifiedGroupKFold(Year_Race) and GroupKFold(Year)",
    "selected_candidate": selected["name"],
    "stress_worst_auc": selected_summary["worst_auc"],
    "stress_q20_auc": selected_summary["q20_auc"],
    "stress_mean_auc": selected_summary["mean_auc"],
    "year_race_oof_auc": oof_auc,
    "final_n_estimators": final_n_estimators,
    "submission_path": str(WORK / "submission.csv"),
    "oof_path": str(WORK / "oof_predictions.csv.gz"),
    "test_predictions_path": str(WORK / "test_predictions.csv.gz"),
}
print("RESULT_JSON=" + json.dumps(review, sort_keys=True))
