import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None
from sklearn.model_selection import GroupKFold

import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
DOMAIN_COL = "Race"
TESTING_RACE = "Pre-Season Testing"
RANDOM_STATE = 42
TESTING_WEIGHT = 0.10
EXPERT_BLEND_WEIGHT = 0.70

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz")).reset_index(drop=True)
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz")).reset_index(drop=True)
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
cat_cols = [
    c for c in feature_cols if train[c].dtype == "object" or test[c].dtype == "object"
]

for c in cat_cols:
    tr = train[c].astype("object").where(train[c].notna(), "__MISSING__").astype(str)
    te = test[c].astype("object").where(test[c].notna(), "__MISSING__").astype(str)
    cats = pd.Index(pd.concat([tr, te], ignore_index=True).unique())
    train[c] = pd.Categorical(tr, categories=cats)
    test[c] = pd.Categorical(te, categories=cats)

is_testing = train[DOMAIN_COL].astype(str).eq(TESTING_RACE).values
test_is_testing = test[DOMAIN_COL].astype(str).eq(TESTING_RACE).values
groups = train["Year"].astype(str) + "__" + train[DOMAIN_COL].astype(str)


def make_model(seed, small=False):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180 if small else 350,
        learning_rate=0.045 if small else 0.035,
        num_leaves=31 if small else 63,
        min_child_samples=60,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
        force_col_wise=True,
    )


def fit_model(row_idx, seed, sample_weight=None, small=False):
    row_idx = np.asarray(row_idx)
    if len(row_idx) == 0 or np.unique(y[row_idx]).size < 2:
        row_idx = np.arange(len(train))
        sample_weight = None
    model = make_model(seed, small=small)
    kwargs = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    if cat_cols:
        kwargs["categorical_feature"] = cat_cols
    model.fit(train.loc[row_idx, feature_cols], y[row_idx], **kwargs)
    return model


def predict(model, frame):
    return np.clip(model.predict_proba(frame[feature_cols])[:, 1], 1e-6, 1 - 1e-6)


def safe_auc(yt, yp):
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    if len(yt) == 0 or np.unique(yt).size < 2:
        return float("nan")
    return float(roc_auc_score(yt, yp))


if StratifiedGroupKFold is not None:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(splitter.split(train, y, groups))
else:
    splitter = GroupKFold(n_splits=5)
    folds = list(splitter.split(train, y, groups))

oof = {
    "exclude_testing": np.zeros(len(train), dtype=np.float32),
    "downweight_testing": np.zeros(len(train), dtype=np.float32),
    "testing_expert_blend": np.zeros(len(train), dtype=np.float32),
}

for fold, (tr_idx, va_idx) in enumerate(folds, 1):
    race_tr_idx = tr_idx[~is_testing[tr_idx]]
    testing_tr_idx = tr_idx[is_testing[tr_idx]]

    race_model = fit_model(race_tr_idx, RANDOM_STATE + fold)
    race_pred = predict(race_model, train.loc[va_idx])

    oof["exclude_testing"][va_idx] = race_pred

    weights = np.ones(len(tr_idx), dtype=np.float32)
    weights[is_testing[tr_idx]] = TESTING_WEIGHT
    down_model = fit_model(tr_idx, RANDOM_STATE + 100 + fold, sample_weight=weights)
    oof["downweight_testing"][va_idx] = predict(down_model, train.loc[va_idx])

    expert_pred = race_pred.copy()
    if len(testing_tr_idx) > 20 and np.unique(y[testing_tr_idx]).size == 2:
        testing_model = fit_model(testing_tr_idx, RANDOM_STATE + 200 + fold, small=True)
        va_testing_local = is_testing[va_idx]
        if va_testing_local.any():
            expert_only = predict(testing_model, train.loc[va_idx[va_testing_local]])
            expert_pred[va_testing_local] = (1.0 - EXPERT_BLEND_WEIGHT) * expert_pred[
                va_testing_local
            ] + EXPERT_BLEND_WEIGHT * expert_only
    oof["testing_expert_blend"][va_idx] = expert_pred

metrics = {}
for name, pred in oof.items():
    metrics[name] = {
        "auc_with_testing_rows": safe_auc(y, pred),
        "auc_without_testing_rows": safe_auc(y[~is_testing], pred[~is_testing]),
        "auc_testing_rows_only": safe_auc(y[is_testing], pred[is_testing]),
    }
    print(
        f"{name}: "
        f"Year-Race grouped CV ROC AUC with testing rows = {metrics[name]['auc_with_testing_rows']:.6f}, "
        f"without testing rows = {metrics[name]['auc_without_testing_rows']:.6f}, "
        f"testing rows only = {metrics[name]['auc_testing_rows_only']:.6f}"
    )


def selection_key(item):
    m = item[1]
    primary = m["auc_without_testing_rows"]
    if np.isnan(primary):
        primary = m["auc_with_testing_rows"]
    return primary


best_name, best_metrics = max(metrics.items(), key=selection_key)
print(f"Selected ablation: {best_name}")
print(
    f"Selected CV ROC AUC without testing rows: {best_metrics['auc_without_testing_rows']:.6f}"
)
print(
    f"Selected CV ROC AUC with testing rows: {best_metrics['auc_with_testing_rows']:.6f}"
)

if best_name == "exclude_testing":
    final_model = fit_model(np.where(~is_testing)[0], RANDOM_STATE + 1000)
    test_pred = predict(final_model, test)
elif best_name == "downweight_testing":
    full_idx = np.arange(len(train))
    full_weights = np.ones(len(train), dtype=np.float32)
    full_weights[is_testing] = TESTING_WEIGHT
    final_model = fit_model(full_idx, RANDOM_STATE + 1000, sample_weight=full_weights)
    test_pred = predict(final_model, test)
else:
    race_model = fit_model(np.where(~is_testing)[0], RANDOM_STATE + 1000)
    test_pred = predict(race_model, test)
    testing_idx = np.where(is_testing)[0]
    if (
        len(testing_idx) > 20
        and np.unique(y[testing_idx]).size == 2
        and test_is_testing.any()
    ):
        testing_model = fit_model(testing_idx, RANDOM_STATE + 2000, small=True)
        expert_test_pred = predict(testing_model, test.loc[test_is_testing])
        test_pred[test_is_testing] = (1.0 - EXPERT_BLEND_WEIGHT) * test_pred[
            test_is_testing
        ] + EXPERT_BLEND_WEIGHT * expert_test_pred

submission = sample[[ID_COL]].copy()
submission[TARGET] = np.clip(test_pred, 1e-6, 1 - 1e-6)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(oof[best_name], 1e-6, 1 - 1e-6),
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000904"],
    "metric": "roc_auc",
    "cv_scheme": "5-fold Year-Race grouped CV",
    "selected_ablation": best_name,
    "selected_auc_with_testing_rows": best_metrics["auc_with_testing_rows"],
    "selected_auc_without_testing_rows": best_metrics["auc_without_testing_rows"],
    "ablation_results": metrics,
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "oof_predictions_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
}
print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
