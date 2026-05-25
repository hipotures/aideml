import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

RANDOM_STATE = 979
N_SPLITS = 5
HYPOTHESIS_ID = "000979"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

y = train[target_col].astype(int).values
train_ids = train[id_col].values
test_ids = sample[id_col].values

features = [c for c in train.columns if c not in [target_col, id_col]]
cat_cols = [c for c in features if train[c].dtype == "object"]
cat_idx = [features.index(c) for c in cat_cols]

X = train[features].copy()
X_test = test[features].copy()

for c in cat_cols:
    X[c] = X[c].astype(str).fillna("__MISSING__")
    X_test[c] = X_test[c].astype(str).fillna("__MISSING__")

for c in features:
    if c not in cat_cols:
        med = X[c].median()
        X[c] = X[c].fillna(med)
        X_test[c] = X_test[c].fillna(med)

dry_train = ~X["Compound"].isin(["INTERMEDIATE", "WET"])
wet_train = X["Compound"].isin(["INTERMEDIATE", "WET"])
late_train = (X["TyreLife"] >= 8) & (X["RaceProgress"] >= 0.18)

dry_test = ~X_test["Compound"].isin(["INTERMEDIATE", "WET"])
wet_test = X_test["Compound"].isin(["INTERMEDIATE", "WET"])
late_test = (X_test["TyreLife"] >= 8) & (X_test["RaceProgress"] >= 0.18)

regimes = {
    "global": (np.ones(len(X), dtype=bool), np.ones(len(X_test), dtype=bool)),
    "dry": (dry_train.values, dry_test.values),
    "wet_inter": (wet_train.values, wet_test.values),
    "late_stint": (late_train.values, late_test.values),
}

try:
    from catboost import CatBoostClassifier, Pool
except Exception as e:
    raise ImportError(
        "catboost is required for hypothesis 000979 specialist categorical experts"
    ) from e


def make_model(seed, scale_pos_weight):
    return CatBoostClassifier(
        iterations=550,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        verbose=False,
        thread_count=max(1, min(8, os.cpu_count() or 1)),
        scale_pos_weight=scale_pos_weight,
    )


def safe_auc(y_true, pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, pred)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


base_names = list(regimes.keys())
oof_base = np.full((len(X), len(base_names)), np.nan, dtype=np.float32)
test_base = np.zeros((len(X_test), len(base_names)), dtype=np.float32)
folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for j, name in enumerate(base_names):
    train_mask, test_mask = regimes[name]
    test_fold_preds = np.zeros((len(X_test), N_SPLITS), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(folds.split(X, y)):
        fit_idx = tr_idx[train_mask[tr_idx]]
        pred_idx = va_idx[train_mask[va_idx]]

        if (
            len(fit_idx) < 1000
            or y[fit_idx].sum() < 10
            or (len(fit_idx) - y[fit_idx].sum()) < 10
        ):
            continue

        pos = y[fit_idx].sum()
        neg = len(fit_idx) - pos
        spw = float(np.clip(neg / max(pos, 1), 1.0, 50.0))

        model = make_model(RANDOM_STATE + 17 * fold + j, spw)
        train_pool = Pool(X.iloc[fit_idx], y[fit_idx], cat_features=cat_idx)
        valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_idx)

        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

        if len(pred_idx) > 0:
            oof_base[pred_idx, j] = model.predict_proba(X.iloc[pred_idx])[:, 1]

        if test_mask.any():
            preds = model.predict_proba(X_test)[:, 1]
            masked_preds = np.full(len(X_test), np.nan, dtype=np.float32)
            masked_preds[test_mask] = preds[test_mask]
            test_fold_preds[:, fold] = np.nan_to_num(masked_preds, nan=0.0)

    valid_counts = np.isfinite(oof_base[:, j]).sum()
    if valid_counts == 0:
        continue

    nonzero_test_counts = (test_fold_preds > 0).sum(axis=1)
    summed = test_fold_preds.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        test_base[:, j] = np.where(
            nonzero_test_counts > 0, summed / nonzero_test_counts, np.nan
        )

# Fill specialist gaps with global predictions; if any global gaps remain, use target prior.
prior = float(y.mean())
for j in range(len(base_names)):
    if base_names[j] != "global":
        missing = ~np.isfinite(oof_base[:, j])
        oof_base[missing, j] = oof_base[missing, 0]
        missing_test = ~np.isfinite(test_base[:, j])
        test_base[missing_test, j] = test_base[missing_test, 0]

oof_base = np.nan_to_num(oof_base, nan=prior)
test_base = np.nan_to_num(test_base, nan=prior)

Z = logit(oof_base)
Z_test = logit(test_base)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def monotone_logistic_fit(Z_train, y_train):
    n_features = Z_train.shape[1]

    def objective(params):
        intercept = params[0]
        weights = params[1:]
        pred = sigmoid(intercept + Z_train @ weights)
        return log_loss(y_train, pred, labels=[0, 1])

    init = np.r_[logit(np.array([y_train.mean()]))[0], np.ones(n_features) / n_features]
    bounds = [(None, None)] + [(0.0, 8.0)] * n_features
    res = minimize(
        objective, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500}
    )
    if not res.success:
        print("Warning: monotone stack optimizer did not fully converge:", res.message)
    return res.x[0], res.x[1:]


stack_intercept, stack_weights = monotone_logistic_fit(Z, y)
stack_oof_score = stack_intercept + Z @ stack_weights
stack_test_score = stack_intercept + Z_test @ stack_weights
stack_oof = sigmoid(stack_oof_score)
stack_test = sigmoid(stack_test_score)

platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
platt.fit(stack_oof_score.reshape(-1, 1), y)
oof_pred = platt.predict_proba(stack_oof_score.reshape(-1, 1))[:, 1]
test_pred = platt.predict_proba(stack_test_score.reshape(-1, 1))[:, 1]

auc = roc_auc_score(y, oof_pred)
print(f"5-fold OOF ROC AUC: {auc:.6f}")
print(
    "Base expert OOF AUCs:",
    json.dumps(
        {
            n: (
                None
                if np.isnan(safe_auc(y, oof_base[:, i]))
                else round(float(safe_auc(y, oof_base[:, i])), 6)
            )
            for i, n in enumerate(base_names)
        }
    ),
)
print(
    "Monotone stack weights:",
    json.dumps({n: round(float(w), 6) for n, w in zip(base_names, stack_weights)}),
)

submission = sample.copy()
submission[target_col] = np.clip(test_pred, 1e-6, 1 - 1e-6)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(oof_pred, 1e-6, 1 - 1e-6),
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        id_col: test_ids,
        target_col: np.clip(test_pred, 1e-6, 1 - 1e-6),
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "research_hypotheses_llm_claimed_used": [HYPOTHESIS_ID],
    "validation_metric": "roc_auc",
    "validation_score": float(auc),
    "n_splits": N_SPLITS,
    "experts": base_names,
    "outputs": {
        "submission": os.path.join(WORKING_DIR, "submission.csv"),
        "oof_predictions": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
        "test_predictions": os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    },
}
print(json.dumps(result, indent=2))
