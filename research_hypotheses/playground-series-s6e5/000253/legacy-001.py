import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import expit, logit

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESIS_ID = "000253"
RANDOM_STATE = 2026

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
feature_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]
cat_cols = [c for c in feature_cols if train[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

all_features = pd.concat(
    [train[feature_cols], test[feature_cols]], axis=0, ignore_index=True
)
X_all = all_features.copy()

if cat_cols:
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
    )
    X_all[cat_cols] = enc.fit_transform(X_all[cat_cols].astype(str))
for c in num_cols:
    X_all[c] = pd.to_numeric(X_all[c], errors="coerce")
X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(-999)

X = X_all.iloc[: len(train)].reset_index(drop=True)
X_test = X_all.iloc[len(train) :].reset_index(drop=True)

domain_y = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
domain_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
domain_oof = np.zeros(len(X_all))

for tr_idx, va_idx in domain_cv.split(X_all, domain_y):
    if HAS_LGBM:
        dclf = LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=80,
            reg_lambda=2.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        dclf = HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    dclf.fit(X_all.iloc[tr_idx], domain_y[tr_idx])
    domain_oof[va_idx] = dclf.predict_proba(X_all.iloc[va_idx])[:, 1]

domain_auc = roc_auc_score(domain_y, domain_oof)
p_test = float(len(test)) / float(len(train) + len(test))
p_train = 1.0 - p_test

d_train = np.clip(domain_oof[: len(train)], 0.02, 0.98)
weights = (d_train / (1.0 - d_train)) * (p_train / p_test)
weights = np.clip(weights, 0.25, 4.0)
weights = weights / np.mean(weights)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_weighted = np.zeros(len(train))
oof_unweighted = np.zeros(len(train))
test_pred_folds = []
weighted_fold_aucs = []
plain_fold_aucs = []
baseline_fold_aucs = []


def make_model(seed):
    if HAS_LGBM:
        return LGBMClassifier(
            objective="binary",
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=63,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=120,
            reg_lambda=4.0,
            reg_alpha=0.05,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    return HistGradientBoostingClassifier(
        max_iter=350,
        learning_rate=0.04,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        random_state=seed,
    )


for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    model_w = make_model(RANDOM_STATE + fold)
    model_u = make_model(RANDOM_STATE + 100 + fold)

    model_w.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=weights[tr_idx])
    model_u.fit(X.iloc[tr_idx], y[tr_idx])

    pred_w = model_w.predict_proba(X.iloc[va_idx])[:, 1]
    pred_u = model_u.predict_proba(X.iloc[va_idx])[:, 1]

    oof_weighted[va_idx] = pred_w
    oof_unweighted[va_idx] = pred_u
    test_pred_folds.append(model_w.predict_proba(X_test)[:, 1])

    plain_fold_aucs.append(roc_auc_score(y[va_idx], pred_w))
    weighted_fold_aucs.append(
        roc_auc_score(y[va_idx], pred_w, sample_weight=weights[va_idx])
    )
    baseline_fold_aucs.append(roc_auc_score(y[va_idx], pred_u))

cv_auc = roc_auc_score(y, oof_weighted)
cv_weighted_auc = roc_auc_score(y, oof_weighted, sample_weight=weights)
baseline_auc = roc_auc_score(y, oof_unweighted)

pred_labels = (oof_weighted >= np.quantile(oof_weighted, 1.0 - y.mean())).astype(int)
cm = confusion_matrix(y, pred_labels, labels=[0, 1]).astype(float)
col_sums = np.maximum(cm.sum(axis=0), 1.0)
C = cm / col_sums

test_pred_cv = np.mean(test_pred_folds, axis=0)
test_bin = (test_pred_cv >= np.quantile(oof_weighted, 1.0 - y.mean())).astype(int)
q_test = np.array([(test_bin == 0).mean(), (test_bin == 1).mean()])
try:
    bbse_prior = np.linalg.solve(C, q_test)
    bbse_prior = np.clip(bbse_prior, 0.001, 0.999)
    bbse_prior = bbse_prior / bbse_prior.sum()
    estimated_test_pos_prior = float(bbse_prior[1])
except Exception:
    estimated_test_pos_prior = float(y.mean())

train_prior = float(y.mean())
target_prior = float(
    np.clip(estimated_test_pos_prior, 0.25 * train_prior, 4.0 * train_prior)
)
intercept_delta = logit(target_prior) - logit(train_prior)
intercept_delta = float(np.clip(intercept_delta, -1.5, 1.5))

corrected_oof = expit(logit(np.clip(oof_weighted, 1e-6, 1 - 1e-6)) + intercept_delta)
corrected_auc = roc_auc_score(y, corrected_oof)
corrected_weighted_auc = roc_auc_score(y, corrected_oof, sample_weight=weights)

final_model = make_model(RANDOM_STATE + 999)
final_model.fit(X, y, sample_weight=weights)
test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred_corrected = expit(logit(np.clip(test_pred, 1e-6, 1 - 1e-6)) + intercept_delta)

submission = sample[[ID_COL]].copy()
submission[TARGET] = np.clip(test_pred_corrected, 0.0, 1.0)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_weighted,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_predictions = sample[[ID_COL]].copy()
test_predictions[TARGET] = np.clip(test_pred_corrected, 0.0, 1.0)
test_predictions.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "research_hypotheses_llm_claimed_used": [HYPOTHESIS_ID],
    "metric": "roc_auc",
    "cv_roc_auc_weighted_model": float(cv_auc),
    "cv_importance_weighted_roc_auc": float(cv_weighted_auc),
    "cv_roc_auc_unweighted_baseline": float(baseline_auc),
    "domain_classifier_roc_auc": float(domain_auc),
    "train_positive_prior": train_prior,
    "bbse_estimated_test_positive_prior": estimated_test_pos_prior,
    "logit_intercept_delta": intercept_delta,
    "corrected_cv_roc_auc": float(corrected_auc),
    "corrected_cv_importance_weighted_roc_auc": float(corrected_weighted_auc),
}
with open(os.path.join(WORKING_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"CV ROC AUC: {cv_auc:.6f}")
print(f"Importance-weighted CV ROC AUC: {cv_weighted_auc:.6f}")
print(f"Unweighted baseline CV ROC AUC: {baseline_auc:.6f}")
print(f"Domain classifier ROC AUC: {domain_auc:.6f}")
print(f"BBSE estimated test positive prior: {estimated_test_pos_prior:.6f}")
print(f"Corrected CV ROC AUC: {corrected_auc:.6f}")
print(json.dumps(result, sort_keys=True))
