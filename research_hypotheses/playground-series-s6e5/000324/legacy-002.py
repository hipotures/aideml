import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

SEED = 2026
ID_COL = "id"
TARGET = "PitNextLap"
HYPOTHESIS_ID = "000324"
TESTING_RACE = "Pre-Season Testing"
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
REGIMES = ["dry_race", "wet", "testing"]
REGIME_ALPHA = {"dry_race": 0.70, "wet": 0.75, "testing": 0.75}
MIN_SPECIALIST_ROWS = 800
MIN_SPECIALIST_POS = 20

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_regime_features(df):
    out = df.copy()
    race = out["Race"].astype(str)
    compound = out["Compound"].astype(str)
    out["Race_Year"] = race + "_" + out["Year"].astype(str)
    out["IsTesting"] = (race == TESTING_RACE).astype(np.int8)
    out["IsWetCompound"] = compound.isin(WET_COMPOUNDS).astype(np.int8)
    out["IsDryRace"] = ((out["IsTesting"] == 0) & (out["IsWetCompound"] == 0)).astype(
        np.int8
    )
    return out


def regime_masks(raw_df):
    race = raw_df["Race"].astype(str).reset_index(drop=True)
    compound = raw_df["Compound"].astype(str).reset_index(drop=True)
    testing = race.eq(TESTING_RACE).to_numpy()
    wet_compound = compound.isin(WET_COMPOUNDS).to_numpy()
    return {
        "dry_race": (~testing) & (~wet_compound),
        "wet": (~testing) & wet_compound,
        "testing": testing,
    }


train_x_raw = add_regime_features(train.drop(columns=[TARGET]))
test_x_raw = add_regime_features(test)
feature_cols = [c for c in train_x_raw.columns if c != ID_COL]
cat_cols = [c for c in feature_cols if train_x_raw[c].dtype == "object"]

combined = pd.concat(
    [train_x_raw[feature_cols], test_x_raw[feature_cols]], axis=0, ignore_index=True
)
for c in cat_cols:
    combined[c] = combined[c].astype("category")

X = combined.iloc[: len(train)].reset_index(drop=True)
X_test = combined.iloc[len(train) :].reset_index(drop=True)
y = train[TARGET].astype(int).to_numpy()
groups = train["Year"].astype(str).to_numpy()

train_regimes = regime_masks(train)
test_regimes = regime_masks(test)

model_params = dict(
    objective="binary",
    metric="auc",
    n_estimators=300,
    learning_rate=0.045,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=5.0,
    random_state=SEED,
    n_jobs=os.cpu_count() or 1,
    verbosity=-1,
    force_col_wise=True,
)

domain_params = dict(
    objective="binary",
    metric="auc",
    n_estimators=120,
    learning_rate=0.06,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=3.0,
    random_state=SEED,
    n_jobs=os.cpu_count() or 1,
    verbosity=-1,
    force_col_wise=True,
)


def fit_lgbm(X_part, y_part, weights=None, seed=SEED, specialist=False):
    params = model_params.copy()
    params["random_state"] = seed
    if specialist:
        params["min_child_samples"] = 40
        params["n_estimators"] = 260
    model = LGBMClassifier(**params)
    model.fit(X_part, y_part, sample_weight=weights, categorical_feature=cat_cols)
    return model


def shift_weights(X_source, X_target, seed):
    rng = np.random.default_rng(seed)
    max_each = min(180000, len(X_source), len(X_target))
    src_idx = (
        rng.choice(len(X_source), size=max_each, replace=False)
        if len(X_source) > max_each
        else np.arange(len(X_source))
    )
    tgt_idx = (
        rng.choice(len(X_target), size=max_each, replace=False)
        if len(X_target) > max_each
        else np.arange(len(X_target))
    )

    X_domain = pd.concat(
        [X_source.iloc[src_idx], X_target.iloc[tgt_idx]], axis=0, ignore_index=True
    )
    y_domain = np.r_[
        np.zeros(len(src_idx), dtype=np.int8), np.ones(len(tgt_idx), dtype=np.int8)
    ]

    params = domain_params.copy()
    params["random_state"] = seed
    clf = LGBMClassifier(**params)
    clf.fit(X_domain, y_domain, categorical_feature=cat_cols)

    p_test_domain = clf.predict_proba(X_source)[:, 1]
    p_test_domain = np.clip(p_test_domain, 0.02, 0.98)
    weights = p_test_domain / (1.0 - p_test_domain)
    weights = np.clip(weights, 0.20, 5.00)
    return weights / np.mean(weights)


def apply_specialists(X_tr, y_tr, w_tr, X_pred, base_pred, tr_masks, pred_masks, seed):
    pred = base_pred.copy()
    for i, regime in enumerate(REGIMES):
        tr_mask = tr_masks[regime]
        pred_mask = pred_masks[regime]
        if not pred_mask.any():
            continue

        y_sub = y_tr[tr_mask]
        n_pos = int(y_sub.sum())
        n_neg = int(len(y_sub) - n_pos)
        if (
            len(y_sub) < MIN_SPECIALIST_ROWS
            or n_pos < MIN_SPECIALIST_POS
            or n_neg < MIN_SPECIALIST_POS
        ):
            continue

        model = fit_lgbm(
            X_tr.iloc[tr_mask],
            y_sub,
            weights=w_tr[tr_mask],
            seed=seed + 101 + i,
            specialist=True,
        )
        spec_pred = model.predict_proba(X_pred.iloc[pred_mask])[:, 1]
        alpha = REGIME_ALPHA[regime]
        pred[pred_mask] = alpha * spec_pred + (1.0 - alpha) * pred[pred_mask]

    return np.clip(pred, 0.0, 1.0)


logo = LeaveOneGroupOut()
oof = np.zeros(len(train), dtype=np.float32)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(logo.split(X, y, groups), 1):
    heldout_year = sorted(set(groups[va_idx]))[0]
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    w_tr = shift_weights(X_tr, X_test, SEED + fold)
    pooled = fit_lgbm(X_tr, y_tr, weights=w_tr, seed=SEED + fold)
    base_va = pooled.predict_proba(X_va)[:, 1]

    tr_masks = {k: v[tr_idx] for k, v in train_regimes.items()}
    va_masks = {k: v[va_idx] for k, v in train_regimes.items()}
    pred_va = apply_specialists(
        X_tr, y_tr, w_tr, X_va, base_va, tr_masks, va_masks, SEED + fold
    )

    oof[va_idx] = pred_va
    fold_auc = roc_auc_score(y_va, pred_va)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} held-out Year={heldout_year} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"LeaveOneGroupOut(Year) ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

full_weights = shift_weights(X, X_test, SEED + 999)
pooled_full = fit_lgbm(X, y, weights=full_weights, seed=SEED + 999)
base_test = pooled_full.predict_proba(X_test)[:, 1]
test_pred = apply_specialists(
    X,
    y,
    full_weights,
    X_test,
    base_test,
    train_regimes,
    test_regimes,
    SEED + 999,
)

target_col = [c for c in sample.columns if c != ID_COL][0]
submission = pd.DataFrame(
    {
        ID_COL: sample[ID_COL].values,
        target_col: np.clip(test_pred, 0.0, 1.0),
    }
)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(
    json.dumps(
        {
            "metric": "roc_auc",
            "validation": "LeaveOneGroupOut_by_Year",
            "cv_auc": float(cv_auc),
            "fold_auc_mean": float(np.mean(fold_scores)),
            "fold_auc": [float(x) for x in fold_scores],
            "research_hypotheses_llm_claimed_used": [HYPOTHESIS_ID],
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        },
        sort_keys=True,
    )
)
