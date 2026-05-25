import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

TARGET = "PitNextLap"
ID_COL = "id"

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values
test_ids = sample[ID_COL].values

train_features = train.drop(columns=[TARGET])
test_features = test.copy()


def add_hypothesis_000068_features(df):
    df = df.copy()

    expected_life = {
        "SOFT": 18.0,
        "MEDIUM": 28.0,
        "HARD": 38.0,
        "INTERMEDIATE": 22.0,
        "WET": 18.0,
    }
    old_thr = {
        "SOFT": 16.0,
        "MEDIUM": 25.0,
        "HARD": 35.0,
        "INTERMEDIATE": 18.0,
        "WET": 14.0,
    }
    very_old_thr = {
        "SOFT": 24.0,
        "MEDIUM": 36.0,
        "HARD": 48.0,
        "INTERMEDIATE": 28.0,
        "WET": 22.0,
    }

    comp = df["Compound"].astype(str)
    tyre = df["TyreLife"].astype(float)
    progress = df["RaceProgress"].astype(float)

    exp = comp.map(expected_life).fillna(28.0).astype(float)
    old = comp.map(old_thr).fillna(25.0).astype(float)
    very_old = comp.map(very_old_thr).fillna(36.0).astype(float)

    df["H000068_TyreLife_expected_ratio"] = tyre / exp
    df["H000068_TyreLife_old_margin"] = tyre - old
    df["H000068_TyreLife_very_old_margin"] = tyre - very_old
    df["H000068_Is_old_for_compound"] = (tyre >= old).astype(int)
    df["H000068_Is_very_old_for_compound"] = (tyre >= very_old).astype(int)
    df["H000068_TyreLife_x_RaceProgress"] = tyre * progress
    df["H000068_TyreLife_x_LateRace"] = tyre * (progress >= 0.70).astype(int)
    df["H000068_AgeRatio_x_RaceProgress"] = (
        df["H000068_TyreLife_expected_ratio"] * progress
    )

    bins = [-0.1, 5, 10, 15, 20, 30, 45, 80, 999]
    df["H000068_TyreLife_bin"] = pd.cut(tyre, bins=bins, labels=False).astype("int16")
    df["H000068_Compound_TyreLife_bin"] = (
        comp + "_" + df["H000068_TyreLife_bin"].astype(str)
    )

    return df


X = add_hypothesis_000068_features(train_features)
X_test = add_hypothesis_000068_features(test_features)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
all_data = pd.concat([X, X_test], axis=0, ignore_index=True)

for col in cat_cols:
    all_data[col] = all_data[col].astype("category")
    codes = all_data[col].cat.codes.astype("int32")
    codes = codes.where(codes >= 0, 0)
    all_data[col] = codes

X = all_data.iloc[: len(X)].reset_index(drop=True)
X_test = all_data.iloc[len(X) :].reset_index(drop=True)

feature_cols = [c for c in X.columns if c != ID_COL]
X_model = X[feature_cols]
X_test_model = X_test[feature_cols]

try:
    from lightgbm import LGBMClassifier

    model_factory = lambda seed: LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier

    model_factory = lambda seed: HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=48,
        l2_regularization=0.05,
        random_state=seed,
    )

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X_model), dtype=float)
test_pred = np.zeros(len(X_test_model), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_model, y), 1):
    X_tr, X_va = X_model.iloc[tr_idx], X_model.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = model_factory(42 + fold)
    model.fit(X_tr, y_tr)

    va_pred = model.predict_proba(X_va)[:, 1]
    te_pred = model.predict_proba(X_test_model)[:, 1]

    oof[va_idx] = va_pred
    test_pred += te_pred / skf.n_splits

    auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(auc)
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(oof)),
        "target": y,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: np.clip(test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result_review = {
    "metric": "roc_auc",
    "oof_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(v) for v in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000068"],
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
print(json.dumps(result_review, indent=2))
