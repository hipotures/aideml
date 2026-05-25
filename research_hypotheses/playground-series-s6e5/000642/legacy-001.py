import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 642
N_SPLITS = 5

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values
test_ids = sample[ID_COL].values

dry_compounds = {"SOFT", "MEDIUM", "HARD"}
wet_compounds = {"INTERMEDIATE", "WET"}
soft_family = {"SOFT"}


def add_regime_features(df):
    df = df.copy()
    compound = df["Compound"].astype(str).str.upper()

    df["is_dry_tyre"] = compound.isin(dry_compounds).astype(np.int8)
    df["is_wet_intermediate"] = compound.isin(wet_compounds).astype(np.int8)
    df["is_unknown_rare_compound"] = (
        ~compound.isin(dry_compounds | wet_compounds)
    ).astype(np.int8)
    df["is_soft_family"] = compound.isin(soft_family).astype(np.int8)

    base_nums = [
        "TyreLife",
        "Cumulative_Degradation",
        "Stint",
        "Position",
        "RaceProgress",
        "LapNumber",
        "LapTime_Delta",
        "LapTime (s)",
        "Position_Change",
    ]

    for col in base_nums:
        if col in df.columns:
            df[f"{col}_x_dry"] = df[col] * df["is_dry_tyre"]
            df[f"{col}_x_wet"] = df[col] * df["is_wet_intermediate"]
            df[f"{col}_x_soft"] = df[col] * df["is_soft_family"]

    df["tyre_life_progress"] = df["TyreLife"] * df["RaceProgress"]
    df["degradation_per_tyre_life"] = df["Cumulative_Degradation"] / (
        df["TyreLife"] + 1.0
    )
    df["stint_progress"] = df["Stint"] * df["RaceProgress"]

    df["dry_degradation_pressure"] = df["degradation_per_tyre_life"] * df["is_dry_tyre"]
    df["wet_reactive_laptime_pressure"] = (
        df["LapTime_Delta"] * df["is_wet_intermediate"]
    )
    df["soft_life_pressure"] = df["TyreLife"] * df["is_soft_family"]

    return df


train_feat = add_regime_features(train.drop(columns=[TARGET]))
test_feat = add_regime_features(test)

features = [c for c in train_feat.columns if c != ID_COL]
cat_cols = [c for c in features if train_feat[c].dtype == "object"]

for c in cat_cols:
    combined = pd.concat([train_feat[c], test_feat[c]], axis=0).astype("category")
    categories = combined.cat.categories
    train_feat[c] = pd.Categorical(train_feat[c], categories=categories)
    test_feat[c] = pd.Categorical(test_feat[c], categories=categories)

X = train_feat[features]
X_test = test_feat[features]

dry_mask_all = train_feat["is_dry_tyre"].values == 1
wet_mask_all = train_feat["is_wet_intermediate"].values == 1
test_dry_mask = test_feat["is_dry_tyre"].values == 1
test_wet_mask = test_feat["is_wet_intermediate"].values == 1

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 80,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 2.0,
    "n_estimators": 2500,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
    "class_weight": None,
}


def fit_lgb(X_tr, y_tr, X_va, y_va, seed_offset=0):
    model = lgb.LGBMClassifier(**{**params, "random_state": RANDOM_STATE + seed_offset})
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(train), dtype=np.float32)
test_pred = np.zeros(len(test), dtype=np.float64)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr_all, y_tr_all = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]

    global_model = fit_lgb(X_tr_all, y_tr_all, X_va, y_va, seed_offset=fold * 10)

    tr_dry_idx = tr_idx[dry_mask_all[tr_idx]]
    tr_wet_idx = tr_idx[wet_mask_all[tr_idx]]

    dry_model = None
    wet_model = None

    if len(np.unique(y[tr_dry_idx])) == 2 and len(tr_dry_idx) >= 500:
        dry_val_idx = va_idx[dry_mask_all[va_idx]]
        eval_idx = (
            dry_val_idx
            if len(dry_val_idx) >= 100 and len(np.unique(y[dry_val_idx])) == 2
            else va_idx
        )
        dry_model = fit_lgb(
            X.iloc[tr_dry_idx],
            y[tr_dry_idx],
            X.iloc[eval_idx],
            y[eval_idx],
            seed_offset=fold * 10 + 1,
        )

    if len(np.unique(y[tr_wet_idx])) == 2 and len(tr_wet_idx) >= 200:
        wet_val_idx = va_idx[wet_mask_all[va_idx]]
        eval_idx = (
            wet_val_idx
            if len(wet_val_idx) >= 50 and len(np.unique(y[wet_val_idx])) == 2
            else va_idx
        )
        wet_model = fit_lgb(
            X.iloc[tr_wet_idx],
            y[tr_wet_idx],
            X.iloc[eval_idx],
            y[eval_idx],
            seed_offset=fold * 10 + 2,
        )

    va_pred = global_model.predict_proba(X_va)[:, 1]
    test_fold_pred = global_model.predict_proba(X_test)[:, 1]

    va_dry = train_feat.iloc[va_idx]["is_dry_tyre"].values == 1
    va_wet = train_feat.iloc[va_idx]["is_wet_intermediate"].values == 1

    if dry_model is not None:
        va_pred[va_dry] = dry_model.predict_proba(X_va.iloc[va_dry])[:, 1]
        test_fold_pred[test_dry_mask] = dry_model.predict_proba(
            X_test.iloc[test_dry_mask]
        )[:, 1]

    if wet_model is not None:
        va_pred[va_wet] = wet_model.predict_proba(X_va.iloc[va_wet])[:, 1]
        test_fold_pred[test_wet_mask] = wet_model.predict_proba(
            X_test.iloc[test_wet_mask]
        )[:, 1]

    oof[va_idx] = va_pred.astype(np.float32)
    test_pred += test_fold_pred / N_SPLITS

    fold_auc = roc_auc_score(y_va, va_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"Mean fold ROC AUC: {np.mean(fold_scores):.6f} +/- {np.std(fold_scores):.6f}")
print(f"OOF ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
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

result = {
    "metric": "roc_auc",
    "oof_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(v) for v in fold_scores],
    "research_hypotheses_llm_claimed_used": ["000642"],
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
    "oof_predictions_path": os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
}
print(json.dumps(result, indent=2))
