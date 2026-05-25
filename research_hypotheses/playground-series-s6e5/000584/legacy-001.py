import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 584
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
y = train[target_col].astype(int).values


def add_hazard_features(df):
    df = df.copy()
    rp = df["RaceProgress"].astype(float).clip(1e-4, 1.0)
    lap = df["LapNumber"].astype(float)
    tyre = df["TyreLife"].astype(float)
    remaining = (lap / rp - lap).clip(0, 120)

    df["EstimatedRaceLaps"] = (lap + remaining).clip(1, 120)
    df["LapsRemaining_Est"] = remaining
    df["TyreLife_log1p"] = np.log1p(tyre)
    df["LapsRemaining_log1p"] = np.log1p(remaining)
    df["NeedToStopIndex"] = tyre * np.log1p(remaining)
    df["DegradationPerLap"] = df["Cumulative_Degradation"].astype(float) / tyre.clip(1)
    df["PositiveDegradationPressure"] = np.maximum(
        df["Cumulative_Degradation"].astype(float), 0
    ) * np.log1p(remaining)
    df["Cooldown"] = np.maximum(0.0, 3.0 - tyre) + 2.0 * df["PitStop"].astype(float)

    age_bins = [-np.inf, 2, 4, 6, 8, 10, 12, 15, 18, 22, 26, 32, 40, 50, 65, np.inf]
    rem_bins = [-np.inf, 1, 2, 3, 5, 8, 12, 16, 22, 30, 45, 65, np.inf]
    age_bin = (
        pd.cut(tyre, age_bins, labels=False, include_lowest=True)
        .astype("int16")
        .astype(str)
    )
    rem_bin = (
        pd.cut(remaining, rem_bins, labels=False, include_lowest=True)
        .astype("int16")
        .astype(str)
    )

    df["TyreAgeBin"] = age_bin
    df["LapsRemainingBin"] = rem_bin
    df["AgeRemainingBin"] = "a" + age_bin + "_r" + rem_bin
    df["CompoundAgeRemainingBin"] = (
        df["Compound"].astype(str) + "_" + df["AgeRemainingBin"]
    )
    return df


train_fe = add_hazard_features(train.drop(columns=[target_col]))
test_fe = add_hazard_features(test)

cat_cols = [
    c
    for c in train_fe.columns
    if train_fe[c].dtype == "object"
    or c
    in ["TyreAgeBin", "LapsRemainingBin", "AgeRemainingBin", "CompoundAgeRemainingBin"]
]

for c in cat_cols:
    both = (
        pd.concat([train_fe[c], test_fe[c]], ignore_index=True)
        .astype("string")
        .fillna("__MISSING__")
    )
    cats = pd.Index(both.unique())
    train_fe[c] = pd.Categorical(
        train_fe[c].astype("string").fillna("__MISSING__"), categories=cats
    )
    test_fe[c] = pd.Categorical(
        test_fe[c].astype("string").fillna("__MISSING__"), categories=cats
    )

hazard_features = [
    "TyreLife",
    "TyreLife_log1p",
    "LapsRemaining_Est",
    "LapsRemaining_log1p",
    "NeedToStopIndex",
    "Cumulative_Degradation",
    "DegradationPerLap",
    "PositiveDegradationPressure",
    "Cooldown",
    "PitStop",
    "Stint",
    "RaceProgress",
    "Position",
    "Position_Change",
    "LapTime_Delta",
    "Compound",
    "TyreAgeBin",
    "LapsRemainingBin",
    "AgeRemainingBin",
    "CompoundAgeRemainingBin",
]

baseline_features = [c for c in train_fe.columns if c != id_col]

hazard_constraints_map = {
    "TyreLife": 1,
    "TyreLife_log1p": 1,
    "LapsRemaining_Est": 1,
    "LapsRemaining_log1p": 1,
    "NeedToStopIndex": 1,
    "Cumulative_Degradation": 1,
    "DegradationPerLap": 1,
    "PositiveDegradationPressure": 1,
    "Cooldown": -1,
    "PitStop": -1,
}
hazard_constraints = [hazard_constraints_map.get(c, 0) for c in hazard_features]

pos = max(y.sum(), 1)
neg = len(y) - pos
scale_pos_weight = neg / pos
n_jobs = max(1, min(8, (os.cpu_count() or 2) - 1))

hazard_params = dict(
    objective="binary",
    metric="auc",
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=31,
    max_depth=6,
    min_child_samples=160,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.95,
    reg_alpha=0.05,
    reg_lambda=2.0,
    scale_pos_weight=scale_pos_weight,
    monotone_constraints=hazard_constraints,
    random_state=SEED,
    n_jobs=n_jobs,
    verbosity=-1,
)

baseline_params = dict(
    objective="binary",
    metric="auc",
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.5,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED + 1,
    n_jobs=n_jobs,
    verbosity=-1,
)

folds = list(
    StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(
        train_fe, y
    )
)


def train_cv(name, features, params):
    oof = np.zeros(len(train_fe), dtype=np.float64)
    test_pred = np.zeros(len(test_fe), dtype=np.float64)
    cat_features = [c for c in features if c in cat_cols]
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        model = lgb.LGBMClassifier(**params)
        X_tr, X_va = train_fe.iloc[tr_idx][features], train_fe.iloc[va_idx][features]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        )

        va_pred = model.predict_proba(X_va)[:, 1]
        te_pred = model.predict_proba(test_fe[features])[:, 1]
        oof[va_idx] = va_pred
        test_pred += te_pred / N_SPLITS

        auc = roc_auc_score(y_va, va_pred)
        scores.append(auc)
        print(f"{name} fold {fold} ROC AUC: {auc:.6f}")

    full_auc = roc_auc_score(y, oof)
    print(f"{name} OOF ROC AUC: {full_auc:.6f}")
    return oof, test_pred, scores, full_auc


haz_oof, haz_test, haz_scores, haz_auc = train_cv(
    "monotone_hazard_000584", hazard_features, hazard_params
)
base_oof, base_test, base_scores, base_auc = train_cv(
    "baseline_lgbm", baseline_features, baseline_params
)

weights = np.linspace(0, 1, 21)
blend_scores = []
for w in weights:
    pred = w * haz_oof + (1 - w) * base_oof
    blend_scores.append(roc_auc_score(y, pred))

best_idx = int(np.argmax(blend_scores))
best_w = float(weights[best_idx])
blend_oof = np.clip(best_w * haz_oof + (1 - best_w) * base_oof, 0, 1)
blend_test = np.clip(best_w * haz_test + (1 - best_w) * base_test, 0, 1)
blend_auc = roc_auc_score(y, blend_oof)

print(f"Best hazard blend weight: {best_w:.2f}")
print(f"Final 5-fold OOF ROC AUC: {blend_auc:.6f}")

pd.DataFrame(
    {"row": np.arange(len(train_fe)), "target": y, "prediction": blend_oof}
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample.copy()
test_pred_df[target_col] = blend_test
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)
test_pred_df.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

review = {
    "research_hypotheses_llm_claimed_used": ["000584"],
    "metric": "5-fold OOF ROC AUC",
    "cv_roc_auc": float(blend_auc),
    "monotone_hazard_oof_auc": float(haz_auc),
    "baseline_oof_auc": float(base_auc),
    "hazard_blend_weight": best_w,
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review))
