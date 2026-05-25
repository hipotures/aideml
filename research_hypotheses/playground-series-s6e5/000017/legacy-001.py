import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESES = ["000017"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
train_ids = train[ID_COL].values
test_ids = sample[ID_COL].values


def add_hypothesis_000017_features(df):
    df = df.copy()

    lap = df["LapNumber"].astype(float)
    tyre_life = df["TyreLife"].astype(float)
    progress = df["RaceProgress"].clip(0.01, 1.0).astype(float)
    deg = df["Cumulative_Degradation"].astype(float)
    compound = df["Compound"].astype(str)

    estimated_total_laps = np.maximum(lap / progress, lap)
    laps_remaining = np.maximum(estimated_total_laps - lap, 0.0)
    laps_remaining_next = np.maximum(laps_remaining - 1.0, 0.0)

    slick = compound.isin(["SOFT", "MEDIUM", "HARD"]).astype(float)
    wet = compound.isin(["INTERMEDIATE", "WET"]).astype(float)

    base_life = np.select(
        [
            compound.eq("SOFT"),
            compound.eq("MEDIUM"),
            compound.eq("HARD"),
            compound.eq("INTERMEDIATE"),
            compound.eq("WET"),
        ],
        [18.0, 26.0, 36.0, 22.0, 18.0],
        default=25.0,
    )

    fresh_slick_best_life = np.maximum.reduce(
        [
            np.full(len(df), 18.0),
            np.full(len(df), 26.0),
            np.full(len(df), 36.0),
        ]
    )

    degradation_penalty = np.clip(deg / 1200.0, -0.35, 0.75)
    current_effective_life_left = np.maximum(
        base_life * (1.0 - degradation_penalty) - tyre_life, -80.0
    )
    current_finish_margin = current_effective_life_left - laps_remaining

    fresh_slick_finish_margin = fresh_slick_best_life - laps_remaining_next
    current_pressure = -current_finish_margin

    current_fails = (current_finish_margin < 0).astype(int)
    fresh_slick_can_finish = (fresh_slick_finish_margin >= 0).astype(int)
    late_race = (laps_remaining <= 18).astype(int)

    df["est_total_laps"] = estimated_total_laps
    df["est_laps_remaining"] = laps_remaining
    df["est_laps_remaining_next"] = laps_remaining_next
    df["current_finish_margin"] = current_finish_margin
    df["current_finish_pressure"] = current_pressure
    df["fresh_slick_finish_margin_next"] = fresh_slick_finish_margin
    df["current_window_fails"] = current_fails
    df["fresh_slick_can_finish_next"] = fresh_slick_can_finish
    df["current_fails_and_fresh_slick_can_finish"] = (
        current_fails * fresh_slick_can_finish
    )
    df["pressure_margin_x_best_fresh_margin"] = (
        current_pressure * fresh_slick_finish_margin
    )
    df["late_current_fails_and_fresh_slick_can_finish"] = (
        late_race * current_fails * fresh_slick_can_finish
    )
    df["late_pressure_margin_x_best_fresh_margin"] = (
        late_race * current_pressure * fresh_slick_finish_margin
    )
    df["slick_current_fails_and_fresh_slick_can_finish"] = (
        slick * current_fails * fresh_slick_can_finish
    )
    df["wet_current_fails_and_fresh_slick_can_finish"] = (
        wet * current_fails * fresh_slick_can_finish
    )

    return df


full = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
full = add_hypothesis_000017_features(full)

cat_cols = full.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    full[col] = full[col].astype("category")

X = full.iloc[: len(train)].reset_index(drop=True)
X_test = full.iloc[len(train) :].reset_index(drop=True)

feature_cols = [c for c in X.columns if c != ID_COL]
X_model = X[feature_cols]
X_test_model = X_test[feature_cols]

try:
    from lightgbm import LGBMClassifier

    model_type = "lightgbm"
    base_params = dict(
        objective="binary",
        n_estimators=1800,
        learning_rate=0.025,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import OrdinalEncoder

    model_type = "sklearn_hgb"

oof = np.zeros(len(train), dtype=float)
test_pred = np.zeros(len(test), dtype=float)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, va_idx) in enumerate(cv.split(X_model, y), 1):
    X_tr, X_va = X_model.iloc[tr_idx].copy(), X_model.iloc[va_idx].copy()
    y_tr, y_va = y[tr_idx], y[va_idx]

    if model_type == "lightgbm":
        model = LGBMClassifier(**base_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_cols,
        )
        va_pred = model.predict_proba(X_va)[:, 1]
        te_pred = model.predict_proba(X_test_model)[:, 1]
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr_enc, X_va_enc, X_te_enc = X_tr.copy(), X_va.copy(), X_test_model.copy()
        if cat_cols:
            X_tr_enc[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
            X_va_enc[cat_cols] = enc.transform(X_va[cat_cols].astype(str))
            X_te_enc[cat_cols] = enc.transform(X_test_model[cat_cols].astype(str))
        model = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=450,
            max_leaf_nodes=63,
            l2_regularization=0.05,
            random_state=42 + fold,
        )
        model.fit(X_tr_enc, y_tr)
        va_pred = model.predict_proba(X_va_enc)[:, 1]
        te_pred = model.predict_proba(X_te_enc)[:, 1]

    oof[va_idx] = va_pred
    test_pred += te_pred / cv.n_splits
    print(f"Fold {fold} ROC AUC: {roc_auc_score(y_va, va_pred):.6f}")

auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {auc:.6f}")

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

review = {
    "metric": "roc_auc",
    "oof_roc_auc": float(auc),
    "research_hypotheses_llm_claimed_used": HYPOTHESES,
    "model": model_type,
    "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
    "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
}
with open(os.path.join(WORKING_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(json.dumps(review, indent=2))
