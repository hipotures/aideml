import os
import gc
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SEED = 538
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
ID_COL = "id"
TARGET = "PitNextLap"

REGIMES = ["dry_race", "wet_or_intermediate", "testing", "final_zone"]
MIN_EXPERT_ROWS = 300
MIN_CLASS_COUNT = 5

os.makedirs(WORK_DIR, exist_ok=True)


def assign_strategy_regime(df):
    compound = df["Compound"].astype(str).str.upper()
    race = df["Race"].astype(str).str.lower()
    rp = pd.to_numeric(df["RaceProgress"], errors="coerce").fillna(0.0)
    stint = pd.to_numeric(df["Stint"], errors="coerce").fillna(0.0)
    tyre = pd.to_numeric(df["TyreLife"], errors="coerce").fillna(0.0)

    testing = race.str.contains(
        "pre-season|preseason|testing", regex=True, na=False
    ).to_numpy()
    wet = compound.isin(["INTERMEDIATE", "WET"]).to_numpy()
    final = ((rp >= 0.88) | ((rp >= 0.80) & (stint >= 2) & (tyre >= 10))).to_numpy()

    regime = np.full(len(df), "dry_race", dtype=object)
    regime[final] = "final_zone"
    regime[wet] = "wet_or_intermediate"
    regime[testing] = "testing"
    return regime


def make_features(df):
    out = df.copy()

    compound = out["Compound"].astype(str).str.upper()
    race = out["Race"].astype(str).str.lower()
    rp = pd.to_numeric(out["RaceProgress"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    tyre = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(0.0)
    stint = pd.to_numeric(out["Stint"], errors="coerce").fillna(0.0)
    lap = pd.to_numeric(out["LapNumber"], errors="coerce").fillna(0.0)
    deg = pd.to_numeric(out["Cumulative_Degradation"], errors="coerce").fillna(0.0)
    lap_delta = pd.to_numeric(out["LapTime_Delta"], errors="coerce").fillna(0.0)
    position = pd.to_numeric(out["Position"], errors="coerce").fillna(0.0)
    pos_change = pd.to_numeric(out["Position_Change"], errors="coerce").fillna(0.0)

    wet = compound.isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    dry = compound.isin(["SOFT", "MEDIUM", "HARD"]).astype(np.int8)
    testing = race.str.contains(
        "pre-season|preseason|testing", regex=True, na=False
    ).astype(np.int8)

    dry_race = ((dry == 1) & (testing == 0)).astype(np.float32)
    first_stint = (stint <= 1).astype(np.float32)

    out["is_wet_compound"] = wet
    out["is_dry_compound"] = dry
    out["is_testing"] = testing
    out["compound_family"] = np.where(wet == 1, "wet", "dry")
    out["strategy_regime"] = assign_strategy_regime(out)

    out["race_progress_left"] = (1.0 - rp).astype(np.float32)
    out["late_race"] = (rp >= 0.80).astype(np.int8)
    out["final_no_stop_zone"] = (
        (rp >= 0.88) | ((rp >= 0.80) & (stint >= 2) & (tyre >= 10))
    ).astype(np.int8)

    out["dry_mandatory_stop_pressure"] = (
        dry_race * first_stint * np.clip((rp - 0.35) / 0.65, 0.0, 1.0)
    ).astype(np.float32)
    out["wet_rule_exemption"] = wet.astype(np.int8)

    out["tyre_life_x_progress"] = (tyre * rp).astype(np.float32)
    out["stint_x_progress"] = (stint * rp).astype(np.float32)
    out["relative_tyre_age"] = (tyre / np.maximum(lap, 1.0)).astype(np.float32)
    out["degradation_per_tyre_life"] = (deg / np.maximum(tyre, 1.0)).astype(np.float32)
    out["abs_laptime_delta"] = np.abs(lap_delta).astype(np.float32)
    out["position_x_progress"] = (position * rp).astype(np.float32)
    out["position_change_abs"] = np.abs(pos_change).astype(np.float32)

    return out


def safe_auc(y_true, pred):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, pred))


def make_model(seed, y_slice):
    pos = float(np.sum(y_slice))
    neg = float(len(y_slice) - pos)
    scale_pos_weight = float(np.clip(neg / max(pos, 1.0), 1.0, 80.0))

    return LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=300,
        learning_rate=0.045,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=max(1, min(8, os.cpu_count() or 1)),
        verbosity=-1,
        force_col_wise=True,
    )


def can_train_expert(y_slice):
    counts = np.bincount(y_slice.astype(int), minlength=2)
    return (
        len(y_slice) >= MIN_EXPERT_ROWS
        and counts[0] >= MIN_CLASS_COUNT
        and counts[1] >= MIN_CLASS_COUNT
    )


def fit_encoder(df, cat_cols):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        enc.fit(df[cat_cols].fillna("__MISSING__").astype(str))
    return enc


def transform_features(df, enc, cat_cols):
    out = df.copy()
    if cat_cols:
        encoded = enc.transform(out[cat_cols].fillna("__MISSING__").astype(str))
        for i, col in enumerate(cat_cols):
            out[col] = encoded[:, i].astype(np.int32)

    for col in out.columns:
        if col not in cat_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)

    return out


def predict_with_regime_experts(
    X_train_raw, X_train_enc, y_train, X_pred_raw, X_pred_enc, cat_cols, seed
):
    global_model = make_model(seed, y_train)
    global_model.fit(X_train_enc, y_train, categorical_feature=cat_cols)
    global_pred = global_model.predict_proba(X_pred_enc)[:, 1]
    routed_pred = global_pred.copy()

    train_regime = X_train_raw["strategy_regime"].astype(str).to_numpy()
    pred_regime = X_pred_raw["strategy_regime"].astype(str).to_numpy()
    expert_status = {}

    for i, regime in enumerate(REGIMES):
        tr_idx = np.flatnonzero(train_regime == regime)
        pr_idx = np.flatnonzero(pred_regime == regime)

        if len(pr_idx) == 0:
            expert_status[regime] = {"prediction_rows": 0, "status": "unused"}
            continue

        y_reg = y_train[tr_idx]
        counts = np.bincount(y_reg.astype(int), minlength=2)

        if can_train_expert(y_reg):
            model = make_model(seed + 101 * (i + 1), y_reg)
            model.fit(X_train_enc.iloc[tr_idx], y_reg, categorical_feature=cat_cols)
            routed_pred[pr_idx] = model.predict_proba(X_pred_enc.iloc[pr_idx])[:, 1]
            expert_status[regime] = {
                "training_rows": int(len(tr_idx)),
                "training_positives": int(counts[1]),
                "prediction_rows": int(len(pr_idx)),
                "status": "expert",
            }
        else:
            expert_status[regime] = {
                "training_rows": int(len(tr_idx)),
                "training_positives": int(counts[1]),
                "prediction_rows": int(len(pr_idx)),
                "status": "global_fallback",
            }

    return routed_pred, global_pred, expert_status


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()

train_fe = make_features(train.drop(columns=[TARGET]))
test_fe = make_features(test)

X = train_fe.drop(columns=[ID_COL])
X_test = test_fe.drop(columns=[ID_COL])

cat_cols = [
    col
    for col in X.columns
    if X[col].dtype == "object" or str(X[col].dtype).startswith("category")
]

oof = np.zeros(len(X), dtype=np.float32)
global_oof = np.zeros(len(X), dtype=np.float32)
fold_reports = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr_raw = X.iloc[tr_idx].reset_index(drop=True)
    X_va_raw = X.iloc[va_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    enc = fit_encoder(X_tr_raw, cat_cols)
    X_tr_enc = transform_features(X_tr_raw, enc, cat_cols)
    X_va_enc = transform_features(X_va_raw, enc, cat_cols)

    pred, global_pred, status = predict_with_regime_experts(
        X_tr_raw, X_tr_enc, y_tr, X_va_raw, X_va_enc, cat_cols, SEED + fold
    )

    oof[va_idx] = pred
    global_oof[va_idx] = global_pred

    fold_auc = safe_auc(y_va, pred)
    fold_global_auc = safe_auc(y_va, global_pred)
    fold_reports.append(
        {
            "fold": fold,
            "regime_gated_auc": fold_auc,
            "global_fallback_auc": fold_global_auc,
            "experts": status,
        }
    )
    print(
        f"fold {fold}: regime_gated_roc_auc={fold_auc:.6f}, global_only_roc_auc={fold_global_auc:.6f}"
    )

    del X_tr_raw, X_va_raw, X_tr_enc, X_va_enc, pred, global_pred
    gc.collect()

cv_auc = safe_auc(y, oof)
global_cv_auc = safe_auc(y, global_oof)

regime_report = {}
train_regime = X["strategy_regime"].astype(str).to_numpy()
for regime in REGIMES:
    mask = train_regime == regime
    regime_report[regime] = {
        "rows": int(mask.sum()),
        "positives": int(y[mask].sum()),
        "regime_gated_auc": safe_auc(y[mask], oof[mask]) if mask.sum() else None,
        "global_only_auc": safe_auc(y[mask], global_oof[mask]) if mask.sum() else None,
    }

enc_full = fit_encoder(X, cat_cols)
X_full_enc = transform_features(X, enc_full, cat_cols)
X_test_enc = transform_features(X_test, enc_full, cat_cols)

test_pred, _, final_expert_status = predict_with_regime_experts(
    X, X_full_enc, y, X_test, X_test_enc, cat_cols, SEED + 1000
)
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = sample[[ID_COL]].copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train), dtype=np.int32),
        "target": y.astype(int),
        "prediction": oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_pred_df = sample[[ID_COL]].copy()
test_pred_df[TARGET] = test_pred
test_pred_df.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": cv_auc,
    "global_only_cv_roc_auc": global_cv_auc,
    "regime_report": regime_report,
    "fold_reports": fold_reports,
    "final_expert_status": final_expert_status,
    "research_hypotheses_llm_claimed_used": ["000538"],
    "artifacts": {
        "submission": os.path.join(WORK_DIR, "submission.csv"),
        "oof_predictions": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
        "test_predictions": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    },
}

with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"CV ROC AUC: {cv_auc:.6f}")
print(json.dumps(result, indent=2))
