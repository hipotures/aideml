import os
import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]
GROUP_COLS = ["Year", "Race", "Driver"]
N_SPLITS = 5
N_THREADS = max(1, os.cpu_count() or 1)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values


def add_time_to_next_pit(df):
    df = df.copy()
    df["_orig_row"] = np.arange(len(df))
    df = df.sort_values(GROUP_COLS + ["LapNumber", ID_COL]).reset_index(drop=True)

    lower = np.zeros(len(df), dtype=np.float32)
    upper = np.zeros(len(df), dtype=np.float32)

    for _, idx in df.groupby(GROUP_COLS, sort=False).groups.items():
        idx = np.asarray(list(idx))
        laps = df.loc[idx, "LapNumber"].to_numpy()
        pit = df.loc[idx, "PitStop"].to_numpy()
        max_lap = laps.max()
        pit_laps = laps[pit == 1]

        for j, row_idx in enumerate(idx):
            future = pit_laps[pit_laps > laps[j]]
            if len(future):
                t = max(float(future[0] - laps[j]), 1.0)
                lower[row_idx] = t
                upper[row_idx] = t
            else:
                censor = max(float(max_lap - laps[j] + 1), 1.0)
                lower[row_idx] = censor
                upper[row_idx] = np.inf

    df["aft_lower"] = lower
    df["aft_upper"] = upper
    return (
        df.sort_values("_orig_row").drop(columns=["_orig_row"]).reset_index(drop=True)
    )


def make_features(train_df, test_df):
    full = pd.concat(
        [train_df.drop(columns=[TARGET], errors="ignore"), test_df],
        axis=0,
        ignore_index=True,
    )

    full["race_driver"] = full["Race"].astype(str) + "_" + full["Driver"].astype(str)
    full["year_race"] = full["Year"].astype(str) + "_" + full["Race"].astype(str)
    full["tyre_frac"] = full["TyreLife"] / full["LapNumber"].clip(lower=1)
    full["laps_remaining_est"] = (
        full["LapNumber"] / full["RaceProgress"].clip(0.01, 1.0)
    ) - full["LapNumber"]
    full["degradation_per_lap"] = full["Cumulative_Degradation"] / full[
        "TyreLife"
    ].clip(lower=1)
    full["is_wet_compound"] = full["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    full["recent_pit_context"] = (
        (full["PitStop"] == 1) | (full["TyreLife"] <= 2)
    ).astype(int)

    cat_cols = CAT_COLS + ["race_driver", "year_race"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    full[cat_cols] = enc.fit_transform(full[cat_cols].astype(str))

    features = [c for c in full.columns if c != ID_COL]
    full[features] = full[features].replace([np.inf, -np.inf], np.nan).fillna(-999)

    X_train = full.iloc[: len(train_df)][features].reset_index(drop=True)
    X_test = full.iloc[len(train_df) :][features].reset_index(drop=True)
    return X_train, X_test, features


def make_aft_dmatrix(X_part, lower, upper):
    dmat = xgb.DMatrix(X_part)
    dmat.set_float_info("label_lower_bound", lower.astype(np.float32))
    dmat.set_float_info("label_upper_bound", upper.astype(np.float32))
    return dmat


def predict_aft(booster, dmat):
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is not None:
        try:
            return booster.predict(dmat, iteration_range=(0, int(best_iter) + 1))
        except TypeError:
            best_ntree_limit = getattr(booster, "best_ntree_limit", 0)
            if best_ntree_limit:
                return booster.predict(dmat, ntree_limit=best_ntree_limit)
    return booster.predict(dmat)


train_aft = add_time_to_next_pit(train)
X, X_test, features = make_features(train, test)
groups = train[GROUP_COLS].astype(str).agg("|".join, axis=1).values

oof_cls = np.zeros(len(train), dtype=np.float32)
oof_aft = np.zeros(len(train), dtype=np.float32)
test_cls = np.zeros(len(test), dtype=np.float32)
test_aft_pred = np.zeros(len(test), dtype=np.float32)

gkf = GroupKFold(n_splits=N_SPLITS)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_depth=5,
        learning_rate=0.045,
        n_estimators=900,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=8,
        reg_lambda=2.0,
        random_state=2026 + fold,
        n_jobs=N_THREADS,
        early_stopping_rounds=60,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_cls[va_idx] = clf.predict_proba(X_va)[:, 1]
    test_cls += clf.predict_proba(X_test)[:, 1] / N_SPLITS

    dtr = make_aft_dmatrix(
        X_tr,
        train_aft.loc[tr_idx, "aft_lower"].values,
        train_aft.loc[tr_idx, "aft_upper"].values,
    )
    dva = make_aft_dmatrix(
        X_va,
        train_aft.loc[va_idx, "aft_lower"].values,
        train_aft.loc[va_idx, "aft_upper"].values,
    )
    dte = xgb.DMatrix(X_test)

    aft_params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "tree_method": "hist",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.4,
        "max_depth": 4,
        "eta": 0.04,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 12,
        "lambda": 3.0,
        "seed": 4026 + fold,
        "nthread": N_THREADS,
    }

    aft = xgb.train(
        aft_params,
        dtr,
        num_boost_round=750,
        evals=[(dva, "valid")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )

    pred_time_va = np.clip(predict_aft(aft, dva), 0.25, 100.0)
    pred_time_te = np.clip(predict_aft(aft, dte), 0.25, 100.0)

    oof_aft[va_idx] = np.clip(1.0 - np.exp(-1.0 / pred_time_va), 0.0005, 0.9995)
    test_aft_pred += (
        np.clip(1.0 - np.exp(-1.0 / pred_time_te), 0.0005, 0.9995) / N_SPLITS
    )

    fold_blend = 0.75 * oof_cls[va_idx] + 0.25 * oof_aft[va_idx]
    print(f"fold {fold} roc_auc={roc_auc_score(y_va, fold_blend):.6f}")

blend_weights = np.linspace(0.0, 1.0, 21)

try:
    from joblib import Parallel, delayed

    workers = min(16, os.cpu_count() or 1)
    print(f"Evaluating {len(blend_weights)} blend candidates with {workers} workers")
    scores = Parallel(n_jobs=workers, prefer="threads")(
        delayed(roc_auc_score)(y, w * oof_cls + (1.0 - w) * oof_aft)
        for w in blend_weights
    )
except Exception:
    print(f"Evaluating {len(blend_weights)} blend candidates with 1 workers")
    scores = [
        roc_auc_score(y, w * oof_cls + (1.0 - w) * oof_aft) for w in blend_weights
    ]

best_i = int(np.argmax(scores))
best_w = float(blend_weights[best_i])
best_auc = float(scores[best_i])

oof_pred = best_w * oof_cls + (1.0 - best_w) * oof_aft
test_pred = best_w * test_cls + (1.0 - best_w) * test_aft_pred
test_pred = np.clip(test_pred, 0.0005, 0.9995)

print(f"cv_roc_auc={best_auc:.6f}")
print(f"best_classifier_weight={best_w:.2f}")

pred_by_id = pd.Series(test_pred, index=test[ID_COL].values)
submission = sample[[ID_COL]].copy()
submission[TARGET] = submission[ID_COL].map(pred_by_id).astype(float).values
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "research_hypotheses_llm_claimed_used": ["000348"],
    "metric": "roc_auc",
    "cv_roc_auc": best_auc,
    "best_classifier_weight": best_w,
    "n_features": len(features),
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)
