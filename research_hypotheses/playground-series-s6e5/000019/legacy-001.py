import json
import os
import re
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42

train = pd.read_csv(INPUT_DIR / "train.csv.gz")
test = pd.read_csv(INPUT_DIR / "test.csv.gz")
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")
test_ids = test[ID_COL].copy()


def add_h000019_features(df):
    out = df.copy()
    group_cols = ["Year", "Race", "Driver"]
    order_cols = group_cols + ["LapNumber", ID_COL]
    needed = order_cols + ["Compound"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns for hypothesis 000019: {missing}")

    work = out[needed].copy()
    work["_orig_order"] = np.arange(len(work))
    work["_compound_norm"] = work["Compound"].astype(str).str.upper()
    work["_is_wet"] = (
        work["_compound_norm"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    )

    work = work.sort_values(order_cols, kind="mergesort")
    work["_wet_seen"] = (
        work.groupby(group_cols, sort=False)["_is_wet"].cummax().astype(np.int8)
    )

    dry_seen_sum = np.zeros(len(work), dtype=np.int16)
    for comp in ("SOFT", "MEDIUM", "HARD"):
        col = f"_seen_{comp.lower()}"
        work[col] = work["_compound_norm"].eq(comp).astype(np.int8)
        work[col] = work.groupby(group_cols, sort=False)[col].cummax().astype(np.int8)
        dry_seen_sum += work[col].to_numpy(dtype=np.int16)

    raw_dry_debt = np.maximum(0, 2 - dry_seen_sum).astype(np.int8)
    wet_seen = work["_wet_seen"].to_numpy(dtype=np.int8)
    exempted_dry_debt = np.where(wet_seen == 1, 0, raw_dry_debt).astype(np.int8)

    work["h000019_wet_seen_sofar"] = wet_seen
    work["h000019_remaining_dry_debt"] = exempted_dry_debt
    work["h000019_dry_debt_cleared_by_wet"] = (
        (wet_seen == 1) & (raw_dry_debt > 0)
    ).astype(np.int8)

    restore = work.sort_values("_orig_order", kind="mergesort")
    feature_cols = [
        "h000019_wet_seen_sofar",
        "h000019_remaining_dry_debt",
        "h000019_dry_debt_cleared_by_wet",
    ]
    for col in feature_cols:
        out[col] = restore[col].to_numpy()
    return out


def safe_feature_names(cols):
    seen = {}
    result = []
    for col in cols:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") or "feature"
        if base[0].isdigit():
            base = "f_" + base
        n = seen.get(base, 0)
        seen[base] = n + 1
        result.append(base if n == 0 else f"{base}_{n}")
    return result


n_train = len(train)
test[TARGET] = np.nan
all_df = pd.concat([train, test], ignore_index=True, sort=False)
all_df = add_h000019_features(all_df)

feature_cols = [c for c in all_df.columns if c not in [ID_COL, TARGET]]
X_all = all_df[feature_cols].copy()

cat_cols_original = [c for c in feature_cols if X_all[c].dtype == "object"]
for c in cat_cols_original:
    X_all[c] = X_all[c].astype("category")

renamed_cols = safe_feature_names(feature_cols)
name_map = dict(zip(feature_cols, renamed_cols))
X_all.columns = renamed_cols
cat_features = [name_map[c] for c in cat_cols_original]

X = X_all.iloc[:n_train].reset_index(drop=True)
X_test = X_all.iloc[n_train:].reset_index(drop=True)
y = train[TARGET].astype(int).reset_index(drop=True)

groups = (
    train["Year"].astype(str)
    + "|"
    + train["Race"].astype(str)
    + "|"
    + train["Driver"].astype(str)
)

try:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(X, y, groups))
except Exception as exc:
    print(f"StratifiedGroupKFold unavailable; falling back to StratifiedKFold: {exc}")
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(X, y))

pos = int(y.sum())
neg = int(len(y) - pos)
scale_pos_weight = neg / max(pos, 1)
n_jobs = max(1, min(8, os.cpu_count() or 1))

base_params = dict(
    objective="binary",
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.0,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=n_jobs,
    verbosity=-1,
    force_col_wise=True,
)

oof = np.zeros(len(X), dtype=np.float32)
fold_aucs = []
best_iterations = []

for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
    model = lgb.LGBMClassifier(**base_params)
    model.fit(
        X.iloc[tr_idx],
        y.iloc[tr_idx],
        eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_features,
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    best_iter = int(model.best_iteration_ or base_params["n_estimators"])
    best_iterations.append(best_iter)
    val_pred = model.predict_proba(X.iloc[va_idx], num_iteration=best_iter)[:, 1]
    oof[va_idx] = val_pred.astype(np.float32)

    fold_auc = roc_auc_score(y.iloc[va_idx], val_pred)
    fold_aucs.append(float(fold_auc))
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f} (best_iter={best_iter})")

cv_auc = roc_auc_score(y, oof)
print(f"OOF ROC AUC: {cv_auc:.6f}")

final_estimators = int(
    np.clip(np.median(best_iterations), 50, base_params["n_estimators"])
)
final_params = dict(base_params)
final_params["n_estimators"] = final_estimators

final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(X, y, categorical_feature=cat_features)

test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

pd.DataFrame(
    {
        "row": np.arange(len(y)),
        "target": y.to_numpy(),
        "prediction": oof,
    }
).to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

pred_col = [c for c in sample.columns if c != ID_COL][0]
test_pred_frame = pd.DataFrame({ID_COL: test_ids.to_numpy(), pred_col: test_pred})
submission = sample[[ID_COL]].merge(test_pred_frame, on=ID_COL, how="left")
submission[pred_col] = submission[pred_col].fillna(float(oof.mean()))

submission.to_csv(WORK_DIR / "submission.csv", index=False)
submission.to_csv(WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip")

review = {
    "research_hypotheses_llm_claimed_used": ["000019"],
    "evaluation_metric": "roc_auc",
    "cv_roc_auc": float(cv_auc),
    "fold_roc_auc": [float(x) for x in fold_aucs],
    "final_model_n_estimators": int(final_estimators),
    "feature_count": int(X.shape[1]),
}
with open(WORK_DIR / "result_review.json", "w") as f:
    json.dump(review, f, indent=2)

print(f"Saved submission: {WORK_DIR / 'submission.csv'}")
print(f"Saved OOF predictions: {WORK_DIR / 'oof_predictions.csv.gz'}")
print(f"Saved test predictions: {WORK_DIR / 'test_predictions.csv.gz'}")
