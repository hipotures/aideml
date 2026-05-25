import os
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

SEED = 2026
INPUT_DIR = Path("./input")
WORK_DIR = Path("./working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
FRESH_SLICK_BLOCK = os.environ.get("FRESH_SLICK_BLOCK", "compact").lower()
if FRESH_SLICK_BLOCK not in {"compact", "baseline", "full", "none"}:
    raise ValueError("FRESH_SLICK_BLOCK must be one of: compact, baseline, full, none")


def clean_columns(df):
    used = set()
    out = []
    for col in df.columns:
        new = re.sub(r"[^A-Za-z0-9_]+", "_", str(col)).strip("_")
        if not new:
            new = "feature"
        base, i = new, 1
        while new in used:
            i += 1
            new = f"{base}_{i}"
        used.add(new)
        out.append(new)
    df = df.copy()
    df.columns = out
    return df


def add_fresh_slick_features(df, mode):
    df = df.copy()

    lap = df["LapNumber"].astype(float)
    progress = df["RaceProgress"].astype(float).clip(1e-4, 1.0)
    total_laps = np.clip(lap / progress, lap, 80.0)

    remaining_now = np.maximum(0.0, total_laps - lap)
    remaining_after_next = np.maximum(0.0, total_laps - lap - 1.0)

    slick_life = {"SOFT": 18.0, "MEDIUM": 28.0, "HARD": 38.0}
    compound_life = {
        "SOFT": 18.0,
        "MEDIUM": 28.0,
        "HARD": 38.0,
        "INTERMEDIATE": 18.0,
        "WET": 22.0,
    }

    compound = df["Compound"].astype(str).str.upper()
    current_limit = compound.map(compound_life).fillna(28.0).astype(float)
    current_margin = current_limit - (df["TyreLife"].astype(float) + remaining_now)
    needs_stop = current_margin < 0.0

    margins = {}
    viable = {}
    for comp, life in slick_life.items():
        margin = life - remaining_after_next
        margins[comp] = margin
        viable[comp] = margin >= 0.0

    different_can_finish = np.zeros(len(df), dtype=bool)
    for comp in slick_life:
        different_can_finish |= viable[comp] & (compound.to_numpy() != comp)

    if mode in {"baseline", "full"}:
        df["FinishWindow_CurrentMargin"] = current_margin
        df["FinishWindow_NeedsStop"] = needs_stop.astype(np.int8)
        df["FinishWindow_RemainingAfterNext"] = remaining_after_next

    if mode in {"compact", "full"}:
        soft_v = viable["SOFT"]
        med_v = viable["MEDIUM"]
        hard_v = viable["HARD"]

        option_count = (
            soft_v.astype(np.int8) + med_v.astype(np.int8) + hard_v.astype(np.int8)
        )
        softest_viable = np.select(
            [soft_v, med_v, hard_v],
            [1, 2, 3],
            default=0,
        ).astype(np.int8)

        df["FreshSlick_OptionCount"] = option_count
        df["FreshSlick_SoftestViable"] = softest_viable
        df["FreshSlick_BestFinishMargin"] = np.maximum.reduce(
            [margins["SOFT"], margins["MEDIUM"], margins["HARD"]]
        )
        df["FreshSlick_DifferentCanFinish"] = different_can_finish.astype(np.int8)
        df["NeedsStop_And_DifferentSlickCanFinish"] = (
            needs_stop.to_numpy() & different_can_finish
        ).astype(np.int8)
        df["FreshSlick_SOFT_FinishMargin"] = margins["SOFT"]
        df["FreshSlick_MEDIUM_FinishMargin"] = margins["MEDIUM"]
        df["FreshSlick_HARD_FinishMargin"] = margins["HARD"]

    if mode == "full":
        df["FreshSlick_AnyCanFinish_alias"] = (
            viable["SOFT"] | viable["MEDIUM"] | viable["HARD"]
        ).astype(np.int8)
        df["FreshSlick_SOFT_CanFinish_alias"] = viable["SOFT"].astype(np.int8)
        df["FreshSlick_MEDIUM_CanFinish_alias"] = viable["MEDIUM"].astype(np.int8)
        df["FreshSlick_HARD_CanFinish_alias"] = viable["HARD"].astype(np.int8)
        df["FreshSlick_DifferentOptionCount_alias"] = sum(
            (viable[comp] & (compound.to_numpy() != comp)).astype(np.int8)
            for comp in slick_life
        )

    return df


train = clean_columns(pd.read_csv(INPUT_DIR / "train.csv.gz"))
test = clean_columns(pd.read_csv(INPUT_DIR / "test.csv.gz"))
sample = pd.read_csv(INPUT_DIR / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
train_base = train.drop(columns=[TARGET])
n_train = len(train_base)

combined = pd.concat([train_base, test], axis=0, ignore_index=True)
combined = add_fresh_slick_features(combined, FRESH_SLICK_BLOCK)

drop_cols = [ID_COL]
features = [c for c in combined.columns if c not in drop_cols]

cat_cols = [c for c in ["Compound", "Driver", "Race"] if c in features]
for c in cat_cols:
    combined[c] = combined[c].astype("category")

X = combined.iloc[:n_train][features].copy()
X_test = combined.iloc[n_train:][features].copy()

pos = max(float(y.sum()), 1.0)
neg = float(len(y) - y.sum())
scale_pos_weight = neg / pos

params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=1200,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_alpha=0.05,
    reg_lambda=1.5,
    scale_pos_weight=scale_pos_weight,
    random_state=SEED,
    n_jobs=max(1, os.cpu_count() or 1),
    verbosity=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof = np.zeros(n_train, dtype=float)
test_pred = np.zeros(len(X_test), dtype=float)
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits

    fold_auc = roc_auc_score(y[va_idx], oof[va_idx])
    fold_scores.append(fold_auc)
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = roc_auc_score(y, oof)
print(f"5-fold OOF ROC AUC: {cv_auc:.6f}")
print(f"Fresh slick block mode: {FRESH_SLICK_BLOCK}")

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0.0, 1.0)
submission.to_csv(WORK_DIR / "submission.csv", index=False)
submission.to_csv(WORK_DIR / "test_predictions.csv.gz", index=False, compression="gzip")

oof_df = pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": np.clip(oof, 0.0, 1.0),
    }
)
oof_df.to_csv(WORK_DIR / "oof_predictions.csv.gz", index=False, compression="gzip")

result = {
    "metric_name": "roc_auc",
    "metric_value": float(cv_auc),
    "fold_scores": [float(x) for x in fold_scores],
    "validation": "5-fold StratifiedKFold out-of-fold",
    "fresh_slick_block_mode": FRESH_SLICK_BLOCK,
    "research_hypotheses_llm_claimed_used": ["000003"],
}

for filename in ["result.json", "review.json"]:
    with open(WORK_DIR / filename, "w") as f:
        json.dump(result, f, indent=2)
