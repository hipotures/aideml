import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"

y = train[target_col].astype(int).values
test_ids = sample[id_col].values

feature_cols = [c for c in train.columns if c not in [target_col, id_col]]
cat_cols = [c for c in feature_cols if train[c].dtype == "object"]
num_cols = [c for c in feature_cols if c not in cat_cols]

all_feat = pd.concat(
    [train[feature_cols], test[feature_cols]], axis=0, ignore_index=True
)

for c in num_cols:
    all_feat[c] = all_feat[c].astype("float32")
for c in cat_cols:
    all_feat[c] = all_feat[c].astype(str).fillna("__MISSING__")

# Small, deterministic domain features.
all_feat["TyreLife_x_Progress"] = all_feat["TyreLife"] * all_feat["RaceProgress"]
all_feat["Deg_per_TyreLife"] = all_feat["Cumulative_Degradation"] / (
    all_feat["TyreLife"] + 1.0
)
all_feat["LapFrac_x_Stint"] = all_feat["RaceProgress"] * all_feat["Stint"]
all_feat["Abs_Position_Change"] = all_feat["Position_Change"].abs()
all_feat["IsWetCompound"] = (
    all_feat["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")
)
all_feat["LapTime_Delta_abs"] = all_feat["LapTime_Delta"].abs()

cat_cols = [c for c in all_feat.columns if all_feat[c].dtype == "object"]
enc = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
)
if cat_cols:
    all_feat[cat_cols] = enc.fit_transform(all_feat[cat_cols]).astype("float32")

X_all = all_feat.astype("float32")
X = X_all.iloc[: len(train)].reset_index(drop=True)
X_test = X_all.iloc[len(train) :].reset_index(drop=True)

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAS_LGBM = False

# Adversarial train-vs-test model.
adv_y = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
adv_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=801)
adv_oof = np.zeros(len(X_all), dtype="float32")

for fold, (tr_idx, va_idx) in enumerate(adv_cv.split(X_all, adv_y), 1):
    if HAS_LGBM:
        adv_model = LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.045,
            num_leaves=31,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=80,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=801 + fold,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        adv_model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=801 + fold,
        )
    adv_model.fit(X_all.iloc[tr_idx], adv_y[tr_idx])
    adv_oof[va_idx] = adv_model.predict_proba(X_all.iloc[va_idx])[:, 1]

adv_auc = roc_auc_score(adv_y, adv_oof)
train_test_likeness = adv_oof[: len(train)]

# Most-test-like validation split from adversarial scores.
adv_threshold = np.quantile(train_test_likeness, 0.80)
adv_val_idx = np.where(train_test_likeness >= adv_threshold)[0]
adv_tr_idx = np.where(train_test_likeness < adv_threshold)[0]

# Smooth sample weights for weighted expert.
w = train_test_likeness.copy()
w = (w - w.min()) / (w.max() - w.min() + 1e-9)
sample_weight = (0.65 + 1.70 * w).astype("float32")

groups = train["Race"].astype(str).values
gkf = GroupKFold(n_splits=5)

candidates = [
    {
        "name": "base",
        "weighted": False,
        "params": dict(
            n_estimators=650, learning_rate=0.035, num_leaves=48, min_child_samples=70
        ),
    },
    {
        "name": "weighted_test_like",
        "weighted": True,
        "params": dict(
            n_estimators=650, learning_rate=0.035, num_leaves=48, min_child_samples=70
        ),
    },
]

results = []
test_pred_pool = []
oof_pool = []

for cand_i, cand in enumerate(candidates):
    oof = np.zeros(len(train), dtype="float32")
    fold_scores = []
    test_fold_preds = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
        if HAS_LGBM:
            model = LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                metric="auc",
                n_estimators=cand["params"]["n_estimators"],
                learning_rate=cand["params"]["learning_rate"],
                num_leaves=cand["params"]["num_leaves"],
                min_child_samples=cand["params"]["min_child_samples"],
                subsample=0.88,
                colsample_bytree=0.88,
                reg_alpha=0.15,
                reg_lambda=1.2,
                random_state=1000 + 31 * cand_i + fold,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            model = HistGradientBoostingClassifier(
                max_iter=350,
                learning_rate=cand["params"]["learning_rate"],
                max_leaf_nodes=cand["params"]["num_leaves"],
                l2_regularization=0.05,
                random_state=1000 + 31 * cand_i + fold,
            )

        fit_kwargs = {}
        if cand["weighted"]:
            fit_kwargs["sample_weight"] = sample_weight[tr_idx]

        model.fit(X.iloc[tr_idx], y[tr_idx], **fit_kwargs)
        val_pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        oof[va_idx] = val_pred
        fold_scores.append(roc_auc_score(y[va_idx], val_pred))
        test_fold_preds.append(model.predict_proba(X_test)[:, 1])

    race_cv_auc = roc_auc_score(y, oof)

    if HAS_LGBM:
        adv_model = LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            metric="auc",
            n_estimators=cand["params"]["n_estimators"],
            learning_rate=cand["params"]["learning_rate"],
            num_leaves=cand["params"]["num_leaves"],
            min_child_samples=cand["params"]["min_child_samples"],
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.15,
            reg_lambda=1.2,
            random_state=2000 + cand_i,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        adv_model = HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=cand["params"]["learning_rate"],
            max_leaf_nodes=cand["params"]["num_leaves"],
            l2_regularization=0.05,
            random_state=2000 + cand_i,
        )

    fit_kwargs = {}
    if cand["weighted"]:
        fit_kwargs["sample_weight"] = sample_weight[adv_tr_idx]
    adv_model.fit(X.iloc[adv_tr_idx], y[adv_tr_idx], **fit_kwargs)
    adv_holdout_pred = adv_model.predict_proba(X.iloc[adv_val_idx])[:, 1]
    adv_holdout_auc = roc_auc_score(y[adv_val_idx], adv_holdout_pred)

    stability_gap = abs(race_cv_auc - adv_holdout_auc)
    results.append(
        {
            "name": cand["name"],
            "race_cv_auc": float(race_cv_auc),
            "adv_holdout_auc": float(adv_holdout_auc),
            "fold_auc_std": float(np.std(fold_scores)),
            "stability_gap": float(stability_gap),
        }
    )
    oof_pool.append(oof)
    test_pred_pool.append(np.mean(test_fold_preds, axis=0))

best_race = max(r["race_cv_auc"] for r in results)
kept = [
    i
    for i, r in enumerate(results)
    if r["race_cv_auc"] >= best_race - 0.004 and r["stability_gap"] <= 0.035
]
if not kept:
    kept = [
        int(np.argmax([r["race_cv_auc"] - 0.25 * r["stability_gap"] for r in results]))
    ]

final_oof = np.mean([oof_pool[i] for i in kept], axis=0)
final_test_pred = np.mean([test_pred_pool[i] for i in kept], axis=0)
final_auc = roc_auc_score(y, final_oof)

submission = sample.copy()
submission[target_col] = np.clip(final_test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": np.clip(final_oof, 0, 1),
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        id_col: test_ids,
        target_col: np.clip(final_test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

review = {
    "metric": "roc_auc",
    "race_held_out_oof_auc": float(final_auc),
    "adversarial_train_vs_test_auc": float(adv_auc),
    "candidate_results": results,
    "kept_models": [results[i]["name"] for i in kept],
    "research_hypotheses_llm_claimed_used": ["000801"],
    "files_written": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
    ],
}

print(f"race_held_out_oof_auc={final_auc:.6f}")
print(f"adversarial_train_vs_test_auc={adv_auc:.6f}")
print(json.dumps(review, indent=2))
