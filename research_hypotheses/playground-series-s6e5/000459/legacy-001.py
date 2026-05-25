import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostError, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None


INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_features(df):
    out = df.copy()
    for c in ["Driver", "Race", "Compound"]:
        out[c] = out[c].fillna("__MISSING__").astype(str)

    year_s = out["Year"].astype(str)
    out["Race_Year"] = out["Race"] + "_" + year_s
    out["Driver_Year"] = out["Driver"] + "_" + year_s
    out["Driver_Race"] = out["Driver"] + "_" + out["Race"]
    out["Driver_Race_Year"] = out["Driver"] + "_" + out["Race"] + "_" + year_s

    tyre = pd.to_numeric(out["TyreLife"], errors="coerce").clip(lower=1)
    lap = pd.to_numeric(out["LapNumber"], errors="coerce").clip(lower=1)
    progress = pd.to_numeric(out["RaceProgress"], errors="coerce")
    degradation = pd.to_numeric(out["Cumulative_Degradation"], errors="coerce")

    out["TyreLife_to_Lap"] = tyre / lap
    out["Degradation_per_TyreLife"] = degradation / tyre
    out["TyreLife_x_RaceProgress"] = tyre * progress
    out["Is_Wet_Compound"] = (
        out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    )
    return out


train_fe = add_features(train)
test_fe = add_features(test)

feature_cols = [c for c in train_fe.columns if c not in [ID_COL, TARGET]]
X = train_fe[feature_cols].copy()
X_test = test_fe[feature_cols].copy()
y = train_fe[TARGET].astype(int).values

cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
cat_feature_indices = [feature_cols.index(c) for c in cat_cols]

for c in cat_cols:
    X[c] = X[c].fillna("__MISSING__").astype(str)
    X_test[c] = X_test[c].fillna("__MISSING__").astype(str)

num_cols = [c for c in feature_cols if c not in cat_cols]
medians = X[num_cols].median()
X[num_cols] = X[num_cols].fillna(medians).astype(np.float32)
X_test[num_cols] = X_test[num_cols].fillna(medians).astype(np.float32)

sparse_id_cols = {
    "Driver",
    "Race",
    "Race_Year",
    "Driver_Year",
    "Driver_Race",
    "Driver_Race_Year",
}
feature_weights = [0.35 if c in sparse_id_cols else 1.0 for c in feature_cols]
first_use_penalties = [
    (
        2.5
        if c in {"Driver", "Driver_Year", "Driver_Race", "Driver_Race_Year"}
        else 1.25 if c in {"Race", "Race_Year"} else 0.0
    )
    for c in feature_cols
]

base_params = dict(
    iterations=650,
    learning_rate=0.055,
    depth=7,
    l2_leaf_reg=8.0,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    bootstrap_type="Bernoulli",
    subsample=0.8,
    rsm=0.9,
    max_ctr_complexity=1,
    leaf_estimation_iterations=5,
    random_seed=RANDOM_STATE,
    thread_count=max(1, min(16, os.cpu_count() or 4)),
    allow_writing_files=False,
    od_type="Iter",
    od_wait=60,
    verbose=False,
)


def make_splits():
    groups = train_fe["Race_Year"].values
    if StratifiedGroupKFold is not None:
        try:
            sgkf = StratifiedGroupKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
            )
            splits = list(sgkf.split(X, y, groups=groups))
            if all(len(np.unique(y[val_idx])) == 2 for _, val_idx in splits):
                return splits, "StratifiedGroupKFold_by_Race_Year"
        except Exception:
            pass

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(X, y)), "StratifiedKFold"


def fit_model(kind, fold, train_pool, valid_pool):
    params = base_params.copy()
    params["random_seed"] = RANDOM_STATE + fold * 17

    if kind == "normal":
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        return model, "normal"

    attempts = [
        (
            "feature_weights_and_first_feature_use_penalties",
            {
                "feature_weights": feature_weights,
                "first_feature_use_penalties": first_use_penalties,
            },
        ),
        ("feature_weights_only", {"feature_weights": feature_weights}),
    ]

    last_error = None
    for mode, extra_params in attempts:
        try:
            model = CatBoostClassifier(**params, **extra_params)
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            return model, mode
        except CatBoostError as exc:
            last_error = exc

    raise last_error


splits, cv_name = make_splits()
test_pool = Pool(X_test, cat_features=cat_feature_indices)

oof_normal = np.zeros(len(X), dtype=np.float32)
oof_delayed = np.zeros(len(X), dtype=np.float32)
test_normal = np.zeros(len(X_test), dtype=np.float64)
test_delayed = np.zeros(len(X_test), dtype=np.float64)

fold_scores = {
    "normal_catboost": [],
    "id_delayed_catboost": [],
    "fixed_50_50_late_fusion": [],
}
delayed_modes = []

for fold, (tr_idx, val_idx) in enumerate(splits, start=1):
    train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_feature_indices)
    valid_pool = Pool(X.iloc[val_idx], y[val_idx], cat_features=cat_feature_indices)

    normal_model, _ = fit_model("normal", fold, train_pool, valid_pool)
    delayed_model, delayed_mode = fit_model("delayed", fold, train_pool, valid_pool)
    delayed_modes.append(delayed_mode)

    normal_val = normal_model.predict_proba(valid_pool)[:, 1]
    delayed_val = delayed_model.predict_proba(valid_pool)[:, 1]
    blend_val = 0.5 * normal_val + 0.5 * delayed_val

    oof_normal[val_idx] = normal_val
    oof_delayed[val_idx] = delayed_val

    test_normal += normal_model.predict_proba(test_pool)[:, 1] / N_SPLITS
    test_delayed += delayed_model.predict_proba(test_pool)[:, 1] / N_SPLITS

    fold_scores["normal_catboost"].append(roc_auc_score(y[val_idx], normal_val))
    fold_scores["id_delayed_catboost"].append(roc_auc_score(y[val_idx], delayed_val))
    fold_scores["fixed_50_50_late_fusion"].append(roc_auc_score(y[val_idx], blend_val))

    print(
        f"Fold {fold}: "
        f"normal_auc={fold_scores['normal_catboost'][-1]:.6f}, "
        f"id_delayed_auc={fold_scores['id_delayed_catboost'][-1]:.6f}, "
        f"fusion_auc={fold_scores['fixed_50_50_late_fusion'][-1]:.6f}"
    )

oof_blend = 0.5 * oof_normal + 0.5 * oof_delayed
test_blend = 0.5 * test_normal + 0.5 * test_delayed

oof_scores = {
    "normal_catboost": roc_auc_score(y, oof_normal),
    "id_delayed_catboost": roc_auc_score(y, oof_delayed),
    "fixed_50_50_late_fusion": roc_auc_score(y, oof_blend),
}

deploy_candidates = {
    "id_delayed_catboost": oof_scores["id_delayed_catboost"],
    "fixed_50_50_late_fusion": oof_scores["fixed_50_50_late_fusion"],
}
selected_model = max(deploy_candidates, key=deploy_candidates.get)

if selected_model == "id_delayed_catboost":
    final_oof = oof_delayed
    final_test = test_delayed
else:
    final_oof = oof_blend
    final_test = test_blend

final_test = np.clip(final_test, 0.0, 1.0)

pred_col = [c for c in sample.columns if c != ID_COL][0]
submission = sample.copy()
submission[pred_col] = final_test
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train_fe), dtype=np.int64),
        "target": y,
        "prediction": np.clip(final_oof, 0.0, 1.0),
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000459"],
    "metric": "roc_auc",
    "cv_strategy": cv_name,
    "oof_auc": {k: float(v) for k, v in oof_scores.items()},
    "mean_fold_auc": {k: float(np.mean(v)) for k, v in fold_scores.items()},
    "selected_submission_model": selected_model,
    "selected_oof_auc": float(oof_scores[selected_model]),
    "delayed_training_modes": sorted(set(delayed_modes)),
    "files": {
        "submission": os.path.join(WORK_DIR, "submission.csv"),
        "oof_predictions": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
        "test_predictions": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
        "result_review": os.path.join(WORK_DIR, "result_review.json"),
    },
}

with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(f"CV strategy: {cv_name}")
for name, score in oof_scores.items():
    print(f"OOF ROC AUC {name}: {score:.6f}")
print(f"Selected submission model: {selected_model}")
print(f"Selected OOF ROC AUC: {oof_scores[selected_model]:.6f}")
print("RESULT_REVIEW_JSON:", json.dumps(review, sort_keys=True))
