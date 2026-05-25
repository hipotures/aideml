import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
y = train[target_col].astype(int).to_numpy()


def add_features(df):
    out = df.copy()
    out["Race_Year"] = out["Race"].astype(str) + "_" + out["Year"].astype(str)
    tyre = out["TyreLife"].clip(lower=1)
    lap = out["LapNumber"].clip(lower=1)
    out["Deg_per_TyreLife"] = out["Cumulative_Degradation"] / tyre
    out["TyreLife_frac_race"] = out["TyreLife"] / lap
    out["Lap_to_Tyre_ratio"] = out["LapNumber"] / tyre
    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["Abs_LapTime_Delta"] = out["LapTime_Delta"].abs()
    out["WetCompound"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    out["FreshTyre"] = (out["TyreLife"] <= 2).astype(np.int8)
    out["LateRace"] = (out["RaceProgress"] >= 0.75).astype(np.int8)
    return out


train_fe = add_features(train.drop(columns=[target_col]))
test_fe = add_features(test)

cat_cols_all = [
    c
    for c in train_fe.columns
    if train_fe[c].dtype == "object" or str(train_fe[c].dtype) == "category"
]
for c in cat_cols_all:
    combined = pd.concat(
        [train_fe[c].astype(str), test_fe[c].astype(str)],
        ignore_index=True,
    )
    dtype = pd.CategoricalDtype(categories=pd.unique(combined))
    train_fe[c] = train_fe[c].astype(str).astype(dtype)
    test_fe[c] = test_fe[c].astype(str).astype(dtype)

groups = train_fe["Race_Year"].astype(str).to_numpy()
group_splits = list(GroupKFold(n_splits=N_SPLITS).split(train_fe, y, groups))

chrono_order = np.lexsort((train_fe[id_col].to_numpy(), train_fe["Year"].to_numpy()))
gap = max(1000, len(train_fe) // 150)
tss = TimeSeriesSplit(n_splits=N_SPLITS, gap=gap)
time_splits = [
    (chrono_order[tr], chrono_order[va]) for tr, va in tss.split(chrono_order)
]

pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))
threads = max(1, min(8, os.cpu_count() or 1))

base_params = dict(
    objective="binary",
    boosting_type="gbdt",
    n_estimators=650,
    learning_rate=0.045,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    min_child_samples=150,
    reg_alpha=0.05,
    reg_lambda=1.5,
    scale_pos_weight=pos_weight,
    random_state=SEED,
    n_jobs=threads,
    verbosity=-1,
    force_col_wise=True,
)

candidate_specs = [
    {
        "name": "lgbm_full",
        "drop": [],
        "params": dict(num_leaves=31, min_child_samples=140, reg_lambda=1.2),
    },
    {
        "name": "lgbm_no_driver",
        "drop": ["Driver"],
        "params": dict(num_leaves=31, min_child_samples=180, reg_lambda=2.0),
    },
    {
        "name": "lgbm_conservative_no_driver_race",
        "drop": ["Driver", "Race", "Race_Year"],
        "params": dict(
            num_leaves=23, min_child_samples=240, reg_lambda=3.0, colsample_bytree=0.95
        ),
    },
]


def feature_list(drop):
    return [c for c in train_fe.columns if c != id_col and c not in set(drop)]


def make_model(params_update, seed_offset=0):
    params = base_params.copy()
    params.update(params_update)
    params["random_state"] = SEED + seed_offset
    return lgb.LGBMClassifier(**params)


def safe_auc(y_true, pred):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, pred)


def cv_scores(features, params_update, splits, label):
    X = train_fe[features]
    cat_cols = [c for c in features if str(X[c].dtype) == "category"]
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        model = make_model(params_update, seed_offset=fold)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[
                lgb.early_stopping(45, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        pred = model.predict_proba(X.iloc[va_idx])[:, 1]
        auc = safe_auc(y[va_idx], pred)
        scores.append(auc)
        print(f"{label} fold {fold}: {auc:.6f}")

    return np.array(scores, dtype=float)


stress_results = []
for spec in candidate_specs:
    features = feature_list(spec["drop"])
    print(f"Stress testing {spec['name']} with {len(features)} features")

    g_scores = cv_scores(
        features, spec["params"], group_splits, f"{spec['name']} GroupKFold"
    )
    t_scores = cv_scores(
        features, spec["params"], time_splits, f"{spec['name']} TimeSeriesGap"
    )

    g_mean, t_mean = np.nanmean(g_scores), np.nanmean(t_scores)
    g_std, t_std = np.nanstd(g_scores, ddof=1), np.nanstd(t_scores, ddof=1)
    max_std = float(max(g_std, t_std))
    stress_score = float(min(g_mean, t_mean) - 0.5 * max_std)

    row = {
        "name": spec["name"],
        "drop": spec["drop"],
        "group_auc_mean": float(g_mean),
        "group_auc_std": float(g_std),
        "time_auc_mean": float(t_mean),
        "time_auc_std": float(t_std),
        "max_std": max_std,
        "stress_score": stress_score,
    }
    stress_results.append(row)
    print(f"{spec['name']} stress score: {stress_score:.6f}, max std: {max_std:.6f}")

max_stds = np.array([r["max_std"] for r in stress_results], dtype=float)
best_score = max(r["stress_score"] for r in stress_results)
std_cutoff = max(0.025, float(np.nanpercentile(max_stds, 75)))
score_cutoff = best_score - 0.015

survivor_names = {
    r["name"]
    for r in stress_results
    if r["max_std"] <= std_cutoff and r["stress_score"] >= score_cutoff
}
if not survivor_names:
    survivor_names = {max(stress_results, key=lambda r: r["stress_score"])["name"]}

survivors = [s for s in candidate_specs if s["name"] in survivor_names]
print("Stable members kept:", ", ".join(s["name"] for s in survivors))

member_oof = np.zeros((len(train_fe), len(survivors)), dtype=np.float32)
member_test = np.zeros((len(test_fe), len(survivors)), dtype=np.float32)
member_auc = {}

for m, spec in enumerate(survivors):
    features = feature_list(spec["drop"])
    X = train_fe[features]
    X_test = test_fe[features]
    cat_cols = [c for c in features if str(X[c].dtype) == "category"]

    fold_test_preds = []
    print(f"Training final grouped OOF member {spec['name']}")

    for fold, (tr_idx, va_idx) in enumerate(group_splits, 1):
        model = make_model(spec["params"], seed_offset=100 + 10 * m + fold)
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[
                lgb.early_stopping(45, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        member_oof[va_idx, m] = model.predict_proba(X.iloc[va_idx])[:, 1]
        fold_test_preds.append(model.predict_proba(X_test)[:, 1])

    member_test[:, m] = np.mean(fold_test_preds, axis=0)
    member_auc[spec["name"]] = float(roc_auc_score(y, member_oof[:, m]))
    print(f"{spec['name']} grouped OOF ROC AUC: {member_auc[spec['name']]:.6f}")

avg_oof = member_oof.mean(axis=1)
avg_test = member_test.mean(axis=1)
raw_auc = roc_auc_score(y, avg_oof)

cal_oof = np.zeros(len(y), dtype=np.float32)
cal_splits = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (tr_idx, va_idx) in enumerate(cal_splits.split(avg_oof.reshape(-1, 1), y), 1):
    sigmoid = LogisticRegression(max_iter=1000, solver="lbfgs")
    sigmoid.fit(avg_oof[tr_idx].reshape(-1, 1), y[tr_idx])

    sig_tr = sigmoid.predict_proba(avg_oof[tr_idx].reshape(-1, 1))[:, 1]
    sig_va = sigmoid.predict_proba(avg_oof[va_idx].reshape(-1, 1))[:, 1]

    if len(np.unique(sig_tr)) >= 10:
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        isotonic.fit(sig_tr, y[tr_idx])
        iso_va = isotonic.predict(sig_va)
        cal_oof[va_idx] = 0.5 * sig_va + 0.5 * iso_va
    else:
        cal_oof[va_idx] = sig_va

cal_oof = np.clip(cal_oof, 1e-6, 1 - 1e-6)
cal_auc = roc_auc_score(y, cal_oof)

final_sigmoid = LogisticRegression(max_iter=1000, solver="lbfgs")
final_sigmoid.fit(avg_oof.reshape(-1, 1), y)
sig_oof_all = final_sigmoid.predict_proba(avg_oof.reshape(-1, 1))[:, 1]
sig_test = final_sigmoid.predict_proba(avg_test.reshape(-1, 1))[:, 1]

final_isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
final_isotonic.fit(sig_oof_all, y)
iso_test = final_isotonic.predict(sig_test)

test_pred = np.clip(0.5 * sig_test + 0.5 * iso_test, 1e-6, 1 - 1e-6)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": cal_oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000703"],
    "metric": "roc_auc",
    "final_group_oof_auc_raw_stable_average": float(raw_auc),
    "final_group_oof_auc_calibrated_crossfit": float(cal_auc),
    "stable_members": [s["name"] for s in survivors],
    "member_oof_auc": member_auc,
    "stress_results": stress_results,
    "cv_description": "5-fold GroupKFold by Race_Year plus 5-fold forward TimeSeriesSplit with a chronology gap; final OOF uses GroupKFold and OOF-only sigmoid/isotonic calibration.",
}
with open(os.path.join(WORK_DIR, "review.json"), "w") as f:
    json.dump(review, f, indent=2)

print(f"Raw stable-member 5-fold GroupKFold OOF ROC AUC: {raw_auc:.6f}")
print(f"Calibrated cross-fit OOF ROC AUC: {cal_auc:.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000703"],
            "metric": "roc_auc",
            "cv_auc": float(cal_auc),
            "stable_members": [s["name"] for s in survivors],
        }
    )
)
