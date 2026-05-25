import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

SEED = 20260524
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)


def add_features(df):
    out = df.copy()
    wet = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    out["WetFlag"] = wet
    out["RaceRemaining"] = 1.0 - out["RaceProgress"]
    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["TyreLife_x_Remaining"] = out["TyreLife"] * out["RaceRemaining"]
    out["DegradationPerTyreLap"] = out["Cumulative_Degradation"] / (
        out["TyreLife"] + 1.0
    )
    out["AbsLapDelta"] = out["LapTime_Delta"].abs()
    out["Position_x_Progress"] = out["Position"] * out["RaceProgress"]
    out["LateRaceOldTyre"] = out["TyreLife"] * (out["RaceProgress"] > 0.66).astype(
        np.int8
    )
    out["RacePhase"] = pd.cut(
        out["RaceProgress"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["early", "mid", "late"],
    ).astype(str)
    out["WetPhase"] = np.where(wet == 1, "wet_", "dry_") + out["RacePhase"].astype(str)
    return out


def prepare_categories(train, test, cat_cols):
    for c in cat_cols:
        cats = pd.Index(pd.concat([train[c], test[c]], axis=0).astype(str).unique())
        train[c] = pd.Categorical(train[c].astype(str), categories=cats)
        test[c] = pd.Categorical(test[c].astype(str), categories=cats)
    return train, test


def make_group_folds(train, y):
    groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
    return list(GroupKFold(n_splits=5).split(train, y, groups))


def make_time_folds(train):
    order = train.sort_values(["Year", "id"]).index.to_numpy()
    gap = min(1000, max(0, len(train) // 400))
    folds = []
    for tr_pos, va_pos in TimeSeriesSplit(n_splits=5, gap=gap).split(order):
        folds.append((order[tr_pos], order[va_pos]))
    return folds


def build_model(seed):
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.055,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def run_cv(train, y, test, features, cat_features, folds, make_test=False):
    oof = np.full(len(train), np.nan, dtype=np.float32)
    fold_aucs = []
    test_pred = np.zeros(len(test), dtype=np.float64) if make_test else None

    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = build_model(SEED + fold)
        X_tr = train.iloc[tr_idx][features]
        X_va = train.iloc[va_idx][features]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_features,
            callbacks=[
                lgb.early_stopping(35, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        pred = model.predict_proba(X_va)[:, 1]
        oof[va_idx] = pred
        fold_aucs.append(roc_auc_score(y_va, pred))

        if make_test:
            test_pred += model.predict_proba(test[features])[:, 1] / len(folds)

    return oof, np.array(fold_aucs), test_pred


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_regime_calibrators(scores, y, regimes):
    global_cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    global_cal.fit(logit(scores), y)

    calibrators = {}
    for r in pd.Series(regimes).unique():
        idx = np.asarray(regimes == r)
        pos = y[idx].sum()
        neg = idx.sum() - pos
        if idx.sum() >= 500 and pos >= 10 and neg >= 10:
            cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            cal.fit(logit(scores[idx]), y[idx])
            calibrators[r] = cal
    return global_cal, calibrators


def apply_regime_calibrators(scores, regimes, global_cal, calibrators):
    out = np.zeros(len(scores), dtype=np.float64)
    regimes = np.asarray(regimes)
    for r in pd.Series(regimes).unique():
        idx = regimes == r
        cal = calibrators.get(r, global_cal)
        out[idx] = cal.predict_proba(logit(np.asarray(scores)[idx]))[:, 1]
    return out


def leak_free_calibrated_oof(raw_oof, y, regimes):
    calibrated = np.zeros(len(raw_oof), dtype=np.float64)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    for tr_idx, va_idx in skf.split(raw_oof.reshape(-1, 1), y):
        global_cal, calibrators = fit_regime_calibrators(
            raw_oof[tr_idx], y[tr_idx], regimes[tr_idx]
        )
        calibrated[va_idx] = apply_regime_calibrators(
            raw_oof[va_idx], regimes[va_idx], global_cal, calibrators
        )

    final_global, final_cals = fit_regime_calibrators(raw_oof, y, regimes)
    return calibrated, final_global, final_cals


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train["PitNextLap"].astype(int).to_numpy()
train = train.drop(columns=["PitNextLap"])

train = add_features(train)
test = add_features(test)

base_num = [
    "Year",
    "LapNumber",
    "LapTime (s)",
    "LapTime_Delta",
    "PitStop",
    "Position",
    "Position_Change",
    "RaceProgress",
    "Stint",
    "TyreLife",
    "Cumulative_Degradation",
]
base_cat = ["Race", "Driver", "Compound"]

extra_num = [
    "WetFlag",
    "RaceRemaining",
    "TyreLife_x_Progress",
    "TyreLife_x_Remaining",
    "DegradationPerTyreLap",
    "AbsLapDelta",
    "Position_x_Progress",
    "LateRaceOldTyre",
]
extra_cat = ["RacePhase", "WetPhase"]

all_cat = base_cat + extra_cat
train, test = prepare_categories(train, test, all_cat)

candidates = [
    {
        "name": "base",
        "features": base_num + base_cat,
        "cat": base_cat,
    },
    {
        "name": "regime_state_features",
        "features": base_num + base_cat + extra_num + extra_cat,
        "cat": base_cat + extra_cat,
    },
]

group_folds = make_group_folds(train, y)
time_folds = make_time_folds(train)

results = []
for cand in candidates:
    _, g_auc, _ = run_cv(
        train, y, test, cand["features"], cand["cat"], group_folds, make_test=False
    )
    _, t_auc, _ = run_cv(
        train, y, test, cand["features"], cand["cat"], time_folds, make_test=False
    )
    results.append(
        {
            "name": cand["name"],
            "group_mean": float(g_auc.mean()),
            "group_std": float(g_auc.std()),
            "time_mean": float(t_auc.mean()),
            "time_std": float(t_auc.std()),
            "score": float(
                0.5 * (g_auc.mean() + t_auc.mean()) - 0.25 * (g_auc.std() + t_auc.std())
            ),
        }
    )

baseline = results[0]
accepted_names = [baseline["name"]]
for r in results[1:]:
    improves_both = (r["group_mean"] > baseline["group_mean"]) and (
        r["time_mean"] > baseline["time_mean"]
    )
    low_variance = (
        r["group_std"] <= baseline["group_std"] + 0.002
        and r["time_std"] <= baseline["time_std"] + 0.002
    )
    if improves_both and low_variance:
        accepted_names.append(r["name"])

selected_result = max(
    [r for r in results if r["name"] in accepted_names], key=lambda x: x["score"]
)
selected = next(c for c in candidates if c["name"] == selected_result["name"])

raw_oof, final_fold_aucs, raw_test = run_cv(
    train, y, test, selected["features"], selected["cat"], group_folds, make_test=True
)

train_regime = train["WetPhase"].astype(str).to_numpy()
test_regime = test["WetPhase"].astype(str).to_numpy()

cal_oof, global_cal, regime_cals = leak_free_calibrated_oof(raw_oof, y, train_regime)
cal_test = apply_regime_calibrators(raw_test, test_regime, global_cal, regime_cals)

final_oof = np.clip(0.5 * raw_oof + 0.5 * cal_oof, 0, 1)
final_test = np.clip(0.5 * raw_test + 0.5 * cal_test, 0, 1)

raw_auc = roc_auc_score(y, raw_oof)
cal_auc = roc_auc_score(y, cal_oof)
final_auc = roc_auc_score(y, final_oof)

submission = sample.copy()
submission["PitNextLap"] = final_test
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": final_oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

report = {
    "research_hypotheses_llm_claimed_used": ["000858"],
    "metric": "roc_auc",
    "selected_candidate": selected["name"],
    "accepted_candidates": accepted_names,
    "candidate_dual_axis_results": results,
    "group_cv_raw_fold_auc": [float(x) for x in final_fold_aucs],
    "group_cv_raw_mean_auc": float(final_fold_aucs.mean()),
    "raw_oof_auc": float(raw_auc),
    "regime_sigmoid_calibrated_oof_auc": float(cal_auc),
    "final_raw_calibrated_blend_oof_auc": float(final_auc),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
    "oof_predictions_path": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    "test_predictions_path": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
}

print(f"Validation ROC AUC: {final_auc:.6f}")
print(json.dumps(report, indent=2))
