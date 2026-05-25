import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
TARGET = "PitNextLap"
ID_COL = "id"
SEED = 42
N_SPLITS = 5

os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
WET_COMPOUNDS = ["INTERMEDIATE", "WET"]


def add_recurrent_hazard_features(df):
    out = df.copy()
    group_cols = ["Year", "Race", "Driver"]
    sort_cols = group_cols + ["LapNumber", ID_COL]
    tmp = out.reset_index(names="_row").sort_values(sort_cols).copy()

    g = tmp.groupby(group_cols, sort=False)
    tmp["PriorPitCount"] = (
        (g["PitStop"].cumsum() - tmp["PitStop"]).clip(0, 8).astype("int16")
    )

    for comp in DRY_COMPOUNDS + WET_COMPOUNDS:
        cur = (tmp["Compound"].astype(str) == comp).astype("int8")
        tmp[f"Seen_{comp}"] = (
            cur.groupby([tmp[c] for c in group_cols], sort=False)
            .cumsum()
            .gt(0)
            .astype("int8")
        )

    tmp["DryCompoundSeenCount"] = (
        tmp[[f"Seen_{c}" for c in DRY_COMPOUNDS]].sum(axis=1).astype("int8")
    )
    tmp["WetCompoundSeen"] = (
        tmp[[f"Seen_{c}" for c in WET_COMPOUNDS]].max(axis=1).astype("int8")
    )

    dry_rule_active = (
        tmp["WetCompoundSeen"].eq(0) & tmp["DryCompoundSeenCount"].le(1)
    ).astype("float32")
    tmp["RuleDebt"] = (
        np.maximum(tmp["RaceProgress"].astype("float32") - 0.55, 0.0) * dry_rule_active
    ).astype("float32")
    tmp["RuleDebtLate"] = (
        np.maximum(tmp["RaceProgress"].astype("float32") - 0.78, 0.0) * dry_rule_active
    ).astype("float32")

    tmp["TyreLife_bin"] = np.clip(
        np.floor(tmp["TyreLife"].astype(float) / 3), 0, 30
    ).astype("int16")
    tmp["LapNumber_bin"] = np.clip(
        np.floor(tmp["LapNumber"].astype(float) / 5), 0, 20
    ).astype("int16")
    tmp["RaceProgress_bin"] = np.clip(
        np.floor(tmp["RaceProgress"].astype(float) * 20), 0, 20
    ).astype("int16")
    tmp["StintPrior_bin"] = (
        tmp["Stint"].astype(int).clip(1, 8).astype(str)
        + "_"
        + tmp["PriorPitCount"].astype(int).clip(0, 8).astype(str)
    )

    tmp["FinalWindow"] = np.maximum(
        tmp["RaceProgress"].astype("float32") - 0.80, 0.0
    ).astype("float32")
    tmp["EarlyStintCooldown"] = (tmp["TyreLife"].astype(float) <= 2).astype("int8")
    tmp["DegradationPerTyreLap"] = (
        tmp["Cumulative_Degradation"].astype("float32")
        / np.maximum(tmp["TyreLife"].astype("float32"), 1.0)
    ).astype("float32")
    tmp["AbsLapTimeDelta"] = np.abs(tmp["LapTime_Delta"].astype("float32")).astype(
        "float32"
    )

    tmp = tmp.sort_values("_row").drop(columns=["_row"])
    for c in tmp.select_dtypes(include=["float64"]).columns:
        tmp[c] = tmp[c].astype("float32")
    return tmp


train_fe = add_recurrent_hazard_features(train)
test_fe = add_recurrent_hazard_features(test)

cat_cols = [
    "Compound",
    "Driver",
    "Race",
    "Year",
    "Stint",
    "PriorPitCount",
    "TyreLife_bin",
    "LapNumber_bin",
    "RaceProgress_bin",
    "StintPrior_bin",
]
for c in cat_cols:
    if c in train_fe.columns:
        cats = pd.Index(
            pd.concat(
                [train_fe[c].astype(str), test_fe[c].astype(str)], ignore_index=True
            ).unique()
        )
        train_fe[c] = pd.Categorical(train_fe[c].astype(str), categories=cats)
        test_fe[c] = pd.Categorical(test_fe[c].astype(str), categories=cats)

feature_cols = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
hazard_cols = [
    "Compound",
    "Race",
    "Driver",
    "Year",
    "Stint",
    "PriorPitCount",
    "StintPrior_bin",
    "TyreLife",
    "TyreLife_bin",
    "LapNumber",
    "LapNumber_bin",
    "RaceProgress",
    "RaceProgress_bin",
    "PitStop",
    "Position",
    "Position_Change",
    "LapTime_Delta",
    "AbsLapTimeDelta",
    "Cumulative_Degradation",
    "DegradationPerTyreLap",
    "RuleDebt",
    "RuleDebtLate",
    "FinalWindow",
    "EarlyStintCooldown",
    "DryCompoundSeenCount",
    "WetCompoundSeen",
]
hazard_cols = [c for c in hazard_cols if c in train_fe.columns]
hazard_cat_cols = [c for c in hazard_cols if c in cat_cols]

y = train_fe[TARGET].astype(int).values
groups = train_fe["Year"].astype(str) + "_" + train_fe["Race"].astype(str)

try:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(train_fe, y, groups))
except Exception:
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(train_fe, y, groups))

import lightgbm as lgb


def fit_hazard_model(x_tr, y_tr, x_va=None, y_va=None, n_estimators=2500):
    pos = max(float(np.sum(y_tr)), 1.0)
    neg = max(float(len(y_tr) - np.sum(y_tr)), 1.0)
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.025,
        n_estimators=n_estimators,
        num_leaves=15,
        max_depth=4,
        min_child_samples=180,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        scale_pos_weight=min(neg / pos, 60.0),
        random_state=SEED,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
        force_col_wise=True,
    )
    if x_va is not None:
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            eval_metric="auc",
            categorical_feature=[c for c in hazard_cat_cols if c in x_tr.columns],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(
            x_tr,
            y_tr,
            categorical_feature=[c for c in hazard_cat_cols if c in x_tr.columns],
        )
    return model


def positive_proba(pred):
    if isinstance(pred, pd.DataFrame):
        for col in [1, 1.0, "1", "1.0", True]:
            if col in pred.columns:
                return pred[col].to_numpy(dtype=float)
        return pred.iloc[:, -1].to_numpy(dtype=float)
    arr = np.asarray(pred)
    return arr[:, -1] if arr.ndim == 2 else arr.astype(float)


try:
    from autogluon.tabular import TabularPredictor

    HAS_AG = True
except Exception:
    HAS_AG = False

AG_HYPERPARAMS = {
    "GBM": [{"extra_trees": False, "ag_args": {"name_suffix": "MainGBM"}}]
}
AG_FOLD_TIME = int(os.environ.get("AG_FOLD_TIME", "120"))
AG_FULL_TIME = int(os.environ.get("AG_FULL_TIME", "240"))


def fit_ag_predictor(train_df, valid_df, path, time_limit):
    shutil.rmtree(path, ignore_errors=True)
    predictor = TabularPredictor(
        label=TARGET, eval_metric="roc_auc", path=path, verbosity=0
    )
    fit_kwargs = dict(
        train_data=train_df[feature_cols + [TARGET]],
        hyperparameters=AG_HYPERPARAMS,
        num_bag_folds=0,
        num_stack_levels=0,
        time_limit=time_limit,
        verbosity=0,
    )
    if valid_df is not None:
        fit_kwargs["tuning_data"] = valid_df[feature_cols + [TARGET]]
    predictor.fit(**fit_kwargs)
    return predictor


def fit_main_fallback(x_tr, y_tr, x_va=None, y_va=None):
    pos = max(float(np.sum(y_tr)), 1.0)
    neg = max(float(len(y_tr) - np.sum(y_tr)), 1.0)
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.035,
        n_estimators=2000,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=min(neg / pos, 60.0),
        random_state=SEED + 17,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
        force_col_wise=True,
    )
    cat_in = [c for c in cat_cols if c in x_tr.columns]
    if x_va is not None:
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_va, y_va)],
            eval_metric="auc",
            categorical_feature=cat_in,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(x_tr, y_tr, categorical_feature=cat_in)
    return model


haz_oof = np.zeros(len(train_fe), dtype=float)
ag_oof = np.zeros(len(train_fe), dtype=float)
haz_best_iters = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    print(f"Fold {fold}/{N_SPLITS}")

    xh_tr = train_fe.iloc[tr_idx][hazard_cols]
    xh_va = train_fe.iloc[va_idx][hazard_cols]
    y_tr, y_va = y[tr_idx], y[va_idx]

    haz_model = fit_hazard_model(xh_tr, y_tr, xh_va, y_va)
    haz_oof[va_idx] = haz_model.predict_proba(xh_va)[:, 1]
    haz_best_iters.append(
        int(getattr(haz_model, "best_iteration_", 0) or haz_model.n_estimators)
    )

    if HAS_AG:
        ag_path = os.path.join(WORK_DIR, f"ag_fold_{fold}")
        predictor = fit_ag_predictor(
            train_fe.iloc[tr_idx], train_fe.iloc[va_idx], ag_path, AG_FOLD_TIME
        )
        ag_oof[va_idx] = positive_proba(
            predictor.predict_proba(train_fe.iloc[va_idx][feature_cols])
        )
    else:
        main_model = fit_main_fallback(
            train_fe.iloc[tr_idx][feature_cols],
            y_tr,
            train_fe.iloc[va_idx][feature_cols],
            y_va,
        )
        ag_oof[va_idx] = main_model.predict_proba(train_fe.iloc[va_idx][feature_cols])[
            :, 1
        ]

haz_auc = roc_auc_score(y, haz_oof)
ag_auc = roc_auc_score(y, ag_oof)

weights = np.linspace(0.0, 1.0, 101)
blend_scores = [roc_auc_score(y, w * ag_oof + (1.0 - w) * haz_oof) for w in weights]
best_i = int(np.argmax(blend_scores))
best_w = float(weights[best_i])
blend_oof = best_w * ag_oof + (1.0 - best_w) * haz_oof
blend_auc = float(blend_scores[best_i])

print(f"Hazard OOF ROC AUC: {haz_auc:.6f}")
print(f"AutoGluon OOF ROC AUC: {ag_auc:.6f}")
print(f"Best blend weight on AutoGluon: {best_w:.2f}")
print(f"Blend OOF ROC AUC: {blend_auc:.6f}")

oof = pd.DataFrame(
    {"row": np.arange(len(train_fe)), "target": y, "prediction": blend_oof}
)
oof.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

full_haz_iters = int(max(200, np.median(haz_best_iters) * 1.10))
full_hazard = fit_hazard_model(train_fe[hazard_cols], y, n_estimators=full_haz_iters)
haz_test = full_hazard.predict_proba(test_fe[hazard_cols])[:, 1]

if HAS_AG:
    full_ag_path = os.path.join(WORK_DIR, "ag_full")
    full_ag = fit_ag_predictor(train_fe, None, full_ag_path, AG_FULL_TIME)
    ag_test = positive_proba(full_ag.predict_proba(test_fe[feature_cols]))
else:
    full_main = fit_main_fallback(train_fe[feature_cols], y)
    ag_test = full_main.predict_proba(test_fe[feature_cols])[:, 1]

test_pred = np.clip(best_w * ag_test + (1.0 - best_w) * haz_test, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv": "5-fold grouped by Year_Race",
    "hazard_oof_auc": float(haz_auc),
    "autogluon_oof_auc": float(ag_auc),
    "blend_oof_auc": float(blend_auc),
    "blend_weight_autogluon": best_w,
    "research_hypotheses_llm_claimed_used": ["000508"],
}
with open(os.path.join(WORK_DIR, "result_review.json"), "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
