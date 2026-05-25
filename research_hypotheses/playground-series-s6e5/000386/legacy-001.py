import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

import lightgbm as lgb

RANDOM_STATE = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
id_col = "id"
y = train[target_col].astype(int).values


def add_strategy_features(df):
    out = df.copy()
    total_laps_est = out["LapNumber"] / np.clip(out["RaceProgress"], 1e-4, None)
    remaining_laps = np.maximum(total_laps_est - out["LapNumber"], 0)

    compound_rank = {
        "HARD": 1.0,
        "MEDIUM": 2.0,
        "SOFT": 3.0,
        "INTERMEDIATE": 2.5,
        "WET": 2.5,
    }

    out["remaining_laps_est"] = remaining_laps
    out["wear_pressure"] = out["TyreLife"] / (remaining_laps + 1.0)
    out["degradation_per_lap"] = out["Cumulative_Degradation"] / (out["TyreLife"] + 1.0)
    out["stop_debt"] = out["RaceProgress"] / np.sqrt(out["Stint"].clip(lower=1))
    out["first_stint_late"] = out["Stint"].eq(1).astype(float) * out["RaceProgress"]
    out["compound_wear_rank"] = out["Compound"].map(compound_rank).fillna(2.0)
    out["pit_recent_penalty"] = out["PitStop"].astype(float)
    out["lap_time_delta_pos"] = out["LapTime_Delta"].clip(lower=0)
    out["tyre_x_softness"] = out["TyreLife"] * out["compound_wear_rank"]
    out["degradation_x_softness"] = (
        out["Cumulative_Degradation"] * out["compound_wear_rank"]
    )
    return out


train_fe = add_strategy_features(train.drop(columns=[target_col]))
test_fe = add_strategy_features(test)

feature_cols = [c for c in train_fe.columns if c != id_col]
cat_cols = [c for c in feature_cols if train_fe[c].dtype == "object"]

for c in cat_cols:
    cats = pd.Index(pd.concat([train_fe[c], test_fe[c]], axis=0).astype(str).unique())
    train_fe[c] = pd.Categorical(train_fe[c].astype(str), categories=cats)
    test_fe[c] = pd.Categorical(test_fe[c].astype(str), categories=cats)

specialist_cols = [
    "TyreLife",
    "Cumulative_Degradation",
    "RaceProgress",
    "wear_pressure",
    "degradation_per_lap",
    "stop_debt",
    "first_stint_late",
    "compound_wear_rank",
    "pit_recent_penalty",
    "lap_time_delta_pos",
    "tyre_x_softness",
    "degradation_x_softness",
]
monotone_constraints = [1, 1, 0, 1, 1, 1, 1, 1, -1, 1, 1, 1]

groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    splits = list(splitter.split(train_fe, y, groups))
else:
    splitter = GroupKFold(n_splits=5)
    splits = list(splitter.split(train_fe, y, groups))

main_oof = np.zeros(len(train_fe))
spec_oof = np.zeros(len(train_fe))
main_test = np.zeros(len(test_fe))
spec_test = np.zeros(len(test_fe))
fold_rows = []

main_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.035,
    "num_leaves": 96,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 255,
    "verbosity": -1,
    "seed": RANDOM_STATE,
    "num_threads": max(1, os.cpu_count() or 1),
}

spec_params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.04,
    "num_leaves": 48,
    "min_data_in_leaf": 120,
    "feature_fraction": 0.95,
    "bagging_fraction": 0.90,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 4.0,
    "max_bin": 511,
    "monotone_constraints": monotone_constraints,
    "monotone_constraints_method": "advanced",
    "verbosity": -1,
    "seed": RANDOM_STATE + 100,
    "num_threads": max(1, os.cpu_count() or 1),
}

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr = train_fe.iloc[tr_idx][feature_cols]
    X_va = train_fe.iloc[va_idx][feature_cols]
    y_tr, y_va = y[tr_idx], y[va_idx]

    dtrain = lgb.Dataset(
        X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False
    )
    dvalid = lgb.Dataset(
        X_va, label=y_va, categorical_feature=cat_cols, free_raw_data=False
    )

    main_model = lgb.train(
        main_params,
        dtrain,
        num_boost_round=2500,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
    )

    dtrain_s = lgb.Dataset(
        train_fe.iloc[tr_idx][specialist_cols], label=y_tr, free_raw_data=False
    )
    dvalid_s = lgb.Dataset(
        train_fe.iloc[va_idx][specialist_cols], label=y_va, free_raw_data=False
    )

    spec_model = lgb.train(
        spec_params,
        dtrain_s,
        num_boost_round=1800,
        valid_sets=[dvalid_s],
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
    )

    main_oof[va_idx] = main_model.predict(X_va, num_iteration=main_model.best_iteration)
    spec_oof[va_idx] = spec_model.predict(
        train_fe.iloc[va_idx][specialist_cols],
        num_iteration=spec_model.best_iteration,
    )

    main_test += main_model.predict(
        test_fe[feature_cols], num_iteration=main_model.best_iteration
    ) / len(splits)
    spec_test += spec_model.predict(
        test_fe[specialist_cols], num_iteration=spec_model.best_iteration
    ) / len(splits)

    fold_auc_main = roc_auc_score(y_va, main_oof[va_idx])
    fold_auc_spec = roc_auc_score(y_va, spec_oof[va_idx])
    fold_rows.append(
        {"fold": fold, "main_auc": fold_auc_main, "specialist_auc": fold_auc_spec}
    )
    print(
        f"fold {fold}: main_auc={fold_auc_main:.6f} specialist_auc={fold_auc_spec:.6f}"
    )

stack_X = np.column_stack(
    [
        main_oof,
        spec_oof,
        np.log(np.clip(main_oof, 1e-6, 1 - 1e-6) / np.clip(1 - main_oof, 1e-6, 1)),
        np.log(np.clip(spec_oof, 1e-6, 1 - 1e-6) / np.clip(1 - spec_oof, 1e-6, 1)),
    ]
)
stack_test = np.column_stack(
    [
        main_test,
        spec_test,
        np.log(np.clip(main_test, 1e-6, 1 - 1e-6) / np.clip(1 - main_test, 1e-6, 1)),
        np.log(np.clip(spec_test, 1e-6, 1 - 1e-6) / np.clip(1 - spec_test, 1e-6, 1)),
    ]
)

stacker = LogisticRegression(
    C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE
)
stacker.fit(stack_X, y)
oof_pred = stacker.predict_proba(stack_X)[:, 1]
test_pred = stacker.predict_proba(stack_test)[:, 1]

overall_main_auc = roc_auc_score(y, main_oof)
overall_spec_auc = roc_auc_score(y, spec_oof)
overall_stack_auc = roc_auc_score(y, oof_pred)

submission = sample[[id_col]].copy()
submission[target_col] = np.clip(test_pred, 0, 1)
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

pd.DataFrame(
    {
        id_col: sample[id_col].values,
        target_col: np.clip(test_pred, 0, 1),
    }
).to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_auc_main": float(overall_main_auc),
    "cv_auc_monotone_specialist": float(overall_spec_auc),
    "cv_auc_stacked": float(overall_stack_auc),
    "folds": fold_rows,
    "research_hypotheses_llm_claimed_used": ["000386"],
    "files": {
        "submission": os.path.join(WORK_DIR, "submission.csv"),
        "oof_predictions": os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
        "test_predictions": os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    },
}
print(json.dumps(result, indent=2))
