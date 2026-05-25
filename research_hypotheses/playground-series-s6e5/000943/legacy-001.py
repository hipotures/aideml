import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

SEED = 2026
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def add_bins(df):
    df = df.copy()
    df["TyreLifeBin"] = np.clip(
        np.floor(df["TyreLife"].astype(float) / 3), 0, 30
    ).astype(np.int16)
    df["ProgressBin"] = np.clip(
        np.floor(df["RaceProgress"].astype(float) * 20), 0, 19
    ).astype(np.int16)
    df["LapNumberBin"] = np.clip(
        np.floor(df["LapNumber"].astype(float) / 3), 0, 30
    ).astype(np.int16)
    return df


train = add_bins(train)
test = add_bins(test)

cat_cols = [c for c in ["Race", "Driver", "Compound"] if c in train.columns]
for c in cat_cols:
    cats = pd.Index(
        pd.concat(
            [train[c].astype(str), test[c].astype(str)], ignore_index=True
        ).unique()
    )
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

y = train[TARGET].astype(float).to_numpy()
groups = (train["Year"].astype(str) + "_" + train["Race"].astype(str)).to_numpy()

HAZARD_SPECS = [
    (
        "haz_race_compound_stint_tlife",
        ["Race", "Compound", "Stint", "TyreLifeBin"],
        30.0,
    ),
    ("haz_year_race_progress", ["Year", "Race", "ProgressBin"], 70.0),
    ("haz_race_compound_progress", ["Race", "Compound", "ProgressBin"], 45.0),
    ("haz_compound_stint_tlife", ["Compound", "Stint", "TyreLifeBin"], 25.0),
]


def compound_prior(frame, target, global_mean, smoothing=100.0):
    tmp = frame[["Compound"]].copy()
    tmp["_target"] = target
    stats = tmp.groupby("Compound", observed=True)["_target"].agg(["sum", "count"])
    return (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)


def fit_one_encoder(frame, target, name, keys, smoothing):
    global_mean = float(np.mean(target))
    cp = compound_prior(frame, target, global_mean) if "Compound" in keys else None

    tmp = frame[keys].copy()
    tmp["_target"] = target
    stats = (
        tmp.groupby(keys, observed=True)["_target"].agg(["sum", "count"]).reset_index()
    )

    if cp is not None:
        prior = (
            pd.to_numeric(stats["Compound"].map(cp), errors="coerce")
            .fillna(global_mean)
            .astype(float)
        )
    else:
        prior = global_mean

    stats[name] = (stats["sum"].astype(float) + smoothing * prior) / (
        stats["count"].astype(float) + smoothing
    )
    return {
        "name": name,
        "keys": keys,
        "stats": stats[keys + [name]],
        "global_mean": global_mean,
        "compound_prior": cp,
    }


def fit_hazard_encoders(frame, target):
    return [fit_one_encoder(frame, target, *spec) for spec in HAZARD_SPECS]


def transform_one_encoder(frame, enc):
    name, keys = enc["name"], enc["keys"]
    merged = frame[keys].merge(enc["stats"], on=keys, how="left", sort=False)[name]

    if enc["compound_prior"] is not None:
        fallback = pd.to_numeric(
            frame["Compound"].map(enc["compound_prior"]), errors="coerce"
        )
        fallback = fallback.fillna(enc["global_mean"]).astype(float)
    else:
        fallback = enc["global_mean"]

    return merged.fillna(fallback).astype(np.float32).to_numpy()


def transform_hazard_encoders(frame, encoders):
    out = pd.DataFrame(index=frame.index)
    for enc in encoders:
        out[enc["name"]] = transform_one_encoder(frame, enc)
    return out


def make_oof_hazards(frame, target, group_values, n_splits=4):
    n_unique = len(np.unique(group_values))
    n_splits = min(n_splits, n_unique)
    out = pd.DataFrame(
        index=frame.index, columns=[s[0] for s in HAZARD_SPECS], dtype=np.float32
    )

    splitter = GroupKFold(n_splits=n_splits)
    for tr_idx, va_idx in splitter.split(frame, target, group_values):
        encoders = fit_hazard_encoders(frame.iloc[tr_idx], target[tr_idx])
        out.iloc[va_idx, :] = transform_hazard_encoders(
            frame.iloc[va_idx], encoders
        ).to_numpy()
    return out


base_features = [c for c in train.columns if c not in [ID_COL, TARGET]]
hazard_features = [s[0] for s in HAZARD_SPECS]
feature_cols = base_features + hazard_features


def build_matrix(frame, hazards):
    x = frame[base_features].copy()
    for c in hazard_features:
        x[c] = hazards[c].astype(np.float32).to_numpy()
    return x


lgb_params = dict(
    objective="binary",
    metric="auc",
    n_estimators=900,
    learning_rate=0.035,
    num_leaves=63,
    min_child_samples=60,
    subsample=0.90,
    colsample_bytree=0.90,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=min(8, os.cpu_count() or 1),
    verbosity=-1,
)

outer = GroupKFold(n_splits=5)
oof_pred = np.zeros(len(train), dtype=np.float32)
oof_hazards = pd.DataFrame(index=train.index, columns=hazard_features, dtype=np.float32)
fold_aucs, best_iters = [], []

for fold, (tr_idx, va_idx) in enumerate(outer.split(train, y, groups), 1):
    tr_frame, va_frame = train.iloc[tr_idx], train.iloc[va_idx]
    tr_y, va_y = y[tr_idx], y[va_idx]

    tr_haz = make_oof_hazards(tr_frame, tr_y, groups[tr_idx], n_splits=4)
    va_encoders = fit_hazard_encoders(tr_frame, tr_y)
    va_haz = transform_hazard_encoders(va_frame, va_encoders)
    oof_hazards.iloc[va_idx, :] = va_haz.to_numpy()

    x_tr = build_matrix(tr_frame, tr_haz)
    x_va = build_matrix(va_frame, va_haz)

    model = LGBMClassifier(**{**lgb_params, "random_state": SEED + fold})
    model.fit(
        x_tr,
        tr_y,
        eval_set=[(x_va, va_y)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(70, verbose=False), log_evaluation(0)],
    )

    pred = model.predict_proba(x_va, num_iteration=model.best_iteration_)[:, 1]
    oof_pred[va_idx] = pred.astype(np.float32)
    auc = roc_auc_score(va_y, pred)
    fold_aucs.append(float(auc))
    best_iters.append(int(model.best_iteration_ or lgb_params["n_estimators"]))
    print(f"Fold {fold} ROC AUC: {auc:.6f}")

cv_auc = roc_auc_score(y, oof_pred)
print(f"OOF ROC AUC: {cv_auc:.6f}")

oof_df = pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

full_encoders = fit_hazard_encoders(train, y)
test_hazards = transform_hazard_encoders(test, full_encoders)

x_full = build_matrix(train, oof_hazards)
x_test = build_matrix(test, test_hazards)

final_estimators = int(
    np.clip(np.mean(best_iters) * 1.05, 100, lgb_params["n_estimators"])
)
final_model = LGBMClassifier(
    **{**lgb_params, "n_estimators": final_estimators, "random_state": SEED}
)
final_model.fit(x_full, y, categorical_feature=cat_cols)

test_pred = final_model.predict_proba(x_test)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv": "5-fold GroupKFold by Year/Race",
    "oof_roc_auc": float(cv_auc),
    "fold_aucs": fold_aucs,
    "research_hypotheses_llm_claimed_used": ["000943"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print("RESULT_JSON:", json.dumps(result, sort_keys=True))
