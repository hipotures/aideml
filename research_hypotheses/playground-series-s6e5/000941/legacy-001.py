import os
import gc
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier

from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

SEED = 941
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)
np.random.seed(SEED)


def rank01(values):
    r = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    if len(r) <= 1:
        return np.zeros_like(r, dtype=np.float64)
    return (r - 1.0) / (len(r) - 1.0)


def add_features(df):
    df = df.copy()
    df = df.rename(columns={"LapTime (s)": "LapTime_s"})

    for c in ["Driver", "Race", "Compound"]:
        df[c] = df[c].fillna("MISSING").astype(str)

    lap = (
        pd.to_numeric(df["LapNumber"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    progress = (
        pd.to_numeric(df["RaceProgress"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    progress_safe = np.maximum(progress, 1e-4)
    total_laps = np.clip(lap / progress_safe, 1.0, 120.0)
    total_laps = np.maximum(total_laps, lap)
    remaining = np.maximum(total_laps - lap, 0.0)

    tyre = (
        pd.to_numeric(df["TyreLife"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    deg = (
        pd.to_numeric(df["Cumulative_Degradation"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    delta = (
        pd.to_numeric(df["LapTime_Delta"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    laptime = (
        pd.to_numeric(df["LapTime_s"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    stint = (
        pd.to_numeric(df["Stint"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    )
    pos = (
        pd.to_numeric(df["Position"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    pos_chg = (
        pd.to_numeric(df["Position_Change"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    pitstop = (
        pd.to_numeric(df["PitStop"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )

    df["EstimatedRaceLaps"] = total_laps.astype(np.float32)
    df["LapsRemaining"] = remaining.astype(np.float32)
    df["NextLapProgress"] = np.clip(
        (lap + 1.0) / np.maximum(total_laps, 1.0), 0.0, 1.2
    ).astype(np.float32)
    df["TyreLifePctRace"] = (tyre / np.maximum(total_laps, 1.0)).astype(np.float32)
    df["TyreLifePctLap"] = (tyre / np.maximum(lap, 1.0)).astype(np.float32)
    df["OldTyrePressure"] = (tyre / np.maximum(remaining + 1.0, 1.0)).astype(np.float32)
    df["DegradationPerTyreLap"] = (deg / np.maximum(tyre, 1.0)).astype(np.float32)
    df["DegradationPerRaceLap"] = (deg / np.maximum(lap, 1.0)).astype(np.float32)
    df["LapTimeDeltaAbs"] = np.abs(delta).astype(np.float32)
    df["LapTimePerRaceLap"] = (laptime / np.maximum(lap, 1.0)).astype(np.float32)
    df["PitStopXTyreLife"] = (pitstop * tyre).astype(np.float32)
    df["StintXTyreLife"] = (stint * tyre).astype(np.float32)
    df["ProgressXTyreLife"] = (progress * tyre).astype(np.float32)
    df["ProgressXDegradation"] = (progress * deg).astype(np.float32)
    df["PositionAfterChange"] = np.clip(pos + pos_chg, 1.0, 20.0).astype(np.float32)
    df["FrontPack"] = (pos <= 5).astype(np.int8)
    df["BackPack"] = (pos >= 15).astype(np.int8)

    for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        df[f"CompoundIs{comp.title()}"] = (df["Compound"] == comp).astype(np.int8)

    year_s = df["Year"].astype(str)
    stint_s = df["Stint"].astype(str)
    df["Year_Race"] = year_s + "|" + df["Race"]
    df["Driver_Race"] = df["Driver"] + "|" + df["Race"]
    df["Race_Compound"] = df["Race"] + "|" + df["Compound"]
    df["Driver_Compound"] = df["Driver"] + "|" + df["Compound"]
    df["Compound_Stint"] = df["Compound"] + "|" + stint_s
    df["Race_Stint"] = df["Race"] + "|" + stint_s
    df["Driver_Race_Compound"] = df["Driver"] + "|" + df["Race"] + "|" + df["Compound"]

    return df


def make_weight_grid(n_models, units=10):
    out, cur = [], [0] * n_models

    def rec(i, remaining):
        if i == n_models - 1:
            cur[i] = remaining
            out.append(np.array(cur, dtype=np.float64) / units)
            return
        for v in range(remaining + 1):
            cur[i] = v
            rec(i + 1, remaining - v)

    rec(0, units)
    return out


def monotone_constraints(columns):
    positive = {
        "TyreLife",
        "Cumulative_Degradation",
        "TyreLifePctRace",
        "TyreLifePctLap",
        "OldTyrePressure",
        "DegradationPerTyreLap",
        "DegradationPerRaceLap",
        "LapTimeDeltaAbs",
        "StintXTyreLife",
        "ProgressXTyreLife",
        "ProgressXDegradation",
    }
    negative = {"LapsRemaining"}
    return [1 if c in positive else -1 if c in negative else 0 for c in columns]


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target_col = "PitNextLap"
y = train[target_col].astype(int).to_numpy()
n_train = len(train)

all_raw = pd.concat(
    [train.drop(columns=[target_col]), test],
    axis=0,
    ignore_index=True,
    sort=False,
)
all_feat = add_features(all_raw)
feature_cols = [c for c in all_feat.columns if c != "id"]

cat_cols = [
    c
    for c in feature_cols
    if all_feat[c].dtype == "object" or pd.api.types.is_string_dtype(all_feat[c])
]
num_cols = [c for c in feature_cols if c not in cat_cols]

X_cat = all_feat[feature_cols].copy()
for c in cat_cols:
    X_cat[c] = X_cat[c].fillna("MISSING").astype(str)
for c in num_cols:
    X_cat[c] = (
        pd.to_numeric(X_cat[c], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
    )

X_num = pd.DataFrame(index=all_feat.index)
for c in feature_cols:
    if c in cat_cols:
        X_num[c] = pd.factorize(X_cat[c], sort=True)[0].astype(np.int32)
    else:
        X_num[c] = X_cat[c].astype(np.float32)

X_cat_train = X_cat.iloc[:n_train].reset_index(drop=True)
X_cat_test = X_cat.iloc[n_train:].reset_index(drop=True)
X_num_train = X_num.iloc[:n_train].reset_index(drop=True)
X_num_test = X_num.iloc[n_train:].reset_index(drop=True)

cat_feature_indices = [X_num.columns.get_loc(c) for c in cat_cols]
mono = monotone_constraints(feature_cols)
xgb_mono = "(" + ",".join(str(v) for v in mono) + ")"

model_names = ["catboost", "lightgbm", "xgboost", "extratrees"]
oof = {m: np.zeros(n_train, dtype=np.float64) for m in model_names}
test_pred = {m: np.zeros(len(test), dtype=np.float64) for m in model_names}
fold_scores = {m: [] for m in model_names}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
cat_test_pool = Pool(X_cat_test, cat_features=cat_cols)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num_train, y), 1):
    y_tr, y_va = y[tr_idx], y[va_idx]
    scale_pos = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))

    train_pool = Pool(X_cat_train.iloc[tr_idx], y_tr, cat_features=cat_cols)
    valid_pool = Pool(X_cat_train.iloc[va_idx], y_va, cat_features=cat_cols)
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.055,
        depth=6,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_strength=0.8,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        one_hot_max_size=10,
        max_ctr_complexity=2,
        boosting_type="Plain",
        random_seed=SEED + fold,
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
    )
    cat_model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=75,
        verbose=False,
    )
    oof["catboost"][va_idx] = cat_model.predict_proba(valid_pool)[:, 1]
    test_pred["catboost"] += cat_model.predict_proba(cat_test_pool)[:, 1] / N_SPLITS
    fold_scores["catboost"].append(roc_auc_score(y_va, oof["catboost"][va_idx]))

    lgb_model = LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos,
        monotone_constraints=mono,
        monotone_constraints_method="advanced",
        min_data_per_group=30,
        cat_smooth=20.0,
        random_state=SEED + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    lgb_model.fit(
        X_num_train.iloc[tr_idx],
        y_tr,
        eval_set=[(X_num_train.iloc[va_idx], y_va)],
        eval_metric="auc",
        categorical_feature=cat_feature_indices,
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
    )
    oof["lightgbm"][va_idx] = lgb_model.predict_proba(X_num_train.iloc[va_idx])[:, 1]
    test_pred["lightgbm"] += lgb_model.predict_proba(X_num_test)[:, 1] / N_SPLITS
    fold_scores["lightgbm"].append(roc_auc_score(y_va, oof["lightgbm"][va_idx]))

    xgb_params = dict(
        n_estimators=550,
        learning_rate=0.045,
        max_depth=5,
        min_child_weight=30,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=3.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        max_bin=256,
        scale_pos_weight=scale_pos,
        monotone_constraints=xgb_mono,
        random_state=SEED + fold,
        n_jobs=-1,
    )
    xgb_model = XGBClassifier(**xgb_params)
    try:
        xgb_model.fit(
            X_num_train.iloc[tr_idx],
            y_tr,
            eval_set=[(X_num_train.iloc[va_idx], y_va)],
            early_stopping_rounds=75,
            verbose=False,
        )
    except TypeError:
        xgb_model.set_params(early_stopping_rounds=75)
        xgb_model.fit(
            X_num_train.iloc[tr_idx],
            y_tr,
            eval_set=[(X_num_train.iloc[va_idx], y_va)],
            verbose=False,
        )
    oof["xgboost"][va_idx] = xgb_model.predict_proba(X_num_train.iloc[va_idx])[:, 1]
    test_pred["xgboost"] += xgb_model.predict_proba(X_num_test)[:, 1] / N_SPLITS
    fold_scores["xgboost"].append(roc_auc_score(y_va, oof["xgboost"][va_idx]))

    et_model = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=35,
        max_features=0.75,
        class_weight="balanced",
        random_state=SEED + fold,
        n_jobs=-1,
    )
    et_model.fit(X_num_train.iloc[tr_idx], y_tr)
    oof["extratrees"][va_idx] = et_model.predict_proba(X_num_train.iloc[va_idx])[:, 1]
    test_pred["extratrees"] += et_model.predict_proba(X_num_test)[:, 1] / N_SPLITS
    fold_scores["extratrees"].append(roc_auc_score(y_va, oof["extratrees"][va_idx]))

    print(
        f"fold={fold} "
        + " ".join(f"{m}_auc={fold_scores[m][-1]:.6f}" for m in model_names),
        flush=True,
    )
    del train_pool, valid_pool, cat_model, lgb_model, xgb_model, et_model
    gc.collect()

rank_oof = np.column_stack([rank01(oof[m]) for m in model_names])
rank_test = np.column_stack([rank01(test_pred[m]) for m in model_names])
single_auc = {
    m: float(roc_auc_score(y, rank_oof[:, i])) for i, m in enumerate(model_names)
}

candidates = make_weight_grid(len(model_names), units=10)


def score_weight(w):
    return roc_auc_score(y, rank_oof.dot(w))


try:
    from joblib import Parallel, delayed

    workers = min(16, os.cpu_count() or 1)
    print(
        f"Evaluating {len(candidates)} blend candidates with {workers} workers",
        flush=True,
    )
    scores = Parallel(n_jobs=workers, prefer="threads")(
        delayed(score_weight)(w) for w in candidates
    )
except Exception:
    best_single = int(np.argmax([single_auc[m] for m in model_names]))
    candidates = []
    eye = np.eye(len(model_names), dtype=np.float64)
    candidates.extend(list(eye))
    candidates.append(np.ones(len(model_names), dtype=np.float64) / len(model_names))
    for j in range(len(model_names)):
        if j == best_single:
            continue
        for a in np.linspace(0, 1, 21):
            w = np.zeros(len(model_names), dtype=np.float64)
            w[best_single] = a
            w[j] = 1.0 - a
            candidates.append(w)
    scores = [score_weight(w) for w in candidates]

best_idx = int(np.argmax(scores))
best_weights = candidates[best_idx]
final_oof = np.clip(rank_oof.dot(best_weights), 1e-6, 1 - 1e-6)
final_test = np.clip(rank_test.dot(best_weights), 1e-6, 1 - 1e-6)
cv_auc = float(roc_auc_score(y, final_oof))

sub_target = sample.columns[1]
submission = sample.copy()
submission[sub_target] = final_test
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

oof_df = pd.DataFrame(
    {
        "row": np.arange(n_train, dtype=np.int64),
        "target": y,
        "prediction": final_oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": cv_auc,
    "single_model_oof_auc": single_auc,
    "blend_weights": {m: float(w) for m, w in zip(model_names, best_weights)},
    "research_hypotheses_llm_claimed_used": ["000941"],
}
print(json.dumps(result, sort_keys=True), flush=True)
