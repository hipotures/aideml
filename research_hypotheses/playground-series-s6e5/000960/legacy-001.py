import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

import lightgbm as lgb
from catboost import CatBoostClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False


RANDOM_STATE = 2026
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
n_train = len(train)
n_test = len(test)
threads = max(1, min(8, os.cpu_count() or 1))

base_train = train.drop(columns=[TARGET])
all_df = pd.concat([base_train, test], axis=0, ignore_index=True)

cat_cols = ["Race", "Driver", "Compound"]
num_cols = [c for c in all_df.columns if c not in cat_cols + [ID_COL]]

for c in cat_cols:
    all_df[c] = all_df[c].astype(str).fillna("missing")


def clean_name(s):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")


lap = all_df["LapNumber"].astype(float).clip(lower=1)
progress = all_df["RaceProgress"].astype(float).clip(lower=1e-4, upper=1.0)
tyre = all_df["TyreLife"].astype(float).clip(lower=1)

all_df["EstimatedRaceLaps"] = lap / progress
all_df["EstimatedLapsToGo"] = all_df["EstimatedRaceLaps"] - lap
all_df["TyreLifeToLap"] = tyre / lap
all_df["TyreLifeToRace"] = tyre / all_df["EstimatedRaceLaps"].clip(lower=1)
all_df["DegPerTyreLap"] = all_df["Cumulative_Degradation"] / tyre
all_df["LapDeltaPerTyreLap"] = all_df["LapTime_Delta"] / tyre
all_df["AbsLapTimeDelta"] = all_df["LapTime_Delta"].abs()
all_df["IsWetCompound"] = all_df["Compound"].isin(["WET", "INTERMEDIATE"]).astype(int)
all_df["FreshTyre"] = (all_df["TyreLife"] <= 2).astype(int)
all_df["LateRace"] = (all_df["RaceProgress"] >= 0.75).astype(int)

combo_defs = [
    ("Race", "Driver"),
    ("Race", "Compound"),
    ("Driver", "Compound"),
    ("Race", "Stint", "Compound"),
    ("Year", "Race"),
]
combo_cols = []
for cols in combo_defs:
    name = "__".join(cols)
    combo_cols.append(name)
    all_df[name] = all_df[list(cols)].astype(str).agg("__".join, axis=1)

for c in cat_cols + combo_cols:
    counts = all_df[c].value_counts()
    all_df[f"{c}_freq"] = all_df[c].map(counts).astype("float32")

for c in cat_cols:
    all_df[f"{c}_code"] = pd.factorize(all_df[c], sort=True)[0].astype("int32")

rank_group = ["Year", "Race", "LapNumber"]
rank_source = [
    "TyreLife",
    "Cumulative_Degradation",
    "LapTime (s)",
    "LapTime_Delta",
    "Position",
    "Position_Change",
    "Stint",
]
all_df["RaceLapSize"] = (
    all_df.groupby(rank_group)[ID_COL].transform("size").astype("float32")
)
rank_features = ["RaceLapSize"]
for c in rank_source:
    g = all_df.groupby(rank_group)[c]
    rname = f"{clean_name(c)}_RaceLapRank"
    zname = f"{clean_name(c)}_RaceLapZ"
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    all_df[rname] = g.rank(method="average", pct=True).astype("float32")
    all_df[zname] = (
        ((all_df[c] - mean) / std)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(-6, 6)
        .astype("float32")
    )
    rank_features.extend([rname, zname])

numeric_features = [
    c for c in all_df.columns if c not in cat_cols + combo_cols + [ID_COL]
]
numeric_features = [c for c in numeric_features if c != TARGET]

X_num = all_df[numeric_features].copy()
X_num.columns = [clean_name(c) for c in X_num.columns]
X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(-999).astype("float32")

rank_extra = [
    "LapNumber",
    "RaceProgress",
    "TyreLife",
    "Stint",
    "PitStop",
    "Position",
    "Position_Change",
    "IsWetCompound",
    "FreshTyre",
    "LateRace",
    "Compound_code",
]
rank_cols = [c for c in rank_features + rank_extra if c in all_df.columns]
X_rank = all_df[rank_cols].copy()
X_rank.columns = [clean_name(c) for c in X_rank.columns]
X_rank = X_rank.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")

all_df["Year_cat"] = all_df["Year"].astype(str)
catboost_cat_cols = cat_cols + combo_cols + ["Year_cat"]
catboost_num_cols = [
    "LapNumber",
    "TyreLife",
    "Stint",
    "PitStop",
    "Position",
    "Position_Change",
    "RaceProgress",
    "LapTime (s)",
    "LapTime_Delta",
    "Cumulative_Degradation",
    "EstimatedRaceLaps",
    "EstimatedLapsToGo",
    "TyreLifeToLap",
    "DegPerTyreLap",
    "LapDeltaPerTyreLap",
    "AbsLapTimeDelta",
    "IsWetCompound",
    "FreshTyre",
    "LateRace",
]
X_cat = all_df[catboost_cat_cols + catboost_num_cols].copy()
for c in catboost_cat_cols:
    X_cat[c] = X_cat[c].astype(str).fillna("missing")
for c in catboost_num_cols:
    X_cat[c] = (
        pd.to_numeric(X_cat[c], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(-999)
        .astype("float32")
    )

X_num_train, X_num_test = X_num.iloc[:n_train], X_num.iloc[n_train:]
X_rank_train, X_rank_test = X_rank.iloc[:n_train], X_rank.iloc[n_train:]
X_cat_train, X_cat_test = X_cat.iloc[:n_train], X_cat.iloc[n_train:]

groups = (train["Year"].astype(str) + "__" + train["Race"].astype(str)).values
if HAS_SGK:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X_num_train, y, groups))
else:
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X_num_train, y))


def pos_weight(labels):
    pos = max(1, int(np.sum(labels == 1)))
    neg = max(1, int(np.sum(labels == 0)))
    return neg / pos


base_names = ["lgb_numeric", "race_lap_ranker", "catboost_native_cat"]
oof_base = np.zeros((n_train, len(base_names)), dtype=np.float32)
test_base = np.zeros((n_test, len(base_names)), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    y_tr, y_va = y[tr_idx], y[va_idx]
    spw = pos_weight(y_tr)

    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1600,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE + fold,
        n_jobs=threads,
        verbosity=-1,
    )
    lgb_model.fit(
        X_num_train.iloc[tr_idx],
        y_tr,
        eval_set=[(X_num_train.iloc[va_idx], y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    oof_base[va_idx, 0] = lgb_model.predict_proba(X_num_train.iloc[va_idx])[:, 1]
    test_base[:, 0] += lgb_model.predict_proba(X_num_test)[:, 1] / N_SPLITS

    sw = np.where(y_tr == 1, spw, 1.0)
    rank_model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.055,
        max_iter=420,
        max_leaf_nodes=31,
        min_samples_leaf=45,
        l2_regularization=0.04,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=35,
        random_state=RANDOM_STATE + 100 + fold,
    )
    rank_model.fit(X_rank_train.iloc[tr_idx], y_tr, sample_weight=sw)
    oof_base[va_idx, 1] = rank_model.predict_proba(X_rank_train.iloc[va_idx])[:, 1]
    test_base[:, 1] += rank_model.predict_proba(X_rank_test)[:, 1] / N_SPLITS

    cat_model = CatBoostClassifier(
        iterations=900,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=7.0,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE + 200 + fold,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        verbose=False,
        thread_count=threads,
    )
    cat_model.fit(
        X_cat_train.iloc[tr_idx],
        y_tr,
        cat_features=catboost_cat_cols,
        eval_set=(X_cat_train.iloc[va_idx], y_va),
        use_best_model=True,
        verbose=False,
    )
    oof_base[va_idx, 2] = cat_model.predict_proba(X_cat_train.iloc[va_idx])[:, 1]
    test_base[:, 2] += cat_model.predict_proba(X_cat_test)[:, 1] / N_SPLITS

    fold_scores = [
        roc_auc_score(y_va, oof_base[va_idx, i]) for i in range(len(base_names))
    ]
    print(
        f"fold {fold} base AUCs: "
        + ", ".join(f"{n}={s:.6f}" for n, s in zip(base_names, fold_scores))
    )

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_sigmoid_calibrator(pred, target):
    model = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
    model.fit(logit(pred).reshape(-1, 1), target)
    return model


def apply_sigmoid_calibrator(model, pred):
    return model.predict_proba(logit(pred).reshape(-1, 1))[:, 1]


meta_oof = np.zeros(n_train, dtype=np.float32)
meta_splitter = StratifiedKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + 999
)

for tr_idx, va_idx in meta_splitter.split(oof_base, y):
    cal_tr = np.zeros((len(tr_idx), len(base_names)), dtype=np.float32)
    cal_va = np.zeros((len(va_idx), len(base_names)), dtype=np.float32)

    for j in range(len(base_names)):
        cal = fit_sigmoid_calibrator(oof_base[tr_idx, j], y[tr_idx])
        cal_tr[:, j] = apply_sigmoid_calibrator(cal, oof_base[tr_idx, j])
        cal_va[:, j] = apply_sigmoid_calibrator(cal, oof_base[va_idx, j])

    stacker = LogisticRegression(C=20.0, solver="lbfgs", max_iter=2000)
    stacker.fit(logit(cal_tr), y[tr_idx])
    meta_oof[va_idx] = stacker.predict_proba(logit(cal_va))[:, 1]

stack_auc = roc_auc_score(y, meta_oof)
base_auc = {
    name: float(roc_auc_score(y, oof_base[:, i])) for i, name in enumerate(base_names)
}

final_cal_oof = np.zeros_like(oof_base, dtype=np.float32)
final_cal_test = np.zeros_like(test_base, dtype=np.float32)
for j in range(len(base_names)):
    cal = fit_sigmoid_calibrator(oof_base[:, j], y)
    final_cal_oof[:, j] = apply_sigmoid_calibrator(cal, oof_base[:, j])
    final_cal_test[:, j] = apply_sigmoid_calibrator(cal, test_base[:, j])

final_stacker = LogisticRegression(C=20.0, solver="lbfgs", max_iter=2000)
final_stacker.fit(logit(final_cal_oof), y)
test_pred = final_stacker.predict_proba(logit(final_cal_test))[:, 1]
test_pred = np.clip(test_pred, 0, 1)

sub = sample.copy()
sub[TARGET] = test_pred
sub.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
sub.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

oof_df = pd.DataFrame(
    {
        "row": np.arange(n_train),
        "target": y,
        "prediction": meta_oof,
    }
)
oof_df.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_roc_auc": float(stack_auc),
    "base_oof_roc_auc": base_auc,
    "research_hypotheses_llm_claimed_used": ["000960"],
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"OOF sigmoid-logit stack ROC AUC: {stack_auc:.6f}")
print(json.dumps(result, indent=2))
