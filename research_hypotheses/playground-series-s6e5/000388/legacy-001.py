import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    from sklearn.model_selection import GroupKFold

    HAS_SGK = False

warnings.filterwarnings("ignore")

SEED = 388
TARGET = "PitNextLap"
ID_COL = "id"
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

N_JOBS = max(1, min(8, os.cpu_count() or 1))
OUTER_SPLITS = 5
INNER_SPLITS = 5
EPS = 1e-5


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


def logit_features(pmat):
    pmat = clip_prob(pmat)
    return np.log(pmat / (1.0 - pmat))


def add_features(df):
    df = df.copy()
    tyre = df["TyreLife"].astype(float)
    lap = df["LapNumber"].astype(float)
    progress = df["RaceProgress"].astype(float)
    deg = df["Cumulative_Degradation"].astype(float)
    delta = df["LapTime_Delta"].astype(float)
    stint = df["Stint"].astype(float)
    pitstop = df["PitStop"].astype(float)

    df["ProgressLeft"] = 1.0 - progress
    df["TyreLife_Log1p"] = np.log1p(np.maximum(tyre, 0))
    df["LapNumber_Log1p"] = np.log1p(np.maximum(lap, 0))
    df["TyreLife_RaceProgress"] = tyre * progress
    df["TyreLife_PerLap"] = tyre / np.maximum(lap, 1)
    df["Degradation_PerTyreLife"] = deg / np.maximum(tyre, 1)
    df["LapDelta_PerTyreLife"] = delta / np.maximum(tyre, 1)
    df["CumulativeDeg_x_LogTyreLife"] = deg * np.log1p(np.maximum(tyre, 0))
    df["Positive_LapTime_Delta"] = np.maximum(delta, 0)
    df["Abs_LapTime_Delta"] = np.abs(delta)
    df["Stint_TyreLife"] = stint * tyre
    df["PitStop_x_Progress"] = pitstop * progress
    df["LateRace_TyreLife"] = tyre * np.maximum(progress - 0.55, 0)
    df["DryCompound"] = df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype("int8")
    df["WetCompound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype("int8")
    df["SoftCompound"] = (df["Compound"] == "SOFT").astype("int8")
    df["HardCompound"] = (df["Compound"] == "HARD").astype("int8")
    return df


def make_group_key(df):
    return df["Year"].astype(str) + "__" + df["Race"].astype(str)


def make_rank_key(df):
    return (
        df["Year"].astype(str)
        + "__"
        + df["Race"].astype(str)
        + "__"
        + df["Driver"].astype(str)
    )


def grouped_splits(y, groups, n_splits, seed):
    n_splits = min(n_splits, len(np.unique(groups)))
    if HAS_SGK:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        return list(splitter.split(np.zeros(len(y)), y, groups))
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def scale_pos_weight(y):
    pos = float(np.sum(y))
    neg = float(len(y) - pos)
    return float(np.sqrt(neg / max(pos, 1.0)))


def base_params(y_fit, seed):
    return dict(
        objective="binary",
        metric="auc",
        n_estimators=420,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight(y_fit),
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_col_wise=True,
    )


def monotone_params(y_fit, constraints, seed):
    p = base_params(y_fit, seed)
    p.update(
        n_estimators=360,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=100,
        monotone_constraints=constraints,
    )
    return p


def wear_params(y_fit, seed):
    p = base_params(y_fit, seed)
    p.update(
        n_estimators=450,
        learning_rate=0.032,
        num_leaves=47,
        min_child_samples=60,
        colsample_bytree=0.9,
        reg_lambda=4.0,
    )
    return p


def ranker_params(seed):
    return dict(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_estimators=300,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=N_JOBS,
        verbosity=-1,
        force_col_wise=True,
    )


def ranker_order_and_groups(qids):
    qids = pd.Series(qids).astype(str).reset_index(drop=True)
    order = np.argsort(qids.values, kind="mergesort")
    sorted_qids = qids.iloc[order]
    group_sizes = sorted_qids.groupby(sorted_qids, sort=False).size().to_numpy()
    return order, group_sizes


class BetaCalibrator:
    def __init__(self):
        self.model = None

    def _features(self, p):
        p = clip_prob(p)
        return np.column_stack([np.log(p), np.log1p(-p)])

    def fit(self, p, y):
        if len(np.unique(y)) < 2:
            self.model = None
            return self
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.model.fit(self._features(p), y)
        return self

    def predict(self, p):
        if self.model is None:
            return clip_prob(p)
        return self.model.predict_proba(self._features(p))[:, 1]


def make_blender():
    return LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced"
    )


def cats_for(cols):
    return [c for c in cat_cols if c in cols]


def fit_predict_bundle(train_idx, pred_idx=None, predict_test=False, seed=SEED):
    y_fit = y[train_idx]
    pred_parts = []
    test_parts = []

    model = lgb.LGBMClassifier(**base_params(y_fit, seed + 11))
    model.fit(X.iloc[train_idx][main_cols], y_fit, categorical_feature=main_cat)
    if pred_idx is not None:
        pred_parts.append(model.predict_proba(X.iloc[pred_idx][main_cols])[:, 1])
    if predict_test:
        test_parts.append(model.predict_proba(X_test[main_cols])[:, 1])

    qids = rank_qid.iloc[train_idx].reset_index(drop=True)
    order, group_sizes = ranker_order_and_groups(qids)
    ranker = lgb.LGBMRanker(**ranker_params(seed + 22))
    ranker.fit(
        X.iloc[train_idx].iloc[order][ranker_cols],
        y_fit[order].astype(int),
        group=group_sizes,
        categorical_feature=ranker_cat,
    )
    if pred_idx is not None:
        pred_parts.append(sigmoid(ranker.predict(X.iloc[pred_idx][ranker_cols])))
    if predict_test:
        test_parts.append(sigmoid(ranker.predict(X_test[ranker_cols])))

    mono = lgb.LGBMClassifier(**monotone_params(y_fit, mono_constraints, seed + 33))
    mono.fit(X.iloc[train_idx][mono_cols], y_fit, categorical_feature=mono_cat)
    if pred_idx is not None:
        pred_parts.append(mono.predict_proba(X.iloc[pred_idx][mono_cols])[:, 1])
    if predict_test:
        test_parts.append(mono.predict_proba(X_test[mono_cols])[:, 1])

    wear = lgb.LGBMClassifier(**wear_params(y_fit, seed + 44))
    wear.fit(X.iloc[train_idx][wear_cols], y_fit, categorical_feature=wear_cat)
    if pred_idx is not None:
        pred_parts.append(wear.predict_proba(X.iloc[pred_idx][wear_cols])[:, 1])
    if predict_test:
        test_parts.append(wear.predict_proba(X_test[wear_cols])[:, 1])

    pred_matrix = (
        np.column_stack(pred_parts).astype(np.float32) if pred_idx is not None else None
    )
    test_matrix = (
        np.column_stack(test_parts).astype(np.float32) if predict_test else None
    )
    return pred_matrix, test_matrix


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).to_numpy()
groups = make_group_key(train).to_numpy()
rank_qid = make_rank_key(train)

all_features = pd.concat(
    [train.drop(columns=[TARGET, ID_COL]), test.drop(columns=[ID_COL])],
    axis=0,
    ignore_index=True,
)
all_features = add_features(all_features)

cat_cols = all_features.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    all_features[c] = all_features[c].astype("category")

X = all_features.iloc[: len(train)].reset_index(drop=True)
X_test = all_features.iloc[len(train) :].reset_index(drop=True)

main_cols = X.columns.tolist()
ranker_cols = main_cols[:]
mono_cols = main_cols[:]

wear_seed_cols = [
    "Compound",
    "Race",
    "Driver",
    "Year",
    "LapNumber",
    "RaceProgress",
    "ProgressLeft",
    "TyreLife",
    "TyreLife_Log1p",
    "TyreLife_RaceProgress",
    "TyreLife_PerLap",
    "Stint",
    "Stint_TyreLife",
    "Cumulative_Degradation",
    "Degradation_PerTyreLife",
    "CumulativeDeg_x_LogTyreLife",
    "LapTime_Delta",
    "LapDelta_PerTyreLife",
    "Positive_LapTime_Delta",
    "Abs_LapTime_Delta",
    "PitStop",
    "PitStop_x_Progress",
    "LateRace_TyreLife",
    "DryCompound",
    "WetCompound",
    "SoftCompound",
    "HardCompound",
    "Position",
    "Position_Change",
]
wear_cols = [c for c in wear_seed_cols if c in X.columns]

main_cat = cats_for(main_cols)
ranker_cat = cats_for(ranker_cols)
mono_cat = cats_for(mono_cols)
wear_cat = cats_for(wear_cols)

mono_map = {
    "TyreLife": 1,
    "TyreLife_Log1p": 1,
    "TyreLife_RaceProgress": 1,
    "RaceProgress": 1,
    "Cumulative_Degradation": 1,
    "Degradation_PerTyreLife": 1,
    "CumulativeDeg_x_LogTyreLife": 1,
    "LateRace_TyreLife": 1,
    "PitStop": -1,
}
mono_constraints = [mono_map.get(c, 0) for c in mono_cols]

outer = grouped_splits(y, groups, OUTER_SPLITS, SEED)
stack_rel, cal_rel = outer[0]
stack_idx = np.asarray(stack_rel)
cal_idx = np.asarray(cal_rel)

y_stack = y[stack_idx]
groups_stack = groups[stack_idx]
inner = grouped_splits(y_stack, groups_stack, INNER_SPLITS, SEED + 1)

base_oof = np.zeros((len(stack_idx), 4), dtype=np.float32)

for fold, (tr_rel, va_rel) in enumerate(inner, 1):
    tr_idx = stack_idx[np.asarray(tr_rel)]
    va_idx = stack_idx[np.asarray(va_rel)]
    base_oof[va_rel], _ = fit_predict_bundle(
        tr_idx, pred_idx=va_idx, predict_test=False, seed=SEED + 100 * fold
    )

blend_oof = np.zeros(len(stack_idx), dtype=np.float32)
for fold, (tr_rel, va_rel) in enumerate(inner, 1):
    blender_fold = make_blender()
    blender_fold.fit(logit_features(base_oof[tr_rel]), y_stack[tr_rel])
    blend_oof[va_rel] = blender_fold.predict_proba(logit_features(base_oof[va_rel]))[
        :, 1
    ]

stack_oof_auc = roc_auc_score(y_stack, blend_oof)

blender = make_blender()
blender.fit(logit_features(base_oof), y_stack)

cal_base, test_base = fit_predict_bundle(
    stack_idx, pred_idx=cal_idx, predict_test=True, seed=SEED + 999
)
cal_raw = blender.predict_proba(logit_features(cal_base))[:, 1]
test_raw = blender.predict_proba(logit_features(test_base))[:, 1]

holdout_raw_auc = roc_auc_score(y[cal_idx], cal_raw)

beta = BetaCalibrator().fit(cal_raw, y[cal_idx])
cal_beta = beta.predict(cal_raw)
test_beta = beta.predict(test_raw)
holdout_beta_apparent_auc = roc_auc_score(y[cal_idx], cal_beta)

oof_all = np.zeros(len(train), dtype=np.float32)
oof_all[stack_idx] = blend_oof
oof_all[cal_idx] = cal_raw

pd.DataFrame({"row": np.arange(len(train)), "target": y, "prediction": oof_all}).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame({"row": cal_idx, "target": y[cal_idx], "prediction": cal_raw}).sort_values(
    "row"
).to_csv(
    os.path.join(WORK_DIR, "validation_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

submission = sample[[ID_COL]].copy()
submission[TARGET] = np.clip(test_beta, 0.0, 1.0)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

print(f"stack_group_oof_roc_auc={stack_oof_auc:.6f}")
print(f"untouched_group_holdout_raw_roc_auc={holdout_raw_auc:.6f}")
print(f"untouched_group_holdout_beta_apparent_roc_auc={holdout_beta_apparent_auc:.6f}")
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000388"],
            "metric": "roc_auc",
            "stack_group_oof_roc_auc": float(stack_oof_auc),
            "untouched_group_holdout_raw_roc_auc": float(holdout_raw_auc),
            "untouched_group_holdout_beta_apparent_roc_auc": float(
                holdout_beta_apparent_auc
            ),
            "submission_path": os.path.join(WORK_DIR, "submission.csv"),
        },
        sort_keys=True,
    )
)
