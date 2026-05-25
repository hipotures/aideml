import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

from lightgbm import LGBMClassifier, LGBMRanker, early_stopping, log_evaluation

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 42
N_SPLITS = 5

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values


def add_features(df):
    df = df.copy()
    df["compound_regime"] = np.where(
        df["Compound"].isin(["WET", "INTERMEDIATE"]), "WET_INT", "DRY"
    )
    df["RaceYear"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)
    df["RaceYearLap"] = df["RaceYear"] + "_L" + df["LapNumber"].astype(str)
    df["query_group"] = df["RaceYearLap"] + "_" + df["compound_regime"]

    tyre = df["TyreLife"].replace(0, np.nan)
    lap = df["LapNumber"].replace(0, np.nan)
    progress = df["RaceProgress"].clip(1e-4, 1.0)

    df["degradation_per_tyre_lap"] = df["Cumulative_Degradation"] / tyre
    df["tyre_life_frac_of_race_lap"] = df["TyreLife"] / lap
    df["estimated_total_laps"] = df["LapNumber"] / progress
    df["estimated_laps_remaining"] = df["estimated_total_laps"] - df["LapNumber"]
    df["tyre_life_to_remaining"] = df["TyreLife"] / (
        df["estimated_laps_remaining"].clip(lower=1.0)
    )
    df["stint_x_tyre_life"] = df["Stint"] * df["TyreLife"]
    df["position_x_progress"] = df["Position"] * df["RaceProgress"]
    df["abs_position_change"] = df["Position_Change"].abs()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


train_fe = add_features(train)
test_fe = add_features(test)

drop_cols = [ID_COL, TARGET, "query_group"]
features = [c for c in train_fe.columns if c not in drop_cols]
cat_cols = [
    "Compound",
    "Driver",
    "Race",
    "compound_regime",
    "RaceYear",
    "RaceYearLap",
    "Year",
]

for col in cat_cols:
    all_vals = pd.concat([train_fe[col], test_fe[col]], axis=0).astype(str)
    cats = pd.Index(all_vals.unique())
    train_fe[col] = pd.Categorical(train_fe[col].astype(str), categories=cats)
    test_fe[col] = pd.Categorical(test_fe[col].astype(str), categories=cats)

for col in features:
    if col not in cat_cols:
        med = pd.concat([train_fe[col], test_fe[col]], axis=0).median()
        train_fe[col] = train_fe[col].fillna(med)
        test_fe[col] = test_fe[col].fillna(med)

X = train_fe[features]
X_test = test_fe[features]
race_year_groups = train_fe["RaceYear"].astype(str).values
query_groups = train_fe["query_group"].astype(str).values
test_query_groups = test_fe["query_group"].astype(str).values
train_regime = train_fe["compound_regime"].astype(str).values
test_regime = test_fe["compound_regime"].astype(str).values


def sorted_by_query(X_part, y_part, q_part):
    order = np.argsort(q_part, kind="mergesort")
    q_sorted = q_part[order]
    _, counts = np.unique(q_sorted, return_counts=True)
    return X_part.iloc[order], y_part[order], counts.tolist()


def grouped_softmax_probability(scores, base_prob, groups):
    scores = np.asarray(scores, dtype=float)
    base_prob = np.asarray(base_prob, dtype=float)
    groups = np.asarray(groups).astype(str)
    out = np.zeros_like(scores, dtype=float)

    frame = pd.DataFrame({"g": groups})
    for idx in frame.groupby("g", sort=False).indices.values():
        idx = np.asarray(idx)
        if len(idx) == 1:
            out[idx] = base_prob[idx]
            continue
        s = scores[idx]
        s = (s - s.mean()) / (s.std() + 1e-6)
        z = s - s.max()
        w = np.exp(np.clip(z, -50, 50))
        w = w / w.sum()
        expected_stops = np.clip(base_prob[idx].sum(), 0.0, float(len(idx)))
        out[idx] = np.clip(expected_stops * w, 0.0, 1.0)
    return out


def blend_predictions(clf_pred, rank_pred, regime, w):
    regime = np.asarray(regime).astype(str)
    local_w = np.where(regime == "DRY", w, 0.5 * w)
    return np.clip((1.0 - local_w) * clf_pred + local_w * rank_pred, 0.0, 1.0)


if HAS_SGK:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X, y, groups=race_year_groups))
else:
    splitter = GroupKFold(n_splits=N_SPLITS)
    splits = list(splitter.split(X, y, groups=race_year_groups))

oof_clf = np.zeros(len(train_fe))
oof_rank = np.zeros(len(train_fe))
test_clf_folds = []
test_rank_folds = []

pos = y.sum()
neg = len(y) - pos
scale_pos_weight = neg / max(pos, 1)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    clf = LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE + fold,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="auc",
        categorical_feature=cat_cols,
        callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
    )

    va_clf = clf.predict_proba(X_va)[:, 1]
    te_clf = clf.predict_proba(X_test)[:, 1]
    oof_clf[va_idx] = va_clf
    test_clf_folds.append(te_clf)

    X_rank_tr, y_rank_tr, rank_group = sorted_by_query(X_tr, y_tr, query_groups[tr_idx])
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=900,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        random_state=RANDOM_STATE + 100 + fold,
        n_jobs=-1,
        verbose=-1,
    )
    ranker.fit(
        X_rank_tr,
        y_rank_tr,
        group=rank_group,
        categorical_feature=cat_cols,
        callbacks=[log_evaluation(0)],
    )

    va_rank_raw = ranker.predict(X_va)
    te_rank_raw = ranker.predict(X_test)
    va_rank = grouped_softmax_probability(va_rank_raw, va_clf, query_groups[va_idx])
    te_rank = grouped_softmax_probability(te_rank_raw, te_clf, test_query_groups)

    oof_rank[va_idx] = va_rank
    test_rank_folds.append(te_rank)

    fold_auc = roc_auc_score(
        y_va, blend_predictions(va_clf, va_rank, train_regime[va_idx], 0.20)
    )
    print(f"fold={fold} auc_fixed_w0.20={fold_auc:.6f}")

clf_auc = roc_auc_score(y, oof_clf)
rank_auc = roc_auc_score(y, oof_rank)

candidate_weights = np.linspace(0.0, 0.40, 21)
blend_scores = []
for w in candidate_weights:
    pred = blend_predictions(oof_clf, oof_rank, train_regime, float(w))
    blend_scores.append(roc_auc_score(y, pred))

best_i = int(np.argmax(blend_scores))
best_w = float(candidate_weights[best_i])
best_auc = float(blend_scores[best_i])

test_clf = np.mean(np.column_stack(test_clf_folds), axis=1)
test_rank = np.mean(np.column_stack(test_rank_folds), axis=1)
test_pred = blend_predictions(test_clf, test_rank, test_regime, best_w)

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0.0, 1.0)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

oof_out = pd.DataFrame(
    {
        "row": np.arange(len(train_fe)),
        "target": y,
        "prediction": blend_predictions(oof_clf, oof_rank, train_regime, best_w),
    }
)
oof_out.to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

test_out = submission[[ID_COL, TARGET]].copy()
test_out.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

report = {
    "metric": "roc_auc",
    "cv_classifier_auc": float(clf_auc),
    "cv_rank_aux_auc": float(rank_auc),
    "cv_blended_auc": best_auc,
    "best_dry_rank_blend_weight": best_w,
    "wet_or_intermediate_rank_blend_weight": 0.5 * best_w,
    "research_hypotheses_llm_claimed_used": ["000155"],
    "files_written": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
    ],
}
print(json.dumps(report, indent=2))
