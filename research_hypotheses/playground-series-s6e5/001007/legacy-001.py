import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier, LGBMRanker

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Driver", "Race", "Compound"]
N_SPLITS = 5
RANDOM_STATE = 42

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values


def add_features(df):
    out = df.copy()
    out["WetRegime"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    out["RaceLapKey"] = (
        out["Year"].astype(str)
        + "_"
        + out["Race"].astype(str)
        + "_"
        + out["LapNumber"].astype(str)
        + "_"
        + out["WetRegime"].astype(str)
    )
    out["TyreLife_x_Progress"] = out["TyreLife"] * out["RaceProgress"]
    out["Deg_per_TyreLife"] = out["Cumulative_Degradation"] / (out["TyreLife"] + 1.0)
    out["Lap_frac"] = out["LapNumber"] / (out["LapNumber"].max() + 1.0)
    return out


train_fe = add_features(train.drop(columns=[TARGET]))
test_fe = add_features(test)

feature_cols = [c for c in train_fe.columns if c not in [ID_COL, "RaceLapKey"]]
all_fe = pd.concat(
    [train_fe[feature_cols], test_fe[feature_cols]], axis=0, ignore_index=True
)

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
all_fe[CAT_COLS] = encoder.fit_transform(all_fe[CAT_COLS].astype(str))

X = all_fe.iloc[: len(train_fe)].reset_index(drop=True)
X_test = all_fe.iloc[len(train_fe) :].reset_index(drop=True)

query_keys = train_fe["RaceLapKey"].reset_index(drop=True)
test_query_keys = test_fe["RaceLapKey"].reset_index(drop=True)


def rank_query_groups(keys):
    key_values = keys.values
    order = np.lexsort((np.arange(len(key_values)), key_values))
    sorted_keys = key_values[order]
    _, counts = np.unique(sorted_keys, return_counts=True)
    return order, counts.tolist()


def query_percentile_scores(raw_scores, keys):
    s = pd.Series(raw_scores)
    pct = s.groupby(keys, sort=False).rank(method="average", pct=True)
    return pct.fillna(0.5).values


clf_params = dict(
    objective="binary",
    n_estimators=700,
    learning_rate=0.035,
    num_leaves=48,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

ranker_params = dict(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=500,
    learning_rate=0.04,
    num_leaves=40,
    max_depth=-1,
    min_child_samples=40,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_clf = np.zeros(len(train))
oof_rank = np.zeros(len(train))
oof_blend = np.zeros(len(train))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    X_va = X.iloc[va_idx].reset_index(drop=True)
    y_va = y[va_idx]

    clf = LGBMClassifier(**clf_params)
    clf.fit(X_tr, y_tr)
    clf_pred = clf.predict_proba(X_va)[:, 1]

    train_keys_fold = query_keys.iloc[tr_idx].reset_index(drop=True)
    valid_keys_fold = query_keys.iloc[va_idx].reset_index(drop=True)

    rank_order, rank_groups = rank_query_groups(train_keys_fold)
    ranker = LGBMRanker(**ranker_params)
    ranker.fit(
        X_tr.iloc[rank_order],
        y_tr[rank_order],
        group=rank_groups,
    )

    raw_rank = ranker.predict(X_va)
    rank_pred = query_percentile_scores(raw_rank, valid_keys_fold)

    blend_pred = 0.75 * clf_pred + 0.25 * rank_pred

    oof_clf[va_idx] = clf_pred
    oof_rank[va_idx] = rank_pred
    oof_blend[va_idx] = blend_pred

    print(
        f"fold={fold} "
        f"classifier_auc={roc_auc_score(y_va, clf_pred):.6f} "
        f"rank_auc={roc_auc_score(y_va, rank_pred):.6f} "
        f"blend_auc={roc_auc_score(y_va, blend_pred):.6f}"
    )

clf_auc = roc_auc_score(y, oof_clf)
rank_auc = roc_auc_score(y, oof_rank)
blend_auc = roc_auc_score(y, oof_blend)

print(f"OOF classifier ROC AUC: {clf_auc:.6f}")
print(f"OOF pure rank ROC AUC: {rank_auc:.6f}")
print(f"OOF blended ROC AUC: {blend_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_blend,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

final_clf = LGBMClassifier(**clf_params)
final_clf.fit(X, y)
test_clf = final_clf.predict_proba(X_test)[:, 1]

final_order, final_groups = rank_query_groups(query_keys.reset_index(drop=True))
final_ranker = LGBMRanker(**ranker_params)
final_ranker.fit(
    X.iloc[final_order],
    y[final_order],
    group=final_groups,
)

test_rank_raw = final_ranker.predict(X_test)
test_rank = query_percentile_scores(
    test_rank_raw, test_query_keys.reset_index(drop=True)
)
test_pred = np.clip(0.75 * test_clf + 0.25 * test_rank, 0, 1)

submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred})
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["001007"],
    "metric": "roc_auc",
    "oof_classifier_auc": float(clf_auc),
    "oof_pure_rank_auc": float(rank_auc),
    "oof_blended_auc": float(blend_auc),
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print(json.dumps(review, indent=2))
