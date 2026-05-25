import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from lightgbm import LGBMClassifier
from catboost import CatBoostRanker, Pool

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
CAT_COLS = ["Compound", "Driver", "Race"]
SEED = 42
BLEND_POINTWISE = 0.65

train_raw = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test_raw = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))


def safe_name(col):
    return col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def add_features(df):
    df = df.copy()
    df["Year_Race"] = df["Year"].astype(str) + "_" + df["Race"].astype(str)
    df["query_id"] = df["Year_Race"] + "_lap_" + df["LapNumber"].astype(str)

    q = df.groupby("query_id", sort=False)
    df["query_size"] = q["LapNumber"].transform("size").astype("float32")

    for col in [
        "TyreLife",
        "LapTime_Delta",
        "LapTime (s)",
        "Cumulative_Degradation",
        "Position",
        "RaceProgress",
    ]:
        base = safe_name(col)
        df[f"{base}_q_pct"] = q[col].rank(pct=True).astype("float32")
        df[f"{base}_q_center"] = (df[col] - q[col].transform("mean")).astype("float32")

    df["front_position_pressure_q_pct"] = (
        q["Position"].rank(pct=True, ascending=False).astype("float32")
    )
    df["tyre_life_x_progress"] = (df["TyreLife"] * df["RaceProgress"]).astype("float32")
    df["degradation_per_tyre_lap"] = (
        df["Cumulative_Degradation"] / (df["TyreLife"] + 1.0)
    ).astype("float32")
    df["lap_delta_abs"] = df["LapTime_Delta"].abs().astype("float32")
    df["is_current_pit_lap"] = df["PitStop"].astype("int8")
    return df


def query_percentile(values, query_ids):
    values = pd.Series(values)
    query_ids = pd.Series(query_ids)
    return values.groupby(query_ids, sort=False).rank(pct=True).astype("float32").values


train = add_features(train_raw)
test = add_features(test_raw)

y = train[TARGET].astype(int).values
groups = train["Year_Race"].values
train_query = train["query_id"].astype(str).values
test_query = test["query_id"].astype(str).values

drop_cols = [ID_COL, TARGET, "Year_Race", "query_id"]
features = [c for c in train.columns if c not in drop_cols]
cat_features = [c for c in CAT_COLS if c in features]

for c in cat_features:
    cats = pd.Index(pd.concat([train[c], test[c]], axis=0).astype(str).unique())
    train[c] = pd.Categorical(train[c].astype(str), categories=cats)
    test[c] = pd.Categorical(test[c].astype(str), categories=cats)

rank_train = train[features].copy()
rank_test = test[features].copy()
for c in cat_features:
    rank_train[c] = train[c].cat.codes.astype("int16")
    rank_test[c] = test[c].cat.codes.astype("int16")

point_oof = np.zeros(len(train), dtype=np.float32)
rank_oof_raw = np.zeros(len(train), dtype=np.float32)

logo = LeaveOneGroupOut()
splits = list(logo.split(train, y, groups))
n_splits = len(splits)
print(
    f"Running LeaveOneGroupOut on Year_Race with {n_splits} folds for hypothesis 000749"
)

threads = max(1, os.cpu_count() or 1)

lgb_params = dict(
    objective="binary",
    n_estimators=120,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=80,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=threads,
    verbosity=-1,
    force_col_wise=True,
)

rank_params = dict(
    loss_function="YetiRankPairwise",
    iterations=50,
    learning_rate=0.08,
    depth=5,
    l2_leaf_reg=6.0,
    random_seed=SEED,
    verbose=False,
    allow_writing_files=False,
    thread_count=threads,
)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    clf = LGBMClassifier(**lgb_params)
    clf.fit(
        train.iloc[tr_idx][features],
        y[tr_idx],
        categorical_feature=cat_features,
    )
    point_oof[va_idx] = clf.predict_proba(train.iloc[va_idx][features])[:, 1].astype(
        np.float32
    )

    tr_sorted = tr_idx[np.argsort(train_query[tr_idx], kind="mergesort")]
    va_sorted = va_idx[np.argsort(train_query[va_idx], kind="mergesort")]

    ranker = CatBoostRanker(**rank_params)
    ranker.fit(
        Pool(
            rank_train.iloc[tr_sorted],
            label=y[tr_sorted],
            group_id=train_query[tr_sorted],
        )
    )
    rank_oof_raw[va_sorted] = ranker.predict(
        Pool(rank_train.iloc[va_sorted], group_id=train_query[va_sorted])
    ).astype(np.float32)

    if fold % 10 == 0 or fold == n_splits:
        print(f"Completed fold {fold}/{n_splits}")

rank_oof = query_percentile(rank_oof_raw, train_query)
blend_oof = BLEND_POINTWISE * point_oof + (1.0 - BLEND_POINTWISE) * rank_oof

point_auc = roc_auc_score(y, point_oof)
rank_auc = roc_auc_score(y, rank_oof)
blend_auc = roc_auc_score(y, blend_oof)

print(f"OOF ROC AUC pointwise: {point_auc:.6f}")
print(f"OOF ROC AUC query_ranker: {rank_auc:.6f}")
print(f"OOF ROC AUC blended: {blend_auc:.6f}")

full_point = LGBMClassifier(**{**lgb_params, "n_estimators": 160})
full_point.fit(train[features], y, categorical_feature=cat_features)
full_point_test = full_point.predict_proba(test[features])[:, 1]

full_order = np.argsort(train_query, kind="mergesort")
full_ranker = CatBoostRanker(**{**rank_params, "iterations": 80})
full_ranker.fit(
    Pool(
        rank_train.iloc[full_order],
        label=y[full_order],
        group_id=train_query[full_order],
    )
)

test_order = np.argsort(test_query, kind="mergesort")
full_rank_test_raw = np.empty(len(test), dtype=np.float32)
full_rank_test_raw[test_order] = full_ranker.predict(
    Pool(rank_test.iloc[test_order], group_id=test_query[test_order])
).astype(np.float32)
full_rank_test = query_percentile(full_rank_test_raw, test_query)

test_pred = BLEND_POINTWISE * full_point_test + (1.0 - BLEND_POINTWISE) * full_rank_test
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": blend_oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

test_predictions = sample.copy()
test_predictions[TARGET] = test_pred
test_predictions.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

result = {
    "metric": "roc_auc",
    "validation_score": float(blend_auc),
    "pointwise_oof_auc": float(point_auc),
    "query_ranker_oof_auc": float(rank_auc),
    "research_hypotheses_llm_claimed_used": ["000749"],
    "submission_path": os.path.join(WORK_DIR, "submission.csv"),
}
print(json.dumps(result, indent=2))
