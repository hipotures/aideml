import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

from catboost import CatBoostRanker, Pool
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SEED = 314
N_SPLITS = 5
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

THREADS = max(1, min(8, os.cpu_count() or 1))
TARGET = "PitNextLap"
ID_COL = "id"


def clean_name(name):
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")
    return name if name else "col"


def make_unique(names):
    seen, out = {}, []
    for name in names:
        base = clean_name(name)
        k = seen.get(base, 0)
        out.append(base if k == 0 else f"{base}_{k}")
        seen[base] = k + 1
    return out


def safe_auc(y_true, pred):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, pred)


def fmt(x):
    return "nan" if pd.isna(x) else f"{x:.6f}"


def clean_float(x):
    return None if pd.isna(x) else float(x)


def build_features(train_df, test_df):
    train_x = train_df.drop(columns=[TARGET]).copy()
    test_x = test_df.copy()
    n_train = len(train_x)
    all_x = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    obj_cols = all_x.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        all_x[c] = all_x[c].fillna("NA").astype(str)

    if "Compound" in all_x.columns:
        dry = {"SOFT", "MEDIUM", "HARD"}
        compound_order = {
            "SOFT": 1,
            "MEDIUM": 2,
            "HARD": 3,
            "INTERMEDIATE": 4,
            "WET": 5,
        }
        all_x["is_dry_compound"] = all_x["Compound"].isin(dry).astype(np.int8)
        all_x["compound_order"] = (
            all_x["Compound"].map(compound_order).fillna(0).astype(np.int8)
        )

    if "RaceProgress" in all_x.columns:
        all_x["race_progress_left"] = (1.0 - all_x["RaceProgress"]).astype(np.float32)

    if {"TyreLife", "LapNumber"}.issubset(all_x.columns):
        all_x["tyre_life_frac_lap"] = (
            all_x["TyreLife"] / np.maximum(all_x["LapNumber"], 1)
        ).astype(np.float32)

    if {"Cumulative_Degradation", "TyreLife"}.issubset(all_x.columns):
        all_x["degradation_per_tyre_lap"] = (
            all_x["Cumulative_Degradation"] / np.maximum(all_x["TyreLife"], 1)
        ).astype(np.float32)

    if "LapTime_Delta" in all_x.columns:
        all_x["lap_delta_abs"] = all_x["LapTime_Delta"].abs().astype(np.float32)

    if "Position_Change" in all_x.columns:
        all_x["position_change_abs"] = all_x["Position_Change"].abs().astype(np.float32)
        all_x["position_loss"] = np.maximum(all_x["Position_Change"], 0).astype(
            np.float32
        )
        all_x["position_gain"] = np.maximum(-all_x["Position_Change"], 0).astype(
            np.float32
        )

    group_cols = [c for c in ["Race", "Year", "LapNumber"] if c in all_x.columns]
    if group_cols:
        grouped = all_x.groupby(group_cols, sort=False, observed=True)
        all_x["race_lap_size"] = grouped[ID_COL].transform("size").astype(np.int16)

        if "Compound" in all_x.columns:
            all_x["race_lap_compound_count"] = (
                all_x.groupby(group_cols + ["Compound"], sort=False, observed=True)[
                    ID_COL
                ]
                .transform("size")
                .astype(np.int16)
            )
            all_x["race_lap_compound_share"] = (
                all_x["race_lap_compound_count"] / all_x["race_lap_size"].clip(lower=1)
            ).astype(np.float32)
            all_x["race_lap_compound_nunique"] = (
                grouped["Compound"].transform("nunique").astype(np.int16)
            )

        rel_cols = [
            "TyreLife",
            "LapTime_s",
            "LapTime_Delta",
            "Cumulative_Degradation",
            "Position",
            "Position_Change",
            "Stint",
            "RaceProgress",
        ]
        for col in rel_cols:
            if col not in all_x.columns:
                continue
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std").fillna(0)
            all_x[f"{col}_grp_mean"] = mean.astype(np.float32)
            all_x[f"{col}_grp_std"] = std.astype(np.float32)
            all_x[f"{col}_vs_grp_mean"] = (all_x[col] - mean).astype(np.float32)
            all_x[f"{col}_rank_pct_in_lap"] = (
                grouped[col].rank(method="average", pct=True).astype(np.float32)
            )

    all_x.replace([np.inf, -np.inf], np.nan, inplace=True)

    for c in all_x.select_dtypes(include=["float64"]).columns:
        all_x[c] = all_x[c].astype(np.float32)
    for c in all_x.select_dtypes(include=["int64"]).columns:
        if c != ID_COL:
            all_x[c] = pd.to_numeric(all_x[c], downcast="integer")

    return all_x.iloc[:n_train].reset_index(drop=True), all_x.iloc[
        n_train:
    ].reset_index(drop=True)


def make_query_key(df):
    return (
        df["Race"].astype(str)
        + "|"
        + df["Year"].astype(str)
        + "|"
        + df["LapNumber"].astype(str)
    )


def make_folds(y, query_key):
    if StratifiedGroupKFold is not None:
        try:
            sgkf = StratifiedGroupKFold(
                n_splits=N_SPLITS, shuffle=True, random_state=SEED
            )
            return list(sgkf.split(np.zeros(len(y)), y.astype(int), groups=query_key))
        except Exception as exc:
            print(f"StratifiedGroupKFold failed ({exc}); falling back to GroupKFold.")
    gkf = GroupKFold(n_splits=N_SPLITS)
    return list(gkf.split(np.zeros(len(y)), y, groups=query_key))


def sort_indices_by_group(indices, group_codes):
    return (
        pd.Series(group_codes[indices], index=indices)
        .sort_values(kind="mergesort")
        .index.to_numpy(dtype=np.int64)
    )


def percentile_within_group(scores, groups):
    return (
        pd.DataFrame({"score": scores, "group": np.asarray(groups)})
        .groupby("group", sort=False)["score"]
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )


def prepare_lgb_categories(train_df, test_df, cat_cols):
    train_out = train_df.copy()
    test_out = test_df.copy()
    for c in cat_cols:
        cats = pd.Index(
            pd.concat(
                [train_out[c].astype(str), test_out[c].astype(str)], ignore_index=True
            ).unique()
        )
        dtype = pd.CategoricalDtype(categories=cats)
        train_out[c] = train_out[c].astype(str).astype(dtype)
        test_out[c] = test_out[c].astype(str).astype(dtype)
    return train_out, test_out


def make_lgb(seed):
    return LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=3.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=THREADS,
        verbosity=-1,
        force_col_wise=True,
    )


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

train.columns = make_unique(train.columns)
test.columns = make_unique(test.columns)

y = train[TARGET].astype(np.float32).to_numpy()
X_train_base, X_test_base = build_features(train, test)

feature_cols = [c for c in X_train_base.columns if c != ID_COL]
cat_features = [
    c
    for c in feature_cols
    if X_train_base[c].dtype == "object"
    or str(X_train_base[c].dtype).startswith("category")
]

for c in cat_features:
    X_train_base[c] = X_train_base[c].fillna("NA").astype(str)
    X_test_base[c] = X_test_base[c].fillna("NA").astype(str)

train_query = make_query_key(X_train_base)
test_query = make_query_key(X_test_base)
train_group_codes = pd.factorize(train_query, sort=True)[0].astype(np.int32)
folds = make_folds(y, train_query)

ranker_params = dict(
    loss_function="YetiRank",
    iterations=220,
    learning_rate=0.06,
    depth=6,
    l2_leaf_reg=8.0,
    random_seed=SEED,
    thread_count=THREADS,
    allow_writing_files=False,
    verbose=False,
)

X_rank_train = X_train_base[feature_cols].copy()
X_rank_test = X_test_base[feature_cols].copy()

oof_rank_score = np.zeros(len(X_rank_train), dtype=np.float32)
oof_rank_pct = np.zeros(len(X_rank_train), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    tr_sorted = sort_indices_by_group(tr_idx, train_group_codes)
    params = dict(ranker_params)
    params["random_seed"] = SEED + fold

    ranker = CatBoostRanker(**params)
    ranker.fit(
        Pool(
            X_rank_train.iloc[tr_sorted],
            label=y[tr_sorted],
            cat_features=cat_features,
            group_id=train_group_codes[tr_sorted],
        )
    )

    val_score = ranker.predict(
        Pool(X_rank_train.iloc[val_idx], cat_features=cat_features)
    )
    oof_rank_score[val_idx] = val_score.astype(np.float32)
    oof_rank_pct[val_idx] = percentile_within_group(
        val_score, train_query.iloc[val_idx].to_numpy()
    )

    print(
        f"Fold {fold} ranker raw-score ROC AUC: {fmt(safe_auc(y[val_idx], val_score))}"
    )

full_sorted = sort_indices_by_group(np.arange(len(X_rank_train)), train_group_codes)
final_ranker_params = dict(ranker_params)
final_ranker_params["random_seed"] = SEED + 777
final_ranker = CatBoostRanker(**final_ranker_params)
final_ranker.fit(
    Pool(
        X_rank_train.iloc[full_sorted],
        label=y[full_sorted],
        cat_features=cat_features,
        group_id=train_group_codes[full_sorted],
    )
)

test_rank_score = final_ranker.predict(
    Pool(X_rank_test, cat_features=cat_features)
).astype(np.float32)
test_rank_pct = percentile_within_group(test_rank_score, test_query.to_numpy())

X_train_meta = X_train_base[feature_cols].copy()
X_test_meta = X_test_base[feature_cols].copy()
X_train_meta["ranker_score"] = oof_rank_score
X_train_meta["ranker_pct"] = oof_rank_pct
X_test_meta["ranker_score"] = test_rank_score
X_test_meta["ranker_pct"] = test_rank_pct

meta_features = feature_cols + ["ranker_score", "ranker_pct"]

X_train_base_lgb, X_test_base_lgb = prepare_lgb_categories(
    X_train_base[feature_cols], X_test_base[feature_cols], cat_features
)
X_train_meta_lgb, X_test_meta_lgb = prepare_lgb_categories(
    X_train_meta[meta_features], X_test_meta[meta_features], cat_features
)

oof_base = np.zeros(len(y), dtype=np.float32)
oof_meta = np.zeros(len(y), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(folds, 1):
    base_model = make_lgb(SEED + 100 + fold)
    base_model.fit(
        X_train_base_lgb.iloc[tr_idx],
        y[tr_idx],
        categorical_feature=cat_features,
    )
    oof_base[val_idx] = base_model.predict_proba(X_train_base_lgb.iloc[val_idx])[:, 1]

    meta_model = make_lgb(SEED + 200 + fold)
    meta_model.fit(
        X_train_meta_lgb.iloc[tr_idx],
        y[tr_idx],
        categorical_feature=cat_features,
    )
    oof_meta[val_idx] = meta_model.predict_proba(X_train_meta_lgb.iloc[val_idx])[:, 1]

    print(
        f"Fold {fold} classifier ROC AUC: "
        f"baseline={fmt(safe_auc(y[val_idx], oof_base[val_idx]))}, "
        f"rank_meta={fmt(safe_auc(y[val_idx], oof_meta[val_idx]))}"
    )

baseline_auc = safe_auc(y, oof_base)
meta_auc = safe_auc(y, oof_meta)
ranker_raw_auc = safe_auc(y, oof_rank_score)
ranker_pct_auc = safe_auc(y, oof_rank_pct)

print(f"5-fold CV ROC AUC with hypothesis 000314 ranker meta-features: {fmt(meta_auc)}")
print(f"5-fold CV ROC AUC baseline without ranker meta-features: {fmt(baseline_auc)}")
print(f"OOF CatBoost ranker raw-score ROC AUC: {fmt(ranker_raw_auc)}")
print(f"OOF CatBoost ranker percentile ROC AUC: {fmt(ranker_pct_auc)}")

same_comp_col = "race_lap_compound_count"
many_threshold = (
    float(np.nanquantile(X_train_base[same_comp_col], 0.75))
    if same_comp_col in X_train_base
    else np.inf
)


def subgroup_report(name, mask):
    mask = np.asarray(mask, dtype=bool)
    b = safe_auc(y[mask], oof_base[mask])
    m = safe_auc(y[mask], oof_meta[mask])
    delta = m - b if not (pd.isna(m) or pd.isna(b)) else np.nan
    print(
        f"Subgroup {name}: rows={int(mask.sum())}, positives={int(y[mask].sum())}, "
        f"baseline_auc={fmt(b)}, rank_meta_auc={fmt(m)}, delta={fmt(delta)}"
    )
    return {
        "rows": int(mask.sum()),
        "positives": int(y[mask].sum()),
        "baseline_auc": clean_float(b),
        "rank_meta_auc": clean_float(m),
        "delta": clean_float(delta),
    }


dry_mask = (
    X_train_base["is_dry_compound"].to_numpy() == 1
    if "is_dry_compound" in X_train_base
    else np.ones(len(y), dtype=bool)
)
many_compound_mask = (
    X_train_base[same_comp_col].to_numpy() >= many_threshold
    if same_comp_col in X_train_base
    else np.zeros(len(y), dtype=bool)
)

subgroups = {
    "dry_compounds": subgroup_report("dry_compounds", dry_mask),
    "wet_or_intermediate_compounds": subgroup_report(
        "wet_or_intermediate_compounds", ~dry_mask
    ),
    "many_same_compound": subgroup_report("many_same_compound", many_compound_mask),
    "fewer_same_compound": subgroup_report("fewer_same_compound", ~many_compound_mask),
}

final_model = make_lgb(SEED + 999)
final_model.fit(
    X_train_meta_lgb[meta_features],
    y,
    categorical_feature=cat_features,
)
test_pred = final_model.predict_proba(X_test_meta_lgb[meta_features])[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

sub_id_col = sample.columns[0]
sub_target_col = sample.columns[1]
submission = pd.DataFrame(
    {
        sub_id_col: sample[sub_id_col].to_numpy(),
        sub_target_col: test_pred,
    }
)

submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        "row": np.arange(len(y)),
        "target": y,
        "prediction": oof_meta,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000314"],
    "validation_metric": "grouped_5_fold_roc_auc",
    "cv_auc": clean_float(meta_auc),
    "baseline_cv_auc": clean_float(baseline_auc),
    "ranker_meta_delta": clean_float(
        meta_auc - baseline_auc
        if not (pd.isna(meta_auc) or pd.isna(baseline_auc))
        else np.nan
    ),
    "ranker_raw_oof_auc": clean_float(ranker_raw_auc),
    "ranker_percentile_oof_auc": clean_float(ranker_pct_auc),
    "many_same_compound_threshold": clean_float(many_threshold),
    "subgroups": subgroups,
    "saved_files": [
        "./working/submission.csv",
        "./working/oof_predictions.csv.gz",
        "./working/test_predictions.csv.gz",
    ],
}
print(json.dumps(review, indent=2))
