import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

TARGET = "PitNextLap"
ID_COL = "id"
RANDOM_STATE = 20260524
HYPOTHESES_USED = ["000530"]

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values


def add_rule_weather_features(df):
    df = df.copy()
    compound = df["Compound"].astype(str).str.upper()

    df["is_wet_regime"] = compound.isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    df["is_intermediate"] = (compound == "INTERMEDIATE").astype(np.int8)
    df["is_full_wet"] = (compound == "WET").astype(np.int8)
    df["is_dry_compound"] = compound.isin(["SOFT", "MEDIUM", "HARD"]).astype(np.int8)

    df["compound_soft"] = (compound == "SOFT").astype(np.int8)
    df["compound_medium"] = (compound == "MEDIUM").astype(np.int8)
    df["compound_hard"] = (compound == "HARD").astype(np.int8)

    laps_elapsed = df["LapNumber"].astype(float)
    tyre_life = df["TyreLife"].astype(float)
    race_progress = df["RaceProgress"].clip(0.001, 1.0).astype(float)
    est_total_laps = (laps_elapsed / race_progress).replace([np.inf, -np.inf], np.nan)
    est_total_laps = est_total_laps.fillna(laps_elapsed.max())
    laps_remaining = (est_total_laps - laps_elapsed).clip(lower=0)

    df["est_total_laps"] = est_total_laps
    df["laps_remaining"] = laps_remaining
    df["tyre_life_frac_of_race"] = tyre_life / np.maximum(est_total_laps, 1)
    df["stint_frac_of_race_left"] = laps_remaining / np.maximum(est_total_laps, 1)

    nominal_life = np.select(
        [
            compound == "SOFT",
            compound == "MEDIUM",
            compound == "HARD",
            compound == "INTERMEDIATE",
            compound == "WET",
        ],
        [18.0, 28.0, 38.0, 24.0, 32.0],
        default=28.0,
    )
    df["nominal_life"] = nominal_life
    df["tyre_life_over_nominal"] = tyre_life / nominal_life
    df["current_compound_can_finish"] = (
        laps_remaining <= np.maximum(nominal_life - tyre_life, 0)
    ).astype(np.int8)

    fresh_alt_life = np.select(
        [compound == "SOFT", compound == "MEDIUM", compound == "HARD"],
        [38.0, 38.0, 28.0],
        default=32.0,
    )
    df["fresh_alt_can_finish"] = (laps_remaining <= fresh_alt_life).astype(np.int8)

    df["likely_two_dry_compounds_satisfied"] = (
        (df["is_dry_compound"] == 1)
        & ((df["Stint"].astype(float) >= 2) | (df["PitStop"].astype(float) > 0))
    ).astype(np.int8)

    df["dry_rule_pressure"] = (
        (df["is_dry_compound"] == 1)
        & (df["likely_two_dry_compounds_satisfied"] == 0)
        & (df["RaceProgress"].astype(float) > 0.55)
    ).astype(np.int8)

    df["must_stop_pressure"] = (
        df["dry_rule_pressure"]
        * (1 + (df["RaceProgress"].astype(float) > 0.75).astype(np.int8))
        * (1 + (df["current_compound_can_finish"] == 0).astype(np.int8))
    )

    df["wet_crossover_pressure"] = df["is_wet_regime"] * (
        (df["tyre_life_over_nominal"] > 0.75).astype(np.int8)
        + (df["LapTime_Delta"].astype(float) > 2.0).astype(np.int8)
        + (df["Cumulative_Degradation"].astype(float) > 80.0).astype(np.int8)
    )

    df["degradation_per_tyre_lap"] = df["Cumulative_Degradation"].astype(
        float
    ) / np.maximum(tyre_life, 1)
    df["lap_delta_x_tyre_age"] = df["LapTime_Delta"].astype(float) * np.log1p(tyre_life)
    df["progress_x_stint"] = df["RaceProgress"].astype(float) * df["Stint"].astype(
        float
    )
    return df


train_fe = add_rule_weather_features(train.drop(columns=[TARGET]))
test_fe = add_rule_weather_features(test)

feature_cols = [c for c in train_fe.columns if c != ID_COL]
cat_cols = [c for c in feature_cols if train_fe[c].dtype == "object"]

X_all = train_fe[feature_cols].copy()
X_test = test_fe[feature_cols].copy()

if cat_cols:
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
    )
    X_all[cat_cols] = enc.fit_transform(X_all[cat_cols].astype(str))
    X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(-999)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(-999)

wet_mask_all = train_fe["is_wet_regime"].values.astype(bool)
wet_mask_test = test_fe["is_wet_regime"].values.astype(bool)

train_query_key = (
    train_fe["Year"].astype(str)
    + "\x1f"
    + train_fe["Race"].astype(str)
    + "\x1f"
    + train_fe["LapNumber"].astype(str)
).values

try:
    from lightgbm import LGBMClassifier, LGBMRanker

    Model = "lightgbm"
    RankerBackend = "lightgbm"
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier

    LGBMRanker = None
    Model = "sklearn_hgb"
    try:
        from catboost import CatBoostRanker, Pool

        RankerBackend = "catboost"
    except Exception:
        CatBoostRanker = None
        Pool = None
        RankerBackend = None


def make_model(regime):
    if Model == "lightgbm":
        params = dict(
            objective="binary",
            n_estimators=650 if regime == "dry" else 350,
            learning_rate=0.035 if regime == "dry" else 0.045,
            num_leaves=48 if regime == "dry" else 24,
            max_depth=-1,
            min_child_samples=80 if regime == "dry" else 25,
            subsample=0.88,
            colsample_bytree=0.88,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=max(1, os.cpu_count() or 1),
            verbose=-1,
            class_weight=None,
        )
        return LGBMClassifier(**params)

    return HistGradientBoostingClassifier(
        max_iter=300 if regime == "dry" else 180,
        learning_rate=0.045,
        l2_regularization=0.05,
        max_leaf_nodes=31 if regime == "dry" else 15,
        random_state=RANDOM_STATE,
    )


def make_ranker(fold):
    if RankerBackend == "lightgbm":
        return LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=320,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=45,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.03,
            reg_lambda=1.0,
            label_gain=[0, 1],
            random_state=RANDOM_STATE + fold,
            n_jobs=max(1, os.cpu_count() or 1),
            verbose=-1,
        )

    if RankerBackend == "catboost":
        return CatBoostRanker(
            loss_function="YetiRank",
            iterations=260,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=4.0,
            random_seed=RANDOM_STATE + fold,
            thread_count=max(1, os.cpu_count() or 1),
            allow_writing_files=False,
            verbose=False,
        )

    return None


def sorted_group_indices(idx):
    idx = np.asarray(idx)
    order = np.argsort(train_query_key[idx], kind="mergesort")
    sorted_idx = idx[order]
    _, group_sizes = np.unique(train_query_key[sorted_idx], return_counts=True)
    return sorted_idx, group_sizes.astype(int).tolist()


def fit_predict_ranker(tr_idx, val_idx, fold):
    prior = float(np.mean(y[tr_idx]))
    if RankerBackend is None or len(np.unique(y[tr_idx])) < 2:
        return np.full(len(val_idx), prior), np.full(len(test), prior)

    sorted_idx, group_sizes = sorted_group_indices(tr_idx)
    if len(group_sizes) == 0 or max(group_sizes) < 2:
        return np.full(len(val_idx), prior), np.full(len(test), prior)

    try:
        ranker = make_ranker(fold)
        if RankerBackend == "lightgbm":
            ranker.fit(X_all.iloc[sorted_idx], y[sorted_idx], group=group_sizes)
            val_pred = ranker.predict(X_all.iloc[val_idx])
            test_pred = ranker.predict(X_test)
        else:
            train_pool = Pool(
                X_all.iloc[sorted_idx],
                label=y[sorted_idx],
                group_id=train_query_key[sorted_idx],
            )
            ranker.fit(train_pool)
            val_pred = ranker.predict(Pool(X_all.iloc[val_idx]))
            test_pred = ranker.predict(Pool(X_test))

        return np.asarray(val_pred, dtype=float), np.asarray(test_pred, dtype=float)
    except Exception as exc:
        print(f"Fold {fold} ranker fallback used: {exc}")
        return np.full(len(val_idx), prior), np.full(len(test), prior)


def fit_predict_regime(X_tr, y_tr, X_val, X_te, regime):
    if len(np.unique(y_tr)) < 2:
        p = float(np.mean(y_tr)) if len(y_tr) else 0.0
        return np.full(len(X_val), p), np.full(len(X_te), p)

    model = make_model(regime)
    model.fit(X_tr, y_tr)

    if hasattr(model, "predict_proba"):
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_te)[:, 1]
    else:
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_te)

    return np.clip(val_pred, 1e-6, 1 - 1e-6), np.clip(test_pred, 1e-6, 1 - 1e-6)


def rank01(values):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    fill = float(np.median(arr[finite])) if finite.any() else 0.0
    arr = np.where(finite, arr, fill)

    if len(arr) <= 1:
        return np.full(len(arr), 0.5)
    if np.nanmax(arr) == np.nanmin(arr):
        return np.full(len(arr), 0.5)

    ranks = pd.Series(arr).rank(method="average").to_numpy(dtype=float)
    return (ranks - 1.0) / (len(arr) - 1.0)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_binary = np.zeros(len(train), dtype=float)
oof_ranker = np.zeros(len(train), dtype=float)
test_pred_binary_folds = np.zeros((len(test), skf.n_splits), dtype=float)
test_pred_ranker_folds = np.zeros((len(test), skf.n_splits), dtype=float)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all, y), 1):
    rank_val_pred, rank_test_pred = fit_predict_ranker(tr_idx, val_idx, fold)
    oof_ranker[val_idx] = rank_val_pred
    test_pred_ranker_folds[:, fold - 1] = rank_test_pred

    fold_test_pred = np.zeros(len(test), dtype=float)

    for regime, is_wet in [("dry", False), ("wet", True)]:
        tr_reg = tr_idx[wet_mask_all[tr_idx] == is_wet]
        val_reg = val_idx[wet_mask_all[val_idx] == is_wet]
        te_reg = np.where(wet_mask_test == is_wet)[0]

        if len(val_reg) == 0:
            continue

        val_pred, te_pred = fit_predict_regime(
            X_all.iloc[tr_reg],
            y[tr_reg],
            X_all.iloc[val_reg],
            X_test.iloc[te_reg] if len(te_reg) else X_test.iloc[:0],
            regime,
        )
        oof_binary[val_reg] = val_pred
        if len(te_reg):
            fold_test_pred[te_reg] = te_pred

    missing_test = fold_test_pred == 0
    if missing_test.any():
        fold_test_pred[missing_test] = float(np.mean(y[tr_idx]))

    test_pred_binary_folds[:, fold - 1] = fold_test_pred

    fold_binary_auc = roc_auc_score(y[val_idx], oof_binary[val_idx])
    fold_ranker_auc = roc_auc_score(y[val_idx], oof_ranker[val_idx])
    print(
        f"Fold {fold} ROC AUC: binary={fold_binary_auc:.6f} "
        f"ranker_sidecar={fold_ranker_auc:.6f}"
    )

base_cv_auc = roc_auc_score(y, oof_binary)
ranker_cv_auc = roc_auc_score(y, oof_ranker)

test_pred_binary = np.clip(test_pred_binary_folds.mean(axis=1), 1e-6, 1 - 1e-6)
test_pred_ranker = test_pred_ranker_folds.mean(axis=1)

binary_oof_rank = rank01(oof_binary)
ranker_oof_rank = rank01(oof_ranker)
binary_test_rank = rank01(test_pred_binary)
ranker_test_rank = rank01(test_pred_ranker)

best_weight = 0.0
best_auc = -np.inf
for weight in np.linspace(0.0, 0.30, 16):
    blend_oof = (1.0 - weight) * binary_oof_rank + weight * ranker_oof_rank
    auc = roc_auc_score(y, blend_oof)
    if auc > best_auc:
        best_auc = auc
        best_weight = float(weight)

oof_final = np.clip(
    (1.0 - best_weight) * binary_oof_rank + best_weight * ranker_oof_rank,
    1e-6,
    1 - 1e-6,
)
test_pred_final = np.clip(
    (1.0 - best_weight) * binary_test_rank + best_weight * ranker_test_rank,
    1e-6,
    1 - 1e-6,
)

pd.DataFrame(
    {"row": np.arange(len(train)), "target": y, "prediction": oof_final}
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

test_pred_df = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred_final})
test_pred_df.to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)
test_pred_df.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

print(f"Base 5-fold CV ROC AUC: {base_cv_auc:.6f}")
print(f"Ranker sidecar OOF ROC AUC: {ranker_cv_auc:.6f}")
print(f"Best rank-average blend weight: {best_weight:.3f}")
print(f"5-fold CV ROC AUC: {best_auc:.6f}")
print(
    json.dumps(
        {
            "metric": "roc_auc",
            "cv_score": float(best_auc),
            "base_cv_score": float(base_cv_auc),
            "ranker_sidecar_cv_score": float(ranker_cv_auc),
            "ranker_backend": RankerBackend,
            "rank_blend_weight": float(best_weight),
            "research_hypotheses_llm_claimed_used": HYPOTHESES_USED,
            "submission_path": os.path.join(WORKING_DIR, "submission.csv"),
        }
    )
)
