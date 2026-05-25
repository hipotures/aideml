import os
import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

warnings.filterwarnings("ignore")

SEED = 2026
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
INPUT = Path("./input")
WORK = Path("./working")
WORK.mkdir(parents=True, exist_ok=True)


class RankTransformer:
    def fit(self, p, y=None):
        self.sorted_ = np.sort(np.asarray(p, dtype=np.float64))
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=np.float64)
        return np.searchsorted(self.sorted_, p, side="right") / (
            len(self.sorted_) + 1.0
        )


class BetaCalibrator:
    def fit(self, p, y):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        y = np.asarray(y, dtype=int)
        self.constant_ = None
        if len(np.unique(y)) < 2:
            self.constant_ = float(np.mean(y))
            return self
        x = np.column_stack([np.log(p), np.log1p(-p)])
        self.model_ = LogisticRegression(C=1000.0, max_iter=1000, solver="lbfgs")
        self.model_.fit(x, y)
        return self

    def predict(self, p):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
        if self.constant_ is not None:
            return np.full(len(p), self.constant_, dtype=np.float64)
        x = np.column_stack([np.log(p), np.log1p(-p)])
        return self.model_.predict_proba(x)[:, 1]


class VennAbersStyleCalibrator:
    def __init__(self, n_bins=80):
        self.n_bins = n_bins

    def fit(self, p, y):
        p = np.asarray(p, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        order = np.argsort(p)
        p_sorted = p[order]
        y_sorted = y[order]
        bins = np.array_split(
            np.arange(len(p_sorted)), min(self.n_bins, max(2, len(p_sorted) // 200))
        )

        centers, probs, weights = [], [], []
        for idx in bins:
            if len(idx) == 0:
                continue
            n = len(idx)
            pos = float(y_sorted[idx].sum())
            lower = pos / (n + 1.0)
            upper = (pos + 1.0) / (n + 1.0)
            va_prob = upper / (1.0 - lower + upper)
            centers.append(float(np.median(p_sorted[idx])))
            probs.append(float(np.clip(va_prob, 1e-6, 1 - 1e-6)))
            weights.append(float(n))

        centers = np.asarray(centers)
        probs = np.asarray(probs)
        weights = np.asarray(weights)

        if len(np.unique(centers)) < 2:
            self.constant_ = float(np.average(probs, weights=weights))
            self.iso_ = None
        else:
            self.constant_ = None
            self.iso_ = IsotonicRegression(
                y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip"
            )
            self.iso_.fit(centers, probs, sample_weight=weights)
        return self

    def predict(self, p):
        p = np.asarray(p, dtype=np.float64)
        if self.iso_ is None:
            return np.full(len(p), self.constant_, dtype=np.float64)
        return np.clip(self.iso_.predict(p), 1e-6, 1 - 1e-6)


def make_features(df):
    df = df.copy()
    eps = 1e-6
    df["TyreLife_x_Progress"] = df["TyreLife"] * df["RaceProgress"]
    df["TyreLife_per_Lap"] = df["TyreLife"] / (df["LapNumber"] + eps)
    df["Deg_per_TyreLife"] = df["Cumulative_Degradation"] / (df["TyreLife"] + eps)
    df["LapTime_per_Progress"] = df["LapTime (s)"] / (df["RaceProgress"] + eps)
    df["Abs_Position_Change"] = df["Position_Change"].abs()
    df["IsWetCompound"] = df["Compound"].isin(["WET", "INTERMEDIATE"]).astype(int)
    df["LateRace_TyreLife"] = df["RaceProgress"] * df["TyreLife"]
    return df


def get_splits(x, y, groups):
    if StratifiedGroupKFold is not None:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        return list(cv.split(x, y, groups))
    cv = GroupKFold(n_splits=N_SPLITS)
    return list(cv.split(x, y, groups))


def train_lgbm(x_tr, y_tr, x_va, y_va, x_te, cat_cols, variant, seed):
    import lightgbm as lgb

    if variant == "full":
        params = dict(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
        )
    else:
        params = dict(
            n_estimators=700,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            min_child_samples=120,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=0.10,
            reg_lambda=3.0,
        )

    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=max(1, min(8, os.cpu_count() or 1)),
        verbose=-1,
        **params,
    )
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_va, y_va)],
        eval_metric="auc",
        categorical_feature=[c for c in cat_cols if c in x_tr.columns],
        callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(0)],
    )
    va = model.predict_proba(x_va)[:, 1]
    te = model.predict_proba(x_te)[:, 1]
    return va, te


def train_catboost(x_tr, y_tr, x_va, y_va, x_te, cat_cols, seed):
    from catboost import CatBoostClassifier, Pool

    cat_idx = [x_tr.columns.get_loc(c) for c in cat_cols if c in x_tr.columns]
    tr_pool = Pool(x_tr, y_tr, cat_features=cat_idx)
    va_pool = Pool(x_va, y_va, cat_features=cat_idx)
    te_pool = Pool(x_te, cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=700,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        thread_count=max(1, min(8, os.cpu_count() or 1)),
        verbose=False,
    )
    model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
    va = model.predict_proba(va_pool)[:, 1]
    te = model.predict_proba(te_pool)[:, 1]
    return va, te


def transformed_matrix(
    kind, base, y=None, fit_idx=None, apply_idx=None, test_base=None
):
    if fit_idx is None:
        fit_idx = np.arange(base.shape[0])
    if apply_idx is None:
        apply_idx = np.arange(base.shape[0])

    out = np.zeros((len(apply_idx), base.shape[1]), dtype=np.float32)
    test_out = None if test_base is None else np.zeros_like(test_base, dtype=np.float32)

    for j in range(base.shape[1]):
        if kind == "rank":
            cal = RankTransformer().fit(base[fit_idx, j])
        elif kind == "beta":
            cal = BetaCalibrator().fit(base[fit_idx, j], y[fit_idx])
        elif kind == "venn_abers":
            cal = VennAbersStyleCalibrator().fit(base[fit_idx, j], y[fit_idx])
        else:
            raise ValueError(kind)

        out[:, j] = cal.predict(base[apply_idx, j])
        if test_base is not None:
            test_out[:, j] = cal.predict(test_base[:, j])

    return out, test_out


train = pd.read_csv(INPUT / "train.csv.gz")
test = pd.read_csv(INPUT / "test.csv.gz")
sample = pd.read_csv(INPUT / "sample_submission.csv.gz")

y = train[TARGET].astype(int).to_numpy()
groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)

train_features = train.drop(columns=[TARGET])
all_features = pd.concat([train_features, test], axis=0, ignore_index=True)
all_features = make_features(all_features)

cat_cols = all_features.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    all_features[c] = all_features[c].astype(str).astype("category")

feature_cols = [c for c in all_features.columns if c != ID_COL]
x_all = all_features[feature_cols]
x = x_all.iloc[: len(train)].reset_index(drop=True)
x_test = x_all.iloc[len(train) :].reset_index(drop=True)

tyre_cols = [
    c
    for c in [
        "Compound",
        "Race",
        "Year",
        "LapNumber",
        "RaceProgress",
        "Stint",
        "TyreLife",
        "PitStop",
        "Cumulative_Degradation",
        "LapTime_Delta",
        "Position",
        "TyreLife_x_Progress",
        "TyreLife_per_Lap",
        "Deg_per_TyreLife",
        "IsWetCompound",
        "LateRace_TyreLife",
    ]
    if c in x.columns
]

splits = get_splits(x, y, groups)
model_specs = [
    ("lgb_full", "lgb", feature_cols, "full"),
    ("lgb_tyre", "lgb", tyre_cols, "tyre"),
]

try:
    import catboost  # noqa: F401

    model_specs.append(("cat_full", "cat", feature_cols, "full"))
except Exception:
    pass

oof_base = np.zeros((len(train), len(model_specs)), dtype=np.float32)
test_base = np.zeros((len(test), len(model_specs)), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    print(f"Training fold {fold}/{N_SPLITS}")
    for m, (name, family, cols, variant) in enumerate(model_specs):
        x_tr, x_va, x_te = x.iloc[tr_idx][cols], x.iloc[va_idx][cols], x_test[cols]
        fold_cat_cols = [c for c in cat_cols if c in cols]

        if family == "lgb":
            va_pred, te_pred = train_lgbm(
                x_tr,
                y[tr_idx],
                x_va,
                y[va_idx],
                x_te,
                fold_cat_cols,
                variant,
                SEED + fold + m,
            )
        else:
            x_tr_cb = x_tr.copy()
            x_va_cb = x_va.copy()
            x_te_cb = x_te.copy()
            for c in fold_cat_cols:
                x_tr_cb[c] = x_tr_cb[c].astype(str)
                x_va_cb[c] = x_va_cb[c].astype(str)
                x_te_cb[c] = x_te_cb[c].astype(str)
            va_pred, te_pred = train_catboost(
                x_tr_cb,
                y[tr_idx],
                x_va_cb,
                y[va_idx],
                x_te_cb,
                fold_cat_cols,
                SEED + fold + m,
            )

        oof_base[va_idx, m] = va_pred
        test_base[:, m] += te_pred / N_SPLITS
        print(f"  {name}: fold_auc={roc_auc_score(y[va_idx], va_pred):.6f}")

    gc.collect()

base_auc = roc_auc_score(y, np.mean(oof_base, axis=1))
print(f"Base mean OOF ROC AUC: {base_auc:.6f}")

variants = ["rank", "beta", "venn_abers"]
variant_results = {}

for kind in variants:
    meta_oof = np.zeros(len(train), dtype=np.float32)
    fold_aucs = []

    for tr_idx, va_idx in splits:
        x_tr_meta, _ = transformed_matrix(
            kind, oof_base, y=y, fit_idx=tr_idx, apply_idx=tr_idx
        )
        x_va_meta, _ = transformed_matrix(
            kind, oof_base, y=y, fit_idx=tr_idx, apply_idx=va_idx
        )

        blender = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        blender.fit(x_tr_meta, y[tr_idx])
        pred = blender.predict_proba(x_va_meta)[:, 1]
        meta_oof[va_idx] = pred
        fold_aucs.append(roc_auc_score(y[va_idx], pred))

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    stability_score = mean_auc - 0.25 * std_auc
    variant_results[kind] = {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "stability_score": stability_score,
        "fold_aucs": fold_aucs,
        "oof": meta_oof,
    }
    print(
        f"{kind}: grouped_cv_auc={mean_auc:.6f}, "
        f"fold_std_public_lb_proxy={std_auc:.6f}, stability_score={stability_score:.6f}"
    )

selected = max(variant_results, key=lambda k: variant_results[k]["stability_score"])
selected_oof = variant_results[selected]["oof"]
selected_auc = roc_auc_score(y, selected_oof)

full_meta_x, full_test_x = transformed_matrix(
    selected,
    oof_base,
    y=y,
    fit_idx=np.arange(len(train)),
    apply_idx=np.arange(len(train)),
    test_base=test_base,
)
final_blender = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
final_blender.fit(full_meta_x, y)
test_pred = final_blender.predict_proba(full_test_x)[:, 1]
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = sample.copy()
submission[TARGET] = test_pred
submission.to_csv(WORK / "submission.csv", index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": selected_oof,
    }
).to_csv(WORK / "oof_predictions.csv.gz", index=False, compression="gzip")

submission.to_csv(WORK / "test_predictions.csv.gz", index=False, compression="gzip")

review = {
    "research_hypotheses_llm_claimed_used": ["000737"],
    "metric": "grouped_5fold_roc_auc",
    "selected_calibration_input": selected,
    "selected_oof_auc": float(selected_auc),
    "selected_mean_fold_auc": variant_results[selected]["mean_auc"],
    "selected_fold_std": variant_results[selected]["std_auc"],
    "base_models": [m[0] for m in model_specs],
    "submission_path": str(WORK / "submission.csv"),
}
print(f"Selected blend input: {selected}")
print(f"Selected grouped 5-fold OOF ROC AUC: {selected_auc:.6f}")
print(json.dumps(review, sort_keys=True))
