import os
import re
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
INPUT_DIR = "./input"
WORKING_DIR = "./working"
N_JOBS = max(1, min(8, os.cpu_count() or 1))


def clip_prob(p, eps=1e-6):
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def logit(p):
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def expit(z):
    z = np.clip(np.asarray(z, dtype=float), -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


class MonotoneBetaCalibrator:
    def __init__(self):
        self.params_ = np.array([1.0, 1.0, 0.0], dtype=float)
        self.fallback_reason_ = None

    def fit(self, p, y):
        p = clip_prob(p)
        y = np.asarray(y, dtype=float)
        f1 = np.log(p)
        f2 = -np.log1p(-p)

        try:
            from scipy.optimize import minimize

            def objective(theta):
                a, b, c = theta
                z = c + a * f1 + b * f2
                pred = expit(z)
                loss = np.mean(np.logaddexp(0.0, z) - y * z)
                reg = 1e-4 * ((a - 1.0) ** 2 + (b - 1.0) ** 2 + c**2)
                diff = pred - y
                grad = np.array(
                    [
                        np.mean(diff * f1) + 2e-4 * (a - 1.0),
                        np.mean(diff * f2) + 2e-4 * (b - 1.0),
                        np.mean(diff) + 2e-4 * c,
                    ]
                )
                return loss + reg, grad

            res = minimize(
                objective,
                self.params_,
                jac=True,
                method="L-BFGS-B",
                bounds=[(1e-6, 30.0), (1e-6, 30.0), (-30.0, 30.0)],
                options={"maxiter": 200},
            )
            if np.all(np.isfinite(res.x)):
                self.params_ = res.x.astype(float)
            if not res.success:
                self.fallback_reason_ = str(res.message)
        except Exception as exc:
            self.params_ = np.array([1.0, 1.0, 0.0], dtype=float)
            self.fallback_reason_ = f"identity fallback: {exc}"
        return self

    def transform(self, p):
        p = clip_prob(p)
        a, b, c = self.params_
        z = c + a * np.log(p) - b * np.log1p(-p)
        return clip_prob(expit(z))


def feature_frame(df, heavy=False):
    X = df.drop(columns=[c for c in [ID_COL, TARGET] if c in df.columns]).copy()

    if "Year" in X.columns:
        X["YearCat"] = X["Year"].astype(str)
    if "Stint" in X.columns:
        X["StintCat"] = X["Stint"].astype(str)

    if heavy:
        eps = 1e-3

        if "Compound" in X.columns:
            compound = X["Compound"].astype(str)
            hardness = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
            X["CompoundHardness"] = compound.map(hardness).fillna(0).astype(float)
            X["IsWetCompound"] = compound.isin(["INTERMEDIATE", "WET"]).astype(int)

        if {"LapNumber", "RaceProgress"}.issubset(X.columns):
            lap = X["LapNumber"].astype(float)
            progress = X["RaceProgress"].astype(float).clip(lower=eps)
            total_laps = (
                (lap / progress)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(60)
                .clip(1, 120)
            )
            X["EstimatedTotalLaps"] = total_laps
            X["LapsRemaining"] = (total_laps - lap).clip(-5, 120)
            X["RaceProgressSq"] = progress**2

        if {"TyreLife", "LapNumber"}.issubset(X.columns):
            tyre = X["TyreLife"].astype(float)
            lap = X["LapNumber"].astype(float)
            X["TyreLifeToLapRatio"] = tyre / (lap + 1.0)
            X["LapMinusTyreLife"] = lap - tyre

        if {"TyreLife", "EstimatedTotalLaps"}.issubset(X.columns):
            X["TyreLifeRaceFrac"] = X["TyreLife"].astype(float) / (
                X["EstimatedTotalLaps"].astype(float) + 1.0
            )

        if {"Cumulative_Degradation", "TyreLife"}.issubset(X.columns):
            tyre = X["TyreLife"].astype(float)
            deg = X["Cumulative_Degradation"].astype(float)
            X["DegPerTyreLife"] = deg / (tyre + 1.0)
            X["DegTimesTyreLife"] = deg * tyre

        if "LapTime_Delta" in X.columns:
            delta = X["LapTime_Delta"].astype(float)
            X["AbsLapTimeDelta"] = delta.abs()

        if {"LapTime (s)", "LapTime_Delta"}.issubset(X.columns):
            X["LapDeltaRatio"] = X["LapTime_Delta"].astype(float) / (
                X["LapTime (s)"].astype(float) + eps
            )

        if {"PitStop", "TyreLife"}.issubset(X.columns):
            X["PitStopTimesTyreLife"] = X["PitStop"].astype(float) * X[
                "TyreLife"
            ].astype(float)

        if {"Stint", "TyreLife"}.issubset(X.columns):
            X["StintTimesTyreLife"] = X["Stint"].astype(float) * X["TyreLife"].astype(
                float
            )

        if "Position_Change" in X.columns:
            X["AbsPositionChange"] = X["Position_Change"].astype(float).abs()

    return X


def sanitize_feature_names(X):
    X = X.copy()
    seen = {}
    names = []
    for col in X.columns:
        name = re.sub(r"[^0-9A-Za-z_]+", "_", str(col)).strip("_") or "feature"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    X.columns = names
    return X


def align_lgb_categories(train_X, test_X):
    train_X = train_X.copy()
    test_X = test_X.copy()
    cat_cols = train_X.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        tr = train_X[col].where(train_X[col].notna(), "__NA__").astype(str)
        te = test_X[col].where(test_X[col].notna(), "__NA__").astype(str)
        cats = pd.Index(pd.concat([tr, te], ignore_index=True).unique())
        train_X[col] = pd.Categorical(tr, categories=cats)
        test_X[col] = pd.Categorical(te, categories=cats)

    return train_X, test_X, cat_cols


def stringify_categories(train_X, test_X):
    train_X = train_X.copy()
    test_X = test_X.copy()
    cat_cols = train_X.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in cat_cols:
        train_X[col] = train_X[col].where(train_X[col].notna(), "__NA__").astype(str)
        test_X[col] = test_X[col].where(test_X[col].notna(), "__NA__").astype(str)

    return train_X, test_X, cat_cols


def make_ohe():
    try:
        return OneHotEncoder(
            handle_unknown="ignore", min_frequency=5, sparse_output=True
        )
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_splits(train_df, y):
    groups = train_df["Year"].astype(str) + "_" + train_df["Race"].astype(str)
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=SEED
        )
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
        if all(len(np.unique(y[val])) == 2 for _, val in splits):
            return splits, "StratifiedGroupKFold(Year_Race)"
    except Exception:
        pass

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros(len(y)), y)), "StratifiedKFold"


def predict_lgb(model, X):
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is not None and best_iter > 0:
        return model.predict_proba(X, num_iteration=best_iter)[:, 1]
    return model.predict_proba(X)[:, 1]


def count_ties(pred):
    return int(len(pred) - np.unique(np.round(pred, 12)).size)


def new_meta_model():
    return LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000)


os.makedirs(WORKING_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
scale_pos_weight = float((len(y) - y.sum()) / max(y.sum(), 1))

splits, split_name = make_splits(train, y)
print(f"Using {split_name} with {len(splits)} folds")
print(f"Positive rate: {y.mean():.6f}")

X_lgb_train = sanitize_feature_names(feature_frame(train, heavy=True))
X_lgb_test = sanitize_feature_names(feature_frame(test, heavy=True))
X_lgb_train, X_lgb_test, lgb_cat_cols = align_lgb_categories(X_lgb_train, X_lgb_test)

X_cb_train, X_cb_test, cb_cat_cols = stringify_categories(
    feature_frame(train, heavy=True),
    feature_frame(test, heavy=True),
)
cb_cat_idx = [X_cb_train.columns.get_loc(c) for c in cb_cat_cols]
cb_test_pool = Pool(X_cb_test, cat_features=cb_cat_idx)

X_lin_train, X_lin_test, lin_cat_cols = stringify_categories(
    feature_frame(train, heavy=False),
    feature_frame(test, heavy=False),
)
lin_num_cols = [c for c in X_lin_train.columns if c not in lin_cat_cols]

base_names = ["lgb_heavy", "cat_native", "linear_raw"]
oof_probs = {name: np.zeros(len(train), dtype=float) for name in base_names}
test_probs = {name: np.zeros(len(test), dtype=float) for name in base_names}

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    print(f"Fold {fold}/{len(splits)}")

    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        min_child_samples=80,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED + fold,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    lgb_model.fit(
        X_lgb_train.iloc[tr_idx],
        y[tr_idx],
        eval_set=[(X_lgb_train.iloc[va_idx], y[va_idx])],
        eval_metric="auc",
        categorical_feature=lgb_cat_cols,
        callbacks=[lgb.early_stopping(70, verbose=False), lgb.log_evaluation(0)],
    )
    pred = predict_lgb(lgb_model, X_lgb_train.iloc[va_idx])
    oof_probs["lgb_heavy"][va_idx] = pred
    test_probs["lgb_heavy"] += predict_lgb(lgb_model, X_lgb_test) / len(splits)
    print(f"  lgb_heavy auc: {roc_auc_score(y[va_idx], pred):.6f}")

    cb_model = CatBoostClassifier(
        iterations=650,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=6.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED + fold,
        auto_class_weights="Balanced",
        bootstrap_type="Bernoulli",
        subsample=0.85,
        od_type="Iter",
        od_wait=70,
        allow_writing_files=False,
        verbose=False,
        thread_count=N_JOBS,
    )
    cb_train_pool = Pool(X_cb_train.iloc[tr_idx], y[tr_idx], cat_features=cb_cat_idx)
    cb_valid_pool = Pool(X_cb_train.iloc[va_idx], y[va_idx], cat_features=cb_cat_idx)
    cb_model.fit(cb_train_pool, eval_set=cb_valid_pool, use_best_model=True)
    pred = cb_model.predict_proba(cb_valid_pool)[:, 1]
    oof_probs["cat_native"][va_idx] = pred
    test_probs["cat_native"] += cb_model.predict_proba(cb_test_pool)[:, 1] / len(splits)
    print(f"  cat_native auc: {roc_auc_score(y[va_idx], pred):.6f}")

    transformers = []
    if lin_num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                lin_num_cols,
            )
        )
    if lin_cat_cols:
        transformers.append(("cat", make_ohe(), lin_cat_cols))

    linear_model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(transformers=transformers, sparse_threshold=0.3),
            ),
            (
                "clf",
                LogisticRegression(
                    C=0.7,
                    solver="saga",
                    penalty="l2",
                    max_iter=250,
                    class_weight="balanced",
                    random_state=SEED + fold,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )
    linear_model.fit(X_lin_train.iloc[tr_idx], y[tr_idx])
    pred = linear_model.predict_proba(X_lin_train.iloc[va_idx])[:, 1]
    oof_probs["linear_raw"][va_idx] = pred
    test_probs["linear_raw"] += linear_model.predict_proba(X_lin_test)[:, 1] / len(
        splits
    )
    print(f"  linear_raw auc: {roc_auc_score(y[va_idx], pred):.6f}")

base_oof = np.column_stack([clip_prob(oof_probs[name]) for name in base_names])
base_test = np.column_stack([clip_prob(test_probs[name]) for name in base_names])
meta_oof_X = np.column_stack([logit(base_oof[:, i]) for i in range(base_oof.shape[1])])
meta_test_X = np.column_stack(
    [logit(base_test[:, i]) for i in range(base_test.shape[1])]
)

raw_blend_oof = clip_prob(base_oof.mean(axis=1))
raw_blend_test = clip_prob(base_test.mean(axis=1))

sigmoid_stack_oof = np.zeros(len(train), dtype=float)
beta_stack_oof = np.zeros(len(train), dtype=float)
beta_notes = []

for tr_idx, va_idx in splits:
    meta = new_meta_model()
    meta.fit(meta_oof_X[tr_idx], y[tr_idx])
    p_tr = meta.predict_proba(meta_oof_X[tr_idx])[:, 1]
    p_va = meta.predict_proba(meta_oof_X[va_idx])[:, 1]
    sigmoid_stack_oof[va_idx] = clip_prob(p_va)

    beta = MonotoneBetaCalibrator().fit(p_tr, y[tr_idx])
    beta_stack_oof[va_idx] = beta.transform(p_va)
    if beta.fallback_reason_:
        beta_notes.append(beta.fallback_reason_)

final_meta = new_meta_model()
final_meta.fit(meta_oof_X, y)
sigmoid_stack_test = clip_prob(final_meta.predict_proba(meta_test_X)[:, 1])
sigmoid_stack_train_fit = clip_prob(final_meta.predict_proba(meta_oof_X)[:, 1])

final_beta = MonotoneBetaCalibrator().fit(sigmoid_stack_train_fit, y)
beta_stack_test = final_beta.transform(sigmoid_stack_test)
if final_beta.fallback_reason_:
    beta_notes.append(final_beta.fallback_reason_)

candidate_oof = {
    "raw_blend": raw_blend_oof,
    "sigmoid_stack": sigmoid_stack_oof,
    "beta_stack": beta_stack_oof,
}
candidate_test = {
    "raw_blend": raw_blend_test,
    "sigmoid_stack": sigmoid_stack_test,
    "beta_stack": beta_stack_test,
}

base_metrics = {
    name: float(roc_auc_score(y, clip_prob(oof_probs[name]))) for name in base_names
}
stack_metrics = []
for name, pred in candidate_oof.items():
    stack_metrics.append(
        {
            "name": name,
            "auc": float(roc_auc_score(y, pred)),
            "ties_rounded_12dp": count_ties(pred),
        }
    )

preference = {"raw_blend": 0, "sigmoid_stack": 1, "beta_stack": 2}
best = sorted(
    stack_metrics, key=lambda d: (d["auc"], preference[d["name"]]), reverse=True
)[0]
best_name = best["name"]
best_oof = clip_prob(candidate_oof[best_name])
best_test = clip_prob(candidate_test[best_name])
best_auc = float(roc_auc_score(y, best_oof))

if np.array_equal(sample[ID_COL].values, test[ID_COL].values):
    ordered_test_pred = best_test
else:
    ordered_test_pred = (
        pd.Series(best_test, index=test[ID_COL]).reindex(sample[ID_COL]).values
    )
    if np.isnan(ordered_test_pred).any():
        raise ValueError("Could not align test predictions to sample_submission ids.")

submission = sample[[ID_COL]].copy()
submission[TARGET] = clip_prob(ordered_test_pred)

submission_path = os.path.join(WORKING_DIR, "submission.csv")
oof_path = os.path.join(WORKING_DIR, "oof_predictions.csv.gz")
test_pred_path = os.path.join(WORKING_DIR, "test_predictions.csv.gz")
report_path = os.path.join(WORKING_DIR, "result_review.json")

submission.to_csv(submission_path, index=False, float_format="%.10f")
submission.to_csv(test_pred_path, index=False, compression="gzip", float_format="%.10f")

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": best_oof,
    }
).to_csv(oof_path, index=False, compression="gzip", float_format="%.10f")

report = {
    "evaluation_metric": "roc_auc",
    "cv_strategy": split_name,
    "base_oof_auc": base_metrics,
    "final_candidate_auc_and_ties": stack_metrics,
    "selected_model": best_name,
    "selected_oof_auc": best_auc,
    "submission_path": submission_path,
    "oof_predictions_path": oof_path,
    "test_predictions_path": test_pred_path,
    "beta_notes": sorted(set(beta_notes))[:5],
    "research_hypotheses_llm_claimed_used": ["000810"],
}

with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"OOF ROC AUC ({best_name}): {best_auc:.6f}")
print(json.dumps(report, indent=2))
