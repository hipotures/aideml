import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv.gz")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv.gz")
SAMPLE_PATH = os.path.join(INPUT_DIR, "sample_submission.csv.gz")

TARGET = "PitNextLap"
ID_COL = "id"
N_SPLITS = 5
EMBARGO = 800

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values

cat_cols = ["Compound", "Driver", "Race"]
base_cols = [c for c in train.columns if c not in [TARGET, ID_COL]]

race_order = {
    r: i
    for i, r in enumerate(
        sorted(pd.concat([train["Race"], test["Race"]], axis=0).astype(str).unique())
    )
}


def add_features(df):
    out = df.copy()
    out["Race_ord"] = out["Race"].astype(str).map(race_order).fillna(-1).astype(int)
    out["LapFrac_x_TyreLife"] = out["RaceProgress"] * out["TyreLife"]
    out["TyreLife_sq"] = out["TyreLife"] ** 2
    out["LapNumber_sq"] = out["LapNumber"] ** 2
    out["Deg_per_TyreLife"] = out["Cumulative_Degradation"] / (out["TyreLife"] + 1.0)
    out["LateRace_oldTyre"] = out["RaceProgress"] * np.log1p(out["TyreLife"])
    out["Stint_TyreLife"] = out["Stint"] * out["TyreLife"]
    out["RecentPit_x_Stint"] = out["PitStop"] * out["Stint"]
    out["Abs_Position_Change"] = out["Position_Change"].abs()
    out["LapTime_Delta_abs"] = out["LapTime_Delta"].abs()
    out["Driver_Race"] = out["Driver"].astype(str) + "_" + out["Race"].astype(str)
    out["Race_Compound"] = out["Race"].astype(str) + "_" + out["Compound"].astype(str)
    return out


train_fe = add_features(train)
test_fe = add_features(test)

feature_cols = [c for c in train_fe.columns if c not in [TARGET, ID_COL]]
cat_features = [c for c in feature_cols if train_fe[c].dtype == "object"]
num_features = [c for c in feature_cols if c not in cat_features]

sort_cols = ["Year", "Race_ord", "LapNumber", ID_COL]
order = np.argsort(train_fe[sort_cols].to_records(index=False), kind="mergesort")
fold_blocks = np.array_split(order, N_SPLITS)


def make_lgbm(seed):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=48,
            max_depth=-1,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=90,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    except Exception:
        return None


def make_cat(seed):
    try:
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=700,
            learning_rate=0.045,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=max(1, os.cpu_count() or 1),
            l2_leaf_reg=6.0,
        )
    except Exception:
        return None


def make_logistic():
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                num_features,
            ),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", min_frequency=20),
                ),
                cat_features,
            ),
        ],
        sparse_threshold=0.3,
    )
    return make_pipeline(
        pre,
        LogisticRegression(
            C=1.0,
            max_iter=600,
            solver="saga",
            n_jobs=-1,
            class_weight="balanced",
            random_state=17,
        ),
    )


def fit_predict_model(model_name, tr_idx, va_idx, fit_all=False):
    if model_name == "lgbm":
        model = make_lgbm(100 + len(tr_idx) % 1000)
        if model is None:
            return None, None
        model.fit(
            train_fe.iloc[tr_idx][feature_cols],
            y[tr_idx],
            categorical_feature=[c for c in cat_features],
        )
        va_pred = (
            model.predict_proba(train_fe.iloc[va_idx][feature_cols])[:, 1]
            if va_idx is not None
            else None
        )
        te_pred = model.predict_proba(test_fe[feature_cols])[:, 1] if fit_all else None
        return va_pred, te_pred

    if model_name == "cat":
        model = make_cat(200 + len(tr_idx) % 1000)
        if model is None:
            return None, None
        cat_idx = [feature_cols.index(c) for c in cat_features]
        model.fit(
            train_fe.iloc[tr_idx][feature_cols],
            y[tr_idx],
            cat_features=cat_idx,
        )
        va_pred = (
            model.predict_proba(train_fe.iloc[va_idx][feature_cols])[:, 1]
            if va_idx is not None
            else None
        )
        te_pred = model.predict_proba(test_fe[feature_cols])[:, 1] if fit_all else None
        return va_pred, te_pred

    model = make_logistic()
    horizon_cols = [
        "Year",
        "Race",
        "Compound",
        "Race_ord",
        "LapNumber",
        "RaceProgress",
        "Stint",
        "TyreLife",
        "TyreLife_sq",
        "LapFrac_x_TyreLife",
        "Deg_per_TyreLife",
        "LateRace_oldTyre",
        "PitStop",
        "Position",
        "Position_Change",
        "LapTime_Delta",
        "Cumulative_Degradation",
    ]
    old_feature_cols = feature_cols[:]
    try:
        globals()["feature_cols"] = horizon_cols
        model.fit(train_fe.iloc[tr_idx][horizon_cols], y[tr_idx])
        va_pred = (
            model.predict_proba(train_fe.iloc[va_idx][horizon_cols])[:, 1]
            if va_idx is not None
            else None
        )
        te_pred = model.predict_proba(test_fe[horizon_cols])[:, 1] if fit_all else None
    finally:
        globals()["feature_cols"] = old_feature_cols
    return va_pred, te_pred


model_names = ["lgbm", "cat", "horizon_logit"]
oof_base = np.full((len(train_fe), len(model_names)), np.nan, dtype=np.float32)

for fold, va_idx in enumerate(fold_blocks, 1):
    first_pos = np.where(order == va_idx[0])[0][0]
    train_end = max(0, first_pos - EMBARGO)
    tr_idx = order[:train_end]

    if len(tr_idx) < 5000 or len(np.unique(y[tr_idx])) < 2:
        tr_idx = order[:first_pos]

    print(f"Fold {fold}: train={len(tr_idx)} valid={len(va_idx)} embargo={EMBARGO}")

    for m, name in enumerate(model_names):
        pred, _ = fit_predict_model(name, tr_idx, va_idx, fit_all=False)
        if pred is None:
            print(f"  {name}: unavailable, using fold prior")
            pred = np.full(len(va_idx), y[tr_idx].mean(), dtype=float)
        oof_base[va_idx, m] = np.clip(pred, 1e-6, 1 - 1e-6)
        print(f"  {name} fold AUC: {roc_auc_score(y[va_idx], oof_base[va_idx, m]):.6f}")

valid_mask = np.isfinite(oof_base).all(axis=1)
stacker = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=777)
stacker.fit(oof_base[valid_mask], y[valid_mask])
oof_stack = np.full(len(train_fe), np.nan, dtype=np.float32)
oof_stack[valid_mask] = stacker.predict_proba(oof_base[valid_mask])[:, 1]

cv_auc = roc_auc_score(y[valid_mask], oof_stack[valid_mask])
print(f"Purged walk-forward stacked ROC AUC: {cv_auc:.6f}")

pd.DataFrame(
    {
        "row": np.arange(len(train_fe))[valid_mask],
        "target": y[valid_mask],
        "prediction": oof_stack[valid_mask],
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

full_idx = np.arange(len(train_fe))
test_base = np.zeros((len(test_fe), len(model_names)), dtype=np.float32)

for m, name in enumerate(model_names):
    _, te_pred = fit_predict_model(name, full_idx, None, fit_all=True)
    if te_pred is None:
        te_pred = np.full(len(test_fe), y.mean(), dtype=float)
    test_base[:, m] = np.clip(te_pred, 1e-6, 1 - 1e-6)

test_pred = stacker.predict_proba(test_base)[:, 1]
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_pred})
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)
submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

review = {
    "research_hypotheses_llm_claimed_used": ["000694"],
    "validation_metric": "roc_auc",
    "validation_score": float(cv_auc),
    "cv_scheme": "5-fold purged walk-forward with sigmoid logistic OOF stacker",
}
print(json.dumps(review, sort_keys=True))
