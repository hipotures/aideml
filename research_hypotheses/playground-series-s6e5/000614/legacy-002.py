import os
import json
import warnings
import numpy as np
import pandas as pd

from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
HYPOTHESES = ["000614"]

os.makedirs("./working", exist_ok=True)

train = pd.read_csv("./input/train.csv.gz")
test = pd.read_csv("./input/test.csv.gz")
sample = pd.read_csv("./input/sample_submission.csv.gz")

y = train[TARGET].astype(int).values


def add_features(df):
    df = df.copy()
    df["TyreLife_Compound"] = (
        df["TyreLife"].round().astype(int).astype(str)
        + "_"
        + df["Compound"].astype(str)
    )
    df["Race_Year"] = df["Race"].astype(str) + "_" + df["Year"].astype(str)
    df["Driver_Race"] = df["Driver"].astype(str) + "_" + df["Race"].astype(str)
    df["TyreLife_x_Progress"] = df["TyreLife"] * df["RaceProgress"]
    df["Lap_x_Progress"] = df["LapNumber"] * df["RaceProgress"]
    df["Deg_per_TyreLife"] = df["Cumulative_Degradation"] / (df["TyreLife"] + 1.0)
    df["LapTime_Delta_abs"] = df["LapTime_Delta"].abs()
    df["IsFreshTyre"] = (df["TyreLife"] <= 2).astype(int)
    df["LateRace"] = (df["RaceProgress"] >= 0.75).astype(int)
    return df


train_fe = add_features(train.drop(columns=[TARGET]))
test_fe = add_features(test)

features = [c for c in train_fe.columns if c != ID_COL]
cat_cols = [
    "Driver",
    "Race",
    "Compound",
    "TyreLife_Compound",
    "Race_Year",
    "Driver_Race",
]
cat_cols = [c for c in cat_cols if c in features]
num_cols = [c for c in features if c not in cat_cols]

X = train_fe[features]
X_test = test_fe[features]

groups = train_fe["Race_Year"].astype(str).values
try:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X, y, groups))
except Exception:
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(splitter.split(X, y))


def make_ordinal_preprocess():
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


model_specs = []

try:
    from lightgbm import LGBMClassifier

    model_specs.append(
        (
            "lightgbm",
            LGBMClassifier(
                n_estimators=900,
                learning_rate=0.035,
                num_leaves=48,
                max_depth=-1,
                min_child_samples=80,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.0,
                objective="binary",
                metric="auc",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            "native_cat",
        )
    )
except Exception as e:
    print(f"LightGBM unavailable: {e}")

try:
    from xgboost import XGBClassifier

    model_specs.append(
        (
            "xgboost",
            make_pipeline(
                make_ordinal_preprocess(),
                XGBClassifier(
                    n_estimators=750,
                    learning_rate=0.035,
                    max_depth=5,
                    min_child_weight=8,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    reg_alpha=0.05,
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    enable_categorical=False,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
            "sklearn",
        )
    )
except Exception as e:
    print(f"XGBoost unavailable: {e}")

try:
    from catboost import CatBoostClassifier

    model_specs.append(
        (
            "catboost",
            CatBoostClassifier(
                iterations=700,
                learning_rate=0.045,
                depth=6,
                l2_leaf_reg=5.0,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            ),
            "catboost",
        )
    )
except Exception as e:
    print(f"CatBoost unavailable: {e}")

model_specs.append(
    (
        "extratrees",
        make_pipeline(
            make_ordinal_preprocess(),
            ExtraTreesClassifier(
                n_estimators=450,
                max_features="sqrt",
                min_samples_leaf=8,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        "sklearn",
    )
)


def prepare_native_cat(df):
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype("category")
    return out


def prepare_catboost(df):
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype(str).fillna("missing")
    return out


def fit_predict(spec_name, model, mode, x_tr, y_tr, x_va, x_te):
    if mode == "native_cat":
        x_tr_m = prepare_native_cat(x_tr)
        x_va_m = prepare_native_cat(x_va)
        x_te_m = prepare_native_cat(x_te)
        fit_kwargs = {}
        if spec_name == "lightgbm":
            fit_kwargs["categorical_feature"] = cat_cols
        model.fit(x_tr_m, y_tr, **fit_kwargs)
        va_pred = model.predict_proba(x_va_m)[:, 1]
        te_pred = model.predict_proba(x_te_m)[:, 1]
    elif mode == "catboost":
        x_tr_m = prepare_catboost(x_tr)
        x_va_m = prepare_catboost(x_va)
        x_te_m = prepare_catboost(x_te)
        cat_idx = [x_tr_m.columns.get_loc(c) for c in cat_cols]
        model.fit(x_tr_m, y_tr, cat_features=cat_idx)
        va_pred = model.predict_proba(x_va_m)[:, 1]
        te_pred = model.predict_proba(x_te_m)[:, 1]
    else:
        model.fit(x_tr, y_tr)
        va_pred = model.predict_proba(x_va)[:, 1]
        te_pred = model.predict_proba(x_te)[:, 1]
    return va_pred, te_pred


oof = {}
test_pred = {}
fold_scores = {name: [] for name, _, _ in model_specs}

for name, _, _ in model_specs:
    oof[name] = np.zeros(len(train_fe), dtype=np.float32)
    test_pred[name] = np.zeros(len(test_fe), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    x_tr, x_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    print(f"Fold {fold}/{N_SPLITS}")

    for name, base_model, mode in model_specs:
        model = clone(base_model)
        va_pred, te_pred = fit_predict(name, model, mode, x_tr, y_tr, x_va, X_test)
        oof[name][va_idx] = va_pred
        test_pred[name] += te_pred / N_SPLITS
        score = roc_auc_score(y_va, va_pred)
        fold_scores[name].append(score)
        print(f"  {name} AUC: {score:.6f}")

model_names = list(oof.keys())
oof_mat = np.column_stack([oof[n] for n in model_names])
test_mat = np.column_stack([test_pred[n] for n in model_names])

rank_oof_mat = np.column_stack(
    [
        rankdata(oof_mat[:, i], method="average") / len(oof_mat)
        for i in range(oof_mat.shape[1])
    ]
)
rank_test_mat = np.column_stack(
    [
        rankdata(test_mat[:, i], method="average") / len(test_mat)
        for i in range(test_mat.shape[1])
    ]
)


def candidate_weights(m):
    weights = []
    for i in range(m):
        w = np.zeros(m)
        w[i] = 1.0
        weights.append(w)

    if m == 1:
        return weights

    grid = np.linspace(0.0, 1.0, 11)
    if m == 2:
        for a in grid:
            weights.append(np.array([a, 1.0 - a]))
    elif m == 3:
        for a in grid:
            for b in grid:
                c = 1.0 - a - b
                if c >= -1e-12:
                    w = np.array([a, b, max(0.0, c)])
                    weights.append(w / w.sum())
    elif m == 4:
        for a in grid:
            for b in grid:
                for c in grid:
                    d = 1.0 - a - b - c
                    if d >= -1e-12:
                        w = np.array([a, b, c, max(0.0, d)])
                        weights.append(w / w.sum())
    else:
        rng = np.random.default_rng(RANDOM_STATE)
        weights.extend(rng.dirichlet(np.ones(m), size=300))

    unique = {}
    for w in weights:
        unique[tuple(np.round(w, 10))] = w
    return list(unique.values())


weights = candidate_weights(len(model_names))
print(f"Evaluating {2 * len(weights)} blend candidates")

best = {"auc": -1.0, "weights": None, "rank": False}
for use_rank, mat in [(False, oof_mat), (True, rank_oof_mat)]:
    for w in weights:
        pred = mat @ w
        auc = roc_auc_score(y, pred)
        if auc > best["auc"]:
            best = {"auc": auc, "weights": w.copy(), "rank": use_rank}

final_test = (rank_test_mat if best["rank"] else test_mat) @ best["weights"]
final_oof = (rank_oof_mat if best["rank"] else oof_mat) @ best["weights"]
final_test = np.clip(final_test, 0, 1)
final_oof = np.clip(final_oof, 0, 1)

print("Model CV AUCs:")
for n in model_names:
    print(
        f"{n}: mean={np.mean(fold_scores[n]):.6f}, folds={[round(v, 6) for v in fold_scores[n]]}"
    )

cv_score = roc_auc_score(y, final_oof)
print(f"OOF ROC AUC: {cv_score:.6f}")
print(
    "Best ensemble:",
    dict(zip(model_names, np.round(best["weights"], 4))),
    "rank_average=",
    best["rank"],
)

submission = sample.copy()
submission[TARGET] = final_test
submission.to_csv("./working/submission.csv", index=False)

pd.DataFrame(
    {
        "row": np.arange(len(y)),
        "target": y,
        "prediction": final_oof,
    }
).to_csv("./working/oof_predictions.csv.gz", index=False, compression="gzip")

test_predictions = sample.copy()
test_predictions[TARGET] = final_test
test_predictions.to_csv(
    "./working/test_predictions.csv.gz", index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "cv_score": float(cv_score),
    "research_hypotheses_llm_claimed_used": HYPOTHESES,
    "models": model_names,
    "ensemble_weights": {n: float(w) for n, w in zip(model_names, best["weights"])},
    "rank_average": bool(best["rank"]),
    "submission_path": "./working/submission.csv",
    "oof_path": "./working/oof_predictions.csv.gz",
    "test_predictions_path": "./working/test_predictions.csv.gz",
}
print(json.dumps(result, indent=2))
