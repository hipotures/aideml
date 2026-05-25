import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
WORKING_DIR = "./working"
os.makedirs(WORKING_DIR, exist_ok=True)

RANDOM_STATE = 408
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
EPS = 1e-6

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

y = train[TARGET].astype(int).values
test_ids = sample[ID_COL].values


def add_features(df):
    out = df.copy()
    out["is_testing"] = (out["Race"].astype(str) == "Pre-Season Testing").astype(int)
    out["is_wet_compound"] = out["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    out["is_slick"] = out["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(int)

    keys = ["Year", "Race", "Driver"]
    order_cols = keys + ["LapNumber", ID_COL]
    ordered = out.sort_values(order_cols, kind="mergesort").copy()

    slick_seen_cols = []
    for compound in ["SOFT", "MEDIUM", "HARD"]:
        col = f"seen_{compound.lower()}"
        ordered[col] = (
            ordered["Compound"].eq(compound) & ordered["is_slick"].eq(1)
        ).astype(int)
        ordered[col] = ordered.groupby(keys, sort=False)[col].cummax()
        slick_seen_cols.append(col)

    ordered["slick_compounds_seen"] = ordered[slick_seen_cols].sum(axis=1).astype(float)
    out["slick_compounds_seen"] = ordered.sort_index()["slick_compounds_seen"].values

    out["mandatory_stop_owed"] = (
        (out["is_testing"].eq(0))
        & (out["is_wet_compound"].eq(0))
        & (out["slick_compounds_seen"] < 2)
        & (out["RaceProgress"] < 0.92)
    ).astype(int)

    out["tyre_life_x_progress"] = out["TyreLife"] * out["RaceProgress"]
    out["stint_x_progress"] = out["Stint"] * out["RaceProgress"]
    out["lap_frac"] = out["LapNumber"] / out.groupby(["Year", "Race"])[
        "LapNumber"
    ].transform("max").clip(lower=1)
    out["degradation_per_lap"] = out["Cumulative_Degradation"] / out["TyreLife"].clip(
        lower=1
    )
    out["abs_position_change"] = out["Position_Change"].abs()
    out["recent_pit_or_first_lap"] = (
        (out["PitStop"] == 1) | (out["TyreLife"] <= 2)
    ).astype(int)

    for c in ["LapTime (s)", "LapTime_Delta", "Cumulative_Degradation", "TyreLife"]:
        grp = out.groupby(["Year", "Race"])[c]
        med = grp.transform("median")
        q75 = grp.transform(lambda s: s.quantile(0.75))
        q25 = grp.transform(lambda s: s.quantile(0.25))
        out[c + "_race_centered"] = out[c] - med
        out[c + "_race_iqr_scaled"] = (out[c] - med) / (q75 - q25 + 1e-3)

    return out


all_df = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
all_df = add_features(all_df)

cat_cols = ["Driver", "Race", "Compound"]
num_cols = [c for c in all_df.columns if c not in cat_cols + [ID_COL]]

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
all_df[cat_cols] = encoder.fit_transform(all_df[cat_cols].astype(str)).astype(np.int32)

features = cat_cols + num_cols
X_all = all_df[features].replace([np.inf, -np.inf], np.nan)
X_train = X_all.iloc[: len(train)].reset_index(drop=True)
X_test = X_all.iloc[len(train) :].reset_index(drop=True)

regime_all = pd.DataFrame(
    {
        "testing": all_df["is_testing"].values.astype(bool),
        "wet": (
            (all_df["is_testing"].values == 0) & (all_df["is_wet_compound"].values == 1)
        ),
        "dry_owed": (
            (all_df["is_testing"].values == 0)
            & (all_df["is_wet_compound"].values == 0)
            & (all_df["mandatory_stop_owed"].values == 1)
        ),
        "dry_free": (
            (all_df["is_testing"].values == 0)
            & (all_df["is_wet_compound"].values == 0)
            & (all_df["mandatory_stop_owed"].values == 0)
        ),
    }
)
regime_train = regime_all.iloc[: len(train)].reset_index(drop=True)
regime_test = regime_all.iloc[len(train) :].reset_index(drop=True)


def make_model(seed):
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            n_estimators=900,
            learning_rate=0.035,
            num_leaves=48,
            max_depth=-1,
            min_child_samples=70,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=450,
            learning_rate=0.045,
            max_leaf_nodes=48,
            l2_regularization=0.05,
            random_state=seed,
        )


def predict_proba_positive(model, X):
    if len(X) == 0:
        return np.array([], dtype=float)
    p = model.predict_proba(X)
    if p.shape[1] == 1:
        return np.full(len(X), float(model.classes_[0]))
    return p[:, list(model.classes_).index(1)]


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_predict_model(X_tr, y_tr, X_va, X_te, seed):
    if len(np.unique(y_tr)) < 2:
        const = float(np.mean(y_tr))
        return np.full(len(X_va), const), np.full(len(X_te), const)

    model = make_model(seed)
    fit_kwargs = {}
    if "LGBMClassifier" in type(model).__name__:
        fit_kwargs["categorical_feature"] = cat_cols

    model.fit(X_tr, y_tr, **fit_kwargs)
    return predict_proba_positive(model, X_va), predict_proba_positive(model, X_te)


experts = ["global", "testing", "wet", "dry_owed", "dry_free"]
oof_expert = {name: np.full(len(train), np.nan) for name in experts}
test_expert_folds = {name: [] for name in experts}

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    va_pred, te_pred = fit_predict_model(X_tr, y_tr, X_va, X_test, RANDOM_STATE + fold)
    oof_expert["global"][va_idx] = va_pred
    test_expert_folds["global"].append(te_pred)

    for name in ["testing", "wet", "dry_owed", "dry_free"]:
        tr_mask = regime_train.loc[tr_idx, name].values
        va_mask = regime_train.loc[va_idx, name].values
        te_mask = regime_test[name].values

        specialist_va = va_pred.copy()
        specialist_te = te_pred.copy()

        if tr_mask.sum() >= 200 and len(np.unique(y_tr[tr_mask])) == 2:
            sub_va_pred, sub_te_pred = fit_predict_model(
                X_tr.loc[tr_mask],
                y_tr[tr_mask],
                X_va.loc[va_mask] if va_mask.any() else X_va.iloc[:0],
                X_test.loc[te_mask] if te_mask.any() else X_test.iloc[:0],
                RANDOM_STATE + 100 * fold + len(name),
            )
            if va_mask.any():
                specialist_va[va_mask] = sub_va_pred
            if te_mask.any():
                specialist_te[te_mask] = sub_te_pred

        oof_expert[name][va_idx] = specialist_va
        test_expert_folds[name].append(specialist_te)

    fold_auc = roc_auc_score(y_va, oof_expert["global"][va_idx])
    print(f"fold {fold} global_auc={fold_auc:.6f}")

for name in experts:
    if np.isnan(oof_expert[name]).any():
        oof_expert[name] = np.where(
            np.isnan(oof_expert[name]), oof_expert["global"], oof_expert[name]
        )

oof_matrix = np.column_stack([logit(oof_expert[name]) for name in experts])
test_matrix = np.column_stack(
    [logit(np.mean(test_expert_folds[name], axis=0)) for name in experts]
)

meta_oof = np.zeros(len(train))
meta_test_folds = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(oof_matrix, y), 1):
    meta = LogisticRegression(
        C=0.5,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE + fold,
    )
    meta.fit(oof_matrix[tr_idx], y[tr_idx])
    meta_oof[va_idx] = meta.predict_proba(oof_matrix[va_idx])[:, 1]
    meta_test_folds.append(meta.predict_proba(test_matrix)[:, 1])

test_pred = np.mean(meta_test_folds, axis=0)
cv_auc = roc_auc_score(y, meta_oof)
global_auc = roc_auc_score(y, oof_expert["global"])

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": meta_oof,
    }
).to_csv(
    os.path.join(WORKING_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

pd.DataFrame(
    {
        ID_COL: test_ids,
        TARGET: test_pred,
    }
).to_csv(
    os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
    index=False,
    compression="gzip",
)

submission = sample.copy()
submission[TARGET] = np.clip(test_pred, 0, 1)
submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)

print(f"global_oof_roc_auc={global_auc:.6f}")
print(f"regime_specialist_meta_oof_roc_auc={cv_auc:.6f}")
print(
    json.dumps(
        {
            "metric": "roc_auc",
            "validation_score": float(cv_auc),
            "global_baseline_score": float(global_auc),
            "research_hypotheses_llm_claimed_used": ["000408"],
            "saved_files": [
                "./working/submission.csv",
                "./working/oof_predictions.csv.gz",
                "./working/test_predictions.csv.gz",
            ],
        }
    )
)
