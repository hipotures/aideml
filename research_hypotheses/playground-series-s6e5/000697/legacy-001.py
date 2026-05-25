import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold

    HAS_SGK = True
except Exception:
    HAS_SGK = False

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

SEED = 42
INPUT_DIR = "./input"
WORK_DIR = "./working"
os.makedirs(WORK_DIR, exist_ok=True)

REGIMES = ["dry_low_deg", "dry_high_deg", "wet_intermediate", "late_finish_window"]


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


try:
    from lightgbm import LGBMClassifier

    USE_LGBM = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier

    USE_LGBM = False


def build_lgbm(seed, specialist=False):
    if USE_LGBM:
        return LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=300 if specialist else 450,
            learning_rate=0.045,
            num_leaves=31 if specialist else 63,
            min_child_samples=80 if specialist else 50,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.2 if specialist else 0.05,
            reg_lambda=2.0 if specialist else 1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=max(1, (os.cpu_count() or 2) - 1),
            verbose=-1,
            force_col_wise=True,
        )

    num_cols = None
    cat_cols = None

    class FallbackModel:
        def __init__(self):
            self.pipe = None

        def fit(self, X, y, categorical_feature=None):
            nonlocal num_cols, cat_cols
            cat_cols = [
                c for c in X.columns if str(X[c].dtype) in ("object", "category")
            ]
            num_cols = [c for c in X.columns if c not in cat_cols]
            self.pipe = Pipeline(
                [
                    (
                        "prep",
                        ColumnTransformer(
                            [
                                ("cat", make_ohe(), cat_cols),
                                ("num", StandardScaler(), num_cols),
                            ]
                        ),
                    ),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=220 if specialist else 320,
                            learning_rate=0.055,
                            max_leaf_nodes=31,
                            l2_regularization=0.2 if specialist else 0.05,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            self.pipe.fit(X, y)
            return self

        def predict_proba(self, X):
            return self.pipe.predict_proba(X)

    return FallbackModel()


def add_features(train, test):
    train = train.copy()
    test = test.copy()
    train["_part"] = "train"
    test["_part"] = "test"
    all_df = pd.concat([train, test], axis=0, ignore_index=True)

    all_df["event_key"] = all_df["Year"].astype(str) + "__" + all_df["Race"].astype(str)
    all_df["wet_flag"] = (
        all_df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(np.int8)
    )
    all_df["dry_flag"] = (1 - all_df["wet_flag"]).astype(np.int8)

    rp = all_df["RaceProgress"].clip(lower=0.01)
    all_df["est_total_laps_raw"] = all_df["LapNumber"] / rp
    event_total = all_df.groupby("event_key")["est_total_laps_raw"].transform("median")
    fallback_total = float(np.nanmedian(all_df["est_total_laps_raw"]))
    all_df["est_total_laps"] = event_total.fillna(fallback_total).clip(30, 95)

    all_df["laps_remaining"] = (all_df["est_total_laps"] - all_df["LapNumber"]).clip(
        lower=0
    )
    all_df["lap_frac_est"] = (
        all_df["LapNumber"] / all_df["est_total_laps"].clip(lower=1)
    ).clip(0, 1.5)
    all_df["tyre_life_frac"] = (
        all_df["TyreLife"] / all_df["est_total_laps"].clip(lower=1)
    ).clip(0, 2)

    all_df["race_median_deg"] = all_df.groupby("event_key")[
        "Cumulative_Degradation"
    ].transform("median")
    all_df["race_median_deg"] = all_df["race_median_deg"].fillna(
        all_df["Cumulative_Degradation"].median()
    )
    all_df["deg_per_tyre_lap"] = all_df["Cumulative_Degradation"] / all_df[
        "TyreLife"
    ].clip(lower=1)
    all_df["abs_laptime_delta"] = all_df["LapTime_Delta"].abs()
    all_df["is_late_race_window"] = (
        (all_df["RaceProgress"] >= 0.84) | (all_df["laps_remaining"] <= 8)
    ).astype(np.int8)
    all_df["is_fresh_stint"] = (all_df["TyreLife"] <= 2).astype(np.int8)

    train_fe = (
        all_df[all_df["_part"] == "train"]
        .drop(columns=["_part"])
        .reset_index(drop=True)
    )
    test_fe = (
        all_df[all_df["_part"] == "test"].drop(columns=["_part"]).reset_index(drop=True)
    )
    return train_fe, test_fe


def assign_regime(df, deg_threshold):
    wet = df["wet_flag"].values == 1
    late = df["is_late_race_window"].values == 1
    high = df["race_median_deg"].values >= deg_threshold

    regime = np.full(len(df), "dry_low_deg", dtype=object)
    regime[high] = "dry_high_deg"
    regime[late & ~wet] = "late_finish_window"
    regime[wet] = "wet_intermediate"
    return regime


def fit_gate(X_gate, labels):
    cat_cols = ["Compound", "Race", "Year"]
    num_cols = [
        "wet_flag",
        "race_median_deg",
        "est_total_laps",
        "RaceProgress",
        "laps_remaining",
        "is_late_race_window",
    ]
    gate = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", make_ohe(), cat_cols),
                        ("num", StandardScaler(), num_cols),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=0.5,
                    max_iter=300,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="auto",
                    random_state=SEED,
                ),
            ),
        ]
    )
    gate.fit(X_gate[cat_cols + num_cols], labels)
    return gate, cat_cols + num_cols


def gate_proba(gate, gate_cols, X_gate):
    small = gate.predict_proba(X_gate[gate_cols])
    classes = list(gate.named_steps["clf"].classes_)
    out = np.zeros((len(X_gate), len(REGIMES)), dtype=np.float32)
    for j, cls in enumerate(classes):
        if cls in REGIMES:
            out[:, REGIMES.index(cls)] = small[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    out = np.divide(out, np.maximum(row_sum, 1e-12))
    return out


def blend_predictions(global_pred, expert_preds, gate_probs):
    mixture = np.sum(expert_preds * gate_probs, axis=1)
    conf = gate_probs.max(axis=1)
    strength = np.clip((conf - 0.35) / 0.40, 0.0, 1.0)
    pred = strength * mixture + (1.0 - strength) * global_pred
    pred = 0.88 * pred + 0.12 * global_pred
    return np.clip(pred, 1e-6, 1 - 1e-6)


train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

target = train["PitNextLap"].astype(int).values
train_fe, test_fe = add_features(train, test)

dry_train = train_fe["wet_flag"].values == 0
deg_threshold = float(np.nanmedian(train_fe.loc[dry_train, "race_median_deg"]))
train_regime = assign_regime(train_fe, deg_threshold)

drop_cols = ["id", "PitNextLap", "event_key", "est_total_laps_raw"]
feature_cols = [c for c in train_fe.columns if c not in drop_cols]
X = train_fe[feature_cols].copy()
X_test = test_fe[feature_cols].copy()

cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
cat_cols += ["Year"]
cat_cols = sorted(set([c for c in cat_cols if c in feature_cols]))

for c in cat_cols:
    cats = pd.Index(pd.concat([X[c], X_test[c]], axis=0).astype(str).unique())
    X[c] = pd.Categorical(X[c].astype(str), categories=cats)
    X_test[c] = pd.Categorical(X_test[c].astype(str), categories=cats)

groups = train_fe["event_key"].astype(str).values
if HAS_SGK:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = splitter.split(X, target, groups)
else:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = splitter.split(X, target)

oof = np.zeros(len(X), dtype=np.float32)
test_pred = np.zeros(len(X_test), dtype=np.float32)
fold_aucs = []

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = target[tr_idx], target[va_idx]

    global_model = build_lgbm(SEED + fold, specialist=False)
    if USE_LGBM:
        global_model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    else:
        global_model.fit(X_tr, y_tr)

    global_va = global_model.predict_proba(X_va)[:, 1]
    global_te = global_model.predict_proba(X_test)[:, 1]

    expert_va = np.zeros((len(va_idx), len(REGIMES)), dtype=np.float32)
    expert_te = np.zeros((len(X_test), len(REGIMES)), dtype=np.float32)

    for r_i, regime in enumerate(REGIMES):
        mask = train_regime[tr_idx] == regime
        if mask.sum() >= 1200 and len(np.unique(y_tr[mask])) == 2:
            model = build_lgbm(SEED + 100 * fold + r_i, specialist=True)
            if USE_LGBM:
                model.fit(X_tr.iloc[mask], y_tr[mask], categorical_feature=cat_cols)
            else:
                model.fit(X_tr.iloc[mask], y_tr[mask])
            expert_va[:, r_i] = model.predict_proba(X_va)[:, 1]
            expert_te[:, r_i] = model.predict_proba(X_test)[:, 1]
        else:
            expert_va[:, r_i] = global_va
            expert_te[:, r_i] = global_te

    gate, gate_cols = fit_gate(train_fe.iloc[tr_idx], train_regime[tr_idx])
    gate_va = gate_proba(gate, gate_cols, train_fe.iloc[va_idx])
    gate_te = gate_proba(gate, gate_cols, test_fe)

    pred_va = blend_predictions(global_va, expert_va, gate_va)
    pred_te = blend_predictions(global_te, expert_te, gate_te)

    oof[va_idx] = pred_va
    test_pred += pred_te / 5.0

    fold_auc = roc_auc_score(y_va, pred_va)
    fold_aucs.append(float(fold_auc))
    print(f"Fold {fold} ROC AUC: {fold_auc:.6f}")

cv_auc = float(roc_auc_score(target, oof))
print(f"CV ROC AUC: {cv_auc:.6f}")

submission = sample.copy()
submission["PitNextLap"] = np.clip(test_pred, 1e-6, 1 - 1e-6)
submission.to_csv(os.path.join(WORK_DIR, "submission.csv"), index=False)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": target,
        "prediction": oof,
    }
).to_csv(
    os.path.join(WORK_DIR, "oof_predictions.csv.gz"), index=False, compression="gzip"
)

submission.to_csv(
    os.path.join(WORK_DIR, "test_predictions.csv.gz"), index=False, compression="gzip"
)

result = {
    "research_hypotheses_llm_claimed_used": ["000697"],
    "metric": "roc_auc",
    "cv_roc_auc": cv_auc,
    "fold_roc_auc": fold_aucs,
    "deg_threshold": deg_threshold,
    "used_lightgbm": USE_LGBM,
}
with open(os.path.join(WORK_DIR, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, sort_keys=True))
