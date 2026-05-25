import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

TARGET = "PitNextLap"
ID = "id"
INPUT = "./input"
WORKING = "./working"
os.makedirs(WORKING, exist_ok=True)

train = pd.read_csv(f"{INPUT}/train.csv.gz").rename(
    columns={"LapTime (s)": "LapTime_s"}
)
test = pd.read_csv(f"{INPUT}/test.csv.gz").rename(columns={"LapTime (s)": "LapTime_s"})
sample = pd.read_csv(f"{INPUT}/sample_submission.csv.gz")
y = train[TARGET].astype(int).to_numpy()


def cut_str(s, bins, prefix):
    codes = pd.cut(s, bins=bins, labels=False, include_lowest=True)
    codes = pd.Series(codes, index=s.index).fillna(-1).astype(int).astype(str)
    return prefix + codes


def add_features(df):
    df = df.copy()
    for c in ["Compound", "Driver", "Race"]:
        df[c] = df[c].astype(str).fillna("NA")

    rp = df["RaceProgress"].clip(0.005, 1.0)
    tyre = df["TyreLife"].clip(lower=1)
    df["EstimatedTotalLaps"] = (df["LapNumber"] / rp).clip(1, 120)
    df["LapsRemaining"] = (df["EstimatedTotalLaps"] - df["LapNumber"]).clip(0, 120)
    df["TyreLifeFrac"] = df["TyreLife"] / (df["EstimatedTotalLaps"] + 1)
    df["TyreLifeToLap"] = df["TyreLife"] / (df["LapNumber"] + 1)
    df["DegPerTyreLap"] = df["Cumulative_Degradation"] / tyre
    df["AbsLapDelta"] = df["LapTime_Delta"].abs()
    df["PositionFrac"] = df["Position"] / 20.0
    df["IsWetCompound"] = df["Compound"].isin(["INTERMEDIATE", "WET"]).astype(int)
    df["IsSlickCompound"] = df["Compound"].isin(["SOFT", "MEDIUM", "HARD"]).astype(int)
    df["LateRace"] = (df["RaceProgress"] >= 0.75).astype(int)

    df["Year_cat"] = df["Year"].astype(str)
    df["Stint_cat"] = df["Stint"].astype(str)
    df["LapNumber_int"] = df["LapNumber"].round().clip(1, 120).astype(int).astype(str)
    df["TyreLife_int"] = df["TyreLife"].round().clip(1, 120).astype(int).astype(str)
    df["phase_bin"] = cut_str(
        df["RaceProgress"], [-0.01, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.01], "phase_"
    )
    df["tyre_bin"] = cut_str(
        df["TyreLife"], [0, 3, 5, 8, 12, 16, 22, 30, 45, 120], "tyre_"
    )
    df["lap_bin"] = cut_str(
        df["LapNumber"], [0, 5, 10, 15, 20, 30, 40, 55, 120], "lap_"
    )
    df["deg_bin"] = cut_str(
        df["DegPerTyreLap"], [-np.inf, -10, -3, 0, 2, 5, 10, 20, np.inf], "deg_"
    )

    df["compound_stint"] = df["Compound"] + "_" + df["Stint_cat"]
    df["race_phase"] = df["Race"] + "_" + df["phase_bin"]
    df["state_key"] = df["Compound"] + "_" + df["phase_bin"] + "_" + df["tyre_bin"]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


all_x = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True, sort=False)
all_x = add_features(all_x)

cat_cols = [
    "Compound",
    "Driver",
    "Race",
    "Year_cat",
    "Stint_cat",
    "LapNumber_int",
    "TyreLife_int",
    "phase_bin",
    "tyre_bin",
    "lap_bin",
    "deg_bin",
    "compound_stint",
    "race_phase",
    "state_key",
]
for c in cat_cols:
    all_x[c] = all_x[c].astype(str).astype("category")

train_x = all_x.iloc[: len(train)].copy()
test_x = all_x.iloc[len(train) :].copy()
feature_cols = [c for c in train_x.columns if c != ID]
lgb_cat_cols = [c for c in cat_cols if c in feature_cols]

HAZARD_KEYS = [
    ("Race", "LapNumber_int"),
    ("Race", "Compound", "TyreLife_int"),
    ("Compound", "TyreLife_int"),
    ("Stint_cat", "TyreLife_int"),
    ("Race", "phase_bin", "Compound"),
    ("compound_stint", "tyre_bin"),
    ("state_key",),
    ("Race", "Stint_cat", "lap_bin"),
]
HAZARD_SMOOTH = [80, 90, 70, 60, 100, 70, 60, 90]


def key_frame(frame, keys):
    out = frame[list(keys)].copy()
    for c in keys:
        out[c] = out[c].astype(str).fillna("NA")
    return out


def hazard_matrix(fit_frame, fit_y, apply_frame, prior, loo=False):
    cols = {}
    fit_y = np.asarray(fit_y, dtype=float)
    for i, keys in enumerate(HAZARD_KEYS):
        keys = list(keys)
        name = "haz_" + "_".join(keys)
        smooth = HAZARD_SMOOTH[i]
        fk = key_frame(fit_frame, keys)
        fk["_target"] = fit_y
        stats = (
            fk.groupby(keys, observed=True)["_target"]
            .agg(["sum", "count"])
            .reset_index()
        )

        if loo:
            app = fk.copy()
            app["_ord"] = np.arange(len(app))
            merged = app.merge(stats, on=keys, how="left", sort=False)
            vals = (merged["sum"] - merged["_target"] + prior * smooth) / (
                merged["count"] - 1.0 + smooth
            )
        else:
            app = key_frame(apply_frame, keys)
            app["_ord"] = np.arange(len(app))
            merged = app.merge(stats, on=keys, how="left", sort=False)
            vals = (merged["sum"] + prior * smooth) / (merged["count"] + smooth)

        cols[name] = vals.fillna(prior).to_numpy(dtype=float)[
            np.argsort(merged["_ord"].to_numpy())
        ]
    return pd.DataFrame(cols)


AUX_NUM = [
    "LapNumber",
    "TyreLife",
    "RaceProgress",
    "LapsRemaining",
    "DegPerTyreLap",
    "LapTime_s",
    "LapTime_Delta",
    "AbsLapDelta",
    "Cumulative_Degradation",
    "Position",
    "Position_Change",
    "Stint",
    "PitStop",
    "IsWetCompound",
    "LateRace",
]
AUX_CAT = [
    "Compound",
    "Race",
    "Year_cat",
    "Stint_cat",
    "phase_bin",
    "tyre_bin",
    "lap_bin",
]


def make_aux_frame(fit_frame, fit_y, apply_frame, prior, loo=False):
    haz = hazard_matrix(fit_frame, fit_y, apply_frame, prior, loo=loo)
    base = apply_frame[AUX_NUM + AUX_CAT].reset_index(drop=True).copy()
    for c in AUX_CAT:
        base[c] = base[c].astype(str)
    return pd.concat([base, haz], axis=1)


def make_ohe():
    try:
        return OneHotEncoder(
            handle_unknown="ignore", min_frequency=20, sparse_output=True
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def regime_predict(fit_frame, fit_y, apply_frame, prior):
    haz = hazard_matrix(fit_frame, fit_y, apply_frame, prior, loo=False)
    weights = np.array([0.24, 0.18, 0.16, 0.12, 0.12, 0.10, 0.05, 0.03])
    weights = weights / weights.sum()
    return np.clip(haz.to_numpy().dot(weights), 1e-6, 1 - 1e-6)


def rank_for_auc(pred, frame):
    s = pd.Series(np.asarray(pred, dtype=float))
    global_rank = s.rank(method="average", pct=True).to_numpy()
    key = (
        frame["Year"].astype(str).reset_index(drop=True)
        + "|"
        + frame["Race"].astype(str).reset_index(drop=True)
    )
    group_rank = s.groupby(key).rank(method="average", pct=True).to_numpy()
    ranked = 0.5 * global_rank + 0.5 * group_rank
    return np.clip(np.nan_to_num(ranked, nan=np.nanmean(global_rank)), 1e-6, 1 - 1e-6)


try:
    from sklearn.model_selection import StratifiedGroupKFold

    groups = (
        train_x["Year"].astype(str)
        + "|"
        + train_x["Race"].astype(str)
        + "|"
        + train_x["Driver"].astype(str)
    )
    splits = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(
            train_x, y, groups
        )
    )
except Exception:
    splits = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(train_x, y)
    )

model_names = ["tabular_lgbm", "hazard_logit", "latent_catboost", "regime_hazard"]
oof = {m: np.zeros(len(train_x), dtype=float) for m in model_names}
test_rank_sum = {m: np.zeros(len(test_x), dtype=float) for m in model_names}
prior = float(y.mean())

latent_cols = [
    "Compound",
    "Driver",
    "Race",
    "Year_cat",
    "Stint_cat",
    "phase_bin",
    "tyre_bin",
    "deg_bin",
    "lap_bin",
    "compound_stint",
    "race_phase",
    "state_key",
    "LapNumber",
    "TyreLife",
    "RaceProgress",
    "LapsRemaining",
    "DegPerTyreLap",
    "Position",
    "Position_Change",
    "PitStop",
    "Cumulative_Degradation",
    "LapTime_Delta",
]
latent_cat = [c for c in latent_cols if c in cat_cols]

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    print(f"Fold {fold}/5")
    X_tr = train_x.iloc[tr_idx]
    X_va = train_x.iloc[va_idx]
    y_tr = y[tr_idx]
    y_va = y[va_idx]
    pos_weight = max(1.0, (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum()))

    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=900,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=4.0,
        scale_pos_weight=pos_weight,
        random_state=1000 + fold,
        n_jobs=-1,
        verbosity=-1,
    )
    lgbm.fit(
        X_tr[feature_cols],
        y_tr,
        eval_set=[(X_va[feature_cols], y_va)],
        eval_metric="auc",
        categorical_feature=lgb_cat_cols,
        callbacks=[early_stopping(80), log_evaluation(0)],
    )
    pred_va = lgbm.predict_proba(X_va[feature_cols])[:, 1]
    pred_te = lgbm.predict_proba(test_x[feature_cols])[:, 1]
    oof["tabular_lgbm"][va_idx] = rank_for_auc(pred_va, X_va)
    test_rank_sum["tabular_lgbm"] += rank_for_auc(pred_te, test_x) / len(splits)

    aux_tr = make_aux_frame(X_tr, y_tr, X_tr, prior, loo=True)
    aux_va = make_aux_frame(X_tr, y_tr, X_va, prior, loo=False)
    aux_te = make_aux_frame(X_tr, y_tr, test_x, prior, loo=False)
    aux_num = [c for c in aux_tr.columns if c not in AUX_CAT]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler(with_mean=False)),
                    ]
                ),
                aux_num,
            ),
            ("cat", make_ohe(), AUX_CAT),
        ]
    )
    logit = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="saga",
                    max_iter=350,
                    tol=1e-3,
                    n_jobs=-1,
                    random_state=2000 + fold,
                ),
            ),
        ]
    )
    logit.fit(aux_tr, y_tr)
    pred_va = logit.predict_proba(aux_va)[:, 1]
    pred_te = logit.predict_proba(aux_te)[:, 1]
    oof["hazard_logit"][va_idx] = rank_for_auc(pred_va, X_va)
    test_rank_sum["hazard_logit"] += rank_for_auc(pred_te, test_x) / len(splits)

    cb_tr = X_tr[latent_cols].copy()
    cb_va = X_va[latent_cols].copy()
    cb_te = test_x[latent_cols].copy()
    for c in latent_cat:
        cb_tr[c] = cb_tr[c].astype(str)
        cb_va[c] = cb_va[c].astype(str)
        cb_te[c] = cb_te[c].astype(str)

    cb = CatBoostClassifier(
        iterations=500,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=8.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=3000 + fold,
        class_weights=[1.0, pos_weight],
        allow_writing_files=False,
        verbose=False,
    )
    cb.fit(
        cb_tr,
        y_tr,
        cat_features=latent_cat,
        eval_set=(cb_va, y_va),
        early_stopping_rounds=70,
        use_best_model=True,
    )
    pred_va = cb.predict_proba(cb_va)[:, 1]
    pred_te = cb.predict_proba(cb_te)[:, 1]
    oof["latent_catboost"][va_idx] = rank_for_auc(pred_va, X_va)
    test_rank_sum["latent_catboost"] += rank_for_auc(pred_te, test_x) / len(splits)

    pred_va = regime_predict(X_tr, y_tr, X_va, prior)
    pred_te = regime_predict(X_tr, y_tr, test_x, prior)
    oof["regime_hazard"][va_idx] = rank_for_auc(pred_va, X_va)
    test_rank_sum["regime_hazard"] += rank_for_auc(pred_te, test_x) / len(splits)


def greedy_auc_stack(mat, target, names, max_rounds=80, tol=1e-7):
    single_auc = np.array(
        [roc_auc_score(target, mat[:, i]) for i in range(mat.shape[1])]
    )
    best = int(np.argmax(single_auc))
    counts = np.zeros(mat.shape[1], dtype=int)
    counts[best] = 1
    current = mat[:, best].copy()
    best_auc = float(single_auc[best])

    for _ in range(max_rounds - 1):
        total = counts.sum()
        cand_auc = []
        for j in range(mat.shape[1]):
            cand = (current * total + mat[:, j]) / (total + 1)
            cand_auc.append(roc_auc_score(target, cand))
        j = int(np.argmax(cand_auc))
        if cand_auc[j] <= best_auc + tol:
            break
        current = (current * total + mat[:, j]) / (total + 1)
        counts[j] += 1
        best_auc = float(cand_auc[j])

    weights = counts / counts.sum()
    return weights, current, best_auc, single_auc


oof_mat = np.column_stack([oof[m] for m in model_names])
test_mat = np.column_stack([test_rank_sum[m] for m in model_names])
weights, stack_oof, stack_auc, single_auc = greedy_auc_stack(oof_mat, y, model_names)
stack_test = np.clip(test_mat.dot(weights), 1e-6, 1 - 1e-6)

submission = sample.copy()
submission[TARGET] = stack_test
submission.to_csv(f"{WORKING}/submission.csv", index=False)
submission.to_csv(f"{WORKING}/test_predictions.csv.gz", index=False, compression="gzip")

pd.DataFrame(
    {
        "row": np.arange(len(y)),
        "target": y,
        "prediction": stack_oof,
    }
).to_csv(f"{WORKING}/oof_predictions.csv.gz", index=False, compression="gzip")

print(f"5-fold OOF ROC AUC: {stack_auc:.6f}")
print(
    "Base OOF AUC:",
    json.dumps({m: float(a) for m, a in zip(model_names, single_auc)}, sort_keys=True),
)
print(
    "Greedy stack weights:",
    json.dumps({m: float(w) for m, w in zip(model_names, weights)}, sort_keys=True),
)
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000896"],
            "validation_metric": "5_fold_oof_roc_auc",
            "validation_score": float(stack_auc),
            "submission_path": f"{WORKING}/submission.csv",
            "oof_path": f"{WORKING}/oof_predictions.csv.gz",
            "test_predictions_path": f"{WORKING}/test_predictions.csv.gz",
        },
        sort_keys=True,
    )
)
