import os
import gc
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

SEED = 20260525
N_SPLITS = 5
TARGET = "PitNextLap"
ID_COL = "id"
INPUT_DIR = "./input"
WORKING_DIR = "./working"
THREADS = max(1, min(8, os.cpu_count() or 1))

try:
    from pytabkit import RealMLP_TD_Classifier

    PYTABKIT_IMPORT_ERROR = None
except Exception as e:
    RealMLP_TD_Classifier = None
    PYTABKIT_IMPORT_ERROR = repr(e)


def clean_pred(p):
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(p, 1e-6, 1 - 1e-6)


def positive_proba(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = np.asarray(p)
        if p.ndim == 2:
            return clean_pred(p[:, 1])
        return clean_pred(p)
    return clean_pred(model.predict(X))


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce").astype(float)
    b = pd.to_numeric(b, errors="coerce").astype(float)
    return a / np.where(np.abs(b) < 1e-9, np.nan, b)


def build_features(train, test):
    all_df = pd.concat(
        [train.drop(columns=[TARGET], errors="ignore"), test],
        axis=0,
        ignore_index=True,
    )
    all_df = all_df.rename(columns={"LapTime (s)": "LapTime_s"})
    all_df = all_df.drop(columns=[ID_COL], errors="ignore")

    object_cols = all_df.select_dtypes(include=["object"]).columns.tolist()
    for c in object_cols:
        all_df[c] = all_df[c].fillna("NA").astype(str)

    all_df["YearCat"] = all_df["Year"].astype(str)
    all_df["RaceYear"] = all_df["Race"].astype(str) + "_" + all_df["YearCat"]
    all_df["DriverRace"] = (
        all_df["Driver"].astype(str) + "_" + all_df["Race"].astype(str)
    )
    all_df["DriverRaceYear"] = all_df["DriverRace"] + "_" + all_df["YearCat"]
    all_df["CompoundStint"] = (
        all_df["Compound"].astype(str) + "_" + all_df["Stint"].astype(str)
    )

    cat_cols = list(
        dict.fromkeys(
            object_cols
            + ["YearCat", "RaceYear", "DriverRace", "DriverRaceYear", "CompoundStint"]
        )
    )

    lap = pd.to_numeric(all_df["LapNumber"], errors="coerce")
    tyre = pd.to_numeric(all_df["TyreLife"], errors="coerce")
    progress = pd.to_numeric(all_df["RaceProgress"], errors="coerce")
    degr = pd.to_numeric(all_df["Cumulative_Degradation"], errors="coerce")
    lap_time = pd.to_numeric(all_df["LapTime_s"], errors="coerce")
    lap_delta = pd.to_numeric(all_df["LapTime_Delta"], errors="coerce")
    pos = pd.to_numeric(all_df["Position"], errors="coerce")
    pos_chg = pd.to_numeric(all_df["Position_Change"], errors="coerce")
    stint = pd.to_numeric(all_df["Stint"], errors="coerce")
    pitstop = pd.to_numeric(all_df["PitStop"], errors="coerce")

    max_lap = all_df.groupby("RaceYear")["LapNumber"].transform("max")
    compound = all_df["Compound"].astype(str)

    all_df["IsWetCompound"] = compound.isin(["INTERMEDIATE", "WET"]).astype(np.float32)
    all_df["IsSoftCompound"] = (compound == "SOFT").astype(np.float32)
    all_df["IsHardCompound"] = (compound == "HARD").astype(np.float32)
    all_df["TyreLife_sq"] = tyre**2
    all_df["TyreLife_log1p"] = np.log1p(tyre.clip(lower=0))
    all_df["TyreLife_x_RaceProgress"] = tyre * progress
    all_df["TyreLife_per_Lap"] = safe_div(tyre, lap)
    all_df["DegradationPerTyreLap"] = safe_div(degr, tyre)
    all_df["DegradationPerRaceProgress"] = safe_div(degr, progress)
    all_df["LapTime_log1p"] = np.log1p(lap_time.clip(lower=0))
    all_df["AbsLapTimeDelta"] = lap_delta.abs()
    all_df["PositiveLapTimeDelta"] = lap_delta.clip(lower=0)
    all_df["NegativeLapTimeDelta"] = (-lap_delta).clip(lower=0)
    all_df["LapsRemainingEst"] = max_lap - lap
    all_df["LapFracOfRaceMax"] = safe_div(lap, max_lap)
    all_df["Position_x_RaceProgress"] = pos * progress
    all_df["AbsPositionChange"] = pos_chg.abs()
    all_df["GainedPositions"] = (-pos_chg).clip(lower=0)
    all_df["LostPositions"] = pos_chg.clip(lower=0)
    all_df["Stint_x_TyreLife"] = stint * tyre
    all_df["PitStop_x_TyreLife"] = pitstop * tyre
    all_df["LateRaceOldTyre"] = progress * np.log1p(tyre.clip(lower=0))

    for c in cat_cols:
        counts = all_df[c].map(all_df[c].value_counts()).astype(np.float32)
        all_df[f"{c}_count_log"] = np.log1p(counts)

    for c in cat_cols:
        all_df[c] = all_df[c].fillna("NA").astype(str)

    num_cols = [c for c in all_df.columns if c not in cat_cols]
    for c in num_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )

    train_feat = all_df.iloc[: len(train)].reset_index(drop=True)
    test_feat = all_df.iloc[len(train) :].reset_index(drop=True)

    medians = train_feat[num_cols].median().fillna(0)
    train_feat[num_cols] = train_feat[num_cols].fillna(medians).astype(np.float32)
    test_feat[num_cols] = test_feat[num_cols].fillna(medians).astype(np.float32)

    return train_feat, test_feat, cat_cols, num_cols


def make_ohe():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=30,
            sparse_output=True,
            dtype=np.float32,
        )
    except TypeError:
        try:
            return OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=30,
                sparse=True,
                dtype=np.float32,
            )
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def make_lgb_frames(train_feat, test_feat, cat_cols, num_cols):
    full = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)
    for c in cat_cols:
        full[c] = pd.Categorical(full[c].astype(str))
    for c in num_cols:
        full[c] = full[c].astype(np.float32)
    return full.iloc[: len(train_feat)].copy(), full.iloc[len(train_feat) :].copy()


def make_plain_frames(train_feat, test_feat, cat_cols, num_cols):
    train_out = train_feat.copy()
    test_out = test_feat.copy()
    for df in (train_out, test_out):
        for c in cat_cols:
            df[c] = df[c].astype(str)
        for c in num_cols:
            df[c] = df[c].astype(np.float32)
    return train_out, test_out


def fit_neural_predict(X_tr, y_tr, X_va, y_va, X_te, num_cols, cat_cols, seed):
    if RealMLP_TD_Classifier is not None:
        try:
            scaler = StandardScaler()
            Xtr = X_tr.copy()
            Xva = X_va.copy()
            Xte = X_te.copy()
            Xtr[num_cols] = scaler.fit_transform(Xtr[num_cols]).astype(np.float32)
            Xva[num_cols] = scaler.transform(Xva[num_cols]).astype(np.float32)
            Xte[num_cols] = scaler.transform(Xte[num_cols]).astype(np.float32)

            for c in cat_cols:
                cats = pd.Index(
                    pd.concat([X_tr[c], X_va[c], X_te[c]], ignore_index=True)
                    .astype(str)
                    .unique()
                )
                Xtr[c] = pd.Categorical(Xtr[c].astype(str), categories=cats)
                Xva[c] = pd.Categorical(Xva[c].astype(str), categories=cats)
                Xte[c] = pd.Categorical(Xte[c].astype(str), categories=cats)

            model = RealMLP_TD_Classifier(
                device="cpu",
                random_state=seed,
                n_cv=1,
                n_refit=0,
                val_fraction=0.15,
                n_threads=THREADS,
                tmp_folder=os.path.join(WORKING_DIR, "pytabkit_tmp"),
                verbosity=0,
                n_epochs=35,
                batch_size=4096,
                predict_batch_size=8192,
                hidden_sizes=[128, 64],
                p_drop=0.10,
                wd=1e-4,
                use_early_stopping=True,
            )
            try:
                model.fit(Xtr, y_tr, Xva, y_va)
            except TypeError:
                model.fit(Xtr, y_tr)
            return (
                positive_proba(model, Xva),
                positive_proba(model, Xte),
                "pytabkit_realmlp",
            )
        except Exception as e:
            print(
                f"PyTabKit RealMLP failed on this fold ({type(e).__name__}: {e}); using sklearn MLP fallback."
            )

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="NA")),
            ("onehot", make_ohe()),
        ]
    )
    pre = ColumnTransformer(
        [("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        sparse_threshold=0.35,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(96, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=4096,
        learning_rate_init=0.002,
        max_iter=25,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=4,
        random_state=seed,
        verbose=False,
    )
    model = Pipeline([("prep", pre), ("mlp", mlp)])
    model.fit(X_tr, y_tr)
    return (
        positive_proba(model, X_va),
        positive_proba(model, X_te),
        "sklearn_mlp_fallback",
    )


def main():
    os.makedirs(WORKING_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv.gz"))
    test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv.gz"))
    sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv.gz"))

    y = train[TARGET].astype(int).to_numpy()
    train_feat, test_feat, cat_cols, num_cols = build_features(train, test)

    X_lgb, X_lgb_test = make_lgb_frames(train_feat, test_feat, cat_cols, num_cols)
    X_plain, X_plain_test = make_plain_frames(train_feat, test_feat, cat_cols, num_cols)

    print(f"Using {len(num_cols)} numeric and {len(cat_cols)} categorical features.")
    if PYTABKIT_IMPORT_ERROR:
        print(
            f"PyTabKit unavailable ({PYTABKIT_IMPORT_ERROR}); neural component will use sklearn MLP fallback."
        )

    oof_lgb = np.zeros(len(train), dtype=np.float64)
    oof_cat = np.zeros(len(train), dtype=np.float64)
    oof_nn = np.zeros(len(train), dtype=np.float64)
    test_lgb = np.zeros(len(test), dtype=np.float64)
    test_cat = np.zeros(len(test), dtype=np.float64)
    test_nn = np.zeros(len(test), dtype=np.float64)
    nn_backends = []

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_lgb, y), 1):
        print(f"Fold {fold}/{N_SPLITS}")

        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            n_estimators=1800,
            learning_rate=0.035,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=SEED + fold,
            n_jobs=THREADS,
            force_col_wise=True,
            verbosity=-1,
        )
        lgb_model.fit(
            X_lgb.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X_lgb.iloc[va_idx], y[va_idx])],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof_lgb[va_idx] = positive_proba(lgb_model, X_lgb.iloc[va_idx])
        test_lgb += positive_proba(lgb_model, X_lgb_test) / N_SPLITS

        cat_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=900,
            learning_rate=0.055,
            depth=6,
            l2_leaf_reg=8.0,
            random_strength=0.6,
            bootstrap_type="Bernoulli",
            subsample=0.85,
            rsm=0.90,
            od_type="Iter",
            od_wait=80,
            random_seed=SEED + fold,
            allow_writing_files=False,
            verbose=False,
            thread_count=THREADS,
        )
        cat_model.fit(
            X_plain.iloc[tr_idx],
            y[tr_idx],
            cat_features=cat_cols,
            eval_set=(X_plain.iloc[va_idx], y[va_idx]),
            use_best_model=True,
        )
        oof_cat[va_idx] = positive_proba(cat_model, X_plain.iloc[va_idx])
        test_cat += positive_proba(cat_model, X_plain_test) / N_SPLITS

        nn_val, nn_test_fold, backend = fit_neural_predict(
            X_plain.iloc[tr_idx],
            y[tr_idx],
            X_plain.iloc[va_idx],
            y[va_idx],
            X_plain_test,
            num_cols,
            cat_cols,
            SEED + fold,
        )
        oof_nn[va_idx] = nn_val
        test_nn += nn_test_fold / N_SPLITS
        nn_backends.append(backend)

        print(
            f"Fold {fold} AUC: "
            f"LightGBM={roc_auc_score(y[va_idx], oof_lgb[va_idx]):.6f}, "
            f"CatBoost={roc_auc_score(y[va_idx], oof_cat[va_idx]):.6f}, "
            f"Neural={roc_auc_score(y[va_idx], oof_nn[va_idx]):.6f} ({backend})"
        )

        del lgb_model, cat_model
        gc.collect()

    lgb_auc = roc_auc_score(y, oof_lgb)
    cat_auc = roc_auc_score(y, oof_cat)
    nn_auc = roc_auc_score(y, oof_nn)
    trees_only_auc = roc_auc_score(y, clean_pred(0.5 * oof_lgb + 0.5 * oof_cat))

    blend_candidates = [
        ("lgb55_cat40_nn05", 0.55, 0.40, 0.05),
        ("lgb50_cat40_nn10", 0.50, 0.40, 0.10),
        ("lgb45_cat45_nn10", 0.45, 0.45, 0.10),
        ("lgb45_cat40_nn15", 0.45, 0.40, 0.15),
        ("lgb40_cat50_nn10", 0.40, 0.50, 0.10),
    ]

    scored = []
    for name, wl, wc, wn in blend_candidates:
        pred = clean_pred(wl * oof_lgb + wc * oof_cat + wn * oof_nn)
        scored.append((roc_auc_score(y, pred), name, wl, wc, wn))

    best_auc, best_name, wl, wc, wn = max(scored, key=lambda x: x[0])
    blend_oof = clean_pred(wl * oof_lgb + wc * oof_cat + wn * oof_nn)
    blend_test = clean_pred(wl * test_lgb + wc * test_cat + wn * test_nn)

    print(f"OOF ROC AUC LightGBM: {lgb_auc:.6f}")
    print(f"OOF ROC AUC CatBoost: {cat_auc:.6f}")
    print(f"OOF ROC AUC Neural diversity model: {nn_auc:.6f}")
    print(f"OOF ROC AUC trees-only 50/50 reference: {trees_only_auc:.6f}")
    print(f"OOF ROC AUC selected blend ({best_name}): {best_auc:.6f}")

    sub = sample.copy()
    sub[TARGET] = blend_test
    sub.to_csv(os.path.join(WORKING_DIR, "submission.csv"), index=False)
    sub.to_csv(
        os.path.join(WORKING_DIR, "test_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    oof_df = pd.DataFrame(
        {
            "row": np.arange(len(train)),
            "target": y,
            "prediction": blend_oof,
        }
    )
    oof_df.to_csv(
        os.path.join(WORKING_DIR, "oof_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    review = {
        "metric": "roc_auc",
        "cv_auc": float(best_auc),
        "lightgbm_oof_auc": float(lgb_auc),
        "catboost_oof_auc": float(cat_auc),
        "neural_diversity_oof_auc": float(nn_auc),
        "trees_only_reference_auc": float(trees_only_auc),
        "selected_blend": {
            "name": best_name,
            "weights": {
                "lightgbm": float(wl),
                "catboost": float(wc),
                "realmlp_or_neural_component": float(wn),
            },
        },
        "neural_backends": dict(Counter(nn_backends)),
        "research_hypotheses_llm_claimed_used": ["000091"],
        "files_written": [
            "./working/submission.csv",
            "./working/oof_predictions.csv.gz",
            "./working/test_predictions.csv.gz",
        ],
    }
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
