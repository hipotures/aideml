# Hypothesis 000032: reduce SDSS17 auxiliary-domain weight to test domain-shift robustness
# Auto-generated from response.py on 2026-06-11.
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

from aide_solution_helpers import (
    load_competition_data,
    working_dir,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, early_stopping

    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from joblib import Parallel, delayed

    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False

RANDOM_STATE = 42
N_SPLITS = 5
AUX_SAMPLE_WEIGHT = 0.20
BLEND_STEP = 0.05
BLEND_REFINEMENT_STEP = 0.01
BLEND_REFINEMENT_RADIUS = 0.12
CLASS_OFFSET_STEP = 0.3
CLASS_OFFSET_MAX = 1.5
TEMP_STEP = 0.1
TEMP_MIN = 0.55
TEMP_MAX = 1.45
ENSEMBLE_LOGIT_EPS = 1e-12

NUMERIC_BASE_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
CLASS_ORDER = np.array(["GALAXY", "QSO", "STAR"])
_LGBM_CUDA_SMOKE_OK = None

CATEGORICAL_COLS = [
    "spectral_type",
    "galaxy_population",
    "spectral_type__galaxy_population",
    "spectral_type__sky_alpha_bin_24",
    "galaxy_population__sky_alpha_bin_24",
    "spectral_type__sky_cell_bin",
    "sky_alpha_bin_24",
    "sky_delta_bin_18",
    "sky_cell_bin",
    "spectral_type__galaxy_population__sky_delta_bin_18",
    "mag_min_band",
    "mag_max_band",
    "spectral_type__band_min_band",
    "galaxy_population__band_max_band",
    "sky_alpha_bin_48",
    "sky_delta_bin_36",
    "sky_cell_bin_48",
    "spectral_type__sky_alpha_bin_48",
    "galaxy_population__sky_alpha_bin_48",
    "spectral_type__sky_cell_bin_48",
    "redshift_bin_20",
    "spectral_type__redshift_bin_20",
    "galaxy_population__redshift_bin_20",
    "galactic_l_bin_24",
    "galactic_b_bin_12",
    "galactic_cell_bin",
    "spectral_type__galactic_cell_bin",
    "galaxy_population__galactic_cell_bin",
]

CAT_CONFIG = dict(
    iterations=900, learning_rate=0.08, depth=8, l2_leaf_reg=3.0, random_seed=42
)
XGB_CONFIG = dict(
    n_estimators=650,
    learning_rate=0.08,
    max_depth=7,
    min_child_weight=1,
    subsample=0.90,
    colsample_bytree=0.80,
    reg_lambda=1.0,
    random_state=99,
)
LGBM_CONFIG = dict(
    n_estimators=900,
    learning_rate=0.045,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=45,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    random_state=42,
)


def clean_mag_columns(df, cols):
    out = df.copy()
    for col in cols:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.where(s > -9000, np.nan)
    return out


def load_auxiliary_sdss17():
    required = {"alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"}
    candidates = [
        (Path("./input/star_classification.csv"), ",", {}),
        (Path("./input/star_classification.csv.gz"), ",", {}),
        (Path("./input/original_sdss17/star_classification.csv"), ",", {}),
    ]
    aux = None
    for path, sep, kwargs in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, sep=sep, **kwargs)
        except Exception:
            continue
        if required.issubset(df.columns):
            aux = df.copy()
            break
    if aux is None:
        return None
    aux = aux[["alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"]].copy()
    aux = clean_mag_columns(aux, ["u", "g", "r", "i", "z"])
    for col in NUMERIC_BASE_COLS:
        aux[col] = pd.to_numeric(aux[col], errors="coerce")
    aux["class"] = aux["class"].astype(str).str.strip()
    aux = aux.dropna(subset=["class"])
    aux["id"] = -(np.arange(len(aux), dtype=np.int64) + 1)
    aux["spectral_type"] = "sdss17_aux"
    aux["galaxy_population"] = "sdss17_aux"
    return aux[
        [
            "id",
            "alpha",
            "delta",
            "u",
            "g",
            "r",
            "i",
            "z",
            "redshift",
            "spectral_type",
            "galaxy_population",
            "class",
        ]
    ]


def add_all_features(train_df, test_df):
    out_train = train_df.copy()
    out_test = test_df.copy()

    def add_color(df):
        out = df.copy()
        for a, b in [
            ("u", "g"),
            ("u", "r"),
            ("u", "i"),
            ("u", "z"),
            ("g", "r"),
            ("g", "i"),
            ("g", "z"),
            ("r", "i"),
            ("r", "z"),
        ]:
            out[f"{a}_{b}"] = out[a] - out[b]
        return out

    def add_band(df):
        out = df.copy()
        mags_df = out[["u", "g", "r", "i", "z"]].apply(pd.to_numeric, errors="coerce")
        mags = mags_df.to_numpy()
        idx = np.arange(5, dtype=float)
        sumy = np.nansum(mags, axis=1)
        sumxy = np.nansum(mags * idx[None, :], axis=1)
        slope = (5.0 * sumxy - 10.0 * sumy) / 50.0
        intercept = (sumy - 10.0 * slope) / 5.0
        resid = mags - (intercept[:, None] + slope[:, None] * idx[None, :])
        d1 = np.diff(mags, axis=1)
        d2 = np.diff(d1, axis=1)
        out["mag_sed_slope"] = slope
        out["mag_sed_resid_mean_abs"] = np.nanmean(np.abs(resid), axis=1)
        out["mag_sed_resid_std"] = np.nanstd(resid, axis=1)
        out["mag_sed_d2_mean"] = np.nanmean(d2, axis=1)
        out["mag_sed_d2_abs_mean"] = np.nanmean(np.abs(d2), axis=1)
        out["mag_sed_d2_std"] = np.nanstd(d2, axis=1)
        out["mag_mean"] = np.nanmean(mags, axis=1)
        out["mag_std"] = np.nanstd(mags, axis=1)
        out["mag_range"] = np.nanmax(mags, axis=1) - np.nanmin(mags, axis=1)
        out["band_min_mag"] = mags_df.min(axis=1)
        out["band_max_mag"] = mags_df.max(axis=1)
        out["mag_min_band"] = mags_df.idxmin(axis=1)
        out["mag_max_band"] = mags_df.idxmax(axis=1)
        out["spectral_type__band_min_band"] = (
            out["spectral_type"].astype(str) + "__" + out["mag_min_band"].astype(str)
        )
        out["galaxy_population__band_max_band"] = (
            out["galaxy_population"].astype(str)
            + "__"
            + out["mag_max_band"].astype(str)
        )
        return out

    def add_sky(df):
        out = df.copy()
        a = np.deg2rad(pd.to_numeric(out["alpha"], errors="coerce").to_numpy())
        d = np.deg2rad(pd.to_numeric(out["delta"], errors="coerce").to_numpy())
        out["alpha_sin"] = np.sin(a)
        out["alpha_cos"] = np.cos(a)
        out["delta_sin"] = np.sin(d)
        out["delta_cos"] = np.cos(d)
        out["sky_x"] = np.cos(d) * np.cos(a)
        out["sky_y"] = np.cos(d) * np.sin(a)
        out["sky_z"] = np.sin(d)
        out["sky_xy"] = out["sky_x"] * out["sky_y"]
        out["sky_xz"] = out["sky_x"] * out["sky_z"]
        out["sky_yz"] = out["sky_y"] * out["sky_z"]
        out["sky_x2_minus_y2"] = (
            out["sky_x"] * out["sky_x"] - out["sky_y"] * out["sky_y"]
        )
        for k in (2, 3, 4, 5, 6, 7, 8):
            out[f"alpha_sin{k}"] = np.sin(k * a)
            out[f"alpha_cos{k}"] = np.cos(k * a)
            out[f"delta_sin{k}"] = np.sin(k * d)
            out[f"delta_cos{k}"] = np.cos(k * d)
        return out

    for fn in (add_color, add_band, add_sky):
        out_train = fn(out_train)
        out_test = fn(out_test)

    combined = pd.concat(
        [out_train[["alpha", "delta"]], out_test[["alpha", "delta"]]], ignore_index=True
    ).apply(pd.to_numeric, errors="coerce")
    alpha = np.deg2rad(combined["alpha"].fillna(combined["alpha"].median()).to_numpy())
    delta = np.deg2rad(combined["delta"].fillna(combined["delta"].median()).to_numpy())
    nn = NearestNeighbors(
        n_neighbors=17, algorithm="ball_tree", metric="haversine", n_jobs=-1
    )
    nn.fit(np.column_stack([delta, alpha]))
    distances, _ = nn.kneighbors(np.column_stack([delta, alpha]), return_distance=True)
    arcmin = distances[:, 1:] * (180.0 / np.pi) * 60.0
    feats = pd.DataFrame(
        {
            "sky_nn5_arcmin": arcmin[:, 4],
            "sky_nn10_arcmin": arcmin[:, 9],
            "sky_nn16_arcmin": arcmin[:, 15],
        }
    )
    for k, col in [
        (5, "sky_nn5_arcmin"),
        (10, "sky_nn10_arcmin"),
        (16, "sky_nn16_arcmin"),
    ]:
        feats[f"sky_nn{k}_density"] = np.log1p(
            k / np.maximum(feats[col].to_numpy() ** 2, 1e-6)
        )
    feats["sky_nn5_density_ratio_5_16"] = (
        5 / np.maximum(feats["sky_nn5_arcmin"].to_numpy() ** 2, 1e-6)
    ) / np.maximum(
        16 / np.maximum(feats["sky_nn16_arcmin"].to_numpy() ** 2, 1e-6), 1e-12
    )
    n_train = len(out_train)
    out_train[feats.columns] = feats.iloc[:n_train].to_numpy()
    out_test[feats.columns] = feats.iloc[n_train:].to_numpy()

    def add_bins(df):
        out = df.copy()
        alpha = np.mod(pd.to_numeric(out["alpha"], errors="coerce").to_numpy(), 360.0)
        delta = pd.to_numeric(out["delta"], errors="coerce").to_numpy()
        ab24 = (
            pd.Series(
                pd.cut(
                    alpha,
                    np.linspace(0, 360, 25),
                    labels=False,
                    include_lowest=True,
                    right=False,
                ),
                index=out.index,
            )
            .astype(float)
            .fillna(-1)
            .astype(int)
        )
        db18 = (
            pd.Series(
                pd.cut(
                    delta, np.linspace(-90, 90, 19), labels=False, include_lowest=True
                ),
                index=out.index,
            )
            .astype(float)
            .fillna(-1)
            .astype(int)
        )
        ab48 = (
            pd.Series(
                pd.cut(
                    alpha,
                    np.linspace(0, 360, 49),
                    labels=False,
                    include_lowest=True,
                    right=False,
                ),
                index=out.index,
            )
            .astype(float)
            .fillna(-1)
            .astype(int)
        )
        db36 = (
            pd.Series(
                pd.cut(
                    delta, np.linspace(-90, 90, 37), labels=False, include_lowest=True
                ),
                index=out.index,
            )
            .astype(float)
            .fillna(-1)
            .astype(int)
        )
        out["sky_alpha_bin_24"] = ab24.astype(str)
        out["sky_delta_bin_18"] = db18.astype(str)
        out["sky_cell_bin"] = (ab24 * 19 + db18).astype(str)
        out["sky_alpha_bin_48"] = ab48.astype(str)
        out["sky_delta_bin_36"] = db36.astype(str)
        out["sky_cell_bin_48"] = (ab48 * 36 + db36).astype(str)
        out["sky_alpha_bin_offset_deg"] = (
            (alpha - (ab24 * 15.0 + 7.5) + 180.0) % 360.0
        ) - 180.0
        out["sky_delta_bin_offset_deg"] = delta - (-90.0 + db18 * 10.0 + 5.0)
        return out

    out_train = add_bins(out_train)
    out_test = add_bins(out_test)

    # Combined train/test covariate frequencies and ranks use only features available at prediction time.
    for col in [
        "sky_alpha_bin_24",
        "sky_delta_bin_18",
        "sky_cell_bin",
        "sky_alpha_bin_48",
        "sky_delta_bin_36",
        "sky_cell_bin_48",
    ]:
        freq = pd.concat(
            [out_train[col].astype(str), out_test[col].astype(str)], ignore_index=True
        ).value_counts(normalize=True)
        out_train[f"{col}_freq"] = out_train[col].astype(str).map(freq).fillna(0.0)
        out_test[f"{col}_freq"] = out_test[col].astype(str).map(freq).fillna(0.0)

    for df in (out_train, out_test):
        df["spectral_type__galaxy_population"] = (
            df["spectral_type"].astype(str) + "__" + df["galaxy_population"].astype(str)
        )
        df["spectral_type__sky_alpha_bin_24"] = (
            df["spectral_type"].astype(str) + "__" + df["sky_alpha_bin_24"].astype(str)
        )
        df["galaxy_population__sky_alpha_bin_24"] = (
            df["galaxy_population"].astype(str)
            + "__"
            + df["sky_alpha_bin_24"].astype(str)
        )
        df["spectral_type__sky_cell_bin"] = (
            df["spectral_type"].astype(str) + "__" + df["sky_cell_bin"].astype(str)
        )
        df["spectral_type__galaxy_population__sky_delta_bin_18"] = (
            df["spectral_type"].astype(str)
            + "__"
            + df["galaxy_population"].astype(str)
            + "__"
            + df["sky_delta_bin_18"].astype(str)
        )
        df["spectral_type__sky_alpha_bin_48"] = (
            df["spectral_type"].astype(str) + "__" + df["sky_alpha_bin_48"].astype(str)
        )
        df["galaxy_population__sky_alpha_bin_48"] = (
            df["galaxy_population"].astype(str)
            + "__"
            + df["sky_alpha_bin_48"].astype(str)
        )
        df["spectral_type__sky_cell_bin_48"] = (
            df["spectral_type"].astype(str) + "__" + df["sky_cell_bin_48"].astype(str)
        )
        z = pd.to_numeric(df["redshift"], errors="coerce")
        df["redshift_abs"] = z.abs()
        df["redshift_sq"] = z**2
        df["redshift_log_abs"] = np.log1p(z.abs())
        df["redshift_signed_log_abs"] = np.sign(z) * np.log1p(z.abs())

    zc = pd.concat(
        [
            pd.to_numeric(out_train["redshift"], errors="coerce"),
            pd.to_numeric(out_test["redshift"], errors="coerce"),
        ],
        ignore_index=True,
    )
    q = np.unique(np.quantile(zc[np.isfinite(zc)], np.linspace(0, 1, 21)))
    codes = (
        pd.cut(zc, bins=q, labels=False, include_lowest=True, duplicates="drop")
        .astype(float)
        .fillna(-1)
        .astype(int)
    )
    out_train["redshift_bin_20"] = "zbin_" + codes.iloc[:n_train].astype(str)
    out_test["redshift_bin_20"] = "zbin_" + codes.iloc[n_train:].astype(str)

    for df in (out_train, out_test):
        df["spectral_type__redshift_bin_20"] = (
            df["spectral_type"].astype(str) + "__" + df["redshift_bin_20"].astype(str)
        )
        df["galaxy_population__redshift_bin_20"] = (
            df["galaxy_population"].astype(str)
            + "__"
            + df["redshift_bin_20"].astype(str)
        )

    freq = pd.concat(
        [out_train["redshift_bin_20"], out_test["redshift_bin_20"]], ignore_index=True
    ).value_counts(normalize=True)
    out_train["redshift_bin_20_freq"] = (
        out_train["redshift_bin_20"].map(freq).fillna(0.0)
    )
    out_test["redshift_bin_20_freq"] = out_test["redshift_bin_20"].map(freq).fillna(0.0)

    def add_galactic(df):
        out = df.copy()
        ra = np.deg2rad(pd.to_numeric(out["alpha"], errors="coerce").to_numpy())
        dec = np.deg2rad(pd.to_numeric(out["delta"], errors="coerce").to_numpy())
        x_eq = np.cos(dec) * np.cos(ra)
        y_eq = np.cos(dec) * np.sin(ra)
        z_eq = np.sin(dec)
        xg = -0.0548755604 * x_eq - 0.8734370902 * y_eq - 0.4838350155 * z_eq
        yg = 0.4941094279 * x_eq - 0.44482963 * y_eq + 0.7469822445 * z_eq
        zg = -0.8676661490 * x_eq - 0.1980763734 * y_eq + 0.4559837762 * z_eq
        l = (np.degrees(np.arctan2(yg, xg)) + 360.0) % 360.0
        b = np.degrees(np.arcsin(np.clip(zg, -1, 1)))
        out["galactic_l"] = l
        out["galactic_b"] = b
        out["galactic_l_sin"] = np.sin(np.deg2rad(l))
        out["galactic_l_cos"] = np.cos(np.deg2rad(l))
        out["galactic_b_sin"] = np.sin(np.deg2rad(b))
        out["galactic_b_cos"] = np.cos(np.deg2rad(b))
        out["galactic_b_abs"] = np.abs(b)
        lb = (
            pd.Series(
                pd.cut(
                    l,
                    np.linspace(0, 360, 25),
                    labels=False,
                    include_lowest=True,
                    right=False,
                ),
                index=out.index,
            )
            .astype(float)
            .fillna(-1)
            .astype(int)
        )
        bb = (
            pd.Series(
                pd.cut(b, np.linspace(-90, 90, 13), labels=False, include_lowest=True),
                index=out.index,
            )
            .astype(float)
            .fillna(-1)
            .astype(int)
        )
        out["galactic_l_bin_24"] = lb.astype(str)
        out["galactic_b_bin_12"] = bb.astype(str)
        out["galactic_cell_bin"] = (lb * 12 + bb).astype(str)
        out["spectral_type__galactic_cell_bin"] = (
            out["spectral_type"].astype(str)
            + "__"
            + out["galactic_cell_bin"].astype(str)
        )
        out["galaxy_population__galactic_cell_bin"] = (
            out["galaxy_population"].astype(str)
            + "__"
            + out["galactic_cell_bin"].astype(str)
        )
        return out

    out_train = add_galactic(out_train)
    out_test = add_galactic(out_test)

    for col in ["galactic_l_bin_24", "galactic_b_bin_12", "galactic_cell_bin"]:
        freq = pd.concat(
            [out_train[col].astype(str), out_test[col].astype(str)], ignore_index=True
        ).value_counts(normalize=True)
        out_train[f"{col}_freq"] = out_train[col].astype(str).map(freq).fillna(0.0)
        out_test[f"{col}_freq"] = out_test[col].astype(str).map(freq).fillna(0.0)

    ranks = (
        pd.concat(
            [
                out_train[["u", "g", "r", "i", "z", "redshift"]].astype(float),
                out_test[["u", "g", "r", "i", "z", "redshift"]].astype(float),
            ],
            ignore_index=True,
        )
        .rank(pct=True)
        .add_suffix("_qrank")
    )
    out_train[ranks.columns] = ranks.iloc[:n_train].to_numpy()
    out_test[ranks.columns] = ranks.iloc[n_train:].to_numpy()
    return out_train, out_test


def compute_aux_locality_weights(main_df, full_df, cols):
    base = main_df[cols].apply(pd.to_numeric, errors="coerce")
    full = full_df[cols].apply(pd.to_numeric, errors="coerce")
    med = base.median(numeric_only=True)
    mad = (base - med).abs().median(numeric_only=True).replace(0.0, 1.0)
    base_std = ((base - med) / mad).fillna(0.0)
    full_std = ((full - med) / mad).fillna(0.0)
    base_dist = np.sqrt((base_std**2).sum(axis=1).to_numpy())
    full_dist = np.sqrt((full_std**2).sum(axis=1).to_numpy())
    scale = np.quantile(base_dist, 0.90)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.clip(np.exp(-full_dist / scale), 0.08, 1.0)


def compute_fold_sample_weights(
    train_idx, is_main_train, y_train, aux_base_weight, aux_locality
):
    weights = np.ones(len(train_idx), dtype=float)
    local_is_main = is_main_train[train_idx]
    local_is_aux = ~local_is_main
    if not np.any(local_is_aux):
        return weights
    y_local = np.asarray(y_train, dtype=object)[train_idx].astype(str)
    ser = pd.Series(y_local)
    main_counts = ser[local_is_main].value_counts()
    aux_counts = ser[local_is_aux].value_counts()
    raw_ratio = {
        cls: (
            float(main_counts.get(cls, 0.0)) / float(aux_n)
            if aux_n > 0 and main_counts.get(cls, 0.0) > 0
            else 1.0
        )
        for cls, aux_n in aux_counts.items()
    }
    center = np.median(list(raw_ratio.values())) if raw_ratio else 1.0
    if not np.isfinite(center) or center <= 0:
        center = 1.0
    y_aux = y_local[local_is_aux]
    locality_aux = aux_locality[train_idx][local_is_aux]
    class_mult = np.array(
        [raw_ratio.get(cls, 1.0) / center for cls in y_aux], dtype=float
    )
    weights[local_is_aux] = (
        aux_base_weight
        * np.clip(class_mult, 0.15, 5.0)
        * np.clip(locality_aux, 0.08, 1.0)
    )
    return weights


def make_cat_model():
    return dict(
        iterations=CAT_CONFIG["iterations"],
        learning_rate=CAT_CONFIG["learning_rate"],
        depth=CAT_CONFIG["depth"],
        l2_leaf_reg=CAT_CONFIG["l2_leaf_reg"],
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=CAT_CONFIG["random_seed"],
        verbose=False,
    )


def fit_cat_with_fallback(X_tr, y_tr, X_val, y_val, cat_features, sample_weight):
    params = make_cat_model()
    fit_kwargs = dict(
        cat_features=cat_features,
        sample_weight=sample_weight,
        verbose=False,
        eval_set=(X_val, y_val),
        early_stopping_rounds=60,
    )
    try:
        model = CatBoostClassifier(
            task_type="GPU", devices="0", gpu_ram_part=0.8, **params
        )
        model.fit(X_tr, y_tr, **fit_kwargs)
        return model
    except Exception as exc:
        print(
            f"CatBoost GPU fit failed ({type(exc).__name__}); falling back to CPU CatBoost.",
            flush=True,
        )
        model = CatBoostClassifier(**params)
        try:
            model.fit(X_tr, y_tr, **fit_kwargs)
        except Exception:
            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_features,
                sample_weight=sample_weight,
                verbose=False,
            )
        return model


def fit_xgb_with_fallback(X_tr, y_tr, X_val, y_val, sample_weight):
    if not HAS_XGB:
        return None
    base = dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_jobs=-1,
        verbosity=0,
        **XGB_CONFIG,
    )
    for device_params in (
        {"tree_method": "hist", "device": "cuda"},
        {"tree_method": "hist"},
    ):
        model = XGBClassifier(**base, **device_params)
        try:
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                verbose=False,
                early_stopping_rounds=40,
            )
            return model
        except Exception:
            try:
                model.fit(
                    X_tr,
                    y_tr,
                    sample_weight=sample_weight,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                return model
            except Exception:
                continue
    return None


def lightgbm_cuda_smoke_ok():
    global _LGBM_CUDA_SMOKE_OK
    if _LGBM_CUDA_SMOKE_OK is not None:
        return _LGBM_CUDA_SMOKE_OK
    if not HAS_LGBM:
        _LGBM_CUDA_SMOKE_OK = False
        return False
    smoke_code = "import numpy as np\nfrom lightgbm import LGBMClassifier\nX=np.random.default_rng(0).normal(size=(96,6)); y=np.random.default_rng(1).integers(0,3,size=96)\nm=LGBMClassifier(objective='multiclass',num_class=3,n_estimators=8,device_type='cuda',verbosity=-1)\nm.fit(X,y)\nprint('LIGHTGBM_CUDA_OK')"
    try:
        result = subprocess.run(
            [sys.executable, "-c", smoke_code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        _LGBM_CUDA_SMOKE_OK = (
            result.returncode == 0 and "LIGHTGBM_CUDA_OK" in result.stdout
        )
    except Exception:
        _LGBM_CUDA_SMOKE_OK = False
    if not _LGBM_CUDA_SMOKE_OK:
        print("LightGBM CUDA smoke test failed; using CPU fallback.", flush=True)
    return _LGBM_CUDA_SMOKE_OK


def fit_lgbm_with_fallback(X_tr, y_tr, X_val, y_val, sample_weight):
    if not HAS_LGBM:
        return None
    attempts = []
    if lightgbm_cuda_smoke_ok():
        attempts.append(("cuda", {"device_type": "cuda"}))
    attempts.append(("cpu", {}))
    for label, device_params in attempts:
        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1,
            **device_params,
            **LGBM_CONFIG,
        )
        try:
            model.fit(
                X_tr,
                y_tr,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[early_stopping(50, verbose=False)],
            )
            return model
        except Exception as exc:
            if label == "cuda":
                print(
                    f"LightGBM CUDA fit failed ({type(exc).__name__}); falling back to CPU.",
                    flush=True,
                )
            try:
                model.fit(X_tr, y_tr, sample_weight=sample_weight)
                return model
            except Exception:
                continue
    return None


def align_numeric_proba(probabilities, model_classes, num_classes):
    out = np.zeros((probabilities.shape[0], num_classes), dtype=np.float64)
    for j, cls in enumerate(model_classes):
        idx = int(cls)
        if 0 <= idx < num_classes:
            out[:, idx] = probabilities[:, j]
    return out


def score_predictions(y_true, probs, class_order):
    return float(balanced_accuracy_score(y_true, class_order[np.argmax(probs, axis=1)]))


def softmax_from_logits(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(shifted)
    return expv / np.sum(expv, axis=1, keepdims=True)


def aggregate_logits(weights, model_names, model_probs):
    logits = np.zeros_like(model_probs[model_names[0]], dtype=np.float64)
    for w, name in zip(weights, model_names):
        logits += w * np.log(np.clip(model_probs[name], ENSEMBLE_LOGIT_EPS, 1.0))
    return logits


def blend_candidate_score(weights, model_names, oof_probs, y_true, class_order):
    pred = class_order[
        np.argmax(aggregate_logits(weights, model_names, oof_probs), axis=1)
    ]
    return float(balanced_accuracy_score(y_true, pred))


def generate_simplex_candidates(n_models, step):
    if n_models == 1:
        return [(1.0,)]
    values = np.round(np.arange(0.0, 1.0 + 1e-12, step), 10)
    if n_models == 2:
        return [(float(w), float(1.0 - w)) for w in values]
    out = []
    for w1 in values:
        for w2 in values:
            w3 = 1.0 - w1 - w2
            if -1e-12 <= w3 <= 1.0 + 1e-12:
                out.append((float(w1), float(w2), float(max(0.0, w3))))
            if w3 < -1e-12:
                break
    return out


def generate_local_simplex_candidates(
    center, step=BLEND_REFINEMENT_STEP, radius=BLEND_REFINEMENT_RADIUS
):
    center = np.asarray(center, dtype=float)
    center = center / center.sum()
    deltas = np.round(np.arange(-radius, radius + 1e-12, step), 10)
    candidates = set()
    if len(center) == 2:
        for d in deltas:
            w0 = center[0] + d
            if 0 <= w0 <= 1:
                candidates.add((float(np.round(w0, 10)), float(np.round(1 - w0, 10))))
    else:
        for d0 in deltas:
            for d1 in deltas:
                w0 = center[0] + d0
                w1 = center[1] + d1
                w2 = 1 - w0 - w1
                if 0 <= w0 <= 1 and 0 <= w1 <= 1 and 0 <= w2 <= 1:
                    candidates.add(
                        (
                            float(np.round(w0, 10)),
                            float(np.round(w1, 10)),
                            float(np.round(w2, 10)),
                        )
                    )
    return sorted(candidates)


def evaluate_candidate_scores(candidates, model_names, oof_probs, y_true, class_order):
    if HAS_JOBLIB and len(candidates) > 120:
        n_jobs = min(16, os.cpu_count() or 1)
        print(
            f"Evaluating {len(candidates)} blend candidates with {n_jobs} workers",
            flush=True,
        )
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(blend_candidate_score)(
                w, tuple(model_names), oof_probs, y_true, class_order
            )
            for w in candidates
        )
    print(f"Evaluating {len(candidates)} blend candidates sequentially", flush=True)
    return [
        blend_candidate_score(w, model_names, oof_probs, y_true, class_order)
        for w in candidates
    ]


def select_best_simplex_weights(y_true, preds_by_model, model_order, class_order):
    if len(model_order) == 1:
        return (1.0,)
    coarse = generate_simplex_candidates(len(model_order), BLEND_STEP)
    coarse_scores = evaluate_candidate_scores(
        coarse, model_order, preds_by_model, y_true, class_order
    )
    best = coarse[int(np.argmax(coarse_scores))]
    local = [c for c in generate_local_simplex_candidates(best) if c not in set(coarse)]
    if local:
        local_scores = evaluate_candidate_scores(
            local, model_order, preds_by_model, y_true, class_order
        )
        candidates = coarse + local
        scores = list(coarse_scores) + list(local_scores)
        best = candidates[int(np.argmax(scores))]
    return tuple(float(w) for w in best)


def generate_temp_offset_candidates():
    temps = np.round(np.arange(TEMP_MIN, TEMP_MAX + 1e-12, TEMP_STEP), 6)
    offsets = np.round(
        np.arange(-CLASS_OFFSET_MAX, CLASS_OFFSET_MAX + 1e-12, CLASS_OFFSET_STEP), 6
    )
    return [
        (float(t), float(a), float(b)) for t in temps for a in offsets for b in offsets
    ]


def apply_temp_and_offsets(logits, params):
    t, a, b = params
    out = np.array(logits, copy=True) / max(float(t), 1e-12)
    out[:, 0] += float(a)
    out[:, 1] += float(b)
    return out


def temp_offset_score(params, logits, y_true, class_order):
    pred = class_order[np.argmax(apply_temp_and_offsets(logits, params), axis=1)]
    return float(balanced_accuracy_score(y_true, pred))


def evaluate_temp_offset_scores(candidates, logits, y_true, class_order, fold_id):
    if HAS_JOBLIB and len(candidates) > 120:
        n_jobs = min(16, os.cpu_count() or 1)
        print(
            f"Evaluating {len(candidates)} temp-offset candidates with {n_jobs} workers (fold={fold_id})",
            flush=True,
        )
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(temp_offset_score)(p, logits, y_true, class_order)
            for p in candidates
        )
    print(
        f"Evaluating {len(candidates)} temp-offset candidates sequentially (fold={fold_id})",
        flush=True,
    )
    return [temp_offset_score(p, logits, y_true, class_order) for p in candidates]


def main():
    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()
        for col in NUMERIC_BASE_COLS:
            train[col] = pd.to_numeric(train[col], errors="coerce")
            test[col] = pd.to_numeric(test[col], errors="coerce")

        aux = load_auxiliary_sdss17()
        if aux is not None:
            print(f"Loaded auxiliary SDSS17 rows: {len(aux)}", flush=True)
            train_full = pd.concat([train.copy(), aux], ignore_index=True, sort=False)
        else:
            print(
                "No auxiliary SDSS17 rows loaded; using competition train only.",
                flush=True,
            )
            train_full = train.copy()

        n_main = len(train)
        main_targets = train["class"].astype(str).to_numpy()
        all_targets = train_full["class"].astype(str).to_numpy()
        is_main_train = np.zeros(len(train_full), dtype=bool)
        is_main_train[:n_main] = True
        aux_idx = np.where(~is_main_train)[0]

        aux_locality = np.ones(len(train_full), dtype=float)
        if len(aux_idx) > 0:
            aux_locality = compute_aux_locality_weights(
                train_full.iloc[:n_main], train_full, NUMERIC_BASE_COLS
            )
            aux_locality[:n_main] = 1.0
            print(
                f"Auxiliary locality weights: min={aux_locality.min():.4f}, mean={aux_locality.mean():.4f}, max={aux_locality.max():.4f}",
                flush=True,
            )

        train_full, test = add_all_features(train_full, test)

        for col in CATEGORICAL_COLS:
            if col not in train_full.columns:
                train_full[col] = "missing"
                test[col] = "missing"
            train_full[col] = train_full[col].astype(str).fillna("missing")
            test[col] = test[col].astype(str).fillna("missing")

        numeric_cols = [
            c
            for c in train_full.columns
            if c not in ("id", "class") and c not in CATEGORICAL_COLS
        ]
        med = train_full[numeric_cols].median(numeric_only=True)
        for col in numeric_cols:
            fill = med.get(col, 0.0)
            train_full[col] = train_full[col].fillna(fill)
            test[col] = test[col].fillna(fill)

        feature_cols = [c for c in train_full.columns if c not in ("id", "class")]
        X_cat = train_full[feature_cols]
        X_test_cat = test[feature_cols]
        cat_features = [i for i, c in enumerate(feature_cols) if c in CATEGORICAL_COLS]

        class_order = CLASS_ORDER.copy()
        class_to_idx = {cls: i for i, cls in enumerate(class_order)}
        all_targets_code = (
            pd.Series(all_targets).map(class_to_idx).to_numpy(dtype=np.int64)
        )
        num_classes = len(class_order)

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train_cat_encoded = encoder.fit_transform(
            train_full[CATEGORICAL_COLS].astype(str)
        ).astype(np.int32)
        test_cat_encoded = encoder.transform(test[CATEGORICAL_COLS].astype(str)).astype(
            np.int32
        )

        train_noncat = X_cat.drop(columns=CATEGORICAL_COLS).reset_index(drop=True)
        test_noncat = X_test_cat.drop(columns=CATEGORICAL_COLS).reset_index(drop=True)
        train_tree = pd.concat(
            [
                train_noncat,
                pd.DataFrame(
                    train_cat_encoded,
                    columns=[f"cat_{c}" for c in CATEGORICAL_COLS],
                    index=train_noncat.index,
                ),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        test_tree = pd.concat(
            [
                test_noncat,
                pd.DataFrame(
                    test_cat_encoded,
                    columns=[f"cat_{c}" for c in CATEGORICAL_COLS],
                    index=test_noncat.index,
                ),
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        main_indices = np.arange(n_main, dtype=np.int64)
        folds = list(skf.split(main_indices, main_targets))
        print(
            f"Prepared {len(folds)} stratified folds on competition rows only.",
            flush=True,
        )

    model_names = ["catboost", "xgboost", "lightgbm"]
    oof_probs = {
        name: np.zeros((n_main, num_classes), dtype=np.float64) for name in model_names
    }
    fold_predictions = []
    fold_scores = {name: [] for name in model_names}
    fold_times = {name: [] for name in model_names}
    model_available = {"catboost": True, "xgboost": HAS_XGB, "lightgbm": HAS_LGBM}

    with aide_stage("fit_predict_fold_stage"):
        for fold_idx, (tr_pos, va_pos) in enumerate(folds, start=1):
            main_tr_idx = main_indices[tr_pos]
            main_va_idx = main_indices[va_pos]
            train_idx = (
                main_tr_idx
                if len(aux_idx) == 0
                else np.concatenate([main_tr_idx, aux_idx])
            )
            y_tr = all_targets[train_idx]
            y_va = all_targets[main_va_idx]
            y_tr_code = all_targets_code[train_idx]
            y_va_code = all_targets_code[main_va_idx]

            fold_aux_weight = compute_fold_sample_weights(
                train_idx, is_main_train, all_targets, AUX_SAMPLE_WEIGHT, aux_locality
            )
            balanced_sw = compute_sample_weight(
                class_weight="balanced", y=y_tr_code
            ).astype(float)
            tree_sw = balanced_sw * fold_aux_weight

            X_tr_cat = X_cat.iloc[train_idx]
            X_va_cat = X_cat.iloc[main_va_idx]
            X_tr_tree = train_tree.iloc[train_idx]
            X_va_tree = train_tree.iloc[main_va_idx]
            fold_record = {"va_idx": main_va_idx, "oof": {}, "test": {}}

            log_stage(
                f"event=progress|stage=fit_predict_fold_stage|fold={fold_idx}|model=catboost"
            )
            start = time.perf_counter()
            cat_model = fit_cat_with_fallback(
                X_tr_cat, y_tr, X_va_cat, y_va, cat_features, tree_sw
            )
            cat_val_prob = (
                pd.DataFrame(
                    cat_model.predict_proba(X_va_cat), columns=cat_model.classes_
                )
                .reindex(columns=class_order, fill_value=0.0)
                .to_numpy()
            )
            cat_test_prob = (
                pd.DataFrame(
                    cat_model.predict_proba(X_test_cat), columns=cat_model.classes_
                )
                .reindex(columns=class_order, fill_value=0.0)
                .to_numpy()
            )
            oof_probs["catboost"][main_va_idx] = cat_val_prob
            fold_record["oof"]["catboost"] = cat_val_prob
            fold_record["test"]["catboost"] = cat_test_prob
            score = score_predictions(y_va, cat_val_prob, class_order)
            elapsed = time.perf_counter() - start
            fold_scores["catboost"].append(score)
            fold_times["catboost"].append(elapsed)
            print(
                f"Fold {fold_idx} | catboost | balanced_accuracy={score:.6f} | runtime_s={elapsed:.1f}",
                flush=True,
            )

            log_stage(
                f"event=progress|stage=fit_predict_fold_stage|fold={fold_idx}|model=xgboost"
            )
            start = time.perf_counter()
            if model_available["xgboost"]:
                xgb_model = fit_xgb_with_fallback(
                    X_tr_tree, y_tr_code, X_va_tree, y_va_code, tree_sw
                )
                if xgb_model is None:
                    model_available["xgboost"] = False
                else:
                    xgb_val_prob = align_numeric_proba(
                        xgb_model.predict_proba(X_va_tree),
                        xgb_model.classes_,
                        num_classes,
                    )
                    xgb_test_prob = align_numeric_proba(
                        xgb_model.predict_proba(test_tree),
                        xgb_model.classes_,
                        num_classes,
                    )
                    oof_probs["xgboost"][main_va_idx] = xgb_val_prob
                    fold_record["oof"]["xgboost"] = xgb_val_prob
                    fold_record["test"]["xgboost"] = xgb_test_prob
                    score = score_predictions(y_va, xgb_val_prob, class_order)
                    elapsed = time.perf_counter() - start
                    fold_scores["xgboost"].append(score)
                    fold_times["xgboost"].append(elapsed)
                    print(
                        f"Fold {fold_idx} | xgboost | balanced_accuracy={score:.6f} | runtime_s={elapsed:.1f}",
                        flush=True,
                    )
            if not model_available["xgboost"]:
                print(
                    f"Fold {fold_idx} | xgboost | unavailable_after_failure | runtime_s={time.perf_counter() - start:.1f}",
                    flush=True,
                )

            log_stage(
                f"event=progress|stage=fit_predict_fold_stage|fold={fold_idx}|model=lightgbm"
            )
            start = time.perf_counter()
            if model_available["lightgbm"]:
                lgb_model = fit_lgbm_with_fallback(
                    X_tr_tree, y_tr_code, X_va_tree, y_va_code, tree_sw
                )
                if lgb_model is None:
                    model_available["lightgbm"] = False
                else:
                    lgb_val_prob = align_numeric_proba(
                        lgb_model.predict_proba(X_va_tree),
                        lgb_model.classes_,
                        num_classes,
                    )
                    lgb_test_prob = align_numeric_proba(
                        lgb_model.predict_proba(test_tree),
                        lgb_model.classes_,
                        num_classes,
                    )
                    oof_probs["lightgbm"][main_va_idx] = lgb_val_prob
                    fold_record["oof"]["lightgbm"] = lgb_val_prob
                    fold_record["test"]["lightgbm"] = lgb_test_prob
                    score = score_predictions(y_va, lgb_val_prob, class_order)
                    elapsed = time.perf_counter() - start
                    fold_scores["lightgbm"].append(score)
                    fold_times["lightgbm"].append(elapsed)
                    print(
                        f"Fold {fold_idx} | lightgbm | balanced_accuracy={score:.6f} | runtime_s={elapsed:.1f}",
                        flush=True,
                    )
            if not model_available["lightgbm"]:
                print(
                    f"Fold {fold_idx} | lightgbm | unavailable_after_failure | runtime_s={time.perf_counter() - start:.1f}",
                    flush=True,
                )

            fold_predictions.append(fold_record)

    with aide_stage("score_stage"):
        valid_model_names = []
        for name in model_names:
            if (
                not model_available[name]
                or len(fold_scores[name]) != N_SPLITS
                or np.any(oof_probs[name].sum(axis=1) <= 0)
            ):
                print(
                    f"Model {name} excluded: incomplete predictions or training failed.",
                    flush=True,
                )
                continue
            valid_model_names.append(name)
            print(
                f"Model summary | {name} | mean_cv_balanced_accuracy={np.mean(fold_scores[name]):.6f} | oof_balanced_accuracy={score_predictions(main_targets, oof_probs[name], class_order):.6f} | mean_runtime_s={np.mean(fold_times[name]):.1f}",
                flush=True,
            )

        if not valid_model_names:
            raise RuntimeError(
                "No model in the panel produced complete OOF predictions."
            )

        common_model_names = [
            name
            for name in valid_model_names
            if all(name in fold["oof"] for fold in fold_predictions)
        ]
        if not common_model_names:
            raise RuntimeError("No model with complete fold predictions for blending.")

        oof_logit_uncal = np.zeros((n_main, num_classes), dtype=np.float64)
        oof_logit_base = np.zeros((n_main, num_classes), dtype=np.float64)
        test_logit_base = np.zeros((len(test), num_classes), dtype=np.float64)
        temp_offset_candidates = generate_temp_offset_candidates()
        print(
            f"Candidate temperature-offset grid size: {len(temp_offset_candidates)}",
            flush=True,
        )

        fold_weight_report = []
        fold_temp_report = []
        for fold_id, fold_record in enumerate(fold_predictions, start=1):
            va_idx = fold_record["va_idx"]
            va_targets = main_targets[va_idx]
            va_model_probs = {
                name: fold_record["oof"][name] for name in common_model_names
            }
            best_weights = select_best_simplex_weights(
                va_targets, va_model_probs, common_model_names, class_order
            )
            fold_weight_report.append(
                "fold="
                + str(fold_id)
                + " "
                + ", ".join(
                    f"{n}={w:.3f}" for n, w in zip(common_model_names, best_weights)
                )
            )

            va_logits = aggregate_logits(
                best_weights, common_model_names, va_model_probs
            )
            oof_logit_uncal[va_idx] = va_logits
            temp_scores = evaluate_temp_offset_scores(
                temp_offset_candidates, va_logits, va_targets, class_order, fold_id
            )
            best_temp_offset = (
                temp_offset_candidates[int(np.argmax(temp_scores))]
                if temp_scores
                else (1.0, 0.0, 0.0)
            )
            fold_temp_report.append(
                f"fold={fold_id} T={best_temp_offset[0]:.2f}, GAL={best_temp_offset[1]:+.2f}, QSO={best_temp_offset[2]:+.2f}"
            )
            oof_logit_base[va_idx] = apply_temp_and_offsets(va_logits, best_temp_offset)

            fold_test_logits = aggregate_logits(
                best_weights, common_model_names, fold_record["test"]
            )
            test_logit_base += apply_temp_and_offsets(
                fold_test_logits, best_temp_offset
            )

        test_logit_base /= float(N_SPLITS)
        base_score = score_predictions(
            main_targets, softmax_from_logits(oof_logit_uncal), class_order
        )
        best_oof_matrix = softmax_from_logits(oof_logit_base)
        best_oof_pred = class_order[np.argmax(best_oof_matrix, axis=1)]
        best_test_probs = softmax_from_logits(test_logit_base)
        best_test_pred = class_order[np.argmax(best_test_probs, axis=1)]
        best_oof_score = score_predictions(main_targets, best_oof_matrix, class_order)

        print(
            f"Base blend score (before fold-wise temp+offset): {base_score:.6f}",
            flush=True,
        )
        print(
            f"Fold-specific blend weight vectors: {' | '.join(fold_weight_report)}",
            flush=True,
        )
        print(
            f"Fold-specific temp-offset parameters: {' | '.join(fold_temp_report)}",
            flush=True,
        )
        print(
            f"Primary validation metric (fold-specific blend + fold-wise temp+offset): {best_oof_score:.6f}",
            flush=True,
        )

    with aide_stage("write_outputs_stage"):
        write_oof_predictions(
            pd.DataFrame(
                {
                    "row": np.arange(n_main, dtype=np.int64),
                    "target": main_targets,
                    "prediction": best_oof_pred,
                }
            )
        )

        test_prob_frame = pd.DataFrame(best_test_probs, columns=class_order)
        test_prob_frame.insert(0, "id", test["id"].to_numpy())
        test_prob_frame = sample_sub[["id"]].merge(test_prob_frame, on="id", how="left")
        if test_prob_frame[class_order.tolist()].isna().any().any():
            fallback = np.nanmean(best_test_probs, axis=0)
            for i, cls in enumerate(class_order):
                test_prob_frame[cls] = test_prob_frame[cls].fillna(fallback[i])
        write_test_predictions(test_prob_frame)

        pred_map = pd.Series(best_test_pred, index=test["id"])
        submission = sample_sub[["id"]].copy()
        submission["class"] = submission["id"].map(pred_map)
        if submission["class"].isna().any():
            fallback_label = class_order[
                int(np.argmax(np.nanmean(best_test_probs, axis=0)))
            ]
            submission["class"] = submission["class"].fillna(fallback_label)
        write_submission(submission)
        print(f"Artifacts written to {working_dir()}", flush=True)


if __name__ == "__main__":
    main()
