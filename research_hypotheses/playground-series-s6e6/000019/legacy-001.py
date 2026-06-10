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
    write_validation_predictions,
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


RANDOM_STATE = 42
N_SPLITS = 5
AUX_SAMPLE_WEIGHT = 0.35
NUMERIC_BASE_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

CAT_CONFIG = {
    "iterations": 900,
    "learning_rate": 0.08,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
}

XGB_CONFIG = {
    "n_estimators": 650,
    "learning_rate": 0.08,
    "max_depth": 7,
    "min_child_weight": 1,
    "subsample": 0.90,
    "colsample_bytree": 0.80,
    "reg_lambda": 1.0,
    "random_state": 99,
}

LGBM_CONFIG = {
    "n_estimators": 900,
    "learning_rate": 0.045,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 45,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "random_state": 42,
}

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


def clean_mag_columns(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = s.where(s > -9000, np.nan)
    return out


def load_auxiliary_sdss17() -> pd.DataFrame | None:
    required = {"alpha", "delta", "u", "g", "r", "i", "z", "redshift", "class"}
    candidates = [
        (Path("./input/star_classification.csv"), ",", {}),
        (Path("./input/star_classification.csv.gz"), ",", {}),
        (Path("./input/original_sdss17/star_classification.csv"), ",", {}),
        (Path("./input/star_classification.txt"), r"\s+", {"engine": "python"}),
        (
            Path("./input/original_sdss17/star_classification.txt"),
            r"\s+",
            {"engine": "python"},
        ),
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
    if aux.empty:
        return None
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


def add_color_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for left, right in [
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
        out[f"{left}_{right}"] = out[left] - out[right]
    return out


def add_band_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mags = (
        out[["u", "g", "r", "i", "z"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    )
    idx = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    sumy = np.nansum(mags, axis=1)
    sumxy = np.nansum(mags * idx[None, :], axis=1)
    slope = (5.0 * sumxy - 10.0 * sumy) / 50.0
    intercept = (sumy - 10.0 * slope) / 5.0
    trend = intercept[:, None] + slope[:, None] * idx[None, :]
    resid = mags - trend
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
    return out


def add_band_extrema_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mags = out[["u", "g", "r", "i", "z"]].apply(pd.to_numeric, errors="coerce")
    out["band_min_mag"] = mags.min(axis=1)
    out["band_max_mag"] = mags.max(axis=1)
    out["mag_min_band"] = mags.idxmin(axis=1)
    out["mag_max_band"] = mags.idxmax(axis=1)
    return out


def add_extrema_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type__band_min_band"] = (
        out["spectral_type"].astype(str).fillna("missing")
        + "__"
        + out["mag_min_band"].astype(str).fillna("missing")
    )
    out["galaxy_population__band_max_band"] = (
        out["galaxy_population"].astype(str).fillna("missing")
        + "__"
        + out["mag_max_band"].astype(str).fillna("missing")
    )
    return out


def add_sky_circular_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alpha_rad = np.deg2rad(pd.to_numeric(out["alpha"], errors="coerce").to_numpy())
    delta_rad = np.deg2rad(pd.to_numeric(out["delta"], errors="coerce").to_numpy())
    out["alpha_sin"] = np.sin(alpha_rad)
    out["alpha_cos"] = np.cos(alpha_rad)
    out["delta_sin"] = np.sin(delta_rad)
    out["delta_cos"] = np.cos(delta_rad)
    return out


def add_sky_cartesian_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alpha_rad = np.deg2rad(pd.to_numeric(out["alpha"], errors="coerce").to_numpy())
    delta_rad = np.deg2rad(pd.to_numeric(out["delta"], errors="coerce").to_numpy())
    out["sky_x"] = np.cos(delta_rad) * np.cos(alpha_rad)
    out["sky_y"] = np.cos(delta_rad) * np.sin(alpha_rad)
    out["sky_z"] = np.sin(delta_rad)
    return out


def add_sky_cartesian_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    x = pd.to_numeric(out["sky_x"], errors="coerce")
    y = pd.to_numeric(out["sky_y"], errors="coerce")
    z = pd.to_numeric(out["sky_z"], errors="coerce")
    out["sky_xy"] = x * y
    out["sky_xz"] = x * z
    out["sky_yz"] = y * z
    out["sky_x2_minus_y2"] = x * x - y * y
    return out


def add_sky_harmonic_features(df: pd.DataFrame, order: int) -> pd.DataFrame:
    out = df.copy()
    a = np.deg2rad(pd.to_numeric(out["alpha"], errors="coerce").to_numpy())
    d = np.deg2rad(pd.to_numeric(out["delta"], errors="coerce").to_numpy())
    k = float(order)
    out[f"alpha_sin{k:g}"] = np.sin(k * a)
    out[f"alpha_cos{k:g}"] = np.cos(k * a)
    out[f"delta_sin{k:g}"] = np.sin(k * d)
    out[f"delta_cos{k:g}"] = np.cos(k * d)
    return out


def add_sky_bin_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alpha = pd.to_numeric(out["alpha"], errors="coerce").to_numpy()
    delta = pd.to_numeric(out["delta"], errors="coerce").to_numpy()
    alpha_wrapped = np.mod(alpha, 360.0)
    alpha_bin = pd.cut(
        alpha_wrapped,
        bins=np.linspace(0.0, 360.0, 25),
        labels=False,
        include_lowest=True,
        right=False,
    )
    delta_bin = pd.cut(
        delta,
        bins=np.linspace(-90.0, 90.0, 19),
        labels=False,
        include_lowest=True,
    )
    alpha_bin = (
        pd.Series(alpha_bin, index=out.index).astype(float).fillna(-1).astype(int)
    )
    delta_bin = (
        pd.Series(delta_bin, index=out.index).astype(float).fillna(-1).astype(int)
    )
    out["sky_alpha_bin_24"] = alpha_bin.astype(str)
    out["sky_delta_bin_18"] = delta_bin.astype(str)
    out["sky_cell_bin"] = (alpha_bin * 19 + delta_bin).astype(str)
    return out


def add_sky_fine_bin_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alpha = pd.to_numeric(out["alpha"], errors="coerce").to_numpy()
    delta = pd.to_numeric(out["delta"], errors="coerce").to_numpy()
    alpha_wrapped = np.mod(alpha, 360.0)
    alpha_bin = pd.cut(
        alpha_wrapped,
        bins=np.linspace(0.0, 360.0, 49),
        labels=False,
        include_lowest=True,
        right=False,
    )
    delta_bin = pd.cut(
        delta,
        bins=np.linspace(-90.0, 90.0, 37),
        labels=False,
        include_lowest=True,
    )
    alpha_bin = (
        pd.Series(alpha_bin, index=out.index).astype(float).fillna(-1).astype(int)
    )
    delta_bin = (
        pd.Series(delta_bin, index=out.index).astype(float).fillna(-1).astype(int)
    )
    out["sky_alpha_bin_48"] = alpha_bin.astype(str)
    out["sky_delta_bin_36"] = delta_bin.astype(str)
    out["sky_cell_bin_48"] = (alpha_bin * 36 + delta_bin).astype(str)
    return out


def add_sky_bin_offset_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alpha = pd.to_numeric(out["alpha"], errors="coerce").to_numpy()
    delta = pd.to_numeric(out["delta"], errors="coerce").to_numpy()
    alpha_wrapped = np.mod(alpha, 360.0)
    alpha_bin = pd.to_numeric(out["sky_alpha_bin_24"], errors="coerce").to_numpy()
    delta_bin = pd.to_numeric(out["sky_delta_bin_18"], errors="coerce").to_numpy()
    alpha_centers = np.full(len(out), np.nan, dtype=float)
    delta_centers = np.full(len(out), np.nan, dtype=float)
    a_edges = np.linspace(0.0, 360.0, 25)
    d_edges = np.linspace(-90.0, 90.0, 19)
    a_width = 360.0 / 24.0
    d_width = 180.0 / 18.0
    am = np.isfinite(alpha_bin) & (alpha_bin >= 0) & (alpha_bin < 24)
    dm = np.isfinite(delta_bin) & (delta_bin >= 0) & (delta_bin < 18)
    alpha_centers[am] = a_edges[alpha_bin[am].astype(np.int64)] + 0.5 * a_width
    delta_centers[dm] = d_edges[delta_bin[dm].astype(np.int64)] + 0.5 * d_width
    out["sky_alpha_bin_center_deg"] = alpha_centers
    out["sky_alpha_bin_offset_deg"] = (
        (alpha_wrapped - alpha_centers + 180.0) % 360.0
    ) - 180.0
    out["sky_delta_bin_center_deg"] = delta_centers
    out["sky_delta_bin_offset_deg"] = delta - delta_centers
    return out


def add_sky_neighbor_density_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    out_train = train_df.copy()
    out_test = test_df.copy()
    combined = pd.concat(
        [out_train[["alpha", "delta"]], out_test[["alpha", "delta"]]],
        ignore_index=True,
    ).apply(pd.to_numeric, errors="coerce")
    alpha = np.deg2rad(combined["alpha"].fillna(combined["alpha"].median()).to_numpy())
    delta = np.deg2rad(combined["delta"].fillna(combined["delta"].median()).to_numpy())
    coords = np.column_stack([delta, alpha])
    nn = NearestNeighbors(
        n_neighbors=17, algorithm="ball_tree", metric="haversine", n_jobs=-1
    )
    nn.fit(coords)
    distances, _ = nn.kneighbors(coords, return_distance=True)
    arcmin = distances[:, 1:] * (180.0 / np.pi) * 60.0
    r5 = arcmin[:, 4]
    r10 = arcmin[:, 9]
    r16 = arcmin[:, 15]
    density5 = 5.0 / np.maximum(r5 * r5, 1e-6)
    density10 = 10.0 / np.maximum(r10 * r10, 1e-6)
    density16 = 16.0 / np.maximum(r16 * r16, 1e-6)
    feats = pd.DataFrame(
        {
            "sky_nn5_arcmin": r5,
            "sky_nn10_arcmin": r10,
            "sky_nn16_arcmin": r16,
            "sky_nn5_density": np.log1p(density5),
            "sky_nn10_density": np.log1p(density10),
            "sky_nn16_density": np.log1p(density16),
            "sky_nn_density_ratio_5_16": density5 / np.maximum(density16, 1e-12),
        }
    )
    n_train = len(out_train)
    out_train[feats.columns] = feats.iloc[:n_train].to_numpy()
    out_test[feats.columns] = feats.iloc[n_train:].to_numpy()
    return out_train, out_test


def add_frequency_features(train_df: pd.DataFrame, test_df: pd.DataFrame, cols):
    out_train = train_df.copy()
    out_test = test_df.copy()
    for col in cols:
        train_key = out_train[col].astype(str).fillna("missing")
        test_key = out_test[col].astype(str).fillna("missing")
        freq = pd.concat([train_key, test_key], ignore_index=True).value_counts(
            normalize=True
        )
        out_train[f"{col}_freq"] = train_key.map(freq).fillna(0.0).astype(float)
        out_test[f"{col}_freq"] = test_key.map(freq).fillna(0.0).astype(float)
    return out_train, out_test


def add_galaxy_population_sky_alpha_interaction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["galaxy_population__sky_alpha_bin_24"] = (
        out["galaxy_population"].astype(str).fillna("missing")
        + "__"
        + out["sky_alpha_bin_24"].astype(str).fillna("missing")
    )
    return out


def add_spectral_sky_alpha_interaction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type__sky_alpha_bin_24"] = (
        out["spectral_type"].astype(str).fillna("missing")
        + "__"
        + out["sky_alpha_bin_24"].astype(str).fillna("missing")
    )
    return out


def add_spectral_sky_cell_interaction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type__sky_cell_bin"] = (
        out["spectral_type"].astype(str).fillna("missing")
        + "__"
        + out["sky_cell_bin"].astype(str).fillna("missing")
    )
    return out


def add_spectral_population_interaction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type__galaxy_population"] = (
        out["spectral_type"].astype(str).fillna("missing")
        + "__"
        + out["galaxy_population"].astype(str).fillna("missing")
    )
    return out


def add_triple_spectral_population_sky_delta_interaction(
    df: pd.DataFrame,
) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type__galaxy_population__sky_delta_bin_18"] = (
        out["spectral_type"].astype(str).fillna("missing")
        + "__"
        + out["galaxy_population"].astype(str).fillna("missing")
        + "__"
        + out["sky_delta_bin_18"].astype(str).fillna("missing")
    )
    return out


def add_spectral_sky_alpha_interaction_fine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type__sky_alpha_bin_48"] = (
        out["spectral_type"].astype(str).fillna("missing")
        + "__"
        + out["sky_alpha_bin_48"].astype(str).fillna("missing")
    )
    return out


def add_galaxy_sky_alpha_interaction_fine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["galaxy_population__sky_alpha_bin_48"] = (
        out["galaxy_population"].astype(str).fillna("missing")
        + "__"
        + out["sky_alpha_bin_48"].astype(str).fillna("missing")
    )
    return out


def add_spectral_sky_cell_interaction_fine(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type__sky_cell_bin_48"] = (
        out["spectral_type"].astype(str).fillna("missing")
        + "__"
        + out["sky_cell_bin_48"].astype(str).fillna("missing")
    )
    return out


def add_redshift_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z = pd.to_numeric(out["redshift"], errors="coerce")
    az = z.abs()
    out["redshift_abs"] = az
    out["redshift_sq"] = z**2
    out["redshift_log_abs"] = np.log1p(az)
    out["redshift_signed_log_abs"] = np.sign(z) * np.log1p(az)
    return out


def add_redshift_bin_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, n_bins: int = 20
):
    out_train = train_df.copy()
    out_test = test_df.copy()
    combined = pd.concat(
        [
            pd.to_numeric(out_train["redshift"], errors="coerce"),
            pd.to_numeric(out_test["redshift"], errors="coerce"),
        ],
        ignore_index=True,
    )
    z = combined.to_numpy(dtype=float)
    finite = np.isfinite(z)
    n_train = len(out_train)
    code = pd.Series(np.full(len(combined), np.nan), index=combined.index, dtype=float)
    if finite.sum() >= 2:
        q = np.unique(np.quantile(z[finite], np.linspace(0.0, 1.0, n_bins + 1)))
        q = q[np.isfinite(q)]
        if len(q) >= 3:
            try:
                code = pd.cut(
                    combined,
                    bins=q,
                    labels=False,
                    include_lowest=True,
                    duplicates="drop",
                ).astype(float)
            except Exception:
                code = pd.Series(np.nan, index=combined.index, dtype=float)
    code = code.fillna(-1.0).round(0).astype(int)
    out_train["redshift_bin_20"] = ("zbin_" + code.iloc[:n_train].astype(str)).astype(
        str
    )
    out_test["redshift_bin_20"] = ("zbin_" + code.iloc[n_train:].astype(str)).astype(
        str
    )
    return out_train, out_test


def add_redshift_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type__redshift_bin_20"] = (
        out["spectral_type"].astype(str).fillna("missing")
        + "__"
        + out["redshift_bin_20"].astype(str).fillna("missing")
    )
    out["galaxy_population__redshift_bin_20"] = (
        out["galaxy_population"].astype(str).fillna("missing")
        + "__"
        + out["redshift_bin_20"].astype(str).fillna("missing")
    )
    return out


def add_galactic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ra = np.deg2rad(pd.to_numeric(out["alpha"], errors="coerce").to_numpy())
    dec = np.deg2rad(pd.to_numeric(out["delta"], errors="coerce").to_numpy())
    x_eq = np.cos(dec) * np.cos(ra)
    y_eq = np.cos(dec) * np.sin(ra)
    z_eq = np.sin(dec)
    m00, m01, m02 = -0.0548755604, -0.8734370902, -0.4838350155
    m10, m11, m12 = 0.4941094279, -0.44482963, 0.7469822445
    m20, m21, m22 = -0.8676661490, -0.1980763734, 0.4559837762
    x_gal = m00 * x_eq + m01 * y_eq + m02 * z_eq
    y_gal = m10 * x_eq + m11 * y_eq + m12 * z_eq
    z_gal = m20 * x_eq + m21 * y_eq + m22 * z_eq
    l_rad = np.arctan2(y_gal, x_gal)
    b_rad = np.arcsin(np.clip(z_gal, -1.0, 1.0))
    l_deg = (np.degrees(l_rad) + 360.0) % 360.0
    b_deg = np.degrees(b_rad)
    out["galactic_l"] = l_deg
    out["galactic_b"] = b_deg
    lr = np.deg2rad(l_deg)
    br = np.deg2rad(b_deg)
    out["galactic_l_sin"] = np.sin(lr)
    out["galactic_l_cos"] = np.cos(lr)
    out["galactic_b_sin"] = np.sin(br)
    out["galactic_b_cos"] = np.cos(br)
    out["galactic_b_abs"] = np.abs(pd.to_numeric(out["galactic_b"], errors="coerce"))
    return out


def add_galactic_bin_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    l = np.mod(pd.to_numeric(out["galactic_l"], errors="coerce").to_numpy(), 360.0)
    b = pd.to_numeric(out["galactic_b"], errors="coerce").to_numpy()
    l_bin = pd.cut(
        l,
        bins=np.linspace(0.0, 360.0, 25),
        labels=False,
        include_lowest=True,
        right=False,
    )
    b_bin = pd.cut(
        b,
        bins=np.linspace(-90.0, 90.0, 13),
        labels=False,
        include_lowest=True,
    )
    l_bin = pd.Series(l_bin, index=out.index).astype(float).fillna(-1).astype(int)
    b_bin = pd.Series(b_bin, index=out.index).astype(float).fillna(-1).astype(int)
    out["galactic_l_bin_24"] = l_bin.astype(str)
    out["galactic_b_bin_12"] = b_bin.astype(str)
    out["galactic_cell_bin"] = (l_bin * 12 + b_bin).astype(str)
    return out


def add_spectral_galactic_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spectral_type__galactic_cell_bin"] = (
        out["spectral_type"].astype(str).fillna("missing")
        + "__"
        + out["galactic_cell_bin"].astype(str).fillna("missing")
    )
    out["galaxy_population__galactic_cell_bin"] = (
        out["galaxy_population"].astype(str).fillna("missing")
        + "__"
        + out["galactic_cell_bin"].astype(str).fillna("missing")
    )
    return out


def add_rank_features(train_df: pd.DataFrame, test_df: pd.DataFrame, cols):
    train_vals = train_df[cols].astype(float)
    test_vals = test_df[cols].astype(float)
    combined = pd.concat([train_vals, test_vals], axis=0, ignore_index=True)
    rank_df = combined.rank(pct=True, method="average").add_suffix("_qrank")
    n_train = len(train_df)
    out_train = train_df.copy()
    out_test = test_df.copy()
    out_train[rank_df.columns] = rank_df.iloc[:n_train].to_numpy()
    out_test[rank_df.columns] = rank_df.iloc[n_train:].to_numpy()
    return out_train, out_test


def add_all_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    out_train = train_df.copy()
    out_test = test_df.copy()

    out_train = add_color_features(out_train)
    out_test = add_color_features(out_test)
    out_train = add_band_profile_features(out_train)
    out_test = add_band_profile_features(out_test)
    out_train = add_band_extrema_features(out_train)
    out_test = add_band_extrema_features(out_test)
    out_train = add_extrema_interaction_features(out_train)
    out_test = add_extrema_interaction_features(out_test)
    out_train = add_sky_circular_features(out_train)
    out_test = add_sky_circular_features(out_test)
    out_train = add_sky_cartesian_features(out_train)
    out_test = add_sky_cartesian_features(out_test)
    out_train = add_sky_cartesian_interaction_features(out_train)
    out_test = add_sky_cartesian_interaction_features(out_test)

    for order in (2, 3, 4, 5, 6, 7, 8):
        out_train = add_sky_harmonic_features(out_train, order)
        out_test = add_sky_harmonic_features(out_test, order)

    # These transductive aggregates use only covariates from train+test and never use the target.
    out_train, out_test = add_sky_neighbor_density_features(out_train, out_test)
    out_train = add_sky_bin_features(out_train)
    out_test = add_sky_bin_features(out_test)
    out_train = add_sky_fine_bin_features(out_train)
    out_test = add_sky_fine_bin_features(out_test)
    out_train = add_sky_bin_offset_features(out_train)
    out_test = add_sky_bin_offset_features(out_test)
    out_train, out_test = add_frequency_features(
        out_train,
        out_test,
        cols=[
            "sky_alpha_bin_24",
            "sky_delta_bin_18",
            "sky_cell_bin",
            "sky_alpha_bin_48",
            "sky_delta_bin_36",
            "sky_cell_bin_48",
        ],
    )

    out_train = add_galaxy_population_sky_alpha_interaction(out_train)
    out_test = add_galaxy_population_sky_alpha_interaction(out_test)
    out_train = add_spectral_sky_alpha_interaction(out_train)
    out_test = add_spectral_sky_alpha_interaction(out_test)
    out_train = add_spectral_sky_alpha_interaction_fine(out_train)
    out_test = add_spectral_sky_alpha_interaction_fine(out_test)
    out_train = add_galaxy_sky_alpha_interaction_fine(out_train)
    out_test = add_galaxy_sky_alpha_interaction_fine(out_test)
    out_train = add_spectral_sky_cell_interaction(out_train)
    out_test = add_spectral_sky_cell_interaction(out_test)
    out_train = add_spectral_sky_cell_interaction_fine(out_train)
    out_test = add_spectral_sky_cell_interaction_fine(out_test)
    out_train = add_triple_spectral_population_sky_delta_interaction(out_train)
    out_test = add_triple_spectral_population_sky_delta_interaction(out_test)

    out_train = add_redshift_shape_features(out_train)
    out_test = add_redshift_shape_features(out_test)
    out_train, out_test = add_redshift_bin_features(out_train, out_test, n_bins=20)
    out_train = add_redshift_interaction_features(out_train)
    out_test = add_redshift_interaction_features(out_test)
    out_train, out_test = add_frequency_features(
        out_train, out_test, cols=["redshift_bin_20"]
    )

    out_train = add_spectral_population_interaction(out_train)
    out_test = add_spectral_population_interaction(out_test)

    out_train = add_galactic_features(out_train)
    out_test = add_galactic_features(out_test)
    out_train = add_galactic_bin_features(out_train)
    out_test = add_galactic_bin_features(out_test)
    out_train, out_test = add_frequency_features(
        out_train,
        out_test,
        cols=["galactic_l_bin_24", "galactic_b_bin_12", "galactic_cell_bin"],
    )
    out_train = add_spectral_galactic_interactions(out_train)
    out_test = add_spectral_galactic_interactions(out_test)

    out_train, out_test = add_rank_features(
        out_train,
        out_test,
        cols=["u", "g", "r", "i", "z", "redshift"],
    )
    return out_train, out_test


def compute_aux_locality_weights(
    main_df: pd.DataFrame, full_df: pd.DataFrame, cols
) -> np.ndarray:
    if len(main_df) == 0 or len(full_df) == 0:
        return np.ones(len(full_df), dtype=float)
    base = pd.DataFrame()
    full = pd.DataFrame()
    for col in cols:
        base[col] = pd.to_numeric(main_df[col], errors="coerce")
        full[col] = pd.to_numeric(full_df[col], errors="coerce")
    med = base.median(numeric_only=True)
    mad = (base - med).abs().median(numeric_only=True).replace(0.0, 1.0)
    mad = mad.where(mad > 0.0, 1.0)
    base_std = ((base - med) / mad).fillna(0.0)
    full_std = ((full - med) / mad).fillna(0.0)
    base_dist = np.sqrt((base_std**2).sum(axis=1).to_numpy())
    full_dist = np.sqrt((full_std**2).sum(axis=1).to_numpy())
    scale = np.quantile(base_dist, 0.90)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.median(base_dist))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return np.clip(np.exp(-full_dist / scale).astype(float), 0.08, 1.0)


def compute_fold_sample_weights(
    train_idx: np.ndarray,
    is_main_train: np.ndarray,
    y_train: np.ndarray,
    aux_base_weight: float,
    aux_locality: np.ndarray,
) -> np.ndarray:
    weights = np.ones(len(train_idx), dtype=float)
    local_is_main = is_main_train[train_idx]
    local_is_aux = ~local_is_main
    if not np.any(local_is_aux):
        return weights
    y_local = np.asarray(y_train, dtype=object)[train_idx].astype(str)
    ser = pd.Series(y_local)
    main_counts = ser[local_is_main].value_counts()
    aux_counts = ser[local_is_aux].value_counts()
    if len(main_counts) == 0 or len(aux_counts) == 0:
        weights[local_is_aux] = (
            aux_base_weight
            * np.asarray(aux_locality, dtype=float)[train_idx][local_is_aux]
        )
        return weights
    raw_ratio = {}
    for cls, aux_n in aux_counts.items():
        main_n = main_counts.get(cls, 0.0)
        raw_ratio[cls] = (
            float(main_n) / float(aux_n) if aux_n > 0 and main_n > 0 else 1.0
        )
    ratio_vals = np.array(list(raw_ratio.values()), dtype=float)
    center = float(np.median(ratio_vals)) if len(ratio_vals) else 1.0
    if not np.isfinite(center) or center <= 0:
        center = 1.0
    y_aux = y_local[local_is_aux]
    locality_aux = np.asarray(aux_locality, dtype=float)[train_idx][local_is_aux]
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
        iterations=int(CAT_CONFIG["iterations"]),
        learning_rate=float(CAT_CONFIG["learning_rate"]),
        depth=int(CAT_CONFIG["depth"]),
        l2_leaf_reg=float(CAT_CONFIG["l2_leaf_reg"]),
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        random_seed=int(CAT_CONFIG["random_seed"]),
        verbose=False,
    )


def fit_cat_with_fallback(X_tr, y_tr, X_val, y_val, cat_features, sample_weight):
    params = make_cat_model()
    fit_kwargs = {
        "cat_features": cat_features,
        "sample_weight": sample_weight,
        "verbose": False,
        "eval_set": (X_val, y_val),
        "early_stopping_rounds": 60,
    }
    try:
        model = CatBoostClassifier(
            task_type="GPU", devices="0", gpu_ram_part=0.8, **params
        )
        model.fit(X_tr, y_tr, **fit_kwargs)
        return model
    except Exception:
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
    fit_kwargs = {
        "sample_weight": sample_weight,
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    for device_params in (
        {"tree_method": "hist", "device": "cuda"},
        {"tree_method": "hist"},
    ):
        model = XGBClassifier(**base, **device_params)
        try:
            model.fit(X_tr, y_tr, **fit_kwargs, early_stopping_rounds=40)
            return model
        except Exception:
            try:
                model.fit(X_tr, y_tr, **fit_kwargs)
                return model
            except Exception:
                continue
    return None


def lightgbm_cuda_smoke_ok() -> bool:
    global _LGBM_CUDA_SMOKE_OK
    if _LGBM_CUDA_SMOKE_OK is not None:
        return _LGBM_CUDA_SMOKE_OK
    if not HAS_LGBM:
        _LGBM_CUDA_SMOKE_OK = False
        return _LGBM_CUDA_SMOKE_OK
    smoke_code = """
import numpy as np
from lightgbm import LGBMClassifier
rng = np.random.default_rng(0)
X = rng.normal(size=(96, 6))
y = rng.integers(0, 3, size=96)
m = LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=8,
    learning_rate=0.1,
    num_leaves=31,
    device_type="cuda",
    verbosity=-1,
)
m.fit(X, y)
print("LIGHTGBM_CUDA_OK")
"""
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


def align_numeric_proba(
    probabilities: np.ndarray, model_classes: np.ndarray, num_classes: int
) -> np.ndarray:
    out = np.zeros((probabilities.shape[0], num_classes), dtype=np.float64)
    for j, cls in enumerate(model_classes):
        idx = int(cls)
        if 0 <= idx < num_classes:
            out[:, idx] = probabilities[:, j]
    return out


def score_predictions(
    y_true: np.ndarray, probs: np.ndarray, class_order: np.ndarray
) -> float:
    pred = class_order[np.argmax(probs, axis=1)]
    return float(balanced_accuracy_score(y_true, pred))


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
                "Auxiliary locality weights: "
                f"min={aux_locality.min():.4f}, mean={aux_locality.mean():.4f}, max={aux_locality.max():.4f}",
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
        num_medians = train_full[numeric_cols].median(numeric_only=True)
        for col in numeric_cols:
            fill_value = num_medians.get(col, 0.0)
            train_full[col] = train_full[col].fillna(fill_value)
            test[col] = test[col].fillna(fill_value)

        feature_cols = [c for c in train_full.columns if c not in ("id", "class")]
        X_cat = train_full[feature_cols]
        X_test_cat = test[feature_cols]

        class_order = CLASS_ORDER.copy()
        class_to_idx = {cls: i for i, cls in enumerate(class_order)}
        target_code = pd.Series(all_targets).map(class_to_idx).to_numpy(dtype=np.int64)
        main_target_code = target_code[:n_main]
        num_classes = len(class_order)

        cat_features = [i for i, c in enumerate(feature_cols) if c in CATEGORICAL_COLS]

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
    test_probs = {
        name: np.zeros((len(test), num_classes), dtype=np.float64)
        for name in model_names
    }
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
            y_tr_code = target_code[train_idx]
            y_va_code = target_code[main_va_idx]

            fold_aux_weight = compute_fold_sample_weights(
                train_idx,
                is_main_train,
                all_targets,
                AUX_SAMPLE_WEIGHT,
                aux_locality,
            )
            balanced_sw = compute_sample_weight(
                class_weight="balanced", y=y_tr_code
            ).astype(float)
            tree_sw = balanced_sw * fold_aux_weight

            X_tr_cat = X_cat.iloc[train_idx]
            X_va_cat = X_cat.iloc[main_va_idx]
            X_tr_tree = train_tree.iloc[train_idx]
            X_va_tree = train_tree.iloc[main_va_idx]

            log_stage(
                f"event=progress|stage=fit_predict_fold_stage|fold={fold_idx}|model=catboost"
            )
            start = time.perf_counter()
            cat_model = fit_cat_with_fallback(
                X_tr_cat,
                y_tr,
                X_va_cat,
                y_va,
                cat_features,
                fold_aux_weight,
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
            test_probs["catboost"] += cat_test_prob
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
                    X_tr_tree,
                    y_tr_code,
                    X_va_tree,
                    y_va_code,
                    tree_sw,
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
                    test_probs["xgboost"] += xgb_test_prob
                    score = score_predictions(y_va, xgb_val_prob, class_order)
                    elapsed = time.perf_counter() - start
                    fold_scores["xgboost"].append(score)
                    fold_times["xgboost"].append(elapsed)
                    print(
                        f"Fold {fold_idx} | xgboost | balanced_accuracy={score:.6f} | runtime_s={elapsed:.1f}",
                        flush=True,
                    )
            if not model_available["xgboost"]:
                elapsed = time.perf_counter() - start
                print(
                    f"Fold {fold_idx} | xgboost | unavailable_after_failure | runtime_s={elapsed:.1f}",
                    flush=True,
                )

            log_stage(
                f"event=progress|stage=fit_predict_fold_stage|fold={fold_idx}|model=lightgbm"
            )
            start = time.perf_counter()
            if model_available["lightgbm"]:
                lgb_model = fit_lgbm_with_fallback(
                    X_tr_tree,
                    y_tr_code,
                    X_va_tree,
                    y_va_code,
                    tree_sw,
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
                    test_probs["lightgbm"] += lgb_test_prob
                    score = score_predictions(y_va, lgb_val_prob, class_order)
                    elapsed = time.perf_counter() - start
                    fold_scores["lightgbm"].append(score)
                    fold_times["lightgbm"].append(elapsed)
                    print(
                        f"Fold {fold_idx} | lightgbm | balanced_accuracy={score:.6f} | runtime_s={elapsed:.1f}",
                        flush=True,
                    )
            if not model_available["lightgbm"]:
                elapsed = time.perf_counter() - start
                print(
                    f"Fold {fold_idx} | lightgbm | unavailable_after_failure | runtime_s={elapsed:.1f}",
                    flush=True,
                )

    with aide_stage("score_stage"):
        valid_models = []
        for name in model_names:
            if not model_available[name]:
                print(f"Model {name} excluded: training failed.", flush=True)
                continue
            if len(fold_scores[name]) != N_SPLITS:
                print(f"Model {name} excluded: missing fold predictions.", flush=True)
                continue
            row_mass = oof_probs[name].sum(axis=1)
            if not np.all(np.isfinite(row_mass)) or np.any(row_mass <= 0.0):
                print(
                    f"Model {name} excluded: incomplete OOF probabilities.", flush=True
                )
                continue
            test_probs[name] /= float(N_SPLITS)
            mean_fold = float(np.mean(fold_scores[name]))
            oof_score = score_predictions(main_targets, oof_probs[name], class_order)
            mean_time = (
                float(np.mean(fold_times[name])) if fold_times[name] else float("nan")
            )
            print(
                f"Model summary | {name} | mean_cv_balanced_accuracy={mean_fold:.6f} | "
                f"oof_balanced_accuracy={oof_score:.6f} | mean_runtime_s={mean_time:.1f}",
                flush=True,
            )
            valid_models.append((oof_score, mean_fold, name))

        if not valid_models:
            raise RuntimeError(
                "No model in the panel produced complete OOF predictions."
            )

        valid_models.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_oof_score, best_mean_cv, best_model = valid_models[0]
        best_oof_probs = oof_probs[best_model]
        best_test_probs = test_probs[best_model]
        best_oof_pred = class_order[np.argmax(best_oof_probs, axis=1)]
        best_test_pred = class_order[np.argmax(best_test_probs, axis=1)]

        print(f"Selected best single model: {best_model}", flush=True)
        print(f"Mean CV balanced accuracy: {best_mean_cv:.6f}", flush=True)
        print(f"OOF balanced accuracy: {best_oof_score:.6f}", flush=True)
        print(
            f"Primary validation metric ({best_model}): {best_oof_score:.6f}",
            flush=True,
        )

    with aide_stage("write_outputs_stage"):
        oof_frame = pd.DataFrame(
            {
                "row": np.arange(n_main, dtype=np.int64),
                "target": main_targets,
                "prediction": best_oof_pred,
            }
        )
        write_oof_predictions(oof_frame)

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
