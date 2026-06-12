from __future__ import annotations

import json
import shutil
import warnings
import contextlib
import inspect
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

AIDE_AG_CONFIG = {'aux_file': 'star_classification.csv',
 'class_balance': 'balanced',
 'eval_metric': 'balanced_accuracy',
 'fit_args': {'auto_stack': False, 'fit_weighted_ensemble': True, 'save_space': True},
 'hyperparameters': {'CAT': [{'ag_args_fit': {'num_gpus': 1},
                              'devices': '0',
                              'gpu_ram_part': 0.8,
                              'task_type': 'GPU'}],
                     'GBM': [{'ag_args_fit': {'num_gpus': 1}, 'device': 'cuda'}],
                     'XGB': [{'ag_args': {'priority': 999},
                              'ag_args_fit': {'num_gpus': 1},
                              'device': 'cuda',
                              'tree_method': 'hist'}]},
 'included_model_types': ['XGB', 'GBM', 'CAT'],
 'preprocess_timeout': 180,
 'presets': 'medium_quality',
 'project_name': 'playground-series-s6e6',
 'time_limit': 600,
 'use_gpu': True,
 'validation_strategy': 'holdout'}
RESULT_MARKER = 'AIDE_RESULT_JSON:'
FORBIDDEN_SPLIT_MARKER = '__is_train__'
FORBIDDEN_ROW_ID = '__aide_row_id__'
CLASS_WEIGHT_COL = "__aide_class_weight__"


def preprocess(df: pd.DataFrame, aux: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    out = df.copy()

    def _safe_to_float(series):
        return pd.to_numeric(series, errors="coerce")

    numeric_cols = [
        c
        for c in ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
        if c in out.columns
    ]
    for c in numeric_cols:
        out[c] = _safe_to_float(out[c])

    # --- Colors ---
    color_pairs = [
        ("u", "g"),
        ("u", "r"),
        ("u", "i"),
        ("u", "z"),
        ("g", "r"),
        ("g", "i"),
        ("g", "z"),
        ("r", "i"),
        ("r", "z"),
    ]
    for a, b in color_pairs:
        if a in out.columns and b in out.columns:
            out[f"{a}_{b}_color"] = out[a] - out[b]
            out[f"{a}_{b}_color_abs"] = (out[a] - out[b]).abs()

    # --- Band profile features on ugriz ---
    bands = [c for c in ["u", "g", "r", "i", "z"] if c in out.columns]
    if len(bands) == 5:
        mags = out[bands].to_numpy(dtype=np.float64, copy=False)
        band_idx = np.arange(mags.shape[1], dtype=np.float64)
        idx_bar = band_idx.mean()
        x_var = np.sum((band_idx - idx_bar) ** 2.0)

        mag_mean = mags.mean(axis=1)
        mag_std = mags.std(axis=1)
        mag_min = mags.min(axis=1)
        mag_max = mags.max(axis=1)
        out["mag_mean"] = mag_mean
        out["mag_std"] = mag_std
        out["mag_range"] = mag_max - mag_min
        out["mag_min"] = mag_min
        out["mag_max"] = mag_max

        slope = np.sum((mags - mag_mean[:, None]) * (band_idx - idx_bar), axis=1) / (
            x_var if x_var else 1.0
        )
        intercept = mag_mean - slope * idx_bar
        pred = slope[:, None] * band_idx + intercept[:, None]
        residual = mags - pred

        out["mag_spectral_slope"] = slope
        out["mag_resid_mean_abs"] = np.abs(residual).mean(axis=1)
        out["mag_resid_std"] = residual.std(axis=1)

        dif1 = np.diff(mags, axis=1)
        dif2 = np.diff(dif1, axis=1)
        out["mag_d1_mean"] = dif1.mean(axis=1)
        out["mag_d1_std"] = dif1.std(axis=1)
        out["mag_d1_abs_mean"] = np.abs(dif1).mean(axis=1)
        out["mag_d2_mean"] = dif2.mean(axis=1)
        out["mag_d2_std"] = dif2.std(axis=1)
        out["mag_d2_abs_mean"] = np.abs(dif2).mean(axis=1)

        band_labels = np.array(bands)
        argmin_idx = np.argmin(mags, axis=1)
        argmax_idx = np.argmax(mags, axis=1)
        out["mag_min_band"] = pd.Series(band_labels[argmin_idx], index=out.index)
        out["mag_max_band"] = pd.Series(band_labels[argmax_idx], index=out.index)

    # --- Redshift transforms ---
    if "redshift" in out.columns:
        z = out["redshift"].to_numpy(dtype=np.float64)
        z_abs = np.abs(z)
        z_signed = np.sign(z) * np.log1p(z_abs)
        out["redshift_abs"] = z_abs
        out["redshift_sq"] = z**2
        out["redshift_log1p_abs"] = np.log1p(z_abs)
        out["redshift_signed_log1p_abs"] = z_signed

        try:
            out["redshift_bin"] = pd.qcut(
                out["redshift"], q=10, labels=False, duplicates="drop"
            )
        except ValueError:
            out["redshift_bin"] = 0

    # --- Categorical mappings and interactions ---
    if "spectral_type" in out.columns:
        spec = out["spectral_type"].astype("string").fillna("NA")
        spec_code = spec.map({"M": 0.0, "O/B": 1.0, "G/K": 2.0, "A/F": 3.0}).fillna(
            -1.0
        )
    else:
        spec = pd.Series(["NA"] * len(out), index=out.index, dtype="string")
        spec_code = pd.Series([-1.0] * len(out), index=out.index)

    if "galaxy_population" in out.columns:
        gal_pop = out["galaxy_population"].astype("string").fillna("NA")
        pop_code = gal_pop.map({"Blue_Cloud": 0.0, "Red_Sequence": 1.0}).fillna(-1.0)
    else:
        gal_pop = pd.Series(["NA"] * len(out), index=out.index, dtype="string")
        pop_code = pd.Series([-1.0] * len(out), index=out.index)

    out["spectral_type"] = out.get("spectral_type", spec)
    out["galaxy_population"] = out.get("galaxy_population", gal_pop)
    out["spectral_type_code"] = spec_code.values
    out["galaxy_population_code"] = pop_code.values

    if "redshift" in out.columns:
        out["redshift_x_spec_code"] = out["redshift"] * spec_code
        out["redshift_x_pop_code"] = out["redshift"] * pop_code
        out["redshift_abs_x_spec_code"] = out["redshift_abs"] * spec_code
        out["redshift_abs_x_pop_code"] = out["redshift_abs"] * pop_code
        out["redshift_signed_log1p_abs_x_pop_code"] = (
            out["redshift_signed_log1p_abs"] * pop_code
        )

    if "redshift_bin" in out.columns:
        out["redshift_bin"] = out["redshift_bin"].astype("Int64")
        out["redshift_bin"] = out["redshift_bin"].fillna(0).astype(np.int16)

    # --- Sky features (RA/Dec trig, cartesian, harmonics, bins) ---
    if {"alpha", "delta"}.issubset(out.columns):
        alpha = out["alpha"].to_numpy(dtype=np.float64)
        delta = out["delta"].to_numpy(dtype=np.float64)

        ar = np.deg2rad(alpha)
        dr = np.deg2rad(delta)

        sin_a = np.sin(ar)
        cos_a = np.cos(ar)
        sin_d = np.sin(dr)
        cos_d = np.cos(dr)

        out["alpha_sin"] = sin_a
        out["alpha_cos"] = cos_a
        out["delta_sin"] = sin_d
        out["delta_cos"] = cos_d

        x = cos_d * cos_a
        y = cos_d * sin_a
        z = sin_d
        out["sky_x"] = x
        out["sky_y"] = y
        out["sky_z"] = z
        out["sky_xy"] = x * y
        out["sky_xz"] = x * z
        out["sky_yz"] = y * z
        out["sky_rxy"] = np.hypot(x, y)
        out["sky_radius"] = np.sqrt(x * x + y * y + z * z)

        for k in range(2, 9):
            out[f"alpha_sin_{k}"] = np.sin(k * ar)
            out[f"alpha_cos_{k}"] = np.cos(k * ar)
            out[f"delta_sin_{k}"] = np.sin(k * dr)
            out[f"delta_cos_{k}"] = np.cos(k * dr)

        a_coarse = np.clip((alpha / 15.0).astype(np.int16), 0, 23)
        d_coarse = np.clip(((delta + 90.0) / 10.0).astype(np.int16), 0, 17)
        a_fine = np.clip((alpha / 5.0).astype(np.int16), 0, 71)
        d_fine = np.clip(((delta + 90.0) / 2.0).astype(np.int16), 0, 89)

        out["alpha_coarse_bin"] = a_coarse
        out["delta_coarse_bin"] = d_coarse
        out["alpha_fine_bin"] = a_fine
        out["delta_fine_bin"] = d_fine

        a_coarse_center = (a_coarse + 0.5) * 15.0
        d_coarse_center = (d_coarse + 0.5) * 10.0 - 90.0
        a_fine_center = (a_fine + 0.5) * 5.0
        d_fine_center = (d_fine + 0.5) * 2.0 - 90.0

        out["alpha_coarse_center"] = a_coarse_center
        out["delta_coarse_center"] = d_coarse_center
        out["alpha_fine_center"] = a_fine_center
        out["delta_fine_center"] = d_fine_center
        out["alpha_coarse_center_offset"] = alpha - a_coarse_center
        out["delta_coarse_center_offset"] = delta - d_coarse_center
        out["alpha_fine_center_offset"] = alpha - a_fine_center
        out["delta_fine_center_offset"] = delta - d_fine_center

        def _neighbor_density(a_bin, d_bin, a_size, d_size):
            a_i = np.asarray(a_bin, dtype=np.int32)
            d_i = np.asarray(d_bin, dtype=np.int32)
            flat = a_i * d_size + d_i
            counts = np.bincount(flat, minlength=a_size * d_size).astype(np.float64)
            self_count = counts[flat]
            neigh_sum = np.zeros_like(self_count, dtype=np.float64)
            for da in (-1, 0, 1):
                for dd in (-1, 0, 1):
                    a2 = np.clip(a_i + da, 0, a_size - 1)
                    d2 = np.clip(d_i + dd, 0, d_size - 1)
                    neigh_sum += counts[a2 * d_size + d2]
            return self_count, neigh_sum

        out["sky_cell_count_coarse"], out["sky_cell_plus1_coarse"] = _neighbor_density(
            a_coarse, d_coarse, 24, 18
        )
        out["sky_cell_count_fine"], out["sky_cell_plus1_fine"] = _neighbor_density(
            a_fine, d_fine, 72, 90
        )

    # --- Galactic coordinates from RA/Dec (approx transformation) ---
    if {"alpha", "delta"}.issubset(out.columns):
        ra = np.deg2rad(out["alpha"].to_numpy(dtype=np.float64))
        dc = np.deg2rad(out["delta"].to_numpy(dtype=np.float64))
        v_x = np.cos(dc) * np.cos(ra)
        v_y = np.cos(dc) * np.sin(ra)
        v_z = np.sin(dc)

        # Equatorial (ICRS) -> Galactic rotation matrix
        r00, r01, r02 = (-0.0548755604, -0.8734370902, -0.4838350155)
        r10, r11, r12 = (0.4941094279, -0.4448296300, 0.7469822445)
        r20, r21, r22 = (-0.8676661490, -0.1980763734, 0.4559837762)

        g_x = r00 * v_x + r01 * v_y + r02 * v_z
        g_y = r10 * v_x + r11 * v_y + r12 * v_z
        g_z = r20 * v_x + r21 * v_y + r22 * v_z

        l = np.degrees(np.arctan2(g_y, g_x))
        l = (l + 360.0) % 360.0
        b = np.degrees(np.arcsin(np.clip(g_z, -1.0, 1.0)))

        out["gal_l"] = l
        out["gal_b"] = b
        out["gal_l_sin"] = np.sin(np.deg2rad(l))
        out["gal_l_cos"] = np.cos(np.deg2rad(l))
        out["gal_b_sin"] = np.sin(np.deg2rad(b))
        out["gal_b_cos"] = np.cos(np.deg2rad(b))
        out["gal_b_abs"] = np.abs(b)

        out["gal_l_bin"] = np.clip((l / 15.0).astype(np.int16), 0, 23)
        out["gal_b_bin"] = np.clip(((b + 90.0) / 10.0).astype(np.int16), 0, 17)

        out["gal_l_center"] = (out["gal_l_bin"].to_numpy(dtype=np.float64) + 0.5) * 15.0
        out["gal_b_center"] = (
            out["gal_b_bin"].to_numpy(dtype=np.float64) + 0.5
        ) * 10.0 - 90.0
        out["gal_l_offset"] = l - out["gal_l_center"]
        out["gal_b_offset"] = b - out["gal_b_center"]

    # --- Crossed categoricals ---
    out["spec_x_pop"] = (
        spec.astype("string").fillna("NA") + "|" + gal_pop.astype("string").fillna("NA")
    )
    if "redshift_bin" in out.columns:
        out["spec_x_redshift_bin"] = (
            spec.astype("string").fillna("NA")
            + "|"
            + out["redshift_bin"].astype("string").fillna("NA")
        )
        out["pop_x_redshift_bin"] = (
            gal_pop.astype("string").fillna("NA")
            + "|"
            + out["redshift_bin"].astype("string").fillna("NA")
        )
    if "alpha_coarse_bin" in out.columns and "delta_coarse_bin" in out.columns:
        sky_cell = (
            "a"
            + out["alpha_coarse_bin"].astype("string")
            + "_d"
            + out["delta_coarse_bin"].astype("string")
        )
        out["sky_cell_cat"] = sky_cell
        out["spec_x_sky_cell"] = spec.astype("string").fillna("NA") + "|" + sky_cell
        out["pop_x_sky_cell"] = gal_pop.astype("string").fillna("NA") + "|" + sky_cell
    if "gal_l_bin" in out.columns and "gal_b_bin" in out.columns:
        gal_cell = (
            "l"
            + out["gal_l_bin"].astype("string")
            + "_b"
            + out["gal_b_bin"].astype("string")
        )
        out["gal_cell_cat"] = gal_cell
        out["spec_x_gal_cell"] = spec.astype("string").fillna("NA") + "|" + gal_cell

    # --- Covariate-only frequency features ---
    freq_candidates = [
        "alpha_coarse_bin",
        "delta_coarse_bin",
        "alpha_fine_bin",
        "delta_fine_bin",
        "sky_cell_cat",
        "redshift_bin",
        "spec_x_pop",
        "spec_x_sky_cell",
        "pop_x_sky_cell",
        "gal_cell_cat",
        "gal_l_bin",
        "gal_b_bin",
        "spectral_type",
        "galaxy_population",
    ]
    for c in freq_candidates:
        if c in out.columns:
            freq = out[c].astype("string").value_counts(normalize=True, dropna=False)
            out[f"{c}_freq"] = (
                out[c].astype("string").map(freq).fillna(0.0).astype(np.float64)
            )

    # --- Percentile-rank covariates ---
    for c in ["u", "g", "r", "i", "z", "redshift"]:
        if c in out.columns:
            out[f"{c}_prank"] = out[c].rank(pct=True, method="average")

    # --- Optional auxiliary-derived standardization/quantile anchoring ---
    if isinstance(aux, pd.DataFrame) and not aux.empty:
        aux_clean = aux.copy()
        if set(["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]).issubset(
            aux_clean.columns
        ):
            for c in ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]:
                if c in aux_clean.columns:
                    v = pd.to_numeric(aux_clean[c], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    if c in {"u", "g", "r", "i", "z"}:
                        v = v[v > -9000]
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        continue
                    v_sorted = np.sort(v)
                    v_mean = float(v.mean())
                    v_std = float(v.std()) or 1.0
                    out[f"{c}_aux_q"] = np.searchsorted(
                        v_sorted, out[c].to_numpy(dtype=np.float64), side="right"
                    ) / len(v_sorted)
                    out[f"{c}_aux_z"] = (
                        out[c].to_numpy(dtype=np.float64) - v_mean
                    ) / v_std

            for a, b in color_pairs:
                if a in aux_clean.columns and b in aux_clean.columns:
                    av = pd.to_numeric(aux_clean[a], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    bv = pd.to_numeric(aux_clean[b], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    if a in {"u", "g", "r", "i", "z"}:
                        m = (av > -9000) & (bv > -9000)
                    else:
                        m = np.isfinite(av) & np.isfinite(bv)
                    av = av[m]
                    bv = bv[m]
                    if av.size == 0:
                        continue
                    c_aux = av - bv
                    c_aux = c_aux[np.isfinite(c_aux)]
                    if c_aux.size == 0:
                        continue
                    c_sorted = np.sort(c_aux)
                    c_mean = float(c_aux.mean())
                    c_std = float(c_aux.std()) or 1.0
                    if f"{a}_{b}_color" in out.columns:
                        v = out[f"{a}_{b}_color"].to_numpy(dtype=np.float64)
                        out[f"{a}_{b}_color_aux_q"] = np.searchsorted(
                            c_sorted, v, side="right"
                        ) / len(c_sorted)
                        out[f"{a}_{b}_color_aux_z"] = (v - c_mean) / c_std

    return out


def _force_autogluon_cpu_resources() -> None:
    if AIDE_AG_CONFIG.get("use_gpu") is not False:
        return
    try:
        from autogluon.common.utils.resource_utils import ResourceManager
    except Exception:
        return
    ResourceManager.get_gpu_count = staticmethod(lambda: 0)
    ResourceManager.get_gpu_count_torch = staticmethod(lambda cuda_only=False: 0)


def _read_csv(data_dir: Path, stem: str) -> pd.DataFrame:
    gz_path = data_dir / f"{stem}.csv.gz"
    csv_path = data_dir / f"{stem}.csv"
    if gz_path.exists():
        return pd.read_csv(gz_path)
    return pd.read_csv(csv_path)


def _read_aux_csv(input_dir: Path) -> pd.DataFrame | None:
    aux_file = AIDE_AG_CONFIG.get("aux_file")
    if not aux_file:
        return None
    aux_path = input_dir / str(aux_file)
    if not aux_path.exists():
        raise FileNotFoundError(f"Configured aux file not found: {aux_path}")
    return pd.read_csv(aux_path)


def _preprocess_accepts_aux() -> bool:
    signature = inspect.signature(preprocess)
    return len(signature.parameters) == 2


def _run_preprocess(combined: pd.DataFrame, aux_df: pd.DataFrame | None) -> pd.DataFrame:
    if _preprocess_accepts_aux():
        if aux_df is None:
            aux_df = pd.DataFrame()
        return preprocess(combined.copy(), aux_df.copy())
    return preprocess(combined.copy())


def _balanced_sample_weight(labels: pd.Series) -> np.ndarray:
    labels = pd.Series(labels).reset_index(drop=True)
    counts = labels.value_counts(dropna=False)
    if counts.empty:
        raise ValueError("Cannot compute class weights for empty labels")
    weights_by_class = len(labels) / (len(counts) * counts.astype(float))
    return labels.map(weights_by_class).astype(float).to_numpy()


def _positive_probability(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    *,
    model: str | None = None,
) -> pd.Series:
    proba = predictor.predict_proba(data, model=model)
    return _positive_probability_from_proba(proba)


def _positive_probability_from_proba(proba) -> pd.Series:
    if isinstance(proba, pd.Series):
        return proba.reset_index(drop=True)
    for positive_class in (1, 1.0, "1", "1.0", True):
        if positive_class in proba.columns:
            return proba[positive_class].reset_index(drop=True)
    return proba.iloc[:, -1].reset_index(drop=True)


def _prediction_from_proba(proba) -> pd.Series:
    if isinstance(proba, pd.Series):
        return proba.reset_index(drop=True)
    return proba.idxmax(axis=1).reset_index(drop=True)


def _values_from_proba(proba, *, eval_metric: str) -> pd.Series:
    if eval_metric == "roc_auc":
        return _positive_probability_from_proba(proba)
    return _prediction_from_proba(proba)


def _predict_values(
    predictor: TabularPredictor,
    data: pd.DataFrame,
    *,
    eval_metric: str,
    model: str | None = None,
) -> pd.Series:
    if eval_metric == "roc_auc":
        return _positive_probability(predictor, data, model=model)
    pred = predictor.predict(data, model=model)
    return pd.Series(pred).reset_index(drop=True)


def _make_combined_frame(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_part = train_df.copy()
    test_part = test_df.copy()
    return pd.concat([train_part, test_part], ignore_index=True, sort=False)


def _validate_preprocessed_frame(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    target_col: str,
    train_rows: int,
    test_rows: int,
) -> pd.DataFrame:
    if not isinstance(after, pd.DataFrame):
        raise TypeError("preprocess(df) must return a pandas DataFrame")
    if len(after) != len(before):
        raise ValueError(
            f"preprocess changed row count: {len(after)} != {len(before)}. "
            "AutoGluon preprocess(df) must preserve one output row per input row. "
            "If row removal was intentional, such as outlier filtering, rewrite it "
            "without dropping rows: add an outlier flag, clipped/winsorized value, "
            "imputed clean value, anomaly score, or distance-from-normal feature instead."
        )
    if target_col in after.columns:
        raise ValueError(f"preprocess created forbidden target column: {target_col}")
    if FORBIDDEN_SPLIT_MARKER in after.columns:
        raise ValueError(f"preprocess created forbidden split marker column: {FORBIDDEN_SPLIT_MARKER}")
    if FORBIDDEN_ROW_ID in after.columns:
        raise ValueError(f"preprocess created forbidden row id column: {FORBIDDEN_ROW_ID}")
    if CLASS_WEIGHT_COL in after.columns:
        raise ValueError(f"preprocess created forbidden class weight column: {CLASS_WEIGHT_COL}")

    ordered = after.reset_index(drop=True)
    if len(ordered.iloc[:train_rows]) != train_rows:
        raise ValueError("preprocess changed number of train rows")
    if len(ordered.iloc[train_rows:]) != test_rows:
        raise ValueError("preprocess changed number of test rows")
    return ordered


def _configured_metric() -> str:
    return AIDE_AG_CONFIG["eval_metric"]


def _should_stratify_holdout(target: pd.Series) -> bool:
    unique_count = target.nunique(dropna=True)
    if unique_count == 2:
        return True
    if pd.api.types.is_object_dtype(target) or pd.api.types.is_categorical_dtype(target):
        return True
    return False


def _json_safe_scalar(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _leaderboard_records(predictor: TabularPredictor) -> list[dict]:
    keep_columns = [
        "model",
        "score_val",
        "score_test",
        "eval_metric",
        "fit_time",
        "fit_time_marginal",
        "pred_time_val",
        "pred_time_val_marginal",
        "stack_level",
        "can_infer",
        "fit_order",
    ]
    try:
        leaderboard = predictor.leaderboard(silent=True)
    except Exception as exc:
        return [{"error": f"leaderboard unavailable: {exc}"}]
    records = []
    for row in leaderboard.to_dict(orient="records"):
        records.append(
            {
                column: _json_safe_scalar(row.get(column))
                for column in keep_columns
                if column in row
            }
        )
    return records


def _artifact_dir(working_dir: Path) -> Path:
    return Path(os.environ.get("AIDE_NODE_ARTIFACT_DIR", str(working_dir)))


def _save_submission(submission: pd.DataFrame, working_dir: Path) -> Path:
    submission_path = working_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    artifact_dir = _artifact_dir(working_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_submission_path = artifact_dir / "submission.csv"
    if artifact_submission_path.resolve() != submission_path.resolve():
        shutil.copy2(submission_path, artifact_submission_path)
    return submission_path


def _save_prediction_artifact(frame: pd.DataFrame, working_dir: Path, filename: str) -> Path:
    if not filename.endswith(".gz"):
        filename = f"{filename}.gz"
    working_path = working_dir / filename
    working_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(working_path, index=False, compression="gzip")
    artifact_dir = _artifact_dir(working_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.resolve() != working_path.resolve():
        shutil.copy2(working_path, artifact_path)
    return working_path


def _safe_prediction_name(name: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
    return safe or "model"


def _save_autogluon_prediction_artifacts(
    predictor: TabularPredictor,
    *,
    train_target: pd.Series,
    test_model: pd.DataFrame,
    test_ids: pd.Series,
    test_pred: pd.Series,
    eval_metric: str,
    working_dir: Path,
    id_col: str,
    target_col: str,
    valid_data: pd.DataFrame | None,
    valid_pred: pd.Series | None,
) -> dict:
    artifacts = {}
    test_predictions = pd.DataFrame({
        id_col: pd.Series(test_ids).reset_index(drop=True),
        target_col: pd.Series(test_pred).reset_index(drop=True),
    })
    test_path = _save_prediction_artifact(
        test_predictions,
        working_dir,
        "test_predictions.csv",
    )
    artifacts["test_predictions"] = str(test_path)

    try:
        per_model = []
        for model_name in predictor.model_names():
            try:
                model_oof_proba = predictor.predict_proba_oof(
                    model=model_name,
                    transformed=False,
                    as_multiclass=True,
                )
                model_oof_pred = _values_from_proba(
                    model_oof_proba,
                    eval_metric=eval_metric,
                )
                if len(model_oof_pred) != len(train_target):
                    raise ValueError(
                        f"OOF row count {len(model_oof_pred)} != train rows {len(train_target)}"
                    )
                safe_model_name = _safe_prediction_name(model_name)
                model_oof_frame = pd.DataFrame({
                    "row": np.arange(len(train_target)),
                    "target": pd.Series(train_target).reset_index(drop=True),
                    "prediction": model_oof_pred.reset_index(drop=True),
                    "model": model_name,
                })
                model_oof_path = _save_prediction_artifact(
                    model_oof_frame,
                    working_dir,
                    f"model_predictions/{safe_model_name}-oof.csv",
                )
                model_test_pred = _predict_values(
                    predictor,
                    test_model,
                    eval_metric=eval_metric,
                    model=model_name,
                )
                model_test_frame = pd.DataFrame({
                    id_col: pd.Series(test_ids).reset_index(drop=True),
                    target_col: pd.Series(model_test_pred).reset_index(drop=True),
                    "model": model_name,
                })
                model_test_path = _save_prediction_artifact(
                    model_test_frame,
                    working_dir,
                    f"model_predictions/{safe_model_name}-test.csv",
                )
                per_model.append({
                    "model": model_name,
                    "oof_predictions": str(model_oof_path),
                    "test_predictions": str(model_test_path),
                    "rows": int(len(model_oof_frame)),
                    "test_rows": int(len(model_test_frame)),
                })
            except Exception as exc:
                per_model.append({
                    "model": model_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        artifacts["model_predictions"] = per_model
        artifacts["model_predictions_ok"] = sum(1 for item in per_model if "error" not in item)

        oof_proba = predictor.predict_proba_oof(
            transformed=False,
            as_multiclass=True,
        )
        oof_pred = _values_from_proba(oof_proba, eval_metric=eval_metric)
        if len(oof_pred) != len(train_target):
            raise ValueError(f"OOF row count {len(oof_pred)} != train rows {len(train_target)}")
        oof_frame = pd.DataFrame({
            "row": np.arange(len(train_target)),
            "target": pd.Series(train_target).reset_index(drop=True),
            "prediction": oof_pred.reset_index(drop=True),
        })
        oof_path = _save_prediction_artifact(
            oof_frame,
            working_dir,
            "oof_predictions.csv",
        )
        artifacts["oof_predictions"] = str(oof_path)
        artifacts["oof_rows"] = int(len(oof_frame))
    except Exception as exc:
        artifacts["oof_error"] = f"{type(exc).__name__}: {exc}"

    if valid_data is not None and valid_pred is not None:
        validation_frame = pd.DataFrame({
            "row": np.arange(len(valid_data)),
            "target": valid_data[target_col].reset_index(drop=True),
            "prediction": pd.Series(valid_pred).reset_index(drop=True),
        })
        validation_path = _save_prediction_artifact(
            validation_frame,
            working_dir,
            "validation_predictions.csv",
        )
        artifacts["validation_predictions"] = str(validation_path)
        artifacts["validation_rows"] = int(len(validation_frame))
    return artifacts


def _make_submission(
    sample_submission: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    test_ids: pd.Series,
    test_pred: pd.Series,
) -> pd.DataFrame:
    prediction_frame = pd.DataFrame({
        id_col: pd.Series(test_ids).reset_index(drop=True),
        target_col: pd.Series(test_pred).reset_index(drop=True),
    })
    if prediction_frame[id_col].duplicated().any():
        raise ValueError(f"test data contains duplicate {id_col} values")

    submission = sample_submission.copy()
    mapped = submission[[id_col]].merge(
        prediction_frame,
        on=id_col,
        how="left",
        validate="one_to_one",
    )[target_col]
    if mapped.isna().any():
        missing = int(mapped.isna().sum())
        raise ValueError(f"missing predictions for {missing} sample_submission ids")

    submission[target_col] = mapped.to_numpy()
    return submission.sort_values(id_col, kind="mergesort").reset_index(drop=True)


@contextlib.contextmanager
def _preprocess_timeout(seconds: int):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    class PreprocessTimeoutError(TimeoutError):
        pass

    def _raise_preprocess_timeout(_signum, _frame):
        raise PreprocessTimeoutError(
            "AIDE AutoGluon preprocess exceeded the dedicated timeout of "
            f"{seconds} seconds. This timeout is separate from AutoGluon "
            "training time_limit. Analyze preprocess(df) and remove or replace "
            "time-consuming operations such as Python callbacks over groups or "
            "rolling windows, polynomial fitting, row-wise loops, or repeated "
            "full-frame copies."
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _raise_preprocess_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextlib.contextmanager
def _quiet_model_output(working_dir: Path):
    artifact_dir = _artifact_dir(working_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "autogluon_stdout.log"

    class _TeeWriter:
        def __init__(self, primary, log_file):
            self.primary = primary
            self.log_file = log_file

        def write(self, text):
            written = self.primary.write(text)
            self.log_file.write(text)
            if hasattr(self.primary, "flush"):
                self.primary.flush()
            self.log_file.flush()
            return written

        def flush(self):
            if hasattr(self.primary, "flush"):
                self.primary.flush()
            self.log_file.flush()

        def fileno(self):
            if hasattr(self.primary, "fileno"):
                return self.primary.fileno()
            fallback = getattr(sys, "__stderr__", None) or getattr(sys, "__stdout__", None)
            if fallback is not None and hasattr(fallback, "fileno"):
                return fallback.fileno()
            return self.log_file.fileno()

        def isatty(self):
            return bool(self.primary.isatty()) if hasattr(self.primary, "isatty") else False

        @property
        def encoding(self):
            return getattr(self.primary, "encoding", "utf-8")

    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        stdout_writer = _TeeWriter(sys.stdout, log_file)
        stderr_writer = _TeeWriter(sys.stderr, log_file)
        log_handler = logging.StreamHandler(stderr_writer)
        log_handler.setFormatter(logging.Formatter("%(message)s"))
        logger_names = ["", "autogluon"]
        loggers = [logging.getLogger(name) for name in logger_names]
        previous_states = [
            (
                logger.level,
                logger.disabled,
                list(logger.handlers),
                logger.propagate,
            )
            for logger in loggers
        ]
        for logger in loggers:
            logger.disabled = False
            logger.setLevel(logging.INFO)
            logger.handlers = [log_handler]
            logger.propagate = False
        try:
            with contextlib.redirect_stdout(stdout_writer), contextlib.redirect_stderr(stderr_writer):
                yield
        finally:
            for logger, (level, disabled, handlers, propagate) in zip(
                loggers,
                previous_states,
            ):
                logger.setLevel(level)
                logger.disabled = disabled
                logger.handlers = handlers
                logger.propagate = propagate


def main() -> None:
    _force_autogluon_cpu_resources()

    input_dir = Path("./input")
    working_dir = Path("./working")
    working_dir.mkdir(parents=True, exist_ok=True)

    train_df = _read_csv(input_dir, "train")
    test_df = _read_csv(input_dir, "test")
    sample_submission = _read_csv(input_dir, "sample_submission")
    aux_df = _read_aux_csv(input_dir)
    id_col = sample_submission.columns[0]
    target_col = sample_submission.columns[1]
    if target_col not in train_df.columns:
        raise ValueError(f"Target column {target_col!r} not found in train data")

    y_train = train_df[target_col].reset_index(drop=True)
    train_features = train_df.drop(columns=[target_col, id_col], errors="ignore")
    test_features = test_df.drop(columns=[id_col], errors="ignore")
    combined = _make_combined_frame(train_features, test_features)
    if aux_df is not None:
        print(
            "AIDE AutoGluon: loaded aux file "
            f"{AIDE_AG_CONFIG.get('aux_file')} rows={len(aux_df)} "
            f"cols={len(aux_df.columns)} "
            f"passed_to_preprocess={_preprocess_accepts_aux()}",
            flush=True,
        )
    print("AIDE AutoGluon: starting preprocess", flush=True)
    preprocess_started_at = time.time()
    with _preprocess_timeout(int(AIDE_AG_CONFIG.get("preprocess_timeout", 180))):
        preprocessed = _run_preprocess(combined, aux_df)
    preprocess_time = time.time() - preprocess_started_at
    feature_count = int(len(preprocessed.columns))
    print(
        f"AIDE AutoGluon: finished preprocess rows={len(preprocessed)} cols={feature_count}",
        flush=True,
    )
    preprocessed = _validate_preprocessed_frame(
        combined,
        preprocessed,
        target_col=target_col,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )

    train_fe = preprocessed.iloc[:len(train_df)].copy()
    test_fe = preprocessed.iloc[len(train_df):].copy()
    train_model = train_fe.copy()
    train_model[target_col] = y_train.to_numpy()
    if AIDE_AG_CONFIG.get("class_balance") == "balanced":
        train_model[CLASS_WEIGHT_COL] = _balanced_sample_weight(y_train)
    test_model = test_fe.copy()

    eval_metric = _configured_metric()
    fit_args = dict(AIDE_AG_CONFIG.get("fit_args", {}) or {})
    bagged_mode = int(fit_args.get("num_bag_folds") or 0) > 0 or bool(fit_args.get("auto_stack"))
    defer_save_space = bool(bagged_mode and fit_args.pop("save_space", False))
    if bagged_mode:
        train_data = train_model
        valid_data = None
        print(
            "AIDE AutoGluon: bagged mode detected; using internal OOF validation without tuning_data",
            flush=True,
        )
    elif AIDE_AG_CONFIG.get("validation_strategy") == "holdout":
        stratify = train_model[target_col] if _should_stratify_holdout(train_model[target_col]) else None
        train_data, valid_data = train_test_split(
            train_model,
            test_size=AIDE_AG_CONFIG.get("validation_fraction", 0.2),
            random_state=AIDE_AG_CONFIG.get("seed", 42),
            stratify=stratify,
        )
    else:
        train_data = train_model
        valid_data = None

    model_dir = working_dir / "autogluon_model"
    shutil.rmtree(model_dir, ignore_errors=True)
    fit_kwargs = {
        "train_data": train_data,
        "presets": AIDE_AG_CONFIG["presets"],
        "time_limit": AIDE_AG_CONFIG["time_limit"],
    }
    if AIDE_AG_CONFIG.get("use_gpu") is not None:
        fit_kwargs["num_gpus"] = 1 if AIDE_AG_CONFIG["use_gpu"] else 0
    if valid_data is not None:
        fit_kwargs["tuning_data"] = valid_data
    if AIDE_AG_CONFIG.get("included_model_types"):
        fit_kwargs["included_model_types"] = AIDE_AG_CONFIG["included_model_types"]
    if AIDE_AG_CONFIG.get("hyperparameters"):
        fit_kwargs["hyperparameters"] = AIDE_AG_CONFIG["hyperparameters"]
    fit_kwargs.update(fit_args)
    training_started_at = time.time()
    with _quiet_model_output(working_dir):
        print("AIDE AutoGluon: starting fit", flush=True)
        predictor_kwargs = {
            "label": target_col,
            "eval_metric": eval_metric,
            "path": str(model_dir),
            "verbosity": 2,
        }
        if AIDE_AG_CONFIG.get("class_balance") == "balanced":
            predictor_kwargs["sample_weight"] = CLASS_WEIGHT_COL
            predictor_kwargs["weight_evaluation"] = False
        predictor = TabularPredictor(**predictor_kwargs)
        predictor.fit(**fit_kwargs)
        print("AIDE AutoGluon: finished fit", flush=True)
    actual_eval_metric = str(getattr(predictor.eval_metric, "name", predictor.eval_metric))
    training_time = time.time() - training_started_at
    model_records = _leaderboard_records(predictor)

    with _quiet_model_output(working_dir):
        print("AIDE AutoGluon: starting validation and prediction", flush=True)
        valid_pred = None
        if valid_data is None:
            leaderboard = predictor.leaderboard(silent=True)
            score_candidates = [
                col for col in ("score_val", "score_test") if col in leaderboard.columns
            ]
            metric_value = float(leaderboard[score_candidates[0]].max()) if score_candidates else float("nan")
            lower_is_better = False
        elif eval_metric == "roc_auc":
            valid_pred = _positive_probability(
                predictor,
                valid_data.drop(columns=[target_col, CLASS_WEIGHT_COL], errors="ignore"),
            )
            metric_value = float(roc_auc_score(valid_data[target_col], valid_pred))
            lower_is_better = False
        else:
            valid_pred = _predict_values(
                predictor,
                valid_data.drop(columns=[target_col, CLASS_WEIGHT_COL], errors="ignore"),
                eval_metric=eval_metric,
            )
            scores = predictor.evaluate(valid_data, silent=True)
            metric_value = float(scores.get(eval_metric))
            lower_is_better = False

        test_pred = _predict_values(
            predictor,
            test_model,
            eval_metric=eval_metric,
        )
        prediction_artifacts = _save_autogluon_prediction_artifacts(
            predictor,
            train_target=y_train,
            test_model=test_model,
            test_ids=test_df[id_col],
            test_pred=test_pred,
            eval_metric=eval_metric,
            working_dir=working_dir,
            id_col=id_col,
            target_col=target_col,
            valid_data=valid_data,
            valid_pred=valid_pred,
        )
        if defer_save_space:
            try:
                predictor.save_space(remove_data=True, remove_fit_stack=True)
                prediction_artifacts["save_space_after_artifacts"] = True
            except Exception as exc:
                prediction_artifacts["save_space_error"] = f"{type(exc).__name__}: {exc}"
    submission = _make_submission(
        sample_submission,
        id_col=id_col,
        target_col=target_col,
        test_ids=test_df[id_col],
        test_pred=test_pred,
    )
    submission_path = _save_submission(submission, working_dir)
    artifact_submission_path = _artifact_dir(working_dir) / "submission.csv"
    print("AIDE AutoGluon: finished validation and prediction", flush=True)
    print(f"AIDE AutoGluon: submission saved to {submission_path}", flush=True)
    if artifact_submission_path.resolve() != submission_path.resolve():
        print(f"AIDE AutoGluon: artifact submission saved to {artifact_submission_path}", flush=True)
    if prediction_artifacts.get("oof_predictions"):
        print(f"AIDE AutoGluon: OOF predictions saved to {prediction_artifacts['oof_predictions']}", flush=True)
    elif prediction_artifacts.get("oof_error"):
        print(f"AIDE AutoGluon: OOF predictions unavailable: {prediction_artifacts['oof_error']}", flush=True)
    if prediction_artifacts.get("validation_predictions"):
        print(f"AIDE AutoGluon: validation predictions saved to {prediction_artifacts['validation_predictions']}", flush=True)
    print(f"AIDE AutoGluon: test predictions saved to {prediction_artifacts['test_predictions']}", flush=True)

    summary = "AutoGluon preprocess wrapper completed."
    run_stats = {
        "feature_count": feature_count,
        "preprocess_time": float(preprocess_time),
        "training_time": float(training_time),
        "eval_metric": actual_eval_metric,
        "models": model_records,
        "prediction_artifacts": prediction_artifacts,
    }
    print(f"Validation {eval_metric}: {metric_value:.6f}")
    print("Submission saved successfully.")
    print(RESULT_MARKER + " " + json.dumps({
        "is_bug": False,
        "summary": summary,
        "metric": metric_value,
        "eval_metric": actual_eval_metric,
        "lower_is_better": lower_is_better,
        "run_stats": run_stats,
        
    }, sort_keys=True))


# AIDE executes generated code via exec with a custom globals dict, where __name__ is not
# guaranteed to be "__main__". Call main() directly so the wrapper actually runs
# inside the interpreter.
main()
