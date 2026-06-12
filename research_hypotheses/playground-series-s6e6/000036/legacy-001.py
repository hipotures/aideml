from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from catboost import CatBoostClassifier

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)


class EcdFGenerator:
    def __init__(
        self,
        anchor_features: List[str],
        group_columns: List[str],
        min_group_size: int = 2500,
        lower_tail: float = 0.001,
        upper_tail: float = 0.999,
        rank_clip: float = 1e-6,
        rank_cols_for_distance: List[str] | None = None,
    ):
        self.anchor_features = anchor_features
        self.group_columns = group_columns
        self.min_group_size = min_group_size
        self.lower_tail = lower_tail
        self.upper_tail = upper_tail
        self.rank_clip = rank_clip
        self.rank_cols_for_distance = rank_cols_for_distance or []
        self._reset_state()

    def _reset_state(self):
        self.pooled_refs: Dict[str, np.ndarray] = {}
        self.group_refs: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.rank_mean: np.ndarray | None = None
        self.rank_inv_cov: np.ndarray | None = None

    @staticmethod
    def _clean_numeric(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return arr[np.isfinite(arr)]

    @staticmethod
    def _sorted(arr: np.ndarray) -> np.ndarray:
        vals = np.sort(arr.astype(float))
        return vals

    @staticmethod
    def _winsor_q(
        df: pd.DataFrame, cols: List[str], lo: float, hi: float
    ) -> Dict[str, Tuple[float, float]]:
        bounds = {}
        for c in cols:
            raw = df[c].to_numpy(dtype=float)
            raw = raw[np.isfinite(raw)]
            if raw.size == 0:
                bounds[c] = (0.0, 1.0)
            else:
                bounds[c] = (np.quantile(raw, lo), np.quantile(raw, hi))
        return bounds

    @staticmethod
    def _apply_winsor(
        df: pd.DataFrame, cols: List[str], bounds: Dict[str, Tuple[float, float]]
    ) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            lo, hi = bounds[c]
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                out[c] = out[c].clip(lower=lo, upper=hi)
        return out

    @staticmethod
    def _ecdf(x: np.ndarray, ref_sorted: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        out = np.full(arr.shape, 0.5, dtype=float)
        if ref_sorted.size == 0:
            return out
        if ref_sorted.size == 1:
            out[:] = 0.5
            return out
        mask = np.isfinite(arr)
        ranks = np.searchsorted(ref_sorted, arr[mask], side="right")
        out[mask] = ranks / ref_sorted.size
        return out

    def _transform_from_rank(self, p: np.ndarray) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        p = np.clip(p, self.rank_clip, 1.0 - self.rank_clip)
        logit_p = np.log(p / (1.0 - p))
        norm_p = norm.ppf(p)
        lower = p
        upper = 1.0 - p
        tail2 = 2.0 * np.minimum(lower, upper)
        tail2 = np.clip(tail2, 1e-15, 1.0)
        nlog_tail2 = -np.log(tail2)
        return p, logit_p, norm_p, lower, upper, tail2, nlog_tail2

    def fit(self, df: pd.DataFrame) -> None:
        self._reset_state()

        # pooled refs
        for col in self.anchor_features:
            arr = self._clean_numeric(df[col].to_numpy(dtype=float))
            arr = arr[(arr >= -1.0e9) & (arr <= 1.0e9)]
            self.pooled_refs[col] = self._sorted(arr)

        # group-specific refs
        for gcol in self.group_columns:
            self.group_refs[gcol] = {}
            for gval, sub in df.groupby(df[gcol], dropna=True):
                if len(sub) < self.min_group_size:
                    continue
                self.group_refs[gcol][str(gval)] = {}
                for col in self.anchor_features:
                    arr = self._clean_numeric(sub[col].to_numpy(dtype=float))
                    arr = arr[(arr >= -1.0e9) & (arr <= 1.0e9)]
                    self.group_refs[gcol][str(gval)][col] = self._sorted(arr)

        # rank-space Gaussian statistics for distance features
        rank_cols = [
            c for c in self.rank_cols_for_distance if c in self.anchor_features
        ]
        if len(rank_cols) == 0:
            self.rank_mean = None
            self.rank_inv_cov = None
            return

        rank_matrix = []
        for c in rank_cols:
            p = self._ecdf(
                df[c].to_numpy(dtype=float), self.pooled_refs.get(c, np.array([0.5]))
            )
            _, _, z, _, _, _, _ = self._transform_from_rank(p)
            rank_matrix.append(z)
        rank_matrix = np.column_stack(rank_matrix)
        if rank_matrix.size == 0 or rank_matrix.shape[0] < 2:
            self.rank_mean = (
                np.zeros(rank_matrix.shape[1])
                if rank_matrix.ndim == 2
                else np.array([0.0])
            )
            self.rank_inv_cov = np.eye(len(self.rank_mean))
            return

        rank_matrix = np.nan_to_num(rank_matrix, nan=0.0)
        self.rank_mean = rank_matrix.mean(axis=0)
        if rank_matrix.shape[1] == 1:
            var = rank_matrix.var(axis=0, ddof=1)
            var = float(var[0]) if np.ndim(var) else float(var)
            self.rank_inv_cov = np.array([[1.0 / max(var, 1e-6)]])
        else:
            cov = np.cov(rank_matrix, rowvar=False)
            if not np.all(np.isfinite(cov)):
                cov = np.eye(rank_matrix.shape[1])
            cov = cov + np.eye(rank_matrix.shape[1]) * 1e-6
            self.rank_inv_cov = np.linalg.pinv(cov)

    def transform(
        self, df: pd.DataFrame, verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        if verbose:
            log_stage("building ECDF-tail features")

        out = df.copy()
        fallback_counts: Dict[str, int] = {}
        feature_names: List[str] = []
        meta_tail_rates: Dict[str, float] = {}

        for col in self.anchor_features:
            # pooled features
            pooled_ref = self.pooled_refs.get(col, np.array([0.5]))
            p_pool = self._ecdf(out[col].to_numpy(dtype=float), pooled_ref)

            p, lp, z, tlow, thigh, t2, nlog = self._transform_from_rank(p_pool)
            out[f"{col}_pct_pool"] = p
            out[f"{col}_logit_pool"] = lp
            out[f"{col}_norm_pool"] = z
            out[f"{col}_tail_low_pool"] = tlow
            out[f"{col}_tail_high_pool"] = thigh
            out[f"{col}_tail2_pool"] = t2
            out[f"{col}_nlog_tail2_pool"] = nlog
            feature_names.extend(
                [
                    f"{col}_pct_pool",
                    f"{col}_logit_pool",
                    f"{col}_norm_pool",
                    f"{col}_tail_low_pool",
                    f"{col}_tail_high_pool",
                    f"{col}_tail2_pool",
                    f"{col}_nlog_tail2_pool",
                ]
            )

            # metadata-conditional ECDFs
            for gcol in self.group_columns:
                gvals = out[gcol].astype("string").fillna("__MISSING__").to_numpy()
                grp_vals = np.full(len(out), np.nan, dtype=float)
                grp_logit = np.full(len(out), np.nan, dtype=float)
                grp_norm = np.full(len(out), np.nan, dtype=float)
                grp_tlow = np.full(len(out), np.nan, dtype=float)
                grp_thigh = np.full(len(out), np.nan, dtype=float)
                grp_t2 = np.full(len(out), np.nan, dtype=float)
                grp_nlog = np.full(len(out), np.nan, dtype=float)

                fallback_count = 0
                for g in np.unique(gvals):
                    idx = gvals == g
                    if not idx.any():
                        continue
                    group_ref = self.group_refs.get(gcol, {}).get(str(g), {}).get(col)
                    if group_ref is None or len(group_ref) < self.min_group_size:
                        group_ref = pooled_ref
                        fallback_count += int(idx.sum())
                    p_grp = self._ecdf(
                        out.loc[idx, col].to_numpy(dtype=float), group_ref
                    )
                    gp, gl, gz, glow, ghigh, gt2, gnlog = self._transform_from_rank(
                        p_grp
                    )

                    grp_vals[idx] = gp
                    grp_logit[idx] = gl
                    grp_norm[idx] = gz
                    grp_tlow[idx] = glow
                    grp_thigh[idx] = ghigh
                    grp_t2[idx] = gt2
                    grp_nlog[idx] = gnlog

                if verbose:
                    fallback_counts[f"{col}|{gcol}"] = fallback_count

                out[f"{col}_pct_{gcol}"] = grp_vals
                out[f"{col}_logit_{gcol}"] = grp_logit
                out[f"{col}_norm_{gcol}"] = grp_norm
                out[f"{col}_tail_low_{gcol}"] = grp_tlow
                out[f"{col}_tail_high_{gcol}"] = grp_thigh
                out[f"{col}_tail2_{gcol}"] = grp_t2
                out[f"{col}_nlog_tail2_{gcol}"] = grp_nlog
                feature_names.extend(
                    [
                        f"{col}_pct_{gcol}",
                        f"{col}_logit_{gcol}",
                        f"{col}_norm_{gcol}",
                        f"{col}_tail_low_{gcol}",
                        f"{col}_tail_high_{gcol}",
                        f"{col}_tail2_{gcol}",
                        f"{col}_nlog_tail2_{gcol}",
                    ]
                )

            pool_tail = out[f"{col}_tail2_pool"]
            meta_tail_rates[f"{col}_tail_1pct"] = float(np.mean(pool_tail < 0.01))
            meta_tail_rates[f"{col}_tail_2p5pct"] = float(np.mean(pool_tail < 0.025))
            meta_tail_rates[f"{col}_tail_5pct"] = float(np.mean(pool_tail < 0.05))

        # row-level tail summaries
        tail_pool_cols = [f"{c}_tail2_pool" for c in self.anchor_features]
        tail_pool = out[tail_pool_cols]
        out["tail_surprise_sum_pool"] = (
            -np.log(np.clip(2.0 * np.minimum(tail_pool, 1.0 - tail_pool), 1e-15, 1.0))
        ).sum(axis=1)
        out["tail_surprise_mean_pool"] = (
            -np.log(np.clip(2.0 * np.minimum(tail_pool, 1.0 - tail_pool), 1e-15, 1.0))
        ).mean(axis=1)
        out["tail_surprise_max_pool"] = (
            -np.log(np.clip(2.0 * np.minimum(tail_pool, 1.0 - tail_pool), 1e-15, 1.0))
        ).max(axis=1)
        out["n_tail_le_1pct_pool"] = (tail_pool < 0.01).sum(axis=1)
        out["n_tail_le_2p5pct_pool"] = (tail_pool < 0.025).sum(axis=1)
        out["n_tail_le_5pct_pool"] = (tail_pool < 0.05).sum(axis=1)

        blue_anchors = {"u", "g", "u_g", "g_r", "r_i"}
        red_anchors = {"r", "i", "z", "i_z", "u_r", "g_z"}
        blue_tail_cols = [
            f"{c}_tail2_pool" for c in self.anchor_features if c in blue_anchors
        ]
        red_tail_cols = [
            f"{c}_tail2_pool" for c in self.anchor_features if c in red_anchors
        ]
        if blue_tail_cols:
            out["n_tail_blue_5pct_pool"] = (out[blue_tail_cols] < 0.05).sum(axis=1)
        else:
            out["n_tail_blue_5pct_pool"] = 0.0
        if red_tail_cols:
            out["n_tail_red_5pct_pool"] = (out[red_tail_cols] < 0.05).sum(axis=1)
        else:
            out["n_tail_red_5pct_pool"] = 0.0

        # rank-space distance summary
        rank_features = [
            f"{c}_norm_pool"
            for c in self.rank_cols_for_distance
            if f"{c}_norm_pool" in out.columns
        ]
        if (
            self.rank_mean is not None
            and self.rank_inv_cov is not None
            and len(rank_features) > 0
        ):
            z = np.nan_to_num(out[rank_features].to_numpy(dtype=float), nan=0.0)
            diff = z - self.rank_mean
            out["copula_rank_mdistance_pool"] = np.sqrt(
                np.maximum(np.einsum("ij,jk,ik->i", diff, self.rank_inv_cov, diff), 0.0)
            )
        else:
            out["copula_rank_mdistance_pool"] = 0.0

        # ensure categorical columns are not accidentally numeric-converted
        for g in self.group_columns:
            if g in out.columns:
                out[g] = out[g].astype("string")

        out = out.drop(columns=["id"], errors="ignore")
        return out, {
            "group_fallback_counts": fallback_counts,
            "tail_rates": meta_tail_rates,
            "feature_names": feature_names,
        }


def add_color_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_z"] = out["g"] - out["z"]
    return out


def build_model(seed: int = 42, use_gpu: bool = True) -> CatBoostClassifier:
    if use_gpu:
        return CatBoostClassifier(
            iterations=400,
            learning_rate=0.08,
            depth=8,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            random_seed=seed,
            auto_class_weights="Balanced",
            verbose=False,
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.8,
        )
    return CatBoostClassifier(
        iterations=400,
        learning_rate=0.08,
        depth=8,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=seed,
        auto_class_weights="Balanced",
        verbose=False,
        task_type="CPU",
    )


def main() -> None:
    with aide_stage("build_features_stage"):
        train, test, sample_sub = load_competition_data()

        if {"class"} != {"class"}:
            raise RuntimeError("Unexpected data format from load_competition_data().")

        X_train_raw = train.drop(columns=["class", "id"]).copy()
        y = train["class"].astype("string").to_numpy()
        X_test_raw = test.drop(columns=["id"]).copy()

        classes = np.array(sorted(pd.Series(y).unique()))
        y_encoded = np.array([np.where(classes == v)[0][0] for v in y], dtype=int)

        numeric_clip_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

        anchor_features = [
            "alpha",
            "delta",
            "u",
            "g",
            "r",
            "i",
            "z",
            "redshift",
            "u_g",
            "g_r",
            "r_i",
            "i_z",
            "u_r",
            "g_z",
        ]
        group_columns = ["spectral_type", "galaxy_population"]
        rank_distance_cols = ["u", "g", "r", "i", "z", "redshift"]

        transformer = EcdFGenerator(
            anchor_features=anchor_features,
            group_columns=group_columns,
            min_group_size=2500,
            lower_tail=0.001,
            upper_tail=0.999,
            rank_cols_for_distance=rank_distance_cols,
        )

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)

        oof_proba = np.zeros((len(train), len(classes)), dtype=float)
        test_proba = np.zeros((len(test), len(classes)), dtype=float)
        fold_scores: List[float] = []
        all_meta = []

    with aide_stage("fit_predict_fold_stage"):
        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_train_raw, y_encoded), start=1
        ):
            tr = X_train_raw.iloc[train_idx].copy()
            val = X_train_raw.iloc[val_idx].copy()

            # Transductive, target-free pooled statistics: fold-train + test
            pool_ref = pd.concat([tr, X_test_raw], axis=0, ignore_index=True)

            bounds = EcdFGenerator._winsor_q(
                pool_ref, numeric_clip_cols, lo=0.001, hi=0.999
            )
            tr = EcdFGenerator._apply_winsor(tr, numeric_clip_cols, bounds)
            val = EcdFGenerator._apply_winsor(val, numeric_clip_cols, bounds)
            test_fold = EcdFGenerator._apply_winsor(
                X_test_raw, numeric_clip_cols, bounds
            )

            tr = add_color_features(tr)
            val = add_color_features(val)
            test_fold = add_color_features(test_fold)

            trans_ref = pd.concat([tr, test_fold], axis=0, ignore_index=True)
            transformer.fit(trans_ref)

            tr_fe, tr_meta = transformer.transform(tr, verbose=False)
            val_fe, val_meta = transformer.transform(val, verbose=False)
            test_fe, test_meta = transformer.transform(test_fold, verbose=False)

            tr_meta["fold"] = fold_idx
            val_meta["fold"] = fold_idx
            test_meta["fold"] = fold_idx
            all_meta.append(tr_meta)
            all_meta.append(val_meta)
            all_meta.append(test_meta)

            # log fallback behavior / no-op checks
            log_stage(
                f"fold={fold_idx} group_fallback_counts="
                f"{ {k: int(v) for k, v in tr_meta['group_fallback_counts'].items()} }"
            )
            log_stage(
                f"fold={fold_idx} mean tail rates="
                f"{ {k: float(v) for k, v in tr_meta['tail_rates'].items()} }"
            )

            # Cat features (string/object)
            cat_cols = [
                c
                for c in tr_fe.columns
                if tr_fe[c].dtype == "string" or tr_fe[c].dtype == "object"
            ]
            cat_idxs = [tr_fe.columns.get_loc(c) for c in cat_cols]

            print(
                f"fold={fold_idx} training CatBoostClassifier (GPU attempt)", flush=True
            )
            X_tr = tr_fe
            X_val = val_fe
            y_tr = y_encoded[train_idx]
            y_val = y_encoded[val_idx]

            try:
                model = build_model(seed=42 + fold_idx, use_gpu=True)
                model.fit(
                    X_tr,
                    y_tr,
                    cat_features=cat_idxs,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                )
            except Exception as e:
                print(
                    f"fold={fold_idx} GPU fitting failed: {repr(e)}. Falling back to CPU.",
                    flush=True,
                )
                model = build_model(seed=42 + fold_idx, use_gpu=False)
                model.fit(
                    X_tr,
                    y_tr,
                    cat_features=cat_idxs,
                    eval_set=(X_val, y_val),
                    use_best_model=True,
                )

            val_pred_proba = model.predict_proba(X_val)
            oof_proba[val_idx, :] = val_pred_proba
            test_proba += model.predict_proba(test_fe) / 5.0

            val_pred = np.argmax(val_pred_proba, axis=1)
            bal = balanced_accuracy_score(y_val, val_pred)
            fold_scores.append(float(bal))
            print(f"fold={fold_idx} balanced_accuracy={bal:.6f}", flush=True)

    with aide_stage("score_stage"):
        oof_pred = classes[np.argmax(oof_proba, axis=1)]
        oof_true = classes[y_encoded]
        oof_score = balanced_accuracy_score(oof_true, oof_pred)
        print(f"cv_balanced_accuracy_mean={np.mean(fold_scores):.6f}", flush=True)
        print(f"cv_balanced_accuracy_std={np.std(fold_scores):.6f}", flush=True)
        print(f"oof_balanced_accuracy={oof_score:.6f}", flush=True)

    with aide_stage("write_outputs_stage"):
        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train), dtype=int),
                "target": oof_true,
                "prediction": oof_pred.astype(str),
            }
        )
        write_oof_predictions(oof_df)

        test_pred_df = pd.DataFrame(test_proba, columns=list(classes))
        test_pred_df.insert(0, "id", test["id"].to_numpy())
        write_test_predictions(test_pred_df)

        submission = sample_sub.copy()
        submission["class"] = classes[np.argmax(test_proba, axis=1)]
        write_submission(submission)

        # Optional: keep probability artifact consistent with labels for downstream checks
        log_stage(
            f"submission_shape={submission.shape}, test_predictions_shape={test_pred_df.shape}, oof_predictions_shape={oof_df.shape}"
        )


if __name__ == "__main__":
    main()
