import numpy as np
import pandas as pd

from scipy import sparse
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from aide_solution_helpers import (
    load_competition_data,
    write_submission,
    write_oof_predictions,
    write_test_predictions,
    aide_stage,
    log_stage,
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Angular periodicity for 0-360 using trig basis (sin/cos), then splined downstream.
    alpha_rad = np.deg2rad(out["alpha"].to_numpy(dtype=np.float32))
    out["alpha_sin"] = np.sin(alpha_rad)
    out["alpha_cos"] = np.cos(alpha_rad)

    out["u_g"] = out["u"] - out["g"]
    out["g_r"] = out["g"] - out["r"]
    out["r_i"] = out["r"] - out["i"]
    out["i_z"] = out["i"] - out["z"]
    out["u_r"] = out["u"] - out["r"]
    out["g_i"] = out["g"] - out["i"]
    out["r_z"] = out["r"] - out["z"]
    out["u_z"] = out["u"] - out["z"]
    return out


def _make_spline() -> SplineTransformer:
    kwargs = dict(
        degree=3,
        n_knots=6,
        include_bias=False,
        extrapolation="constant",
    )
    try:
        return SplineTransformer(**kwargs, knots="quantile", sparse_output=True)
    except TypeError:
        return SplineTransformer(**kwargs, knots="quantile", sparse=True)


def _to_sparse(x):
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(x)


class SplineFeatureBuilder:
    def __init__(
        self, spline_columns, cat_columns, interaction_pairs, interaction_rank=3
    ):
        self.spline_columns = spline_columns
        self.cat_columns = cat_columns
        self.interaction_pairs = interaction_pairs
        self.interaction_rank = interaction_rank
        self.spline_ = {}
        self.selector_ = None
        self.scaler_ = None
        self.ohe_ = None

    def _fit_splines(self, X):
        for col in self.spline_columns:
            st = _make_spline()
            st.fit(X[[col]].to_numpy(dtype=np.float32))
            self.spline_[col] = st

    def _fit_cats(self, X):
        try:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse=True)

        self.ohe_.fit(X[self.cat_columns].astype(str))

    def _build_numeric_blocks(self, X):
        blocks = []
        transformed = {}

        for col in self.spline_columns:
            Xt = self.spline_[col].transform(X[[col]].to_numpy(dtype=np.float32))
            transformed[col] = _to_sparse(Xt)
            blocks.append(transformed[col])

        for left, right in self.interaction_pairs:
            A = transformed[left]
            B = transformed[right]

            ka = min(self.interaction_rank, A.shape[1])
            kb = min(self.interaction_rank, B.shape[1])

            for ia in range(ka):
                a_col = A[:, ia]
                for ib in range(kb):
                    b_col = B[:, ib]
                    blocks.append(_to_sparse(a_col.multiply(b_col)))

        return blocks

    def fit_transform(self, X):
        X = X.copy()
        self._fit_splines(X)
        self._fit_cats(X)

        numeric_blocks = self._build_numeric_blocks(X)
        X_num = hstack(numeric_blocks, format="csr")
        X_cat = self.ohe_.transform(X[self.cat_columns].astype(str))
        X_all = hstack([X_num, X_cat], format="csr")

        self.selector_ = VarianceThreshold(threshold=1e-10)
        X_sel = self.selector_.fit_transform(X_all)

        self.scaler_ = StandardScaler(with_mean=False)
        X_scaled = self.scaler_.fit_transform(X_sel)
        return X_scaled

    def transform(self, X):
        check_is_fitted(self, ["spline_", "ohe_", "selector_", "scaler_"])
        numeric_blocks = self._build_numeric_blocks(X)
        X_num = hstack(numeric_blocks, format="csr")
        X_cat = self.ohe_.transform(X[self.cat_columns].astype(str))
        X_all = hstack([X_num, X_cat], format="csr")
        X_sel = self.selector_.transform(X_all)
        X_scaled = self.scaler_.transform(X_sel)
        return X_scaled


def main():
    train, test, sample_sub = load_competition_data()

    with aide_stage("build_features_stage"):
        train_feat = build_features(train)
        test_feat = build_features(test)

        cat_columns = ["spectral_type", "galaxy_population"]

        spline_columns = [
            "alpha_sin",
            "alpha_cos",
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
            "g_i",
            "r_z",
            "u_z",
        ]

        # Optional compact tensor-product-like spline interactions.
        interaction_pairs = [
            ("redshift", "u_g"),
            ("redshift", "g_r"),
            ("redshift", "r_i"),
            ("g_r", "r_i"),
        ]

        X = train_feat.drop(columns=["class", "id"], errors="ignore")
        y = train_feat["class"].to_numpy()
        X_test = test_feat.drop(columns=["id"], errors="ignore")

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        class_names = le.classes_

    with aide_stage("make_folds_stage"):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(skf.split(X, y_enc))

    oof_proba = np.zeros((len(X), len(class_names)), dtype=np.float32)

    with aide_stage("fit_predict_fold_stage"):
        for fold_idx, (train_idx, valid_idx) in enumerate(folds, start=1):
            log_stage(
                f"fit_predict_fold_stage: fold={fold_idx} model=QuantileSplineLogistic"
            )
            X_tr = X.iloc[train_idx].reset_index(drop=True)
            y_tr = y_enc[train_idx]
            X_va = X.iloc[valid_idx].reset_index(drop=True)
            y_va = y_enc[valid_idx]

            builder = SplineFeatureBuilder(
                spline_columns=spline_columns,
                cat_columns=cat_columns,
                interaction_pairs=interaction_pairs,
                interaction_rank=3,
            )

            X_tr_t = builder.fit_transform(X_tr)
            X_va_t = builder.transform(X_va)

            model = LogisticRegression(
                class_weight="balanced",
                solver="saga",
                multi_class="multinomial",
                max_iter=2500,
                n_jobs=-1,
            )

            model.fit(X_tr_t, y_tr)
            va_proba = model.predict_proba(X_va_t)
            oof_proba[valid_idx] = va_proba

    with aide_stage("score_stage"):
        oof_pred = oof_proba.argmax(axis=1)
        oof_true = le.inverse_transform(y_enc)
        oof_pred_lbl = le.inverse_transform(oof_pred)

        oof_balanced_accuracy = balanced_accuracy_score(oof_true, oof_pred_lbl)
        print(f"OOF Balanced Accuracy: {oof_balanced_accuracy:.6f}", flush=True)

        # Optional fold-level metric checks could be added by computing inside loop if needed.

    with aide_stage("fit_predict_fold_stage"):
        # Refit on full training data for final predictions.
        log_stage("fit_predict_fold_stage: final fit QuantileSplineLogistic")
        final_builder = SplineFeatureBuilder(
            spline_columns=spline_columns,
            cat_columns=cat_columns,
            interaction_pairs=interaction_pairs,
            interaction_rank=3,
        )
        X_full_t = final_builder.fit_transform(X)
        final_model = LogisticRegression(
            class_weight="balanced",
            solver="saga",
            multi_class="multinomial",
            max_iter=2500,
            n_jobs=-1,
        )
        final_model.fit(X_full_t, y_enc)

        X_test_t = final_builder.transform(X_test)
        test_proba = final_model.predict_proba(X_test_t)

    with aide_stage("write_outputs_stage"):
        # Required leakage-free artifacts.
        oof_df = pd.DataFrame(
            {
                "row": np.arange(len(train)),
                "target": le.inverse_transform(y_enc),
                "prediction": le.inverse_transform(oof_pred),
            }
        )
        write_oof_predictions(oof_df)

        test_prob_cols = [f"{c}_prob" for c in class_names]
        test_pred_probs_df = pd.DataFrame(test_proba, columns=test_prob_cols)
        test_pred_probs_df.insert(0, "id", test["id"].to_numpy())
        write_test_predictions(test_pred_probs_df)

        test_preds = le.inverse_transform(test_proba.argmax(axis=1))
        submission = sample_sub.copy()
        submission["class"] = test_preds
        write_submission(submission)


if __name__ == "__main__":
    main()
