import os
import gc
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

TARGET = "PitNextLap"
ID_COL = "id"
SEED = 2026
N_SPLITS = 5
MAX_LEAF_TREES = 128
HASH_FEATURES = 4096


def sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def rank01(a):
    return pd.Series(a).rank(method="average").to_numpy(dtype=np.float64) / (
        len(a) + 1.0
    )


def make_ordinal_encoder():
    try:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
            dtype=np.float32,
        )
    except TypeError:
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def hashed_leaf_csr(leaves, n_features=HASH_FEATURES):
    leaves = np.asarray(leaves)
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    n, t = leaves.shape
    tree_offsets = (np.arange(t, dtype=np.int64) * 1000003 + 17) % n_features
    idx = (leaves.astype(np.int64, copy=False) * 9176 + tree_offsets) % n_features
    indptr = np.arange(0, n * t + 1, t, dtype=np.int64)
    data = np.full(n * t, 1.0 / np.sqrt(max(t, 1)), dtype=np.float32)
    return sparse.csr_matrix(
        (data, idx.ravel().astype(np.int32, copy=False), indptr),
        shape=(n, n_features),
    )


class LeafMLP:
    def __init__(
        self,
        n_features=HASH_FEATURES,
        hidden=48,
        lr=0.003,
        epochs=7,
        batch_size=4096,
        l2=1e-5,
        patience=2,
        seed=SEED,
    ):
        self.n_features = n_features
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.patience = patience
        self.seed = seed
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def _init_weights(self):
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(
            0, np.sqrt(2.0 / self.n_features), (self.n_features, self.hidden)
        ).astype(np.float32)
        self.b1 = np.zeros(self.hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden), self.hidden).astype(
            np.float32
        )
        self.b2 = np.zeros(1, dtype=np.float32)

    def _adam_update(self, param, grad, m, v, step):
        grad = grad.astype(np.float32, copy=False)
        m *= self.beta1
        m += (1.0 - self.beta1) * grad
        v *= self.beta2
        v += (1.0 - self.beta2) * (grad * grad)
        m_hat = m / (1.0 - self.beta1**step)
        v_hat = v / (1.0 - self.beta2**step)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def fit(self, leaves_train, y_train, leaves_val, y_val):
        self._init_weights()
        y_train = y_train.astype(np.float32)
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        pos_weight = np.float32(np.clip(neg / max(pos, 1.0), 1.0, 25.0))

        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)

        rng = np.random.default_rng(self.seed)
        best_auc = -np.inf
        best_state = None
        stale = 0
        step = 0

        for epoch in range(self.epochs):
            order = rng.permutation(len(y_train))
            for start in range(0, len(order), self.batch_size):
                step += 1
                idx = order[start : start + self.batch_size]
                Xb = hashed_leaf_csr(leaves_train[idx], self.n_features)
                yb = y_train[idx]

                z = Xb @ self.W1 + self.b1
                h = np.maximum(z, 0.0)
                p = sigmoid(h @ self.W2 + self.b2[0]).astype(np.float32)

                sw = np.where(yb > 0, pos_weight, 1.0).astype(np.float32)
                err = (p - yb) * sw
                denom = max(float(sw.sum()), 1.0)

                gW2 = (h.T @ err) / denom + self.l2 * self.W2
                gb2 = np.array([err.sum() / denom], dtype=np.float32)

                gz = (err[:, None] * self.W2[None, :]) * (z > 0.0)
                gz /= denom
                gW1 = Xb.T @ gz + self.l2 * self.W1
                gb1 = gz.sum(axis=0)

                self._adam_update(self.W1, np.asarray(gW1), mW1, vW1, step)
                self._adam_update(self.b1, np.asarray(gb1), mb1, vb1, step)
                self._adam_update(self.W2, np.asarray(gW2), mW2, vW2, step)
                self._adam_update(self.b2, gb2, mb2, vb2, step)

            val_pred = self.predict_proba(leaves_val)
            auc = roc_auc_score(y_val, val_pred)
            print(f"    leaf_mlp epoch {epoch + 1}: val_auc={auc:.6f}")

            if auc > best_auc + 1e-5:
                best_auc = auc
                best_state = (
                    self.W1.copy(),
                    self.b1.copy(),
                    self.W2.copy(),
                    self.b2.copy(),
                )
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.W1, self.b1, self.W2, self.b2 = best_state
        return self

    def predict_proba(self, leaves, batch_size=16384):
        preds = np.empty(leaves.shape[0], dtype=np.float32)
        for start in range(0, leaves.shape[0], batch_size):
            end = min(start + batch_size, leaves.shape[0])
            Xb = hashed_leaf_csr(leaves[start:end], self.n_features)
            z = Xb @ self.W1 + self.b1
            h = np.maximum(z, 0.0)
            preds[start:end] = sigmoid(h @ self.W2 + self.b2[0]).astype(np.float32)
        return preds


def lgb_leaf_matrix(model, X, max_trees=MAX_LEAF_TREES):
    best_iter = getattr(model, "best_iteration_", None)
    if best_iter is None or best_iter <= 0:
        best_iter = getattr(model, "n_estimators", max_trees)
    num_iter = min(int(best_iter), max_trees)
    leaves = model.booster_.predict(X, pred_leaf=True, num_iteration=num_iter)
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    return leaves.astype(np.int16, copy=False)


def main():
    input_dir = "./input"
    out_dir = "./working"
    os.makedirs(out_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(input_dir, "train.csv.gz"))
    test = pd.read_csv(os.path.join(input_dir, "test.csv.gz"))
    sample = pd.read_csv(os.path.join(input_dir, "sample_submission.csv.gz"))

    y = train[TARGET].astype(np.int8).to_numpy()
    feature_cols = [c for c in train.columns if c not in (TARGET, ID_COL)]
    X_raw = train[feature_cols].copy()
    T_raw = test[feature_cols].copy()

    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for c in cat_cols:
        X_raw[c] = X_raw[c].fillna("__NA__").astype(str)
        T_raw[c] = T_raw[c].fillna("__NA__").astype(str)

    for c in num_cols:
        med = pd.concat([X_raw[c], T_raw[c]], ignore_index=True).median()
        X_raw[c] = (
            pd.to_numeric(X_raw[c], errors="coerce").fillna(med).astype(np.float32)
        )
        T_raw[c] = (
            pd.to_numeric(T_raw[c], errors="coerce").fillna(med).astype(np.float32)
        )

    X_lgb, T_lgb = X_raw.copy(), T_raw.copy()
    for c in cat_cols:
        cats = pd.Index(pd.concat([X_lgb[c], T_lgb[c]], ignore_index=True).unique())
        X_lgb[c] = pd.Categorical(X_lgb[c], categories=cats)
        T_lgb[c] = pd.Categorical(T_lgb[c], categories=cats)

    X_xgb, T_xgb = X_raw.copy(), T_raw.copy()
    if cat_cols:
        enc = make_ordinal_encoder()
        enc.fit(pd.concat([X_xgb[cat_cols], T_xgb[cat_cols]], ignore_index=True))
        tr_cat = enc.transform(X_xgb[cat_cols]).astype(np.float32)
        te_cat = enc.transform(T_xgb[cat_cols]).astype(np.float32)
        for j, c in enumerate(cat_cols):
            X_xgb[c] = tr_cat[:, j]
            T_xgb[c] = te_cat[:, j]
    X_xgb = X_xgb.astype(np.float32)
    T_xgb = T_xgb.astype(np.float32)

    X_cat, T_cat = X_raw.copy(), T_raw.copy()
    cat_idx = [X_cat.columns.get_loc(c) for c in cat_cols]
    threads = max(1, os.cpu_count() or 1)

    member_names = ["lgb", "xgb", "cat", "leaf_mlp"]
    oof = {m: np.zeros(len(train), dtype=np.float32) for m in member_names}
    test_pred = {m: np.zeros(len(test), dtype=np.float32) for m in member_names}

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y), 1):
        print(f"\nFold {fold}/{N_SPLITS}")
        y_tr, y_va = y[tr_idx], y[va_idx]
        pos_weight = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1))

        lgb_model = LGBMClassifier(
            objective="binary",
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=63,
            min_child_samples=80,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=SEED + fold,
            n_jobs=threads,
            verbosity=-1,
        )
        fit_kwargs = {}
        if cat_cols:
            fit_kwargs["categorical_feature"] = cat_cols
        lgb_model.fit(
            X_lgb.iloc[tr_idx],
            y_tr,
            eval_set=[(X_lgb.iloc[va_idx], y_va)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            **fit_kwargs,
        )
        oof["lgb"][va_idx] = lgb_model.predict_proba(X_lgb.iloc[va_idx])[:, 1]
        test_pred["lgb"] += lgb_model.predict_proba(T_lgb)[:, 1] / N_SPLITS

        leaves_tr = lgb_leaf_matrix(lgb_model, X_lgb.iloc[tr_idx])
        leaves_va = lgb_leaf_matrix(lgb_model, X_lgb.iloc[va_idx])
        leaves_te = lgb_leaf_matrix(lgb_model, T_lgb)

        leaf_mlp = LeafMLP(seed=SEED + 100 * fold)
        leaf_mlp.fit(leaves_tr, y_tr, leaves_va, y_va)
        oof["leaf_mlp"][va_idx] = leaf_mlp.predict_proba(leaves_va)
        test_pred["leaf_mlp"] += leaf_mlp.predict_proba(leaves_te) / N_SPLITS

        del leaves_tr, leaves_va, leaves_te, leaf_mlp
        gc.collect()

        xgb_model = XGBClassifier(
            n_estimators=420,
            max_depth=5,
            learning_rate=0.04,
            min_child_weight=50,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            max_bin=256,
            scale_pos_weight=pos_weight,
            random_state=SEED + fold,
            n_jobs=threads,
        )
        xgb_model.fit(
            X_xgb.iloc[tr_idx],
            y_tr,
            eval_set=[(X_xgb.iloc[va_idx], y_va)],
            verbose=False,
        )
        oof["xgb"][va_idx] = xgb_model.predict_proba(X_xgb.iloc[va_idx])[:, 1]
        test_pred["xgb"] += xgb_model.predict_proba(T_xgb)[:, 1] / N_SPLITS

        cat_model = CatBoostClassifier(
            iterations=550,
            learning_rate=0.045,
            depth=6,
            l2_leaf_reg=6.0,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            bootstrap_type="Bernoulli",
            subsample=0.85,
            rsm=0.85,
            random_seed=SEED + fold,
            thread_count=threads,
            od_type="Iter",
            od_wait=80,
            allow_writing_files=False,
            verbose=False,
        )
        cat_model.fit(
            X_cat.iloc[tr_idx],
            y_tr,
            cat_features=cat_idx if cat_idx else None,
            eval_set=(X_cat.iloc[va_idx], y_va),
            use_best_model=True,
            verbose=False,
        )
        oof["cat"][va_idx] = cat_model.predict_proba(X_cat.iloc[va_idx])[:, 1]
        test_pred["cat"] += cat_model.predict_proba(T_cat)[:, 1] / N_SPLITS

        fold_auc = {m: roc_auc_score(y_va, oof[m][va_idx]) for m in member_names}
        print("  fold_auc:", {k: round(v, 6) for k, v in fold_auc.items()})

        del lgb_model, xgb_model, cat_model
        gc.collect()

    member_auc = {m: float(roc_auc_score(y, oof[m])) for m in member_names}
    blend_oof = np.mean([rank01(oof[m]) for m in member_names], axis=0)
    blend_test = np.mean([rank01(test_pred[m]) for m in member_names], axis=0)
    blend_test = np.clip(blend_test, 1e-6, 1 - 1e-6)

    cv_auc = float(roc_auc_score(y, blend_oof))
    print(f"\n5-fold OOF ROC AUC: {cv_auc:.6f}")
    print("Member OOF AUC:", {k: round(v, 6) for k, v in member_auc.items()})

    sub_target = TARGET if TARGET in sample.columns else sample.columns[1]
    submission = sample.copy()
    submission[sub_target] = blend_test
    submission.to_csv(os.path.join(out_dir, "submission.csv"), index=False)

    pd.DataFrame(
        {
            "row": np.arange(len(train), dtype=np.int32),
            "target": y.astype(np.int8),
            "prediction": blend_oof.astype(np.float32),
        }
    ).to_csv(
        os.path.join(out_dir, "oof_predictions.csv.gz"), index=False, compression="gzip"
    )

    test_predictions = sample.copy()
    test_predictions[sub_target] = blend_test.astype(np.float32)
    test_predictions.to_csv(
        os.path.join(out_dir, "test_predictions.csv.gz"),
        index=False,
        compression="gzip",
    )

    report = {
        "research_hypotheses_llm_claimed_used": ["001046"],
        "metric": "roc_auc",
        "cv_auc": cv_auc,
        "member_auc": member_auc,
        "blend": "equal_rank_average_lgb_xgb_cat_leaf_mlp",
        "files": {
            "submission": "./working/submission.csv",
            "oof_predictions": "./working/oof_predictions.csv.gz",
            "test_predictions": "./working/test_predictions.csv.gz",
        },
    }
    with open(os.path.join(out_dir, "result_review.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
