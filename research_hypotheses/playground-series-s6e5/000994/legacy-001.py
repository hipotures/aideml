import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
os.makedirs("./working", exist_ok=True)

RANDOM_STATE = 42
TARGET = "PitNextLap"
ID_COL = "id"
GROUP_COL = "Race"
HYPOTHESES = ["000994"]

train = pd.read_csv("./input/train.csv.gz")
test = pd.read_csv("./input/test.csv.gz")
sample = pd.read_csv("./input/sample_submission.csv.gz")

y = train[TARGET].astype(int).values
features = [c for c in train.columns if c not in [TARGET, ID_COL]]
cat_cols = [c for c in features if train[c].dtype == "object"]

train_x = train[features].copy()
test_x = test[features].copy()

if cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    all_cats = pd.concat([train_x[cat_cols], test_x[cat_cols]], axis=0)
    enc.fit(all_cats)
    train_x[cat_cols] = enc.transform(train_x[cat_cols]).astype(np.int32)
    test_x[cat_cols] = enc.transform(test_x[cat_cols]).astype(np.int32)

for c in features:
    if c not in cat_cols:
        train_x[c] = train_x[c].astype(np.float32)
        test_x[c] = test_x[c].astype(np.float32)

groups = train[GROUP_COL].astype(str).values


def rank01(a):
    s = pd.Series(a)
    return s.rank(method="average").values / (len(s) + 1.0)


def clip_prob(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


class BetaCalibrator:
    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def fit(self, p, y):
        p = clip_prob(p)
        x = np.column_stack([np.log(p), np.log1p(-p)])
        self.model.fit(x, y)
        return self

    def predict(self, p):
        p = clip_prob(p)
        x = np.column_stack([np.log(p), np.log1p(-p)])
        return self.model.predict_proba(x)[:, 1]


class VennAbersLite:
    def __init__(self):
        self.iso0 = IsotonicRegression(out_of_bounds="clip")
        self.iso1 = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p, y):
        p = np.asarray(p, dtype=float)
        y = np.asarray(y, dtype=int)
        self.iso0.fit(np.r_[p, p], np.r_[y, np.zeros_like(y)])
        self.iso1.fit(np.r_[p, p], np.r_[y, np.ones_like(y)])
        return self

    def predict(self, p):
        p0 = clip_prob(self.iso0.predict(np.asarray(p, dtype=float)))
        p1 = clip_prob(self.iso1.predict(np.asarray(p, dtype=float)))
        return clip_prob(p1 / (1.0 - p0 + p1))


def fit_calibrator(kind, p, y_cal):
    if kind == "raw":
        return None
    if kind == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(p, y_cal)
    if kind == "beta":
        return BetaCalibrator().fit(p, y_cal)
    if kind == "venn_abers":
        return VennAbersLite().fit(p, y_cal)
    raise ValueError(kind)


def apply_calibrator(cal, kind, p):
    if kind == "raw":
        return clip_prob(p)
    if kind == "isotonic":
        return clip_prob(cal.predict(p))
    return clip_prob(cal.predict(p))


model_params = dict(
    objective="binary",
    n_estimators=900,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=80,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,
    reg_lambda=1.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

score_sources = ["raw_score", "rank_score"]
calibration_kinds = ["raw", "isotonic", "beta", "venn_abers"]
cv_scores = {f"{src}_{kind}": [] for src in score_sources for kind in calibration_kinds}
oof = np.zeros(len(train), dtype=float)

gkf = GroupKFold(n_splits=5)

for fold, (dev_idx, val_idx) in enumerate(gkf.split(train_x, y, groups), 1):
    dev_groups = groups[dev_idx]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.18, random_state=RANDOM_STATE + fold
    )
    fit_rel, cal_rel = next(
        splitter.split(train_x.iloc[dev_idx], y[dev_idx], dev_groups)
    )
    fit_idx = dev_idx[fit_rel]
    cal_idx = dev_idx[cal_rel]

    model = LGBMClassifier(**model_params)
    model.fit(
        train_x.iloc[fit_idx],
        y[fit_idx],
        eval_set=[(train_x.iloc[cal_idx], y[cal_idx])],
        eval_metric="auc",
        callbacks=[],
    )

    cal_raw = model.predict_proba(train_x.iloc[cal_idx])[:, 1]
    val_raw = model.predict_proba(train_x.iloc[val_idx])[:, 1]
    cal_scores = {"raw_score": cal_raw, "rank_score": rank01(cal_raw)}
    val_scores = {"raw_score": val_raw, "rank_score": rank01(val_raw)}

    fold_best_auc = -1
    fold_best_pred = None

    for src in score_sources:
        for kind in calibration_kinds:
            key = f"{src}_{kind}"
            cal = fit_calibrator(kind, cal_scores[src], y[cal_idx])
            pred = apply_calibrator(cal, kind, val_scores[src])
            auc = roc_auc_score(y[val_idx], pred)
            cv_scores[key].append(auc)
            if auc > fold_best_auc:
                fold_best_auc = auc
                fold_best_pred = pred

    oof[val_idx] = fold_best_pred
    print(f"fold {fold}: best untouched grouped validation AUC = {fold_best_auc:.6f}")

mean_scores = {k: float(np.mean(v)) for k, v in cv_scores.items()}
best_key = max(mean_scores, key=mean_scores.get)
best_src, best_kind = best_key.rsplit("_", 1)
if best_key.endswith("venn_abers"):
    best_src = best_key.replace("_venn_abers", "")
    best_kind = "venn_abers"

overall_oof_auc = roc_auc_score(y, oof)
print(f"selected final mapping: {best_key}")
print(f"OOF AUC using per-fold best calibrated mapping: {overall_oof_auc:.6f}")
print("calibration comparison:", json.dumps(mean_scores, sort_keys=True))

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": clip_prob(oof),
    }
).to_csv("./working/oof_predictions.csv.gz", index=False, compression="gzip")

full_splitter = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=RANDOM_STATE)
fit_idx, cal_idx = next(full_splitter.split(train_x, y, groups))

final_model = LGBMClassifier(**model_params)
final_model.fit(
    train_x.iloc[fit_idx],
    y[fit_idx],
    eval_set=[(train_x.iloc[cal_idx], y[cal_idx])],
    eval_metric="auc",
    callbacks=[],
)

cal_raw = final_model.predict_proba(train_x.iloc[cal_idx])[:, 1]
test_raw = final_model.predict_proba(test_x)[:, 1]

if best_src == "rank_score":
    final_cal_score = rank01(cal_raw)
    final_test_score = rank01(test_raw)
else:
    final_cal_score = cal_raw
    final_test_score = test_raw

final_calibrator = fit_calibrator(best_kind, final_cal_score, y[cal_idx])
test_pred = apply_calibrator(final_calibrator, best_kind, final_test_score)

submission = sample[[ID_COL]].copy()
submission[TARGET] = clip_prob(test_pred)
submission.to_csv("./working/submission.csv", index=False)

test_predictions = submission.copy()
test_predictions.to_csv(
    "./working/test_predictions.csv.gz", index=False, compression="gzip"
)

result = {
    "metric": "roc_auc",
    "validation_auc": float(overall_oof_auc),
    "selected_calibration": best_key,
    "research_hypotheses_llm_claimed_used": HYPOTHESES,
    "submission_path": "./working/submission.csv",
    "oof_path": "./working/oof_predictions.csv.gz",
    "test_predictions_path": "./working/test_predictions.csv.gz",
}
print(json.dumps(result, indent=2, sort_keys=True))
