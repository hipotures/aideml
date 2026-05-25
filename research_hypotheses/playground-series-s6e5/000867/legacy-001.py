import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

INPUT = "./input"
WORKING = "./working"
os.makedirs(WORKING, exist_ok=True)

train = pd.read_csv(f"{INPUT}/train.csv.gz")
test = pd.read_csv(f"{INPUT}/test.csv.gz")
sample = pd.read_csv(f"{INPUT}/sample_submission.csv.gz")

target_col = "PitNextLap"
id_col = "id"
y = train[target_col].astype(int).values

features = [c for c in train.columns if c not in [target_col, id_col]]
cat_cols = [c for c in features if train[c].dtype == "object"]
num_cols = [c for c in features if c not in cat_cols]

X = train[features].copy()
X_test = test[features].copy()

# Add compact domain features without using future rows.
for df in (X, X_test):
    df["laps_left_est"] = (
        df["LapNumber"] / np.maximum(df["RaceProgress"], 1e-3) - df["LapNumber"]
    ).clip(-5, 100)
    df["tyre_life_x_progress"] = df["TyreLife"] * df["RaceProgress"]
    df["degradation_per_lap"] = df["Cumulative_Degradation"] / np.maximum(
        df["TyreLife"], 1
    )
    df["is_current_pitlap"] = df["PitStop"].astype(int)

features = X.columns.tolist()
cat_cols = [c for c in features if X[c].dtype == "object"]
num_cols = [c for c in features if c not in cat_cols]

pre = ColumnTransformer(
    [
        ("num", SimpleImputer(strategy="median"), num_cols),
        (
            "cat",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
            cat_cols,
        ),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

experts = {
    "hgb": HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.055,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    ),
    "extra_trees": ExtraTreesClassifier(
        n_estimators=260,
        min_samples_leaf=25,
        max_features=0.75,
        n_jobs=-1,
        random_state=43,
        class_weight="balanced_subsample",
    ),
    "logit": LogisticRegression(
        C=0.8, max_iter=700, solver="lbfgs", class_weight="balanced"
    ),
}


def rank01(a):
    s = pd.Series(a)
    return s.rank(method="average").to_numpy() / (len(s) + 1.0)


def eval_model_cv(model, splitter, split_args):
    scores = []
    for tr_idx, va_idx in splitter.split(*split_args):
        pipe = make_pipeline(pre, model)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        pred = pipe.predict_proba(X.iloc[va_idx])[:, 1]
        scores.append(roc_auc_score(y[va_idx], pred))
    return np.array(scores)


groups = train["Year"].astype(str) + "_" + train["Race"].astype(str)
gkf = GroupKFold(n_splits=5)

order_cols = ["Year", "id"] if "Year" in train.columns else [id_col]
ordered_idx = train.sort_values(order_cols).index.to_numpy()
tscv = TimeSeriesSplit(n_splits=5)
forward_splits = [
    (ordered_idx[tr], ordered_idx[va]) for tr, va in tscv.split(ordered_idx)
]


class FixedSplit:
    def __init__(self, splits):
        self.splits = splits

    def split(self, X=None, y=None, groups=None):
        yield from self.splits


selected = []
diagnostics = {}

for name, model in experts.items():
    group_scores = eval_model_cv(model, gkf, (X, y, groups))
    forward_scores = eval_model_cv(model, FixedSplit(forward_splits), (X, y, None))
    mean_both = min(group_scores.mean(), forward_scores.mean())
    stability_penalty = (
        group_scores.std()
        + forward_scores.std()
        + abs(group_scores.mean() - forward_scores.mean())
    )
    stability_weight = max(1e-6, mean_both - 0.5) / (1.0 + 4.0 * stability_penalty)

    diagnostics[name] = {
        "group_auc_mean": float(group_scores.mean()),
        "group_auc_std": float(group_scores.std()),
        "forward_auc_mean": float(forward_scores.mean()),
        "forward_auc_std": float(forward_scores.std()),
        "stability_weight": float(stability_weight),
    }

    if (
        group_scores.mean() >= 0.60
        and forward_scores.mean() >= 0.60
        and stability_weight > 0
    ):
        selected.append((name, model, stability_weight))

if not selected:
    best = max(diagnostics, key=lambda k: diagnostics[k]["stability_weight"])
    selected = [(best, experts[best], diagnostics[best]["stability_weight"])]

weights = np.array([w for _, _, w in selected], dtype=float)
weights = weights / weights.sum()

oof_expert = np.zeros((len(train), len(selected)))
test_expert = np.zeros((len(test), len(selected)))

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
    for j, (name, model, _) in enumerate(selected):
        pipe = make_pipeline(pre, model)
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof_expert[va_idx, j] = pipe.predict_proba(X.iloc[va_idx])[:, 1]

for j, (name, model, _) in enumerate(selected):
    pipe = make_pipeline(pre, model)
    pipe.fit(X, y)
    test_expert[:, j] = pipe.predict_proba(X_test)[:, 1]

oof_rank = np.column_stack(
    [rank01(oof_expert[:, j]) for j in range(oof_expert.shape[1])]
)
test_rank = np.column_stack(
    [rank01(test_expert[:, j]) for j in range(test_expert.shape[1])]
)

oof_pred = np.average(oof_rank, axis=1, weights=weights)
test_pred = np.average(test_rank, axis=1, weights=weights)

cv_auc = roc_auc_score(y, oof_pred)
print(f"Stability-weighted grouped 5-fold OOF ROC AUC: {cv_auc:.6f}")
print("Selected experts:", ", ".join([n for n, _, _ in selected]))
print(
    json.dumps(
        {
            "research_hypotheses_llm_claimed_used": ["000867"],
            "expert_diagnostics": diagnostics,
            "selected_experts": [n for n, _, _ in selected],
            "metric": "roc_auc",
            "grouped_oof_auc": float(cv_auc),
        },
        indent=2,
    )
)

pd.DataFrame(
    {
        "row": np.arange(len(train)),
        "target": y,
        "prediction": oof_pred,
    }
).to_csv(f"{WORKING}/oof_predictions.csv.gz", index=False, compression="gzip")

pd.DataFrame(
    {
        id_col: sample[id_col].values,
        target_col: test_pred,
    }
).to_csv(f"{WORKING}/test_predictions.csv.gz", index=False, compression="gzip")

submission = sample.copy()
submission[target_col] = test_pred
submission.to_csv(f"{WORKING}/submission.csv", index=False)
