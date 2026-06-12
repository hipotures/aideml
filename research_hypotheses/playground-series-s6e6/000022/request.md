You are a research scientist and Kaggle competition strategist. Your job is to
investigate this machine learning problem using live web search, compare public
techniques and adjacent competition patterns, and propose concise, testable
hypotheses for an automated ML experiment system. The system will later turn
one selected hypothesis into Python solution code, execute it, score it, and
store the result. Do not write a full solution script. Return only structured
JSON matching the provided schema.

# Research task
Generate initial hypotheses for feature search. Use live web search to identify
feature engineering ideas, preprocessing directions, data representations,
validation traps, and simple baseline algorithms that are relevant to this
competition or closely related machine learning problems.

# Initial hypothesis policy
An initial hypothesis is a self-contained first solution direction focused on a
distinct feature/preprocessing/data-representation family. The main
experimental variable must be the feature strategy, not advanced model tuning.
Use a simple, consistent baseline model panel only as a measuring instrument
for the feature family and for observing which basic algorithm families fit
the engineered features best. Describe the panel by simple, diverse model
families, not as a fixed magic list of model names to copy every time. Include
concrete model names only when they are clearly appropriate for the task and
expected runtime. Do not make each hypothesis depend on a different arbitrary
algorithm subset unless the feature family clearly requires it.
The baseline model panel should normally be reused across initial hypotheses
unless the feature family clearly requires a different estimator. Hypothesis
novelty must come from the feature/preprocessing/data-representation strategy,
not from changing panel composition.

Do not propose heavy ensembling, stacking, calibration pipelines,
hyperparameter search, seed search, or advanced model-specific tricks in
initial hypotheses. Those belong to later algorithm/ensemble/tuning phases
after initial feature-search hypotheses have produced scores.

# Novelty dimensions
A new initial feature-search hypothesis is materially different only if at
least one of these dimensions changes:
- feature representation family
- preprocessing mechanism
- data source usage
- fold-dependent or statistical feature mechanism
- physical or domain-specific representation
- dimensionality-reduction or embedding method
- missingness or outlier treatment

Changing only the model panel, hyperparameters, parameter names, bin count,
threshold values, seeds, wording, or evaluation wrapper is not novel. Changing
the binning mechanism or fold-safe statistical mechanism can be novel; changing
only the number of bins is not.

# Output contract
Return exactly 1 concise new initial feature-search
hypotheses. Do not target a specific previous node or code block. Use the
prior results only to avoid repeating approaches that have already been tried
and, when scores are available, as evidence about which feature families and
simple baseline algorithms look promising. Treat previous hypothesis text only
as novelty and comparison context, not as execution evidence.
Do not force weak novelty. If the requested number of hypotheses must be
returned but only a weak or near-duplicate idea remains, still return the
hypothesis, set novelty_confidence to "low", and explicitly describe the
duplication or weak-novelty risk in risk.

The prompt may include previously stored hypotheses as short Title, Summary,
and Rationale text blocks. Use them only to understand what feature directions
have already been proposed and to avoid near-duplicates.

# Prior research history
If recent research summaries are included, use them as context for choosing a
distinct next direction. When score summaries are present, treat them as weak
evidence about which directions looked promising after that research
checkpoint.

# Context section meanings
The context below is plain text. It may include task details, data overview,
runtime options, previous hypotheses, recent research summaries, and examples
of working solution code. Use working solution examples only to understand what
has already been tried and what performed well or poorly.

# Required JSON output shape
Return JSON with: summary; hypotheses[].title; hypotheses[].summary;
hypotheses[].feature_family; hypotheses[].feature_strategy;
hypotheses[].baseline_model_panel; hypotheses[].model_panel_rationale;
hypotheses[].validation_strategy; hypotheses[].materialization_hint;
hypotheses[].expected_signal; hypotheses[].novelty_confidence;
hypotheses[].risk; hypotheses[].sources. The hypotheses array must contain
exactly 1 items.

# Field meanings
- title: short human-readable name for this initial hypothesis.
- summary: one-sentence UI/memory summary.
- feature_family: short stable label for the feature/preprocessing family,
  such as categorical_frequency_counts, numeric_ratios_logs,
  time_window_aggregations, group_statistics_fold_safe,
  auxiliary_data_features, missingness_outlier_features, or text_tfidf_features.
- feature_strategy: concrete plan for which features, transformations,
  encodings, imputations, reductions, or data representations this hypothesis
  should build. This is the main hypothesis.
- baseline_model_panel: simple, diverse model-family panel to evaluate the
  feature family. Keep it basic and comparable across initial hypotheses; avoid
  hardcoding the same few model names in every hypothesis unless the task
  clearly supports that panel. No stacking, heavy blending, or deep tuning.
- model_panel_rationale: why this simple panel is enough to measure the
  feature signal and compare basic algorithm fit.
- validation_strategy: general validation choice, for example 5-fold
  StratifiedKFold for classification unless task metadata clearly requires
  group/time-aware folds.
- materialization_hint: guidance for the later code-materialization prompt.
  Describe how to turn the hypothesis into a staged solution, but do not write
  code or repeat global artifact/cache contracts.
- expected_signal: what should be visible in CV, per-model diagnostics,
  runtime, or output logs if the feature family has value.
- novelty_confidence: one of high, medium, or low. Use low when the idea is
  weakly novel, partly redundant with previous hypotheses, or mainly a
  fallback because the system requires another hypothesis.
- risk: leakage, overfitting, runtime, data availability, or no-op risks.
  When novelty_confidence is low, risk must explicitly describe the overlap or
  duplication risk.
- sources: concise URLs or source names used for this idea; use an empty array
  when none are available.

# Current task and prior-result summary
## Task description
## Goal
Predict the stellar class for each object in the test set.
For each row in `test.csv`, predict the `class` label. The target column in
`train.csv` is `class`; the identifier column is `id`.
## Evaluation
Submissions are evaluated using balanced accuracy. Higher is better.
Competition-specific modeling hint: if using CatBoost for this multiclass task,
include `auto_class_weights="Balanced"` unless explicitly testing a different
class-weighting strategy; this has empirically improved local CV and public
leaderboard score for this competition.
Analogous balanced-class settings should be used for other multiclass tree
models unless explicitly testing a different class-weighting strategy: for
LightGBM use `class_weight="balanced"`, and for XGBoost pass fold-specific
`sample_weight=compute_sample_weight(class_weight="balanced", y=y_train)` to
`.fit()`.
The submission file must contain a header and exactly these columns:
```csv
id,class
577347,STAR
577348,GALAXY
577349,QSO
```
`class` must contain one of `GALAXY`, `STAR`, or `QSO`.
## Data description
- **train.csv** - training data with the multiclass target column `class`
- **test.csv** - test data without the target column
- **sample_submission.csv** - sample submission in the required format
Additional auxiliary data description for `star_classification.csv`:
Original SDSS17 Stellar Classification Dataset.
This is the original real-world dataset that inspired the synthetic Playground
Series S6E6 competition data. It can be used as raw auxiliary data, but it is
not automatically merged with train.csv or test.csv.
Common columns with the competition data:
alpha, delta, u, g, r, i, z, redshift, class.
Columns present in this original dataset but not in the competition files:
obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.
Competition columns not present in this original dataset:
id, spectral_type, galaxy_population.
Generated code should decide whether and how to use this file. Any merge,
filtering, cleaning of sentinel magnitudes, or column mapping must be done
explicitly by the generated solution code.

## Data overview
```
playground-series-s6e6.zip (61.4 MB)
sample_submission.csv (247436 lines)
sample_submission.csv.gz (247436 lines)
test.csv (247436 lines)
test.csv.gz (247436 lines)
train.csv (577348 lines)
train.csv.gz (577348 lines)
original_sdss17/
star_classification.csv (100001 lines)
star_classification.txt (18 lines)```
-> original_sdss17/star_classification.csv has 100000 rows and 18 columns.
Here is some information about the columns:
MJD (int64) has range: 51608.00 - 58932.00, 0 nan values
alpha (float64) has range: 0.01 - 360.00, 0 nan values
cam_col (int64) has 6 unique values: [2, 5, 3, 4, 6, 1]
class (object) has 3 unique values: ['GALAXY', 'QSO', 'STAR']
delta (float64) has range: -18.79 - 83.00, 0 nan values
fiber_ID (int64) has range: 1.00 - 1000.00, 0 nan values
field_ID (int64) has range: 11.00 - 989.00, 0 nan values
g (float64) has range: -9999.00 - 31.60, 0 nan values
i (float64) has range: 9.47 - 32.14, 0 nan values
obj_ID (float64) has range: 1237645942904389888.00 - 1237680531356386304.00, 0 nan values
plate (int64) has range: 266.00 - 12547.00, 0 nan values
r (float64) has range: 9.82 - 29.57, 0 nan values
redshift (float64) has range: -0.01 - 7.01, 0 nan values
rerun_ID (int64) has 1 unique values: [301]
run_ID (int64) has range: 109.00 - 8162.00, 0 nan values
spec_obj_ID (float64) has range: 299519089380976640.00 - 14126940609093851136.00, 0 nan values
u (float64) has range: -9999.00 - 32.78, 0 nan values
z (float64) has range: -9999.00 - 29.38, 0 nan values
-> original_sdss17/star_classification.txt has content:
Original SDSS17 Stellar Classification Dataset.
This is the original real-world dataset that inspired the synthetic Playground
Series S6E6 competition data. It can be used as raw auxiliary data, but it is
not automatically merged with train.csv or test.csv.
Common columns with the competition data:
alpha, delta, u, g, r, i, z, redshift, class.
Columns present in this original dataset but not in the competition files:
obj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.
Competition columns not present in this original dataset:
id, spectral_type, galaxy_population.
Generated code should decide whether and how to use this file. Any merge,
filtering, cleaning of sentinel magnitudes, or column mapping must be done
explicitly by the generated solution code.
-> sample_submission.csv has 247435 rows and 2 columns.
Here is some information about the columns:
class (object) has 1 unique values: ['GALAXY']
id (int64) has range: 577347.00 - 824781.00, 0 nan values
-> sample_submission.csv.gz has 247435 rows and 2 columns.
Here is some information about the columns:
class (object) has 1 unique values: ['GALAXY']
id (int64) has range: 577347.00 - 824781.00, 0 nan values
-> test.csv has 247435 rows and 11 columns.
Here is some information about the columns:
alpha (float64) has range: 0.01 - 360.00, 0 nan values
delta (float64) has range: -17.96 - 79.17, 0 nan values
g (float64) has range: 13.37 - 27.17, 0 nan values
galaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud']
i (float64) has range: 10.03 - 24.57, 0 nan values
id (int64) has range: 577347.00 - 824781.00, 0 nan values
r (float64) has range: 10.39 - 25.29, 0 nan values
redshift (float64) has range: -0.01 - 7.01, 0 nan values
spectral_type (object) has 4 unique values: ['G/K', 'M', 'O/B', 'A/F']
u (float64) has range: 13.90 - 27.84, 0 nan values
z (float64) has range: 10.63 - 25.70, 0 nan values
-> test.csv.gz has 247435 rows and 11 columns.
Here is some information about the columns:
alpha (float64) has range: 0.01 - 360.00, 0 nan values
delta (float64) has range: -17.96 - 79.17, 0 nan values
g (float64) has range: 13.37 - 27.17, 0 nan values
galaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud']
i (float64) has range: 10.03 - 24.57, 0 nan values
id (int64) has range: 577347.00 - 824781.00, 0 nan values
r (float64) has range: 10.39 - 25.29, 0 nan values
redshift (float64) has range: -0.01 - 7.01, 0 nan values
spectral_type (object) has 4 unique values: ['G/K', 'M', 'O/B', 'A/F']
u (float64) has range: 13.90 - 27.84, 0 nan values
z (float64) has range: 10.63 - 25.70, 0 nan values
-> train.csv has 577347 rows and 12 columns.
Here is some information about the columns:
alpha (float64) has range: 0.01 - 360.00, 0 nan values
class (object) has 3 unique values: ['GALAXY', 'QSO', 'STAR']
delta (float64) has range: -17.97 - 79.16, 0 nan values
g (float64) has range: 13.54 - 27.62, 0 nan values
galaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud']
i (float64) has range: 11.96 - 27.91, 0 nan values
id (int64) has range: 0.00 - 577346.00, 0 nan values
r (float64) has range: 12.58 - 25.25, 0 nan values
redshift (float64) has range: -0.01 - 7.01, 0 nan values
spectral_type (object) has 4 unique values: ['M', 'O/B', 'G/K', 'A/F']
u (float64) has range: -0.14 - 28.25, 0 nan values
z (float64) has range: 11.68 - 26.83, 0 nan values
-> train.csv.gz has 577347 rows and 12 columns.
Here is some information about the columns:
alpha (float64) has range: 0.01 - 360.00, 0 nan values
class (object) has 3 unique values: ['GALAXY', 'QSO', 'STAR']
delta (float64) has range: -17.97 - 79.16, 0 nan values
g (float64) has range: 13.54 - 27.62, 0 nan values
galaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud']
i (float64) has range: 11.96 - 27.91, 0 nan values
id (int64) has range: 0.00 - 577346.00, 0 nan values
r (float64) has range: 12.58 - 25.25, 0 nan values
redshift (float64) has range: -0.01 - 7.01, 0 nan values
spectral_type (object) has 4 unique values: ['M', 'O/B', 'G/K', 'A/F']
u (float64) has range: -0.14 - 28.25, 0 nan values
z (float64) has range: 11.68 - 26.83, 0 nan values

## Metric direction
maximize

## Runtime options
- agent mode: legacy
- gpu: true
- aux: star_classification.csv
- aux file: star_classification.csv
- research model: gpt-5.5
- research reasoning effort: medium
- materialize after hypothesis: false
- execute after materialization: false

## Existing hypotheses
### 1
Title: Color-engineered two-stage classifier
Summary: Build explicit color indices and piecewise features from `u,g,r,i,z,redshift`, then train a hierarchical model that first separates `STAR` vs non-`STAR`, and only then splits `GALAXY` vs `QSO`.
Rationale: Astrophysical classification is strongly driven by color-color structure and redshift; literature on SDSS star/galaxy/QSO work repeatedly shows that compact, physically meaningful features outperform raw magnitudes alone. A hierarchical decision boundary also matches the known class geometry better than forcing a single flat 3-way split.
### 2
Title: Auxiliary SDSS pretrain then domain-stack
Summary: Treat `original_sdss17/star_classification.csv` as a second supervised domain, pretrain on the original SDSS rows using only shared columns, then fine-tune or stack on the synthetic competition train set.
Rationale: The original SDSS file is a natural auxiliary label source with the same target semantics and overlapping feature space. Domain shift is likely real because the competition data includes synthetic categorical fields and altered value ranges, so a two-domain strategy can learn stable astrophysical structure from the real data while adapting to the competition distribution.
### 3
Title: Photometric Color Stack
Summary: Convert the ugriz magnitudes into physically meaningful color and slope features to expose the main class-separating signal.
Rationale: Feature family: numeric_ratios_logs Feature strategy: Keep the raw magnitudes and add a dense set of pairwise color indices and simple spectral-shape summaries such as u-g, g-r, r-i, i-z, u-r, g-i, r-z, u-z, adjacent-band slopes, brightness summaries like mean/min/max/std across ugriz, and a few robust nonlinear transforms of redshift and the color gaps; do not use id or any target-derived feature. Baseline model panel: Balanced logistic regression, a shallow tree ensemble, and a class-weighted gradient-boosted tr…
### 4
Title: Fold-Safe Categorical and Binned Features
Summary: Treat the small categorical columns and coarse numeric bins as first-class signals instead of leaving them as raw text or continuous values only.
Rationale: Feature family: group_statistics_fold_safe Feature strategy: One-hot encode spectral_type and galaxy_population, add fold-safe frequency/likelihood encodings for those categories, bin redshift and selected magnitudes into quantiles, and create small cross features such as spectral_type x redshift_bin and galaxy_population x color_bin; keep all encodings out-of-fold to avoid leakage. Baseline model panel: Balanced logistic regression, categorical-boosting style trees, and a class-weighted gradient-boosted tree mode…
### 5
Title: Auxiliary SDSS Transfer Features
Summary: Use the provided original SDSS table as an auxiliary labeled source to build robust, merged representation features for the competition rows.
Rationale: Feature family: auxiliary_data_features Feature strategy: Explicitly clean the original SDSS magnitudes, align the shared columns with the competition schema, and derive a compact feature set from the auxiliary table such as class-conditional centroid distances, fold-safe nearest-class prototype scores, and distributional priors for redshift and color patterns; avoid direct row-level identity joins and use the auxiliary data only through stable aggregate mappings. Baseline model panel: Balanced logistic regression…
### 6
Title: Unsupervised Photometric Locus Features
Summary: Represent each object by its unsupervised position, cluster affinity, and local density in color-magnitude-redshift space before applying simple balanced classifiers.
Rationale: Feature family: unsupervised_locus_density_embeddings Feature strategy: Build leak-free unsupervised features using only non-target columns from combined train and test: raw ugriz magnitudes, adjacent and broad color indices, redshift, and one-hot spectral_type/galaxy_population. Standardize numeric inputs, then add PCA components, KMeans or Gaussian-mixture cluster distances/probabilities, nearest-cluster margin features, and kNN local-density proxies such as mean distance to the 10/25/50 nearest neighbors. Do no…
### 7
Title: Photometric-Redshift Consistency Features
Summary: Add fold-safe predicted-redshift and redshift-residual features that measure how well each object's ugriz colors explain its observed redshift.
Rationale: Feature family: self_supervised_photoz_residual_features Feature strategy: Keep the already useful raw magnitudes, colors, categorical one-hots, and redshift, but add a staged self-supervised representation: within each training fold, fit a modest regressor to predict redshift from ugriz magnitudes, color indices, alpha/delta, spectral_type, and galaxy_population, then create OOF features such as predicted_redshift, redshift_minus_predicted, absolute_residual, squared_residual, residual divided by observed redshif…
### 8
Title: Source-Derived Photometric Sky Feature Baseline
Summary: Rebuild the strong step-4 feature recipe from the best public-score run and measure it with a simple balanced model panel instead of the original heavy ensemble.
Rationale: Feature family: source_derived_photometric_sky_features Feature strategy: Rebuild the strong step-4 feature recipe from source run 2-step132-remote-rerun artifact 20260608T030220-1ee322aa-4: color indices, band-profile/extrema features, redshift transforms and bins, sky trigonometric/cartesian/harmonic/bin/density features, galactic-coordinate features, categorical crosses, covariate-only frequency/rank features, and optional locality-weighted SDSS17 auxiliary rows. The hypothesis is to test this feature represent…
### 9
Title: Explicit Photometric Sky Formula Features
Summary: Test a fully specified photometric, sky, redshift, galactic, frequency, and rank feature block with simple balanced models.
Rationale: Feature family: explicit_photometric_sky_formula_features Feature strategy: implement the exact formulas listed in feature_strategy: color differences; ugriz SED slope/residual/second-difference/aggregate/extrema features; alpha/delta trigonometric, 3D, interaction, harmonic, bin, offset, and neighbor-density features; redshift transforms and redshift-category interactions; galactic coordinate/bin/interaction features; categorical crosses; train+test covariate-only frequencies; and percentile ranks for u,g,r,i,z,r…

## Best working solutions
### Solution 1
Local CV score: 0.96652
Code:
```python
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
alpha_rad = np.deg2rad(pd.to_numeric(out["alpha"], errors="coerce").to_numpy(…
```
### Solution 2
Local CV score: 0.96544
Code:
```python
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
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
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_SPLITS = 5
CLASS_NAMES = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
def join_parts(*parts: pd.Series) -> pd.Series:
out = parts[0].astype("string").fillna("__MISSING__")
for part in parts[1:]:
out = out + "__" + part.astype("string").fillna("__MISSING__")
return out
def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
numeric_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
base_cat_cols = ["spectral_type", "galaxy_population"]
keep_cols = numeric_cols + base_cat_cols
log_stage("Covariate-only combined train+test features; no labels used")
combined = pd.concat(
[train_df[keep_cols].copy(), test_df[keep_cols].copy()],
axis=0,
ignore_index=True,
)
for col in numeric_cols:
combined[col] = pd.to_numeric(combined[col], errors="coerce").astype(np.float64)
for col in base_cat_cols:
combined[col] = combined[col].astype("string").fillna("__MISSING__")
mags = combined[["u", "g", "r", "i", "z"]].to_numpy(dtype=np.float64)
band_names = np.array(["u", "g", "r", "i", "z"], dtype=object)
combined["u_g"] = combined["u"] - combined["g"]
combined["u_r"] = combined["u"] - combined["r"]
combined["u_i"] = combined["u"] - combined["i"]
combined["u_z"] = combined["u"] - combined["z"]
combined["g_r"] = combined["g"] - combined["r"]
combined["g_i"] = combined["g"] - combined["i"]
combined["g_z"] = combined["g"] - combined["z"]
combined["r_i"] = combined["r"] - combined["i"]
combined["r_z"] = combined["r"] - combined["z"]
k = np.arange(5, dtype=np.float64)
mag_sum = mags.sum(axis=1)
mag_sed_slope = (5.0 * (mags * k).sum(axis=1) - 10.0 * mag_sum) / 50.0
mag_sed_intercept = (mag_sum - 10.0 * mag_sed_slope) / 5.0
mag_fit = mag_sed_intercept[:, None] + mag_sed_slope[:, None] * k[None, :]
mag_resid = mags - mag_fit
mag_d2 = np.diff(mags, n=2, axis=1)
min_idx = np.argmin(mags, axis=1)
max_idx = np.argmax(mags, axis=1)
combined["mag_sed_slope"] = mag_sed_slope
combined["mag_sed_intercept"] = mag_sed_intercept
combined["mag_sed_resid_mean_abs"] = np.abs(mag_resid).mean(axis=1)
combined["mag_sed_resid_std"] = mag_resid.std(axis=1)
combined["mag_sed_d2_mean"] = mag_d2.mean(axis=1)
combined["mag_sed_d2_abs_mean"] = np.abs(mag_d2).mean(axis=1)
combined["mag_sed_d2_std"] = mag_d2.std(axis=1)
combined["mag_mean"] = mags.mean(axis=1)
combined["mag_std"] = mags.std(axis=1)
combined["mag_range"] = mags.max(axis=1) - mags.min(axis=1)
combined["band_min_mag"] = mags.min(axis=1)
combined["band_max_mag"] = mags.max(axis=1)
combined["mag_min_band"] = band_names[min_idx]
combined["mag_max_band"] = band_names[max_idx]
combined["band_min_band"] = combined["mag_min_band"]
combined["band_max_band"] = combined["mag_max_band"]
alpha_deg = np.mod(combined["alpha"].to_numpy(dtype=np.float64), 360.0)
delta_deg = combined["delta"].to_numpy(dtype=np.float64)
alpha_rad = np.deg2rad(alpha_deg)
delta_rad = np.deg2rad(delta_deg)
alpha_sin = np.sin(alpha_rad)
alpha_cos = np.cos(alpha_rad)
delta_sin = np.sin(delta_rad)
delta_cos = np.cos(delta_rad)
combined["alpha_sin"] = alpha_sin
combined["alpha_cos"] = alpha_cos
combined["delta_sin"] = delta_sin
combined["delta_cos"] = delta_cos
sky_x = delta_cos * alpha_cos
sky_y = delta_cos * alpha_sin
sky_z = delta_sin
combined["sky_x"] = sky_x
combined["sky_y"] = sky_y
combined["sky_z"] = sky_z
combined["sky_xy"] = sky_x * sky_y
combined["sky_xz"] = sky_x * sky_z
combined["sky_yz"] = sky_y * sky_z
combined["sky_x2_minus_y2"] = sky_x**2 - sky_y**2
for order in [2, 3, 4, 5, 6, 7, 8]:
combined[f"alpha_sin{order}"] = np.sin(order * alpha_rad)
combined[f"alpha_cos{order}"] = np.cos(order * alpha_rad)
combined[f"delta_sin{order}"] = np.sin(order * delta_rad)
combined[f"delta_cos{order}"] = np.cos(order * delta_rad)
sky_alpha_bin_24 = np.floor(alpha_deg / 15.0).astype(np.int16)
sky_delta_bin_18 = np.clip(np.floor((delta_deg + 90.0) / 10.0), 0, 17).astype(
np.int16
)
sky_cell_bin = (sky_alpha_bin_24 * 19 + sky_delta_bin_18).astype(np.int32)
sky_alpha_bin_48 = np.floor(alpha_deg / 7.5).astype(np.int16)
sky_delta_bin_36 = np.clip(np.floor((delta_deg + 90.0) / 5.0), 0, 35).astype(
np.int16
)
sky_cell_bin_48 = (sky_alpha_bin_48 * 36 + sky_delta_bin_36).astype(np.int32)
combined["sky_alpha_bin_24"] = sky_alpha_bin_24
combined["sky_delta_bin_18"] = sky_delta_bin_18
combined["sky_cell_bin"] = sky_cell_bin
combined["sky_alpha_bin_48"] = sky_alpha_bin_48
combined["sky_delta_bin_36"] = sky_delta_bin_36
combined["sky_cell_bin_48"] = sky_cell_bin_48
sky_alpha_bin_center_deg = (sky_alpha_bin_24 + 0.5) * 15.0
sky_delta_bin_center_deg = (sky_delta_bin_18 + 0.5) * 10.0 - 90.0
combined["sky_alpha_bin_center_deg"] = sky_alpha_bin_center_deg
combined["sky_alpha_bin_offset_deg"] = alpha_deg - sky_alpha_bin_center_deg
combined["sky_delta_bin_center_deg"] = sky_delta_bin_center_deg
combined["sky_delta_bin_offset_deg"] = delta_deg - sky_delta_bin_center_deg
sky_points = np.column_stack([delta_rad, alpha_rad])
nn_model = NearestNeighbors(
n_neighbors=17,
metric="haversine",
algorithm="ball_tree",
n_jobs=-1,
)
nn_model.fit(sky_points)
nn_dist, _ = nn_model.kneighbors(sky_points)
nn_arcmin = nn_dist * (180.0 / np.pi) * 60.0
r5 = nn_arcmin[:, 5]
r10 = nn_arcmin[:, 10]
r16 = nn_arcmin[:, 16]
d5 = np.log1p(5.0 / np.maximum(r5**2, 1e-6))
d10 = np.log1p(10.0 / np.maximum(r10**2, 1e-6))
d16 =…
```
### Solution 3
Local CV score: 0.96413
Code:
```python
import gc
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from aide_solution_helpers import (
aide_stage,
load_competition_data,
log_stage,
working_dir,
write_oof_predictions,
write_submission,
write_test_predictions,
)
warnings.filterwarnings("ignore")
SEED = 2026
N_SPLITS = 5
INNER_SPLITS = 4
CLASS_NAMES = np.array(["GALAXY", "QSO", "STAR"])
CLASS_TO_INT = {name: idx for idx, name in enumerate(CLASS_NAMES)}
def build_base_frame(df: pd.DataFrame) -> pd.DataFrame:
base = df[["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]].copy()
base["ug"] = df["u"] - df["g"]
base["gr"] = df["g"] - df["r"]
base["ri"] = df["r"] - df["i"]
base["iz"] = df["i"] - df["z"]
base["ur"] = df["u"] - df["r"]
base["gi"] = df["g"] - df["i"]
base["spectral_type"] = df["spectral_type"].fillna("missing").astype(str)
base["galaxy_population"] = df["galaxy_population"].fillna("missing").astype(str)
return base
def make_bin_edges(series: pd.Series, n_bins: int) -> np.ndarray:
quantiles = np.linspace(0.0, 1.0, n_bins + 1)
edges = np.unique(np.quantile(series.to_numpy(dtype=float), quantiles))
if edges.size < 2:
return np.array([-np.inf, np.inf], dtype=float)
edges[0] = -np.inf
edges[-1] = np.inf
return edges.astype(float)
def apply_bins(series: pd.Series, edges: np.ndarray, prefix: str) -> pd.Series:
codes = pd.cut(series, bins=edges, labels=False, include_lowest=True)
codes = pd.Series(codes, index=series.index).fillna(-1).astype(int).astype(str)
return prefix + "_" + codes
def build_likelihood_maps(
cat_series: pd.Series, y: np.ndarray, priors: np.ndarray, alpha: float
):
stats = pd.crosstab(cat_series, y).reindex(
columns=np.arange(len(CLASS_NAMES)), fill_value=0
)
totals = stats.sum(axis=1).astype(float)
maps = {}
for class_idx, class_name in enumerate(CLASS_NAMES):
probs = (stats[class_idx] + alpha * priors[class_idx]) / (totals + alpha)
maps[class_name] = probs.to_dict()
return maps
def add_fold_safe_encodings(
train_df: pd.DataFrame,
valid_df: pd.DataFrame,
test_df: pd.DataFrame,
y_train: np.ndarray,
cat_cols,
alpha: float = 20.0,
):
train_df = train_df.copy()
valid_df = valid_df.copy()
test_df = test_df.copy()
priors = np.bincount(y_train, minlength=len(CLASS_NAMES)).astype(np.float64)
priors /= priors.sum()
inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
for col in cat_cols:
freq_map = train_df[col].value_counts(normalize=True).to_dict()
freq_col = f"{col}_freq"
train_df[freq_col] = train_df[col].map(freq_map).fillna(0.0).astype(np.float32)
valid_df[freq_col] = valid_df[col].map(freq_map).fillna(0.0).astype(np.float32)
test_df[freq_col] = test_df[col].map(freq_map).fillna(0.0).astype(np.float32)
inner_encoded = {
class_name: np.empty(len(train_df), dtype=np.float32)
for class_name in CLASS_NAMES
}
for inner_train_idx, inner_valid_idx in inner_cv.split(train_df, y_train):
inner_train = train_df.iloc[inner_train_idx]
inner_valid = train_df.iloc[inner_valid_idx]
inner_y = y_train[inner_train_idx]
maps = build_likelihood_maps(inner_train[col], inner_y, priors, alpha)
for class_idx, class_name in enumerate(CLASS_NAMES):
inner_encoded[class_name][inner_valid_idx] = (
inner_valid[col]
.map(maps[class_name])
.fillna(priors[class_idx])
.to_numpy(np.float32)
)
full_maps = build_likelihood_maps(train_df[col], y_train, priors, alpha)
for class_idx, class_name in enumerate(CLASS_NAMES):
enc_col = f"{col}_p_{class_name.lower()}"
train_df[enc_col] = inner_encoded[class_name]
valid_df[enc_col] = (
valid_df[col]
.map(full_maps[class_name])
.fillna(priors[class_idx])
.astype(np.float32)
)
test_df[enc_col] = (
test_df[col]
.map(full_maps[class_name])
.fillna(priors[class_idx])
.astype(np.float32)
)
return train_df, valid_df, test_df
def engineer_fold_features(train_base, valid_base, test_base, y_train):
train_feat = train_base.copy().reset_index(drop=True)
valid_feat = valid_base.copy().reset_index(drop=True)
test_feat = test_base.copy().reset_index(drop=True)
bin_specs = {
"redshift_bin": ("redshift", 8),
"u_bin": ("u", 8),
"g_bin": ("g", 8),
"r_bin": ("r", 8),
"color_gr_bin": ("gr", 8),
}
for new_col, (src_col, n_bins) in bin_specs.items():
edges = make_bin_edges(train_feat[src_col], n_bins)
train_feat[new_col] = apply_bins(train_feat[src_col], edges, new_col)
valid_feat[new_col] = apply_bins(valid_feat[src_col], edges, new_col)
test_feat[new_col] = apply_bins(test_feat[src_col], edges, new_col)
for frame in (train_feat, valid_feat, test_feat):
frame["spectral_x_redshift_bin"] = (
frame["spectral_type"] + "__" + frame["redshift_bin"]
)
frame["population_x_color_bin"] = (
frame["galaxy_population"] + "__" + frame["color_gr_bin"]
)
train_feat, valid_feat, test_feat = add_fold_safe_encodings(
train_feat,
valid_feat,
test_feat,
y_train,
cat_cols=["spectral_type", "galaxy_population"],
alpha=20.0,
)
cat_cols = [
"spectral_type",
"galaxy_population",
"redshift_bin",
"u_bin",
"g_bin",
"r_bin",
"color_gr_bin",
"spectral_x_redshift_bin",
"population_x_color_bin",
]
numeric_cols = [
"alpha",
"delta",
"u",
"g",
"r",
"i",
"z",
"redshift",
"ug",
"gr",
"ri",
"iz",
"ur",
"gi",
"spectral_type_freq",
"galaxy_population_freq",
"spectral_type_p_galaxy",
"spectral_type_p_qso",
"spectral_type_p_star",
"galaxy_population_p_galaxy",
"galaxy_population_p_qso",
"galaxy_population_p_star",
]
for frame in (train_feat, valid_feat, test_feat):
frame[numeric_cols] = frame[numeric_cols].astype(np.float32)
for col in cat_cols:
frame[col] = frame[col].astype(str)
return train_feat, valid_feat, test_feat, numeric_cols, cat_cols
def make_s…
```
### Solution 4
Local CV score: 0.96366
Code:
```python
import os
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from aide_solution_helpers import (
aide_stage,
load_competition_data,
log_stage,
working_dir,
write_oof_predictions,
write_submission,
write_test_predictions,
)
RANDOM_STATE = 42
N_FOLDS = 5
N_JOBS = min(16, os.cpu_count() or 1)
ID_COL = "id"
TARGET_COL = "class"
CLASS_NAMES = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
CLASS_TO_INT = {label: idx for idx, label in enumerate(CLASS_NAMES)}
SHARED_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
PHOT_COLS = ["u", "g", "r", "i", "z"]
CAT_COLS = ["spectral_type", "galaxy_population"]
def find_auxiliary_path() -> Path:
candidates = [
Path("./input/star_classification.csv"),
Path("./input/original_sdss17/star_classification.csv"),
]
for candidate in candidates:
if candidate.exists():
return candidate
raise FileNotFoundError(
"Expected auxiliary data at ./input/star_classification.csv "
"or ./input/original_sdss17/star_classification.csv"
)
def clean_numeric_frame(frame: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
frame = frame.copy()
for col in numeric_cols:
values = pd.to_numeric(frame[col], errors="coerce")
if col in PHOT_COLS:
values = values.mask(values <= -999)
frame[col] = values.astype(np.float32)
return frame
def prepare_competition_features(
train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
feature_cols = [c for c in train.columns if c not in (ID_COL, TARGET_COL)]
numeric_cols = [c for c in feature_cols if c not in CAT_COLS]
# This concatenation is covariate-only and target-free; it is used only to align category levels.
combined = pd.concat(
[train[feature_cols].copy(), test[feature_cols].copy()],
axis=0,
ignore_index=True,
)
for col in numeric_cols:
values = pd.to_numeric(combined[col], errors="coerce")
if col in PHOT_COLS:
values = values.mask(values <= -999)
combined[col] = values.astype(np.float32)
for col in CAT_COLS:
combined[col] = combined[col].fillna("missing").astype(str).astype("category")
train_features = combined.iloc[: len(train)].copy()
test_features = combined.iloc[len(train) :].copy()
return train_features, test_features, feature_cols
def prepare_auxiliary_features(aux: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
aux = aux.loc[aux[TARGET_COL].isin(CLASS_TO_INT)].copy()
aux_x = clean_numeric_frame(aux[SHARED_COLS], SHARED_COLS)
aux_y = aux[TARGET_COL].map(CLASS_TO_INT).to_numpy(dtype=np.int32)
return aux_x, aux_y
def build_lgbm(seed: int, n_estimators: int) -> lgb.LGBMClassifier:
return lgb.LGBMClassifier(
objective="multiclass",
num_class=len(CLASS_NAMES),
n_estimators=n_estimators,
learning_rate=0.03,
num_leaves=63,
min_child_samples=40,
subsample=0.8,
subsample_freq=1,
colsample_bytree=0.8,
reg_lambda=1.0,
class_weight="balanced",
random_state=seed,
n_jobs=N_JOBS,
verbosity=-1,
)
with aide_stage("build_features_stage"):
working_dir()
train, test, sample_sub = load_competition_data()
aux_path = find_auxiliary_path()
aux = pd.read_csv(aux_path)
y = train[TARGET_COL].map(CLASS_TO_INT).to_numpy(dtype=np.int32)
train_features, test_features, feature_cols = prepare_competition_features(
train, test
)
aux_features, aux_y = prepare_auxiliary_features(aux)
with aide_stage("make_folds_stage"):
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
folds = list(skf.split(train_features, y))
with aide_stage("fit_predict_fold_stage"):
log_stage(
f"fold=0/{N_FOLDS}|model=aux_lgbm_shared|rows={len(aux_features)}|path={aux_path.name}"
)
aux_model = build_lgbm(seed=RANDOM_STATE, n_estimators=1200)
aux_model.fit(aux_features, aux_y)
# The auxiliary model is trained on a separate labeled domain, so these stack features do not leak competition targets.
aux_train_proba = aux_model.predict_proba(train_features[SHARED_COLS])
aux_test_proba = aux_model.predict_proba(test_features[SHARED_COLS])
stacked_train = train_features.copy()
stacked_test = test_features.copy()
for class_idx, class_name in enumerate(CLASS_NAMES):
col_name = f"aux_proba_{class_name.lower()}"
stacked_train[col_name] = aux_train_proba[:, class_idx].astype(np.float32)
stacked_test[col_name] = aux_test_proba[:, class_idx].astype(np.float32)
oof_proba = np.zeros((len(train), len(CLASS_NAMES)), dtype=np.float32)
test_proba = np.zeros((len(test), len(CLASS_NAMES)), dtype=np.float32)
for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
log_stage(f"fold={fold}/{N_FOLDS}|model=comp_lgbm_stacked")
x_train = stacked_train.iloc[train_idx]
x_valid = stacked_train.iloc[valid_idx]
y_train = y[train_idx]
y_valid = y[valid_idx]
model = build_lgbm(seed=RANDOM_STATE + fold, n_estimators=3000)
model.fit(
x_train,
y_train,
eval_set=[(x_valid, y_valid)],
eval_metric="multi_logloss",
categorical_feature=CAT_COLS,
callbacks=[lgb.early_stopping(100, verbose=False)],
)
valid_proba = model.predict_proba(x_valid, num_iteration=model.best_iteration_)
fold_test_proba = model.predict_proba(
stacked_test, num_iteration=model.best_iteration_
)
oof_proba[valid_idx] = valid_proba.astype(np.float32)
test_proba += fold_test_proba.astype(np.float32) / N_FOLDS
valid_pred = valid_proba.argmax(axis=1)
fold_score = balanced_accuracy_score(y_valid, valid_pred)
print(f"Fold {fold} balanced_accuracy: {fold_score:.6f}", flush=True)
with aide_stage("score_stage"):
oof_pred = oof_proba.argmax(axis=1)
cv_score = balanced_accuracy_score(y, oof_pred)
print(f"CV balanced_accuracy: {cv_score:.6f}", flush=True)
with aide_stage("write_outputs_stage"):
oof_labels = CLASS_NAMES[oof_pred]
test_pred = test_proba.argmax(axis=1)
test_labels = CLASS_NAMES[test_pred]
oof_df = pd.DataFrame(
{
"row": np.arange(len(train), dtype=np.int64),
"target": CLASS_NAMES[y],
"prediction": oof_labels,
}
)
write_oof_predictions(oof_df)
test_pred_df = pd.DataFrame(
{
ID_COL: sample_sub[ID_COL].values,
TARGET_COL: test_labels,
f"{TARGET…
```
### Solution 5
Local CV score: 0.96259
Code:
```python
import os
import warnings
from collections import OrderedDict
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from aide_solution_helpers import (
aide_stage,
load_competition_data,
log_stage,
working_dir,
write_oof_predictions,
write_submission,
write_test_predictions,
write_validation_predictions,
)
warnings.filterwarnings("ignore")
MAG_COLS = ["u", "g", "r", "i", "z"]
BASE_NUMERIC_COLS = ["alpha", "delta", "redshift"] + MAG_COLS
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]
RANDOM_STATE = 42
N_SPLITS = 5
def build_features(df: pd.DataFrame) -> pd.DataFrame:
feat = pd.DataFrame(index=df.index)
for col in BASE_NUMERIC_COLS:
feat[col] = pd.to_numeric(df[col], errors="coerce")
# Sentinel cleanup is applied row-wise only; auxiliary data is intentionally unused
# so the measured change reflects the photometric color-stack hypothesis itself.
for col in MAG_COLS:
feat.loc[feat[col] <= -1000, col] = np.nan
color_defs = [
("u_g", "u", "g"),
("g_r", "g", "r"),
("r_i", "r", "i"),
("i_z", "i", "z"),
("u_r", "u", "r"),
("g_i", "g", "i"),
("r_z", "r", "z"),
("u_z", "u", "z"),
]
for name, left, right in color_defs:
feat[name] = feat[left] - feat[right]
feat["ug_minus_gr"] = feat["u_g"] - feat["g_r"]
feat["gr_minus_ri"] = feat["g_r"] - feat["r_i"]
feat["ri_minus_iz"] = feat["r_i"] - feat["i_z"]
feat["slope_balance"] = feat["u_g"] - feat["i_z"]
feat["blue_curvature"] = feat["u"] - 2.0 * feat["g"] + feat["r"]
feat["mid_curvature"] = feat["g"] - 2.0 * feat["r"] + feat["i"]
feat["red_curvature"] = feat["r"] - 2.0 * feat["i"] + feat["z"]
mag_frame = feat[MAG_COLS]
color_cols = [name for name, _, _ in color_defs]
color_frame = feat[color_cols]
adjacent_color_frame = feat[["u_g", "g_r", "r_i", "i_z"]]
feat["mag_mean"] = mag_frame.mean(axis=1)
feat["mag_min"] = mag_frame.min(axis=1)
feat["mag_max"] = mag_frame.max(axis=1)
feat["mag_std"] = mag_frame.std(axis=1)
feat["mag_range"] = feat["mag_max"] - feat["mag_min"]
feat["color_mean"] = color_frame.mean(axis=1)
feat["color_std"] = color_frame.std(axis=1)
feat["color_range"] = color_frame.max(axis=1) - color_frame.min(axis=1)
feat["adjacent_color_mean"] = adjacent_color_frame.mean(axis=1)
feat["adjacent_color_std"] = adjacent_color_frame.std(axis=1)
feat["color_energy"] = (
feat["u_g"].pow(2)
+ feat["g_r"].pow(2)
+ feat["r_i"].pow(2)
+ feat["i_z"].pow(2)
)
feat["ug_gr_ratio"] = feat["u_g"] / (1.0 + feat["g_r"].abs())
feat["ri_iz_ratio"] = feat["r_i"] / (1.0 + feat["i_z"].abs())
feat["uz_over_range"] = feat["u_z"] / (1.0 + feat["mag_range"].abs())
red = feat["redshift"].fillna(0.0)
feat["redshift_abs"] = red.abs()
feat["redshift_log1p_abs"] = np.log1p(red.abs())
feat["redshift_signed_log1p"] = np.sign(red) * np.log1p(red.abs())
feat["redshift_sq"] = red.pow(2)
feat["redshift_sqrt_abs"] = np.sqrt(red.abs())
feat["redshift_color_mix"] = red * feat["g_r"].fillna(0.0)
feat["redshift_color_gap"] = red * feat["u_g"].fillna(0.0)
for col in CATEGORICAL_COLS:
feat[col] = df[col].astype("string")
return feat
def make_preprocessor(numeric_cols, categorical_cols):
numeric_pipe = Pipeline(
steps=[
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler()),
]
)
categorical_pipe = Pipeline(
steps=[
("imputer", SimpleImputer(strategy="most_frequent")),
("onehot", OneHotEncoder(handle_unknown="ignore")),
]
)
return ColumnTransformer(
transformers=[
("num", numeric_pipe, numeric_cols),
("cat", categorical_pipe, categorical_cols),
],
remainder="drop",
)
def make_logreg(numeric_cols, categorical_cols):
return Pipeline(
steps=[
("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
(
"model",
LogisticRegression(
C=1.0,
solver="saga",
max_iter=200,
class_weight="balanced",
multi_class="multinomial",
n_jobs=min(16, os.cpu_count() or 1),
random_state=RANDOM_STATE,
),
),
]
)
def make_extra_trees(numeric_cols, categorical_cols):
return Pipeline(
steps=[
("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
(
"model",
ExtraTreesClassifier(
n_estimators=250,
max_depth=8,
min_samples_leaf=5,
max_features="sqrt",
class_weight="balanced_subsample",
random_state=RANDOM_STATE,
n_jobs=min(16, os.cpu_count() or 1),
),
),
]
)
def make_xgb(numeric_cols, categorical_cols, num_class):
return Pipeline(
steps=[
("preprocess", make_preprocessor(numeric_cols, categorical_cols)),
(
"model",
XGBClassifier(
objective="multi:softprob",
num_class=num_class,
n_estimators=350,
learning_rate=0.05,
max_depth=6,
min_child_weight=4,
subsample=0.85,
colsample_bytree=0.85,
reg_lambda=1.0,
random_state=RANDOM_STATE,
tree_method="hist",
device="cuda",
eval_metric="mlogloss",
n_jobs=min(16, os.cpu_count() or 1),
verbosity=0,
),
),
]
)
def main():
_ = working_dir()
with aide_stage("build_features_stage"):
train, test, sample_sub = load_competition_data()
X_train = build_features(train)
X_test = build_features(test)
y = train["class"].astype(str).to_numpy()
test_ids = sample_sub["id"].copy()
numeric_cols = [c for c in X_train.columns if c not in CATEGORICAL_COLS]
with aide_stage("make_folds_stage"):
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_
skf = StratifiedKFold(
n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)
folds = list(skf.split(X_train, y_encoded))
model_builders = OrderedDict(
[
("balanced_logreg", lambda: make_logreg(numeric_cols, CATEGORICAL_COLS)),
(
"shallow_extratrees",
lambda: make_extra_trees(numeric_cols, CATEGORICAL_COLS),
),
(
"balanced_xgb_gpu",
lambda: make_xgb(numeric_cols, CATEGORICAL…
```

## Worst working solutions
### Solution 1
Local CV score: 0.96113
Code:
```python
import warnings
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from aide_solution_helpers import (
aide_stage,
load_competition_data,
log_stage,
working_dir,
write_oof_predictions,
write_submission,
write_test_predictions,
write_validation_predictions,
)
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_SPLITS = 5
CLASS_NAMES = np.array(["GALAXY", "QSO", "STAR"], dtype=object)
CAT_COLS = ["spectral_type", "galaxy_population"]
MAG_COLS = ["u", "g", "r", "i", "z"]
def build_features(frame: pd.DataFrame) -> pd.DataFrame:
df = frame.copy()
for col in ["alpha", "delta", "redshift"] + MAG_COLS:
df[col] = pd.to_numeric(df[col], errors="coerce")
for col in MAG_COLS:
df.loc[df[col] <= -999, col] = np.nan
for col in CAT_COLS:
df[col] = df[col].fillna("missing").astype(str)
color_pairs = [
("u", "g"),
("g", "r"),
("r", "i"),
("i", "z"),
("u", "r"),
("g", "i"),
("r", "z"),
("u", "z"),
]
for left, right in color_pairs:
df[f"color_{left}_{right}"] = df[left] - df[right]
df["color_curve_blue"] = df["color_u_g"] - df["color_g_r"]
df["color_curve_red"] = df["color_r_i"] - df["color_i_z"]
df["mag_mean"] = df[MAG_COLS].mean(axis=1)
df["mag_std"] = df[MAG_COLS].std(axis=1, ddof=0)
df["mag_range"] = df[MAG_COLS].max(axis=1) - df[MAG_COLS].min(axis=1)
df["missing_mag_count"] = df[MAG_COLS].isna().sum(axis=1)
redshift = df["redshift"].fillna(0.0)
redshift_abs = redshift.abs()
df["redshift_abs"] = redshift_abs
df["redshift_sq"] = redshift * redshift
df["redshift_log1p_abs"] = np.log1p(redshift_abs)
df["redshift_sqrt_abs"] = np.sqrt(redshift_abs)
df["redshift_pos"] = np.clip(redshift, 0.0, None)
df["redshift_neg"] = np.clip(-redshift, 0.0, None)
for knot in (0.2, 0.5, 1.0, 2.0):
suffix = str(knot).replace(".", "_")
df[f"redshift_relu_{suffix}"] = np.clip(redshift - knot, 0.0, None)
df["redshift_x_color_u_g"] = redshift * df["color_u_g"].fillna(0.0)
df["redshift_x_color_g_r"] = redshift * df["color_g_r"].fillna(0.0)
df["redshift_x_color_r_i"] = redshift * df["color_r_i"].fillna(0.0)
df["redshift_x_color_i_z"] = redshift * df["color_i_z"].fillna(0.0)
return df
def fit_catboost(
x_train: pd.DataFrame,
y_train: pd.Series,
x_valid: pd.DataFrame,
y_valid: pd.Series,
cat_features: list[str],
model_name: str,
) -> CatBoostClassifier:
base_params = {
"loss_function": "Logloss",
"eval_metric": "Logloss",
"iterations": 1000,
"learning_rate": 0.05,
"depth": 8,
"l2_leaf_reg": 5.0,
"random_seed": RANDOM_STATE,
"auto_class_weights": "Balanced",
"od_type": "Iter",
"od_wait": 100,
"allow_writing_files": False,
"thread_count": -1,
"verbose": False,
}
last_error = None
for use_gpu in (True, False):
params = base_params.copy()
if use_gpu:
params.update(
{
"task_type": "GPU",
"devices": "0",
"gpu_ram_part": 0.8,
}
)
else:
log_stage(f"event=info|model={model_name}|device=cpu_fallback")
model = CatBoostClassifier(**params)
try:
model.fit(
x_train,
y_train,
cat_features=cat_features,
eval_set=(x_valid, y_valid),
use_best_model=True,
verbose=False,
)
return model
except Exception as exc:
last_error = exc
if use_gpu:
log_stage(
f"event=warning|model={model_name}|device=gpu_failed|error_type={exc.__class__.__name__}"
)
else:
raise
raise last_error
def positive_class_probability(
model: CatBoostClassifier, x: pd.DataFrame
) -> np.ndarray:
class_list = list(model.classes_)
positive_index = class_list.index(1)
return model.predict_proba(x)[:, positive_index]
def hard_route_predictions(p_star: np.ndarray, p_qso_cond: np.ndarray) -> np.ndarray:
predictions = np.full(p_star.shape[0], "GALAXY", dtype=object)
qso_mask = (p_star < 0.5) & (p_qso_cond >= 0.5)
star_mask = p_star >= 0.5
predictions[qso_mask] = "QSO"
predictions[star_mask] = "STAR"
return predictions
def hierarchical_probabilities(
p_star: np.ndarray, p_qso_cond: np.ndarray
) -> np.ndarray:
p_nonstar = 1.0 - p_star
probs = np.column_stack(
[
p_nonstar * (1.0 - p_qso_cond),
p_nonstar * p_qso_cond,
p_star,
]
)
row_sums = np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
return probs / row_sums
def main() -> None:
_ = working_dir()
_ = write_validation_predictions
train, test, sample_sub = load_competition_data()
with aide_stage("build_features_stage"):
y = train["class"].astype(str).reset_index(drop=True)
train_features = build_features(train.drop(columns=["class"])).reset_index(
drop=True
)
test_features = build_features(test).reset_index(drop=True)
feature_cols = [col for col in train_features.columns if col != "id"]
x = train_features[feature_cols]
x_test = test_features[feature_cols]
with aide_stage("make_folds_stage"):
splitter = StratifiedKFold(
n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)
folds = list(splitter.split(x, y))
n_train = len(x)
n_test = len(x_test)
oof_star_prob = np.zeros(n_train, dtype=np.float32)
oof_qso_cond_prob = np.zeros(n_train, dtype=np.float32)
test_star_prob = np.zeros(n_test, dtype=np.float32)
test_qso_cond_prob = np.zeros(n_test, dtype=np.float32)
fold_scores = []
with aide_stage("fit_predict_fold_stage"):
for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
x_train = x.iloc[train_idx]
x_valid = x.iloc[valid_idx]
y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]
y_stage1_train = (y_train == "STAR").astype(int)
y_stage1_valid = (y_valid == "STAR").astype(int)
log_stage(f"event=info|fold={fold}|model=catboost_stage1_star_vs_nonstar")
model_stage1 = fit_catboost(
x_train=x_train,
y_train=y_stage1_train,
x_valid=x_valid,
y_valid=y_stage1_valid,
cat_features=CAT_COLS,
model_name=f"fold{fold}_stage1",
)
p_star_valid = positive_class_probability(model_stage1, x_valid)
p_star_test = positive_class_probability(model_stage1, x_test)
nonstar_train_mask = y_train != "STAR"
nonstar_valid_mask = y_valid != "STAR"
x_train_stage2 = x_train.loc[nonstar_train_mask]
y_train_stage2 = (y_train.loc[nonstar_train_mask] == "QSO").astype(int)
x_valid_stage2 = x_valid.loc[nonstar_valid_mask]
y_valid_stage2…
```
### Solution 2
Local CV score: 0.96137
Code:
```python
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from aide_solution_helpers import (
aide_stage,
load_competition_data,
log_stage,
working_dir,
write_oof_predictions,
write_submission,
write_test_predictions,
)
RANDOM_STATE = 42
N_SPLITS = 5
TARGET = "class"
ID_COL = "id"
MAG_COLS = ["u", "g", "r", "i", "z"]
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]
COLOR_FEATURES = [
("u_minus_g", "u", "g"),
("g_minus_r", "g", "r"),
("r_minus_i", "r", "i"),
("i_minus_z", "i", "z"),
("u_minus_r", "u", "r"),
("g_minus_i", "g", "i"),
("r_minus_z", "r", "z"),
]
RESIDUAL_FEATURES = [
"predicted_redshift",
"redshift_minus_predicted",
"redshift_abs_residual",
"redshift_squared_residual",
"redshift_scaled_residual",
"photoz_residual_sign",
"photoz_abs_residual_gt_0p05",
"photoz_abs_residual_gt_0p10",
"photoz_abs_residual_gt_0p20",
"photoz_consistent_within_5pct_scale",
]
def add_color_features(frame: pd.DataFrame) -> pd.DataFrame:
frame = frame.copy()
for feature_name, left, right in COLOR_FEATURES:
frame[feature_name] = frame[left] - frame[right]
frame["redshift_log1p"] = np.log1p(frame["redshift"].clip(lower=-0.999999))
frame["magnitude_mean"] = frame[MAG_COLS].mean(axis=1)
frame["magnitude_std"] = frame[MAG_COLS].std(axis=1)
return frame
def prepare_base_features(train: pd.DataFrame, test: pd.DataFrame):
train = train.copy()
test = test.copy()
for frame in (train, test):
for col in MAG_COLS:
frame[col] = frame[col].replace(-9999, np.nan)
for col in CATEGORICAL_COLS:
frame[col] = frame[col].fillna("missing").astype(str)
train = add_color_features(train)
test = add_color_features(test)
# This concatenation uses only covariates and no target column, so it is safe for one-hot alignment.
combined = pd.concat(
[train.drop(columns=[TARGET]), test],
axis=0,
ignore_index=True,
)
combined = pd.get_dummies(combined, columns=CATEGORICAL_COLS, dummy_na=False)
train_base = combined.iloc[: len(train)].reset_index(drop=True)
test_base = combined.iloc[len(train) :].reset_index(drop=True)
train_base = train_base.drop(columns=[ID_COL]).astype(np.float32)
test_base = test_base.drop(columns=[ID_COL]).astype(np.float32)
photoz_feature_cols = [
col
for col in train_base.columns
if col != "redshift" and not col.startswith("redshift_")
]
classifier_feature_cols = list(train_base.columns)
return train_base, test_base, photoz_feature_cols, classifier_feature_cols
def fit_photoz_oof(
train_base: pd.DataFrame,
test_base: pd.DataFrame,
photoz_feature_cols,
folds,
):
x_photoz_train = train_base[photoz_feature_cols].copy()
x_photoz_test = test_base[photoz_feature_cols].copy()
fill_values = x_photoz_train.median(axis=0)
x_photoz_train = x_photoz_train.fillna(fill_values)
x_photoz_test = x_photoz_test.fillna(fill_values)
y_redshift = train_base["redshift"].values
oof_pred = np.zeros(len(train_base), dtype=np.float32)
regressor = HistGradientBoostingRegressor(
loss="squared_error",
learning_rate=0.05,
max_depth=6,
max_iter=150,
min_samples_leaf=100,
l2_regularization=1.0,
random_state=RANDOM_STATE,
)
for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
log_stage(
f"event=info|stage=fit_predict_fold_stage|fold={fold}|model=photoz_regressor"
)
regressor.fit(x_photoz_train.iloc[train_idx], y_redshift[train_idx])
oof_pred[valid_idx] = regressor.predict(x_photoz_train.iloc[valid_idx]).astype(
np.float32
)
regressor.fit(x_photoz_train, y_redshift)
test_pred = regressor.predict(x_photoz_test).astype(np.float32)
return oof_pred, test_pred
def build_photoz_features(observed_redshift, predicted_redshift) -> pd.DataFrame:
observed_redshift = np.asarray(observed_redshift, dtype=np.float32)
predicted_redshift = np.asarray(predicted_redshift, dtype=np.float32)
residual = observed_redshift - predicted_redshift
abs_residual = np.abs(residual)
scale = 1.0 + np.abs(observed_redshift)
return pd.DataFrame(
{
"predicted_redshift": predicted_redshift,
"redshift_minus_predicted": residual,
"redshift_abs_residual": abs_residual,
"redshift_squared_residual": residual**2,
"redshift_scaled_residual": residual / scale,
"photoz_residual_sign": np.sign(residual).astype(np.float32),
"photoz_abs_residual_gt_0p05": (abs_residual > 0.05).astype(np.float32),
"photoz_abs_residual_gt_0p10": (abs_residual > 0.10).astype(np.float32),
"photoz_abs_residual_gt_0p20": (abs_residual > 0.20).astype(np.float32),
"photoz_consistent_within_5pct_scale": (
abs_residual <= 0.05 * scale
).astype(np.float32),
}
)
def build_classifier_matrices(
train_base: pd.DataFrame,
test_base: pd.DataFrame,
train_photoz: pd.DataFrame,
test_photoz: pd.DataFrame,
):
x_train = pd.concat(
[train_base.reset_index(drop=True), train_photoz.reset_index(drop=True)], axis=1
)
x_test = pd.concat(
[test_base.reset_index(drop=True), test_photoz.reset_index(drop=True)], axis=1
)
fill_values = x_train.median(axis=0)
x_train = x_train.fillna(fill_values).astype(np.float32)
x_test = x_test.fillna(fill_values).astype(np.float32)
return x_train, x_test
def make_model_builders():
return {
"ridge": lambda: Pipeline(
[
("scaler", StandardScaler()),
("model", RidgeClassifier(alpha=2.0, class_weight="balanced")),
]
),
"extra_trees": lambda: ExtraTreesClassifier(
n_estimators=250,
max_depth=16,
min_samples_leaf=5,
class_weight="balanced",
n_jobs=-1,
random_state=RANDOM_STATE,
),
"catboost": lambda: CatBoostClassifier(
loss_function="MultiClass",
auto_class_weights="Balanced",
iterations=1200,
learning_rate=0.05,
depth=7,
l2_leaf_reg=5.0,
random_seed=RANDOM_STATE,
task_type="GPU",
devices="0",
gpu_ram_part=0.8,
od_type="Iter",
od_wait=100,
verbose=False,
),
}
def main():
working_dir()
train, test, sample_sub = load_co…
```
### Solution 3
Local CV score: 0.96163
Code:
```python
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from aide_solution_helpers import (
aide_stage,
load_competition_data,
log_stage,
working_dir,
write_oof_predictions,
write_submission,
write_test_predictions,
write_validation_predictions,
)
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_SPLITS = 5
ID_COL = "id"
TARGET_COL = "class"
MAG_COLS = ["u", "g", "r", "i", "z"]
SHARED_NUMERIC_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
COLOR_COLS = ["u_g", "g_r", "r_i", "i_z", "u_z", "g_z"]
BASE_NUMERIC_COLS = SHARED_NUMERIC_COLS + COLOR_COLS + ["mag_mean", "log_redshift"]
TRANSFER_COLS = [
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
"u_z",
"log_redshift",
]
CATEGORICAL_COLS = ["spectral_type", "galaxy_population"]
def clean_and_engineer(frame: pd.DataFrame) -> pd.DataFrame:
frame = frame.copy()
for col in SHARED_NUMERIC_COLS:
frame[col] = pd.to_numeric(frame[col], errors="coerce")
for col in MAG_COLS:
frame.loc[frame[col] <= -999.0, col] = np.nan
frame["u_g"] = frame["u"] - frame["g"]
frame["g_r"] = frame["g"] - frame["r"]
frame["r_i"] = frame["r"] - frame["i"]
frame["i_z"] = frame["i"] - frame["z"]
frame["u_z"] = frame["u"] - frame["z"]
frame["g_z"] = frame["g"] - frame["z"]
frame["mag_mean"] = frame[MAG_COLS].mean(axis=1)
frame["log_redshift"] = np.sign(frame["redshift"]) * np.log1p(
np.abs(frame["redshift"])
)
return frame
def softmax_rows(scores: np.ndarray) -> np.ndarray:
shifted = scores - scores.max(axis=1, keepdims=True)
exp_scores = np.exp(shifted)
return exp_scores / exp_scores.sum(axis=1, keepdims=True)
def min_distance_to_centers(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
x_norm = np.sum(x * x, axis=1, keepdims=True)
c_norm = np.sum(centers * centers, axis=1)[None, :]
d2 = x_norm + c_norm - 2.0 * (x @ centers.T)
return np.sqrt(np.maximum(d2.min(axis=1), 0.0)).astype(np.float32)
def fit_auxiliary_transfer(aux_df: pd.DataFrame, class_names: list[str]):
aux_values = aux_df[TRANSFER_COLS].replace([np.inf, -np.inf], np.nan)
aux_medians = aux_values.median()
aux_filled = aux_values.fillna(aux_medians)
scaler = StandardScaler()
aux_scaled = scaler.fit_transform(aux_filled).astype(np.float32)
centroids = {}
spreads = {}
prototypes = {}
aux_labels = aux_df[TARGET_COL].astype(str).to_numpy()
for class_name in class_names:
class_matrix = aux_scaled[aux_labels == class_name]
centroids[class_name] = class_matrix.mean(axis=0).astype(np.float32)
spreads[class_name] = np.clip(class_matrix.std(axis=0), 0.05, None).astype(
np.float32
)
n_clusters = int(min(4, max(1, class_matrix.shape[0])))
kmeans = MiniBatchKMeans(
n_clusters=n_clusters,
random_state=RANDOM_STATE,
batch_size=4096,
n_init=10,
)
kmeans.fit(class_matrix)
prototypes[class_name] = kmeans.cluster_centers_.astype(np.float32)
return aux_medians, scaler, centroids, spreads, prototypes
def make_auxiliary_features(
frame: pd.DataFrame,
class_names: list[str],
aux_bundle,
) -> pd.DataFrame:
aux_medians, scaler, centroids, spreads, prototypes = aux_bundle
values = frame[TRANSFER_COLS].replace([np.inf, -np.inf], np.nan).fillna(aux_medians)
scaled = scaler.transform(values).astype(np.float32)
feature_data = {}
centroid_scores = []
prototype_scores = []
for class_name in class_names:
center = centroids[class_name]
spread = spreads[class_name]
diff = scaled - center
centroid_dist = np.sqrt(np.mean(diff * diff, axis=1)).astype(np.float32)
pattern_maha = np.sqrt(np.mean((diff / spread) ** 2, axis=1)).astype(np.float32)
proto_dist = min_distance_to_centers(scaled, prototypes[class_name])
feature_data[f"aux_centroid_dist_{class_name}"] = centroid_dist
feature_data[f"aux_pattern_maha_{class_name}"] = pattern_maha
feature_data[f"aux_proto_dist_{class_name}"] = proto_dist
centroid_scores.append((-centroid_dist).reshape(-1, 1))
prototype_scores.append((-proto_dist).reshape(-1, 1))
centroid_probs = softmax_rows(np.hstack(centroid_scores))
prototype_probs = softmax_rows(np.hstack(prototype_scores))
for idx, class_name in enumerate(class_names):
feature_data[f"aux_centroid_prob_{class_name}"] = centroid_probs[:, idx].astype(
np.float32
)
feature_data[f"aux_proto_prob_{class_name}"] = prototype_probs[:, idx].astype(
np.float32
)
return pd.DataFrame(feature_data, index=frame.index)
def run_model_cv(
model_name: str,
x: np.ndarray,
x_test: np.ndarray,
y: np.ndarray,
folds: list[tuple[np.ndarray, np.ndarray]],
n_classes: int,
):
oof_proba = np.zeros((x.shape[0], n_classes), dtype=np.float32)
test_proba = np.zeros((x_test.shape[0], n_classes), dtype=np.float32)
fold_scores = []
for fold, (train_idx, valid_idx) in enumerate(folds, start=1):
log_stage(f"model={model_name}|fold={fold}|event=fit_start")
x_train = x[train_idx]
y_train = y[train_idx]
x_valid = x[valid_idx]
y_valid = y[valid_idx]
if model_name == "logreg":
scaler = StandardScaler()
x_train_fit = scaler.fit_transform(x_train)
x_valid_fit = scaler.transform(x_valid)
x_test_fit = scaler.transform(x_test)
model = LogisticRegression(
C=1.0,
max_iter=400,
solver="lbfgs",
multi_class="multinomial",
class_weight="balanced",
random_state=RANDOM_STATE,
)
model.fit(x_train_fit, y_train)
valid_proba = model.predict_proba(x_valid_fit).astype(np.float32)
test_fold_proba = model.predict_proba(x_test_fit).astype(np.float32)
elif model_name == "extratrees":
model = ExtraTreesClassifier(
n_estimators=220,
max_depth=24,
min_samples_leaf=4,
max_features=0.8,
class_weight="balanced",
n_jobs=-1,
random_state=RANDOM_STATE + fold,
)
model.fit(x_train, y_train)
valid_proba = model.predict_proba(x_valid).astype(np.float32)
test_fold_proba = model.…
```
