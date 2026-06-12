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

# Hard boundary for prior examples
Do not imitate prior hypotheses that use intermediate supervised models,
hierarchical classifiers, pretraining, stacking, calibration, label
propagation, KNN class posteriors, class-conditional density models, threshold
tuning, or other multi-stage prediction architectures as the hypothesis
mechanism. If such ideas are absent from the Existing hypotheses section, that
absence is intentional. Treat the visible Existing hypotheses as examples of
feature/preprocessing/data-representation directions to avoid duplicating, not
as permission to introduce supervised meta-features or algorithmic search.

# Novelty dimensions
A new initial feature-search hypothesis is materially different only if at
least one of these dimensions changes:
- feature representation family
- preprocessing mechanism
- data source usage
- covariate-only statistical feature mechanism
- physical or domain-specific representation
- dimensionality-reduction or embedding method
- missingness or outlier treatment

Changing only the model panel, hyperparameters, parameter names, bin count,
threshold values, seeds, wording, or evaluation wrapper is not novel. Changing
the binning mechanism or target-free statistical mechanism can be novel;
changing only the number of bins is not.

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

## Existing hypotheses
### 1
Title: Photometric Color Stack
Summary: Convert the ugriz magnitudes into physically meaningful color and slope features to expose the main class-separating signal.
Rationale: Feature family: numeric_ratios_logs Feature strategy: Keep the raw magnitudes and add a dense set of pairwise color indices and simple spectral-shape summaries such as u-g, g-r, r-i, i-z, u-r, g-i, r-z, u-z, adjacent-band slopes, brightness summaries like mean/min/max/std across ugriz, and a few robust nonlinear transforms of redshift and the color gaps; do not use id or any target-derived feature. Baseline model panel: Balanced logistic regression, a shallow tree ensemble, and a class-weighted gradient-boosted tr…
### 2
Title: Fold-Safe Categorical and Binned Features
Summary: Treat the small categorical columns and coarse numeric bins as first-class signals instead of leaving them as raw text or continuous values only.
Rationale: Feature family: group_statistics_fold_safe Feature strategy: One-hot encode spectral_type and galaxy_population, add fold-safe frequency/likelihood encodings for those categories, bin redshift and selected magnitudes into quantiles, and create small cross features such as spectral_type x redshift_bin and galaxy_population x color_bin; keep all encodings out-of-fold to avoid leakage. Baseline model panel: Balanced logistic regression, categorical-boosting style trees, and a class-weighted gradient-boosted tree mode…
### 3
Title: Auxiliary SDSS Transfer Features
Summary: Use the provided original SDSS table as an auxiliary labeled source to build robust, merged representation features for the competition rows.
Rationale: Feature family: auxiliary_data_features Feature strategy: Explicitly clean the original SDSS magnitudes, align the shared columns with the competition schema, and derive a compact feature set from the auxiliary table such as class-conditional centroid distances, fold-safe nearest-class prototype scores, and distributional priors for redshift and color patterns; avoid direct row-level identity joins and use the auxiliary data only through stable aggregate mappings. Baseline model panel: Balanced logistic regression…
### 4
Title: Unsupervised Photometric Locus Features
Summary: Represent each object by its unsupervised position, cluster affinity, and local density in color-magnitude-redshift space before applying simple balanced classifiers.
Rationale: Feature family: unsupervised_locus_density_embeddings Feature strategy: Build leak-free unsupervised features using only non-target columns from combined train and test: raw ugriz magnitudes, adjacent and broad color indices, redshift, and one-hot spectral_type/galaxy_population. Standardize numeric inputs, then add PCA components, KMeans or Gaussian-mixture cluster distances/probabilities, nearest-cluster margin features, and kNN local-density proxies such as mean distance to the 10/25/50 nearest neighbors. Do no…
### 5
Title: Photometric-Redshift Consistency Features
Summary: Add fold-safe predicted-redshift and redshift-residual features that measure how well each object's ugriz colors explain its observed redshift.
Rationale: Feature family: self_supervised_photoz_residual_features Feature strategy: Keep the already useful raw magnitudes, colors, categorical one-hots, and redshift, but add a staged self-supervised representation: within each training fold, fit a modest regressor to predict redshift from ugriz magnitudes, color indices, alpha/delta, spectral_type, and galaxy_population, then create OOF features such as predicted_redshift, redshift_minus_predicted, absolute_residual, squared_residual, residual divided by observed redshif…
### 6
Title: Source-Derived Photometric Sky Feature Baseline
Summary: Rebuild the strong step-4 feature recipe from the best public-score run and measure it with a simple balanced model panel instead of the original heavy ensemble.
Rationale: Feature family: source_derived_photometric_sky_features Feature strategy: Rebuild the strong step-4 feature recipe from source run 2-step132-remote-rerun artifact 20260608T030220-1ee322aa-4: color indices, band-profile/extrema features, redshift transforms and bins, sky trigonometric/cartesian/harmonic/bin/density features, galactic-coordinate features, categorical crosses, covariate-only frequency/rank features, and optional locality-weighted SDSS17 auxiliary rows. The hypothesis is to test this feature represent…
### 7
Title: Explicit Photometric Sky Formula Features
Summary: Test a fully specified photometric, sky, redshift, galactic, frequency, and rank feature block with simple balanced models.
Rationale: Feature family: explicit_photometric_sky_formula_features Feature strategy: implement the exact formulas listed in feature_strategy: color differences; ugriz SED slope/residual/second-difference/aggregate/extrema features; alpha/delta trigonometric, 3D, interaction, harmonic, bin, offset, and neighbor-density features; redshift transforms and redshift-category interactions; galactic coordinate/bin/interaction features; categorical crosses; train+test covariate-only frequencies; and percentile ranks for u,g,r,i,z,r…
### 8
Title: Relative Flux And Luminosity Proxies
Summary: Convert ugriz magnitudes into linear relative-flux, flux-share, spectral-moment, and redshift-scaled luminosity-proxy features to test whether a physical photometric representation adds signal beyond magnitude-space colors.
Rationale: Feature family: physical_flux_luminosity_features Feature strategy: Keep the raw numeric and categorical columns, but add a parallel representation where each magnitude band is converted to relative flux as 10^(-0.4*m) after clipping only invalid/sentinel values. Derive total_flux, log_total_flux, per-band flux fractions, adjacent and broad flux ratios, normalized SED moments over approximate SDSS filter wavelengths, flux-weighted mean wavelength, flux concentration/entropy, blue-to-red flux balance, and redshift-…
### 9
Title: Self-Supervised Metadata Consistency Features
Summary: Predict `spectral_type` and `galaxy_population` from photometry/redshift/sky covariates and use the fold-safe probability, entropy, margin, and mismatch residuals as class-prediction features.
Rationale: Feature family: self_supervised_metadata_consistency_features Feature strategy: Keep the raw numeric columns, one-hot `spectral_type` and `galaxy_population`, and a compact set of adjacent color indices, then add two auxiliary covariate-only learners: one predicts `spectral_type` from ugriz colors, redshift, alpha, and delta, and the other predicts `galaxy_population` from the same inputs. For each training fold, generate out-of-fold features: full predicted probability vectors, max probability, entropy, top-two m…
### 10
Title: Sky-Local Photometric Residualization
Summary: Create covariate-only sky-sector calibration features by expressing each object's magnitudes and colors relative to local alpha/delta neighborhood medians.
Rationale: Feature family: sky_local_photometric_residual_features Feature strategy: Keep raw ugriz, redshift, alpha/delta, spectral_type, galaxy_population, and basic adjacent color indices, then add unsupervised sky-local residual features. Partition the sky using fixed RA/Dec bins or spherical KMeans/HEALPix-like cells on combined train/test covariates. For each band and key color, compute global medians and sky-cell medians, optionally smoothed with neighboring cells. Add residuals such as u_minus_cell_median, g_r_minus_…
