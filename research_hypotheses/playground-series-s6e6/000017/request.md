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

Do not propose heavy ensembling, stacking, calibration pipelines,
hyperparameter search, seed search, or advanced model-specific tricks in
initial hypotheses. Those belong to later algorithm/ensemble/tuning phases
after initial feature-search hypotheses have produced scores.

# Output contract
Return exactly 1 concise new initial feature-search
hypotheses. Do not target a specific previous node or code block. Use the
prior results only to avoid repeating approaches that have already been tried
and, when scores are available, as evidence about which feature families and
simple baseline algorithms look promising. Treat unexecuted hypotheses only as
anti-duplication context. Treat buggy hypotheses as implementation warnings,
not as evidence that the feature family is weak. Do not debug broken code.

When existing_hypotheses is present, it contains hypotheses already stored for
this task. Use it as anti-duplication context: the new hypothesis must be
materially different in feature family, preprocessing strategy, and expected
signal. When current_run_hypotheses is present, it contains hypotheses already
created earlier in this same run, optionally with materialization and score
metadata. Use it to avoid duplicates and to understand what has already been
materialized or scored.

# Prior research history
If previous_research_summaries is present, it lists recent completed research
proposals. Each entry includes its summary plus the maximum local CV score and
Kaggle public score observed afterwards when available. Try to propose ideas
that are unique relative to those earlier summaries, or explicitly develop the
strongest methods from them into a new testable direction.

# Context field meanings
best_working_solutions contains the highest-scoring code snippets that ran
successfully. worst_working_solutions contains the lowest-scoring code
snippets that still ran successfully. local_cv_score is the validation metric.
kaggle_public_score is included only when a completed Kaggle public
leaderboard score is available for that exact node. Use these examples only to
understand what has already been tried and what performed well or poorly.
current_run_hypotheses contains earlier hypotheses from this same run, including
their id, title, summary, rationale, expected_effect, risk, and optional
materialization/score metadata.
existing_hypotheses contains previously stored task hypotheses, including
disabled hypotheses; disabled entries are still anti-duplication context, not
recommendations. runtime_options records the current agent mode, GPU setting,
auxiliary file selection, and research execution/materialization settings.
target_hypothesis_id, when present, is the ID that will be assigned to this
single generated hypothesis.

# Required JSON output shape
Return JSON with: summary; hypotheses[].title; hypotheses[].summary;
hypotheses[].feature_family; hypotheses[].feature_strategy;
hypotheses[].baseline_model_panel; hypotheses[].model_panel_rationale;
hypotheses[].validation_strategy; hypotheses[].materialization_hint;
hypotheses[].expected_signal; hypotheses[].risk; hypotheses[].sources. The
hypotheses array must contain exactly 1 items.

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
- risk: leakage, overfitting, runtime, data availability, or no-op risks.
- sources: concise URLs or source names used for this idea; use an empty array
  when none are available.

# Current task and prior-result summary
```json
{
  "task_desc": "## Goal\nPredict the stellar class for each object in the test set.\n\nFor each row in `test.csv`, predict the `class` label. The target column in\n`train.csv` is `class`; the identifier column is `id`.\n\n## Evaluation\nSubmissions are evaluated using balanced accuracy. Higher is better.\n\nCompetition-specific modeling hint: if using CatBoost for this multiclass task,\ninclude `auto_class_weights=\"Balanced\"` unless explicitly testing a different\nclass-weighting strategy; this has empirically improved local CV and public\nleaderboard score for this competition.\nAnalogous balanced-class settings should be used for other multiclass tree\nmodels unless explicitly testing a different class-weighting strategy: for\nLightGBM use `class_weight=\"balanced\"`, and for XGBoost pass fold-specific\n`sample_weight=compute_sample_weight(class_weight=\"balanced\", y=y_train)` to\n`.fit()`.\n\nThe submission file must contain a header and exactly these columns:\n\n```csv\nid,class\n577347,STAR\n577348,GALAXY\n577349,QSO\n```\n\n`class` must contain one of `GALAXY`, `STAR`, or `QSO`.\n\n## Data description\n- **train.csv** - training data with the multiclass target column `class`\n- **test.csv** - test data without the target column\n- **sample_submission.csv** - sample submission in the required format\n\nAdditional auxiliary data description for `star_classification.csv`:\n\nOriginal SDSS17 Stellar Classification Dataset.\n\nThis is the original real-world dataset that inspired the synthetic Playground\nSeries S6E6 competition data. It can be used as raw auxiliary data, but it is\nnot automatically merged with train.csv or test.csv.\n\nCommon columns with the competition data:\nalpha, delta, u, g, r, i, z, redshift, class.\n\nColumns present in this original dataset but not in the competition files:\nobj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.\n\nCompetition columns not present in this original dataset:\nid, spectral_type, galaxy_population.\n\nGenerated code should decide whether and how to use this file. Any merge,\nfiltering, cleaning of sentinel magnitudes, or column mapping must be done\nexplicitly by the generated solution code.\n",
  "data_overview": "```\nplayground-series-s6e6.zip (61.4 MB)\nsample_submission.csv (247436 lines)\nsample_submission.csv.gz (247436 lines)\ntest.csv (247436 lines)\ntest.csv.gz (247436 lines)\ntrain.csv (577348 lines)\ntrain.csv.gz (577348 lines)\noriginal_sdss17/\n    star_classification.csv (100001 lines)\n    star_classification.txt (18 lines)```\n\n-> original_sdss17/star_classification.csv has 100000 rows and 18 columns.\nHere is some information about the columns:\nMJD (int64) has range: 51608.00 - 58932.00, 0 nan values\nalpha (float64) has range: 0.01 - 360.00, 0 nan values\ncam_col (int64) has 6 unique values: [2, 5, 3, 4, 6, 1]\nclass (object) has 3 unique values: ['GALAXY', 'QSO', 'STAR']\ndelta (float64) has range: -18.79 - 83.00, 0 nan values\nfiber_ID (int64) has range: 1.00 - 1000.00, 0 nan values\nfield_ID (int64) has range: 11.00 - 989.00, 0 nan values\ng (float64) has range: -9999.00 - 31.60, 0 nan values\ni (float64) has range: 9.47 - 32.14, 0 nan values\nobj_ID (float64) has range: 1237645942904389888.00 - 1237680531356386304.00, 0 nan values\nplate (int64) has range: 266.00 - 12547.00, 0 nan values\nr (float64) has range: 9.82 - 29.57, 0 nan values\nredshift (float64) has range: -0.01 - 7.01, 0 nan values\nrerun_ID (int64) has 1 unique values: [301]\nrun_ID (int64) has range: 109.00 - 8162.00, 0 nan values\nspec_obj_ID (float64) has range: 299519089380976640.00 - 14126940609093851136.00, 0 nan values\nu (float64) has range: -9999.00 - 32.78, 0 nan values\nz (float64) has range: -9999.00 - 29.38, 0 nan values\n\n-> original_sdss17/star_classification.txt has content:\n\nOriginal SDSS17 Stellar Classification Dataset.\n\nThis is the original real-world dataset that inspired the synthetic Playground\nSeries S6E6 competition data. It can be used as raw auxiliary data, but it is\nnot automatically merged with train.csv or test.csv.\n\nCommon columns with the competition data:\nalpha, delta, u, g, r, i, z, redshift, class.\n\nColumns present in this original dataset but not in the competition files:\nobj_ID, run_ID, rerun_ID, cam_col, field_ID, spec_obj_ID, plate, MJD, fiber_ID.\n\nCompetition columns not present in this original dataset:\nid, spectral_type, galaxy_population.\n\nGenerated code should decide whether and how to use this file. Any merge,\nfiltering, cleaning of sentinel magnitudes, or column mapping must be done\nexplicitly by the generated solution code.\n\n\n-> sample_submission.csv has 247435 rows and 2 columns.\nHere is some information about the columns:\nclass (object) has 1 unique values: ['GALAXY']\nid (int64) has range: 577347.00 - 824781.00, 0 nan values\n\n-> sample_submission.csv.gz has 247435 rows and 2 columns.\nHere is some information about the columns:\nclass (object) has 1 unique values: ['GALAXY']\nid (int64) has range: 577347.00 - 824781.00, 0 nan values\n\n-> test.csv has 247435 rows and 11 columns.\nHere is some information about the columns:\nalpha (float64) has range: 0.01 - 360.00, 0 nan values\ndelta (float64) has range: -17.96 - 79.17, 0 nan values\ng (float64) has range: 13.37 - 27.17, 0 nan values\ngalaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud']\ni (float64) has range: 10.03 - 24.57, 0 nan values\nid (int64) has range: 577347.00 - 824781.00, 0 nan values\nr (float64) has range: 10.39 - 25.29, 0 nan values\nredshift (float64) has range: -0.01 - 7.01, 0 nan values\nspectral_type (object) has 4 unique values: ['G/K', 'M', 'O/B', 'A/F']\nu (float64) has range: 13.90 - 27.84, 0 nan values\nz (float64) has range: 10.63 - 25.70, 0 nan values\n\n-> test.csv.gz has 247435 rows and 11 columns.\nHere is some information about the columns:\nalpha (float64) has range: 0.01 - 360.00, 0 nan values\ndelta (float64) has range: -17.96 - 79.17, 0 nan values\ng (float64) has range: 13.37 - 27.17, 0 nan values\ngalaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud']\ni (float64) has range: 10.03 - 24.57, 0 nan values\nid (int64) has range: 577347.00 - 824781.00, 0 nan values\nr (float64) has range: 10.39 - 25.29, 0 nan values\nredshift (float64) has range: -0.01 - 7.01, 0 nan values\nspectral_type (object) has 4 unique values: ['G/K', 'M', 'O/B', 'A/F']\nu (float64) has range: 13.90 - 27.84, 0 nan values\nz (float64) has range: 10.63 - 25.70, 0 nan values\n\n-> train.csv has 577347 rows and 12 columns.\nHere is some information about the columns:\nalpha (float64) has range: 0.01 - 360.00, 0 nan values\nclass (object) has 3 unique values: ['GALAXY', 'QSO', 'STAR']\ndelta (float64) has range: -17.97 - 79.16, 0 nan values\ng (float64) has range: 13.54 - 27.62, 0 nan values\ngalaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud']\ni (float64) has range: 11.96 - 27.91, 0 nan values\nid (int64) has range: 0.00 - 577346.00, 0 nan values\nr (float64) has range: 12.58 - 25.25, 0 nan values\nredshift (float64) has range: -0.01 - 7.01, 0 nan values\nspectral_type (object) has 4 unique values: ['M', 'O/B', 'G/K', 'A/F']\nu (float64) has range: -0.14 - 28.25, 0 nan values\nz (float64) has range: 11.68 - 26.83, 0 nan values\n\n-> train.csv.gz has 577347 rows and 12 columns.\nHere is some information about the columns:\nalpha (float64) has range: 0.01 - 360.00, 0 nan values\nclass (object) has 3 unique values: ['GALAXY', 'QSO', 'STAR']\ndelta (float64) has range: -17.97 - 79.16, 0 nan values\ng (float64) has range: 13.54 - 27.62, 0 nan values\ngalaxy_population (object) has 2 unique values: ['Red_Sequence', 'Blue_Cloud']\ni (float64) has range: 11.96 - 27.91, 0 nan values\nid (int64) has range: 0.00 - 577346.00, 0 nan values\nr (float64) has range: 12.58 - 25.25, 0 nan values\nredshift (float64) has range: -0.01 - 7.01, 0 nan values\nspectral_type (object) has 4 unique values: ['M', 'O/B', 'G/K', 'A/F']\nu (float64) has range: -0.14 - 28.25, 0 nan values\nz (float64) has range: 11.68 - 26.83, 0 nan values",
  "best_working_solutions": [],
  "worst_working_solutions": [],
  "previous_research_summaries": [],
  "existing_hypotheses": [
    {
      "id": "000001",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "Prune AutoGluon-flagged constant and duplicate-like features",
      "summary": "Remove the feature columns the seed AutoGluon logs already reported as useless or unused, starting with constant/missing/sentinel indicators and then testing a broader duplicate-like cleanup.",
      "rationale": "The seed log reports many original features that AutoGluon drops or leaves unused. Keeping them still expands preprocessing, memory, and split-search surface for tree models. A hard pruning pass is a low-risk way to reduce noisy candidate splits while preserving the stronger photometric, color, sky, redshift, and categorical features.",
      "expected_effect": "A small balanced_accuracy gain is plausible because XGBoost and CatBoost see fewer redundant or constant candidates, and the seed already shows only a subset of engineered columns surviving AutoGluon processing.",
      "risk": "Some columns marked unused on one seed split may become useful after a later feature addition or different validation split, so test constant-only before the broader duplicate-like cleanup."
    },
    {
      "id": "000002",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "Auxiliary class-conditional distance features",
      "summary": "Use class labels in original_sdss17/star_classification.csv to compute robust distances from each competition row to GALAXY, STAR, and QSO reference loci without matching objects by sky position.",
      "rationale": "The seed uses the auxiliary data mostly as an unlabeled reference. The auxiliary file has the same core photometric columns plus class labels, so it can provide a deterministic external teacher. Robust class-conditional distances in magnitude, color, and redshift space may be especially useful around QSO/STAR and QSO/GALAXY boundaries.",
      "expected_effect": "Balanced accuracy may improve because the model receives compact class-locus proximity signals instead of having to infer external class structure indirectly from raw features.",
      "risk": "This relies on labeled public external data. If the experiment policy disallows external labels, disable this hypothesis. Distribution mismatch between SDSS auxiliary rows and competition rows could also overfit public validation."
    },
    {
      "id": "000003",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "Auxiliary coarse color-redshift class priors",
      "summary": "Build smoothed P(class | color, redshift) lookup features from the labeled auxiliary file using coarse quantile bins, then map those priors onto train and test rows.",
      "rationale": "Class-conditional distances and grid priors encode different information. The grid prior gives the model an empirical local class distribution in color-redshift space, which tree models can use more directly than raw density patterns from the auxiliary data.",
      "expected_effect": "The model may gain calibrated local priors for rare or ambiguous regions, improving QSO and STAR recall under balanced accuracy.",
      "risk": "Sparse bins can create noisy priors. Keep the first grid coarse and smoothed, and avoid deriving bin edges from competition labels."
    },
    {
      "id": "000004",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "Activate or remove the dead id feature block",
      "summary": "Verify whether seed id-derived features actually reach preprocess, then either pass id into preprocess and drop raw id afterward or remove the inactive block.",
      "rationale": "The seed plan attributes score to id-derived synthetic ordering features, but the wrapper can drop id before preprocess. If the id block is inactive, current conclusions about id signal are misleading. If it is activated, deterministic rank, gap, block, and modulo features may expose synthetic generation artifacts.",
      "expected_effect": "If id features were inactive, the cleanup clarifies the baseline. If active id features help, the run may recover generation-order signal not available through photometry alone.",
      "risk": "Raw or poorly handled id can overfit synthetic ordering. The experiment must ensure raw id is not passed as a direct model feature unless explicitly tested."
    },
    {
      "id": "000005",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "ID-block residual features for synthetic batch artifacts",
      "summary": "After id features are confirmed active, add local block-count and within-block residual features over several id block sizes.",
      "rationale": "Simple id modulo and rank features may be too crude to capture synthetic generation batches. Local block residuals can expose shifts in redshift, magnitude, and colors within nearby id ranges while avoiding raw id as a direct predictor.",
      "expected_effect": "Balanced accuracy may improve if the playground generator emits local id batches with shifted photometric or redshift distributions.",
      "risk": "This hypothesis depends on id activation from hypothesis 000004. Block aggregates over combined train/test can be fragile if public/private test ordering differs."
    },
    {
      "id": "000006",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "Ordered spectral and galaxy-population interactions",
      "summary": "Replace arbitrary categorical codes with semantically ordered spectral_type and galaxy_population features plus interactions with colors, redshift, and magnitude summaries.",
      "rationale": "The seed encodes categories through frequency and arbitrary categorical codes, and galaxy_population_code/freq were reported unused. Domain ordering can give tree models cleaner monotonic splits and interaction signals, especially for spectral types O/B, A/F, G/K, and M.",
      "expected_effect": "The model may better exploit stable synthetic categories while reducing reliance on arbitrary code values.",
      "risk": "If the synthetic category labels are not physically consistent, the imposed order can add misleading interactions. Keep the first run compact and compare with the pruning variant separately."
    },
    {
      "id": "000007",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "Redshift RBF gates for hard QSO regimes",
      "summary": "Add local radial-basis redshift features and color interactions around known difficult redshift ranges, especially the QSO/STAR overlap region.",
      "rationale": "The seed has global redshift transforms and interactions, but no local basis functions. Balanced accuracy rewards class-specific improvements, and QSO errors can concentrate in narrow redshift regimes where global trees may need many splits.",
      "expected_effect": "The model may gain sharper decision gates for QSO/STAR and QSO/GALAXY overlap, improving minority-class recall without changing the training profile.",
      "risk": "Hand-picked centers can overfit local validation. Validate as a standalone preprocessing change before combining with auxiliary class priors."
    },
    {
      "id": "000008",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "Faint-end photometric noise proxy features",
      "summary": "Create deterministic proxies for photometric uncertainty from relative faintness, then normalize adjacent colors by those proxies.",
      "rationale": "The competition data lacks explicit photometric errors. Magnitudes still imply a rough noise regime: fainter objects should have less reliable colors. Giving the model color reliability proxies can help classify ambiguous STAR/QSO and QSO/GALAXY cases.",
      "expected_effect": "The model may learn when color features are less trustworthy, improving boundary decisions without external labels.",
      "risk": "The proxy is heuristic and can duplicate magnitude information. Keep it separate from pruning and auxiliary-label experiments."
    },
    {
      "id": "000009",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "Sky and auxiliary-sky family ablations",
      "summary": "Test whole-family pruning of sky geometry and auxiliary sky percentile features instead of dropping isolated columns.",
      "rationale": "Direct catalog matching by alpha/delta appears weak, so sky features should be treated as possible generator artifacts rather than reliable physical identity keys. Large sky feature families can help local validation while hurting public/private generalization.",
      "expected_effect": "The best ablation may improve generalization by removing fragile coordinate artifacts while retaining robust occupancy signals.",
      "risk": "Some sky features may be genuinely useful synthetic priors. Treat this as an ablation family and do not combine with new auxiliary-label features in the first run."
    },
    {
      "id": "000010",
      "enabled": true,
      "agent_modes": [
        "autogluon"
      ],
      "compatible_with_current_agent": false,
      "title": "Reallocate AutoGluon time from LightGBM to XGB and CatBoost",
      "summary": "Keep the seed preprocessing fixed but test a bounded AutoGluon profile that gives more capacity to XGBoost and CatBoost, since the seed ensemble weight is dominated by XGBoost and CatBoost.",
      "rationale": "The seed log shows XGBoost as the strongest base model and LightGBM with little or no effective ensemble contribution in the referenced run. A single controlled profile change can test whether time is better spent on multiple XGB configurations and a focused CatBoost run instead of an equally weighted default model set.",
      "expected_effect": "If LightGBM contributes little under the seed profile, reallocating time may improve the weighted ensemble or produce a stronger XGB-only/CAT-supported solution.",
      "risk": "This is a model-control experiment, not preprocessing. It can interact with all feature changes, so test it separately after preprocessing hypotheses are ranked."
    },
    {
      "id": "000011",
      "enabled": true,
      "agent_modes": [
        "legacy"
      ],
      "compatible_with_current_agent": true,
      "title": "Color-engineered two-stage classifier",
      "summary": "Build explicit color indices and piecewise features from `u,g,r,i,z,redshift`, then train a hierarchical model that first separates `STAR` vs non-`STAR`, and only then splits `GALAXY` vs `QSO`.",
      "rationale": "Astrophysical classification is strongly driven by color-color structure and redshift; literature on SDSS star/galaxy/QSO work repeatedly shows that compact, physically meaningful features outperform raw magnitudes alone. A hierarchical decision boundary also matches the known class geometry better than forcing a single flat 3-way split.",
      "expected_effect": "Higher recall on the minority/confusable classes, especially `QSO`, without sacrificing the easy `STAR` region.",
      "risk": "If the synthetic train/test distribution differs from the original SDSS geometry, hand-crafted splits may overfit a visually intuitive but suboptimal boundary."
    },
    {
      "id": "000012",
      "enabled": true,
      "agent_modes": [
        "legacy"
      ],
      "compatible_with_current_agent": true,
      "title": "Auxiliary SDSS pretrain then domain-stack",
      "summary": "Treat `original_sdss17/star_classification.csv` as a second supervised domain, pretrain on the original SDSS rows using only shared columns, then fine-tune or stack on the synthetic competition train set.",
      "rationale": "The original SDSS file is a natural auxiliary label source with the same target semantics and overlapping feature space. Domain shift is likely real because the competition data includes synthetic categorical fields and altered value ranges, so a two-domain strategy can learn stable astrophysical structure from the real data while adapting to the competition distribution.",
      "expected_effect": "Better generalization on ambiguous objects and improved robustness on rare boundary cases, especially if the synthetic data preserves the original class semantics but shifts feature distributions.",
      "risk": "Negative transfer is possible if the synthetic generation process warps the joint distribution enough that the original SDSS signal is only weakly aligned."
    },
    {
      "id": "000013",
      "enabled": false,
      "agent_modes": [
        "legacy"
      ],
      "compatible_with_current_agent": false,
      "title": "Calibrated weighted ensemble with per-class decision tuning",
      "summary": "Ensemble multiple balanced tree models, then calibrate their probabilities on out-of-fold predictions and tune class decisions to maximize balanced accuracy rather than raw log loss.",
      "rationale": "Boosted tree outputs are often miscalibrated, and balanced accuracy is sensitive to per-class recall rather than probability quality. A calibrated OOF ensemble can reduce overconfident errors and make the final argmax more reliable, especially when class frequencies and class confusions are asymmetric.",
      "expected_effect": "Small but consistent gain from better error allocation across classes, particularly on the harder minority class and on borderline `GALAXY`/`QSO` cases.",
      "risk": "Calibration can help only if the base models are already strong; if the ensemble is weak, recalibration may polish probabilities without improving classification recall."
    },
    {
      "id": "000014",
      "enabled": true,
      "agent_modes": [
        "legacy"
      ],
      "compatible_with_current_agent": true,
      "title": "Photometric Color Stack",
      "summary": "Convert the ugriz magnitudes into physically meaningful color and slope features to expose the main class-separating signal.",
      "rationale": "Feature family: numeric_ratios_logs Feature strategy: Keep the raw magnitudes and add a dense set of pairwise color indices and simple spectral-shape summaries such as u-g, g-r, r-i, i-z, u-r, g-i, r-z, u-z, adjacent-band slopes, brightness summaries like mean/min/max/std across ugriz, and a few robust nonlinear transforms of redshift and the color gaps; do not use id or any target-derived feature. Baseline model panel: Balanced logistic regression, a shallow tree ensemble, and a class-weighted gradient-boosted tree model. Model panel rationale: This feature family should be close to linearly separable in parts but still benefit from nonlinearity, so a linear model, a tree ensemble, and a b…",
      "expected_effect": "CV should improve mainly from stronger STAR/QSO separation and from fewer confident but wrong GALAXY predictions; tree models should gain more than a plain linear baseline if the colors are useful.",
      "risk": "If the synthetic data already encodes class almost directly in redshift and one or two magnitudes, the extra color features may be redundant; overly aggressive nonlinear transforms can also add noise."
    },
    {
      "id": "000015",
      "enabled": true,
      "agent_modes": [
        "legacy"
      ],
      "compatible_with_current_agent": true,
      "title": "Fold-Safe Categorical and Binned Features",
      "summary": "Treat the small categorical columns and coarse numeric bins as first-class signals instead of leaving them as raw text or continuous values only.",
      "rationale": "Feature family: group_statistics_fold_safe Feature strategy: One-hot encode spectral_type and galaxy_population, add fold-safe frequency/likelihood encodings for those categories, bin redshift and selected magnitudes into quantiles, and create small cross features such as spectral_type x redshift_bin and galaxy_population x color_bin; keep all encodings out-of-fold to avoid leakage. Baseline model panel: Balanced logistic regression, categorical-boosting style trees, and a class-weighted gradient-boosted tree model. Model panel rationale: These features are deliberately low-dimensional and tabular, so a linear baseline checks whether the bins/encodings already separate classes while tree mo…",
      "expected_effect": "If useful, the fold-local encodings should lift validation without large runtime cost, especially on the minority class recall that balanced accuracy rewards; gains should be visible even before complex models are introduced.",
      "risk": "Any leakage in the encodings will inflate CV, and overly fine bins may fragment the data enough to hurt balanced accuracy more than they help."
    },
    {
      "id": "000016",
      "enabled": true,
      "agent_modes": [
        "legacy"
      ],
      "compatible_with_current_agent": true,
      "title": "Auxiliary SDSS Transfer Features",
      "summary": "Use the provided original SDSS table as an auxiliary labeled source to build robust, merged representation features for the competition rows.",
      "rationale": "Feature family: auxiliary_data_features Feature strategy: Explicitly clean the original SDSS magnitudes, align the shared columns with the competition schema, and derive a compact feature set from the auxiliary table such as class-conditional centroid distances, fold-safe nearest-class prototype scores, and distributional priors for redshift and color patterns; avoid direct row-level identity joins and use the auxiliary data only through stable aggregate mappings. Baseline model panel: Balanced logistic regression, a balanced tree ensemble, and a class-weighted gradient-boosted tree model. Model panel rationale: If the auxiliary table adds real transfer signal, even simple models should pic…",
      "expected_effect": "A real gain should appear as a consistent bump across folds with modest variance, especially on confusing QSO versus STAR regions where external SDSS structure may help.",
      "risk": "The auxiliary dataset may have domain shift relative to the synthetic competition data, and careless use of labels or merges can create leakage or brittle features that do not transfer to test."
    }
  ],
  "current_run_hypotheses": [],
  "runtime_options": {
    "data_dir": "aide/example_tasks/playground-series-s6e6",
    "agent": {
      "mode": "legacy",
      "gpu": true,
      "aux": "star_classification.csv",
      "aux_file_name": "star_classification.csv"
    },
    "research": {
      "mode": "hypothesis",
      "model": "gpt-5.5",
      "reasoning_effort": "high",
      "timeout": 900,
      "materialize": false,
      "execute": false
    }
  },
  "target_hypothesis_id": "000017",
  "hypothesis_count": 1
}
```
