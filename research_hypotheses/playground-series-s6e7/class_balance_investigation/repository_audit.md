# S6E7 class-balancing repository audit

Audit date: 2026-07-11 (Europe/Warsaw)  
Repository revision: `29c3c1654da95e9066e8ba258840f19292612a51`  
Scope: mandatory pre-training audit only; no training launched.

## Gate decision (updated after Main protocol approval)

The historical `class_balance: balanced` path described below computed weights
from all labels before the holdout split and is retained only as historical,
non-comparable evidence. Main approved the proposed Stage A constants and an
implementation correction. The working-tree implementation now computes and
normalizes inverse-frequency weights only from the explicit post-split training
partition, aligns them by index, validates them, and logs the exact mapping.
Custom weighting rejects bagged/auto-stack and non-holdout validation rather
than claiming fold safety. No training has been launched; Main code review is
the remaining gate.

Main also decided that Stage C must preserve the strongest recorded pipeline's
exact transductive train+test covariate transforms and label them explicitly;
feature fitting will not be redesigned during this investigation.

## Initial-audit checklist (13/13 inspected)

1. **Training entry points.** The generated fixed AutoGluon wrapper is built by
   `aide/autogluon_preprocess.py::build_autogluon_wrapper` (lines 607 onward),
   and its generated `main()` is at source-template lines 1236-1409. Historical
   profile reruns use `scripts/rerun_autogluon_profile.py` (profile construction
   around 699-788 and CLI `main` around 1357-1485). A standalone simple runner
   exists at `scripts/autogluon_baseline.py:38-70`. The strongest-feature
   one-off runner is
   `logs/experiments/s6e7_top_boost_transforms.py:529-715`.

2. **Configuration system.** Typed defaults are in
   `aide/utils/config.py:191-207`; YAML profiles/defaults are in
   `aide/utils/config.yaml:177-1703`; resolution/override precedence is
   `aide/autogluon_preprocess.py:445-523`. Project selection is environment
   driven (`AIDE_PROJECT_NAME=playground-series-s6e7`, metric
   `balanced_accuracy`, data directory
   `aide/example_tasks/playground-series-s6e7`). The current global profile is
   not an experiment protocol and must not be inherited implicitly.

3. **Data-loading path.** The generated wrapper reads compressed or plain
   train/test/sample files with `_read_csv` at
   `aide/autogluon_preprocess.py:667-672`, optional auxiliary data at 675-683,
   and loads them in generated `main` at 1245-1248. Target and ID are inferred
   from the sample-submission columns at 1249-1251; the target/ID are removed
   before feature generation at 1256-1258.

4. **Feature-generation paths.** Generated hypotheses supply `preprocess`,
   invoked on the concatenated train+test covariate frame at
   `aide/autogluon_preprocess.py:1259-1284`, then sliced back at 1285-1291.
   The recorded combined top-feature runner follows the same transductive
   pattern at `logs/experiments/s6e7_top_boost_transforms.py:588-600` and its
   feature union is defined at 93-114. AutoGluon then performs its internal
   feature generation. Any frequency/median/group statistic in user
   `preprocess` therefore sees test covariates unless explicitly redesigned.

5. **Validation implementation.** The wrapper detects explicit bagging and
   uses AutoGluon OOF without `tuning_data`; otherwise `validation_strategy:
   holdout` uses stratified `train_test_split` with configured fraction and
   seed (`aide/autogluon_preprocess.py:1303-1323`). With neither condition,
   AutoGluon chooses its internal validation. The strongest-feature runner has
   equivalent logic at `logs/experiments/s6e7_top_boost_transforms.py:777-836`.
   Stage A should use an explicit holdout to keep row assignment auditable.

6. **Metric implementation.** The project metric is `balanced_accuracy`.
   `TabularPredictor(eval_metric=...)` receives it at
   `aide/autogluon_preprocess.py:1343-1354`; explicit holdout evaluation uses
   `predictor.evaluate` at 1368-1384. Historical prediction artifacts were
   independently recomputed with `sklearn.metrics.balanced_accuracy_score` and
   matched their stored scores exactly.

7. **Experiment/result storage.** Per-run artifacts live under
   `logs/<run>/artifacts/<timestamp>/` (`aide_result.json`, `solution.py`,
   prediction artifacts, submission). The repository-wide structured index is
   `logs/submission_index.json`. Calibration records are under
   `logs/autogluon_fast_profile_cv_public/s6e7_profile_20260709T204443Z/`.
   The top-feature one-off result is
   `logs/experiments/s6e7_top_boost_transforms/metrics.json`. This investigation
   uses the sibling persistent artifacts in this directory and must write each
   future run to a unique directory/log.

8. **Target and class distribution.** Current data fingerprints and counts:

   | item | value |
   |---|---|
   | train | 690,088 rows x 15 columns |
   | test | 295,753 rows x 14 columns |
   | auxiliary | 50,000 rows x 16 columns |
   | target | `health_condition` |
   | `at-risk` | 592,561 (0.858674546) |
   | `unhealthy` | 57,724 (0.083647303) |
   | `fit` | 39,803 (0.057678151) |
   | train SHA-256 | `8d74774c5ed7b8aba981116d59d3d8cd753731c8a1eb8920b965016b20443cad` |
   | test SHA-256 | `0066684a879ca44b7f5e554d2507d4118d20e5f751056a57c82e603b75c0c30a` |
   | sample SHA-256 | `6e8fed8306350028ac6cd894452801c6363995cf1ea3b52014654475dd88d95b` |
   | auxiliary SHA-256 | `697b806faa2f55d6e0bb1e6283b522b3de58db8e8aaaf0fd2314d63abbc0a659` |

9. **Seeds/folds/split.** Historical neutral reference candidates use a
   stratified 20% holdout and seed 1729. Reproducing the split from current data
   yields 552,070 train / 138,018 validation rows, with train counts
   `at-risk=474049`, `unhealthy=46179`, `fit=31842` and validation counts
   `at-risk=118512`, `unhealthy=11545`, `fit=7961`. Sorted ID fingerprints are
   train `5d82c6531913afea458a6e821be7dafeafb8679971010edc97ad2b0f8bed1a80`
   and validation
   `15c4f89f3870a4a8ff1b4b3aed386a321d0439374277fef7b48e8491d481d27e`.
   The proposed Stage A protocol freezes these IDs.

10. **Models/budget/ensembling.** Existing fair-one reference profiles request
    exactly XGB, GBM, CAT; `medium_quality`; 600 seconds total; one explicitly
    prioritized configuration per family; GPU; no bagging, stacking, or
    weighted ensemble. See `aide/utils/config.yaml:201-264` and 449-479.
    Current `full_boost` is CPU, 600 seconds, holdout, no ensemble/stacking
    (`aide/utils/config.yaml:818-845`). The initial Stage A GPU block failed
    because the execution environment exposed zero GPUs. Main subsequently
    froze an otherwise identical CPU-only fair-one block with
    `fit_weighted_ensemble=false`, so family scores, not an ensemble policy,
    determine verification.

11. **Exact `class_balance: balanced` execution.** See the detailed trace
    below. It is a custom sample-weight column, not AutoGluon's built-in
    `balance_weight` string.

12. **Strongest recorded feature pipeline.** The strongest externally scored
    single AutoGluon submission in the structured submission-lab evidence is
    source artifact
    `logs/2-romantic-guan-of-eternity/artifacts/20260707T214845-49a4adb6-50/`
    (public 0.95019, local 0.950553137). Its 16-feature pipeline keeps the 13 raw
    covariates and adds categorical-profile frequency, within-profile numeric
    median deviation, and rounded full-covariate-template frequency
    (`solution.py:242-295`). Artifact evidence: `aide_result.json`, solution
    SHA-256 `80c78c9254d99e628a951fea623130e5b1bb1dc94ded447a7673e292506499f8`,
    all XGB/GBM/CAT trained and inferable. This pipeline is **transductive**:
    its frequencies and medians are learned from concatenated train+test
    covariates. Main must decide whether Stage C preserves this exact recorded
    behavior for fidelity or converts it to fold-fitted transforms (which would
    be a different pipeline). The separate 111-feature union at
    `logs/experiments/s6e7_top_boost_transforms/metrics.json` scored local
    0.950237054 but lacks equivalent external evidence and is not the audit's
    primary strongest-pipeline candidate.

13. **Neutral necessary-preprocessing pipeline.** `preprocess(df): return
    df.copy()` after the wrapper drops ID/target. It leaves the 7 numeric and 6
    categorical raw covariates (including missing values) to AutoGluon's native
    feature generator. This is the existing 13-feature neutral pipeline and
    requires no auxiliary data or learned preprocessing statistic.

## `class_balance: balanced` semantics and execution trace

1. The resolved profile exposes `class_balance` through
   `aide/autogluon_preprocess.py:559-569`.
2. The generated wrapper computes `N/(K*n_c)` with
   `_balanced_sample_weight` at `aide/autogluon_preprocess.py:698-705`.
3. In generated `main`, this mapping is computed from **all** `y_train` and
   attached before validation splitting (`aide/autogluon_preprocess.py:1297-1299`).
4. `TabularPredictor` is initialized with `sample_weight` set to the column name
   and `weight_evaluation=False` (`aide/autogluon_preprocess.py:1343-1353`).
5. AutoGluon 1.5.0 extracts the training column and normalizes it to mean one in
   `.venv/.../autogluon/tabular/trainer/abstract_trainer.py:4328-4346`.
6. XGB receives it as `XGBClassifier.fit(sample_weight=...)`
   (`.../models/xgboost/xgboost_model.py:79-80,200`); GBM passes it as the
   LightGBM Dataset `weight` (`.../models/lgb/lgb_model.py:169-216,470-521`);
   CAT passes it to `catboost.Pool(weight=...)`
   (`.../models/catboost/catboost_model.py:122-158`). Thus all three intended
   families receive the signal.
7. `weight_evaluation=False` means validation weights are ignored for metric
   evaluation and validation-weight arguments are not passed to base models.

Exact current full-data weights (already mean one over all rows):

| class | count | weight |
|---|---:|---:|
| `at-risk` | 592,561 | 0.388195195656 |
| `unhealthy` | 57,724 | 3.984986025454 |
| `fit` | 39,803 | 5.779195873008 |

Fold-safe weights from the frozen seed-1729 training partition would be
0.388194750613, 3.985000397006, and 5.779264284069 respectively. The numerical
difference is tiny because the split is stratified; the semantic violation is
still real. There is no resampling, no model-specific balancing parameter, and
no prediction-time prior correction.

## Leakage and fold-safety conclusions

- **Validation-label leakage: confirmed.** Full-label counts determine the
  mapping before the holdout split.
- **Test-label leakage: absent.** Test labels are unavailable and are not used
  in weight construction.
- **Bagging fold safety: failed.** A single full-training mapping is constructed
  before AutoGluon creates internal folds.
- **Feature transduction: confirmed for learned user transforms.** The wrapper
  deliberately concatenates train and test covariates before `preprocess`.
  Identity preprocessing is neutral; frequency/median pipelines use test
  covariates. This is separate from label leakage but affects how Stage C is
  described.
- **Index alignment:** the generated column stays aligned while splitting and
  AutoGluon extracts it from each frame. There is no observed positional
  misalignment, but future implementations should still assert index equality,
  positivity, and finiteness.

## Historical equivalence/reuse candidates

Two neutral-pipeline artifacts are an exceptionally close legacy pair:

- unweighted: `logs/2-smiling-topaz-oarfish/artifacts/20260710T002338`
- current balanced: `logs/2-smiling-topaz-oarfish/artifacts/20260710T010321`

Their generated `solution.py` files differ by exactly one inserted line,
`'class_balance': 'balanced'`; both recorded the same 13 features, 20% holdout,
seed 1729, XGB/GBM/CAT, model hyperparameters, 600-second budget, no ensemble,
and the stored validation target sequence has identical SHA-256
`150650daaa6967630a06605052d9c65c53bebd5d46a56346a2f78908891952a6`.
They provide strong evidence for the legacy observed gap (0.880162059 vs
0.949285091), including exact per-class diagnostics in `results.json`.

They are **not sufficient to skip Stage A** under the goal's strict reuse rule:
the artifacts do not record input-file hashes or the training-code Git revision,
and the balanced member has the fold-safety defect. They should be treated as
historical verification targets, not as fully equivalent reusable final runs.

## Exact source ranges requiring Main review

1. `aide/autogluon_preprocess.py:698-705` — weight formula and validation.
2. `aide/autogluon_preprocess.py:1245-1299` — data/feature path and weight
   calculation before split.
3. `aide/autogluon_preprocess.py:1303-1354` — split, fit kwargs, predictor
   weighting semantics.
4. `aide/autogluon_preprocess.py:1360-1396` — selected validation metric and
   prediction-artifact path.
5. `logs/experiments/s6e7_top_boost_transforms.py:93-114,588-647` — strongest
   union's transductive feature calculation and duplicate unsafe weighting path.
6. `.venv/lib/python3.12/site-packages/autogluon/tabular/trainer/abstract_trainer.py:4328-4346`
   and the three model ranges cited above — proof of family signal propagation.

## Corrected Stage A implementation addendum

- `aide/autogluon_preprocess.py:215-298` defines the centralized configuration,
  inverse-frequency formula, validation, and unsupported-context checks.
- `aide/autogluon_preprocess.py:1377-1419` resolves balancing, rejects unsafe
  validation contexts, splits first, derives weights from `train_data[target]`,
  preserves the split indices, and logs the exact normalized mapping.
- `aide/autogluon_preprocess.py:1438-1449` activates AutoGluon's sample-weight
  column only for a non-`none` method.
- Legacy `class_balance: balanced` resolves to corrected
  `inverse_frequency, alpha=1.0`; explicit `none` and mapping-based
  `inverse_frequency` are supported.
- Frozen CPU profiles are at `aide/utils/config.yaml:233-286`. A structured
  comparison confirms that their only resolved difference is `class_balance`.
- Focused weighting and CPU-profile tests are at
  `tests/test_autogluon_preprocess.py:423-548`.

## Stage A resource-policy revision

The first run-1 attempt failed before data loading because a custom logs root
changed source-workspace lookup. The corrected attempt reached AutoGluon but
all XGB/GBM/CAT configurations failed before fitting because zero GPUs were
detected. Both attempts are non-comparable infrastructure evidence and provide
no score.

Main revised the frozen resource control to CPU-only for both variants. Each
profile requests exactly one configuration per required family at priority 100;
XGB explicitly uses `device=cpu, tree_method=hist`; all families declare zero
GPU resources; CAT GPU task/device/RAM options and GBM/XGB CUDA options are
absent. Seed, split, features, preset, time limit, scheduling, validation,
ensemble, bagging, and stacking controls remain unchanged.
