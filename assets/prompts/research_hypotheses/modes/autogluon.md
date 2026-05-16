Allowed implementation scope:
- The AIDE coding agent will later implement a Python feature procedure named
  `preprocess(df)`. AIDE will wrap that procedure in its own AutoGluon training
  script.
- The generated hypotheses must be feature/preprocessing hypotheses that can
  realistically be tested inside a single `preprocess(df: pd.DataFrame) ->
  pd.DataFrame` procedure and then handed to AutoGluon.

Do not propose AutoGluon optimizer/configuration hypotheses. Exclude ideas about
presets, time_limit, included_model_types, hyperparameters,
validation_strategy, bagging, stacking, weighted ensembles, refit_full, final
full-train refits, or changing AutoGluon fit arguments. Those are out of scope
even if they might improve score.

Example of the kind of implementation AutoGluon mode can build later:

{{AUTOGLUON_PREPROCESS_EXAMPLE}}

If no task-specific example is provided, infer the allowed shape from the rule
above: a single target-free `preprocess(df)` function that adds safe tabular
features from the available input columns.

Good hypotheses for this scope:
- target-free tabular feature blocks
- causal cumulative features using current and past rows only
- safe categorical/frequency/statistical encodings that do not use the target
- domain-specific feature families grounded in the task context and available input columns
- robust transformations, interactions, clipping, missingness flags
- features that AutoGluon can consume directly after preprocessing

Bad hypotheses for this scope:
- manual LightGBM/CatBoost/XGBoost training
- custom cross-validation loops
- out-of-fold target encoding
- target-derived encodings
- custom rank averaging of multiple trained predictors
- calibration pipelines
- AutoGluon preset, model-list, hyperparameter, time-limit, or fit-argument tuning
- final refit/bagging/stacking experiments that require changing AutoGluon fit logic
- instructions like "start from node X" or "copy step Y"

Scope-specific quality rules:
- Every `implementation_hint` should read like something that can become a
  `preprocess(df)` feature block.
- Do not recommend manual model training or custom validation.
- Do not recommend tuning AutoGluon itself. Only propose feature/preprocess
  hypotheses.
- Remove any hypothesis requiring custom CV, manual model training, OOF target
  encoding, or custom ensembling.
- Remove any hypothesis about AutoGluon presets, fit settings, bagging,
  stacking, final refit, model lists, or hyperparameter optimization.
