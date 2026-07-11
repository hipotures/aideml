# Class-balance investigation status

State: **capped180 CPU Stage A pair completed comparably. Fold-safe inverse
frequency alpha 1 improves balanced accuracy from `0.880358218` to
`0.949580122` (`+0.069221904`); no Stage B run is authorized or launched**.

Completed:

- all 13 mandatory audit items inspected and recorded;
- current data/class counts and SHA-256 fingerprints captured;
- `class_balance: balanced` traced through AutoGluon 1.5.0 to XGB/GBM/CAT;
- fold-safety defect confirmed (weights computed before holdout/fold creation);
- strongest recorded public feature pipeline and neutral raw-13 pipeline identified;
- closest historical Stage A pair extracted with per-class diagnostics;
- proposed Stage A constants and required outputs written to the manifest.
- Main approved the Stage A constants and exact transductive Stage C behavior;
- centralized fold-safe holdout weighting implemented with legacy alias support;
- unsafe bagged/internal-validation custom weighting rejected explicitly;
- exact Stage A profiles and dry-run-validated reproduction commands generated;
- focused AutoGluon preprocessing tests passed.
- run 1 was authorized and launched at revision `53916cd`, but the runner
  failed before training because the custom `--logs-dir` changed where it
  searched for the existing source workspace input;
- the failed attempt produced only a unique generated `solution.py` and logs,
  with no model/result/submission artifacts.
- corrected attempt 2 resolved the source workspace and preserved the exact
  approved configuration, but AutoGluon detected zero GPUs and rejected all
  three required one-GPU model configurations before fitting;
- attempt 2 produced structured error artifacts but no metrics, predictions,
  trained models, or submission.
- both failed attempts are retained as non-comparable infrastructure evidence;
- Main revised Stage A to CPU-only: one CPU-compatible configuration each for
  XGB, GBM, and CAT, identical resources/settings across both variants;
- GPU-only model parameters were removed and profile names now state `cpu` and
  `fairone` explicitly.
- CPU unweighted run produced valid CAT predictions and a balanced accuracy of
  `0.873304239`, but CatBoost used 596.7 of 600 seconds and no XGB/GBM model was
  trained;
- the run is marked invalid/non-comparable under the mandatory-family rule.
- Main approved `ag_args_fit.max_time_limit=180` identically for XGB, GBM, and
  CAT in both profiles; the 540-second model allocation preserves the 600-second
  predictor total and reserves 60 seconds for overhead;
- capped profile identifiers are versioned separately from the invalid
  uncapped CAT-only run.
- Main authorized only the capped180 CPU unweighted run at revision `fab529a`;
  it completed in 488.10 seconds with all three mandatory families trained and
  inferable, no failed/skipped family, and XGB selected;
- structured validation predictions reproduce balanced accuracy
  `0.880358218`, with recall `0.989148778` (at-risk), `0.823386748`
  (unhealthy), and `0.828539128` (fit);
- current input SHA-256 values, frozen train/validation ID hashes, validation
  target-sequence hash, resolved profile, source immutability, and all artifact
  hashes were verified and recorded in `results.json`;
- the run is valid as Stage A run 1. The expected non-bagged per-model
  prediction-export warning does not affect the selected-model validation
  predictions or family training/inference checks.
- Main separately authorized the capped180 fold-safe inverse-frequency alpha-1
  run at the same `fab529a` training revision; it completed in 227.06 seconds
  with XGB, GBM, and CAT trained and inferable, no failed/skipped family, and
  LightGBM selected;
- the weighted run's resolved configuration is byte-for-structure equal to run
  1 after removing `class_balance`; data, split, source, and code hashes match;
- the sole exact class-weight log record reports training-only weights
  `at-risk=0.3881947506`, `unhealthy=3.9850003970`, and `fit=5.7792642841`,
  whose training-partition mean is exactly one;
- weighted recall is `0.934572026` (at-risk), `0.963533997` (unhealthy), and
  `0.950634342` (fit), changing recall by `-0.054576752`, `+0.140147250`, and
  `+0.122095214`, respectively, versus the capped unweighted run;
- all weighted aggregate, family, per-class, confusion, prediction-distribution,
  runtime, warning, configuration, and artifact-hash diagnostics are recorded
  in `results.json`.

Pending Main review:

1. interpret the completed capped180 Stage A pair in `results.json`;
2. decide whether any later-stage investigation should be designed and
   separately authorized.

Next Luna action after Main review: launch no command unless explicitly
authorized. No Stage B run was launched.
