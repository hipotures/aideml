# Class-balance investigation status

State: **CAT-only CPU run remains non-comparable; Main approved and configured
an equal 180-second per-family cap with 60 seconds reserved overhead; capped
profiles are verified but unexecuted; weighted run remains unlaunched**.

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

Pending Main review:

1. review `aide/autogluon_preprocess.py:215-298,1377-1449`;
2. review the renamed CPU profiles at `aide/utils/config.yaml:233-286`;
3. review and separately authorize the capped180 CPU unweighted rerun.

Next Luna action after Main review: launch no command unless explicitly
authorized. Weighted run remains unlaunched.
