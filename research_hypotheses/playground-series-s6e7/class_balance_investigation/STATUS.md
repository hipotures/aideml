# Class-balance investigation status

State: **Stage A resource policy revised to a frozen CPU-only block; both CPU
commands are dry-run pending; no CPU training launched; weighted run remains
unlaunched**.

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

Pending Main review:

1. review `aide/autogluon_preprocess.py:215-298,1377-1449`;
2. review the renamed CPU profiles at `aide/utils/config.yaml:233-286`;
3. authorize only the CPU run-1 command after dry-run verification.

Next Luna action after Main review: launch no command unless explicitly
authorized. Weighted run remains unlaunched.
