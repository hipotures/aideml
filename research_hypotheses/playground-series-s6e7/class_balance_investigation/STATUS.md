# Class-balance investigation status

State: **Stage A implementation and exact run plan complete; awaiting Main code
review; no training launched**.

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

Pending Main review:

1. review `aide/autogluon_preprocess.py:215-298,1377-1449`;
2. review the frozen profiles at `aide/utils/config.yaml:233-294`;
3. authorize sequential execution of the two commands in `stage_a_plan.md`.

Next Luna action after approval: run Stage A sequentially with redirected
per-run logs, verify required-family completion and artifact hashes, then append
the structured diagnostics to `results.json`.
