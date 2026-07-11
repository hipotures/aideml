# Class-balance investigation status

State: **Stage B alpha `0.50` completed comparably at `0.943514733`. Main
selected alpha `0.25` as the final inverse-frequency screen; its profile is
frozen but uncommitted and unexecuted, so Main cannot launch it yet**.

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
- Main selected only neutral raw-13 inverse-frequency alpha `0.75` for Stage B;
  alpha `0.25` and `0.5` profiles were not created;
- the Stage B profile is frozen at revision `5b43616` and resolves identically
  to the successful Stage A alpha-1 CPU fair-one block after removing
  `class_balance`, with one XGB/GBM/CAT configuration each, 180-second family
  caps, a 600-second predictor limit, and 60 seconds reserved overhead;
- `stage_b_configs.json` and `stage_b_plan.md` record the exact configuration,
  unique session/log/artifact identifiers, and reproduction command;
- generated-wrapper compilation and dry-run source selection passed; focused
  invariants passed (`10 passed`), ruff passed, and JSON/diff checks passed. No
  training had been launched at the freeze gate.
- Main authorized only the frozen alpha-0.75 command at revision `50f13f8`; it
  completed in 225.07 seconds with XGB, GBM, and CAT trained and inferable, no
  failed/skipped family, and LightGBM selected;
- the resolved profile is identical to the successful Stage A CPU block after
  removing `class_balance`; data, split, source, and code hashes match;
- exactly one training-only weight record reports `at-risk=0.5632106551`,
  `unhealthy=3.2300176498`, and `fit=4.2686253090`, with training mean one;
- recall is `0.944781963` (at-risk), `0.951407536` (unhealthy), and
  `0.947117196` (fit); the exact confusion matrix, prediction distribution,
  family scores/times, runtime, warnings, and artifact hashes are in
  `results.json`;
- alpha `0.75` is `0.001811224` below alpha `1.0` balanced accuracy and
  `0.067410680` above the unweighted Stage A run. No other Stage B experiment
  was launched.
- Main selected only neutral raw-13 inverse-frequency alpha `0.50` next; a
  single CPU capped180 fair-one profile and unique reproduction identifiers are
  prepared, while alpha `0.25` was not created;
- the alpha-0.50 profile resolves identically to the Stage A reference after
  removing `class_balance`, with XGB/GBM/CAT present, 180-second family caps,
  and the 600-second predictor limit;
- generated-wrapper compilation, profile-calibration validation, the one-source
  dry-run, focused invariants (`10 passed`), ruff, JSON, and diff checks passed
  for alpha `0.50`;
- at the preparation gate, the profile and research updates were uncommitted on
  base revision `50f13f8`, so no training log or artifact was created then;
- the alpha-0.50 profile was committed and separately authorized at revision
  `835e0cf`; it completed in 213.05 seconds with XGB, GBM, and CAT trained and
  inferable, no failed/skipped family, and XGBoost selected;
- the resolved profile matches the Stage A block after removing
  `class_balance`; data, frozen split, source, and code hashes match;
- exactly one training-only weight record reports `at-risk=0.7411670637`,
  `unhealthy=2.3746820367`, and `fit=2.8597466510`, with training mean one;
- recall is `0.951076684` (at-risk), `0.942399307` (unhealthy), and
  `0.937068208` (fit). Full confusion, prediction distribution, family
  scores/times, runtime, warnings, and artifact hashes are in `results.json`;
- alpha `0.50` is `0.004254165` below alpha `0.75`, `0.006065389` below alpha
  `1.0`, and `0.063156515` above the unweighted Stage A run. No alpha `0.25` or
  other balancing method was launched.
- Main selected only neutral raw-13 inverse-frequency alpha `0.25` as the final
  required inverse-frequency screen; no other balancing method was prepared;
- the alpha-0.25 profile resolves identically to the Stage A reference after
  removing `class_balance`, with XGB/GBM/CAT present, 180-second family caps,
  and the 600-second predictor limit;
- the exact profile, session, log path, artifact root, and reproduction command
  are frozen in the Stage B config and plan;
- generated-wrapper compilation, profile-calibration validation, the one-source
  dry-run, focused invariants (`10 passed`), ruff, JSON, and diff checks passed
  for alpha `0.25`;
- the profile and research changes are uncommitted on base revision `8b58b1f`.
  Main cannot authorize launch until commits are possible and the resulting
  revision is recorded. No alpha-0.25 training log or artifact exists.

Pending Main review:

1. review the frozen alpha-0.25 configuration and verification evidence;
2. commit the profile and research state when commits are possible;
3. only then decide whether to authorize the exact prepared command.

Next Luna action after Main review: launch no command unless explicitly
authorized from a committed revision. Alpha `0.25` remains unexecuted.
