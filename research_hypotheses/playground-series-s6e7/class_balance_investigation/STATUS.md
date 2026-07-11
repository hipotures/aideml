# Class-balance investigation status

State: **partial random oversampling ratio `0.25` is prepared only at base
revision `5256f07`; no ratio-0.25 training has been authorized or launched**.

The profile
`s6e7_class_balance_stage_b_partial_random_oversample_ratio025_cpu_capped180_fairone_seed1729_10m`
preserves the frozen CPU capped-180 reference outside `class_balance`. After
the seed-1729 stratified holdout, it resamples only training rows from at-risk
`474049`, unhealthy `46179`, and fit `31842` to at-risk `474049`, unhealthy
`118513`, and fit `118513` (total `711075`, added `159005`). Validation is not
resampled and no sample weights are configured for this method.

Prepared command (do not execute until separately authorized from a committed
revision):

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_partial_random_oversample_ratio025
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 f26e4d0a1755c73b --profile s6e7_class_balance_stage_b_partial_random_oversample_ratio025_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-b-partial-random-oversample-ratio025-cpu-capped180-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_partial_random_oversample_ratio025/run_cpu_capped180_attempt_1.log 2>&1
```

Reserved outer log:
`logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_partial_random_oversample_ratio025/run_cpu_capped180_attempt_1.log`.
Expected single runtime record: `AIDE_RUNTIME|class_resampling` with method
`partial_random_oversample`, ratio `0.25`, seed `1729`, the exact before/after
counts above, and `added=159005`. Ratio `.15` is not yet authorized.

Effective-number beta `0.999995` is closed: its comparable execution completed
at balanced accuracy `0.9463823272415373` (artifact `20260711T163831`). Beta
`.99999`, cap 3, and all other methods remain unauthorized.

Fixed held-out probability capture is prepared only, not authorized or run. It
uses exactly one rerun of the frozen unweighted profile
`s6e7_class_balance_stage_a_none_cpu_capped180_fairone_seed1729_10m`, exporting
the single seed-1729 holdout probabilities rather than CV OOF predictions.
The label-free prior-power sweep evaluates only tau `0`, `.25`, `.5`, `.75`,
and `1` after capture. Its priors are frozen from the post-split training
counts only: at-risk `474049`, unhealthy `46179`, and fit `31842`; labels are
used only for separate scoring diagnostics.

Completed:

- Stage C prepared only from the strongest 16-feature transductive source
  (selector `feff2f3363b23b0b2a6bde1516db26e44427f197461b7d69265f9f4a92cd2edb`);
  three profiles are unexecuted and training remains unauthorized.

- Fixed held-out probability capture completed once (artifact `20260711T173849`);
  per-family prior-power sweep (`tau=0,.25,.5,.75,1`) is recorded in
  `heldout_prior_power_sweep.json`. This is not full-CV OOF; no next training is authorized.

- Read-only prior-power boundary extension completed at revision `12e00f4` with
  no training. Tau `1.5` declines for every family versus its best tau; global
  best remains GBM at tau `1.0` (`0.9487465298656267`). No next training is authorized.

- Partial random oversampling ratio `.25` executed once at artifact
  `20260711T171117`, balanced accuracy `0.9413978706130188`; ratio `.15` and
  other methods remain not yet authorized.

- Effective-number beta `0.999995` completed comparably at balanced accuracy
  `0.9463823272415373` (artifact `20260711T163831`); beta `.99999` and any next
  method remain unauthorized.

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
- at the preparation gate, the profile and research changes were uncommitted on
  base revision `8b58b1f`, so no alpha-0.25 log or artifact existed then;
- the alpha-0.25 profile was committed and separately authorized at revision
  `8a64c26`; it completed in 215.05 seconds with XGB, GBM, and CAT trained and
  inferable, no failed/skipped family, and XGBoost selected;
- the resolved profile matches the Stage A block after removing
  `class_balance`; data, frozen split, source, and code hashes match;
- exactly one training-only weight record reports `at-risk=0.8915070763`,
  `unhealthy=1.5957669730`, and `fit=1.7511789127`, with training mean one;
- recall is `0.973639800` (at-risk), `0.878388913` (unhealthy), and
  `0.897625926` (fit). Full confusion, prediction distribution, family
  scores/times, runtime, warnings, and artifact hashes are in `results.json`;
- alpha `0.25` is `0.026963186` below alpha `0.50`, `0.031217352` below alpha
  `0.75`, `0.033028576` below alpha `1.0`, and `0.036193329` above unweighted.
  No other Stage B method was launched.
- Main selected only clipped inverse-frequency alpha `1.0`, raw cap `4.0` next;
  cap `3` and all other balancing methods were not prepared;
- the centralized parser and weight helper now implement base inverse-frequency
  weights, exponentiation, pre-normalization clipping, and training-mean-one
  normalization while preserving legacy none/inverse behavior and indices;
- finite positive cap validation, training-only derivation, exact method/alpha/
  cap/mapping logging, and the existing bagging/internal-validation rejection
  are covered by focused tests;
- the cap-4 profile preserves the Stage A CPU capped180 block exactly outside
  `class_balance`, with unique session, log, artifact, and reproduction IDs;
- generated-wrapper compilation, calibration validation, one-source dry-run,
  focused balancing tests (`18 passed`), the complete preprocessing/runner test
  files, ruff, JSON, and diff checks passed;
- cap-4 implementation, profile, result, and research records are committed at
  `f704d31`; its comparable artifact is `20260711T161037` and no further
  clipped run is pending;
- effective-number beta `0.999995` completed comparably and is closed; its
  artifact is `20260711T163831`. Beta `.99999` remains unauthorized.

Pending Main review:

1. review the partial-random-oversample ratio-0.25 implementation, profile,
   and verification evidence;
2. commit the ratio-0.25 preparation when commits are possible;
3. only then decide whether to authorize its exact prepared command.

Next Luna action after Main review: launch no command unless explicitly
authorized from a committed revision. Partial random oversampling ratio `.25`
remains unexecuted; ratio `.15` and effective-number beta `.99999` remain
unauthorized.
