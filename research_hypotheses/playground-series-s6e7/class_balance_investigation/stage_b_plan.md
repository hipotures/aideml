# Stage B alpha-0.75 execution plan

Status: **frozen and verified, not executed**. Main selected exactly one Stage B
experiment: neutral raw-13 preprocessing with fold-safe inverse-frequency
balancing at alpha `0.75`. This plan does not define alpha `0.25` or `0.5` and
does not authorize training.

The profile is frozen at repository revision
`5b436165c8677503beb064cb466625b869fdfb2c`. It uses the same source artifact,
seed-1729 stratified holdout, CPU fair-one XGB/GBM/CAT block, 180-second
per-family caps, 600-second predictor limit, and disabled stacking/ensembling
as the successful Stage A reference. Its resolved configuration differs from
the Stage A alpha-1 reference only in `class_balance.alpha`.

Profile:
`s6e7_class_balance_stage_b_inverse_frequency_alpha075_cpu_capped180_fairone_seed1729_10m`

Expected unique artifact root:
`logs/2-smiling-topaz-oarfish/artifacts/<timestamp>/`

Reserved outer log:
`logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha075/run_cpu_capped180_attempt_1.log`

## Reproduction command

Run only after separate explicit training authorization.

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha075
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 f26e4d0a1755c73b --profile s6e7_class_balance_stage_b_inverse_frequency_alpha075_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-b-alpha075-cpu-capped180-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha075/run_cpu_capped180_attempt_1.log 2>&1
```

## Required post-run verification

- data, frozen split-ID, source, code, configuration, and artifact hashes;
- exact equality with the Stage A alpha-1 reference after removing
  `class_balance`;
- exactly one training-only `AIDE_RUNTIME|class_weights` record for alpha
  `0.75`;
- XGB, GBM, and CAT all trained and inferable;
- aggregate and family balanced accuracy, per-class recall, confusion matrix,
  prediction counts/proportions, fit/prediction/runtime, warnings, and failures;
- no bagging, stacking, weighted ensemble, or additional Stage B experiment.
