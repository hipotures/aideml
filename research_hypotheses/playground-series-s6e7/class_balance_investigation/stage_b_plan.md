# Stage B inverse-frequency execution plan

Status: **clipped inverse-frequency alpha `1.0`, pre-normalization cap `4.0`,
completed comparably at balanced accuracy `0.9491452350142495`**. Cap `3` and
all other methods remain unauthorized; no next method is selected.

The profile is frozen at repository revision
`5b436165c8677503beb064cb466625b869fdfb2c`. It uses the same source artifact,
seed-1729 stratified holdout, CPU fair-one XGB/GBM/CAT block, 180-second
per-family caps, 600-second predictor limit, and disabled stacking/ensembling
as the successful Stage A reference. Its resolved configuration differs from
the Stage A alpha-1 reference only in `class_balance` (method and cap).

The authorized execution used revision
`50f13f8572d86730e3293050154f5d85bae85f42`. Artifact:
`logs/2-smiling-topaz-oarfish/artifacts/20260711T135608`.

Profile:
`s6e7_class_balance_stage_b_inverse_frequency_alpha075_cpu_capped180_fairone_seed1729_10m`

Expected unique artifact root:
`logs/2-smiling-topaz-oarfish/artifacts/<timestamp>/`

Reserved outer log:
`logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha075/run_cpu_capped180_attempt_1.log`

## Reproduction command

Executed once after separate explicit training authorization.

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

All checks passed. XGB, GBM, and CAT trained and are inferable; LightGBM was
selected at balanced accuracy `0.9477688983566126`. The sole training-only
weight record contains `at-risk=0.5632106551439511`,
`unhealthy=3.2300176498031856`, and `fit=4.2686253090070965`. The result is
`0.0018112237374613427` below Stage A alpha `1.0`. Full diagnostics and hashes
are recorded in `results.json`.

## Alpha 0.50 execution

Main selected neutral raw-13 fold-safe inverse-frequency alpha `0.50` as the
only next experiment. Its CPU fair-one profile is identical to the successful
Stage A capped180 block except `class_balance.alpha`. It was committed and
executed after separate Main authorization at revision
`835e0cff25849e0e5887f49b6f72f56d963c0a71`.

Profile:
`s6e7_class_balance_stage_b_inverse_frequency_alpha050_cpu_capped180_fairone_seed1729_10m`

Expected unique artifact root:
`logs/2-smiling-topaz-oarfish/artifacts/20260711T152757/`

Reserved outer log:
`logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha050/run_cpu_capped180_attempt_1.log`

### Executed reproduction command

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha050
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 f26e4d0a1755c73b --profile s6e7_class_balance_stage_b_inverse_frequency_alpha050_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-b-alpha050-cpu-capped180-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha050/run_cpu_capped180_attempt_1.log 2>&1
```

Required post-run verification is identical to alpha `0.75`, with the exact
training-only weight record required to state alpha `0.50`. All checks passed:
XGB, GBM, and CAT trained and are inferable; XGBoost was selected at balanced
accuracy `0.943514732929528`; the sole training-only mapping is
`at-risk=0.7411670636696837`, `unhealthy=2.374682036696347`, and
`fit=2.859746651022219`. Full diagnostics and hashes are in `results.json`.
The completed alpha-0.50 command does not itself authorize any further run.

## Alpha 0.25 execution

Main selected neutral raw-13 fold-safe inverse-frequency alpha `0.25` as the
final required inverse-frequency screen. Its CPU fair-one profile is identical
to the successful Stage A capped180 block except `class_balance.alpha`. It was
committed and executed after separate Main authorization at revision
`8a64c262e962d7a43a44b44dfc545dc03542e09d`.

Profile:
`s6e7_class_balance_stage_b_inverse_frequency_alpha025_cpu_capped180_fairone_seed1729_10m`

Expected unique artifact root:
`logs/2-smiling-topaz-oarfish/artifacts/20260711T154503/`

Reserved outer log:
`logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha025/run_cpu_capped180_attempt_1.log`

### Executed reproduction command

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha025
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 f26e4d0a1755c73b --profile s6e7_class_balance_stage_b_inverse_frequency_alpha025_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-b-alpha025-cpu-capped180-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_inverse_frequency_alpha025/run_cpu_capped180_attempt_1.log 2>&1
```

Required post-run verification is identical to the earlier inverse-frequency
screens. All checks passed: XGB, GBM, and CAT trained and are inferable; XGBoost
was selected at balanced accuracy `0.9165515465098321`; the sole training-only
mapping is `at-risk=0.8915070763078106`, `unhealthy=1.595766972983883`, and
`fit=1.751178912691916`. Full diagnostics and hashes are in `results.json`.
This completed plan does not authorize any other Stage B method.

## Clipped inverse-frequency alpha 1.0, cap 4.0 preparation

Main selected one clipped candidate on neutral raw-13. The base weights are
computed from the holdout training partition only as `N / (K * n_c)`, raised to
alpha `1.0`, clipped at raw weight `4.0`, and then normalized to training-row
mean one. The CPU fair-one model block is otherwise identical to Stage A.

The run completed at committed revision `704c71d`. Artifact:
`logs/2-smiling-topaz-oarfish/artifacts/20260711T161037`.

For the frozen training counts, the expected normalized mapping is
`at-risk=0.4325884766651549`, `unhealthy=4.440722726231404`, and
`fit=4.4574376751058375`.

Profile:
`s6e7_class_balance_stage_b_clipped_inverse_frequency_alpha1_cap4_cpu_capped180_fairone_seed1729_10m`

Expected unique artifact root:
`logs/2-smiling-topaz-oarfish/artifacts/<timestamp>/`

Reserved outer log:
`logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_clipped_inverse_frequency_alpha1_cap4/run_cpu_capped180_attempt_1.log`

### Prepared reproduction command

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_clipped_inverse_frequency_alpha1_cap4
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 f26e4d0a1755c73b --profile s6e7_class_balance_stage_b_clipped_inverse_frequency_alpha1_cap4_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-b-clipped-alpha1-cap4-cpu-capped180-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_b_20260711/stage_b_neutral_clipped_inverse_frequency_alpha1_cap4/run_cpu_capped180_attempt_1.log 2>&1
```

Required post-run verification includes one exact training-only runtime record
with method `clipped_inverse_frequency`, alpha `1.0`, cap `4.0`, and the
normalized mapping; frozen data/split/source hashes; all three required model
families; and the standard structured diagnostics. This preparation does not
authorize execution, cap `3`, or any other method.
