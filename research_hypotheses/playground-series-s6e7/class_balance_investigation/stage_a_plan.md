# Stage A execution plan

Status: the first run-1 attempt failed before data loading because of an invalid
custom logs root. Corrected attempt 2 reached AutoGluon, but the environment
exposed zero GPUs and no family fit succeeded. Both attempts are
non-comparable infrastructure failures. Main revised the frozen resource block
to CPU-only, identically for both variants. The commands below are the revised
CPU plan. The unweighted CPU run completed, but only CatBoost trained because it
consumed 596.7 seconds of the shared 600-second budget. The run is
non-comparable and the weighted CPU run was not launched. Main approved a
180-second maximum per family, giving at most 540 model-fit seconds and a
60-second overhead reserve. The capped commands below are verified but have not
been executed.

Source: the neutral identity source artifact selected by full SHA-256 prefix
`f26e4d0a1755c73b`. Both variants use the same current data fingerprints,
frozen seed-1729 split, code revision, resource policy, model configurations,
and evaluation implementation. Their resolved profiles differ only in
`class_balance`.

Run sequentially. The runner creates a collision-safe timestamped artifact
directory under the source run's normal artifact root. Shell output is
redirected to the investigation's dedicated attempt log and must not be
streamed into model context.

## 1. No balancing, CPU fair-one

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_none
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 f26e4d0a1755c73b --profile s6e7_class_balance_stage_a_none_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-a-cpu-capped180-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_none/run_cpu_capped180_attempt_4.log 2>&1
```

Expected artifact root:
`logs/2-smiling-topaz-oarfish/artifacts/<timestamp>/`. The runner loops until
`mkdir(..., exist_ok=False)` succeeds, so it cannot overwrite an existing
timestamped artifact. The new `run_cpu_capped180_attempt_4.log` is distinct from
the uncapped CAT-only run and both earlier failed attempts.

## 2. Fold-safe inverse frequency alpha 1.0, CPU fair-one

Run only after the CPU first command terminates, its artifacts are verified,
and Main separately authorizes run 2.

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_inverse_frequency_alpha1
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 f26e4d0a1755c73b --profile s6e7_class_balance_stage_a_inverse_frequency_alpha1_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-a-cpu-capped180-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_inverse_frequency_alpha1/run_cpu_capped180_attempt_1.log 2>&1
```

Expected artifact root:
`logs/2-smiling-topaz-oarfish/artifacts/<timestamp>/`.

## Post-run verification before interpretation

- exact data and split-ID hashes match the manifest;
- resolved configurations differ only in `class_balance`;
- XGB, GBM, and CAT all trained and are inferable;
- balanced run log contains one exact `AIDE_RUNTIME|class_weights` record;
- no run used bagging, stacking, or a weighted ensemble;
- structured validation predictions yield balanced accuracy, per-class recall,
  confusion matrix, and prediction counts/proportions;
- record family scores, fit/prediction times, warnings/failures, artifact hashes,
  and current Git revision in `results.json`.
