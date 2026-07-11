# Stage A execution plan

Status: Main approved the protocol; commands were dry-run validated but have
**not** been executed with `--execute`.

Source: the neutral identity source artifact selected by full SHA-256 prefix
`f26e4d0a1755c73b`. Both variants use the same current data fingerprints,
frozen seed-1729 split, code revision, resource policy, model configurations,
and evaluation implementation. Their resolved profiles differ only in
`class_balance`.

Run sequentially. Each runner gets a distinct logs root; the runner creates a
timestamped artifact directory under that root. Shell output is redirected to
the variant's dedicated log and must not be streamed into model context.

## 1. No balancing

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_none
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_none --index logs/submission_index.json --sha256 f26e4d0a1755c73b --profile s6e7_class_balance_stage_a_none_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-a-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_none/run.log 2>&1
```

Expected artifact root:
`logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_none/2-smiling-topaz-oarfish/artifacts/<timestamp>/`.

## 2. Fold-safe inverse frequency, alpha 1.0

Run only after the first command terminates and its artifacts are verified.

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_inverse_frequency_alpha1
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_inverse_frequency_alpha1 --index logs/submission_index.json --sha256 f26e4d0a1755c73b --profile s6e7_class_balance_stage_a_inverse_frequency_alpha1_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-a-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_inverse_frequency_alpha1/run.log 2>&1
```

Expected artifact root:
`logs/class_balance/s6e7_class_balance_stage_a_20260711/stage_a_inverse_frequency_alpha1/2-smiling-topaz-oarfish/artifacts/<timestamp>/`.

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
