# Stage C preparation

Use source selector SHA `feff2f3363b23b0b2a6bde1516db26e44427f197461b7d69265f9f4a92cd2edb` for the exact 16-feature transductive pipeline.

Run these existing frozen profiles sequentially:

```sh
mkdir -p logs/class_balance/s6e7_class_balance_stage_c_20260711/none
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 feff2f3363b23b0b --profile s6e7_class_balance_stage_a_none_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-c-none-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_c_20260711/none/run.log 2>&1

mkdir -p logs/class_balance/s6e7_class_balance_stage_c_20260711/inverse_alpha1
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 feff2f3363b23b0b --profile s6e7_class_balance_stage_a_inverse_frequency_alpha1_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-c-alpha1-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_c_20260711/inverse_alpha1/run.log 2>&1

mkdir -p logs/class_balance/s6e7_class_balance_stage_c_20260711/clipped_cap4
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --logs-dir logs --index logs/submission_index.json --sha256 feff2f3363b23b0b --profile s6e7_class_balance_stage_b_clipped_inverse_frequency_alpha1_cap4_cpu_capped180_fairone_seed1729_10m --profile-calibration --profile-calibration-session-id s6e7-class-balance-stage-c-cap4-20260711 --timeout 4200 --memory-limit-gb 80 --execute --force > logs/class_balance/s6e7_class_balance_stage_c_20260711/clipped_cap4/run.log 2>&1
```

Prior-power is read-only on the `none` probabilities; no fourth training run.
