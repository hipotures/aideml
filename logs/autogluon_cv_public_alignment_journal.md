# AutoGluon CV/Public Alignment Journal

TASK_START_TIME=2026-07-07T23:51:35+02:00
TASK_TIME_BUDGET_HOURS=12
TASK_DEADLINE=2026-07-08T11:51:35+02:00

## 2026-07-07T23:51:35+02:00 - Baseline and setup

Goal: identify an AutoGluon profile whose local CV score best agrees with known Kaggle public balanced accuracy, without making new Kaggle submissions.

Commands executed:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json
uv run python scripts/kaggle_submission_lab.py --output-format json > /tmp/aideml_kaggle_submission_lab.json
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 > /tmp/aideml_kaggle_submission_lab_full.json
uv run pytest tests/test_analyze_autogluon_alignment.py -v
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
uv run python -c '
from aide.utils.config import _load_cfg
from aide.autogluon_preprocess import resolve_autogluon_settings
profiles=["s6e7_align_holdout_balanced_gpu_10m", "s6e7_align_agval_balanced_gpu_10m"]
for profile in profiles:
    cfg=_load_cfg(use_cli_args=False)
    cfg.agent.autogluon.profile=profile
    settings=resolve_autogluon_settings(cfg)
    print(profile, settings["time_limit"], settings["validation_strategy"], settings["class_balance"], settings["use_gpu"], settings["fit_args"])
'
```

Notes:

- The exact required lab command completed and synchronized 60 Kaggle submissions; 70 remote submissions were visible.
- The default JSON registry view was limited to 20 rows, so a full registry JSON was captured with `--registry-limit 0` for analysis.
- Baseline usable rows: 66 (`algo == "AG"`, `remote_status == "COMPLETE"`, numeric local/public scores, `eval_metric == "balanced_accuracy"`).
- Baseline agreement across submitted AG rows: Pearson 0.587971, Spearman 0.393343, MAE 0.000570, signed bias +0.000559.
- Existing profile-eval evidence from `logs/submission_index.json`: `best_boost_gpu_1h` has 4 usable observations with MAE 0.000145 and bias -0.000020, but it is a 3600-second profile and has weak rank agreement on this small sample.
- Added profiles in `aide/utils/config.yaml`:
  - `s6e7_align_holdout_balanced_gpu_10m`: 600s, medium_quality, XGB/GBM/CAT GPU, balanced sample weights, explicit holdout, weighted ensemble enabled.
  - `s6e7_align_agval_balanced_gpu_10m`: 600s, medium_quality, XGB/GBM/CAT GPU, balanced sample weights, AutoGluon-managed validation, weighted ensemble enabled.

Representative source hashes selected for first reruns:

| reason | source sha | public | original local | run | step |
|---|---:|---:|---:|---|---:|
| top public | `4d2b8df165` | 0.95016 | 0.950537 | `2-vociferous-tortoise-of-perspective` | 13 |
| top public, existing 1h comparison | `1070897a05` | 0.95008 | 0.950564 | `2-vociferous-tortoise-of-perspective` | 55 |
| large local/public disagreement | `9f5a6e6e5d` | 0.94925 | 0.950374 | `2-smiling-topaz-oarfish` | 21 |
| high local but weak public | `cdc4cd52a1` | 0.94972 | 0.950670 | `2-whimsical-albatross-from-camelot` | 47 |

Initial rerun plan: run `s6e7_align_holdout_balanced_gpu_10m` first on the four selected hashes, one command at a time. If those results are usable, run the AutoGluon-validation variant on the same set only if the holdout variant does not clearly dominate existing 10-minute evidence.

## 2026-07-07T23:53:50+02:00 - Rerun 1

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --sha 4d2b8df165 --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.950530891138 |
| local - public | +0.000370891138 |
| absolute error | 0.000370891138 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 57.0187s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260707T235234` |

Notes:

- The rerun script defaulted the metadata competition field to `playground-series-s6e6`; future reruns should pass `--competition playground-series-s6e7`. The generated wrapper still used the new profile settings and the project metric from `.env`.

## 2026-07-07T23:55:16+02:00 - Rerun 2

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.950563322301 |
| local - public | +0.000483322301 |
| absolute error | 0.000483322301 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 59.0208s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260707T235408` |

## 2026-07-07T23:56:46+02:00 - Rerun 3

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.950300457417 |
| local - public | +0.001050457417 |
| absolute error | 0.001050457417 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 67.0207s |
| status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260707T235533` |

## 2026-07-07T23:58:08+02:00 - Rerun 4

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.950584075898 |
| local - public | +0.000864075898 |
| absolute error | 0.000864075898 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 58.0228s |
| status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260707T235704` |

## 2026-07-07T23:59:24+02:00 - Rerun 5

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_align_agval_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_agval_balanced_gpu_10m` |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.956098626189 |
| local - public | +0.005938626189 |
| absolute error | 0.005938626189 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 26.0153s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260707T235850` |

Notes:

- This profile's first result is suspiciously over-optimistic. Continue a small number of serial reruns to confirm whether the profile is consistently miscalibrated.

## 2026-07-08T00:00:27+02:00 - Rerun 6

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_align_agval_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_agval_balanced_gpu_10m` |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.956645911375 |
| local - public | +0.006565911375 |
| absolute error | 0.006565911375 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 30.0190s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260707T235948` |

## 2026-07-08T00:01:17+02:00 - Rerun 7

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_align_agval_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_agval_balanced_gpu_10m` |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.956048595407 |
| local - public | +0.006798595407 |
| absolute error | 0.006798595407 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 26.0171s |
| status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T000045` |

## 2026-07-08T00:02:10+02:00 - Rerun 8

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_align_agval_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_agval_balanced_gpu_10m` |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.956492371267 |
| local - public | +0.006772371267 |
| absolute error | 0.006772371267 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 27.0176s |
| status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T000134` |

## 2026-07-08T00:03:17+02:00 - Final comparison

Commands executed:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
uv run pytest tests/test_analyze_autogluon_alignment.py tests/test_autogluon_preprocess.py::test_autogluon_gpu_profiles_use_cuda_gbm tests/test_rerun_autogluon_profile.py::test_main_allows_same_source_with_different_profile_without_force -v
```

Profile-level comparison from `logs/autogluon_cv_public_alignment_analysis.json`:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |
| `s6e7_align_agval_balanced_gpu_10m` | 4 | 0.4021675 | 0.4000000 | 0.0065189 | +0.0065189 | 27.3s |
| `full_boost` | 1 | n/a | n/a | 0.0006087 | +0.0006087 | 609.1s |
| `best_full_1h` | 1 | n/a | n/a | 0.0018547 | -0.0018547 | 3711.7s |

Conclusion:

- Best practical agreement found: `best_boost_gpu_1h`, because its MAE and signed bias are much lower than the new 10-minute profiles and the baseline submitted-row bias. The rank correlations are not reliable at n=4 and are weak for this profile, so the conclusion is based mainly on calibration error and bias.
- Best 10-minute profile tested in this run: `s6e7_align_holdout_balanced_gpu_10m`. It has the strongest Pearson on the selected rerun set, but its positive bias and MAE are worse than the 66-row baseline and far worse than `best_boost_gpu_1h`.
- Rejected profile: `s6e7_align_agval_balanced_gpu_10m`. It consistently reports local scores around 0.956 on sources with public scores around 0.949-0.950, so it is not a usable public-score proxy.

Limitations:

- The profile-rerun sample size is small. Correlation metrics over four observations are unstable, especially because the rerun subsets are not identical across old 1-hour and new 10-minute profiles.
- `best_boost_gpu_1h` is an existing 3600-second profile, so it is not a cheap initial profile. It is the best aligned profile found, not the cheapest aligned profile.
- The new 10-minute holdout profile was intentionally tested on difficult/high-value sources, not a random sample.

Recommended next experiment:

- Create a <=30-minute bagged profile modeled after `best_boost_gpu_1h`, but cheaper: `presets: best`, `validation_strategy: autogluon`, balanced sample weights, GPU boosting, and explicit `fit_args` with `num_bag_folds: 3`, `num_bag_sets: 1`, `num_stack_levels: 0`, `auto_stack: false`. Rerun it on the same four hashes plus one additional high-public source. This directly tests whether the likely calibration source is bagged/OOF validation rather than the 1-hour time budget alone.

## 2026-07-08T00:25:23+02:00 - Continue experiment loop / Rerun 9 planned

Budget check:

| field | value |
|---|---:|
| task start | `2026-07-07T23:51:35+02:00` |
| deadline | `2026-07-08T11:51:35+02:00` |
| elapsed | 0:33:48 |
| estimated remaining | 11:26:11 |
| 12-hour budget reached? | no |
| enough time for <=30m AutoGluon plus preprocessing? | yes |

Profile added for this continuation:

| field | value |
|---|---|
| profile | `s6e7_align_bag3_best_gpu_30m` |
| status | newly created |
| AutoGluon time limit | 1800s |
| preprocessing timeout | 600s |
| presets | `best` |
| validation strategy | `autogluon` |
| class balancing | `balanced` |
| model types | `XGB`, `GBM`, `CAT` |
| GPU settings | CUDA XGB/GBM and GPU CatBoost |
| bag/stack settings | `num_bag_folds=3`, `num_bag_sets=1`, `num_stack_levels=0`, `auto_stack=false` |
| ensemble behavior | `fit_weighted_ensemble=true` |

Intention:

- The two 10-minute profiles produced usable artifacts, but did not produce a strong practical alignment candidate: the holdout profile retained material positive bias, and the AutoGluon-validation 10-minute profile was severely over-optimistic. This 30-minute controlled variant tests whether the stronger calibration seen historically for `best_boost_gpu_1h` comes from bagged/OOF validation and best presets rather than simply the full 1-hour runtime.

Verification before expensive command:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_autogluon_s6e7_align_bag3_best_gpu_30m_profile -v
pgrep -af '[r]erun_autogluon_profile'
```

Result:

- Profile resolver test passed.
- No active `rerun_autogluon_profile` process was visible.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_align_bag3_best_gpu_30m --timeout 3600 --execute
```

Planned source:

| field | value |
|---|---:|
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| run | `2-vociferous-tortoise-of-perspective` |
| step | 13 |

Decision for next experiment:

- If this rerun is successful and runtime is compatible with the remaining budget, recompute profile-level agreement and continue this profile across the same representative SHA set.

## 2026-07-08T00:30:03+02:00 - Rerun 9 result / Rerun 10 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_align_bag3_best_gpu_30m --timeout 3600 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_bag3_best_gpu_30m` |
| profile status | newly created |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.949610439199 |
| local - public | -0.000549560801 |
| absolute error | 0.000549560801 |
| time limit | 1800s |
| process timeout | 3600s |
| exec time | 122.0279s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T002553` |
| submission sha | `cc17ff8df1cc2a29737fc4dd761d177fd7123d5003fb2f160244000feac312a9` |

Preprocessing/runtime notes:

- AutoGluon completed in about 2 minutes despite the 30-minute time limit.
- The log reported sequential fold fitting because Ray is unavailable.
- The profile produced bagged base models and `WeightedEnsemble_L2`.
- The rerun command itself does not append to `logs/submission_index.json`; I refreshed the local index with `--no-remote` and regenerated `logs/autogluon_cv_public_alignment_analysis.json`.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 9:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_bag3_best_gpu_30m` | 1 | n/a | n/a | 0.0005496 | -0.0005496 | 122.0s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |
| `s6e7_align_agval_balanced_gpu_10m` | 4 | 0.4021675 | 0.4000000 | 0.0065189 | +0.0065189 | 27.3s |

Current best profile choice changed?

- No. `best_boost_gpu_1h` remains the best practical agreement candidate on current evidence. The new 30-minute profile has only one observation, so correlation is not meaningful. Its first error is better than the 10-minute holdout profile average but worse than the existing 1-hour profile MAE.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:30:03+02:00` |
| elapsed | 0:38:28 |
| estimated remaining | 11:21:31 |
| 12-hour budget reached? | no |
| enough time for another <=30m AutoGluon rerun? | yes |

Decision for next experiment:

- Continue `s6e7_align_bag3_best_gpu_30m` on the same representative SHA set so the profile has enough observations for rank/error evidence. Next source is the second high-public row with existing 1-hour comparison.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_align_bag3_best_gpu_30m --timeout 3600 --execute
```

Planned source:

| field | value |
|---|---:|
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| run | `2-vociferous-tortoise-of-perspective` |
| step | 55 |

## 2026-07-08T00:46:46+02:00 - Rerun 14 result / Rerun 15 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_align_holdout_best_gpu_30m --timeout 3600 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_best_gpu_30m` |
| profile status | newly created |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.950346747889 |
| local - public | +0.000266747889 |
| absolute error | 0.000266747889 |
| time limit | 1800s |
| process timeout | 3600s |
| exec time | 71.0221s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T004508` |
| submission sha | `634c0529cac921c9a01b8778a5fad0fb3f1765fd55aa6dfda5197dbd944fc1cb` |

Preprocessing/runtime notes:

- Runtime remained about 71 seconds.
- The local ordering of the first two high-public sources matches the public ordering, unlike the bagged 30-minute profile.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 14:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_holdout_best_gpu_30m` | 2 | 1.0000000 | 1.0000000 | 0.0002532 | +0.0002532 | 72.0s |
| `s6e7_align_bag3_best_gpu_30m` | 4 | -0.6763586 | -0.8000000 | 0.0003521 | -0.0001295 | 151.5s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |

Current best profile choice changed?

- Not yet. `s6e7_align_holdout_best_gpu_30m` is promising, but only has two observations. Continue to the disagreement source before changing the best profile choice.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:46:46+02:00` |
| elapsed | 0:55:11 |
| estimated remaining | 11:04:48 |
| 12-hour budget reached? | no |
| enough time for another <=30m AutoGluon rerun? | yes |

Decision for next experiment:

- Continue `s6e7_align_holdout_best_gpu_30m` on the strong historical local/public disagreement source.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_align_holdout_best_gpu_30m --timeout 3600 --execute
```

Planned source:

| field | value |
|---|---:|
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| run | `2-smiling-topaz-oarfish` |
| step | 21 |

## 2026-07-08T00:49:17+02:00 - Rerun 15 result / Rerun 16 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_align_holdout_best_gpu_30m --timeout 3600 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_best_gpu_30m` |
| profile status | newly created |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.950296727513 |
| local - public | +0.001046727513 |
| absolute error | 0.001046727513 |
| time limit | 1800s |
| process timeout | 3600s |
| exec time | 85.0245s |
| status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T004724` |
| submission sha | `5d6d41616416a6f711a421d40f588dacdebbb3d0625701a235dd339bc5301182` |

Preprocessing/runtime notes:

- Runtime was about 85 seconds.
- Rank signal remained strong, but this disagreement source shows large positive bias.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 15:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_holdout_best_gpu_30m` | 3 | 0.8959127 | 1.0000000 | 0.0005177 | +0.0005177 | 76.4s |
| `s6e7_align_bag3_best_gpu_30m` | 4 | -0.6763586 | -0.8000000 | 0.0003521 | -0.0001295 | 151.5s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |

Current best profile choice changed?

- No. The holdout/best profile has the best rank evidence so far among new profiles, but its positive bias and MAE are worse than `best_boost_gpu_1h` and currently worse than the 30-minute bagged profile.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:49:17+02:00` |
| elapsed | 0:57:42 |
| estimated remaining | 11:02:17 |
| 12-hour budget reached? | no |
| enough time for another <=30m AutoGluon rerun? | yes |

Decision for next experiment:

- Complete the four-source matched comparison for `s6e7_align_holdout_best_gpu_30m` before judging whether to keep it or create a less biased holdout variant.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_align_holdout_best_gpu_30m --timeout 3600 --execute
```

Planned source:

| field | value |
|---|---:|
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| run | `2-whimsical-albatross-from-camelot` |
| step | 47 |

## 2026-07-08T00:51:43+02:00 - Rerun 16 result / Rerun 17 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_align_holdout_best_gpu_30m --timeout 3600 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_best_gpu_30m` |
| profile status | newly created |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.950519959553 |
| local - public | +0.000799959553 |
| absolute error | 0.000799959553 |
| time limit | 1800s |
| process timeout | 3600s |
| exec time | 74.0240s |
| status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T004952` |
| submission sha | `f4cd3492a6b6083c44bc2ef2e55123a9a18fa1b5618475c68fa7672e5d13b28e` |

Preprocessing/runtime notes:

- Runtime was about 74 seconds.
- The completed four-source comparison shows the holdout/best profile is still materially over-optimistic.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 16:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_bag3_best_gpu_30m` | 4 | -0.6763586 | -0.8000000 | 0.0003521 | -0.0001295 | 151.5s |
| `s6e7_align_holdout_best_gpu_30m` | 4 | 0.2703368 | 0.4000000 | 0.0005883 | +0.0005883 | 75.8s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |

Current best profile choice changed?

- No. `best_boost_gpu_1h` still has the best MAE and bias. Among newly created profiles, the 30-minute bagged profile has better calibration but poor rank, while the holdout/best profile has better rank but too much positive bias.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:51:43+02:00` |
| elapsed | 1:00:08 |
| estimated remaining | 10:59:51 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Decision for next experiment:

- Test existing `xgb_medium_gpu_balanced_10m` on the same source set to isolate whether the multi-model booster ensemble contributes to holdout over-optimism. This is a controlled model-type change using a 10-minute profile already present in config.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Planned source:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| run | `2-vociferous-tortoise-of-perspective` |
| step | 13 |

## 2026-07-08T00:53:00+02:00 - Rerun 17 result / Rerun 18 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.949878972261 |
| local - public | -0.000281027739 |
| absolute error | 0.000281027739 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 16.0130s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T005221` |
| submission sha | `46a4b351dc033110cbfa40fe5890e06bcfc05b54eb52eb6098e79cd9de8dbe63` |

Preprocessing/runtime notes:

- XGB-only runtime was much shorter than the booster ensemble profiles.
- First point is slightly pessimistic rather than over-optimistic.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 17:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `xgb_medium_gpu_balanced_10m` | 1 | n/a | n/a | 0.0002810 | -0.0002810 | 16.0s |
| `s6e7_align_bag3_best_gpu_30m` | 4 | -0.6763586 | -0.8000000 | 0.0003521 | -0.0001295 | 151.5s |
| `s6e7_align_holdout_best_gpu_30m` | 4 | 0.2703368 | 0.4000000 | 0.0005883 | +0.0005883 | 75.8s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |

Current best profile choice changed?

- No. One XGB-only observation is not enough. It is promising enough to continue because it is cheap and has lower first-point error than the newly created holdout/best profile.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:53:00+02:00` |
| elapsed | 1:01:25 |
| estimated remaining | 10:58:34 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Decision for next experiment:

- Continue `xgb_medium_gpu_balanced_10m` on the second high-public source.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Planned source:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| run | `2-vociferous-tortoise-of-perspective` |
| step | 55 |

## 2026-07-08T00:54:21+02:00 - Rerun 18 result / Rerun 19 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.949912086050 |
| local - public | -0.000167913950 |
| absolute error | 0.000167913950 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 16.0144s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T005340` |
| submission sha | `b26c47c28d37725c29a637e8352cdb254ad1d532b99cdb73aad9b62f28f42679` |

Preprocessing/runtime notes:

- Runtime remained about 16 seconds.
- Calibration error is low, but the first two points are locally ordered opposite to public score.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 18:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `xgb_medium_gpu_balanced_10m` | 2 | -1.0000000 | -1.0000000 | 0.0002245 | -0.0002245 | 16.0s |
| `s6e7_align_bag3_best_gpu_30m` | 4 | -0.6763586 | -0.8000000 | 0.0003521 | -0.0001295 | 151.5s |
| `s6e7_align_holdout_best_gpu_30m` | 4 | 0.2703368 | 0.4000000 | 0.0005883 | +0.0005883 | 75.8s |

Current best profile choice changed?

- No. XGB-only has promising error but insufficient and currently unfavorable rank evidence.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:54:21+02:00` |
| elapsed | 1:02:46 |
| estimated remaining | 10:57:13 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Decision for next experiment:

- Continue `xgb_medium_gpu_balanced_10m` on the strong historical local/public disagreement source.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Planned source:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| run | `2-smiling-topaz-oarfish` |
| step | 21 |

## 2026-07-08T00:55:55+02:00 - Rerun 19 result / Rerun 20 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.950197794297 |
| local - public | +0.000947794297 |
| absolute error | 0.000947794297 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 31.0170s |
| status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T005458` |
| submission sha | `8306cb9d4e5a4ff8cb3b3c5cf93f4dbf18fa1fd4820f491c9d410d4aca59f82a` |

Preprocessing/runtime notes:

- Runtime was about 31 seconds.
- The disagreement source is strongly overestimated, and rank agreement is currently poor.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 19:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `xgb_medium_gpu_balanced_10m` | 3 | -0.9998856 | -1.0000000 | 0.0004656 | +0.0001663 | 21.0s |
| `s6e7_align_bag3_best_gpu_30m` | 4 | -0.6763586 | -0.8000000 | 0.0003521 | -0.0001295 | 151.5s |
| `s6e7_align_holdout_best_gpu_30m` | 4 | 0.2703368 | 0.4000000 | 0.0005883 | +0.0005883 | 75.8s |

Current best profile choice changed?

- No. XGB-only is cheap and has mixed calibration, but rank agreement is currently poor.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:55:55+02:00` |
| elapsed | 1:04:20 |
| estimated remaining | 10:55:39 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Decision for next experiment:

- Complete the four-source XGB-only comparison using the remaining high-local/weak-public source.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Planned source:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| run | `2-whimsical-albatross-from-camelot` |
| step | 47 |

## 2026-07-08T00:57:35+02:00 - Rerun 20 result / Rerun 21 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.949956769491 |
| local - public | +0.000236769491 |
| absolute error | 0.000236769491 |
| time limit | 600s |
| process timeout | 1800s |
| exec time | 15.0143s |
| status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T005630` |
| submission sha | `7da1ac5621a560396bca83402a76257d2ed610b762f78debb180bdbcf1093af8` |

Preprocessing/runtime notes:

- Runtime was about 15 seconds.
- XGB-only remains very cheap but ranks the four representative sources in the opposite public order.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 20:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_bag3_best_gpu_30m` | 4 | -0.6763586 | -0.8000000 | 0.0003521 | -0.0001295 | 151.5s |
| `xgb_medium_gpu_balanced_10m` | 4 | -0.9636407 | -1.0000000 | 0.0004084 | +0.0001839 | 19.5s |
| `s6e7_align_holdout_best_gpu_30m` | 4 | 0.2703368 | 0.4000000 | 0.0005883 | +0.0005883 | 75.8s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |

Current best profile choice changed?

- No. XGB-only is not a good practical proxy on this source set because its rank agreement is strongly negative despite moderate calibration.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:57:35+02:00` |
| elapsed | 1:06:00 |
| estimated remaining | 10:53:59 |
| 12-hour budget reached? | no |
| enough time for another <=30m AutoGluon rerun? | yes |

Decision for next experiment:

- Test `best_boost_gpu_30m`, the allowed-time counterpart to the historically well-calibrated `best_boost_gpu_1h`. It uses `presets: best`, 1800s, GPU XGB/GBM/CAT, AutoGluon default validation, no explicit class balancing, and `save_space=true`.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile best_boost_gpu_30m --timeout 3600 --execute
```

Planned source:

| field | value |
|---|---:|
| profile | `best_boost_gpu_30m` |
| profile status | existing |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| run | `2-vociferous-tortoise-of-perspective` |
| step | 13 |

## 2026-07-08T00:33:10+02:00 - Rerun 10 result / Rerun 11 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_align_bag3_best_gpu_30m --timeout 3600 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_bag3_best_gpu_30m` |
| profile status | newly created |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.949674960896 |
| local - public | -0.000405039104 |
| absolute error | 0.000405039104 |
| time limit | 1800s |
| process timeout | 3600s |
| exec time | 126.0305s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T003039` |
| submission sha | `f3111ec8e7d710b54b1a0063e3fe251086a81d74390e1c7b787ad9ee6a31eddb` |

Preprocessing/runtime notes:

- Runtime stayed near 2 minutes despite the 30-minute modeling limit.
- The result is less positively biased than the 10-minute holdout profile, but the first two sources rank in the wrong order locally versus public score.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 10:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_bag3_best_gpu_30m` | 2 | -1.0000000 | -1.0000000 | 0.0004773 | -0.0004773 | 124.0s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |
| `s6e7_align_agval_balanced_gpu_10m` | 4 | 0.4021675 | 0.4000000 | 0.0065189 | +0.0065189 | 27.3s |

Current best profile choice changed?

- No. `best_boost_gpu_1h` remains best on MAE and bias. The 30-minute bagged profile has promising calibration error relative to the 10-minute holdout profile, but rank evidence is currently unfavorable and the sample size is only two.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:33:10+02:00` |
| elapsed | 0:41:35 |
| estimated remaining | 11:18:24 |
| 12-hour budget reached? | no |
| enough time for another <=30m AutoGluon rerun? | yes |

Decision for next experiment:

- Continue the same profile on a source with strong historical local/public disagreement. This tests whether the 30-minute bagged profile reduces over-optimism on a hard negative example.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_align_bag3_best_gpu_30m --timeout 3600 --execute
```

Planned source:

| field | value |
|---|---:|
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| run | `2-smiling-topaz-oarfish` |
| step | 21 |

## 2026-07-08T00:38:13+02:00 - Rerun 11 result / Rerun 12 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_align_bag3_best_gpu_30m --timeout 3600 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_bag3_best_gpu_30m` |
| profile status | newly created |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.949695324178 |
| local - public | +0.000445324178 |
| absolute error | 0.000445324178 |
| time limit | 1800s |
| process timeout | 3600s |
| exec time | 237.0574s |
| status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T003347` |
| submission sha | `af1cb0d9ed4e4b2c2ae94ecc23834c3d5f4e3a35a60cae0afd5059ea954c7231` |

Preprocessing/runtime notes:

- Runtime increased to about 4 minutes but remains well inside the 30-minute AutoGluon limit and 3600s process timeout.
- The local score increased on the lower-public source, so rank agreement remains suspicious even though mean bias improved.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 11:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_bag3_best_gpu_30m` | 3 | -0.7412227 | -1.0000000 | 0.0004666 | -0.0001698 | 161.7s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |
| `s6e7_align_agval_balanced_gpu_10m` | 4 | 0.4021675 | 0.4000000 | 0.0065189 | +0.0065189 | 27.3s |

Current best profile choice changed?

- No. The 30-minute bagged profile has better MAE/bias than the new 10-minute holdout profile so far, but its rank agreement is poor and `best_boost_gpu_1h` remains much better on MAE and bias.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:38:13+02:00` |
| elapsed | 0:46:38 |
| estimated remaining | 11:13:21 |
| 12-hour budget reached? | no |
| enough time for another <=30m AutoGluon rerun? | yes |

Decision for next experiment:

- Complete the four-source comparison for `s6e7_align_bag3_best_gpu_30m` using the remaining high-local/weak-public source from the original representative set.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_align_bag3_best_gpu_30m --timeout 3600 --execute
```

Planned source:

| field | value |
|---|---:|
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| run | `2-whimsical-albatross-from-camelot` |
| step | 47 |

## 2026-07-08T00:41:26+02:00 - Rerun 12 result / next profile decision

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_align_bag3_best_gpu_30m --timeout 3600 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_bag3_best_gpu_30m` |
| profile status | newly created |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.949711416746 |
| local - public | -0.000008583254 |
| absolute error | 0.000008583254 |
| time limit | 1800s |
| process timeout | 3600s |
| exec time | 121.0325s |
| status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T003849` |
| submission sha | `f3da60963d4ba30f47c16058a4df2dcf90beaba8dd23ca93037ee412df56d4db` |

Preprocessing/runtime notes:

- Runtime returned to about 2 minutes.
- This source was calibrated extremely well, but across the four-source set the profile still has poor rank agreement.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 12:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_bag3_best_gpu_30m` | 4 | -0.6763586 | -0.8000000 | 0.0003521 | -0.0001295 | 151.5s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |
| `s6e7_align_agval_balanced_gpu_10m` | 4 | 0.4021675 | 0.4000000 | 0.0065189 | +0.0065189 | 27.3s |

Current best profile choice changed?

- No. `best_boost_gpu_1h` remains the strongest practical agreement candidate because it still has the lowest MAE and near-zero bias. `s6e7_align_bag3_best_gpu_30m` is a useful cheaper calibration improvement over the 10-minute holdout profile, but its local ranking is currently anti-correlated with public score.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:41:26+02:00` |
| elapsed | 0:49:51 |
| estimated remaining | 11:10:08 |
| 12-hour budget reached? | no |
| enough time for another <=30m AutoGluon rerun? | yes |

Decision for next experiment:

- Create `s6e7_align_holdout_best_gpu_30m`: keep the 10-minute holdout profile's stronger Pearson/rank signal, but switch to `presets: best` and a 30-minute limit to test whether better model fitting reduces the observed positive bias without flattening local scores as much as bagged OOF validation did.

## 2026-07-08T00:42:36+02:00 - Rerun 13 planned

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:42:36+02:00` |
| elapsed | 0:51:01 |
| estimated remaining | 11:08:58 |
| 12-hour budget reached? | no |
| enough time for another <=30m AutoGluon rerun? | yes |

Profile added for this experiment:

| field | value |
|---|---|
| profile | `s6e7_align_holdout_best_gpu_30m` |
| status | newly created |
| AutoGluon time limit | 1800s |
| preprocessing timeout | 600s |
| presets | `best` |
| validation strategy | `holdout` |
| validation fraction | 0.2 |
| class balancing | `balanced` |
| model types | `XGB`, `GBM`, `CAT` |
| GPU settings | CUDA XGB/GBM and GPU CatBoost |
| bag/stack settings | `auto_stack=false`, no explicit bagging |
| ensemble behavior | `fit_weighted_ensemble=true` |

Intention:

- This profile is a controlled variant of `s6e7_align_holdout_balanced_gpu_10m`: keep holdout validation and balanced GPU boosters, but use `presets: best` and a 30-minute limit to test whether calibration improves without losing the holdout profile's stronger local/public rank signal.

Verification before expensive command:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_autogluon_s6e7_align_holdout_best_gpu_30m_profile -v
pgrep -af '[r]erun_autogluon_profile'
```

Result:

- Profile resolver test passed.
- No active `rerun_autogluon_profile` process was visible.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_align_holdout_best_gpu_30m --timeout 3600 --execute
```

Planned source:

| field | value |
|---|---:|
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| run | `2-vociferous-tortoise-of-perspective` |
| step | 13 |

Decision for next experiment:

- If this result is usable and runtime is safe, continue this profile across the same four-source set to compare directly with the previous profiles.

## 2026-07-08T00:44:34+02:00 - Rerun 13 result / Rerun 14 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_align_holdout_best_gpu_30m --timeout 3600 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_best_gpu_30m` |
| profile status | newly created |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.950399653468 |
| local - public | +0.000239653468 |
| absolute error | 0.000239653468 |
| time limit | 1800s |
| process timeout | 3600s |
| exec time | 73.0224s |
| status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T004259` |
| submission sha | `6e54289020c625d6d236fe92ee4706bef17943e6b525ea2f899849cbf89cf0fb` |

Preprocessing/runtime notes:

- Runtime was about 73 seconds, close to the 10-minute holdout profile runtimes despite the 30-minute limit.
- First calibration point is better than the 10-minute holdout average and worse than `best_boost_gpu_1h` average.

Post-experiment commands:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 --no-remote > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
```

Profile-level comparison after Rerun 13:

| profile | n | Pearson | Spearman | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| `best_boost_gpu_1h` | 4 | 0.1232385 | -0.2000000 | 0.0001451 | -0.0000201 | 616.9s |
| `s6e7_align_holdout_best_gpu_30m` | 1 | n/a | n/a | 0.0002397 | +0.0002397 | 73.0s |
| `s6e7_align_bag3_best_gpu_30m` | 4 | -0.6763586 | -0.8000000 | 0.0003521 | -0.0001295 | 151.5s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.8067492 | 0.2000000 | 0.0006922 | +0.0006922 | 60.3s |

Current best profile choice changed?

- No. One observation is too small for the new holdout/best profile. Continue the matched source set.

Budget check after completed experiment:

| field | value |
|---|---:|
| timestamp | `2026-07-08T00:44:34+02:00` |
| elapsed | 0:52:59 |
| estimated remaining | 11:07:00 |
| 12-hour budget reached? | no |
| enough time for another <=30m AutoGluon rerun? | yes |

Decision for next experiment:

- Continue `s6e7_align_holdout_best_gpu_30m` on the second high-public source with existing 1-hour comparison.

About to run:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_align_holdout_best_gpu_30m --timeout 3600 --execute
```

Planned source:

| field | value |
|---|---:|
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| run | `2-vociferous-tortoise-of-perspective` |
| step | 55 |
