# AutoGluon CV/Public Alignment Journal

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
