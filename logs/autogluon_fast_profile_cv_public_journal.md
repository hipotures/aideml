# AutoGluon Fast Profile CV/Public Journal

TASK_START_TIME=2026-07-08T01:15:58+02:00
TASK_TIME_BUDGET_HOURS=12
TASK_DEADLINE=2026-07-08T13:15:58+02:00
TASK_OBJECTIVE=Find fast medium-based AutoGluon screening profile aligned with known public scores

## 2026-07-08T01:15:58+02:00 - Task initialization

Scope:

- Target stage: fast screening profile calibration for many candidate source artifacts.
- Required base preset for candidate profiles: `medium` / `medium_quality`.
- Practical runtime target: preferably under 10 minutes, acceptable up to 15 minutes wall-clock.
- Calibration signal: known Kaggle `public_score` attached to already submitted source artifacts.
- Do not submit reruns to Kaggle.
- Do not modify candidate solution code.
- Keep historical full/best/long reruns separate from fast medium candidate profile ranking.

Initial current-state note:

- The worktree already contains uncommitted edits and historical alignment logs from an earlier attempt that used `best`/30-minute profiles. Those are not valid candidate answers for this task under the current objective.
- This journal is the authoritative record for the current medium-only fast-screening calibration task.

Initial next plan:

- Run the requested lab inspection commands.
- Build a usable source artifact table for `playground-series-s6e7`.
- Select a representative matched source SHA set from public-scored submitted AutoGluon artifacts.
- Inspect profile support and add only fast `medium`/`medium_quality` candidate profiles if existing profiles are insufficient.
- Run one expensive rerun at a time and append every experiment here.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:15:58+02:00` |
| elapsed | 0:00:00 |
| estimated remaining | 12:00:00 |
| 12-hour budget reached? | no |
| enough time for initial inspection? | yes |

## 2026-07-08T01:24:33+02:00 - Initial inspection and medium-only setup

Commands executed:

```bash
uv run python scripts/kaggle_submission_lab.py --output-format json > /tmp/aideml_kaggle_submission_lab.json
uv run python scripts/kaggle_submission_lab.py --output-format json --registry-limit 0 > /tmp/aideml_kaggle_submission_lab_full.json
uv run python scripts/analyze_autogluon_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --output logs/autogluon_cv_public_alignment_analysis.json
uv run pytest tests/test_analyze_fast_autogluon_profile_alignment.py tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_alignment_profiles_are_medium_screening_profiles tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_medium_profile_variants -v
uv run python scripts/analyze_fast_autogluon_profile_alignment.py --lab-json /tmp/aideml_kaggle_submission_lab_full.json --index logs/submission_index.json --competition playground-series-s6e7 --task-start-time 2026-07-08T01:15:58+02:00 --output-json logs/autogluon_fast_profile_cv_public_summary.json --output-csv logs/autogluon_fast_profile_cv_public_summary.csv --sources-csv logs/autogluon_fast_profile_cv_public_sources.csv
pgrep -af '[r]erun_autogluon_profile'
```

Inspection results:

- Usable submitted source artifacts: 66 AutoGluon rows with `remote_status=COMPLETE`, `eval_metric=balanced_accuracy`, numeric local CV, and numeric public score.
- Current task reruns completed after `TASK_START_TIME`: 0.
- Active rerun process before starting experiments: none.
- Historical `best` / 30-minute reruns were separated into `historical_full_reference_rows`; they are not valid candidate answers for this task.
- Removed active config entries for the prior `best` / 30-minute S6E7 candidate profiles and added only medium-based 10-minute variants.

New or updated files:

- `scripts/analyze_fast_autogluon_profile_alignment.py`
- `tests/test_analyze_fast_autogluon_profile_alignment.py`
- `logs/autogluon_fast_profile_cv_public_summary.json`
- `logs/autogluon_fast_profile_cv_public_summary.csv`
- `logs/autogluon_fast_profile_cv_public_sources.csv`

Profiles newly created or modified for this task:

| profile | status | intent | preset | time limit | validation | balance | ensemble |
|---|---|---|---|---:|---|---|---|
| `s6e7_fast_medium_holdout20_nobalance_10m` | newly created | isolate class-balance contribution to holdout optimism | `medium_quality` | 600s | holdout 20% | none | on |
| `s6e7_fast_medium_noensemble_balanced_10m` | newly created | isolate weighted-ensemble contribution to holdout optimism | `medium_quality` | 600s | holdout 20% | balanced | off |

Representative source SHA set selected by the helper:

| sha prefix | public | original local | local-public | run | step |
|---|---:|---:|---:|---|---:|
| `4d2b8df165` | 0.95016 | 0.950537217870 | +0.000377217870 | `2-vociferous-tortoise-of-perspective` | 13 |
| `9f5a6e6e5d` | 0.94925 | 0.950373586439 | +0.001123586439 | `2-smiling-topaz-oarfish` | 21 |
| `cdc4cd52a1` | 0.94972 | 0.950669628467 | +0.000949628467 | `2-whimsical-albatross-from-camelot` | 47 |
| `67b2500c70` | 0.94978 | 0.950375871389 | +0.000595871389 | `2-smiling-topaz-oarfish` | 28 |
| `f658c40156` | 0.95012 | 0.950645806537 | +0.000525806537 | `2-romantic-guan-of-eternity` | 51 |
| `5d49507484` | 0.94931 | 0.950322639621 | +0.001012639621 | `2-smiling-topaz-oarfish` | 3 |
| `0c8ec5b2fd` | 0.94993 | 0.950659698542 | +0.000729698542 | `2-whimsical-albatross-from-camelot` | 69 |
| `48bdb4a69c` | 0.94979 | 0.950521303749 | +0.000731303749 | `2-romantic-guan-of-eternity` | 7 |
| `b07a3b527a` | 0.95009 | 0.950560509646 | +0.000470509646 | `2-romantic-guan-of-eternity` | 1 |
| `b473cc2630` | 0.94939 | 0.950343393219 | +0.000953393219 | `2-smiling-topaz-oarfish` | 27 |
| `117e38ebe5` | 0.95001 | 0.950653008218 | +0.000643008218 | `2-married-stallion-of-courtesy` | 12 |
| `9ea9601b9a` | 0.94979 | 0.950477598464 | +0.000687598464 | `2-vociferous-tortoise-of-perspective` | 5 |

Historical fast-medium profile snapshot, not sufficient to stop:

| profile | n | Pearson | Spearman | top-2 hit | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `s6e7_align_agval_balanced_gpu_10m` | 4 | 0.402167 | 0.400000 | 0.500000 | 0.006519 | +0.006519 | 27.3s |
| `xgb_medium_gpu_balanced_10m` | 4 | -0.963641 | -1.000000 | 0.000000 | 0.000408 | +0.000184 | 19.5s |
| `s6e7_align_holdout_balanced_gpu_10m` | 3 | 0.871318 | 0.500000 | 1.000000 | 0.000799 | +0.000799 | 61.4s |

Interpretation:

- `s6e7_align_holdout_balanced_gpu_10m` has the best historical rank signal among valid medium profiles, but only three correct-competition observations because one previous row for `4d2b8df165` was recorded under the wrong competition.
- `s6e7_align_agval_balanced_gpu_10m` is severely over-optimistic and should not be prioritized.
- `xgb_medium_gpu_balanced_10m` is cheap and fairly calibrated but anti-correlated on the four historical sources.
- These are historical/pre-task rows for this journal, so current-task reruns are still required.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:24:33+02:00` |
| elapsed | 0:08:35 |
| estimated remaining | 11:51:25 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Rerun `s6e7_align_holdout_balanced_gpu_10m` on `4d2b8df165` with `--competition playground-series-s6e7` to complete the correct-competition four-source matched baseline for the strongest existing medium candidate.

## 2026-07-08T01:26:51+02:00 - Rerun 1 result / Rerun 2 planned

Initial command attempted:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

- Status: refused duplicate profile/source evaluation in non-interactive mode.
- Reason: an existing historical row used the same profile and source SHA, but that row was recorded under `playground-series-s6e6`; this task requires `playground-series-s6e7`.
- Action: reran with `--force` to create a correct-competition current-task baseline point.

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --force --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| profile status | existing |
| profile intent | baseline holdout 20%, balanced, weighted ensemble on |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.950521833903 |
| signed error | +0.000361833903 |
| absolute error | 0.000361833903 |
| runtime | 53.0175s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T012536` |
| submission sha | `611aaf7772d2831236c9a4d01f64d2bffe8c88dd1414b50fa3e7f36538588a73` |

Preprocessing/runtime notes:

- Completed inside the 10-minute AutoGluon limit.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- This completed the correct-competition four-source snapshot for `s6e7_align_holdout_balanced_gpu_10m`.
- Updated historical+current medium snapshot for that profile: `n=4`, Pearson 0.791508, Spearman 0.200000, top-2 hit 0.500000, MAE 0.000690, bias +0.000690, average runtime 59.3s.
- The profile remains useful but positively biased; rank agreement is weak on the four-source set.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:26:51+02:00` |
| elapsed | 0:10:53 |
| estimated remaining | 11:49:07 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run newly created `s6e7_fast_medium_holdout20_nobalance_10m` on the same source `4d2b8df165` to isolate whether class balancing is contributing to holdout over-optimism.

## 2026-07-08T01:30:29+02:00 - Rerun 2 result / Rerun 3 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_fast_medium_holdout20_nobalance_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_holdout20_nobalance_10m` |
| profile status | newly created |
| profile intent | isolate class-balance contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.879939455599 |
| signed error | -0.070220544401 |
| absolute error | 0.070220544401 |
| runtime | 160.0355s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T012726` |
| submission sha | `03aa26215fa630a921abe3da2a195661608e1ea8763d7db812bb432e8bd26d9a` |

Preprocessing/runtime notes:

- Completed within the 10-minute AutoGluon limit but took 160s, materially slower than the balanced holdout baseline.
- The score collapse suggests class balancing is essential for this competition/profile family.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- The no-balance profile is not a useful current candidate after this first observation: it is strongly under-optimistic by 0.07022 and would not rank source quality on the same scale as public score.
- Because this failure is large and directly tests the intended ablation, deprioritize matched-set expansion for this profile unless later evidence suggests revisiting class-balance options.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:30:29+02:00` |
| elapsed | 0:14:31 |
| estimated remaining | 11:45:29 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run newly created `s6e7_fast_medium_noensemble_balanced_10m` on `4d2b8df165` to test whether disabling weighted ensemble reduces positive bias while preserving balanced training.

## 2026-07-08T01:32:09+02:00 - Rerun 3 result / Rerun 4 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.950422114854 |
| signed error | +0.000262114854 |
| absolute error | 0.000262114854 |
| runtime | 45.0184s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T013101` |
| submission sha | `cc1d16b01d351518451dfd7398922afbdc2841b3a49e9c510da171587b74bf92` |

Preprocessing/runtime notes:

- Completed quickly at about 45s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- On this first source, disabling weighted ensemble reduced positive bias from +0.0003618 (`s6e7_align_holdout_balanced_gpu_10m`) to +0.0002621 and improved runtime from 53.0s to 45.0s.
- One source is not enough for rank evidence; continue the same four-source matched set.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:32:09+02:00` |
| elapsed | 0:16:11 |
| estimated remaining | 11:43:49 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue `s6e7_fast_medium_noensemble_balanced_10m` on `1070897a05`, the second high-public source from the four-source matched set.

## 2026-07-08T01:33:40+02:00 - Rerun 4 result / Rerun 5 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.950346747889 |
| signed error | +0.000266747889 |
| absolute error | 0.000266747889 |
| runtime | 44.0173s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T013236` |
| submission sha | `634c0529cac921c9a01b8778a5fad0fb3f1765fd55aa6dfda5197dbd944fc1cb` |

Preprocessing/runtime notes:

- Completed in about 44s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Current profile snapshot: `n=2`, Pearson 1.000000, Spearman 1.000000, top-2 hit 1.000000, MAE 0.000264, bias +0.000264, average runtime 44.5s.
- This is promising but only covers two similar high-public sources; continue with a hard local/public disagreement case.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:33:40+02:00` |
| elapsed | 0:17:42 |
| estimated remaining | 11:42:18 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue `s6e7_fast_medium_noensemble_balanced_10m` on `9f5a6e6e5d`, the largest local/public disagreement source in the initial four-source set.

## 2026-07-08T01:35:27+02:00 - Rerun 5 result / Rerun 6 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.950197794297 |
| signed error | +0.000947794297 |
| absolute error | 0.000947794297 |
| runtime | 57.0184s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T013408` |
| submission sha | `8306cb9d4e5a4ff8cb3b3c5cf93f4dbf18fa1fd4820f491c9d410d4aca59f82a` |

Preprocessing/runtime notes:

- Completed in about 57s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Current profile snapshot: `n=3`, Pearson 0.967169, Spearman 1.000000, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000492, bias +0.000492, average runtime 48.7s.
- Bias is worse on this hard low-public source, but still lower than the weighted-ensemble holdout baseline's current MAE/bias.
- Rank evidence remains promising; complete the four-source matched set before judging.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:35:27+02:00` |
| elapsed | 0:19:29 |
| estimated remaining | 11:40:31 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue `s6e7_fast_medium_noensemble_balanced_10m` on `cdc4cd52a1`, the remaining high-local/weak-public source in the initial four-source set.

## 2026-07-08T01:36:59+02:00 - Rerun 6 result / Rerun 7 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.950019649554 |
| signed error | +0.000299649554 |
| absolute error | 0.000299649554 |
| runtime | 38.0181s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T013552` |
| submission sha | `6e93001699b792e6bc424dd4279888b086c849e3d543afc36705945eb3562dd2` |

Preprocessing/runtime notes:

- Completed in about 38s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Four-source `s6e7_fast_medium_noensemble_balanced_10m` snapshot: Pearson 0.615657, Spearman 0.800000, top-2 hit 1.000000, top-3 hit 0.666667, MAE 0.000444, bias +0.000444, average runtime 46.0s, max runtime 57.0s.
- Compared with the weighted-ensemble holdout baseline on the same four sources, no-ensemble has better rank agreement, lower MAE/bias, and lower runtime.
- This is not enough to call a final winner; expand the matched comparison to more representative sources.

Current valid medium comparison:

| profile | n | Pearson | Spearman | top-2 hit | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `s6e7_fast_medium_noensemble_balanced_10m` | 4 | 0.615657 | 0.800000 | 1.000000 | 0.000444 | +0.000444 | 46.0s |
| `s6e7_align_holdout_balanced_gpu_10m` | 4 | 0.791508 | 0.200000 | 0.500000 | 0.000690 | +0.000690 | 59.3s |
| `xgb_medium_gpu_balanced_10m` | 4 | -0.963641 | -1.000000 | 0.000000 | 0.000408 | +0.000184 | 19.5s |
| `s6e7_align_agval_balanced_gpu_10m` | 4 | 0.402167 | 0.400000 | 0.500000 | 0.006519 | +0.006519 | 27.3s |
| `s6e7_fast_medium_holdout20_nobalance_10m` | 1 | n/a | n/a | n/a | 0.070221 | -0.070221 | 160.0s |

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:36:59+02:00` |
| elapsed | 0:21:01 |
| estimated remaining | 11:38:59 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Expand the matched comparison to additional representative sources. Start with `s6e7_fast_medium_noensemble_balanced_10m` on `f658c40156`, a top-public source not in the initial four-source set, then run the weighted-ensemble holdout baseline on the same source.

## 2026-07-08T01:39:03+02:00 - Rerun 7 result / Rerun 8 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha f658c40156 --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `f658c401567a3af2b19f2defd5a949c75490deedd388482eea8e82c8bd36aafb` |
| source public score | 0.95012 |
| source original local score | 0.950645806537 |
| rerun local CV score | 0.950421197605 |
| signed error | +0.000301197605 |
| absolute error | 0.000301197605 |
| runtime | 49.0199s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T013752` |
| submission sha | `8e009b07bcf07c63dddde039b0e589124f6b9ba0627aaaff949630adb2c5e337` |

Preprocessing/runtime notes:

- Completed in about 49s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- `s6e7_fast_medium_noensemble_balanced_10m` snapshot after five sources: Pearson 0.676870, Spearman 0.900000, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000416, bias +0.000416, average runtime 46.6s.
- The fifth point improves rank confidence and keeps runtime suitable for mass screening.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:39:03+02:00` |
| elapsed | 0:23:05 |
| estimated remaining | 11:36:55 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_align_holdout_balanced_gpu_10m` on the same source `f658c40156` for matched weighted-ensemble baseline comparison.

## 2026-07-08T01:40:40+02:00 - Rerun 8 result / Rerun 9 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha f658c40156 --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| profile status | existing |
| profile intent | baseline holdout 20%, balanced, weighted ensemble on |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `f658c401567a3af2b19f2defd5a949c75490deedd388482eea8e82c8bd36aafb` |
| source public score | 0.95012 |
| source original local score | 0.950645806537 |
| rerun local CV score | 0.950539711491 |
| signed error | +0.000419711491 |
| absolute error | 0.000419711491 |
| runtime | 50.0192s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T013928` |
| submission sha | `ab8ecfe8969dd55bcc0d87e1794466d178a5305a00bffe60570624647dbad45a` |

Preprocessing/runtime notes:

- Completed in about 50s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Five-source weighted-ensemble holdout baseline snapshot: Pearson 0.791152, Spearman 0.000000, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000636, bias +0.000636, average runtime 57.4s.
- Five-source no-ensemble snapshot remains stronger for ranking: Spearman 0.900000, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000416, bias +0.000416, average runtime 46.6s.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:40:40+02:00` |
| elapsed | 0:24:42 |
| estimated remaining | 11:35:18 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue the matched comparison with `5d49507484`, a large local/public disagreement source. Run no-ensemble first, then the weighted-ensemble baseline on the same source.

## 2026-07-08T01:42:34+02:00 - Rerun 9 result / Rerun 10 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 5d49507484 --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `5d49507484983b645be5609197a5b2d3921d6ce1e4ee80f3854c54627907b4de` |
| source public score | 0.94931 |
| source original local score | 0.950322639621 |
| rerun local CV score | 0.950362877837 |
| signed error | +0.001052877837 |
| absolute error | 0.001052877837 |
| runtime | 59.0202s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T014112` |
| submission sha | `edffc1ac630a566ef1cc53e9b54760a3f5d8db437c103c13ceadb7d2f6796235` |

Preprocessing/runtime notes:

- Completed in about 59s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Six-source no-ensemble snapshot: Pearson 0.437641, Spearman 0.714286, top-2 hit 1.000000, top-3 hit 0.666667, MAE 0.000522, bias +0.000522, average runtime 48.7s.
- The low-public disagreement source is over-optimistic by +0.001053, weakening rank agreement but not dislodging the top-two sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:42:34+02:00` |
| elapsed | 0:26:36 |
| estimated remaining | 11:33:24 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_align_holdout_balanced_gpu_10m` on `5d49507484` for the matched weighted-ensemble baseline comparison.

## 2026-07-08T01:44:16+02:00 - Rerun 10 result / Rerun 11 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 5d49507484 --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| profile status | existing |
| profile intent | baseline holdout 20%, balanced, weighted ensemble on |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `5d49507484983b645be5609197a5b2d3921d6ce1e4ee80f3854c54627907b4de` |
| source public score | 0.94931 |
| source original local score | 0.950322639621 |
| rerun local CV score | 0.950311133324 |
| signed error | +0.001001133324 |
| absolute error | 0.001001133324 |
| runtime | 55.0187s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T014258` |
| submission sha | `6dde48d1c9f0923bf330a331a0f1cf5e9b4539ba553b93972e4dbe6e5d8955bc` |

Preprocessing/runtime notes:

- Completed in about 55s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Six-source weighted-ensemble baseline snapshot: Pearson 0.858562, Spearman 0.428571, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000697, bias +0.000697, average runtime 57.0s.
- Six-source no-ensemble snapshot remains stronger for screening ranking: Spearman 0.714286, top-2 hit 1.000000, top-3 hit 0.666667, MAE 0.000522, bias +0.000522, average runtime 48.7s.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:44:16+02:00` |
| elapsed | 0:28:18 |
| estimated remaining | 11:31:42 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue the matched comparison with `0c8ec5b2fd`, a high-local/moderate-public source. Run no-ensemble first, then the weighted-ensemble baseline.

## 2026-07-08T01:45:48+02:00 - Rerun 11 result / Rerun 12 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 0c8ec5b2fd --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `0c8ec5b2fdc251164e95ab145577bf5ba55cc15d0d32785f22d23bd4e326119a` |
| source public score | 0.94993 |
| source original local score | 0.950659698542 |
| rerun local CV score | 0.950437498821 |
| signed error | +0.000507498821 |
| absolute error | 0.000507498821 |
| runtime | 45.0171s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T014441` |
| submission sha | `c955e7213d19a05022c8fcd09e7ed274458dc20dfa46aad3269cb690f2a39a3b` |

Preprocessing/runtime notes:

- Completed in about 45s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Seven-source no-ensemble snapshot: Pearson 0.459244, Spearman 0.571429, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000520, bias +0.000520, average runtime 48.2s.
- This source weakened top-k behavior for no-ensemble; the matched baseline result will show whether this is profile-specific or source-set difficulty.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:45:48+02:00` |
| elapsed | 0:29:50 |
| estimated remaining | 11:30:10 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_align_holdout_balanced_gpu_10m` on `0c8ec5b2fd` for matched baseline evidence.

## 2026-07-08T01:47:32+02:00 - Rerun 12 result / Rerun 13 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 0c8ec5b2fd --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| profile status | existing |
| profile intent | baseline holdout 20%, balanced, weighted ensemble on |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `0c8ec5b2fdc251164e95ab145577bf5ba55cc15d0d32785f22d23bd4e326119a` |
| source public score | 0.94993 |
| source original local score | 0.950659698542 |
| rerun local CV score | 0.950599288596 |
| signed error | +0.000669288596 |
| absolute error | 0.000669288596 |
| runtime | 58.0189s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T014613` |
| submission sha | `b12fa62f47dbed74eaf0fa48121a4e4a87f32a895dc91ffc4b4c7b318d27728e` |

Preprocessing/runtime notes:

- Completed in about 58s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Seven-source weighted-ensemble baseline snapshot: Pearson 0.842872, Spearman 0.321429, top-2 hit 0.000000, top-3 hit 0.333333, MAE 0.000693, bias +0.000693, average runtime 57.2s.
- Seven-source no-ensemble snapshot remains better for screening selection: Pearson 0.459244, Spearman 0.571429, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000520, bias +0.000520, average runtime 48.2s.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:47:32+02:00` |
| elapsed | 0:31:34 |
| estimated remaining | 11:28:26 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue the matched comparison with `48bdb4a69c`, a middle-public representative source. Run no-ensemble first, then the weighted-ensemble baseline.

## 2026-07-08T01:49:06+02:00 - Rerun 13 result / Rerun 14 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 48bdb4a69c --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `48bdb4a69c741508967a1861e266e4a5b838c96556aef7ebf2717428c93b8283` |
| source public score | 0.94979 |
| source original local score | 0.950521303749 |
| rerun local CV score | 0.950304877103 |
| signed error | +0.000514877103 |
| absolute error | 0.000514877103 |
| runtime | 45.0186s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T014800` |
| submission sha | `b1e575c481831d2a8067eb09db6399bdf13fff93b56037111faf292bbb941688` |

Preprocessing/runtime notes:

- Completed in about 45s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eight-source no-ensemble snapshot: Pearson 0.459228, Spearman 0.642857, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000519, bias +0.000519, average runtime 47.8s.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:49:06+02:00` |
| elapsed | 0:33:08 |
| estimated remaining | 11:26:52 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_align_holdout_balanced_gpu_10m` on `48bdb4a69c` for matched baseline evidence.

## 2026-07-08T01:50:38+02:00 - Rerun 14 result / Rerun 15 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 48bdb4a69c --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| profile status | existing |
| profile intent | baseline holdout 20%, balanced, weighted ensemble on |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `48bdb4a69c741508967a1861e266e4a5b838c96556aef7ebf2717428c93b8283` |
| source public score | 0.94979 |
| source original local score | 0.950521303749 |
| rerun local CV score | 0.950387867989 |
| signed error | +0.000597867989 |
| absolute error | 0.000597867989 |
| runtime | 48.0177s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T014928` |
| submission sha | `bc06699c7768d784dd899c04c83b7c99507de01d7308ec3f2aee28822d7b42b4` |

Preprocessing/runtime notes:

- Completed in about 48s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eight-source weighted-ensemble baseline snapshot: Pearson 0.808624, Spearman 0.452381, top-2 hit 0.000000, top-3 hit 0.333333, MAE 0.000681, bias +0.000681, average runtime 56.0s.
- Eight-source no-ensemble snapshot: Pearson 0.459228, Spearman 0.642857, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000519, bias +0.000519, average runtime 47.8s.
- No-ensemble remains the stronger screening candidate, but top-2 hit rate is only 0.5; continue expanding before recommending.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:50:38+02:00` |
| elapsed | 0:34:40 |
| estimated remaining | 11:25:20 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue the matched comparison with `b07a3b527a`, a top-public representative source. Run no-ensemble first, then weighted baseline.

## 2026-07-08T01:52:22+02:00 - Rerun 15 result / Rerun 16 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b07a3b527a --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b07a3b527ab89743affa724b885ee944d16eb8516f882afdb7bf38699a201c8e` |
| source public score | 0.95009 |
| source original local score | 0.950560509646 |
| rerun local CV score | 0.950367330218 |
| signed error | +0.000277330218 |
| absolute error | 0.000277330218 |
| runtime | 48.0179s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T015111` |
| submission sha | `8c500e7ace83a762298e20b21035b6de712661eedaef180172f5359836d4351d` |

Preprocessing/runtime notes:

- Completed in about 48s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Nine-source no-ensemble snapshot: Pearson 0.473945, Spearman 0.683333, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000492, bias +0.000492, average runtime 47.8s.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:52:22+02:00` |
| elapsed | 0:36:24 |
| estimated remaining | 11:23:36 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_align_holdout_balanced_gpu_10m` on `b07a3b527a` for matched baseline evidence.

## 2026-07-08T01:53:51+02:00 - Rerun 16 result / Rerun 17 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b07a3b527a --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| profile status | existing |
| profile intent | baseline holdout 20%, balanced, weighted ensemble on |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b07a3b527ab89743affa724b885ee944d16eb8516f882afdb7bf38699a201c8e` |
| source public score | 0.95009 |
| source original local score | 0.950560509646 |
| rerun local CV score | 0.950221921362 |
| signed error | +0.000131921362 |
| absolute error | 0.000131921362 |
| runtime | 43.0169s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T015245` |
| submission sha | `cd5fc8307f2c6714302b7d933a20f608dba9982c005822f66cf6fde46a8853ec` |

Preprocessing/runtime notes:

- Completed in about 43s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Nine-source weighted-ensemble baseline snapshot: Pearson 0.456263, Spearman 0.166667, top-2 hit 0.000000, top-3 hit 0.000000, MAE 0.000620, bias +0.000620, average runtime 54.6s.
- Nine-source no-ensemble snapshot: Pearson 0.473945, Spearman 0.683333, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000492, bias +0.000492, average runtime 47.8s.
- No-ensemble remains clearly better for screening rank behavior on current matched evidence.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:53:51+02:00` |
| elapsed | 0:37:53 |
| estimated remaining | 11:22:07 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue the matched comparison with `b473cc2630`, a low-public large-disagreement source. Run no-ensemble first, then the weighted baseline.

## 2026-07-08T01:55:40+02:00 - Rerun 17 result / Rerun 18 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b473cc2630 --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b473cc26307f2612b3851f3be08b380800928a7ace2b5aef8639ad63d11d1066` |
| source public score | 0.94939 |
| source original local score | 0.950343393219 |
| rerun local CV score | 0.950247535491 |
| signed error | +0.000857535491 |
| absolute error | 0.000857535491 |
| runtime | 61.0209s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T015416` |
| submission sha | `b352dde40ce9ff8fbb46c6c461a9b9263f7561b24b62e2e6832f6c7e388f42cc` |

Preprocessing/runtime notes:

- Completed in about 61s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Ten-source no-ensemble snapshot: Pearson 0.498808, Spearman 0.696970, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000529, bias +0.000529, average runtime 49.1s.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:55:40+02:00` |
| elapsed | 0:39:42 |
| estimated remaining | 11:20:18 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_align_holdout_balanced_gpu_10m` on `b473cc2630` for matched baseline evidence.

## 2026-07-08T01:57:31+02:00 - Rerun 18 result / Rerun 19 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b473cc2630 --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| profile status | existing |
| profile intent | baseline holdout 20%, balanced, weighted ensemble on |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b473cc26307f2612b3851f3be08b380800928a7ace2b5aef8639ad63d11d1066` |
| source public score | 0.94939 |
| source original local score | 0.950343393219 |
| rerun local CV score | 0.950284123505 |
| signed error | +0.000894123505 |
| absolute error | 0.000894123505 |
| runtime | 61.0196s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T015603` |
| submission sha | `6472d7be4a5885cb34dd35ce751a30f4455838ac05a25607baf4808bb87782a1` |

Preprocessing/runtime notes:

- Completed in about 61s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Ten-source weighted-ensemble baseline snapshot: Pearson 0.532172, Spearman 0.272727, top-2 hit 0.000000, top-3 hit 0.000000, MAE 0.000647, bias +0.000647, average runtime 55.2s.
- Ten-source no-ensemble snapshot: Pearson 0.498808, Spearman 0.696970, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000529, bias +0.000529, average runtime 49.1s.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:57:31+02:00` |
| elapsed | 0:41:33 |
| estimated remaining | 11:18:27 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue to the 12-source matched set with `117e38ebe5`, a high-local/top-public source. Run no-ensemble first, then weighted baseline.

## 2026-07-08T01:59:14+02:00 - Rerun 19 result / Rerun 20 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 117e38ebe5 --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `117e38ebe54ee5691eac05df30e6d47ec4b7e9232cffc18786c6f40bb5959bed` |
| source public score | 0.95001 |
| source original local score | 0.950653008218 |
| rerun local CV score | 0.950449898864 |
| signed error | +0.000439898864 |
| absolute error | 0.000439898864 |
| runtime | 51.0188s |
| result status | ok |
| artifact dir | `logs/2-married-stallion-of-courtesy/artifacts/20260708T015800` |
| submission sha | `ed65628fde30053e807d79a18919551806666dd345565e191be30c1e4065aa41` |

Preprocessing/runtime notes:

- Completed in about 51s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eleven-source no-ensemble snapshot: Pearson 0.526824, Spearman 0.636364, top-2 hit 0.000000, top-3 hit 0.333333, MAE 0.000521, bias +0.000521, average runtime 49.3s.
- Top-k behavior weakened after adding this high-local/top-public source, so the final selection is not yet settled.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T01:59:14+02:00` |
| elapsed | 0:43:16 |
| estimated remaining | 11:16:44 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_align_holdout_balanced_gpu_10m` on `117e38ebe5` for matched baseline evidence.

## 2026-07-08T02:00:55+02:00 - Rerun 20 result / Rerun 21 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 117e38ebe5 --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| profile status | existing |
| profile intent | baseline holdout 20%, balanced, weighted ensemble on |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `117e38ebe54ee5691eac05df30e6d47ec4b7e9232cffc18786c6f40bb5959bed` |
| source public score | 0.95001 |
| source original local score | 0.950653008218 |
| rerun local CV score | 0.950628013359 |
| signed error | +0.000618013359 |
| absolute error | 0.000618013359 |
| runtime | 53.0184s |
| result status | ok |
| artifact dir | `logs/2-married-stallion-of-courtesy/artifacts/20260708T015938` |
| submission sha | `a1a9e6d511bce99156a75db1181eb7841cda546f385b29de2bf440b20e95d7a2` |

Preprocessing/runtime notes:

- Completed in about 53s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eleven-source weighted-ensemble baseline snapshot: Pearson 0.557672, Spearman 0.254545, top-2 hit 0.000000, top-3 hit 0.000000, MAE 0.000645, bias +0.000645, average runtime 55.0s.
- Eleven-source no-ensemble snapshot: Pearson 0.526824, Spearman 0.636364, top-2 hit 0.000000, top-3 hit 0.333333, MAE 0.000521, bias +0.000521, average runtime 49.3s.
- Both profiles have poor top-k behavior on the expanded set, but no-ensemble remains better on rank correlation and calibration.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:00:55+02:00` |
| elapsed | 0:44:57 |
| estimated remaining | 11:15:03 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Complete the initial 12-source matched set with `9ea9601b9a`, a middle-public representative source. Run no-ensemble first, then weighted baseline.

## 2026-07-08T02:02:49+02:00 - Rerun 21 result / Rerun 22 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9ea9601b9a --profile s6e7_fast_medium_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | isolate weighted-ensemble contribution to holdout optimism |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9ea9601b9aca0b20f236bb624e4c4c1b46ef23c19a757339266211bacd967e1a` |
| source public score | 0.94979 |
| source original local score | 0.950477598464 |
| rerun local CV score | 0.950388510763 |
| signed error | +0.000598510763 |
| absolute error | 0.000598510763 |
| runtime | 53.0217s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T020122` |
| submission sha | `f544a1339e9dd6865509ed6ef546b702c0d5d074a9c163f7125b284b3fdd92d9` |

Preprocessing/runtime notes:

- Completed in about 53s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Twelve-source no-ensemble snapshot: Pearson 0.519178, Spearman 0.651490, top-2 hit 0.000000, top-3 hit 0.333333, MAE 0.000527, bias +0.000527, average runtime 49.6s.
- No-ensemble has useful rank correlation but is not yet a strong top-k selector on the 12-source set.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:02:49+02:00` |
| elapsed | 0:46:51 |
| estimated remaining | 11:13:09 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_align_holdout_balanced_gpu_10m` on `9ea9601b9a` to complete the 12-source matched weighted-baseline comparison.

## 2026-07-08T02:04:37+02:00 - Rerun 22 result / Holdout-30 variant added

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9ea9601b9a --profile s6e7_align_holdout_balanced_gpu_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_align_holdout_balanced_gpu_10m` |
| profile status | existing |
| profile intent | baseline holdout 20%, balanced, weighted ensemble on |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9ea9601b9aca0b20f236bb624e4c4c1b46ef23c19a757339266211bacd967e1a` |
| source public score | 0.94979 |
| source original local score | 0.950477598464 |
| rerun local CV score | 0.950436473658 |
| signed error | +0.000646473658 |
| absolute error | 0.000646473658 |
| runtime | 58.0207s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T020313` |
| submission sha | `1ded82b8c0c73b07026665c58d37dfed994db65eedc5b98ddb1ffca5803a62fb` |

Preprocessing/runtime notes:

- Completed in about 58s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Twelve-source matched comparison:

| profile | n | Pearson | Spearman | top-2 hit | top-3 hit | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `s6e7_fast_medium_noensemble_balanced_10m` | 12 | 0.519178 | 0.651490 | 0.000000 | 0.333333 | 0.000527 | +0.000527 | 49.6s |
| `s6e7_align_holdout_balanced_gpu_10m` | 12 | 0.557772 | 0.325745 | 0.000000 | 0.000000 | 0.000645 | +0.000645 | 55.3s |

Interpretation:

- No-ensemble remains the better medium candidate on rank correlation, top-3 usefulness, MAE, bias, and runtime.
- Neither profile is a strong top-2 selector on the expanded 12-source set, so further profile search is warranted.

Profile added after this comparison:

| profile | status | intent | preset | time limit | validation | balance | ensemble |
|---|---|---|---|---:|---|---|---|
| `s6e7_fast_medium_holdout30_noensemble_balanced_10m` | newly created | test whether a larger holdout improves top-k/rank stability while preserving no-ensemble calibration | `medium_quality` | 600s | holdout 30% | balanced | off |

Verification:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_alignment_profiles_are_medium_screening_profiles tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_medium_profile_variants -v
```

Result: passed.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:04:37+02:00` |
| elapsed | 0:48:39 |
| estimated remaining | 11:11:21 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Start probing `s6e7_fast_medium_holdout30_noensemble_balanced_10m` on the same representative sources, beginning with `4d2b8df165`.

## 2026-07-08T02:07:20+02:00 - Rerun 23 result / Rerun 24 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_fast_medium_holdout30_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_holdout30_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | larger holdout no-ensemble rank-stability probe |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.949793088337 |
| signed error | -0.000366911663 |
| absolute error | 0.000366911663 |
| runtime | 52.0198s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T020602` |
| submission sha | `38c26618fd2d014b157c45c86d46baa388f3a2537116e3e93e3a7a5cd1a4fc96` |

Preprocessing/runtime notes:

- Completed in about 52s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.
- A first read-only `jq` comparison command had a filter error; no experiment data was affected.

Profile ranking impact:

- One point is not enough for ranking; the first holdout-30 score is modestly pessimistic and has MAE 0.000367.
- Continue on the second high-public source to see whether the local ordering of top sources is preserved.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:07:20+02:00` |
| elapsed | 0:51:22 |
| estimated remaining | 11:08:38 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue `s6e7_fast_medium_holdout30_noensemble_balanced_10m` on `1070897a05`.

## 2026-07-08T02:09:08+02:00 - Rerun 24 result / Rerun 25 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_fast_medium_holdout30_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_holdout30_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | larger holdout no-ensemble rank-stability probe |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.949935523153 |
| signed error | -0.000144476847 |
| absolute error | 0.000144476847 |
| runtime | 49.0197s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T020753` |
| submission sha | `43ff2cc3f7cf8dfa6b9f8eb435e93a73b2cff90eb0fc50733ebf870cf9575c6f` |

Preprocessing/runtime notes:

- Completed in about 49s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Two-source holdout-30 snapshot: Pearson -1.000000, Spearman -1.000000, top-2 hit 1.000000, MAE 0.000256, bias -0.000256, average runtime 50.5s.
- Calibration looks better than holdout-20/no-ensemble, but it reverses the first two high-public sources; continue to a disagreement source before deciding whether this is useful.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:09:08+02:00` |
| elapsed | 0:53:10 |
| estimated remaining | 11:06:50 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue `s6e7_fast_medium_holdout30_noensemble_balanced_10m` on `9f5a6e6e5d`, the low-public disagreement source from the initial four-source set.

## 2026-07-08T02:11:42+02:00 - Rerun 25 result / Rerun 26 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_fast_medium_holdout30_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_holdout30_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | larger holdout no-ensemble rank-stability probe |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.949894301760 |
| signed error | +0.000644301760 |
| absolute error | 0.000644301760 |
| runtime | 66.0182s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T020938` |
| submission sha | `b869bee58c95eec00869236f998cd05dae6a61050768a8a3534cbc3f6165a45c` |

Preprocessing/runtime notes:

- Completed in about 66s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Three-source holdout-30 snapshot: Pearson -0.312677, Spearman -0.500000, top-2 hit 0.500000, top-3 hit 1.000000, MAE 0.000385, bias +0.000044, average runtime 55.7s.
- Bias is much better, but rank correlation is poor. Complete the initial four-source probe before deciding whether to stop this branch.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:11:42+02:00` |
| elapsed | 0:55:44 |
| estimated remaining | 11:04:16 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_holdout30_noensemble_balanced_10m` on `cdc4cd52a1`, the fourth initial source.

## 2026-07-08T02:13:33+02:00 - Rerun 26 result / Rerun 27 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_fast_medium_holdout30_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_holdout30_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | larger holdout no-ensemble rank-stability probe |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.950104418030 |
| signed error | +0.000384418030 |
| absolute error | 0.000384418030 |
| runtime | 60.0182s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T021208` |
| submission sha | `af91bd748cb42f2c7c37e9e91edd923931932fcd3a092565d1e36c2f828b2da5` |

Preprocessing/runtime notes:

- Completed in about 60s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Four-source holdout-30 snapshot: Pearson -0.260563, Spearman -0.400000, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000385, bias +0.000129, average runtime 56.8s.
- Calibration/bias are better than holdout-20 variants, but rank behavior is poor; deprioritize this branch unless later evidence suggests calibration should dominate.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:13:33+02:00` |
| elapsed | 0:57:35 |
| estimated remaining | 11:02:25 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Expand existing `xgb_medium_gpu_balanced_10m` beyond its initial four sources. Start with `f658c40156` and continue across the remaining representative set if runtime remains cheap.

## 2026-07-08T02:16:45+02:00 - Rerun 27 result / Rerun 28 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha f658c40156 --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing fast medium profile expanded |
| profile intent | cheap XGB-only calibration/ranking probe |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `f658c401567a3af2b19f2defd5a949c75490deedd388482eea8e82c8bd36aafb` |
| source public score | 0.95012 |
| source original local score | 0.950645806537 |
| rerun local CV score | 0.949878972261 |
| signed error | -0.000241027739 |
| absolute error | 0.000241027739 |
| runtime | 17.0127s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T021608` |
| submission sha | `46a4b351dc033110cbfa40fe5890e06bcfc05b54eb52eb6098e79cd9de8dbe63` |

Preprocessing/runtime notes:

- Completed in about 17s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Five-source XGB snapshot: Pearson -0.968385, Spearman -0.974679, top-2 hit 0.000000, top-3 hit 0.333333, MAE 0.000375, bias +0.000099, average runtime 19.0s, max runtime 31.0s.
- Calibration remains competitive, but rank ordering is still strongly anti-correlated. This is not a winner yet; continue only because the profile is cheap and a few more representative points can confirm whether this is structural.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:16:45+02:00` |
| elapsed | 1:00:47 |
| estimated remaining | 10:59:13 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Continue `xgb_medium_gpu_balanced_10m` on `5d49507484` as a cheap sixth point; stop expanding this branch early if anti-correlation persists with no top-k improvement.

## 2026-07-08T02:18:28+02:00 - Rerun 28 result / Rerun 29 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 5d49507484 --profile xgb_medium_gpu_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `xgb_medium_gpu_balanced_10m` |
| profile status | existing fast medium profile expanded |
| profile intent | cheap XGB-only calibration/ranking probe |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `5d49507484983b645be5609197a5b2d3921d6ce1e4ee80f3854c54627907b4de` |
| source public score | 0.94931 |
| source original local score | 0.950322639621 |
| rerun local CV score | 0.950197794297 |
| signed error | +0.000887794297 |
| absolute error | 0.000887794297 |
| runtime | 31.0167s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T021732` |
| submission sha | `8306cb9d4e5a4ff8cb3b3c5cf93f4dbf18fa1fd4820f491c9d410d4aca59f82a` |

Preprocessing/runtime notes:

- Completed in about 31s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Six-source XGB snapshot: Pearson -0.975294, Spearman -0.971008, top-2 hit 0.000000, top-3 hit 0.000000, MAE 0.000460, bias +0.000230, average runtime 21.0s, max runtime 31.0s.
- This branch is now stopped: calibration is acceptable, but the rank signal is consistently inverted and top-k behavior deteriorated.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:18:28+02:00` |
| elapsed | 1:02:30 |
| estimated remaining | 10:57:30 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Add `s6e7_fast_medium_noensemble_balanced_seed7_10m`, a seed-only variant of the best current medium/no-ensemble family, then run it first on `4d2b8df165`.

## 2026-07-08T02:21:22+02:00 - Rerun 29 result / Rerun 30 planned

Profile/config changes before this rerun:

- Added `s6e7_fast_medium_noensemble_balanced_seed7_10m`.
- The profile is a seed-only variant of `s6e7_fast_medium_noensemble_balanced_10m`: same `medium_quality`, 600s AutoGluon limit, 600s preprocess timeout, holdout fraction 0.2, balanced sample weights, GPU XGB/GBM/CAT, no weighted ensemble, and `auto_stack: false`; only `seed` changes from 42 to 7.
- Guardrail verification passed:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_alignment_profiles_are_medium_screening_profiles tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_medium_profile_variants -v
```

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_fast_medium_noensemble_balanced_seed7_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed7_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.949910286505 |
| signed error | -0.000249713495 |
| absolute error | 0.000249713495 |
| runtime | 52.0232s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T022010` |
| submission sha | `26f9070a69d2a16e2cbdbd398e0aa7bbf39a80d8ce4e9cd1c9ba59a7c895eb5e` |

Preprocessing/runtime notes:

- Completed in about 52s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- One-source seed-7 snapshot: MAE 0.000250, bias -0.000250, runtime 52.0s.
- This is well calibrated on the first representative source, but it has no ranking evidence yet. Continue to the same initial source set used for other probes.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:21:22+02:00` |
| elapsed | 1:05:24 |
| estimated remaining | 10:54:36 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed7_10m` on `1070897a05`, then continue through the first four representative sources if runtime stays near the existing no-ensemble profile.

## 2026-07-08T02:23:03+02:00 - Rerun 30 result / Rerun 31 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_fast_medium_noensemble_balanced_seed7_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed7_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.949925482855 |
| signed error | -0.000154517145 |
| absolute error | 0.000154517145 |
| runtime | 42.0210s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T022200` |
| submission sha | `587c8523e4c3e1ece31439052eff14ef69630f55409dfea114740ecab77e1dc0` |

Preprocessing/runtime notes:

- Completed in about 42s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Two-source seed-7 snapshot: Pearson -1.000000, Spearman -1.000000, top-2 hit 1.000000, MAE 0.000202, bias -0.000202, average runtime 47.0s.
- Calibration is strong so far, but the first two points are reversed by CV. Continue to four sources before deciding whether this branch should stop.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:23:03+02:00` |
| elapsed | 1:07:05 |
| estimated remaining | 10:52:55 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed7_10m` on `9f5a6e6e5d`, the third initial representative source.

## 2026-07-08T02:24:33+02:00 - Rerun 31 result / Rerun 32 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_fast_medium_noensemble_balanced_seed7_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed7_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.950025183107 |
| signed error | +0.000775183107 |
| absolute error | 0.000775183107 |
| runtime | 41.0175s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T022331` |
| submission sha | `b7617a5f4e1e8a95b43ea75652d7cfee695b2da8e7e147c48ae9389f8a0dcfd5` |

Preprocessing/runtime notes:

- Completed in about 41s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Three-source seed-7 snapshot: Pearson -0.999094, Spearman -1.000000, top-2 hit 0.500000, top-3 hit 1.000000, MAE 0.000393, bias +0.000124, average runtime 45.0s.
- The profile ranks all first-three sources in reverse order, so this is unlikely to beat the seed-42 no-ensemble profile. Run the fourth initial source only to complete the comparable early read.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:24:33+02:00` |
| elapsed | 1:08:35 |
| estimated remaining | 10:51:25 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed7_10m` on `cdc4cd52a1`, then stop or expand this seed branch based on the four-source snapshot.

## 2026-07-08T02:27:07+02:00 - Rerun 32 result / Rerun 33 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_fast_medium_noensemble_balanced_seed7_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed7_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.950059425266 |
| signed error | +0.000339425266 |
| absolute error | 0.000339425266 |
| runtime | 41.0176s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T022600` |
| submission sha | `1fdc2a1ac6facdca32cac304d2336686131540ea77f5cc38847e80889e35f4c9` |

Preprocessing/runtime notes:

- Completed in about 41s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Four-source seed-7 snapshot: Pearson -0.782540, Spearman -0.800000, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000380, bias +0.000178, average runtime 44.0s.
- Stop this seed branch: the calibration is decent, but rank ordering is materially worse than the seed-42 no-ensemble profile.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:27:07+02:00` |
| elapsed | 1:11:09 |
| estimated remaining | 10:48:51 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Add `s6e7_fast_medium_gbmcat_noensemble_balanced_10m`, a seed-42 no-ensemble holdout profile that excludes XGB, then run it first on `4d2b8df165`.

## 2026-07-08T02:29:35+02:00 - Rerun 33 result / Rerun 34 planned

Profile/config changes before this rerun:

- Added `s6e7_fast_medium_gbmcat_noensemble_balanced_10m`.
- The profile keeps the best current seed-42 no-ensemble holdout setup but limits `included_model_types` to `[GBM, CAT]` after the XGB-only profile showed consistently inverted rank behavior.
- Guardrail verification passed:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_alignment_profiles_are_medium_screening_profiles tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_medium_profile_variants -v
```

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.950381883794 |
| signed error | +0.000221883794 |
| absolute error | 0.000221883794 |
| runtime | 40.0188s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T022829` |
| submission sha | `799218b2032a6d7dd474d72d16b331787c5f8126b4e35a015ee0194a0e09084a` |

Preprocessing/runtime notes:

- Completed in about 40s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- One-source GBM/CAT snapshot: MAE 0.000222, bias +0.000222, runtime 40.0s.
- Initial calibration is strong and runtime is acceptable; continue to gather the same early four-source ranking snapshot.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:29:35+02:00` |
| elapsed | 1:13:37 |
| estimated remaining | 10:46:23 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `1070897a05`, the second representative source.

## 2026-07-08T02:31:15+02:00 - Rerun 34 result / Rerun 35 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.950378836516 |
| signed error | +0.000298836516 |
| absolute error | 0.000298836516 |
| runtime | 43.0223s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T023007` |
| submission sha | `cecb1ab6559f741aa353236d5bde1d24871a04701abc737149191de5fab416fa` |

Preprocessing/runtime notes:

- Completed in about 43s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Two-source GBM/CAT snapshot: Pearson 1.000000, Spearman 1.000000, top-2 hit 1.000000, MAE 0.000260, bias +0.000260, average runtime 41.5s.
- Early ordering is better than the XGB-only and seed-7 branches. Continue to the low-public source to see whether the profile separates weak candidates correctly.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:31:15+02:00` |
| elapsed | 1:15:17 |
| estimated remaining | 10:44:43 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `9f5a6e6e5d`, the third representative source.

## 2026-07-08T02:32:49+02:00 - Rerun 35 result / Rerun 36 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.949615804692 |
| signed error | +0.000365804692 |
| absolute error | 0.000365804692 |
| runtime | 33.0184s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T023142` |
| submission sha | `bebe213b7d8e9289906a915ae8491405a44ab90b3106f211f37d0b3eb1cd3d4b` |

Preprocessing/runtime notes:

- Completed in about 33s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Three-source GBM/CAT snapshot: Pearson 0.997112, Spearman 1.000000, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000296, bias +0.000296, average runtime 38.7s.
- This is the first new branch with both strong calibration and correct early ranking. Continue through the fourth initial source and expand if the signal remains positive.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:32:49+02:00` |
| elapsed | 1:16:51 |
| estimated remaining | 10:43:09 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `cdc4cd52a1`, the fourth initial representative source.

## 2026-07-08T02:34:27+02:00 - Rerun 36 result / Rerun 37 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.950034777842 |
| signed error | +0.000314777842 |
| absolute error | 0.000314777842 |
| runtime | 31.0161s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T023322` |
| submission sha | `bc0f0572e61d818117978a5a11559cb927766ddb2b033ea24376737ac714d177` |

Preprocessing/runtime notes:

- Completed in about 31s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Four-source GBM/CAT snapshot: Pearson 0.997131, Spearman 1.000000, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000300, bias +0.000300, average runtime 36.8s.
- This is now the main expansion candidate: it beats the other four-source probes on rank/top-k while staying within fast runtime. Expand across the full 12-source representative set before making a final selection.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:34:27+02:00` |
| elapsed | 1:18:29 |
| estimated remaining | 10:41:31 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `f658c40156`, then continue through the remaining representative sources if the profile remains stable.

## 2026-07-08T02:36:08+02:00 - Rerun 37 result / Rerun 38 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha f658c40156 --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `f658c401567a3af2b19f2defd5a949c75490deedd388482eea8e82c8bd36aafb` |
| source public score | 0.95012 |
| source original local score | 0.950645806537 |
| rerun local CV score | 0.950381883794 |
| signed error | +0.000261883794 |
| absolute error | 0.000261883794 |
| runtime | 41.0174s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T023504` |
| submission sha | `ceeef6d366896885741a74ed303f83e62e85386178c7aa22d3f4d90f1d6f5e95` |

Preprocessing/runtime notes:

- Completed in about 41s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Five-source GBM/CAT snapshot: Pearson 0.997518, Spearman 0.974679, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000293, bias +0.000293, average runtime 37.6s.
- The full-sample expansion remains justified; this profile currently dominates the existing 12-source candidates on rank and calibration, subject to remaining sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:36:08+02:00` |
| elapsed | 1:20:10 |
| estimated remaining | 10:39:50 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `5d49507484`, the next remaining representative source.

## 2026-07-08T02:37:34+02:00 - Rerun 38 result / Rerun 39 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 5d49507484 --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `5d49507484983b645be5609197a5b2d3921d6ce1e4ee80f3854c54627907b4de` |
| source public score | 0.94931 |
| source original local score | 0.950322639621 |
| rerun local CV score | 0.950284123505 |
| signed error | +0.000974123505 |
| absolute error | 0.000974123505 |
| runtime | 34.0180s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T023636` |
| submission sha | `0eade783a907c8d2f4b88d34bb23f6778d2f1afb551e65daf1619b98f2bf86cf` |

Preprocessing/runtime notes:

- Completed in about 34s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Six-source GBM/CAT snapshot: Pearson 0.730560, Spearman 0.927634, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000406, bias +0.000406, average runtime 37.0s.
- The new row increases positive bias, but rank/top-k behavior remains much stronger than the existing full-sample candidates. Continue expansion.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:37:34+02:00` |
| elapsed | 1:21:36 |
| estimated remaining | 10:38:24 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `0c8ec5b2fd`, the next remaining representative source.

## 2026-07-08T02:39:17+02:00 - Rerun 39 result / Rerun 40 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 0c8ec5b2fd --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `0c8ec5b2fdc251164e95ab145577bf5ba55cc15d0d32785f22d23bd4e326119a` |
| source public score | 0.94993 |
| source original local score | 0.950659698542 |
| rerun local CV score | 0.950440714919 |
| signed error | +0.000510714919 |
| absolute error | 0.000510714919 |
| runtime | 47.0183s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T023806` |
| submission sha | `f1bbcaa70bddc6f03a4dd33c9d34e9c8a3d6d99fc1e43f744c200a8c411ffa8c` |

Preprocessing/runtime notes:

- Completed in about 47s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Seven-source GBM/CAT snapshot: Pearson 0.732224, Spearman 0.738769, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000421, bias +0.000421, average runtime 38.4s.
- Top-k weakened on this source, but the profile still beats the current 12-source no-ensemble baseline on Spearman, top-k, MAE, bias, and runtime. Continue to the remaining sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:39:17+02:00` |
| elapsed | 1:23:19 |
| estimated remaining | 10:36:41 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `48bdb4a69c`, the next remaining representative source.

## 2026-07-08T02:41:17+02:00 - Rerun 40 result / Rerun 41 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 48bdb4a69c --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `48bdb4a69c741508967a1861e266e4a5b838c96556aef7ebf2717428c93b8283` |
| source public score | 0.94979 |
| source original local score | 0.950521303749 |
| rerun local CV score | 0.950387018801 |
| signed error | +0.000597018801 |
| absolute error | 0.000597018801 |
| runtime | 46.0205s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T023953` |
| submission sha | `d15ae6429ac50f755a4fa8c91598ef8a57789d770ac572376a664638aa6dfb3a` |

Preprocessing/runtime notes:

- Completed in about 46s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eight-source GBM/CAT snapshot: Pearson 0.714088, Spearman 0.610789, top-2 hit 0.000000, top-3 hit 0.333333, MAE 0.000443, bias +0.000443, average runtime 39.4s.
- Top-k dropped to match the current no-ensemble baseline, but calibration and runtime remain better. Complete the 12-source set before judging the final tradeoff.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:41:17+02:00` |
| elapsed | 1:25:19 |
| estimated remaining | 10:34:41 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `b07a3b527a`, the next remaining representative source.

## 2026-07-08T02:43:02+02:00 - Rerun 41 result / Rerun 42 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b07a3b527a --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b07a3b527ab89743affa724b885ee944d16eb8516f882afdb7bf38699a201c8e` |
| source public score | 0.95009 |
| source original local score | 0.950560509646 |
| rerun local CV score | 0.950367330218 |
| signed error | +0.000277330218 |
| absolute error | 0.000277330218 |
| runtime | 40.0187s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T024148` |
| submission sha | `3df857c61dcf2bd3869d26758e3010b901d8f0b1ef6b31bc4457dc69441217b4` |

Preprocessing/runtime notes:

- Completed in about 40s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Nine-source GBM/CAT snapshot: Pearson 0.721498, Spearman 0.560674, top-2 hit 0.000000, top-3 hit 0.333333, MAE 0.000425, bias +0.000425, average runtime 39.5s.
- Calibration remains better than the current 12-source no-ensemble profile, but Spearman is now lower. Complete the remaining three sources before judging the tradeoff.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:43:02+02:00` |
| elapsed | 1:27:04 |
| estimated remaining | 10:32:56 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `b473cc2630`, the next remaining representative source.

## 2026-07-08T02:44:36+02:00 - Rerun 42 result / Rerun 43 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b473cc2630 --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b473cc26307f2612b3851f3be08b380800928a7ace2b5aef8639ad63d11d1066` |
| source public score | 0.94939 |
| source original local score | 0.950343393219 |
| rerun local CV score | 0.950284123505 |
| signed error | +0.000894123505 |
| absolute error | 0.000894123505 |
| runtime | 35.0168s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T024338` |
| submission sha | `3460d8a83bf8d31f6ac8f23abf9d0d9f062632aa69fbe46772ee3526a3bb2792` |

Preprocessing/runtime notes:

- Completed in about 35s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Ten-source GBM/CAT snapshot: Pearson 0.647872, Spearman 0.652451, top-2 hit 0.000000, top-3 hit 0.333333, MAE 0.000472, bias +0.000472, average runtime 39.0s.
- Spearman is now roughly tied with the current 12-source no-ensemble baseline while MAE and runtime are better. Complete the last two sources before choosing.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:44:36+02:00` |
| elapsed | 1:28:38 |
| estimated remaining | 10:31:22 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `117e38ebe5`, the next remaining representative source.

## 2026-07-08T02:46:10+02:00 - Rerun 43 result / Rerun 44 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 117e38ebe5 --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `117e38ebe54ee5691eac05df30e6d47ec4b7e9232cffc18786c6f40bb5959bed` |
| source public score | 0.95001 |
| source original local score | 0.950653008218 |
| rerun local CV score | 0.950460574770 |
| signed error | +0.000450574770 |
| absolute error | 0.000450574770 |
| runtime | 45.0189s |
| result status | ok |
| artifact dir | `logs/2-married-stallion-of-courtesy/artifacts/20260708T024504` |
| submission sha | `dcf2fcb1f65b5fd46a858ffd42de8b6461831992202845cba325e25ed66b562d` |

Preprocessing/runtime notes:

- Completed in about 45s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eleven-source GBM/CAT snapshot: Pearson 0.664516, Spearman 0.575348, top-2 hit 0.000000, top-3 hit 0.000000, MAE 0.000470, bias +0.000470, average runtime 39.6s.
- This profile is no longer clearly superior on rank/top-k, but it still has better calibration than the current 12-source no-ensemble baseline. Run the final source for the direct 12-source comparison.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:46:10+02:00` |
| elapsed | 1:30:12 |
| estimated remaining | 10:29:48 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` on `9ea9601b9a`, the final remaining representative source.

## 2026-07-08T02:47:44+02:00 - Rerun 44 result / Analysis planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9ea9601b9a --profile s6e7_fast_medium_gbmcat_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | remove anti-ranked XGB from best current no-ensemble family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9ea9601b9aca0b20f236bb624e4c4c1b46ef23c19a757339266211bacd967e1a` |
| source public score | 0.94979 |
| source original local score | 0.950477598464 |
| rerun local CV score | 0.950354906671 |
| signed error | +0.000564906671 |
| absolute error | 0.000564906671 |
| runtime | 43.0175s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T024640` |
| submission sha | `62bdaee7a80fb1ce8c6bcc81963396d81dfebd8a4b6d0c132d32128fa171d361` |

Preprocessing/runtime notes:

- Completed in about 43s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Twelve-source GBM/CAT snapshot: Pearson 0.659968, Spearman 0.657295, top-2 hit 0.000000, top-3 hit 0.000000, MAE 0.000478, bias +0.000478, average runtime 39.9s, max runtime 47.0s.
- Compared with `s6e7_fast_medium_noensemble_balanced_10m` at 12 sources, this profile slightly improves Pearson, Spearman, MAE, bias, and runtime, but it loses the top-3 hit rate. It is useful evidence, not a clear final winner.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:47:44+02:00` |
| elapsed | 1:31:46 |
| estimated remaining | 10:28:14 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Inspect rank ordering for the full 12-source GBM/CAT and three-model no-ensemble profiles, then choose the next medium-only variant aimed at improving top-k without sacrificing the rank/calibration gains.

## 2026-07-08T02:50:10+02:00 - Rank analysis result / Rerun 45 planned

Analysis performed:

- Compared public-rank order with CV-rank order for the two best 12-source profiles:
  - `s6e7_fast_medium_noensemble_balanced_10m`
  - `s6e7_fast_medium_gbmcat_noensemble_balanced_10m`
- Parsed existing GBM/CAT AutoGluon logs to pre-screen LightGBM and CatBoost validation scores before creating another model-family profile.

Findings:

- Both leading 12-source profiles over-rank two mid-public sources and push true top public sources just below the top cutoff.
- GBM/CAT slightly improves full-sample Pearson, Spearman, MAE, bias, and runtime, but loses the baseline profile's only top-3 hit.
- AutoGluon selected LightGBM as best model in inspected GBM/CAT and three-model no-ensemble runs.
- CatBoost-only is not promising from rounded log-score screening: near-zero Spearman and top-k hit rates despite decent MAE.
- A model-family change is less justified than a validation split-shape test. Holdout 30% was already poor on rank, so the next low-cost split probe is a 10% holdout using the current best three-model/no-ensemble family.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:50:10+02:00` |
| elapsed | 1:34:12 |
| estimated remaining | 10:25:48 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Add `s6e7_fast_medium_holdout10_noensemble_balanced_10m`, then run it on the first four representative sources before deciding whether to expand.

## 2026-07-08T02:52:10+02:00 - Rerun 45 result / Rerun 46 planned

Profile/config changes before this rerun:

- Added `s6e7_fast_medium_holdout10_noensemble_balanced_10m`.
- The profile keeps the seed-42 three-model no-ensemble setup and changes only `validation_fraction` from 0.2 to 0.1.
- Guardrail verification passed:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_alignment_profiles_are_medium_screening_profiles tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_medium_profile_variants -v
```

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_fast_medium_holdout10_noensemble_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_holdout10_noensemble_balanced_10m` |
| profile status | newly created |
| profile intent | smaller holdout split-shape probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.952792420169 |
| signed error | +0.002632420169 |
| absolute error | 0.002632420169 |
| runtime | 32.0144s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T025111` |
| submission sha | `769e60fbedbf39ee8d5a3a7273f9fa8d65ce5465ae7c0444f8064a4afb485fc3` |

Preprocessing/runtime notes:

- Completed in about 32s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- One-source holdout-10 snapshot: MAE 0.002632, bias +0.002632, runtime 32.0s.
- Stop this branch immediately: the first source is far more over-optimistic than the 20% holdout family, so additional ranking evidence is unlikely to justify the calibration risk.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:52:10+02:00` |
| elapsed | 1:36:12 |
| estimated remaining | 10:23:48 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Add a second 20% holdout seed variant of the best three-model no-ensemble family and test the first four representative sources.

## 2026-07-08T02:55:24+02:00 - Rerun 46 result / Rerun 47 planned

Profile/config changes before this rerun:

- Added `s6e7_fast_medium_noensemble_balanced_seed123_10m`.
- The profile is a seed-only variant of `s6e7_fast_medium_noensemble_balanced_10m`: same `medium_quality`, 600s AutoGluon limit, 600s preprocess timeout, holdout fraction 0.2, balanced sample weights, GPU XGB/GBM/CAT, no weighted ensemble, and `auto_stack: false`; only `seed` changes from 42 to 123.
- Guardrail verification passed:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_alignment_profiles_are_medium_screening_profiles tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_medium_profile_variants -v
```

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.950403057183 |
| signed error | +0.000243057183 |
| absolute error | 0.000243057183 |
| runtime | 81.0223s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T025341` |
| submission sha | `ab4059844fadb8167522a088c755da185c2f48ed7c676ec7da2766c5459ff733` |

Preprocessing/runtime notes:

- Completed in about 81s, slower than the seed-42 and seed-7 runs but still well within the 10-minute profile limit and 30-minute process timeout.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- One-source seed-123 snapshot: MAE 0.000243, bias +0.000243, runtime 81.0s.
- Calibration on the first source is competitive. Continue to the first-four source snapshot before deciding whether the slower runtime is justified.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:55:24+02:00` |
| elapsed | 1:39:26 |
| estimated remaining | 10:20:34 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `1070897a05`, the second initial representative source.

## 2026-07-08T02:57:28+02:00 - Rerun 47 result / Rerun 48 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.950371160880 |
| signed error | +0.000291160880 |
| absolute error | 0.000291160880 |
| runtime | 68.0202s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T025557` |
| submission sha | `09f50119691ecfdc241d2a32a456e2b314a37c3814f07fb86784b158bbd96a2a` |

Preprocessing/runtime notes:

- Completed in about 68s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Two-source seed-123 snapshot: Pearson 1.000000, Spearman 1.000000, top-2 hit 1.000000, MAE 0.000267, bias +0.000267, average runtime 74.5s.
- Early ranking is better than seed-7 and calibration is competitive. Continue to four sources despite the slower runtime.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:57:28+02:00` |
| elapsed | 1:41:30 |
| estimated remaining | 10:18:30 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `9f5a6e6e5d`, the third initial representative source.

## 2026-07-08T02:59:55+02:00 - Rerun 48 result / Rerun 49 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.950233340800 |
| signed error | +0.000983340800 |
| absolute error | 0.000983340800 |
| runtime | 87.0224s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T025800` |
| submission sha | `d6e226e3c759ee2ccd6c07d191c471823b507e95619effc9a0473dde0a09cbbf` |

Preprocessing/runtime notes:

- Completed in about 87s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Three-source seed-123 snapshot: Pearson 0.995175, Spearman 1.000000, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000506, bias +0.000506, average runtime 78.7s.
- Rank and top-k are strong, but calibration worsened on the low-public source. Run the fourth initial source before deciding whether the slower branch earns a full expansion.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T02:59:55+02:00` |
| elapsed | 1:43:57 |
| estimated remaining | 10:16:03 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `cdc4cd52a1`, the fourth initial representative source.

## 2026-07-08T03:02:01+02:00 - Rerun 49 result / Rerun 50 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.950308986947 |
| signed error | +0.000588986947 |
| absolute error | 0.000588986947 |
| runtime | 72.0208s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T030025` |
| submission sha | `965ae4b32c7c6da17a940fbec758073f064ce372ca826b790995108ce16ee7bd` |

Preprocessing/runtime notes:

- Completed in about 72s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Four-source seed-123 snapshot: Pearson 0.994166, Spearman 1.000000, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000527, bias +0.000527, average runtime 77.0s.
- This branch has the strongest early ranking/top-k behavior so far. Expand to all 12 representative sources despite slower runtime.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:02:01+02:00` |
| elapsed | 1:46:03 |
| estimated remaining | 10:13:57 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `f658c40156`, then continue through the remaining representative sources if top-k remains competitive.

## 2026-07-08T03:04:10+02:00 - Rerun 50 result / Rerun 51 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha f658c40156 --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `f658c401567a3af2b19f2defd5a949c75490deedd388482eea8e82c8bd36aafb` |
| source public score | 0.95012 |
| source original local score | 0.950645806537 |
| rerun local CV score | 0.950403057183 |
| signed error | +0.000283057183 |
| absolute error | 0.000283057183 |
| runtime | 66.0191s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T030236` |
| submission sha | `ab4059844fadb8167522a088c755da185c2f48ed7c676ec7da2766c5459ff733` |

Preprocessing/runtime notes:

- Completed in about 66s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Five-source seed-123 snapshot: Pearson 0.990624, Spearman 0.974679, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000478, bias +0.000478, average runtime 74.8s.
- This remains the best top-k branch so far, with acceptable calibration. Continue expansion.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:04:10+02:00` |
| elapsed | 1:48:12 |
| estimated remaining | 10:11:48 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `5d49507484`, the next remaining representative source.

## 2026-07-08T03:06:31+02:00 - Rerun 51 result / Rerun 52 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 5d49507484 --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `5d49507484983b645be5609197a5b2d3921d6ce1e4ee80f3854c54627907b4de` |
| source public score | 0.94931 |
| source original local score | 0.950322639621 |
| rerun local CV score | 0.950233340800 |
| signed error | +0.000923340800 |
| absolute error | 0.000923340800 |
| runtime | 81.0220s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T030442` |
| submission sha | `d6e226e3c759ee2ccd6c07d191c471823b507e95619effc9a0473dde0a09cbbf` |

Preprocessing/runtime notes:

- Completed in about 81s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Six-source seed-123 snapshot: Pearson 0.993238, Spearman 0.971008, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000552, bias +0.000552, average runtime 75.9s.
- MAE is now worse than the seed-42 no-ensemble full baseline, but rank and top-k remain much better. Continue expansion to see whether the top-k advantage survives the harder sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:06:31+02:00` |
| elapsed | 1:50:33 |
| estimated remaining | 10:09:27 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `0c8ec5b2fd`, the next remaining representative source.

## 2026-07-08T03:08:46+02:00 - Rerun 52 result / Rerun 53 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 0c8ec5b2fd --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `0c8ec5b2fdc251164e95ab145577bf5ba55cc15d0d32785f22d23bd4e326119a` |
| source public score | 0.94993 |
| source original local score | 0.950659698542 |
| rerun local CV score | 0.950437623082 |
| signed error | +0.000507623082 |
| absolute error | 0.000507623082 |
| runtime | 76.0218s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T030703` |
| submission sha | `b063865368435d5536c60ff137e594779f437ede03fe8e4a531ed212eff2b840` |

Preprocessing/runtime notes:

- Completed in about 76s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Seven-source seed-123 snapshot: Pearson 0.924908, Spearman 0.763763, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000546, bias +0.000546, average runtime 75.9s.
- Top-k weakened on this source, but the branch still carries better partial rank/top-k evidence than the previous profiles. Continue expansion.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:08:46+02:00` |
| elapsed | 1:52:48 |
| estimated remaining | 10:07:12 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `48bdb4a69c`, the next remaining representative source.

## 2026-07-08T03:11:13+02:00 - Rerun 53 result / Rerun 54 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 48bdb4a69c --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `48bdb4a69c741508967a1861e266e4a5b838c96556aef7ebf2717428c93b8283` |
| source public score | 0.94979 |
| source original local score | 0.950521303749 |
| rerun local CV score | 0.950267156012 |
| signed error | +0.000477156012 |
| absolute error | 0.000477156012 |
| runtime | 92.0239s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T030917` |
| submission sha | `d3372247b62a3d56a169db82e37f21e81204bfbee10900ab710f2af9f56316ea` |

Preprocessing/runtime notes:

- Completed in about 92s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eight-source seed-123 snapshot: Pearson 0.877825, Spearman 0.819337, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000537, bias +0.000537, average runtime 77.9s.
- This remains the strongest rank/top-k branch despite slower runtime and slightly worse calibration than the seed-42 baseline. Continue the remaining sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:11:13+02:00` |
| elapsed | 1:55:15 |
| estimated remaining | 10:04:45 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `b07a3b527a`, the next remaining representative source.

## 2026-07-08T03:13:54+02:00 - Rerun 54 result / Rerun 55 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b07a3b527a --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b07a3b527ab89743affa724b885ee944d16eb8516f882afdb7bf38699a201c8e` |
| source public score | 0.95009 |
| source original local score | 0.950560509646 |
| rerun local CV score | 0.950545353148 |
| signed error | +0.000455353148 |
| absolute error | 0.000455353148 |
| runtime | 94.0279s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T031149` |
| submission sha | `fefb5d568e3cb130ccdb37730224a06d723ddb35ce16e50eeb05183a43533e35` |

Preprocessing/runtime notes:

- Completed in about 94s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Nine-source seed-123 snapshot: Pearson 0.810733, Spearman 0.789944, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000528, bias +0.000528, average runtime 79.7s.
- The branch no longer preserves top-2, but it still has better Spearman and top-3 hit than the current 12-source profiles. Continue to the final three sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:13:54+02:00` |
| elapsed | 1:57:56 |
| estimated remaining | 10:02:04 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `b473cc2630`, the next remaining representative source.

## 2026-07-08T03:16:16+02:00 - Rerun 55 result / Rerun 56 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b473cc2630 --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b473cc26307f2612b3851f3be08b380800928a7ace2b5aef8639ad63d11d1066` |
| source public score | 0.94939 |
| source original local score | 0.950343393219 |
| rerun local CV score | 0.950233340800 |
| signed error | +0.000843340800 |
| absolute error | 0.000843340800 |
| runtime | 68.0213s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T031432` |
| submission sha | `d6e226e3c759ee2ccd6c07d191c471823b507e95619effc9a0473dde0a09cbbf` |

Preprocessing/runtime notes:

- Completed in about 68s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Ten-source seed-123 snapshot: Pearson 0.837545, Spearman 0.837022, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000560, bias +0.000560, average runtime 78.5s.
- This is the best Spearman/top-3 branch so far, with higher bias and slower runtime. Run the final two representative sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:16:16+02:00` |
| elapsed | 2:00:18 |
| estimated remaining | 9:59:42 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `117e38ebe5`, the next remaining representative source.

## 2026-07-08T03:18:28+02:00 - Rerun 56 result / Rerun 57 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 117e38ebe5 --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `117e38ebe54ee5691eac05df30e6d47ec4b7e9232cffc18786c6f40bb5959bed` |
| source public score | 0.95001 |
| source original local score | 0.950653008218 |
| rerun local CV score | 0.950304363297 |
| signed error | +0.000294363297 |
| absolute error | 0.000294363297 |
| runtime | 63.0219s |
| result status | ok |
| artifact dir | `logs/2-married-stallion-of-courtesy/artifacts/20260708T031653` |
| submission sha | `3feeffd35ed595a54ba030031e5b82d0dd80d791c01b24751c1066d6c29bb4d9` |

Preprocessing/runtime notes:

- Completed in about 63s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eleven-source seed-123 snapshot: Pearson 0.791844, Spearman 0.813847, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000536, bias +0.000536, average runtime 77.1s.
- The profile remains the strongest top-3 branch with materially better Spearman than the other 12-source candidates. Run the final source for direct comparison.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:18:28+02:00` |
| elapsed | 2:02:30 |
| estimated remaining | 9:57:30 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_noensemble_balanced_seed123_10m` on `9ea9601b9a`, the final remaining representative source.

## 2026-07-08T03:20:42+02:00 - Rerun 57 result / Analysis planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9ea9601b9a --profile s6e7_fast_medium_noensemble_balanced_seed123_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_noensemble_balanced_seed123_10m` |
| profile status | newly created |
| profile intent | holdout seed sensitivity probe for best current family |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9ea9601b9aca0b20f236bb624e4c4c1b46ef23c19a757339266211bacd967e1a` |
| source public score | 0.94979 |
| source original local score | 0.950477598464 |
| rerun local CV score | 0.950136708881 |
| signed error | +0.000346708881 |
| absolute error | 0.000346708881 |
| runtime | 62.0239s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T031910` |
| submission sha | `f2a2625c5914a9554ada1c7e5a368fbb59b02d1a5ccbf0b8f6245309e42591c9` |

Preprocessing/runtime notes:

- Completed in about 62s.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Twelve-source seed-123 snapshot: Pearson 0.682998, Spearman 0.759736, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000520, bias +0.000520, average runtime 75.9s, max runtime 94.0s.
- This is the strongest 12-source rank/top-k proxy so far. It beats the seed-42 no-ensemble profile on Pearson, Spearman, top-3, MAE, and bias, but is slower. It beats GBM/CAT on Spearman/top-3 while trailing it on MAE and runtime.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:20:42+02:00` |
| elapsed | 2:04:44 |
| estimated remaining | 9:55:16 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Inspect seed-123 model-selection logs to decide whether a cheaper seed-123 GBM/CAT or GBM-only variant is justified.

## 2026-07-08T03:21:44+02:00 - Model-selection analysis result / Rerun 58 planned

Analysis performed:

- Parsed `autogluon_stdout.log` for all 12 `s6e7_fast_medium_noensemble_balanced_seed123_10m` reruns.

Findings:

- AutoGluon selected `XGBoost` as the best model on all 12 seed-123 sources.
- The seed-123 rank/top-k signal appears to come from the XGB model, not from LightGBM or CatBoost.
- The existing seed-42 XGB-only profile was anti-correlated, so the key difference is the holdout seed, not the model family by itself.
- A seed-123 XGB-only profile is now justified: it should preserve the seed-123 rank signal if it reproduces the XGB validation scores, while avoiding the extra GBM/CAT training time.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:21:44+02:00` |
| elapsed | 2:05:46 |
| estimated remaining | 9:54:14 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Add `s6e7_fast_medium_xgb_seed123_balanced_10m`, then run it first on `4d2b8df165` to confirm it matches the full seed-123 profile's XGB validation score with lower runtime.

## 2026-07-08T03:24:09+02:00 - Rerun 58 result / Rerun 59 planned

Profile/config changes before this rerun:

- Added `s6e7_fast_medium_xgb_seed123_balanced_10m`.
- The profile uses `medium_quality`, 600s AutoGluon limit, 600s preprocess timeout, holdout fraction 0.2, seed 123, balanced sample weights, GPU XGB only, no weighted ensemble, and `auto_stack: false`.
- Guardrail verification passed:

```bash
uv run pytest tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_alignment_profiles_are_medium_screening_profiles tests/test_autogluon_preprocess.py::test_autogluon_s6e7_fast_medium_profile_variants -v
```

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 4d2b8df165 --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `4d2b8df1653886934840da10e99c2f363b03163a31c2949c1daec2c6c983e601` |
| source public score | 0.95016 |
| source original local score | 0.950537217870 |
| rerun local CV score | 0.950403057183 |
| signed error | +0.000243057183 |
| absolute error | 0.000243057183 |
| runtime | 35.0175s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T032301` |
| submission sha | `ab4059844fadb8167522a088c755da185c2f48ed7c676ec7da2766c5459ff733` |

Preprocessing/runtime notes:

- Completed in about 35s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- One-source XGB seed-123 snapshot: MAE 0.000243, bias +0.000243, runtime 35.0s.
- This confirms the XGB-only profile can reproduce the full seed-123 result for the first source with much lower runtime. Expand to all 12 representative sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:24:09+02:00` |
| elapsed | 2:08:11 |
| estimated remaining | 9:51:49 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `1070897a05`, the second representative source.

## 2026-07-08T03:25:53+02:00 - Rerun 59 result / Rerun 60 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 1070897a05 --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `1070897a05ac088919acd4c321348963dd2ef669c4a740b4209e212ebd5296d8` |
| source public score | 0.95008 |
| source original local score | 0.950564387316 |
| rerun local CV score | 0.950371160880 |
| signed error | +0.000291160880 |
| absolute error | 0.000291160880 |
| runtime | 31.0175s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T032452` |
| submission sha | `09f50119691ecfdc241d2a32a456e2b314a37c3814f07fb86784b158bbd96a2a` |

Preprocessing/runtime notes:

- Completed in about 31s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Two-source XGB seed-123 snapshot: Pearson 1.000000, Spearman 1.000000, top-2 hit 1.000000, MAE 0.000267, bias +0.000267, average runtime 33.0s.
- Continue expansion; this is reproducing the full seed-123 signal with materially lower runtime.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:25:53+02:00` |
| elapsed | 2:09:55 |
| estimated remaining | 9:50:05 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `9f5a6e6e5d`, the third representative source.

## 2026-07-08T03:27:27+02:00 - Rerun 60 result / Rerun 61 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9f5a6e6e5d --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9f5a6e6e5d4a1dd5ad2cfda8b636e9d1e932d965360fd78cddd256db350fe4ea` |
| source public score | 0.94925 |
| source original local score | 0.950373586439 |
| rerun local CV score | 0.950233340800 |
| signed error | +0.000983340800 |
| absolute error | 0.000983340800 |
| runtime | 26.0165s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T032633` |
| submission sha | `d6e226e3c759ee2ccd6c07d191c471823b507e95619effc9a0473dde0a09cbbf` |

Preprocessing/runtime notes:

- Completed in about 26s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Three-source XGB seed-123 snapshot: Pearson 0.995175, Spearman 1.000000, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000506, bias +0.000506, average runtime 30.7s.
- Continue expansion; this profile is tracking the full seed-123 metrics at much lower runtime.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:27:27+02:00` |
| elapsed | 2:11:29 |
| estimated remaining | 9:48:31 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `cdc4cd52a1`, the fourth representative source.

## 2026-07-08T03:28:57+02:00 - Rerun 61 result / Rerun 62 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha cdc4cd52a1 --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `cdc4cd52a12fc57152ea36838f9ce7657ff8565c9795f423d40c25af22ae8bde` |
| source public score | 0.94972 |
| source original local score | 0.950669628467 |
| rerun local CV score | 0.950308986947 |
| signed error | +0.000588986947 |
| absolute error | 0.000588986947 |
| runtime | 24.0152s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T032806` |
| submission sha | `965ae4b32c7c6da17a940fbec758073f064ce372ca826b790995108ce16ee7bd` |

Preprocessing/runtime notes:

- Completed in about 24s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Four-source XGB seed-123 snapshot: Pearson 0.994166, Spearman 1.000000, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000527, bias +0.000527, average runtime 29.0s.
- Continue full expansion; this profile preserves the best rank/top-k signal while now running faster than all other full-model candidates.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:28:57+02:00` |
| elapsed | 2:12:59 |
| estimated remaining | 9:47:01 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `f658c40156`, the next representative source.

## 2026-07-08T03:30:51+02:00 - Rerun 62 result / Rerun 63 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha f658c40156 --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `f658c401567a3af2b19f2defd5a949c75490deedd388482eea8e82c8bd36aafb` |
| source public score | 0.95012 |
| source original local score | 0.950645806537 |
| rerun local CV score | 0.950403057183 |
| signed error | +0.000283057183 |
| absolute error | 0.000283057183 |
| runtime | 36.0161s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T032941` |
| submission sha | `ab4059844fadb8167522a088c755da185c2f48ed7c676ec7da2766c5459ff733` |

Preprocessing/runtime notes:

- Completed in about 36s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Five-source XGB seed-123 snapshot: Pearson 0.990624, Spearman 0.974679, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000478, bias +0.000478, average runtime 30.4s.
- Continue expansion; runtime is now competitive with the fastest candidates while preserving the seed-123 ranking.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:30:51+02:00` |
| elapsed | 2:14:53 |
| estimated remaining | 9:45:07 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `5d49507484`, the next representative source.

## 2026-07-08T03:32:39+02:00 - Rerun 63 result / Rerun 64 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 5d49507484 --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `5d49507484983b645be5609197a5b2d3921d6ce1e4ee80f3854c54627907b4de` |
| source public score | 0.94931 |
| source original local score | 0.950322639621 |
| rerun local CV score | 0.950233340800 |
| signed error | +0.000923340800 |
| absolute error | 0.000923340800 |
| runtime | 26.0156s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T033128` |
| submission sha | `d6e226e3c759ee2ccd6c07d191c471823b507e95619effc9a0473dde0a09cbbf` |

Preprocessing/runtime notes:

- Completed in about 26s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Six-source XGB seed-123 snapshot: Pearson 0.993238, Spearman 0.971008, top-2 hit 1.000000, top-3 hit 1.000000, MAE 0.000552, bias +0.000552, average runtime 29.7s.
- Continue expansion; this candidate currently combines the best rank/top-k signal with fast runtime.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:32:39+02:00` |
| elapsed | 2:16:41 |
| estimated remaining | 9:43:19 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `0c8ec5b2fd`, the next representative source.

## 2026-07-08T03:34:23+02:00 - Rerun 64 result / Rerun 65 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 0c8ec5b2fd --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `0c8ec5b2fdc251164e95ab145577bf5ba55cc15d0d32785f22d23bd4e326119a` |
| source public score | 0.94993 |
| source original local score | 0.950659698542 |
| rerun local CV score | 0.950437623082 |
| signed error | +0.000507623082 |
| absolute error | 0.000507623082 |
| runtime | 30.0151s |
| result status | ok |
| artifact dir | `logs/2-whimsical-albatross-from-camelot/artifacts/20260708T033314` |
| submission sha | `b063865368435d5536c60ff137e594779f437ede03fe8e4a531ed212eff2b840` |

Preprocessing/runtime notes:

- Completed in about 30s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Seven-source XGB seed-123 snapshot: Pearson 0.924908, Spearman 0.763763, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000546, bias +0.000546, average runtime 29.7s.
- Continue expansion; XGB-only seed-123 is now the leading candidate if the 12-source metrics match the full seed-123 profile.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:34:23+02:00` |
| elapsed | 2:18:25 |
| estimated remaining | 9:41:35 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `48bdb4a69c`, the next representative source.

## 2026-07-08T03:35:57+02:00 - Rerun 65 result / Rerun 66 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 48bdb4a69c --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `48bdb4a69c741508967a1861e266e4a5b838c96556aef7ebf2717428c93b8283` |
| source public score | 0.94979 |
| source original local score | 0.950521303749 |
| rerun local CV score | 0.950267156012 |
| signed error | +0.000477156012 |
| absolute error | 0.000477156012 |
| runtime | 31.0163s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T033456` |
| submission sha | `d3372247b62a3d56a169db82e37f21e81204bfbee10900ab710f2af9f56316ea` |

Preprocessing/runtime notes:

- Completed in about 31s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eight-source XGB seed-123 snapshot: Pearson 0.877825, Spearman 0.819337, top-2 hit 0.500000, top-3 hit 0.666667, MAE 0.000537, bias +0.000537, average runtime 29.9s.
- Continue expansion to the last four sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:35:57+02:00` |
| elapsed | 2:19:59 |
| estimated remaining | 9:40:01 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `b07a3b527a`, the next representative source.

## 2026-07-08T03:37:34+02:00 - Rerun 66 result / Rerun 67 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b07a3b527a --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b07a3b527ab89743affa724b885ee944d16eb8516f882afdb7bf38699a201c8e` |
| source public score | 0.95009 |
| source original local score | 0.950560509646 |
| rerun local CV score | 0.950545353148 |
| signed error | +0.000455353148 |
| absolute error | 0.000455353148 |
| runtime | 35.0178s |
| result status | ok |
| artifact dir | `logs/2-romantic-guan-of-eternity/artifacts/20260708T033629` |
| submission sha | `fefb5d568e3cb130ccdb37730224a06d723ddb35ce16e50eeb05183a43533e35` |

Preprocessing/runtime notes:

- Completed in about 35s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Nine-source XGB seed-123 snapshot: Pearson 0.810733, Spearman 0.789944, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000528, bias +0.000528, average runtime 30.5s.
- Continue the final three sources for the full comparison.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:37:34+02:00` |
| elapsed | 2:21:36 |
| estimated remaining | 9:38:24 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `b473cc2630`, the next representative source.

## 2026-07-08T03:39:01+02:00 - Rerun 67 result / Rerun 68 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha b473cc2630 --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `b473cc26307f2612b3851f3be08b380800928a7ace2b5aef8639ad63d11d1066` |
| source public score | 0.94939 |
| source original local score | 0.950343393219 |
| rerun local CV score | 0.950233340800 |
| signed error | +0.000843340800 |
| absolute error | 0.000843340800 |
| runtime | 27.0136s |
| result status | ok |
| artifact dir | `logs/2-smiling-topaz-oarfish/artifacts/20260708T033808` |
| submission sha | `d6e226e3c759ee2ccd6c07d191c471823b507e95619effc9a0473dde0a09cbbf` |

Preprocessing/runtime notes:

- Completed in about 27s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Ten-source XGB seed-123 snapshot: Pearson 0.837545, Spearman 0.837022, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000560, bias +0.000560, average runtime 30.1s.
- Continue to the final two sources.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:39:01+02:00` |
| elapsed | 2:23:03 |
| estimated remaining | 9:36:57 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `117e38ebe5`, the next representative source.

## 2026-07-08T03:40:32+02:00 - Rerun 68 result / Rerun 69 planned

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 117e38ebe5 --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `117e38ebe54ee5691eac05df30e6d47ec4b7e9232cffc18786c6f40bb5959bed` |
| source public score | 0.95001 |
| source original local score | 0.950653008218 |
| rerun local CV score | 0.950304363297 |
| signed error | +0.000294363297 |
| absolute error | 0.000294363297 |
| runtime | 31.0155s |
| result status | ok |
| artifact dir | `logs/2-married-stallion-of-courtesy/artifacts/20260708T033933` |
| submission sha | `3feeffd35ed595a54ba030031e5b82d0dd80d791c01b24751c1066d6c29bb4d9` |

Preprocessing/runtime notes:

- Completed in about 31s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Profile ranking impact:

- Eleven-source XGB seed-123 snapshot: Pearson 0.791844, Spearman 0.813847, top-2 hit 0.000000, top-3 hit 0.666667, MAE 0.000536, bias +0.000536, average runtime 30.2s.
- Run the final source for the direct 12-source comparison.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:40:32+02:00` |
| elapsed | 2:24:34 |
| estimated remaining | 9:35:26 |
| 12-hour budget reached? | no |
| enough time for another 10m AutoGluon rerun? | yes |

Next planned experiment:

- Run `s6e7_fast_medium_xgb_seed123_balanced_10m` on `9ea9601b9a`, the final representative source.

## 2026-07-08T03:42:01+02:00 - Rerun 69 result / Selection

Command executed:

```bash
uv run python scripts/rerun_autogluon_profile.py --competition playground-series-s6e7 --sha 9ea9601b9a --profile s6e7_fast_medium_xgb_seed123_balanced_10m --timeout 1800 --execute
```

Result:

| field | value |
|---|---:|
| profile | `s6e7_fast_medium_xgb_seed123_balanced_10m` |
| profile status | newly created |
| profile intent | preserve seed-123 XGB rank signal with lower runtime |
| AutoGluon preset | `medium_quality` |
| AutoGluon time limit | 600s |
| process timeout | 1800s |
| source sha | `9ea9601b9aca0b20f236bb624e4c4c1b46ef23c19a757339266211bacd967e1a` |
| source public score | 0.94979 |
| source original local score | 0.950477598464 |
| rerun local CV score | 0.950136708881 |
| signed error | +0.000346708881 |
| absolute error | 0.000346708881 |
| runtime | 25.0163s |
| result status | ok |
| artifact dir | `logs/2-vociferous-tortoise-of-perspective/artifacts/20260708T034106` |
| submission sha | `f2a2625c5914a9554ada1c7e5a368fbb59b02d1a5ccbf0b8f6245309e42591c9` |

Preprocessing/runtime notes:

- Completed in about 25s.
- Matched the full seed-123 profile's local CV score and submission hash for this source.
- No active rerun process remained after completion.
- Refreshed `/tmp/aideml_kaggle_submission_lab_full.json`, `logs/autogluon_fast_profile_cv_public_summary.json`, `logs/autogluon_fast_profile_cv_public_summary.csv`, and `logs/autogluon_fast_profile_cv_public_sources.csv`.

Final 12-source comparison:

| profile | n | Pearson | Spearman | top-2 hit | top-3 hit | MAE | bias | avg runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `s6e7_fast_medium_xgb_seed123_balanced_10m` | 12 | 0.682998 | 0.759736 | 0.000000 | 0.666667 | 0.000520 | +0.000520 | 29.8s |
| `s6e7_fast_medium_noensemble_balanced_seed123_10m` | 12 | 0.682998 | 0.759736 | 0.000000 | 0.666667 | 0.000520 | +0.000520 | 75.9s |
| `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` | 12 | 0.659968 | 0.657295 | 0.000000 | 0.000000 | 0.000478 | +0.000478 | 39.9s |
| `s6e7_fast_medium_noensemble_balanced_10m` | 12 | 0.519178 | 0.651490 | 0.000000 | 0.333333 | 0.000527 | +0.000527 | 49.6s |
| `s6e7_align_holdout_balanced_gpu_10m` | 12 | 0.557772 | 0.325745 | 0.000000 | 0.000000 | 0.000645 | +0.000645 | 55.3s |

Selection:

- Select `s6e7_fast_medium_xgb_seed123_balanced_10m` as the current fast AutoGluon CV/public proxy profile.
- It uses `medium_quality`, `time_limit: 600`, `preprocess_timeout: 600`, and no Kaggle submissions were made.
- It matches the strongest seed-123 rank/top-k evidence while reducing average runtime from 75.9s to 29.8s.
- It is not the lowest-MAE profile; `s6e7_fast_medium_gbmcat_noensemble_balanced_10m` has lower MAE, but it loses the top-3 hit signal. For selecting promising candidates, rank/top-k behavior is the more useful proxy.

Budget check:

| field | value |
|---|---:|
| timestamp | `2026-07-08T03:42:01+02:00` |
| elapsed | 2:26:03 |
| estimated remaining | 9:33:57 |
| 12-hour budget reached? | no |
| stop rule met? | yes: full 12-source medium/600s candidate selected with best rank/top-k and fast runtime |

Next step:

- Run verification, inspect git status, and commit only the current-task changes.
