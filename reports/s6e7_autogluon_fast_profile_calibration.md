# S6E7 AutoGluon fast-profile calibration

Session: `s6e7_profile_20260709T204443Z`  
Window: 2026-07-09 20:44:43Z to 2026-07-10 08:44:43Z  
Metric: balanced accuracy; source of truth for external performance: completed Kaggle public scores.

## Protocol

- Every candidate used exactly `[XGB, GBM, CAT]`, `medium_quality`, and a
  configured AutoGluon `time_limit` of at most 600 seconds.
- A run counted only when all three families actually trained and were
  inferable. Source solution hashes were checked before and after every run.
- Six source artifacts were frozen for development and four disjoint artifacts
  were frozen for confirmation before finalist selection. No Kaggle submission
  was made.
- Final-process extraction was delegated to the read-only `process_monitor`
  agent. Each monitor assignment matched a post-launch JSONL completion record
  by source, profile, timestamp, and pre-launch line count.

## Development results

All rows below are complete 6/6 development panels with 100% required-family
eligibility.

| Profile | Spearman | Pearson | Median runtime (s) | Result |
| --- | ---: | ---: | ---: | --- |
| 20% holdout, unweighted CPU, seed 1729 | 0.735 | 0.800 | 471.0 | Initial leader |
| 20% holdout, unweighted CUDA, seed 1729 | 0.647 | 0.569 | 341.5 | Finalist for speed/positive rank |
| 20% holdout, balanced CUDA | -0.493 | -0.250 | 40.7 | Rejected |
| 15% holdout, balanced CUDA | -0.754 | -0.735 | 48.6 | Rejected |
| 25% holdout, balanced CUDA | -0.319 | -0.153 | 48.5 | Rejected |
| 20% holdout, balanced CUDA ensemble | -0.783 | -0.476 | 40.7 | Rejected |
| 3-fold bagged OOF, unweighted CUDA | -0.580 | -0.257 | 622.3 | Rejected; wall-clock exceeded 600 s |
| 40% holdout, unweighted CPU, seed 1729 | 0.000 | 0.715 | 558.2 | Failed preregistered stability gate |

The two positive-rank 20% unweighted profiles were frozen before the separate
confirmation panel.

## Confirmation result

| Frozen finalist | Valid | Spearman | Pearson | Median runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| `s6e7_calibration_reference_holdout20_unweighted_cpu_capped180_fairone_seed1729_10m` | 4/4 | 0.800 | 0.507 | 475.6 |
| `s6e7_calibration_reference_holdout20_unweighted_gpu_cuda_fairone_seed1729_10m` | 4/4 | -0.400 | -0.230 | 259.5 |

The CPU profile was therefore selected as the conditional reference profile.
It is the only frozen finalist with positive confirmation rank agreement.

## Seed-stability audit

The selected CPU profile was rerun on the full development panel with two
independent holdout seeds. All 18 runs were valid and all three model families
were eligible.

| Seed | Spearman | Pearson | Median runtime (s) |
| ---: | ---: | ---: | ---: |
| 1729 | 0.735 | 0.800 | 471.0 |
| 2718 | -0.912 | -0.576 | 389.9 |
| 31415 | -0.529 | -0.201 | 456.0 |

This is the central limitation. The development-panel public-score range was
only 0.00065, while the mean within-source standard deviation across the three
seed scores was 0.000857 (maximum 0.001342). Averaging the three seed scores
did not repair rank agreement (Spearman -0.529); their median was also negative
(Spearman -0.765).

## Decision

Use the selected CPU seed-1729 profile only as the best **conditional** local
screening reference observed under this protocol. Do not treat it as a robust
ranker of source artifacts whose public scores differ by only a few ten-thousandths.
The independent confirmation supports that exact configuration, but the
alternate-seed audit shows that a single random holdout is not stable enough to
claim general rank calibration.

Further candidate fishing stopped at the preregistered 40%-holdout gate. A
credible next study needs either a wider-spread set of externally scored source
artifacts or independent labels; the existing source-score spread is below the
observed split-seed noise.

## Reproducibility

- Structured events: `logs/autogluon_fast_profile_cv_public/s6e7_profile_20260709T204443Z/experiments.jsonl`
- Decisions: `logs/autogluon_fast_profile_cv_public/s6e7_profile_20260709T204443Z/decisions.jsonl`
- Aggregate metrics: `logs/autogluon_fast_profile_cv_public/s6e7_profile_20260709T204443Z/profile_metrics.json`
- Frozen source panels: `logs/autogluon_fast_profile_cv_public/s6e7_profile_20260709T204443Z/source_sets.json`
