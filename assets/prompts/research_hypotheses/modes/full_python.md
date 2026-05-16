Allowed implementation scope:
- The generated hypotheses should target a full Python ML solution script.
- Hypotheses may include custom validation, model training, fold-specific
  preprocessing, out-of-fold encodings, model blending, calibration, and longer
  experimental pipelines when justified.

Good hypotheses for this scope:
- concrete feature, modeling, validation, calibration, or ensembling changes
- fold-safe target/statistical encodings when the leakage handling is explicit
- grouped, time-aware, or otherwise task-appropriate validation changes
- bounded model or blending changes with a clear first experiment
- robust preprocessing changes that reduce leakage, instability, or train/test shift

Bad hypotheses for this scope:
- vague "try more models" advice
- broad hyperparameter sweeps without a narrow first test
- large ensembles without an ablation path
- changes that optimize directly on public leaderboard score
- instructions like "start from node X" or "copy step Y"

Scope-specific quality rules:
- Broader modeling is allowed, but avoid vague "try more models" unless it is
  tied to a specific experiment.
- Keep each hypothesis practical enough to become one AIDE coding-agent prompt
  for a full Python ML solution script.
