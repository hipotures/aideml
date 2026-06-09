# Execution contract for refactor pass

The refactor model must not receive the full original generation prompt.

## Task

Refactor the provided `response.py` into a stage-instrumented Python script that uses the fixed `aide_refactor_runtime.py` module.

## Preserve

- original ML behavior;
- original feature engineering;
- original model families;
- original model parameters and seeds;
- original CV/fold construction;
- original target handling;
- original auxiliary-data handling;
- original sample-weight logic;
- original class-order and probability-alignment logic;
- original blending, calibration, postprocessing, and scoring;
- all output files written by the original script;
- original script entrypoint behavior.

## Strict mode

Do not improve the solution.
Do not tune models.
Do not optimize runtime.
Do not fix modeling bugs.
Do not invent new experiments.

## Cache

Default cache mode is `AIDE_CACHE_MODE=off`.

The refactored code must remain correct if all cache operations are disabled or fail.

In phase 1, the refactored code is saved as a sidecar artifact and is not used to replace the original code.
