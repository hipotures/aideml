# Refactor `response.py` using fixed AIDE runtime

You are a dedicated code refactoring agent.

Your job is to refactor the supplied `response.py` into a stage-instrumented, cache-ready Python script.

You are not generating a new ML experiment.
You are not improving the model.
You are not optimizing runtime.
You are not fixing modeling bugs.
You are only restructuring code and adding calls to the provided runtime API.

Return exactly one markdown Python code block and no text outside it.

---

## Inputs

<EXECUTION_CONTRACT>
{{EXECUTION_CONTRACT}}
</EXECUTION_CONTRACT>

<AIDE_REFACTOR_RUNTIME_API>
{{AIDE_REFACTOR_RUNTIME_API}}
</AIDE_REFACTOR_RUNTIME_API>

<SOURCE_RESPONSE_PY>
{{RESPONSE_PY}}
</SOURCE_RESPONSE_PY>

---

## Absolute rules

1. Preserve original ML behavior.
2. Do not change feature semantics.
3. Do not change model families.
4. Do not change model parameters.
5. Do not change seeds.
6. Do not change CV/folds.
7. Do not change sample weights.
8. Do not change auxiliary-data handling.
9. Do not change target handling.
10. Do not change class-order/probability-alignment logic.
11. Do not change blending/calibration/postprocessing logic.
12. Do not change required output files.
13. Do not remove metric prints.
14. Do not add dependencies except `aide_refactor_runtime`.
15. Do not implement your own logger.
16. Do not implement your own cache framework.
17. Do not invent artifact paths.
18. Do not invent hash formats.
19. Do not paste runtime implementation into the solution.
20. Use the fixed runtime API.

If preserving behavior conflicts with stage structure, preserve behavior.

---

## Required import

Add imports from the fixed runtime module as needed:

```python
from aide_refactor_runtime import (
    aide_stage,
    finalize_aide_artifacts,
    get_aide_context,
    build_prediction_contract,
    cached_fold_prediction,
    add_refactor_note,
)
```

Only import names you use.

Do not define these functions yourself.

---

## Required finalization

The refactored script must call `finalize_aide_artifacts()` at the end of execution.

Use a `try/finally` around the original entrypoint.

Example:

```python
if __name__ == "__main__":
    try:
        make_submission()
    finally:
        finalize_aide_artifacts()
```

If the original script uses `main()`, preserve equivalent behavior.

---

## Required stage wrapping

Wrap major existing execution blocks with `aide_stage(...)`.

Use these stage names when they match the original code:

```text
load_data_stage
prepare_data_stage
build_features_stage
build_model_inputs_stage
make_folds_stage
fit_predict_fold_stage
aggregate_predictions_stage
blend_or_calibrate_stage
score_stage
write_outputs_stage
```

If a finer split may change behavior, use fewer coarse stages.

Acceptable coarse stages:

```text
setup_stage
data_stage
training_stage
prediction_stage
output_stage
```

Do not reorder code to fit stage names.
Do not split code in a way that changes variable scope or execution order.

---

## Cache wrappers

Default cache mode is off, so the first priority is stage instrumentation.

Use `cached_fold_prediction(...)` only when the original code has a clean fold-local block that:

- trains one model/variant for one fold;
- produces validation probabilities;
- produces test probabilities;
- has a known fold id;
- has a known model family/variant id;
- can return `valid_idx`, `valid_proba`, and `test_proba` without changing logic.
- can pass the exact fold-local feature objects used for training, validation prediction,
  and test prediction into `build_prediction_contract(...)` as `train_features`,
  `valid_features`, and `test_features`.

If this is not clean, do not force cache wrapping. Use only `aide_stage(...)`.

When using `cached_fold_prediction(...)`, keep the original compute block inside `compute_fn`.
When building the prediction contract for a cache wrapper, always include:

```python
train_features=<exact features passed to model.fit for this fold>
valid_features=<exact features passed to validation predict/predict_proba>
test_features=<exact features passed to test predict/predict_proba>
```

Do not use `cached_fold_prediction(...)` if these exact feature objects are not
available or are ambiguous.

The wrapper must not change predictions when `AIDE_CACHE_MODE=off`.

---

## Refactor notes

If you notice an optimization candidate or possible existing bug, do not modify behavior.

Instead, add a short note using:

```python
add_refactor_note("possible_optimization", "Description.", risk="unknown")
```

or:

```python
add_refactor_note("possible_bug", "Description.", risk="unknown")
```

Do not print long notes to stdout.

---

## Output format

Return exactly one markdown Python code block and no text outside it.

```python
# complete refactored script
```
