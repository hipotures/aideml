# AIDE refactor runtime API

The refactored solution must use this existing runtime module.

Do not reimplement these helpers in the refactored solution.

## Imports

Use only the names you need:

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

## Stage wrapping

```python
with aide_stage("load_data_stage"):
    train = load_csv("train.csv")
    test = load_csv("test.csv")
```

or:

```python
@aide_stage("build_features_stage")
def build_features_stage(train_full, test):
    ...
    return train_full, test
```

## Required finalization

```python
if __name__ == "__main__":
    try:
        make_submission()
    finally:
        finalize_aide_artifacts()
```

## Prediction cache wrapper

Only use when the original code has a clean fold-local fit/predict block.

```python
contract = build_prediction_contract(
    model_family="xgboost",
    variant_id=variant_id,
    fold_id=fold_idx,
    class_order=class_order,
    feature_cols=feature_cols,
    model_params=params,
    fold_indices=valid_idx,
    target_values=y_tr_code,
    sample_weight=model_sw,
    extra={"valid_rows": len(valid_idx), "test_rows": len(X_test)},
)

def compute_xgb_predictions():
    model = fit_xgb_with_fallback(...)
    val_prob = model.predict_proba(X_val_xgb)
    test_prob = model.predict_proba(xgb_test_df)
    return {
        "valid_idx": valid_idx,
        "valid_proba": val_prob,
        "test_proba": test_prob,
        "meta": {"model_family": "xgboost", "variant_id": variant_id, "fold_id": fold_idx},
    }

cached = cached_fold_prediction(
    model_family="xgboost",
    variant_id=variant_id,
    fold_id=fold_idx,
    contract=contract,
    compute_fn=compute_xgb_predictions,
)
```

When `AIDE_CACHE_MODE=off`, this computes normally and does not read/write shared prediction arrays.
