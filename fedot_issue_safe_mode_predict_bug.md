# `safe_mode=True` causes `IndexError` during `predict()` — scaling node column index mismatch

## Description

When using `Fedot` with `safe_mode=True`, the `predict()` call fails with an `IndexError` in the `ScalingImplementation` node. The pipeline trains successfully, but at prediction time the `scaling` node's internal `bool_ids` contain column indices that exceed the actual feature count of the test data.

The root cause is an inconsistency between training-time and prediction-time data preprocessing. During internal cross-validation fitting, the pipeline preprocessor expands the feature space (e.g. from 16 to 73 columns). With `safe_mode=True`, the API-level `data_processor.transform()` at prediction time does **not** reproduce this expansion, so the test data arrives at the pipeline with the original column count — triggering the out-of-bounds access in `EncodedInvariantImplementation._make_new_table()`.

With `safe_mode=False` (the default), the API-level predict preprocessing **does** expand the test data to match the training-time feature count, and prediction succeeds.

## FEDOT version

```
0.7.5
```

## Minimal reproducible example

```python
import pandas as pd
import numpy as np
from fedot.api.main import Fedot
from sklearn.datasets import make_regression

# Generate a simple regression dataset with 16 float features
X, y = make_regression(n_samples=1000, n_features=16, noise=0.1, random_state=42)
X_train = pd.DataFrame(X, columns=[f"f{i}" for i in range(16)])
y_train = pd.Series(y, name="target")
X_test = X_train.copy()

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")

# --- safe_mode=True → FAILS ---
model = Fedot(
    problem="regression",
    timeout=0.5,
    metric="mae",
    preset="fast_train",
    with_tuning=False,
    safe_mode=True,
    n_jobs=1,
)
model.fit(X_train, y_train)

# Inspect corrupted state
for node in model.current_pipeline.nodes:
    fitted = node.fitted_operation
    if hasattr(fitted, "bool_ids"):
        print(
            f"[safe_mode=True] {node.operation.operation_type}: "
            f"bool_ids len={len(fitted.bool_ids)}, "
            f"max={max(fitted.bool_ids) if fitted.bool_ids else None}, "
            f"ids_to_process={fitted.ids_to_process}"
        )

try:
    preds = model.predict(X_test)
    print("[safe_mode=True] predict SUCCESS")
except IndexError as e:
    print(f"[safe_mode=True] predict FAILED: {e}")

# --- safe_mode=False → WORKS ---
model2 = Fedot(
    problem="regression",
    timeout=0.5,
    metric="mae",
    preset="fast_train",
    with_tuning=False,
    safe_mode=False,
    n_jobs=1,
)
model2.fit(X_train, y_train)

try:
    preds2 = model2.predict(X_test)
    print(f"[safe_mode=False] predict SUCCESS, shape={preds2.shape}")
except IndexError as e:
    print(f"[safe_mode=False] predict FAILED: {e}")
```

## Expected behavior

`Fedot.predict()` should succeed regardless of the `safe_mode` setting, producing predictions with shape `(n_samples,)`.

## Actual behavior

With `safe_mode=True`, prediction crashes:

```
IndexError: index 16 is out of bounds for axis 1 with size 16
```

Full traceback (abbreviated):

```
fedot/core/operations/evaluation/operation_implementations/implementation_interfaces.py:150 in _make_new_table
    bool_features = np.array(features[:, self.bool_ids])
IndexError: index 16 is out of bounds for axis 1 with size 16
```

## Analysis

After `Fedot.fit()` completes with `safe_mode=True`:

| Attribute | Value |
|---|---|
| `model.train_data.features.shape` | `(N, 16)` |
| `scaling.bool_ids` | 71 entries, **max index = 72** |
| `scaling.ids_to_process` | `[3, 4]` (2 entries) |
| Total columns expected by pipeline | **73** |

The pipeline was internally trained on 73-feature data (produced by FEDOT's pipeline-level preprocessing during cross-validation fitting). But during `Fedot.predict()`:

- **`safe_mode=False`**: `data_processor.transform()` expands test features from 16 → 73 columns ✅
- **`safe_mode=True`**: `data_processor.transform()` leaves test features at 16 columns ❌

The mismatch between the pipeline's expected feature count (73) and the actual test feature count (16) causes the `IndexError`.

## Workaround

Remove `safe_mode=True` (or explicitly set `safe_mode=False`). This is FEDOT's default and does not trigger the bug.

```python
model = Fedot(problem="regression", timeout=5, metric="mae", safe_mode=False)
```

## Environment

- Python 3.10
- FEDOT 0.7.5
- OS: Linux 6.14.0
