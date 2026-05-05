# LightGBM CUDA build for AIDE AutoGluon profiles

This document describes how LightGBM was rebuilt locally for NVIDIA GPU
training in the AIDE virtual environment, and what has to be changed in
LightGBM source/configuration before compilation on this machine.

## Target environment

- Repository: `/home/xai/DEV/aideml`
- Python environment: `/home/xai/DEV/aideml/.venv`
- OS family: Manjaro / Arch
- GPU: NVIDIA GeForce RTX 4090
- NVIDIA driver observed during setup: `590.48.01`
- CUDA toolkit observed during setup: `13.1`
- LightGBM version: `4.6.0`
- AutoGluon version: `1.5.0`

## Why the OpenCL GPU build failed

LightGBM has two different GPU backends:

- `USE_GPU=ON`, used through `device` / `device_type = gpu`
  - OpenCL backend.
  - This is the backend described by the older LightGBM GPU tutorial.
- `USE_CUDA=ON`, used through `device` / `device_type = cuda`
  - CUDA backend.
  - This is the better fit for NVIDIA-only Linux hosts such as this RTX 4090
    machine.

The OpenCL build was attempted with:

```bash
CMAKE_ARGS="-DUSE_GPU=ON -DOpenCL_LIBRARY=/usr/lib/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/include" \
  uv pip install --no-binary lightgbm lightgbm==4.6.0
```

CMake found OpenCL correctly, but failed on Boost:

```text
Could NOT find Boost (missing: system) (found suitable version "1.89.0")
```

Root cause:

- Arch / Manjaro currently provides Boost `1.89.0`.
- LightGBM 4.6.0 OpenCL GPU build still asks CMake for a `boost_system`
  component.
- On this system Boost 1.89 does not provide a separate
  `boost_systemConfig.cmake` / `libboost_system.so` in the shape expected by
  LightGBM's CMake build.

So the OpenCL path is blocked by a LightGBM/CMake/Boost compatibility issue,
not by the NVIDIA driver or OpenCL headers.

## Why the first CUDA build failed

The CUDA build was attempted with:

```bash
CMAKE_ARGS="-DUSE_CUDA=ON" \
  uv pip install --no-binary lightgbm lightgbm==4.6.0
```

This got past Boost entirely, but failed in `nvcc`:

```text
nvcc fatal: Unsupported gpu architecture 'compute_60'
```

Passing the generic CMake variable was not enough:

```bash
CMAKE_ARGS="-DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89" \
  uv pip install --no-binary lightgbm lightgbm==4.6.0
```

Reason: LightGBM 4.6.0 sets its own `CUDA_ARCHS` list in `CMakeLists.txt`,
including old architectures such as `60`, `61`, and `62`. CUDA 13.1 no longer
supports compiling for `compute_60`, so the build fails before it reaches the
RTX 4090 architecture.

## Required source patch

Use a local copy of the LightGBM 4.6.0 source and patch `CMakeLists.txt`.

The relevant original block starts with:

```cmake
set(CUDA_ARCHS "60" "61" "62" "70" "75")
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
    list(APPEND CUDA_ARCHS "80")
endif()
...
```

For RTX 4090, replace the full architecture construction block with:

```cmake
set(CUDA_ARCHS "89")
```

This targets Ada Lovelace / RTX 4090 directly and avoids unsupported older
architectures.

## Exact rebuild procedure

From the repo root:

```bash
cd /home/xai/DEV/aideml
```

If a previous LightGBM install exists, remove it:

```bash
uv pip uninstall lightgbm
```

Copy the cached source into a patchable temp directory:

```bash
rm -rf /tmp/lightgbm-4.6.0-cuda89
cp -a /home/xai/.cache/uv/sdists-v9/pypi/lightgbm/4.6.0/anEeVlSCMF8pvY1V5BPNW/src \
  /tmp/lightgbm-4.6.0-cuda89
```

Edit:

```text
/tmp/lightgbm-4.6.0-cuda89/CMakeLists.txt
```

Patch the CUDA architecture block to:

```cmake
set(CUDA_ARCHS "89")
```

Install from the patched source:

```bash
CMAKE_ARGS="-DUSE_CUDA=ON" \
  uv pip install --no-binary lightgbm /tmp/lightgbm-4.6.0-cuda89
```

Expected success message:

```text
Built lightgbm @ file:///tmp/lightgbm-4.6.0-cuda89
Installed 1 package
 + lightgbm==4.6.0
```

## Verification

Verify import:

```bash
uv run python -c "import lightgbm as lgb; print(lgb.__version__); print(lgb.__file__)"
```

Verify direct LightGBM CUDA training:

```bash
uv run python -c "import lightgbm as lgb, numpy as np; X=np.random.RandomState(1).rand(2000, 20); y=(X[:,0]+X[:,1]*0.5+np.random.RandomState(2).rand(2000)*0.1>0.8).astype(int); ds=lgb.Dataset(X,label=y); booster=lgb.train({'objective':'binary','metric':'auc','device_type':'cuda','num_leaves':31,'verbose':1}, ds, num_boost_round=5); print('trained', booster.current_iteration())"
```

Observed valid output included:

```text
trained 5
```

LightGBM may warn that `auc` is evaluated on CPU:

```text
Metric auc is not implemented in cuda version. Fall back to evaluation on CPU.
```

That warning is acceptable; training still uses the CUDA backend.

## AutoGluon configuration requirement

AutoGluon 1.5.0 has this LightGBM behavior:

- if `num_gpus != 0`
- and the LightGBM hyperparameters do not already contain `device`
- AutoGluon sets:

```python
params["device"] = "gpu"
```

That means AutoGluon defaults to the OpenCL LightGBM backend for GPU training.
For the CUDA build described above, the AIDE profile must explicitly set:

```yaml
device: cuda
```

The `full_boost_gpu` profile in `aide/utils/config.yaml` therefore uses:

```yaml
full_boost_gpu:
  included_model_types: [XGB, GBM, CAT]
  presets: medium_quality
  time_limit: 600
  use_gpu: true
  validation_strategy: holdout
  hyperparameters:
    GBM:
      - device: cuda
        ag_args_fit:
          num_gpus: 1
    CAT:
      - task_type: GPU
        devices: "0"
        ag_args_fit:
          num_gpus: 1
    XGB:
      - device: cuda
        tree_method: hist
        n_jobs: 8
        ag_args_fit:
          num_gpus: 1
  fit_args:
    save_space: true
    fit_weighted_ensemble: false
    auto_stack: false
```

The important part for LightGBM is:

```yaml
GBM:
  - device: cuda
    ag_args_fit:
      num_gpus: 1
```

Without `device: cuda`, AutoGluon would try `device: gpu`, which expects an
OpenCL-enabled LightGBM build.

## AutoGluon smoke test

A minimal AutoGluon smoke test with only `GBM` confirmed that GPU resources are
passed correctly:

```text
Fitting model: LightGBM ...
    Fitting with cpus=16, gpus=1
```

The test used:

```python
hyperparameters={
    "GBM": [
        {
            "device": "cuda",
            "ag_args_fit": {"num_gpus": 1},
        }
    ]
}
```

## Notes for future rebuilds

- The patched source lives in `/tmp/lightgbm-4.6.0-cuda89`, so it is not a
  durable source of truth.
- If `/tmp` is cleaned, recreate the directory from the cached LightGBM source
  and reapply the `CUDA_ARCHS "89"` patch.
- If LightGBM changes its CMake logic in a future version, re-check whether this
  patch is still needed.
- If CUDA starts failing with a different unsupported architecture, inspect
  `CMakeLists.txt` for the current `CUDA_ARCHS` block before changing AIDE
  config.
- `clinfo` is only useful for diagnosing the OpenCL `device=gpu` backend. It is
  not required for the CUDA `device=cuda` backend used here.
