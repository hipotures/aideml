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

## Why the original OpenCL GPU build failed

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

## Successful OpenCL GPU build for `device_type = gpu`

LightGBM's `device_type = gpu` is the OpenCL backend. It is separate from the
native CUDA backend used by `device_type = cuda`.

This backend was later rebuilt successfully from current LightGBM master on the
same Arch-style system. It does not require installing an older CUDA toolkit,
but it does require working OpenCL packages:

```bash
pacman -Q opencl-nvidia ocl-icd opencl-headers boost boost-libs gcc14
```

The observed installed packages were:

```text
opencl-nvidia
ocl-icd
opencl-headers
boost
boost-libs
gcc14
```

The system OpenCL library must exist:

```bash
test -f /usr/lib/libOpenCL.so && echo ok
```

Clone or reuse LightGBM master and initialize submodules:

```bash
git clone https://github.com/lightgbm-org/LightGBM /tmp/lightgbm-master-aide-debug
git -C /tmp/lightgbm-master-aide-debug submodule update --init --recursive
```

On current Arch / Boost `1.91`, two local source patches were needed.

First, LightGBM's CMake still asks for the removed `boost_system` component
when `USE_GPU=ON`. In `CMakeLists.txt`, change:

```cmake
find_package(Boost 1.56.0 COMPONENTS filesystem system REQUIRED)
```

to:

```cmake
find_package(Boost 1.56.0 COMPONENTS filesystem REQUIRED)
```

Second, vendored Boost.Compute expects the older Boost SHA1 digest type. In
`external_libs/compute/include/boost/compute/detail/sha1.hpp`, replace the
digest block in `operator std::string()` with a Boost-version-compatible
branch:

```cpp
#if BOOST_VERSION >= 109100
            unsigned char digest[20];
            h.get_digest(digest);

            std::ostringstream buf;
            for(int i = 0; i < 20; ++i)
                buf << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned int>(digest[i]);
#else
            unsigned int digest[5];
            h.get_digest(digest);

            std::ostringstream buf;
            for(int i = 0; i < 5; ++i)
                buf << std::hex << std::setfill('0') << std::setw(8) << digest[i];
#endif
```

Build the wheel with the OpenCL backend:

```bash
rm -rf /tmp/lgbm-opencl-build
uv venv /tmp/lgbm-opencl-build --python 3.12
uv pip install --python /tmp/lgbm-opencl-build/bin/python pip

cd /tmp/lightgbm-master-aide-debug
PATH=/tmp/lgbm-opencl-build/bin:$PATH \
CC=/usr/bin/gcc-14 \
CXX=/usr/bin/g++-14 \
  /bin/sh ./build-python.sh bdist_wheel --gpu
```

The resulting wheel is:

```text
/tmp/lightgbm-master-aide-debug/dist/lightgbm-4.6.0.99-py3-none-linux_x86_64.whl
```

Install it into AIDE's normal virtualenv without dependency changes:

```bash
cd /home/xai/DEV/aideml
uv pip install --reinstall --no-deps \
  /tmp/lightgbm-master-aide-debug/dist/lightgbm-4.6.0.99-py3-none-linux_x86_64.whl
```

Use `--no-deps`. Without it, `uv` may upgrade packages such as `numpy` and
`scipy`, which can break ABI compatibility with the already-installed pandas /
scikit-learn stack.

Verify versions:

```bash
uv run python - <<'PY'
import numpy, scipy, pandas, sklearn, lightgbm
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("lightgbm", lightgbm.__version__, lightgbm.__file__)
PY
```

Verify OpenCL GPU training:

```bash
uv run python - <<'PY'
import numpy as np
import lightgbm as lgb

rng = np.random.default_rng(42)
X = rng.normal(size=(3000, 24)).astype(np.float32)
y = (X[:, 0] + 0.25 * X[:, 1] + rng.normal(size=3000) * 0.2 > 0).astype(int)
train = lgb.Dataset(X, label=y)

params = {
    "objective": "binary",
    "metric": "auc",
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "verbose": 1,
    "seed": 42,
}

lgb.train(params, train, num_boost_round=3)
print("uv_opencl_gpu_train_ok")
PY
```

Expected output includes:

```text
This is the GPU trainer!!
Using GPU Device: NVIDIA GeForce RTX 4070 Ti, Vendor: NVIDIA Corporation
uv_opencl_gpu_train_ok
```

In AIDE experiments, use:

```bash
AIDE_LGBM_DEVICE=gpu ./run.sh
```

The experiment code maps this to:

```python
{
    "device_type": "gpu",
    "gpu_platform_id": 0,
    "gpu_device_id": 0,
}
```

Keep `device_type = cuda` disabled for this F1 dataset unless revalidated,
because the native CUDA backend has repeatedly aborted with illegal memory
access in `cuda_best_split_finder.cu`.

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

However, for this F1 pit-stop task the CUDA backend is not a good default for
LightGBM inside AutoGluon:

- the full dataset with native pandas categorical columns can crash with:

  ```text
  [LightGBM] [Fatal] [CUDA] an illegal memory access was encountered
  ```

- this matches the upstream LightGBM CUDA bug report:
  <https://github.com/lightgbm-org/LightGBM/issues/6512>
- lowering `max_bin` enough to avoid the crash, for example to `15`, made the
  standalone AutoGluon `GBM` smoke test train successfully, but with much worse
  validation quality;
- CPU LightGBM remains strong for this dataset, with observed AutoGluon
  validation around `0.9501` for a CPU `LightGBM_r131_BAG_L2` run.

So the practical AIDE profile should keep LightGBM on CPU while using GPU for
CatBoost and XGBoost.

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
      - ag_args_fit:
          num_gpus: 0
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

The important part for LightGBM in the production profile is:

```yaml
GBM:
  - ag_args_fit:
      num_gpus: 0
```

If you explicitly want to test the CUDA backend anyway, use:

```yaml
GBM:
  - device: cuda
    max_bin: 15
    ag_args_fit:
      num_gpus: 1
```

Do not use that as the default quality profile unless it is revalidated on the
target dataset, because it traded stability for a large score drop in the local
smoke test.

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
