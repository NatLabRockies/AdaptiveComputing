# GPU Compatibility Changes for HeatEquation_AC_hero

This document details the changes required to make the `HeatEquation_AC_hero` example compatible with GPU execution using AMReX's CUDA backend.

## 1. Code Changes in `main.cpp`

### A. Fix Variable Shadowing (Initialization Bug)
**Issue:** A local variable `py_thermal_properties` was declared in `main`, shadowing the global variable defined in `thermal_properties.h`. This caused the global pointer to remain `NULL`, leading to a crash when `update_gpu_model_from_python` attempted to use it.

**Change:** Removed the type declaration `PyObject *` to assign directly to the global variable.

```cpp
// Before
PyObject *pName = PyUnicode_DecodeFSDefault("py_thermal_properties");
PyObject *py_thermal_properties = PyImport_Import(pName); // <--- Shadows global variable

// After
PyObject *pName = PyUnicode_DecodeFSDefault("py_thermal_properties");
py_thermal_properties = PyImport_Import(pName); // <--- Assigns to global variable
```

### B. Host/Device Memory Management for Python Embedding
**Issue:** The function `pretrain_kappa_model` passes simulation data to Python/NumPy. When running on GPU, `phi_arr.dataPtr()` points to device memory. Since the standard Python interpreter runs on the CPU, accessing this pointer caused a segmentation fault.

**Change:** Created a temporary `FArrayBox` in host memory (CPU), copied the data from the GPU to this host buffer, and passed the host pointer to NumPy.

```cpp
// Before
const auto& phi_arr = phi.array(mfi);
auto bxg=amrex::grow(mfi.validbox(),1);

npy_intp dims[2] = {bxg.numPts(),1}; 
// ERROR: Passing GPU pointer to CPU Python
PyObject* x_queries = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, const_cast<amrex::Real*>(phi_arr.dataPtr()));
```

```cpp
// After
const auto& phi_arr = phi.array(mfi);
auto bxg=amrex::grow(mfi.validbox(),1);

// Create a host copy of the data for Python access
amrex::FArrayBox host_fab(bxg, 1, amrex::The_Cpu_Arena());
// Copy data from Device (GPU) to Host (CPU)
amrex::Gpu::copy(amrex::Gpu::deviceToHost, phi[mfi].dataPtr(), phi[mfi].dataPtr() + host_fab.size(), host_fab.dataPtr());
amrex::Gpu::streamSynchronize();

npy_intp dims[2] = {bxg.numPts(),1}; 
// Pass CPU pointer to Python
PyObject* x_queries = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, host_fab.dataPtr());
```

## 2. Configuration Changes

### `inputs`
*   Reduced `nsteps` from 5000 to 100 to facilitate faster testing and verification.

## 3. Runtime Environment

To run the executable, the `LD_LIBRARY_PATH` must be set to include the libraries from the custom Python environment, otherwise the dynamic linker will fail to find `libpython3.11.so`.

**Command:**
```bash
export LD_LIBRARY_PATH=/projects/hpcapps/nsawant/AdaptiveComputing/AC_hero/lib:$LD_LIBRARY_PATH
./main3d.gnu.CUDA.ex inputs
```

## 4. Compilation

To compile for GPU, use the `USE_CUDA=TRUE` flag:

```bash
make clean
make AMREX_HOME=/projects/hpcapps/nsawant/marblesLBM/amrex USE_CUDA=TRUE
```
