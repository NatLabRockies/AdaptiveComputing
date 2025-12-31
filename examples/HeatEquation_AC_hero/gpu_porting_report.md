# GPU Implementation Report: Heat Equation AC Hero

## Summary

This report documents the strategy and implementation details for accelerating the Adaptive Computing (AC) surrogate model on GPUs. The primary goal was to enable the C++ AMReX solver to perform inference (prediction) directly on the GPU, avoiding the significant latency of transferring data back and forth to the Python interpreter for every cell in the simulation.

## 1. Strategy: Inference Acceleration

The decision to "code for GPU" followed the **Inference Acceleration** pattern. We did not attempt to run the entire Python `adaptive_computing` library on the GPU. Instead, we identified the specific mathematical operations required to make a prediction and ported *only* those operations to C++/CUDA.

### The Problem
*   The simulation runs on the GPU (AMReX/C++).
*   The surrogate model (Kriging) is trained and managed in Python.
*   Calling Python from a GPU thread is technically impossible.
*   Copying the entire simulation state to the CPU to query Python is prohibitively slow.

### The Solution
We implemented a "Math Port" rather than a "Library Port":
1.  **Identify the Math:** We determined that the surrogate uses a **Kriging (Gaussian Process)** model.
2.  **Port the Formula:** We wrote a lightweight, GPU-compatible C++ function (`predict`) that implements the Kriging prediction formula.
3.  **Bridge the Data:** We created a mechanism to copy the trained model parameters (weights, training points) from Python to the GPU.

## 2. Implementation Details

The implementation consists of three key components defined in `thermal_properties.h`.

### A. The GPU Kernel (`predict`)
This is the "math port." It is a C++ implementation of the Kriging formula. It runs on the GPU and calculates the thermal conductivity for a given temperature.

```cpp
AMREX_GPU_HOST_DEVICE
amrex::Real predict(amrex::Real x) const {
    // Normalize input
    amrex::Real x_norm = (x - X_offset) / X_scale;
    amrex::Real y_norm = beta;
    
    // Compute correlation with all training points
    for (int i = 0; i < n_samples; ++i) {
        amrex::Real dist_sq = (x_norm - X_norma[i]) * (x_norm - X_norma[i]);
        amrex::Real corr = std::exp(-theta[0] * dist_sq);
        y_norm += alpha[i] * corr;
    }
    // Denormalize output
    return y_norm * y_std + y_mean;
}
```

### B. The Data Bridge (`update_gpu_model_from_python`)
This function acts as the coordinator. It is called from `main.cpp` whenever the surrogate is retrained in Python.
1.  It calls a Python helper to extract the raw NumPy arrays (weights, hyperparameters).
2.  It allocates a C++ struct (`KrigingModel`) to hold this data.
3.  It triggers the memory copy to the GPU.

### C. The Plumbing (`copy_numpy_to_device`)
This helper function handles the low-level memory operations required to move data from a Python NumPy array to an AMReX GPU buffer.
1.  **Extract:** Accesses the raw `double*` data pointer from the NumPy object.
2.  **Allocate:** Uses `amrex::The_Arena()->alloc()` to reserve GPU memory.
3.  **Copy:** Uses `amrex::Gpu::copy` to transfer the data.

### D. The Training Data Extractor (`pretrain_kappa_model`)
Located in `main.cpp`, this function handles the *reverse* data flow (GPU $\to$ CPU) required for active learning.
*   **Challenge:** The Python surrogate needs to see the current simulation state (temperature field) to determine if its predictions are still accurate (variance check). The data lives on the GPU.
*   **Solution:**
    1.  Creates a temporary host (CPU) buffer (`host_fab`).
    2.  Copies the simulation data from the GPU to this buffer: `amrex::Gpu::copy(amrex::Gpu::deviceToHost, ...)`
    3.  Wraps this CPU buffer in a NumPy array so Python can read it without further copying.

```cpp
// main.cpp: pretrain_kappa_model
// Create a host copy of the data for Python access
amrex::FArrayBox host_fab(bxg, 1, amrex::The_Cpu_Arena());
amrex::Gpu::copy(amrex::Gpu::deviceToHost, phi[mfi].dataPtr(), phi[mfi].dataPtr() + host_fab.size(), host_fab.dataPtr());
amrex::Gpu::streamSynchronize();
```

### E. The Execution Switch (`get_thermal_conductivity`)
Located in `thermal_properties.h`, this is the main entry point called by the physics solver for every cell.
*   **Role:** It acts as a dispatcher.
*   **Logic:**
    1.  Checks if the GPU model (`d_kriging_model`) is initialized.
    2.  **Fast Path:** If yes, calls `d_kriging_model->predict(temperature)` (The GPU Kernel).
    3.  **Slow Path (CPU only):** If not (or if compiled without CUDA), it falls back to creating Python objects and calling the Python `query` method. This fallback is extremely slow and intended only for debugging or initialization.

### F. The Main Loop (No Data Movement)
In the main time evolution loop (`main.cpp`), the physics update happens entirely on the device. The `get_thermal_conductivity` function is called inside an `AMREX_GPU_DEVICE` lambda, ensuring it runs on the GPU threads.

```cpp
// main.cpp: Main Loop
amrex::ParallelFor(bxg, [=] AMREX_GPU_DEVICE (int i, int j, int k)
{
    // This runs on the GPU. No data leaves the device.
    kappa_arr(i,j,k) = get_thermal_conductivity(phi_arr(i,j,k));
});
```

## 3. Python-Side Acceleration (CuPy)

In addition to the C++ port for the physics loop, we also accelerated the Python-side variance prediction (`predict_variances`) used during the active learning phase.

### Motivation
The active learning loop requires calculating the prediction variance for every grid point to identify areas of high uncertainty. While the physics loop runs on the GPU, this variance check happens in Python. For large grids, calculating the variance on the CPU using NumPy/SciPy becomes a bottleneck.

### Implementation (`gpu_kriging.py`)
We implemented a GPU-accelerated version of the Kriging variance prediction using **CuPy**, a NumPy-compatible library for GPU computing.

1.  **`GPUKriging` Class:** A subclass of SMT's `KRG` model.
2.  **`predict_variances` Override:** Replaces the CPU-based linear algebra with CuPy equivalents:
    *   `scipy.linalg.solve_triangular` $\rightarrow$ `cupyx.scipy.linalg.solve_triangular`
    *   `numpy.dot` $\rightarrow$ `cupy.dot`
    *   `numpy.exp` $\rightarrow$ `cupy.exp`
3.  **Data Flow:**
    *   Input points are transferred to GPU memory (`cp.asarray`).
    *   Model parameters (weights, covariance matrix factors) are transferred once.
    *   Computations happen entirely on the GPU.
    *   Results are transferred back to CPU (`cp.asnumpy`) only when needed by the driver.

### Integration (`gpu_surrogate_wrapper.py`)
To seamlessly integrate this into the existing workflow without modifying the core library:
1.  **Wrapper Class:** Created `GPUSMTGP` which inherits from `SMTGP`.
2.  **Automatic Detection:** The script `py_thermal_properties.py` automatically detects if a GPU is available (via `cupy.cuda.runtime.getDeviceCount()`).
    *   **If GPU available:** Instantiates `GPUSMTGP` (CuPy).
    *   **If CPU only:** Instantiates standard `SMTGP` (NumPy).
3.  **Environment Control:** The C++ application sets an environment variable `AC_ENABLE_GPU_SURROGATE` based on its compilation flags (`AMREX_USE_GPU`). This ensures the Python script respects the application's build configuration.

## 4. Reusability & Future Work

### Modificatiobs for other AC-provided options

*   **Scenario A: Changing Hyperparameters or Data (Same Algorithm)**
    *   **No code changes needed.**
    *   If you change the Kriging kernel parameters, add more training points, or change the scaling, the existing code will automatically copy the new values to the GPU. The `predict` formula remains valid.

*   **Scenario B: Changing the Algorithm (e.g., Neural Network, Random Forest)**
    *   **Yes, new code is needed.**
    *   The `predict` function above is specific to the Kriging formula.
    *   If you switch to a Neural Network, you would need to write a C++ `predict` function that performs matrix multiplications (forward pass).
    *   If you switch to a Random Forest, you would need a C++ `predict` function that traverses decision trees.
    *   The "Plumbing" (`copy_numpy_to_device`) would likely remain reusable, but the "Data Bridge" would need to be updated to extract the new model structure (e.g., layers/weights instead of alpha/theta).

## Conclusion

The current implementation provides a high-performance, GPU-native inference path for Kriging models. It bridges the flexibility of Python training with the speed of C++ GPU execution. Future extensions to other surrogate types will require implementing their specific mathematical prediction formulas in C++.
