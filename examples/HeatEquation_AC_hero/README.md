# Heat Equation AC Hero Example

## Setup and Build Process

### 1. Clone and Checkout
```bash
git clone https://github.com/nileshsawant/AdaptiveComputing.git
cd AdaptiveComputing
git checkout AC_hero_marc
```

### 2. Allocate Resources
```bash
salloc -A hdcomb -t 01:00:00 --nodes=1 --ntasks-per-node=32 --mem=80G --gres=gpu:1 
```

### 3. Load Modules
```bash
module load PrgEnv-gnu/8.5.0
module load cuda/12.3
module load craype-x86-milan
module load anaconda3/2024.06.1
```

### 4. Setup Python Environment
```bash
conda env create --prefix ./AC_hero --file environment.yaml
conda activate ./AC_hero
pip install -e .
python -m pytest
```

### 5. Compile Example
```bash
cd examples/HeatEquation_AC_hero
make USE_CUDA=FALSE
make USE_CUDA=TRUE
```

### 6. Run Example
**Critical Step:** You must export the library path so the executable can find the Python shared libraries.

*Explanation:* While `conda activate` updates your `PATH` to find the python executable, it does not update `LD_LIBRARY_PATH`. The C++ executable needs to link against `libpython3.11.so` at runtime. Without explicitly adding the Conda environment's `lib` directory to `LD_LIBRARY_PATH`, the Linux dynamic linker will fail to find the required shared libraries.

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run CPU version
./main3d.gnu.x86-milan.ex inputs

# Run GPU version
./main3d.gnu.CUDA.ex inputs
```

### 7. Advanced Usage (Beta Features)

#### GPU-Accelerated Training
By default, the GPU executable (`main3d.gnu.CUDA.ex`) uses the GPU for *inference* (prediction) but performs model *training* on the CPU. This is sufficient for small datasets.

For larger datasets, you can enable **GPU-Accelerated Training** to offload the expensive hyperparameter optimization to the GPU.

**To enable:**
Add `use_gpu_kriging=1` to your command line arguments.

```bash
./main3d.gnu.CUDA.ex inputs use_gpu_kriging=1
```

**What this does:**
1.  **Enables GPU Training**: The Python surrogate will use `GPUKrigingTrain` instead of the standard `SMTGP`.
2.  **Accelerated Math**: The Likelihood function and its gradients (used during training) are computed on the GPU using CuPy.
3.  **Identical Results**: The final physics results are identical to the CPU-trained version, though you may see a slight difference in the number of retraining steps due to floating-point precision differences.

