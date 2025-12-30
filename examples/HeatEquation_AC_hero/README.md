# Heat Equation AC Hero Example

This example demonstrates a hybrid C++/Python implementation of a heat equation solver using AMReX.

## Compilation Instructions

### Prerequisites

Load the necessary modules for the Cray environment:

```bash
module load PrgEnv-gnu/8.5.0
module load cuda/12.3
module load craype-x86-milan
```

### Setting Python Environment

The build process requires access to Python headers and libraries. If you are not using an active Conda environment, you must explicitly set the `CONDA_PREFIX` environment variable to point to the location of the Python installation (e.g., the `AC_hero` environment).

```bash
export CONDA_PREFIX=/projects/hpcapps/nsawant/AdaptiveComputing/AC_hero
```

### Compiling for CPU

To compile the CPU-only version:

```bash
make clean
make AMREX_HOME=/projects/hpcapps/nsawant/marblesLBM/amrex USE_CUDA=FALSE
```

This will generate an executable named `main3d.gnu.x86-milan.ex`.

### Compiling for GPU

To compile the GPU-enabled version:

```bash
make clean
make AMREX_HOME=/projects/hpcapps/nsawant/marblesLBM/amrex USE_CUDA=TRUE
```

This will generate an executable named `main3d.gnu.CUDA.ex`.

## Running the Application

To run the application, you must ensure that the Python shared libraries are in your library path.

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Then run the executable with the input file:

```bash
./main3d.gnu.x86-milan.ex inputs
# or
./main3d.gnu.CUDA.ex inputs
```
