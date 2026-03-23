![AGPL License](https://img.shields.io/badge/license-AGPL%20v3-blue)

# Adaptive Computing

The Adaptive Computing (AC) software stack supports goal-based computing, for which a simulation workload is created on the fly, adapting to the results of calculations. Application-specific code defines an objective, which may be to solve an optimization problem or to train a surrogate model with minimal uncertainty. Then, the AC driver decides where in the design parameter space to run simulations to best achieve that objective. This process is iterative and online; as new data is returned from simulations, the AC driver chooses new simulations to run. The AC driver can strategically run simulations on distributed hardware resources (including high performance computing machines, cloud resources, and edge devices) to maximize throughput and obey resource constraints.

## Setup Instructions

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) should be installed on your system. HPC users can try

   ```bash
   module load conda

### Steps

**First, clone the repository:**

```bash
git clone https://github.com/NatLabRockies/AdaptiveComputing.git
cd AdaptiveComputing
```

**Then choose one of the following setup options:**

#### Option 1: Full Setup with Hero Support (Recommended)

Use this option if you plan to use Hero framework features for distributed computing.

1. **Create the conda environment with Hero dependencies**

   ```bash
   mamba env create -f environment.yaml

2. **Activate the conda environment**

   ```bash
   mamba activate AC_hero

3. **Set up Hero credentials (if using Hero features)**

   The Hero environment file will be auto-created from the template when you first use Hero functionality. To configure your credentials:

   ```bash
   # The file will be created automatically, then edit it:
   nano adaptive_computing/hero_utils/set_hero_env_vars.py
   ```

   Replace the placeholder values with your actual Hero credentials (ask your Hero admin for these values).

4. **Add AC to your conda python path**

   ```bash
   pip install -e .

5. **Run the tests**

   ```bash
   python -m pytest

#### Option 2: Basic Setup without Hero

Use this option if you only need core AC functionality without Hero framework features.

1. **Create the conda environment without Hero dependencies**

   ```bash
   mamba env create -f environment_no_hero.yaml

2. **Activate the conda environment**

   ```bash
   mamba activate AC

3. **Hero environment setup (automatic)**

   No manual setup needed! The Hero environment file will be auto-created from the template when first accessed. Since you're not using Hero features, you can ignore any credential-related messages.

4. **Add AC to your conda python path**

   ```bash
   pip install -e .

5. **Run the tests**

   ```bash
   python -m pytest
   
## Examples

The `examples/` directory contains practical demonstrations of AC functionality:

### Getting Started - Hero Introduction

**Start here**: [examples/hero/](examples/hero/) provides a simple introduction to Hero framework concepts:
- **controller.py**: Demonstrates HeroDataset API with different sample addition methods
- **controller_noAC.py**: Shows direct Hero API usage with detailed queue monitoring
- **worker.py**: Simple local worker that processes tasks with basic calculations
- No HPC access required - runs entirely on your local machine
- Perfect for learning Hero concepts and testing your setup

### Advanced - HPC Production Workflows

**Scale up**: [examples/hero_HPC_managers/](examples/hero_HPC_managers/) demonstrates production HPC workflows:
- **Adaptive Computing strategies**: Offline training, offline inference, and online inference
- Automated HPC managers with SSH and SLURM integration
- Real molecular dynamics simulations for conductivity calculation
- Multi-cluster job distribution and error handling
- Requires HPC access and SSH configuration

### Other Examples

- [examples/bayesian_1d_sf/](examples/bayesian_1d_sf/): Single-fidelity Bayesian optimization
- [examples/bayesian_2d_sf/](examples/bayesian_2d_sf/): Multi-dimensional optimization
- [examples/cpp_embedding/](examples/cpp_embedding/): C++ integration
- [examples/query_sf/](examples/query_sf/): Simple query examples

**Recommended learning path**: Start with `examples/hero/` to understand the basics, then explore `examples/hero_HPC_managers/` for production deployment.

## Developer instructions

### Testing

1. Before pushing or opening a pull request, make sure your code passes the test suite:

   ```bash
   # cd to the AdaptiveComputing home directory
   python -m pytest
   # stop the build if there are Python syntax errors or undefined names
   ruff check --select=E9,F63,F7,F82 --target-version=py312 .
   # check for style and potential bugs
   ruff check --target-version=py312 .


### Pull requests

1. Most users of the Adaptive Computing repository have read access. If you attempt to push your code, you will be guided to open a pull request. A bot will comment on the pull request when the tests complete and someone with write access will need to approve and merge your pull request.
