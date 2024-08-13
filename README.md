![AGPL License](https://img.shields.io/badge/license-AGPL%20v3-blue)

# Adaptive Computing

The Adaptive Computing (AC) software stack supports goal-based computing, for which a simulation workload is created on the fly, adapting to the results of calculations. Application-specific code defines an objective, which may be to solve an optimization problem or to train a surrogate model with minimal uncertainty. Then, the AC driver decides where in the design parameter space to run simulations to best achieve that objective. This process is iterative and online; as new data is returned from simulations, the AC driver chooses new simulations to run. The AC driver can strategically run simulations on distributed hardware resources (including high performance computing machines, cloud resources, and edge devices) to maximize throughput and obey resource constraints.

## Setup Instructions

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) should be installed on your system. HPC users can try

   ```bash
   module load conda

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/NREL/AdaptiveComputing.git
   cd AdaptiveComputing

2. **Create the conda environment**

   ```bash
   conda env create -f environment.yaml

3. **Activate the conda environment**

   ```bash
   conda activate AC

4. **Add AC to your conda python path**

   ```bash
   conda develop .

5. **Run the tests**

   ```bash
   python -m pytest
   
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

1. Most users of the Adaptive Computing repository have triage access. If you attempt to push your code, you will be guided to open a pull request. A bot will comment on the pull request when the tests complete and someone with write access will need to approve and merge your pull request.
