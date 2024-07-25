# Adaptive Computing

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
   pytest
   
## Developer instructions

### Testing

1. Before pushing or opening a pull request, make sure your code passes the test suite:

   ```bash
   # cd to the AdaptiveComputing home directory
   pytest
   # stop the build if there are Python syntax errors or undefined names
   ruff check --select=E9,F63,F7,F82 --target-version=py312 .
   # check for style and potential bugs
   ruff check --target-version=py312 .


### Pull requests

1. Most users of the Adaptive Computing repository have triage access. If you attempt to push your code, you will be guided to open a pull request. A bot will comment on the pull request when the tests complete and someone with write access will need to approve and merge your pull request.
