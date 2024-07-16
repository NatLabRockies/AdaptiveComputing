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
   git checkout ac_2.0

2. **Create the conda environment**

   ```bash
   conda env create -f environment.yaml

3. **Activate the conda environment**

   ```bash
   conda activate AC

4. **Install additional dependencies**

   ```bash
   pip install -r requirements.txt

5. **Run the tests**

   ```bash
   pytest
   
2. **Create the conda environment**

   ```bash
