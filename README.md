# AdaptiveComputing
Goal-oriented computing including design optimization and automated surrogate model training

### Getting started
## Installing AC
# Download the source code
git clone https://github.nrel.gov/AC/AdaptiveComputing.git 

# Create a conda environment to ensure the correct python version and libraries are installed
# Install conda or if you are running on an HPC machine, run
module load conda
# Navigating to the AdaptiveComputing root directory.
conda env create -f environment.yaml -n <env_name>
# where <env_name> should be replaced a name of your choosing, (e.g., AC)
# The command prompt should indicate that you are running the conda environment. You can shorten/change the way this displays in the command prompt with the command
conda config --set env_prompt '(AC)'
# where (AC) will be preprended on the command prompt to indicate the AC environment has been loaded.

## Using AC
# Load the python environment and launch Jupyter notebooks
# Note that each time you start a new shell, you need to activate the conda environment
# A) If you are running on a laptop/desktop computer (only tested on M1 mac):
conda activate AC
jupyter notebook
# B) If you are running on an HPC machine (only tested on Eagle):
# if you are running more than just a tutorial, it would be appropriate to request a login node
srun -A acldrd -t 1:00:00 --pty /bin/bash
# Participants of the NREL LDRD project should use the "-A acldrd" option for billing, otherwise omit it.
module load purge
module load conda
source activate AC
# noting the distinction between using the “source” and “conda” keywords on the HPC. Finally, start the notebook without a display
jupyter notebook --no-browser --ip=0.0.0.0 --port=8080
# In a separate terminal, run 
ssh -L 8080:<node_name>:8080 eagle.hpc.nrel.gov
# where <node_name> is replaced by the name of the node where you started the notebook (see the command prompt of that window)
# In a browser window- connect to the second URL printed when the notebook was started. The URL should begin with "http://127.0.0.1:8080/?token=…"
# All calculations will now be done on HPC, and only the results are displayed locally

# From the Jupyter GUI, run the tutorials/example_1d/driver_1d.md

