#!/bin/bash
#SBATCH --time=0:05:00
#SBATCH --nodes=1
# TODO: update the two lines below for your HPC system before running.
# Run 'sinfo' to list partitions and 'sacctmgr show user $USER' for your account.
#SBATCH --partition=debug
#SBATCH --account=degrees
#SBATCH --job-name=mock_simulation
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err

# Generic SLURM script for the AdaptiveComputing HPC tutorial.
# Runs a mock Python simulation (conductivity = temperature^2 / 1000) that
# works on any HPC system with mamba/conda and the AC environment.
#
# To adapt for a real simulation, replace the "Run mock simulation" block
# with your own simulation commands (see script_kestrel.sh for a LAMMPS
# example), then update the grep pattern that extracts the result.
#
# Args passed by manager.py:
#   $1  temperature value
#   $2  Hero task ID
#   $3  machine name (as defined in hpc_config.py)
#   $4  fidelity level (optional, defaults to 0)

temp=$1
task_id=$2
machine_name=$3
i_fidelity=${4:-0}

if [ -z "$temp" ]; then
  echo "Error: No temperature value provided."
  echo "Usage: sbatch script_generic.sh <temp> <task-id> <machine_name> [i_fidelity]"
  exit 1
fi
if [ -z "$task_id" ]; then
  echo "Error: No task-id provided."
  echo "Usage: sbatch script_generic.sh <temp> <task-id> <machine_name> [i_fidelity]"
  exit 1
fi
if [ -z "$machine_name" ]; then
  echo "Error: No machine_name provided."
  echo "Usage: sbatch script_generic.sh <temp> <task-id> <machine_name> [i_fidelity]"
  exit 1
fi

# Load environment and claim the task in Hero before starting computation.
# Note: some systems use 'module load conda' instead of 'module load mamba'.
module load mamba
mamba activate AC
echo "Running command: python -m adaptive_computing.hero_utils.hero_initialize $task_id $machine_name $i_fidelity"
python -m adaptive_computing.hero_utils.hero_initialize $task_id $machine_name $i_fidelity
return_code=$?
if [ $return_code -eq 2 ]; then
    echo "hero_initialize: task already running on another machine. Terminating this job successfully."
    exit 0
elif [ $return_code -ne 0 ]; then
    echo "hero_initialize failed with code $return_code. Exiting."
    exit 1
fi

# Run mock simulation (replace with your real simulation commands)
cd $SLURM_SUBMIT_DIR
echo "Running mock simulation with temp=$temp for task-id=$task_id"
python mock_simulation.py $temp

# Extract the conductivity result from the SLURM stdout file.
# mock_simulation.py prints a line of the form: conductivity=<value>
JOBID=$SLURM_JOB_ID
FILE="out_${JOBID}.out"
result=$(grep "^conductivity=" "$FILE" | awk -F= '{print $2}' | tail -1)
if [[ -z "$result" ]]; then
    result=-1
fi
echo "Conductivity passed to hero_finalize: $result"

# Publish the result to Hero and mark the task as done.
echo "Running command: python -m adaptive_computing.hero_utils.hero_finalize $result $task_id $machine_name $i_fidelity"
python -m adaptive_computing.hero_utils.hero_finalize $result $task_id $machine_name $i_fidelity
if [ $? -ne 0 ]; then
    echo "hero_finalize failed. Exiting."
    exit 1
fi
