#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:05:00
# TODO: update the two lines below for your HPC system before running.
# Run 'qstat -Q' to list queues and check your allocation with your sysadmin.
#PBS -A SAFCombust
#PBS -q debug
#PBS -l filesystems=home
#PBS -N mock_simulation
# Output/error files use PBS defaults: mock_simulation.oJOBID and mock_simulation.eJOBID

# Generic PBS script for the AdaptiveComputing HPC tutorial.
# Runs a mock Python simulation (conductivity = temperature^2 / 1000) that
# works on any HPC system with mamba/conda and the AC environment.
#
# To adapt for a real simulation, replace the "Run mock simulation" block
# with your own simulation commands, then update the awk pattern that
# extracts the result.
#
# Args passed by manager.py via qsub -v:
#   temp         temperature value
#   task_id      Hero task ID

if [ -z "$temp" ]; then
    echo "Error: No temperature value provided."
    echo 'Usage: qsub -v "temp=<val>,task_id=<id>" script_generic_pbs.sh'
    exit 1
fi

if [ -z "$task_id" ]; then
    echo "Error: No task_id provided."
    exit 1
fi

cd "$PBS_O_WORKDIR"

# PBS job scripts don't source ~/.bashrc automatically. Initialize mamba
# explicitly using the module system so this works for any user.
# Note: some systems use 'module load conda' instead of 'module load mamba'.
module load mamba
eval "$(mamba shell hook --shell bash)"
mamba activate AC

# Run mock simulation (replace with your real simulation commands)
echo "Running mock simulation with temp=$temp for task-id=$task_id"
output=$(python mock_simulation.py "$temp")
echo "$output"

# Extract result from simulation output.
# mock_simulation.py prints a line of the form: conductivity=<value>
result=$(echo "$output" | awk -F= '/^conductivity=/{print $2}' | tail -1)
if [ -z "$result" ]; then
    result=-1
fi
echo "Result: conductivity=$result"

# Write result to a file for the manager to pick up and pass to hero_finalize.
# hero_initialize and hero_finalize are called by the manager on the login node,
# which has outbound internet access. Compute nodes may not.
echo "$result" > "result_${task_id}.txt"
