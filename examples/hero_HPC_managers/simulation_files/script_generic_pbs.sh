#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -A SAFCombust
# #PBS -q capacity
#PBS -q debug
#PBS -l filesystems=home
#PBS -N mock_simulation
# Output/error files use PBS defaults: mock_simulation.oJOBID and mock_simulation.eJOBID

# Variables passed via qsub -v: temp, task_id, machine_name, i_fidelity
i_fidelity=${i_fidelity:-0}

if [ -z "$temp" ]; then
    echo "Error: No temperature value provided."
    echo 'Usage: qsub -v "temp=<val>,task_id=<id>,machine_name=<name>[,i_fidelity=<n>]" script_generic_pbs.sh'
    exit 1
fi

if [ -z "$task_id" ]; then
    echo "Error: No task_id provided."
    exit 1
fi

if [ -z "$machine_name" ]; then
    echo "Error: No machine_name provided."
    exit 1
fi

cd "$PBS_O_WORKDIR"

# PBS job scripts don't source ~/.bashrc automatically. Initialize mamba
# explicitly using the module system so this works for any user.
module load mamba
eval "$(mamba shell hook --shell bash)"
mamba activate AC

echo "Running mock simulation with temp=$temp"
output=$(python mock_simulation.py "$temp")
echo "$output"

# Extract result from simulation output.
result=$(echo "$output" | awk -F= '/^conductivity=/{print $2}' | tail -1)
if [ -z "$result" ]; then
    result=-1
fi
echo "Result: conductivity=$result"

# Write result to a file for the manager to pick up and pass to hero_finalize.
# hero_initialize and hero_finalize are called by the manager on the login node,
# which has outbound internet access. Compute nodes may not.
echo "$result" > "result_${task_id}.txt"
