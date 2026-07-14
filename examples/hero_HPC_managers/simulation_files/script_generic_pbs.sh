#!/bin/bash
#PBS -l select=1
#PBS -l walltime=00:05:00
#PBS -A SAFCombust
#PBS -q capacity
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

echo "Running command: python -m adaptive_computing.hero_utils.hero_initialize $task_id $machine_name $i_fidelity"
python -m adaptive_computing.hero_utils.hero_initialize \
    "$task_id" "$machine_name" "$i_fidelity"

return_code=$?

if [ $return_code -eq 2 ]; then
    echo "hero_initialize: task already running on another machine."
    exit 0
elif [ $return_code -ne 0 ]; then
    echo "hero_initialize failed."
    exit 1
fi

echo "Running mock simulation with temp=$temp"

output=$(python mock_simulation.py "$temp")

echo "$output"

result=$(echo "$output" | awk -F= '/^conductivity=/{print $2}' | tail -1)

if [ -z "$result" ]; then
    result=-1
fi

echo "Conductivity passed to hero_finalize: $result"

python -m adaptive_computing.hero_utils.hero_finalize \
    "$result" "$task_id" "$machine_name" "$i_fidelity"

if [ $? -ne 0 ]; then
    echo "hero_finalize failed."
    exit 1
fi
