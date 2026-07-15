#!/bin/bash
#SBATCH --time=0:05:00
#SBATCH --nodes=1
#SBATCH --partition=debug
#SBATCH --account=degrees
#SBATCH --job-name=lammps_test
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err

temp=$1
task_id=$2
machine_name=$3
i_fidelity=${4:-0}

if [ -z "$temp" ]; then
  echo "Error: No temperature value provided."
  echo "Usage: sbatch script_kestrel.sh <temp> <task-id> <machine_name> [i_fidelity]"
  exit 1
fi

if [ -z "$task_id" ]; then
  echo "Error: No task-id provided."
  echo "Usage: sbatch script_kestrel.sh <temp> <task-id> <machine_name> [i_fidelity]"
  exit 1
fi

if [ -z "$machine_name" ]; then
  echo "Error: No machine_name provided."
  echo "Usage: sbatch script_kestrel.sh <temp> <task-id> <machine_name> [i_fidelity]"
  exit 1
fi

cd "$SLURM_SUBMIT_DIR"

# Run LAMMPS
module purge
module load lammps/080223-intelmpi
echo "Running simulation with temp=$temp for task-id=$task_id"
t_lo=`echo "$temp - 0.3" | bc`
t_hi=`echo "$temp + 0.3" | bc`
echo "t=" $temp
srun -n 8 lmp -in in.langevin -v t $temp -v tlo $t_lo -v thi $t_hi

# Extract the conductivity output
JOBID=$SLURM_JOB_ID
FILE="out_${JOBID}.out"
# Search for the line containing "Running average thermal conductivity" and extract the last field
result=$(grep "Running average thermal conductivity" "$FILE" | awk '{print $NF}')
# If grep fails (no matching line found), set result to -1
if [ -z "$result" ]; then
    result=-1
fi
echo "Conductivity passed to python: $result"

# Write result to a file for the manager to pick up and pass to hero_finalize.
# hero_initialize and hero_finalize are called by the manager on the login node,
# which has outbound internet access.
echo "$result" > "result_${task_id}.txt"
