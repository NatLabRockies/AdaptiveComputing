#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --nodes=1
#SBATCH --account=acldrd
#SBATCH --job-name=lammps_test
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err

temp=$1
task_id=$2

if [ -z "$temp" ]; then
  echo "Error: No temperature value provided."
  echo "Usage: sbatch script_vermilion.sh <temp> <task-id>"
  exit 1
fi

if [ -z "$task_id" ]; then
  echo "Error: No task-id provided."
  echo "Usage: sbatch script_vermilion.sh <temp> <task-id>"
  exit 1
fi

cd "$SLURM_SUBMIT_DIR"

# Run LAMMPS
module purge
source /nopt/nrel/apps/210929a/myenv.2110041605
ml lammps/20230802
echo "Running simulation with temp=$temp for task-id=$task_id"
t_lo=`echo "$temp - 0.3" | bc`
t_hi=`echo "$temp + 0.3" | bc`
echo "t=" $temp
srun lmp -in in.langevin -v t $temp -v tlo $t_lo -v thi $t_hi

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
