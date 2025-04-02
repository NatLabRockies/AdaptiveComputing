#!/bin/bash
#SBATCH --time=0:20:00 
#SBATCH --nodes=1
#SBATCH --account=acldrd
#SBATCH --job-name=lammps_test
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err

#temp=1.2
# Get the temperature value from the command-line argument
temp=$1

# Check if the temperature value is provided
if [ -z "$temp" ]; then
  echo "Error: No temperature value provided."
  echo "Usage: sbatch script.sh <temp> <task-id>"
  exit 1
fi

task_id=$2
# Check if the task-id is provided
if [ -z "$task_id" ]; then
  echo "Error: No task-id provided."
  echo "Usage: sbatch script.sh <temp> <task-id>"
  exit 1
fi

# Run hero_initialize_vermillion.py to indicate the job is running and unqueue it.
module load conda
source activate AC_hero
echo "Running command: python hero_initialize_vermillion.py $task_id"
python hero_initialize_vermillion.py $task_id

# Run LAMMPS
source /nopt/nrel/apps/210929a/myenv.2110041605
ml lammps/20230802
cd $SLURM_SUBMIT_DIR
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
if [[ -z "$result" ]]; then
    result=-1
fi
echo "Conductivity passed to python: $result"

# Run hero_finalize_vermillion.py to publish result and mark it as done
module load conda
source activate hero_py3.11
echo "Running command: python hero_finalize_vermillion.py $result $task_id"
python hero_finalize_vermillion.py $result $task_id

