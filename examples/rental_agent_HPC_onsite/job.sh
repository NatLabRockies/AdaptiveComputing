#!/bin/bash
# TODO: update the three lines below for your HPC system.
#SBATCH --job-name=rental_agent_onsite
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# TODO: add your account and partition:
# #SBATCH --account=<your_account>
# #SBATCH --partition=<your_partition>

# =============================================================================
# Arguments passed by manager.py:
#   $1 = task_id       (Hero task UUID)
# =============================================================================
task_id=$1

if [ -z "$task_id" ]; then
    echo "Error: Missing argument 'task_id'"
    echo "Usage: sbatch job.sh <task_id>"
    exit 1
fi

echo "Job started at: $(date)"
echo "SLURM job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Task ID: $task_id"

# --- Resolve paths ---
WORK_DIR="$SLURM_SUBMIT_DIR"
CASE_DIR="$WORK_DIR/cases/$task_id"
SIM_DIR="$WORK_DIR/simulation_files"

echo "Work dir:  $WORK_DIR"
echo "Case dir:  $CASE_DIR"

mkdir -p "$CASE_DIR/logs"

if [ ! -f "$CASE_DIR/config.json" ]; then
    echo "Error: config.json not found at $CASE_DIR/config.json"
    exit 1
fi

echo "Config:"
cat "$CASE_DIR/config.json"

# =============================================================================
# Load environment — TODO: adapt to your HPC module system / conda setup
# =============================================================================
# module load mamba
# source activate AC

# =============================================================================
# Simulation
# =============================================================================
echo "--- Simulation starting at: $(date) ---"
python "$SIM_DIR/mock_simulation.py" "$CASE_DIR" \
    > "$CASE_DIR/logs/simulation.out" 2> "$CASE_DIR/logs/simulation.err"
sim_exit=$?
echo "--- Simulation completed at: $(date) (exit code: $sim_exit) ---"

cat "$CASE_DIR/logs/simulation.out"

if [ $sim_exit -ne 0 ]; then
    echo "Simulation failed. See $CASE_DIR/logs/simulation.err"
    cat "$CASE_DIR/logs/simulation.err"
    exit 1
fi

# --- Extract negated cost from result.json (AC maximizes, so we negate) ---
if [ ! -f "$CASE_DIR/result.json" ]; then
    echo "Error: result.json not written by mock_simulation.py"
    exit 1
fi

cost=$(python3 -c "import json; print(-json.load(open('$CASE_DIR/result.json'))['cost'])")

if [ -z "$cost" ]; then
    echo "Warning: could not extract cost from result.json, defaulting to -1."
    cost=-1
fi
echo "Negated cost (stored as -cost for AC maximization convention): $cost"

# Write result for the manager to pick up via read_result().
echo "$cost" > "$WORK_DIR/result_${task_id}.txt"
echo "Wrote result to result_${task_id}.txt"

echo "Job completed at: $(date)"
