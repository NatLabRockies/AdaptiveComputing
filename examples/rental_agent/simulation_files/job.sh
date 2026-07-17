#!/bin/bash
#SBATCH --job-name=rental_agent
#SBATCH --account=newbridge
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# Output/error files are set by manager.py via --output/--error on the sbatch command line.

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
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Task ID: $task_id  Machine: $machine_name"

# --- Resolve paths ---
SIMULATION_FILES_DIR="$SLURM_SUBMIT_DIR"
AGENT_DIR="$(dirname "$SIMULATION_FILES_DIR")"
CASE_DIR="$AGENT_DIR/cases_agent/$task_id"

echo "Agent dir: $AGENT_DIR"
echo "Case dir:  $CASE_DIR"

mkdir -p "$CASE_DIR/logs"

if [ ! -f "$CASE_DIR/config.json" ]; then
    echo "Error: config.json not found at $CASE_DIR/config.json"
    exit 1
fi
echo "Using config.json:"
cat "$CASE_DIR/config.json"

# =============================================================================
# hero_initialize: mark task as running, dequeue from Hero
# =============================================================================
module load mamba
source activate AC

# =============================================================================
# Simulation: run the mock rental car model
# =============================================================================
echo "--- Mock simulation beginning at: $(date) ---"
python "$SIMULATION_FILES_DIR/mock_simulation.py" "$CASE_DIR" \
    > "$CASE_DIR/logs/simulation.out" 2> "$CASE_DIR/logs/simulation.err"
sim_exit=$?
echo "--- Mock simulation completed at: $(date) (exit code: $sim_exit) ---"

cat "$CASE_DIR/logs/simulation.out"

if [ $sim_exit -ne 0 ]; then
    echo "Simulation failed. See $CASE_DIR/logs/simulation.err"
    cat "$CASE_DIR/logs/simulation.err"
    exit 1
fi

# --- Extract cost from result.json ---
if [ ! -f "$CASE_DIR/result.json" ]; then
    echo "Error: result.json not written by mock_simulation.py"
    exit 1
fi

cost=$(python3 -c "import json; print(-json.load(open('$CASE_DIR/result.json'))['cost'])")

if [ -z "$cost" ]; then
    echo "Warning: Could not extract cost from result.json. Defaulting to -1."
    cost=-1
fi
echo "Negated cost result (stored as -cost for maximization): $cost"

# =============================================================================
# Write result file for the manager to pick up and pass to hero_finalize.
# hero_initialize and hero_finalize are called by the manager on the login node,
# which has outbound internet access. Compute nodes may not.
# =============================================================================
echo "$cost" > "$SLURM_SUBMIT_DIR/result_${task_id}.txt"
echo "Wrote result to result_${task_id}.txt"

echo "Job completed at: $(date)"
