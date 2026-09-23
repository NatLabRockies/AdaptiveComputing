#!/bin/bash
#PBS -N rental_agent
#PBS -A newbridge
#PBS -l walltime=00:10:00
#PBS -l select=1:ncpus=1
# Output/error logs go to cases_agent/<task_id>/logs/ — set via qsub -o / -e.

# =============================================================================
# Arguments passed by manager.py via qsub -v:
#   task_id = Hero task UUID
# =============================================================================

if [ -z "$task_id" ]; then
    echo "Error: Missing variable 'task_id'"
    echo "Usage: qsub -v \"task_id=<uuid>\" script_pbs.sh"
    exit 1
fi

echo "Job started at: $(date)"
echo "Job ID: $PBS_JOBID  Node: $HOSTNAME"
echo "Task ID: $task_id"

# --- Resolve paths ---
# PBS_O_WORKDIR is the directory qsub was called from (simulation_files/).
SIMULATION_FILES_DIR="$PBS_O_WORKDIR"
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
# Load environment  (adapt module names for your PBS system)
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
# =============================================================================
echo "$cost" > "$SIMULATION_FILES_DIR/result_${task_id}.txt"
echo "Wrote result to result_${task_id}.txt"

echo "Job completed at: $(date)"
