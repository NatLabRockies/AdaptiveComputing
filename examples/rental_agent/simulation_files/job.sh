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
#   $2 = machine_name  (e.g. "kestrel")
# =============================================================================
task_id=$1
machine_name=$2

for var_name in task_id machine_name; do
    if [ -z "${!var_name}" ]; then
        echo "Error: Missing argument '$var_name'"
        echo "Usage: sbatch job.sh <task_id> <machine_name>"
        exit 1
    fi
done

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

echo "Running hero_initialize for task_id=$task_id on $machine_name"
python -m adaptive_computing.hero_utils.hero_initialize "$task_id" "$machine_name"
hero_init_code=$?

if [ $hero_init_code -eq 2 ]; then
    echo "hero_initialize: task already claimed by another machine. Exiting cleanly."
    exit 0
elif [ $hero_init_code -ne 0 ]; then
    echo "hero_initialize failed with code $hero_init_code. Exiting."
    exit 1
fi

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
# hero_finalize: publish cost back to the Hero queue
# =============================================================================
echo "Running hero_finalize: result=$cost, task_id=$task_id, machine=$machine_name"
python -m adaptive_computing.hero_utils.hero_finalize "$cost" "$task_id" "$machine_name"
if [ $? -ne 0 ]; then
    echo "hero_finalize failed. Exiting."
    exit 1
fi

echo "Job completed at: $(date)"
