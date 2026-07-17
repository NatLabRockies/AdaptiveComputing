#!/bin/bash

SESSION_NAME="manager_session"
# Get the directory of the currently executing script
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_FILE="$WORK_DIR/kill_scheduler_jobs.output.$1"

# Stop manager.py and kill its tmux session synchronously so that this SSH
# call (from cleanup_remote_managers) does not return until cleanup is done.
# The old approach used "tmux send-keys ... && tmux kill-session" which ran
# asynchronously inside the pane; the SSH call returned before the session was
# actually killed, causing a race condition where run_manager.sh could recreate
# the session and the deferred kill-session would kill the NEW manager.
tmux send-keys -t $SESSION_NAME C-c 2>/dev/null || true
sleep 1
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Cancel all Slurm jobs for queued/running Hero tasks synchronously in this shell.
module load mamba && source activate AC && cd $WORK_DIR && python -u kill_slurm_jobs.py $1 > "$LOG_FILE" 2>&1
