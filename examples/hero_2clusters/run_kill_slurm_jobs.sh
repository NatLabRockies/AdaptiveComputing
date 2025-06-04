#!/bin/bash                                                                                                 

SESSION_NAME="manager_session"
# Get the directory of the currently executing script
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_FILE="$WORK_DIR/kill_slurm_jobs.output"

# Kill manager.py, which was running
tmux send-keys -t $SESSION_NAME C-c
sleep 1
# For all tasks in the Hero queue that have a slurm job id, cancel the slurm job
# Terminate the current tmux session
tmux send-keys -t $SESSION_NAME "cd $WORK_DIR && python -u kill_slurm_jobs.py > \"$LOG_FILE\" 2>&1 && tmux kill-session -t $SESSION_NAME" C-m
