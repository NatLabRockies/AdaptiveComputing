#!/bin/bash                                                                                                 

SESSION_NAME="manager_session"
# Get the directory of the currently executing script
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_FILE="$WORK_DIR/kill_scheduler_jobs.output.$1"

command -v tmux &>/dev/null || module load tmux

# Kill manager.py, which was running
tmux send-keys -t $SESSION_NAME C-c
sleep 1
# For all tasks in the Hero queue that have a scheduler job id, cancel the job
# Terminate the current tmux session
tmux send-keys -t $SESSION_NAME "cd $WORK_DIR && python -u -m adaptive_computing.hpc.kill_scheduler_jobs $1 > \"$LOG_FILE\" 2>&1 && tmux kill-session -t $SESSION_NAME" C-m
