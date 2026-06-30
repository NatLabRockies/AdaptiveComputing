#!/bin/bash
# run_manager.sh — start the AC/Hero queue manager in a persistent tmux session
# Usage: ./run_manager.sh <machine_name>   (e.g.  ./run_manager.sh kestrel)

SESSION_NAME="manager_session"

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$WORK_DIR/manager.$1.log"

# Kill any existing session so we start fresh
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" \
  "cd $WORK_DIR && module load mamba && source activate AC && python -u manager.py $1 > $LOG_FILE 2>&1" Enter

echo "Started tmux session '$SESSION_NAME' — manager log: $LOG_FILE"
