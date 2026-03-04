#!/bin/bash
SESSION_NAME="manager_session"

# Get the directory of the currently executing script
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_FILE="$WORK_DIR/manager.output.$1"

# kill existing session if exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" \
  "cd $WORK_DIR && module load conda && conda activate AC_hero && python -u manager.py $1 > $LOG_FILE 2>&1" Enter

echo "Started background session: $SESSION_NAME"
