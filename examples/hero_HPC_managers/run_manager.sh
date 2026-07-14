#!/bin/bash
SESSION_NAME="manager_session"

# Get the directory of the currently executing script
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# tmux requires TERM to be set to start its server in non-interactive shells.
# Set it here in case the SSH session didn't provide one.
export TERM=${TERM:-xterm-256color}

LOG_FILE="$WORK_DIR/manager.output.$1"

# $1 = machine_name, $2 = i_fidelity (default 0), $3 = env activation command
ENV_ACTIVATE_CMD="${3:-module load mamba && mamba activate AC}"

# kill existing session if exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

tmux new-session -d -s "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" \
  "cd $WORK_DIR && $ENV_ACTIVATE_CMD && python -u manager.py $1 ${2:-0} > $LOG_FILE 2>&1" Enter

echo "Started background session: $SESSION_NAME"
