#!/bin/bash
# Start the AC MCP server in a background tmux session on the current node.
# Run this on whichever login node the agent will also run on.
#
# Usage: ./start_server.sh <storage-dir> [port]
#
# storage-dir  Absolute path to the application directory that will hold
#              registry.json and the experiments/ pickle folder.
#              E.g. /projects/newbridge/kgriffin/stdp-mnist/agent
# port         HTTP port for the MCP server (default: 8765)
#
# The agent connects to http://localhost:<port>/mcp.
# Set AC_MCP_URL in the environment only if you need a non-default port.

STORAGE_DIR=${1:?"Usage: $0 <storage-dir> [port]"}
PORT=${2:-8765}
SESSION="ac_mcp_server"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running on $(hostname). Kill it first with:"
    echo "  tmux kill-session -t $SESSION"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AC_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/kgriffin/.conda-envs/AC/bin/python"

tmux new-session -d -s "$SESSION" \
    "cd '$AC_ROOT' && '$PYTHON' -m ac_mcp.server --storage-dir '$STORAGE_DIR' --port $PORT 2>&1 | tee /tmp/ac_mcp_server.log"

echo "AC MCP server starting on $(hostname):$PORT"
echo "Storage dir: $STORAGE_DIR"
echo "Logs: tmux attach -t $SESSION   or   tail -f /tmp/ac_mcp_server.log"
echo "Stop: tmux kill-session -t $SESSION"
