#!/bin/bash
# Stop the AC MCP server and clean up any remote HPC managers it launched.
#
# Sending SIGHUP via tmux kill-session triggers the server's signal handler,
# which calls cleanup_remote_managers() (cancels scheduler jobs and kills the
# remote manager_session tmux) before exiting.
#
# Run this on the same node where start_server.sh was run.

SESSION="ac_mcp_server"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "No '$SESSION' tmux session found on $(hostname) — nothing to stop."
    exit 0
fi

echo "Stopping AC MCP server (sending SIGHUP to trigger remote cleanup)..."
tmux kill-session -t "$SESSION"
echo "Done. Remote manager cleanup was triggered by the server's SIGHUP handler."
echo "If cleanup did not run (e.g. server was stuck before HPC setup), kill the"
echo "remote manager_session manually:"
echo "  ssh <login-node> 'tmux kill-session -t manager_session'"
