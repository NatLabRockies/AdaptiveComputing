"""
autonomous.py — SSH-based lifecycle management for remote HPC manager daemons.

The manager daemon (manager.py) must run on the HPC login node where the
scheduler (sbatch / qsub) is available.  These utilities start, monitor, and
stop that daemon via SSH + tmux from the controller machine.

Typical usage
-------------
::

    from adaptive_computing.hpc.autonomous import (
        setup_remote_state, run_remote_managers,
        wait_for_managers, cleanup_remote_managers,
    )

    setup_remote_state(
        machine_names    = hpc_config.machine_names,
        remote_usernames = hpc_config.remote_usernames,
        remote_hosts     = hpc_config.remote_hosts,
        remote_dirs      = hpc_config.remote_dirs,
        python_paths     = hpc_config.python_paths,
        # Optional — only needed for multi-hop systems like Aurora:
        proxy_hosts      = getattr(hpc_config, 'proxy_hosts', {}),
    )
    run_remote_managers()
    wait_for_managers()
    # ... run your workflow ...
    cleanup_remote_managers()

Multi-hop SSH (e.g. Aurora at ALCF)
------------------------------------
Some HPC systems do not allow direct SSH to specific login nodes from outside
the facility network, but they do allow SSH to a load-balanced gateway.  Use
``proxy_hosts`` to route through the gateway to a pinned login node::

    # hpc_config.py
    remote_hosts = {'aurora': 'aurora-uan-0010'}        # specific node (node_name)
    proxy_hosts  = {'aurora': 'aurora.alcf.anl.gov'}   # external gateway (hostname)

The default proxy_type is ``"jump"`` (``ssh -J gateway node``).  Set
``proxy_type = {"aurora": "nested"}`` in hpc_config.py when the node only
accepts connections from within the facility network (e.g. ALCF Aurora uses
host-based auth — login nodes trust the gateway but not external keys).
With ``"nested"``, AC runs ``ssh gateway "ssh node cmd"`` so the gateway
initiates the inner SSH using its own credentials.

A signal handler (SIGINT / SIGTERM / SIGHUP) is registered by
``setup_remote_state`` so that Ctrl-C from the controller triggers a clean
remote shutdown automatically.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time

SESSION_NAME = "manager_session"

# ---------------------------------------------------------------------------
# Module-level state (populated by setup_remote_state)
# ---------------------------------------------------------------------------

_machine_names: list[str] = []
_remote_usernames: dict[str, str] = {}
_remote_hosts: dict[str, str] = {}
_remote_dirs: dict[str, str] = {}
_python_paths: dict[str, str] = {}
_proxy_hosts: dict[str, str] = {}   # optional jump / gateway hosts
_proxy_type: dict[str, str] = {}    # 'jump' (default) or 'nested' per machine


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def setup_remote_state(
    machine_names: list[str],
    remote_usernames: dict[str, str],
    remote_hosts: dict[str, str],
    remote_dirs: dict[str, str],
    python_paths: dict[str, str],
    proxy_hosts: dict[str, str] | None = None,
    proxy_type: dict[str, str] | None = None,
) -> None:
    """Populate module-level SSH settings and register a clean-shutdown signal handler.

    Must be called before :func:`run_remote_managers`.  Safe to call from the
    main thread only (Python restricts signal registration to the main thread;
    :mod:`ac_mcp.run_manager` disables signal registration when calling from a
    worker thread).

    Args:
        machine_names:   List of logical machine names (keys for the dicts below).
        remote_usernames: ``{machine_name: ssh_username}``
        remote_hosts:    ``{machine_name: node_name}`` — the specific login node
                         to connect to (e.g. ``"aurora-uan-0010"``).  For simple
                         clusters where direct SSH is allowed, this is the same
                         as the public hostname.
        remote_dirs:     ``{machine_name: absolute_remote_path}`` — directory where
                         ``manager.py`` lives on the remote machine.
        python_paths:    ``{machine_name: absolute_path_to_python}`` — full path to
                         the Python executable in the AC environment on each remote
                         machine, e.g. ``"/home/user/.conda-envs/AC/bin/python"``.
        proxy_hosts:     ``{machine_name: hostname}`` — optional SSH gateway host.
                         When set, all SSH to that machine routes through the gateway.
                         Use for clusters like Aurora where direct SSH to specific
                         login nodes is blocked from outside.
        proxy_type:      ``{machine_name: "jump" | "nested"}`` — how to use the
                         proxy gateway (default ``"jump"``).

                         * ``"jump"``   — ``ssh -J user@gateway user@node cmd``
                           (local machine authenticates to node via gateway TCP tunnel).
                           Works when the node accepts external SSH keys.

                         * ``"nested"`` — ``ssh user@gateway "ssh user@node cmd"``
                           (gateway executes the inner SSH, using its own credentials).
                           Required when the node only accepts connections from within
                           the facility network (e.g. ALCF Aurora, where login nodes
                           use host-based auth from the gateway).
    """
    global _machine_names, _remote_usernames, _remote_hosts, _remote_dirs
    global _python_paths, _proxy_hosts, _proxy_type
    _machine_names = machine_names
    _remote_usernames = remote_usernames
    _remote_hosts = remote_hosts
    _remote_dirs = remote_dirs
    _python_paths = python_paths
    _proxy_hosts = proxy_hosts or {}
    _proxy_type  = proxy_type  or {}

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _signal_handler)


def _signal_handler(sig, frame):
    print(
        f"\nReceived signal {sig}. Canceling all scheduler jobs and "
        "terminating the remote queue managers..."
    )
    cleanup_remote_managers()
    # os._exit avoids sending SystemExit to Hero (which it cannot handle).
    os._exit(0)


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def _build_ssh_cmd(
    machine_name: str,
    remote_cmd: list[str],
    connect_timeout: int = 15,
) -> list[str]:
    """Build the full subprocess command that runs *remote_cmd* on *machine_name*.

    Handles three cases based on ``proxy_hosts`` / ``proxy_type`` config:

    * **No proxy** — direct SSH: ``ssh [opts] user@node cmd``
    * **proxy_type="jump"** (default) — ProxyJump: ``ssh [opts] -J user@gw user@node cmd``
      The local machine authenticates directly to the node via a TCP tunnel through
      the gateway.  Requires the node to accept the local SSH key.
    * **proxy_type="nested"** — nested SSH: ``ssh [opts] user@gw "ssh user@node cmd"``
      The gateway executes the inner SSH command using its own credentials.  Required
      when the node only accepts connections from within the facility network
      (e.g. ALCF Aurora — login nodes use host-based auth from the gateway).

    Args:
        machine_name:    Logical machine name (key into the module-level dicts).
        remote_cmd:      Command and its arguments to run on the target node, as a
                         list.  A single-element list with a shell command string
                         (e.g. ``["bash -l -c 'cmd'"]``) is the common form.
        connect_timeout: SSH ConnectTimeout in seconds.

    Returns:
        A list suitable for passing directly to :func:`subprocess.run`.
    """
    user  = _remote_usernames[machine_name]
    node  = _remote_hosts[machine_name]
    proxy = _proxy_hosts.get(machine_name)
    ptype = _proxy_type.get(machine_name, "jump")

    base_opts = [
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ConnectTimeout={connect_timeout}",
    ]

    if proxy and ptype == "nested":
        # Gateway executes the inner SSH; gateway → node auth is handled internally
        # (e.g. host-based auth on Aurora).  The inner command is passed as a single
        # shell-quoted string so the gateway shell reconstructs it correctly.
        inner_args = [f"{user}@{node}"] + remote_cmd
        inner_str  = " ".join(shlex.quote(a) for a in inner_args)
        return ["ssh"] + base_opts + [f"{user}@{proxy}", f"ssh {inner_str}"]
    elif proxy:
        # ProxyJump: local machine authenticates to node through the gateway TCP tunnel.
        return (["ssh"] + base_opts +
                ["-J", f"{user}@{proxy}", f"{user}@{node}"] + remote_cmd)
    else:
        return ["ssh"] + base_opts + [f"{user}@{node}"] + remote_cmd


# ---------------------------------------------------------------------------
# Hostname check (advisory; never blocks startup)
# ---------------------------------------------------------------------------

def _check_remote_hostname(machine_name: str) -> None:
    """Warn if the SSH connection landed on a different node than configured.

    Load balancers (e.g. ``aurora.alcf.anl.gov``) may route each connection to
    a different login node, breaking tmux session reuse.  Running ``hostname``
    over SSH lets us detect this and suggest the specific node to pin to.
    Skipped when ``proxy_hosts`` is already configured for this machine (the
    user has already opted into pinned-node routing).
    """
    if machine_name in _proxy_hosts:
        return  # already pinned via proxy — nothing to warn about
    configured_host = _remote_hosts[machine_name]
    try:
        result = subprocess.run(
            _build_ssh_cmd(machine_name, ["bash -l -c 'hostname -f'"]),
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0:
            return
        actual_host = result.stdout.strip()
        configured_short = configured_host.split(".")[0]
        actual_short = actual_host.split(".")[0]
        if configured_short != actual_short:
            print(
                f"⚠️  WARNING: remote_hosts for '{machine_name}' is set to "
                f"'{configured_host}', but the SSH connection landed on "
                f"'{actual_short}'.\n"
                f"   tmux sessions may not be reachable if the load balancer "
                f"routes to a different node each time.\n"
                f"   If direct SSH to the node is blocked (e.g. Aurora), use:\n"
                f"       remote_hosts = {{'{machine_name}': '{actual_short}'}}\n"
                f"       proxy_hosts  = {{'{machine_name}': '{configured_host}'}}"
            )
    except (subprocess.TimeoutExpired, Exception):
        pass  # advisory only


# ---------------------------------------------------------------------------
# Manager session polling
# ---------------------------------------------------------------------------

def _is_manager_running(machine_name: str) -> bool:
    """Return True if the manager tmux session exists AND manager.py is running in it."""
    cmd_str = (
        "bash -l -c 'command -v tmux &>/dev/null || module load tmux 2>/dev/null; "
        f"tmux list-panes -t {SESSION_NAME} -F \"#{{pane_current_command}}\" 2>/dev/null "
        "| grep -q python && echo ready || echo not_ready'"
    )
    try:
        result = subprocess.run(
            _build_ssh_cmd(machine_name, [cmd_str]),
            capture_output=True, text=True, timeout=20,
        )
        return result.returncode == 0 and result.stdout.strip() == "ready"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_remote_managers() -> None:
    """Launch the manager daemon on every configured machine via SSH.

    Invokes ``python -m adaptive_computing.hpc.remote_manager start`` on the
    remote login node — no shell scripts need to be copied to the remote
    machine; the module is available through the AC conda environment.

    Skips machines where a ``manager_session`` is already running (to avoid
    duplicate submissions).  Logs success / failure for each machine but does
    not raise on individual failures — call :func:`wait_for_managers` to
    confirm all managers started.
    """
    print("Starting remote managers...")
    for machine_name in _machine_names:
        _check_remote_hostname(machine_name)
        if _is_manager_running(machine_name):
            user  = _remote_usernames[machine_name]
            host  = _remote_hosts[machine_name]
            proxy = _proxy_hosts.get(machine_name)
            ptype = _proxy_type.get(machine_name, "jump")
            if proxy and ptype == "nested":
                kill_cmd = (f"ssh {user}@{proxy} "
                            f"\"ssh {user}@{host} 'tmux kill-session -t {SESSION_NAME}'\"")
            elif proxy:
                kill_cmd = (f"ssh -J {user}@{proxy} {user}@{host} "
                            f"\"tmux kill-session -t {SESSION_NAME}\"")
            else:
                kill_cmd = f"ssh {user}@{host} \"tmux kill-session -t {SESSION_NAME}\""
            print(
                f"⚠️  {machine_name}: manager session already running "
                f"— skipping launch to avoid duplicates\n"
                f"   To abort and start fresh, kill the session then re-run:\n"
                f"     {kill_cmd}"
            )
            continue
        python = _python_paths[machine_name]
        remote_dir = _remote_dirs[machine_name]
        ssh_command = _build_ssh_cmd(
            machine_name,
            [f"{python} -m adaptive_computing.hpc.remote_manager start {machine_name} {remote_dir}"],
            connect_timeout=30,
        )
        print(f"Launching manager on {machine_name}")
        try:
            result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ Manager command sent to {machine_name} successfully")
            else:
                print(f"❌ Failed to start manager on {machine_name}")
                print(f"   STDOUT: {result.stdout}")
                print(f"   STDERR: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⚠️  SSH timeout for {machine_name} (command may still be running)")
        except Exception as e:
            print(f"❌ Error connecting to {machine_name}: {e}")

    print("Remote manager launch commands sent. Waiting for confirmation...")


def wait_for_managers(timeout: int = 60, poll_interval: int = 3) -> None:
    """Block until every remote manager's tmux session is confirmed running.

    Args:
        timeout:       Maximum seconds to wait before raising :class:`RuntimeError`.
        poll_interval: Seconds between polling attempts.

    Raises:
        RuntimeError: If one or more managers have not started within *timeout* seconds.
    """
    print(f"Waiting for remote managers to start (timeout={timeout}s)...")
    deadline = time.time() + timeout
    pending = set(_machine_names)

    while pending:
        if time.time() >= deadline:
            raise RuntimeError(
                f"Timed out after {timeout}s waiting for managers on: "
                + ", ".join(sorted(pending))
                + "\nCheck the manager logs on each machine for details."
            )
        still_pending = set()
        for machine_name in list(pending):
            if _is_manager_running(machine_name):
                print(f"✅ {machine_name}: manager session is running")
            else:
                still_pending.add(machine_name)
        pending = still_pending
        if pending:
            remaining = max(0, int(deadline - time.time()))
            print(
                f"  Still waiting for: {', '.join(sorted(pending))} "
                f"({remaining}s remaining)"
            )
            time.sleep(poll_interval)

    print("All remote managers are running.")


def cleanup_remote_managers() -> None:
    """Cancel all scheduler jobs and stop every remote manager tmux session."""
    print("\nCanceling all scheduler jobs and terminating remote queue managers...")
    for machine_name in _machine_names:
        python = _python_paths[machine_name]
        remote_dir = _remote_dirs[machine_name]
        ssh_command = _build_ssh_cmd(
            machine_name,
            [f"{python} -m adaptive_computing.hpc.remote_manager stop {machine_name} {remote_dir}"],
            connect_timeout=30,
        )
        try:
            subprocess.run(ssh_command, check=True)
            print(f"Remote cleanup completed on {_remote_hosts[machine_name]}.")
        except subprocess.CalledProcessError as e:
            print(
                f"Cleanup command failed on {_remote_hosts[machine_name]} "
                f"(exit {e.returncode}): {e}"
            )
