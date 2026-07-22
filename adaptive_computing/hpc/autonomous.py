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
        machine_names  = hpc_config.machine_names,
        remote_usernames = hpc_config.remote_usernames,
        remote_hosts   = hpc_config.remote_hosts,
        remote_dirs    = hpc_config.remote_dirs,
        python_paths   = hpc_config.python_paths,
    )
    run_remote_managers()
    wait_for_managers()
    # ... run your workflow ...
    cleanup_remote_managers()

A signal handler (SIGINT / SIGTERM / SIGHUP) is registered by
``setup_remote_state`` so that Ctrl-C from the controller triggers a clean
remote shutdown automatically.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

from .remote_manager import SESSION_NAME

# ---------------------------------------------------------------------------
# Module-level state (populated by setup_remote_state)
# ---------------------------------------------------------------------------

_machine_names: list[str] = []
_remote_usernames: dict[str, str] = {}
_remote_hosts: dict[str, str] = {}
_remote_dirs: dict[str, str] = {}
_python_paths: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def setup_remote_state(
    machine_names: list[str],
    remote_usernames: dict[str, str],
    remote_hosts: dict[str, str],
    remote_dirs: dict[str, str],
    python_paths: dict[str, str],
) -> None:
    """Populate module-level SSH settings and register a clean-shutdown signal handler.

    Must be called before :func:`run_remote_managers`.  Safe to call from the
    main thread only (Python restricts signal registration to the main thread;
    :mod:`ac_mcp.run_manager` disables signal registration when calling from a
    worker thread).

    Args:
        machine_names:   List of logical machine names (keys for the dicts below).
        remote_usernames: ``{machine_name: ssh_username}``
        remote_hosts:    ``{machine_name: ssh_hostname_or_ip}``
        remote_dirs:     ``{machine_name: absolute_remote_path}`` — directory where
                         ``manager.py`` lives on the remote machine.
        python_paths:    ``{machine_name: absolute_path_to_python}`` — full path to
                         the Python executable in the AC environment on each remote
                         machine, e.g. ``"/home/user/.conda-envs/AC/bin/python"``.
    """
    global _machine_names, _remote_usernames, _remote_hosts, _remote_dirs, _python_paths
    _machine_names = machine_names
    _remote_usernames = remote_usernames
    _remote_hosts = remote_hosts
    _remote_dirs = remote_dirs
    _python_paths = python_paths

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
# Hostname check (advisory; never blocks startup)
# ---------------------------------------------------------------------------

def _check_remote_hostname(machine_name: str) -> None:
    """Warn if the SSH connection landed on a different node than configured.

    Load balancers (e.g. ``aurora.alcf.anl.gov``) may route each connection to
    a different login node, breaking tmux session reuse.  Running ``hostname``
    over SSH lets us detect this and suggest the specific node to pin to.
    """
    configured_host = _remote_hosts[machine_name]
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=15",
                f"{_remote_usernames[machine_name]}@{configured_host}",
                "bash -l -c 'hostname -f'",
            ],
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
                f"   Recommended fix in hpc_config.py:\n"
                f"       remote_hosts = {{'{machine_name}': '{actual_short}'}}"
            )
    except (subprocess.TimeoutExpired, Exception):
        pass  # advisory only


# ---------------------------------------------------------------------------
# Manager session polling
# ---------------------------------------------------------------------------

def _is_manager_running(machine_name: str) -> bool:
    """Return True if the manager tmux session exists on *machine_name*."""
    ssh_command = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
        (
            "bash -l -c 'command -v tmux &>/dev/null || module load tmux 2>/dev/null; "
            f"tmux list-sessions 2>/dev/null | grep -q {SESSION_NAME} "
            "&& echo ready || echo not_ready'"
        ),
    ]
    try:
        result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=20)
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
            user = _remote_usernames[machine_name]
            host = _remote_hosts[machine_name]
            print(
                f"⚠️  {machine_name}: manager session already running "
                f"— skipping launch to avoid duplicates\n"
                f"   To abort and start fresh, kill the session then re-run:\n"
                f"     ssh {user}@{host} \"tmux kill-session -t {SESSION_NAME}\""
            )
            continue
        python = _python_paths[machine_name]
        remote_dir = _remote_dirs[machine_name]
        ssh_command = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
            f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
            f"{python} -m adaptive_computing.hpc.remote_manager start {machine_name} {remote_dir}",
        ]
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
        ssh_command = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=30",
            f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
            f"{python} -m adaptive_computing.hpc.remote_manager stop {machine_name} {remote_dir}",
        ]
        try:
            subprocess.run(ssh_command, check=True)
            print(f"Remote cleanup completed on {_remote_hosts[machine_name]}.")
        except subprocess.CalledProcessError as e:
            print(
                f"Cleanup command failed on {_remote_hosts[machine_name]} "
                f"(exit {e.returncode}): {e}"
            )
