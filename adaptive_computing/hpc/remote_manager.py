"""
remote_manager.py — Manages the Hero HPC manager tmux session on a remote login node.

Because ``adaptive_computing`` is installed in the AC conda environment and
``python_paths`` in ``hpc_config.py`` points directly to that Python binary,
this module is invoked over SSH with no shell-environment setup required::

    {python_path} -m adaptive_computing.hpc.remote_manager start <machine_name> <work_dir> [<fidelity>]
    {python_path} -m adaptive_computing.hpc.remote_manager stop  <machine_name> <work_dir>

Inside the module, ``sys.executable`` refers to the same Python binary, so it
is reused for launching ``manager.py`` and ``kill_scheduler_jobs`` inside tmux.
No environment activation commands are needed anywhere in the flow.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

SESSION_NAME = "manager_session"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_env() -> dict:
    """Return an environment dict with TERM set (required by tmux)."""
    env = dict(os.environ)
    env.setdefault("TERM", "xterm-256color")
    return env


def _ensure_tmux(env: dict) -> None:
    """Load tmux via the module system if it is not already in PATH.

    Uses ``bash -l`` so that the HPC module system (sourced in login profile)
    is available for ``module load tmux``.  Updates *env* in-place so that
    subsequent subprocess calls using ``env=env`` can find the tmux binary.
    """
    if shutil.which("tmux"):
        return  # already in PATH — nothing to do

    # Ask a login shell to load tmux and report the resulting PATH.
    # We capture the new PATH so we can inject it into env for all later calls.
    result = subprocess.run(
        "bash -l -c 'module load tmux 2>/dev/null && echo \"$PATH\"'",
        shell=True, capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        env["PATH"] = result.stdout.strip()


def _tmux(*args: str, env: dict) -> subprocess.CompletedProcess:
    """Run a tmux subcommand."""
    return subprocess.run(["tmux"] + list(args), env=env, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(machine_name: str, work_dir: str, fidelity: int = 0) -> None:
    """Start ``manager.py`` in a detached tmux session.

    Uses ``sys.executable`` so that ``manager.py`` runs under the same Python
    binary that was used to invoke this module.

    Args:
        machine_name: Logical machine name passed to ``manager.py``.
        work_dir:     Absolute path to the application directory on this
                      machine (contains ``manager.py`` and ``hpc_config.py``).
        fidelity:     Fidelity level index (default 0).
    """
    env = _make_env()
    _ensure_tmux(env)

    log_file = os.path.join(work_dir, f"manager.output.{machine_name}")

    # Kill any stale session to ensure a clean start.
    _tmux("kill-session", "-t", SESSION_NAME, env=env)

    # Start a new detached session using setsid so the tmux server is placed
    # in a new process session.  Without this, HPC systems running systemd
    # with KillUserProcesses=yes will kill the tmux server (and everything in
    # it) as soon as the short-lived SSH command that launched it disconnects.
    result = subprocess.run(
        f"setsid tmux new-session -d -s {SESSION_NAME}",
        shell=True, env=env, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: failed to create tmux session: {result.stderr.strip()}")
        sys.exit(1)

    # Build the shell command that will run inside the tmux session.
    # sys.executable is the full path to the AC Python binary, so no
    # environment activation is needed.
    inner_cmd = (
        f"cd {work_dir!r} && "
        f"{sys.executable!r} -u manager.py {machine_name} {fidelity} "
        f"> {log_file!r} 2>&1"
    )
    _tmux("send-keys", "-t", SESSION_NAME, inner_cmd, "Enter", env=env)
    print(f"Started background session: {SESSION_NAME}")


def stop(machine_name: str, work_dir: str) -> None:
    """Stop the manager tmux session and cancel all tracked scheduler jobs.

    Args:
        machine_name: Logical machine name; passed to ``kill_scheduler_jobs``.
        work_dir:     Absolute path to the application directory on this machine.
    """
    env = _make_env()
    _ensure_tmux(env)

    log_file = os.path.join(work_dir, f"kill_scheduler_jobs.output.{machine_name}")

    # Send Ctrl-C to gracefully stop the running manager loop.
    _tmux("send-keys", "-t", SESSION_NAME, "C-c", env=env)
    time.sleep(1)

    # Cancel all tracked scheduler jobs, then kill the session.
    inner_cmd = (
        f"cd {work_dir!r} && "
        f"{sys.executable!r} -u -m adaptive_computing.hpc.kill_scheduler_jobs {machine_name} "
        f"> {log_file!r} 2>&1 && "
        f"tmux kill-session -t {SESSION_NAME}"
    )
    _tmux("send-keys", "-t", SESSION_NAME, inner_cmd, "Enter", env=env)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    usage = (
        "Usage:\n"
        "  {python} -m adaptive_computing.hpc.remote_manager "
        "start <machine_name> <work_dir> [<fidelity>]\n"
        "  {python} -m adaptive_computing.hpc.remote_manager "
        "stop  <machine_name> <work_dir>"
    ).format(python=sys.executable)
    if len(sys.argv) < 2 or sys.argv[1] not in ("start", "stop"):
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "start":
        if len(sys.argv) < 4:
            print(usage)
            sys.exit(1)
        machine_name = sys.argv[2]
        work_dir     = sys.argv[3]
        fidelity     = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        start(machine_name, work_dir, fidelity)

    elif cmd == "stop":
        if len(sys.argv) < 4:
            print(usage)
            sys.exit(1)
        machine_name = sys.argv[2]
        work_dir     = sys.argv[3]
        stop(machine_name, work_dir)


if __name__ == "__main__":
    main()
