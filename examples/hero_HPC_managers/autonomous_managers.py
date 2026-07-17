import signal
import os
import subprocess
import time

# Internal globals for the signal handler and remote functions
_machine_names = []
_remote_usernames = {}
_remote_hosts = {}
_remote_dirs = {}
_env_activate_cmds = {}

def setup_remote_state(machine_names, remote_usernames, remote_hosts, remote_dirs, env_activate_cmds):
    global _machine_names, _remote_usernames, _remote_hosts, _remote_dirs, _env_activate_cmds
    _machine_names = machine_names
    _remote_usernames = remote_usernames
    _remote_hosts = remote_hosts
    _remote_dirs = remote_dirs
    _env_activate_cmds = env_activate_cmds
    # Register signal handler for clean shutdown on Ctrl-C (SIGINT),
    # termination requests (SIGTERM), and terminal hangup (SIGHUP).
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, signal_handler)

def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}. Canceling all scheduler jobs and then terminating the remote queue managers...")
    cleanup_remote_managers()
    os._exit(0) # unlike sys.exit(0), os._exit(0) avoids sending SystemExit signal to Hero, which it doesn't know how to handle

def _check_remote_hostname(machine_name):
    """Warn if the SSH connection landed on a different node than configured.

    Load balancers (e.g. aurora.alcf.anl.gov) may route each connection to a
    different login node, which breaks tmux session reuse.  Running `hostname`
    over SSH lets us detect this and suggest the specific node to pin to.
    """
    configured_host = _remote_hosts[machine_name]
    try:
        result = subprocess.run(
            ["ssh",
             "-o", "BatchMode=yes",
             "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=15",
             f"{_remote_usernames[machine_name]}@{configured_host}",
             f"bash -l -c 'hostname -f'"],
            capture_output=True, text=True, timeout=20
        )
        if result.returncode != 0:
            return  # Can't check; silently skip
        actual_host = result.stdout.strip()
        # Use short hostnames for comparison and recommendation — FQDNs may not resolve
        configured_short = configured_host.split('.')[0]
        actual_short = actual_host.split('.')[0]
        if configured_short != actual_short:
            print(f"⚠️  WARNING: remote_hosts for '{machine_name}' is set to '{configured_host}', "
                  f"but the SSH connection landed on '{actual_short}'.")
            print(f"   tmux sessions may not be reachable on future connections if the load balancer "
                  f"routes to a different node each time.")
            print(f"   Recommended fix in hpc_config.py:")
            print(f"       remote_hosts = {{'{machine_name}': '{actual_short}'}}")
    except (subprocess.TimeoutExpired, Exception):
        pass  # Hostname check is advisory; never block startup


def _is_manager_running(machine_name):
    """Return True if a manager_session tmux session is active on *machine_name*.

    Uses a single SSH call with a short timeout so callers can poll cheaply.
    Returns False on any connection or timeout error.
    """
    ssh_command = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
        f"bash -l -c 'command -v tmux &>/dev/null || module load tmux 2>/dev/null; tmux list-sessions 2>/dev/null | grep -q manager_session && echo ready || echo not_ready'"
    ]
    try:
        result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=20)
        return result.returncode == 0 and 'ready' in result.stdout
    except Exception:
        return False


def run_remote_managers():
    print("Starting remote managers...")
    for machine_name in _machine_names:
        _check_remote_hostname(machine_name)
        if _is_manager_running(machine_name):
            print(f"⚠️  {machine_name}: manager session already running — skipping launch to avoid duplicates")
            continue
        ssh_command = [
            "ssh",
            "-o", "BatchMode=yes",  # Disable interactive prompts
            "-o", "StrictHostKeyChecking=no",  # Don't prompt for host key verification
            "-o", "ConnectTimeout=30",  # Set connection timeout
            f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
            f"bash -l -c 'cd {_remote_dirs[machine_name]} && nohup ./run_manager.sh {machine_name} 0 \"{_env_activate_cmds[machine_name]}\" > manager_{machine_name}.log 2>&1 &'"
        ]
        print(f"Launching manager on {machine_name}: {' '.join(ssh_command)}")
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
    
    print("Remote manager launch commands completed. Check logs to verify they're running.")

def wait_for_managers(timeout=60, poll_interval=3):
    """
    Block until every remote manager's tmux session is confirmed running,
    or raise RuntimeError if *timeout* seconds elapse first.

    Args:
        timeout (int): Maximum seconds to wait before giving up. Default 60.
        poll_interval (int): Seconds between polling attempts. Default 3.
    """
    print(f"Waiting for remote managers to start (timeout={timeout}s)...")
    deadline = time.time() + timeout
    pending = set(_machine_names)

    while pending:
        if time.time() >= deadline:
            raise RuntimeError(
                f"Timed out after {timeout}s waiting for managers to start on: "
                + ", ".join(sorted(pending))
                + "\nCheck the manager logs on each machine for details."
            )
        still_pending = set()
        for machine_name in list(pending):
            if _is_manager_running(machine_name):
                print(f"\u2705 {machine_name}: manager session is running")
            else:
                still_pending.add(machine_name)
        pending = still_pending
        if pending:
            remaining = max(0, int(deadline - time.time()))
            print(f"  Still waiting for: {', '.join(sorted(pending))} ({remaining}s remaining)")
            time.sleep(poll_interval)

    print("All remote managers are running.")


def cleanup_remote_managers():
    print("\nCanceling all scheduler jobs and then terminating the remote queue managers...")
    for machine_name in _machine_names:
        ssh_command = [
            "ssh",
            "-o", "BatchMode=yes",  # Disable interactive prompts
            "-o", "StrictHostKeyChecking=no",  # Don't prompt for host key verification
            "-o", "ConnectTimeout=30",  # Set connection timeout
            f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
            f"bash -l -c 'cd {_remote_dirs[machine_name]} && ./run_kill_scheduler_jobs.sh {machine_name}'"
        ]
        try:
            subprocess.run(ssh_command, check=True)
            print(f"Remote cleanup completed on {_remote_hosts[machine_name]} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Cleanup command failed on {_remote_hosts[machine_name]} with exit code {e.returncode}:\n{e}")
