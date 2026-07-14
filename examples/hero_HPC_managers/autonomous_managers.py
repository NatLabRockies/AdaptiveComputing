import signal
import os
import subprocess

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
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

def signal_handler(sig, frame):
    print(f"\nReceived signal {sig}. Canceling all slurm jobs and then terminating the remote queue managers...")
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


def run_remote_managers():
    print("Starting remote managers...")
    for machine_name in _machine_names:
        _check_remote_hostname(machine_name)
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

def verify_remote_managers():
    """Check if remote managers are actually running"""
    print("\nVerifying remote managers...")
    for machine_name in _machine_names:
        ssh_command = [
            "ssh",
            "-o", "BatchMode=yes",  # Disable interactive prompts
            "-o", "StrictHostKeyChecking=no",  # Don't prompt for host key verification
            "-o", "ConnectTimeout=15",  # Set connection timeout
            f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
            f"bash -l -c 'cd {_remote_dirs[machine_name]} && command -v tmux &>/dev/null || module load tmux && tmux list-sessions | grep manager_session && echo Manager_session_found || echo No_manager_session'"
        ]
        try:
            result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=15)
            print(f"{machine_name}: {result.stdout.strip()}")
            if result.stderr:
                print(f"  stderr: {result.stderr.strip()}")
        except Exception as e:
            print(f"{machine_name}: Connection failed - {e}")

def cleanup_remote_managers():
    print("\nCanceling all slurm jobs and then terminating the remote queue managers...")
    for machine_name in _machine_names:
        ssh_command = [
            "ssh",
            "-o", "BatchMode=yes",  # Disable interactive prompts
            "-o", "StrictHostKeyChecking=no",  # Don't prompt for host key verification
            "-o", "ConnectTimeout=30",  # Set connection timeout
            f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
            f"bash -l -c 'cd {_remote_dirs[machine_name]} && ./run_kill_slurm_jobs.sh {machine_name}'"
        ]
        try:
            subprocess.run(ssh_command, check=True)
            print(f"Remote cleanup completed on {_remote_hosts[machine_name]} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Cleanup command failed on {_remote_hosts[machine_name]} with exit code {e.returncode}:\n{e}")
