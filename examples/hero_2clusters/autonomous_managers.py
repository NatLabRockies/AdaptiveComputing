import signal
import os
import subprocess

# Internal globals for the signal handler and remote functions
_machine_names = []
_remote_usernames = {}
_remote_hosts = {}
_remote_dirs = {}

def setup_remote_state(machine_names, remote_usernames, remote_hosts, remote_dirs):
    global _machine_names, _remote_usernames, _remote_hosts, _remote_dirs
    _machine_names = machine_names
    _remote_usernames = remote_usernames
    _remote_hosts = remote_hosts
    _remote_dirs = remote_dirs
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

def signal_handler(sig, frame):
    print("\nReceived signal {sig}. Canceling all slurm jobs and then terminating the remote queue managers...")
    cleanup_remote_managers()
    os._exit(0) # unlike sys.exit(0), os._exit(0) avoids sending SystemExit signal to Hero, which it doesn't know how to handle

def run_remote_managers():
    for machine_name in _machine_names:
        ssh_command = [
            "ssh",
            f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
            f"{_remote_dirs[machine_name]}run_manager.sh {machine_name}"
        ]
        subprocess.run(ssh_command)

def cleanup_remote_managers():
    print(f"\nCanceling all slurm jobs and then terminating the remote queue managers...")
    for machine_name in _machine_names:
        ssh_command = [
            "ssh",
            f"{_remote_usernames[machine_name]}@{_remote_hosts[machine_name]}",
            f"{_remote_dirs[machine_name]}run_kill_slurm_jobs.sh {machine_name}"
        ]
        try:
            subprocess.run(ssh_command, check=True)
            print(f"Remote cleanup completed on {_remote_hosts[machine_name]} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Cleanup command failed on {_remote_hosts[machine_name]} with exit code {e.returncode}:\n{e}")