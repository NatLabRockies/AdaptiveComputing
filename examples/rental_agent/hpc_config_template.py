# HPC configuration for the rental_agent mock simulation on Kestrel.
# Edit remote_usernames and remote_hosts to match your login node.

machine_names = ['kestrel']

remote_usernames = {
    'kestrel': 'kgriffin',
}

# Use a specific login node so the manager's tmux session persists.
remote_hosts = {
    'kestrel': 'kl1.hpc.nlr.gov',
}

# Absolute path to the agent directory on Kestrel.
remote_dirs = {
    'kestrel': '/home/kgriffin/AdaptiveComputing/examples/rental_agent/',
}

# SLURM batch script (must live in remote_dirs[machine]/simulation_files/).
batch_scripts = {
    'kestrel': ['job.sh'],
}

# Full path to the Python executable in the AC conda environment on Kestrel.
python_paths = {
    'kestrel': '/home/kgriffin/.conda-envs/AC/bin/python',
}

# Set True for fast test runs (simulation still runs, but no real wait).
debug_run = False
