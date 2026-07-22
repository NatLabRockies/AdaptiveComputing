# hpc_config_template.py — copy to hpc_config.py and fill in your values.
#
# This file is required by HeroHPCManager and by adaptive_computing.hpc.autonomous.
# Every field below must be present; the template shows representative defaults.
#
# IMPORTANT: Use the specific hostname of an individual login node (not a load-
# balancer alias) for remote_hosts, so that tmux sessions are reachable on
# reconnection.

# ── Machine registry ─────────────────────────────────────────────────────────

# Logical names for the HPC machines you want to use.
machine_names = ["machine_a", "machine_b"]

# SSH username on each machine.
remote_usernames = {
    "machine_a": "your_username",
    "machine_b": "your_username",
}

# SSH hostname for each machine.  Use a specific login-node hostname, not a
# load-balancer alias (e.g. kl1.hpc.example.gov, not kestrel.hpc.example.gov).
remote_hosts = {
    "machine_a": "login1.machine-a.example.gov",
    "machine_b": "login1.machine-b.example.gov",
}

# Absolute path on each machine to the directory that contains manager.py.
remote_dirs = {
    "machine_a": "/home/your_username/my_app/",
    "machine_b": "/home/your_username/my_app/",
}

# ── Scheduler ────────────────────────────────────────────────────────────────

# Scheduler type: 'slurm' (default) or 'pbs'.
scheduler = {
    "machine_a": "slurm",
    "machine_b": "slurm",
}

# Batch scripts for each machine, indexed by fidelity level.
# Index 0 = lowest fidelity (single-fidelity workflows only use index 0).
batch_scripts = {
    "machine_a": ["simulation_files/run_sim.sh"],
    "machine_b": ["simulation_files/run_sim.sh"],
}

# ── Python path ──────────────────────────────────────────────────────────────

# Full path to the Python executable in the AC conda environment on each
# remote machine.  Used by autonomous.py to invoke remote_manager.py directly
# without any shell environment activation.
python_paths = {
    "machine_a": "/home/your_username/.conda-envs/AC/bin/python",
    "machine_b": "/home/your_username/.conda-envs/AC/bin/python",
}

# ── Optional: debug / testing flags ──────────────────────────────────────────

# Set debug_run = True to route jobs to a short-wall-time debug partition
# (useful for testing without waiting in the normal queue).
debug_run = False

# Partition names used when debug_run is True.
debug_partitions = {
    "machine_a": "debug",
    "machine_b": "debug",
}

# ── NLR example (uncomment and edit) ─────────────────────────────────────────
# machine_names = ["kestrel"]
# remote_usernames = {"kestrel": "your_nlr_username"}
# remote_hosts     = {"kestrel": "kl1.hpc.nlr.gov"}
# remote_dirs      = {"kestrel": "/home/your_nlr_username/my_app/"}
# scheduler        = {"kestrel": "slurm"}
# batch_scripts    = {"kestrel": ["simulation_files/run_sim.sh"]}
# python_paths    = {"kestrel": "/home/your_nlr_username/.conda-envs/AC/bin/python"}
