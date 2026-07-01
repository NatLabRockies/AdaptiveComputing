# HPC Manager Configuration Template
# Copy this file to hpc_config.py and edit with your actual values

# Machine names to use for HPC job submission
machine_names = ['machine_a', 'machine_b']

# SSH connection details for each machine
remote_usernames = {
    'machine_a': 'your_username',
    'machine_b': 'your_username'
}

# Domain names for ssh. Use the specific hostname of an individual login node so that tmux can run a lightweight processes on the login node and check on that same process later
remote_hosts = {
    'machine_a': 'machine-a.hpc.example.gov',
    'machine_b': 'machine-b.hpc.example.gov'
}

# Remote directory paths to the hero_HPC_managers directory on the remote machines
remote_dirs = {
    'machine_a': '/path/to/AdaptiveComputing/examples/hero_HPC_managers/',
    'machine_b': '/path/to/AdaptiveComputing/examples/hero_HPC_managers/'
}

# SLURM batch scripts to run for each machine, indexed by fidelity level.
# Each entry is a list where index 0 = lowest fidelity, index 1 = next, etc.
# script_generic.sh works on any SLURM system with mamba/conda and the AC
# environment — edit its --partition and --account headers before use.
# To add a higher-fidelity simulation, append a second script to each list.
slurm_scripts = {
    'machine_a': ['simulation_files/script_generic.sh'],  # [fidelity_0, fidelity_1, ...]
    'machine_b': ['simulation_files/script_generic.sh'],
}

# Example configuration for NLR systems (commented out):
# machine_names = ['kestrel', 'vermilion']
# remote_usernames = {'kestrel': 'your_nlr_username', 'vermilion': 'your_nlr_username'}
# remote_hosts = {'kestrel': 'kl1.hpc.nlr.gov', 'vermilion': 'vs-login-1.hpc.nlr.gov'}
# remote_dirs = {
#     'kestrel': '/home/your_nlr_username/AdaptiveComputing/examples/hero_HPC_managers/',
#     'vermilion': '/home/your_nlr_username/AdaptiveComputing/examples/hero_HPC_managers/'
# }
# slurm_scripts = {
#     'kestrel': ['simulation_files/script_kestrel.sh'],    # NLR Kestrel: LAMMPS molecular dynamics
#     'vermilion': ['simulation_files/script_vermilion.sh'] # NLR Vermilion: LAMMPS molecular dynamics
# }
