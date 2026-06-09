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

# SLURM batch scripts to run for each machine
# These are the relative paths to the slurm scripts remote_dirs[machine]/slurm_scripts[machine]
slurm_scripts = {
    'machine_a': 'simulation_files/script_machine_a.sh',
    'machine_b': 'simulation_files/script_machine_b.sh'
}

# Example configuration for NLR systems (commented out):
# machine_names = ['kestrel', 'vermilion']
# remote_usernames = {'kestrel': 'your_nlr_username', 'vermilion': 'your_nlr_username'}
# remote_hosts = {'kestrel': 'kl1.hpc.nlr.gov', 'vermilion': 'vs-login-1.hpc.nlr.gov'}
# remote_dirs = {
#     'kestrel': '/home/your_nlr_username/AdaptiveComputing/examples/hero_HPC_managers/',
#     'vermilion': '/projects/degrees/your_nlr_username/AdaptiveComputing/examples/hero_HPC_managers/'
# }
# slurm_scripts = {'kestrel': 'simulation_files/script_kestrel.sh', 'vermilion': 'simulation_files/script_vermilion.sh'}
