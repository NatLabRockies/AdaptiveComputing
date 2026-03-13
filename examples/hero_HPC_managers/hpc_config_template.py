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

# Remote directory paths where AdaptiveComputing is installed
remote_dirs = {
    'machine_a': '/path/to/AdaptiveComputing/examples/hero_2clusters/',
    'machine_b': '/path/to/AdaptiveComputing/examples/hero_2clusters/'
}

# SLURM batch scripts to run for each machine
# These scripts must exist in the remote_dirs[machine]/simulation_files/ directory
slurm_scripts = {
    'machine_a': 'script_machine_a.sh',
    'machine_b': 'script_machine_b.sh'
}

# Example configuration for NREL systems (commented out):
# machine_names = ['kestrel', 'vermilion']
# remote_usernames = {'kestrel': 'your_nrel_username', 'vermilion': 'your_nrel_username'}
# remote_hosts = {'kestrel': 'kl1.hpc.nrel.gov', 'vermilion': 'vs-login-1.hpc.nrel.gov'}
# remote_dirs = {
#     'kestrel': '/home/your_nrel_username/AdaptiveComputing/examples/hero_2clusters/',
#     'vermilion': '/projects/degrees/your_nrel_username/AdaptiveComputing/examples/hero_2clusters/'
# }
# slurm_scripts = {'kestrel': 'script_kestrel.sh', 'vermilion': 'script_vermilion.sh'}
