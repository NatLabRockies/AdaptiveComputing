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

# Job scheduler type for each machine: 'slurm' or 'pbs'.
# Defaults to 'slurm' if a machine is not listed here.
scheduler = {
    'machine_a': 'slurm',
    'machine_b': 'slurm',
}

# Batch scripts to run for each machine, indexed by fidelity level.
# Each entry is a list where index 0 = lowest fidelity, index 1 = next, etc.
# script_generic.sh works on any SLURM system with mamba/conda and the AC
# environment — edit its --partition and --account headers before use.
# script_generic_pbs.sh is the equivalent for PBS systems (e.g. Aurora).
# To add a higher-fidelity simulation, append a second script to each list.
batch_scripts = {
    'machine_a': ['simulation_files/script_generic.sh'],  # [fidelity_0, fidelity_1, ...]
    'machine_b': ['simulation_files/script_generic.sh'],
}

# Command used to activate the Python environment on each remote machine.
# Common examples:
#   'module load mamba && mamba activate AC'     (HPC clusters)
#   'conda activate myenv'                       (if conda is already in PATH)
#   'source ~/miniconda3/bin/activate myenv'     (explicit conda path)
env_activate_cmds = {
    'machine_a': 'module load mamba && mamba activate AC',
    'machine_b': 'module load mamba && mamba activate AC',
}

# Example configuration for NLR systems (commented out):
# machine_names = ['kestrel', 'vermilion']
# remote_usernames = {'kestrel': 'your_nlr_username', 'vermilion': 'your_nlr_username'}
# remote_hosts = {'kestrel': 'kl1.hpc.nlr.gov', 'vermilion': 'vs-login-1.hpc.nlr.gov'}
# remote_dirs = {
#     'kestrel': '/home/your_nlr_username/AdaptiveComputing/examples/hero_HPC_managers/',
#     'vermilion': '/home/your_nlr_username/AdaptiveComputing/examples/hero_HPC_managers/'
# }
# scheduler = {
#     'kestrel': 'slurm',
#     'vermilion': 'slurm',
# }
# batch_scripts = {
#     'kestrel': ['simulation_files/script_kestrel.sh'],    # NLR Kestrel: LAMMPS molecular dynamics
#     'vermilion': ['simulation_files/script_vermilion.sh'] # NLR Vermilion: LAMMPS molecular dynamics
# }
# env_activate_cmds = {
#     'kestrel': 'module load mamba && mamba activate AC',
#     'vermilion': 'module load mamba && mamba activate AC',
# }

# Example configuration for Aurora (PBS scheduler):
# machine_names = ['aurora']
# remote_usernames = {'aurora': 'your_alcf_username'}
# remote_hosts = {'aurora': 'aurora-uan-0011.alcf.anl.gov'}
# remote_dirs = {'aurora': '/home/your_alcf_username/AdaptiveComputing/examples/hero_HPC_managers/'}
# scheduler = {'aurora': 'pbs'}
# batch_scripts = {'aurora': ['simulation_files/script_generic_pbs.sh']}
# env_activate_cmds = {'aurora': 'module load mamba && mamba activate AC'}
