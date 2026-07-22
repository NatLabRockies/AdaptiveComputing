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
# script_generic_slurm.sh works on any SLURM system with mamba/conda and the AC
# environment — edit its --partition and --account headers before use.
# script_generic_pbs.sh is the equivalent for PBS systems (e.g. Aurora).
# To add a higher-fidelity simulation, append a second script to each list.
batch_scripts = {
    'machine_a': ['simulation_files/script_generic_slurm.sh'],  # [fidelity_0, fidelity_1, ...]
    'machine_b': ['simulation_files/script_generic_slurm.sh'],
}

# Full path to the Python executable in the AC conda environment on each
# remote machine.  The AC environment must be installed there in advance.
# Using the direct path means no shell environment activation is needed.
python_paths = {
    'machine_a': '/home/your_username/.conda-envs/AC/bin/python',
    'machine_b': '/home/your_username/.conda-envs/AC/bin/python',
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
# python_paths = {
#     'kestrel': '/home/your_nlr_username/.conda-envs/AC/bin/python',
#     'vermilion': '/home/your_nlr_username/.conda-envs/AC/bin/python',
# }

# Example configuration for Aurora (PBS scheduler):
# machine_names = ['aurora']
# remote_usernames = {'aurora': 'your_alcf_username'}
# remote_hosts = {'aurora': 'aurora-uan-0011.alcf.anl.gov'}
# remote_dirs = {'aurora': '/home/your_alcf_username/AdaptiveComputing/examples/hero_HPC_managers/'}
# scheduler = {'aurora': 'pbs'}
# batch_scripts = {'aurora': ['simulation_files/script_generic_pbs.sh']}
# python_paths = {'aurora': '/home/your_alcf_username/.conda-envs/AC/bin/python'}
