# This script initializes an AC driver that puts tasks in a Hero queue
# Managers can be launched on Kestrel and Vermillion (two HPC machines) to execute the tasks
import numpy as np
import pickle
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import subprocess

# global variables used by the signal handler
# for testing, use two managers on vermilion on different login nodes
machine_names = ['vermilion1','vermilion2']
remote_usernames = {'vermilion1':'kgriffin','vermilion2':'kgriffin'}
remote_hosts = {'vermilion1':'vs-login-1.hpc.nrel.gov','vermilion2':'vs-login-2.hpc.nrel.gov'} # Note: make sure to specify a specific login node, otherwise it is unlikely you can reattach to a previously started tmux session when cleaning up
remote_dirs = {'vermilion1':'/projects/degrees/kgriffin/AdaptiveComputing/examples/hero_2clusters/','vermilion2':'/projects/degrees/kgriffin/AdaptiveComputing/examples/hero_2clusters/'}

# # machine_names = ['kestrel','vermilion']
# machine_names = ['kestrel']
# # machine_names = ['vermilion']
# remote_usernames = {'kestrel':'kgriffin','vermilion':'kgriffin'}
# remote_hosts = {'kestrel':'kl1.hpc.nrel.gov','vermilion':'vs-login-1.hpc.nrel.gov'} # Note: make sure to specify a specific login node, otherwise it is unlikely you can reattach to a previously started tmux session when cleaning up
# remote_dirs = {'kestrel':'/home/kgriffin/AdaptiveComputing_1.0/AdaptiveComputing/examples/hero_2clusters/','vermilion':'/projects/degrees/kgriffin/AdaptiveComputing/examples/hero_2clusters/'}

def run_remote_managers():
    for machine_name in machine_names:
        ssh_command = [
            "ssh",
            f"{remote_usernames[machine_name]}@{remote_hosts[machine_name]}",
            f"{remote_dirs[machine_name]}run_manager.sh {machine_name}"
        ]
        subprocess.run(ssh_command)

def cleanup_remote_managers():
    print(f"\nCanceling all slurm jobs and then terminating the remote queue managers...")
    for machine_name in machine_names:
        ssh_command = [
            "ssh",
            f"{remote_usernames[machine_name]}@{remote_hosts[machine_name]}",
            f"{remote_dirs[machine_name]}run_kill_slurm_jobs.sh {machine_name}"
        ]
        try:
            subprocess.run(ssh_command, check=True)
            print(f"Remote cleanup completed on {remote_hosts[machine_name]} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Cleanup command failed on {remote_hosts[machine_name]} with exit code {e.returncode}:\n{e}")

def signal_handler(sig, frame):
    print("\nReceived signal {sig}. Canceling all slurm jobs and then terminating the remote queue managers...")
    cleanup_remote_managers()
    sys.exit(0)

# Register the signal handler
import signal
signal.signal(signal.SIGINT, signal_handler)  # For Ctrl-C

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverHero

# Notes:
# How to manually attach to the session: tmux attach-session -t manager_session
# How to manually terminate the session: tmux kill-session -t manager_session

if __name__ == '__main__':
    params = [ContinuousVariable(min=0.8, max=2.0)]
    ac_driver = ActiveLoopDriverHero(simulations=[None], params=params, machine_names=machine_names, surrogate='SMT', acq_func='maximum_variance', blocking=False)
    
    run_remote_managers()

    # Sampling techniques that don't use the surrogate model:
    # 1) queue hero samples at the given x_data values. No initial guess provided.
    # ac_driver.dataset.add_samples(np.array([[0.8],[2.0]]),None,0)
    # 2) add samples with specified x_data and y_data. No hero queueing used.
    ac_driver.dataset.add_samples_nohero(np.array([[0.938],[1.443],[1.641]]),np.array([[2.03],[3.51],[3.81]]),0)
    # 3) queue hero samples at the given x_data values. Initial guesses for y_data provided.
    # ac_driver.dataset.add_samples(np.array([[1.1],[1.2]]),np.array([[2.1],[2.2]]),0)
    # 4) queue hero samples using latin hypercube random sampling for x_data values. No initial guesses for y_data provided.
    # ac_driver.initialize(N_samples_init=3)
    
    # Print the data before and after a hero wait
    # print(f'_x_data = {ac_driver.dataset._x_data}')
    # print(f'_y_data = {ac_driver.dataset._y_data}')
    # print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    # print(f'_hero_todo = {ac_driver.dataset._hero_todo}')
    ac_driver.hero_wait_for_data_and_train()
    # print(f'_x_data = {ac_driver.dataset._x_data}')
    # print(f'_y_data = {ac_driver.dataset._y_data}')
    # print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    # print(f'_hero_todo = {ac_driver.dataset._hero_todo}')

    # Surrogate-based sampling:
    # 1) Manually add points and use the surrogate model to determine the placeholder value for the y_data.
    ac_driver.add_points(np.array([[1.3],[1.7]]),i_fidelity=0)
    # 2) Bayesian optimization. Use an acquisition function to determine which x values to add. y_data placeholder values are computed using the surrogate.
    ac_driver._bopt_initialized = True # Skip additional initialization of the surrogate since already initialized it above
    ac_driver.run(N_steps=5)

    # Print the data before and after a hero wait
    # print(f'_x_data = {ac_driver.dataset._x_data}')
    # print(f'_y_data = {ac_driver.dataset._y_data}')
    # print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    # print(f'_hero_todo = {ac_driver.dataset._hero_todo}')
    ac_driver.hero_wait_for_data_and_train()
    # print(f'_x_data = {ac_driver.dataset._x_data}')
    # print(f'_y_data = {ac_driver.dataset._y_data}')
    # print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    # print(f'_hero_todo = {ac_driver.dataset._hero_todo}')

    cleanup_remote_managers()

    # Save the driver to a pickle file                                                                     
    with open('offline_training.pkl', 'wb') as file:
        pickle.dump(ac_driver, file)

