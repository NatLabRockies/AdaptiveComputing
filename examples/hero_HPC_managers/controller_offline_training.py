# This script initializes an AC driver that puts tasks in a Hero queue
# Managers can be launched on multiple HPC machines to execute the tasks
import numpy as np
import pickle
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverHero

# Notes:
# How to manually attach to the session: tmux attach-session -t manager_session
# How to manually terminate the session: tmux kill-session -t manager_session

if __name__ == '__main__':
    # Import HPC configuration from separate file
    # Note: Copy hpc_config_template.py to hpc_config.py and edit with your values
    try:
        from hpc_config import machine_names, remote_usernames, remote_hosts, remote_dirs, env_activate_cmds
        print("Using HPC configuration from hpc_config.py")
    except ImportError:
        print("ERROR: hpc_config.py not found!")
        print("Please copy hpc_config_template.py to hpc_config.py and edit with your HPC details.")
        sys.exit(1)

    from autonomous_managers import run_remote_managers, cleanup_remote_managers, setup_remote_state, verify_remote_managers
    # register a signal handler and set up the variables it needs to operate
    setup_remote_state(machine_names, remote_usernames, remote_hosts, remote_dirs, env_activate_cmds)
    run_remote_managers()
    
    # Give managers time to start, then verify they're running
    import time
    print("Waiting 10 seconds for managers to start...")
    time.sleep(10)
    verify_remote_managers()

    params = [ContinuousVariable(min=0.8, max=2.0)]
    ac_driver = ActiveLoopDriverHero(simulations=[None], params=params, machine_names=machine_names, output_field_path='y_data', surrogate='SMT_GP', acq_func='maximum_variance', blocking=False)

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
    print(f'Before first wait:')
    print(f'_x_data = {ac_driver.dataset._x_data}')
    print(f'_y_data = {ac_driver.dataset._y_data}')
    print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')
    ac_driver.hero_wait_for_data_and_train()
    print(f'After first wait:')
    print(f'_x_data = {ac_driver.dataset._x_data}')
    print(f'_y_data = {ac_driver.dataset._y_data}')
    print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')

    # Surrogate-based sampling:
    # 1) Manually add points and use the surrogate model to determine the placeholder value for the y_data.
    print(f'About to add_points...')
    ac_driver.add_points(np.array([[1.3],[1.7]]),i_fidelity=0)
    print(f'After add_points:')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')
    # 2) Bayesian optimization. Use an acquisition function to determine which x values to add. y_data placeholder values are computed using the surrogate.
    ac_driver._bopt_initialized = True # Skip additional initialization of the surrogate since already initialized it above
    print(f'About to run(N_steps=5)...')
    ac_driver.run(N_steps=5)
    print(f'After run(N_steps=5):')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')

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

