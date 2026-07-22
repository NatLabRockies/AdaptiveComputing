# This script initializes an AC driver that puts tasks in a Hero queue
# Managers can be launched on multiple HPC machines to execute the tasks
import numpy as np
import pickle
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def print_data(ac_driver):
    print(f"x_data = {ac_driver.dataset.x_data[0]}")
    print(f"y_data = {ac_driver.dataset.y_data[0]}")
    return

# Notes:
# How to manually attach to the session: tmux attach-session -t manager_session
# How to manually terminate the session: tmux kill-session -t manager_session

if __name__ == '__main__':
    # Import HPC configuration from separate file
    # Note: Copy hpc_config_template.py to hpc_config.py and edit with your values
    try:
        from hpc_config import machine_names, remote_usernames, remote_hosts, remote_dirs, python_paths
        print("Using HPC configuration from hpc_config.py")
    except ImportError:
        print("ERROR: hpc_config.py not found!")
        print("Please copy hpc_config_template.py to hpc_config.py and edit with your HPC details.")
        sys.exit(1)

    from adaptive_computing.hpc import run_remote_managers, cleanup_remote_managers, setup_remote_state, wait_for_managers
    # register a signal handler and set up the variables it needs to operate
    setup_remote_state(machine_names, remote_usernames, remote_hosts, remote_dirs, python_paths)
    run_remote_managers()
    wait_for_managers()
    
    # Unpickle the offline trained surrogate (created by controller_offline_training.py)                                                                               
    with open('offline_training.pkl', 'rb') as file:
        ac_driver = pickle.load(file)

    ac_driver.dataset.hero_authenticate(machine_names=machine_names)

    # Clear any stale tasks left over from a previous run.
    # WARNING: this deletes ALL tasks on the shared Hero queue — do not call
    # this when other experiments (e.g. parallel co-scientist chats) have
    # outstanding tasks on the same queue.
    ac_driver.dataset.clear_hero_queue()

    # threshold for trusting surrogate or running a simulation
    variance_threshold = 1e-4

    print_data(ac_driver)
    x_queries = [[0.85],[0.9],[1.1],[1.5],[2.0]]
    print(f"x_queries = {x_queries}")
    # y_var1 = ac_driver.surrogate.predict_variances(x_queries)
    # print(f"y_variances = {y_var1}")
    y_queries = ac_driver.query(x_queries, 'absolute_variance', variance_threshold)
    
    print(f'_x_data = {ac_driver.dataset._x_data}')
    print(f'_y_data = {ac_driver.dataset._y_data}')
    print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')

    # Optionally, use this statement to update the surrogate with whichever hero tasks have completed
    # ac_driver.hero_update_avial_data_and_train()
    # Optionally, use this statement to wait for all outstanding hero tasks to complete and then update the surrogate
    ac_driver.hero_wait_for_data_and_train()

    print(f'_x_data = {ac_driver.dataset._x_data}')
    print(f'_y_data = {ac_driver.dataset._y_data}')
    print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    print(f'_hero_todo = {ac_driver.dataset._hero_todo}')


    print(f"y_queries = {y_queries}")
    print_data(ac_driver)
    # y_var2 = ac_driver.surrogate.predict_variances(x_queries)
    # print(f"y_variances = {y_var2}")
    y_queries = ac_driver.query(x_queries, 'absolute_variance', variance_threshold)
    print(f"y_queries = {y_queries}")
    print_data(ac_driver)
    # expect that the second time, no simulations are launched and the outputs are the same

    ac_driver.hero_wait_for_data_and_train()

    cleanup_remote_managers()

    # Save the driver to a pickle file                                                                     
    with open('online_training.pkl', 'wb') as file:
        pickle.dump(ac_driver, file)
