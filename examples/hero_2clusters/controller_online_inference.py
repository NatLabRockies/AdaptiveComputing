# This script initializes an AC driver that puts tasks in a Hero queue
# Managers can be launched on Kestrel and Vermillion (two HPC machines) to execute the tasks
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

    from autonomous_managers import run_remote_managers, cleanup_remote_managers, setup_remote_state
    # register a signal handler and set up the variables it needs to operate
    setup_remote_state(machine_names, remote_usernames, remote_hosts, remote_dirs)
    run_remote_managers()
    
    # Unpickle the offline trained surrogate (created by controller_offline_training.py)                                                                               
    with open('offline_training.pkl', 'rb') as file:
        ac_driver = pickle.load(file)

    assert(ac_driver.dataset.machine_names == machine_names)

    ac_driver.dataset.hero_authenticate()

    # clear all tasks in the Hero queue since we saved the pickle with an empty Hero queue
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
