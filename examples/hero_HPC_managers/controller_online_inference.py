# This script loads a pre-trained surrogate and refines it with live HPC jobs
# when prediction variance exceeds a threshold.
import numpy as np
import pickle
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def print_data(ac_driver):
    print(f"x_data = {ac_driver.dataset.x_data[0]}")
    print(f"y_data = {ac_driver.dataset.y_data[0]}")


# Notes:
# How to manually attach to the session: tmux attach-session -t manager_session
# How to manually terminate the session: tmux kill-session -t manager_session

if __name__ == '__main__':
    try:
        from hpc_config import machine_names, remote_usernames, remote_hosts, remote_dirs, python_paths
        print("Using HPC configuration from hpc_config.py")
    except ImportError:
        print("ERROR: hpc_config.py not found!")
        print("Please copy hpc_config_template.py to hpc_config.py and edit with your HPC details.")
        sys.exit(1)

    from adaptive_computing.hpc import run_remote_managers, cleanup_remote_managers, setup_remote_state, wait_for_managers
    setup_remote_state(machine_names, remote_usernames, remote_hosts, remote_dirs, python_paths)
    run_remote_managers()
    wait_for_managers()

    with open('offline_training.pkl', 'rb') as f:
        ac_driver = pickle.load(f)

    ac_driver.dataset.hero_authenticate(machine_names=machine_names)

    # Clear any stale tasks left over from a previous run.
    # WARNING: this deletes ALL tasks on the shared Hero queue — do not call
    # this when other experiments (e.g. parallel co-scientist chats) have
    # outstanding tasks on the same queue.
    ac_driver.dataset.clear_hero_queue()

    # Surrogate is trusted when variance is below this threshold.
    # Above it, a Hero task is queued for the background manager to run on HPC.
    variance_threshold = 1e-4

    print_data(ac_driver)
    x_queries = [[0.85], [0.9], [1.1], [1.5], [2.0]]
    print(f"x_queries = {x_queries}")

    # First query: all high-variance points are queued as Hero tasks in one batch;
    # the background manager submits them to Slurm (in parallel) and the surrogate
    # is retrained once when all results are in.
    y_queries = ac_driver.query(x_queries, 'absolute_variance', variance_threshold)

    print(f'_x_data        = {ac_driver.dataset._x_data}')
    print(f'_y_data        = {ac_driver.dataset._y_data}')
    print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    print(f'_hero_todo     = {ac_driver.dataset._hero_todo}')
    print(f"y_queries = {y_queries}")
    print_data(ac_driver)

    # Second query: variance should now be below threshold for all points,
    # so no new Hero tasks are queued and only the surrogate is used.
    y_queries = ac_driver.query(x_queries, 'absolute_variance', variance_threshold)
    print(f"y_queries (second call, expect surrogate-only) = {y_queries}")
    print_data(ac_driver)

    cleanup_remote_managers()

    with open('online_training.pkl', 'wb') as f:
        pickle.dump(ac_driver, f)
    print('Saved online_training.pkl')
