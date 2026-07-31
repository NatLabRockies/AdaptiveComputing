# Loads a pre-trained surrogate from offline_training.pkl and refines it with
# live scheduler jobs when prediction variance exceeds a threshold.
#
# Run this after controller_offline_training.py has produced offline_training.pkl.
#
# Usage (on any node with scheduler access):
#   python controller_online_inference.py [--scheduler {slurm,pbs}]

import argparse
import numpy as np
import pickle
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.local_hero import LocalHeroClient
from manager import create_manager


def print_data(ac_driver):
    print(f"x_data = {ac_driver.dataset.x_data[0]}")
    print(f"y_data = {ac_driver.dataset.y_data[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler', default='slurm', choices=['slurm', 'pbs'],
                        help='Job scheduler used on this HPC system (default: slurm)')
    args = parser.parse_args()

    # Point to the same JSON file used during training so the queue state
    # is consistent across both runs.
    local_hero = LocalHeroClient(
        db_path=os.path.join(os.path.dirname(__file__), 'local_hero_db.json'),
    )

    manager = create_manager(scheduler_type=args.scheduler, hero_client=local_hero)

    with open('offline_training.pkl', 'rb') as f:
        ac_driver = pickle.load(f)

    # Reconnect the dataset to the local JSON backend after loading from pickle,
    # and wire up the inline manager so ac_driver.query() calls run_until_done().
    ac_driver.dataset.hero_authenticate(
        machine_names=[manager.machine_name],
        hero_client=local_hero,
    )
    ac_driver.inline_manager = manager

    # Clear any stale tasks left over from a previous run.
    ac_driver.dataset.clear_hero_queue()

    # Surrogate is trusted when variance is below this threshold.
    # Above it, a real Slurm job is submitted for that query point.
    variance_threshold = 1e-4

    print_data(ac_driver)
    x_queries = [[0.85], [0.9], [1.1], [1.5], [2.0]]
    print(f"x_queries = {x_queries}")

    # First query: all high-variance points are submitted as a parallel batch,
    # the surrogate is retrained once when all results are in.
    y_queries = ac_driver.query(x_queries, 'absolute_variance', variance_threshold)

    print(f'After first query:')
    print(f'_x_data        = {ac_driver.dataset._x_data}')
    print(f'_y_data        = {ac_driver.dataset._y_data}')
    print(f'_unmasked_data = {ac_driver.dataset._unmasked_data}')
    print(f'_hero_todo     = {ac_driver.dataset._hero_todo}')
    print(f'y_queries      = {y_queries}')
    print_data(ac_driver)

    # Second query: all points should now be below threshold.
    y_queries = ac_driver.query(x_queries, 'absolute_variance', variance_threshold)
    print(f"y_queries (second call, expect surrogate-only) = {y_queries}")
    print_data(ac_driver)

    with open('online_training.pkl', 'wb') as f:
        pickle.dump(ac_driver, f)
    print('Saved online_training.pkl')
