# Loads a pre-trained surrogate from offline_training.pkl and uses it for
# purely local inference — no HPC connection, no Slurm jobs, no manager needed.
#
# Run this after controller_offline_training.py has produced offline_training.pkl.
#
# Usage:
#   python controller_offline_inference.py

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def initialize_driver():
    import pickle
    with open('offline_training.pkl', 'rb') as f:
        return pickle.load(f)


def print_data(ac_driver):
    print(f"x_data = {ac_driver.dataset.x_data[0]}")
    print(f"y_data = {ac_driver.dataset.y_data[0]}")


def predict_values(ac_driver, x_queries):
    return ac_driver.surrogate.predict_values(x_queries)


if __name__ == '__main__':
    ac_driver = initialize_driver()
    print_data(ac_driver)

    x_queries = [[0.85], [0.9], [1.1], [1.5], [2.0]]
    print(f"x_queries = {x_queries}")
    y_queries = predict_values(ac_driver, x_queries)
    print(f"y_queries = {y_queries}")
