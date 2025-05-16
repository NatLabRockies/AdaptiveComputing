# This script reads a trained surrogate from a file and uses it in an offline manner (no retraining) for inference
import numpy as np
import pickle
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

if __name__ == '__main__':
    # Unpickle the offline trained surrogate (created by controller_offline_training.py)                                                                               
    with open('offline_training.pkl', 'rb') as file:
        ac_driver = pickle.load(file)

    # The model is queried at prescribed x locations in this simple example script
    x_queries = [[0.85],[0.9],[1.1],[1.5],[2.0]]
    print(f"x_queries = {x_queries}")
    y_queries = ac_driver.surrogate.predict_values(x_queries)
    print(f"y_queries = {y_queries}")

    

