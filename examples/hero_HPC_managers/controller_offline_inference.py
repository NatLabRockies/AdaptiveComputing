# This script reads a trained surrogate from a file and uses it in an offline manner (no retraining) for inference
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def initialize_driver():
    import pickle
    # Unpickle the offline trained surrogate (created by controller_offline_training.py)                                                                               
    with open('offline_training.pkl', 'rb') as file:
        ac_driver = pickle.load(file)
    return ac_driver

def print_data(ac_driver):
    print(f"x_data = {ac_driver.dataset.x_data[0]}")
    print(f"y_data = {ac_driver.dataset.y_data[0]}")
    return

def predict_values(ac_driver, x_queries):
    return ac_driver.surrogate.predict_values(x_queries)

if __name__ == '__main__':
    ac_driver = initialize_driver()
    print_data(ac_driver)
    # The model is queried at prescribed x locations in this simple example script
    x_queries = [[0.85],[0.9],[1.1],[1.5],[2.0]]
    print(f"x_queries = {x_queries}")
    y_queries = predict_values(ac_driver, x_queries)
    print(f"y_queries = {y_queries}")

    

