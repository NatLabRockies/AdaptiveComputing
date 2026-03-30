# This script contains some helping functions that are called by query.cpp
# query.cpp essentially produces the same output as the __main__ function of this script

import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriver

def func_1d(x):
    return (x-3)**2

def initialize_driver():
    params = [ContinuousVariable(min=0, max=10)]
    ac_driver = ActiveLoopDriver(simulations=[func_1d],
                                 params=params,
                                 surrogate='SMT_GP')
    ac_driver.initialize()
    return ac_driver

def print_data(ac_driver):
    print(f"x_data = {ac_driver.dataset.x_data[0]}")
    print(f"y_data = {ac_driver.dataset.y_data[0]}")
    return

if __name__ == '__main__':
    ac_driver = initialize_driver()
    print_data(ac_driver)
    x_queries = [[1],[3],[6],[2.8],[5]]
    print(f"x_queries = {x_queries}")
    y_queries = ac_driver.query(x_queries, 'absolute_variance', 0.1)
    print(f"y_queries = {y_queries}")
    print_data(ac_driver)
    y_queries = ac_driver.query(x_queries, 'absolute_variance', 0.1)
    print(f"y_queries = {y_queries}")
    print_data(ac_driver)
    # expect that the second time, no simulations are launched and the outputs are the same

