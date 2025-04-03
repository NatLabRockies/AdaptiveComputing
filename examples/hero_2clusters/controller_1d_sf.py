# This script initializes an AC driver that puts tasks in a Hero queue
# Workers can be launched on Kestrel and Vermillion (two HPC machines) to execute the tasks
# The c++ version of this script performs the same operations as the __main__ function of this script, but demonstrates c++ embedding.
import numpy as np

import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
#from adaptive_computing.drivers import ActiveLoopDriver
#from adaptive_computing.drivers import ActiveLoopDriverHero

#XXX temp test code:
from adaptive_computing.datasets import HeroDataset

def func_1d(x):
    return (x-3)**2

# def initialize_driver():
#     params = [ContinuousVariable(min=0.7, max=2.0)]
#     #ac_driver = ActiveLoopDriver(simulations=[func_1d],
#     ac_driver = ActiveLoopDriverMFNonBlock(simulations=[func_1d],
#                                  params=params,
#                                  surrogate='SMT')
#     ac_driver.initialize()
#     return ac_driver

def print_data(ac_driver):
    print(f"x_data = {ac_driver.dataset.x_data[0]}")
    print(f"y_data = {ac_driver.dataset.y_data[0]}")
    return

if __name__ == '__main__':
    
    #XXX temp test code:
    params = [ContinuousVariable(min=0.7, max=2.0)]
    machine_names = ['kestrel','vermillion']
    dataset = HeroDataset(params, machine_names, n_fidelity=1)
    # queue hero samples at the given x_data values. No initial guess provided.
    dataset.add_samples(np.array([[1.1]]),None,0)
    dataset.add_samples(np.array([[2.1],[3.1]]),None,0)
    # add samples with specified x_data and y_data. No hero queueing used.
    dataset.add_samples_nohero(np.array([[6.1],[7.1],[8.1]]),np.array([[6.2],[7.2],[8.2]]),0)
    # queue hero samples at the given x_data values. Initial guesses for y_data provided.
    dataset.add_samples(np.array([[4.1],[5.1]]),np.array([[4.2],[5.2]]),0)
    dataset.add_samples(np.array([[1.1]]),np.array([[1.2]]),0)
    print(f'_x_data = {dataset._x_data}')
    print(f'_y_data = {dataset._y_data}')
    print(f'_unmasked_data = {dataset._unmasked_data}')
    print(f'_hero_todo = {dataset._hero_todo}')
    dataset.hero_wait_for_data()
    print(f'_x_data = {dataset._x_data}')
    print(f'_y_data = {dataset._y_data}')
    print(f'_unmasked_data = {dataset._unmasked_data}')
    print(f'_hero_todo = {dataset._hero_todo}')

    # ac_driver = initialize_driver()
    # set the driver to have bounds limits on the output of 0 to inf on the output.
    # print_data(ac_driver)
    # x_queries = [[0.7],[0.9],[1.1],[1.5],[2.0]]
    # print(f"x_queries = {x_queries}")
    # y_queries = ac_driver.query(x_queries, 'absolute_variance', 0.0000001)
    # print(f"y_queries = {y_queries}")
    # print_data(ac_driver)
    # y_queries = ac_driver.query(x_queries, 'absolute_variance', 0.0000001)
    # print(f"y_queries = {y_queries}")
    # print_data(ac_driver)
    # # expect that the second time, no simulations are launched and the outputs are the same

