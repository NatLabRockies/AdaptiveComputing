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
    dataset = HeroDataset(params, n_fidelity=1)
    dataset.add_samples(np.array([[1.1]]),None,0)
    dataset.add_samples(np.array([[4.1],[5.1]]),np.array([[4.2],[5.2]]),0)
    dataset.add_samples_nohero(np.array([[6.1],[7.1],[8.1]]),np.array([[6.2],[7.2],[8.2]]),0)
    dataset.add_samples(np.array([[2.1],[3.1]]),None,0)
    #dataset.add_samples(np.array([[1.1]]),np.array([[1.2]]),0) # this should not queue the sample since y value is provided
    #XXX right now this queues the sample, but I may also want to just add samples (prior calcs/offline training)
    #XXX is there another function for adding samples without the y value? could check if y_data is none and have that be an optional arg
    dataset.hero_wait_for_data()
    #XXX try adding more than one task at a time with add_samples
    #XXX If there is a surrogate, have to manually retrain it. Should add a sync and retrain to the driver (every step if nonblocking, use a wait instead of sync if blocking).

    # ac_driver = initialize_driver()
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

