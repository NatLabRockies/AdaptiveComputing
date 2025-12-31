
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriver

def func_ThermalConductivity(x):
    return  16 + 0.01 * (x-300)

#def func_SpecificHeatCapacity(x):
#    return  500 + 0.1 * (x-300)

#def func_Density(x):
#    return  7910 - 0.4 * (x-300);


def initialize_driver():
    params = [ContinuousVariable(min=300, max=800)]
    ac_driver = ActiveLoopDriver(simulations=[func_ThermalConductivity],
                                 params=params,
                                 surrogate='SMT')
    ac_driver.initialize()
    return ac_driver

def print_data(ac_driver):
    print(f"x_data = {ac_driver.dataset.x_data[0]}")
    print(f"y_data = {ac_driver.dataset.y_data[0]}")
    return

if __name__ == '__main__':
    ac_driver = initialize_driver()
    print_data(ac_driver)
    x_queries = [[320],[350],[400]]
    print(f"x_queries = {x_queries}")
    y_queries = ac_driver.query(x_queries, 'absolute_variance', 0.1)
    print(f"y_queries = {y_queries}")
    print_data(ac_driver)
    y_queries = ac_driver.query(x_queries, 'absolute_variance', 0.1)
    print(f"y_queries = {y_queries}")
    print_data(ac_driver)
    # expect that the second time, no simulations are launched and the outputs are the same

