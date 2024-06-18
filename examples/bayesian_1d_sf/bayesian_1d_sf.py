import matplotlib.pyplot as plt
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverSF
from adaptive_computing.surrogates import SMTWrapper

def func_1d(x):
    return (x-3)**2

def bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverSF(simulation=func_1d,
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 10)

    # plot the result
    plt.figure(figsize=(10, 6))
    plt.scatter(ac_driver.dataset.x_data[0], ac_driver.dataset.y_data[0], marker='o', linestyle='-', color='b')
    plt.xlabel('x_data')
    plt.ylabel('y_data')
    plt.show()

    return ac_driver

if __name__ == "__main__":
    bayesian_1d_sf()
