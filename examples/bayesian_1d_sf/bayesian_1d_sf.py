# add the path to the adaptive_computing module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverSF
from adaptive_computing.surrogates import SMTWrapper
import matplotlib.pyplot as plt
import numpy as np

def func_1d(x):
    return (x-3)**2

def bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverSF(simulation=func_1d,
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 10)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(ac_driver.dataset.x_data[0], ac_driver.dataset.y_data[0], marker='o', linestyle='-', color='b')
    # plt.title('Plot of y_data vs x_data')
    # plt.xlabel('x_data')
    # plt.ylabel('y_data')
    # plt.grid(True)
    # plt.show()

    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    expected_values = [3.0, 0.0]
    print(f'The minimum should be approximately [x,y] = {expected_values}')
    print('The minimum found is [', x_opt[0], ',', y_opt,']')
    computed_values = [x_opt[0], y_opt]
    tolerances = [0.1, 0.1]
    return expected_values, computed_values, tolerances


if __name__ == "__main__":
    bayesian_1d_sf()
