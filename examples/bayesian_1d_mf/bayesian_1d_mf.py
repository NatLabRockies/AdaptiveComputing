import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverCostRatio
import numpy as np

def func_lf(x):
    return (x-3)**2


def func_hf(x):
    return (x-3)**2+0.1*np.sin(x)

def bayesian_1d_mf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverCostRatio(simulations=[func_lf,
                                                func_hf],
                                   fidelity_costs=[1,10],
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 10)

    # plot the result
    plt.figure(figsize=(10, 6))
    plt.scatter(ac_driver.dataset.x_data[0], ac_driver.dataset.y_data[0], marker='o', color='b')
    plt.scatter(ac_driver.dataset.x_data[1], ac_driver.dataset.y_data[1], marker='s', color='r')
    plt.xlabel('x_data')
    plt.ylabel('y_data')
    plt.show()
    
    return ac_driver

if __name__ == "__main__":
    bayesian_1d_mf()
