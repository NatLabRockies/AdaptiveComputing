import matplotlib.pyplot as plt
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverMF
import numpy as np

def func_lf(x):
    return (x-3)**2


def func_hf(x):
    return (x-3)**2+0.1*np.sin(x)

def bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverMF(simulations=[func_lf,
                                                func_hf],
                                   fidelity_costs=[1,10],
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 10)

    return ac_driver

if __name__ == "__main__":
    bayesian_1d_sf()
