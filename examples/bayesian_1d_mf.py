from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverMF
import numpy as np

def func_lf(x):
    return (x-3)**2


def func_hf(x):
    return (x-3)**2+0.1*np.sin(x)

def example_bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverMF(simulations=[func_lf,
                                                func_hf],
                                   fidelity_costs=[1,10],
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 10)



if __name__ == "__main__":
    example_bayesian_1d_sf()