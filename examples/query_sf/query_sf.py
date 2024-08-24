import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriver

def func_1d(x):
    return (x-3)**2

def bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriver(simulations=[func_1d],
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.initialize()

    y = ac_driver.query([[1],[3],[6],[2.8],[5]], 
                        error_criterion='absolute_variance',max_var=0.1)
    

    y = ac_driver.query([[1.1],[3.5],[100]], 
                        error_criterion='percent_variance',max_percent_var=10)
    return ac_driver

if __name__ == "__main__":
    bayesian_1d_sf()
