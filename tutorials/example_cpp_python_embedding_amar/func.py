
import sys
sys.path.insert(0, '../../') # add the path to the AdaptiveComputing directory
from ac_common import *    
import numpy as np
import matplotlib.pyplot as plt


# define the microscale (Kinetic Monte Carlo) simulation
# this function takes Temperature, Pressure, and compositions as input
# returns the flux on the surface
def func_4d(x):
    return (x[0]-3.5)*((x[0]-3.5)) + x[1] + x[2] + x[3]
"""```

Define the design parameters (inputs to the objective function)

```python"""
def init_dataset():
    

    T = Param() # Temperature
    T.type = 'continuous'
    T.min_val = 0
    T.max_val = 300
    P = Param() # Pressure
    P.type = 'continuous'
    P.min_val = 0
    P.max_val = 100
    x0= Param() # composition of species 0
    x0.type = 'continuous'
    x0.min_val = 0
    x0.max_val = 1
    x1= Param() # composition of species 1
    x1.type = 'continuous'
    x1.min_val = 0
    x1.max_val = 1

    params = [T, P, x0, x1]

    # Define the options for surrogate modeling and optimization
    ds_ops = DataSetOptions()
    my_dataset = DataSet(func_4d, params, ds_ops)
    my_dataset.add_lhs_samples(10) # >= the number of input arguments of func_4d + 1 (=5)
    return my_dataset

def init_surrogate(my_dataset):
    # use the SMT implementation of the Gaussian Process model
    from ac_common.surrogate_wrappers import SMTWrapper
    surrogate= SMTWrapper(my_dataset)

def if_query(my_dataset, surrogate, x_queries, threshold_std_mean):
    # Query with a std/mean threshold. Conducts simulations if the standard deviation is too high.
    import numpy as np
    y_queries = my_dataset.query_cpp(surrogate,x_queries,threshold_std_mean=threshold_std_mean)        
    return y_queries
    #return expected_values, computed_values, tolerances
    

if __name__ == '__main__':
    print("Calling func.py")
    #init_dataset()
    #init_surrogate()

