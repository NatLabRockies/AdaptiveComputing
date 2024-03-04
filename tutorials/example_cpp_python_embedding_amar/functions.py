"""
<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (Efficient Global Optimization EGO method) for optimal parameter selection when some of the objective function evaluations have been precomputed and stored in a .csv file.
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - Create DataSet. In continuum simulation, the dataset the flow field. Temperature, Pressure, Composition (mass fraction for the species). </ol>
<ol> - Create Surrogate model. Surrogate model keeps track of which microscale simulations have been performed. Allows us to interpolate between those simulations. </ol>
<ol> - . </ol>
<ol> - . </ol>
</div>

```python
"""

"""```


Define the objective function


```python"""
# define the microscale (Kinetic Monte Carlo) simulation
# this function takes Temperature, Pressure, and compositions as input
# returns the flux on the surface
def func_4d(x):
    return (x[0]-3.5)*np.sin((x[0]-3.5)/(np.pi)) + x[1] + x[2] + x[3]
"""```

Define the design parameters (inputs to the objective function)

```python"""
def init_dataset():
    import sys
    sys.path.insert(0, '../../') # add the path to the AdaptiveComputing directory
    import numpy as np
    from ac_common import *
    if utils.is_notebook():
        get_ipython().run_line_magic('matplotlib', 'notebook')
    import matplotlib.pyplot as plt

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

def init_surrogate():
    # use the SMT implementation of the Gaussian Process model
    from ac_common.surrogate_wrappers import SMTWrapper
    surrogate= SMTWrapper(my_dataset)

def query_surrogate():
    # Query with a std/mean threshold. Conducts simulations if the standard deviation is too high.
    x_queries = np.array([[13],[13.5]])
    threshold_std_mean = 1.0
    y_queries, y_queries_var = my_dataset.query(surrogate,x_queries,threshold_std_mean=threshold_std_mean)
    print(y_queries)
    print(np.sqrt(y_queries_var))
    return expected_values, computed_values, tolerances

if __name__ == '__main__':
    init_dataset()
    init_surrogate()
"""```"""
