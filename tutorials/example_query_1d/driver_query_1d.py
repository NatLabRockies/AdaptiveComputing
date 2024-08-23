"""
<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (Efficient Global Optimization EGO method) for optimal parameter selection when some of the objective function evaluations have been precomputed and stored in a .csv file.
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - Existing function evaluations are read from a .csv file to make use of known data and reduce the computational resources required to achieve convergence.</ol>
<ol> - Otherwise, this tutorial is the same as example_mixed_type .</ol> 
<ol> - The 3D objective function of three different data types is considered.</ol> 
<ol> - $x_0$ is a continous variable with bounds $[0,10]$.</ol> 
<ol> - $x_1$ is an integer variable, which can takes the values ${2,3,4,5,6}$.</ol> 
<ol> - $x_2$ is a categorical variable, which can take the values ${'a','b','c','d'}$.</ol> 
<ol> - The objective function is analytically defined as $f(x_0,x_1,x_2) = (x_0-5)^2+(x_1-4)^2+s(x_2)-5.0$, where $s$ is defined as $s({'a','b','c','d'}) = {10,5,7.5,6}$.</ol> 
<ol> - The global minimum of $f = 0$ occurs at $[x_0, x_1, x_2] = [5, 4, 'b']$. </ol>
</div>

```python
"""
import sys
sys.path.insert(0, '../../') # add the path to the AdaptiveComputing directory
import numpy as np
from ac_common import *
if utils.is_notebook():
    get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
"""```


Define the objective function


```python"""
# define the function
def func_1d(x):
    return (x[0]-3.5)*np.sin((x[0]-3.5)/(np.pi))
"""```

Define the design parameters (inputs to the objective function)

```python"""
def driver_query_1d():
    x0 = Param()
    x0.type = 'continuous'
    x0.min_val = 0
    x0.max_val = 25

    params = [x0]

    # Define the options for surrogate modeling and optimization
    ds_ops = DataSetOptions()
    my_dataset = DataSet(func_1d, params, ds_ops)
    my_dataset.add_lhs_samples(2)
    viz_ops = VizOptions()
    viz_ops.plot_1d=True
    viz_ops.show_exact = True
    # use the SMT implementation of the Gaussian Process model
    from ac_common.surrogates import SMTWrapper
    surrogate= SMTWrapper(my_dataset)
    my_dataset.add_bo_samples(2,surrogate,viz_ops=viz_ops)

    # Query with a std/mean threshold. Conducts simulations if the standard deviation is too high.
    x_queries = np.array([[13],[13.5]])
    threshold_std_mean = 1.0
    y_queries, y_queries_var = my_dataset.query(surrogate,x_queries,threshold_std_mean=threshold_std_mean)
    print(y_queries)
    print(np.sqrt(y_queries_var))

    # Visualize the final result
    my_dataset.add_bo_samples(0,surrogate,viz_ops=viz_ops)

    # Query with a std/total_variation threshold. Conducts simulations if the standard deviation is too high.
    x_queries = np.array([[14],[14.5]])
    threshold_std_tv = 1.0
    y_queries, y_queries_var = my_dataset.query(surrogate,x_queries,threshold_std_tv=threshold_std_tv)
    print(y_queries)
    print(np.sqrt(y_queries_var))

    # Visualize the final result
    my_dataset.add_bo_samples(0,surrogate,viz_ops=viz_ops)
    
    # Query without a threshold. This just probes the surrogate at 3 locations.
    x_queries = np.array([[12],[18],[12.5]])
    y_queries, y_queries_var = my_dataset.query(surrogate,x_queries)
    print(y_queries)
    print(np.sqrt(y_queries_var))

    # Query with a std threshold. Conducts simulations if the standard deviation is too high.
    x_queries = np.array([[12],[18],[12.5]])
    threshold_std = 1.0
    y_queries, y_queries_var = my_dataset.query(surrogate,x_queries,threshold_std=threshold_std)
    print(y_queries)
    print(np.sqrt(y_queries_var))

    # Visualize the final result
    my_dataset.add_bo_samples(0,surrogate,viz_ops=viz_ops)

    # Verify that the standard deviation at all queried points is less than the threshold
    # observed_std < threshold_std
    computed_values = np.clip(y_queries_var[:,0]-threshold_std, a_min = 0.0, a_max = None)
    expected_values = [0.0, 0.0, 0.0]
    tolerances = [1e-12]*len(expected_values)
    return expected_values, computed_values, tolerances
"""```

```python"""
if __name__ == '__main__':
    driver_query_1d()
"""```"""
