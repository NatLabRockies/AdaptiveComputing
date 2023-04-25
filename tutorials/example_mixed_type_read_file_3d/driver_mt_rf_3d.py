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
# define the polynomial function
def func_mt(x):
    # evaluate the categorical variable by doing string comparisons
    if x[2] == 'a':
        s = 10.
    elif x[2] == 'b':
        s = 5.
    elif x[2] == 'c':
        s = 7.5
    elif x[2] == 'd':
        s = 6.
    else:
        raise Exception('Unrecognized value for categorical variable x[2]')
    return pow((x[0]-5.0),2.0) + pow((x[1]-4.0),2.0) + s - 5.0
"""```

Define the design parameters (inputs to the objective function)

```python"""
x0 = Param()
x0.type = 'continuous'
x0.min_val = 0
x0.max_val = 8

# Use an ordered integer when the order of the discrete values has significance,
# that is, we expect neigboring values to have objective function values that are correlated
x1 = Param()
x1.type = 'ordered'
x1.min_val = 2
x1.max_val = 6 # domain: 2,3,4,5,6.

# Use categorical type if the order of the categories is arbitrary.
x2 = Param()
x2.type = 'categorical'
x2.categories = ['a','b','c','d']

params = [x0, x1, x2]
"""```

Define the options for surrogate modeling and optimization

```python"""
options = Options()
options.plot_nd = True
# options.input_data_filenames = 'input_data.csv'
options.input_data_filenames = 'input_data_parameters_only.csv'
options.output_data_filenames = 'output_data.csv'
options.n_init_samp = 0 # must be >= ndim+1, left unspecified, or set to zero if sufficient samples are provided in a .csv
options.n_iter = 25 # number of BayesOpt iterations
options.acq_func = 'EI'
"""```

Perform the optimization

```python"""
import time
t = time.time()
x_opt, y_opt, ind_best, x_data, y_data, gpr = opt(func_mt, params, options)
t = time.time() - t
print('Elapsed time = ', t, ' s')
print('The minimum should be y = 0 at the location [x0_opt, x1_opt, x2_opt] = [5, 4, b]')
print('The minimum found is y = ', y_opt, ' at the location [', x_opt[0],', ',x_opt[1],', ',x2.categories[int(x_opt[2])],']')
"""```

```python

```"""
