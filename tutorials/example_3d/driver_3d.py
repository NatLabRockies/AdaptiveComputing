"""
<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (Efficient Global Optimization EGO method) for optimal parameter selection for a polynomial 3D function
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 3D objective function is analytically defined as $f(x_0,x_1,x_2) = (x_0-3)^2+(x_1-4)^2+(x_2-1)^3$. The domain considered is $x_0 \in [0, 8]$, $x_1 \in [0, 10]$, and $x_2 \in [0, 9]$. The global minimum of $f = 0$ occurs at $[x_0, x_1, x_2] = [3, 4, 1]$. </ol>
</div>

```python
"""
AC_path = '/Users/kgriffin/codes/AdaptiveComputing'
working_dir = AC_path + '/tutorials/example_3d'
import os
os.chdir(working_dir)
import sys
sys.path.insert(0, AC_path)
import numpy as np
from ac_common import *
if utils.is_notebook():
    get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
"""```


Define the objective function


```python"""
# define the polynomial function
def func_3d(x):
    return pow((x[0]-3.0),2.0) + pow((x[1]-4.0),2.0) + pow((x[2]-1.0),2.0)
"""```

Define the design parameters (inputs to the objective function)

```python"""
x0 = Param()
x0.minVal = 0
x0.maxVal = 8

x1 = Param()
x1.minVal = 0
x1.maxVal = 10

x2 = Param()
x2.minVal = 0
x2.maxVal = 9

params = [x0, x1, x2]
"""```

Define the options for surrogate modeling and optimization

```python"""
options = Options()
options.plot_ND = True
options.initial_samples = 10 # must be >= ndim+1
options.n_iter = 30 # number of BayesOpt iterations
# options.acqFunc = 'EI'
options.acqFunc = 'SBO'
# options.acqFunc = 'LCB'
"""```

Perform the optimization

```python"""
import time
t = time.time()
x_opt, y_opt, ind_best, x_data, y_data, gpr = bayesOpt(func_3d, params, options)
t = time.time() - t
print('Elapsed time = ', t, ' s')
print('The minimum should be y = 0 at the location [x0_opt, x1_opt, x2_opt] = [3, 4, 1]')
print('The minimum found is y = ', y_opt, ' at the location', x_opt)
"""```

```python

```"""
