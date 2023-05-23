<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (EGO method) for optimal parameter selection for a simple 1D function
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 1D objective function is analytically defined as $f(x) = (x-3.5) sin((x-3.5)/\pi)$. The global minimum over the domain $x \in [0, 25]$ is $f\approx-15.1251$, which occurs at the parameter value of $x \approx 18.9352$. </ol>
<ol> - The optimization is programmed by calling opt.py which uses SMT's Gaussian Process model.</ol>
<ol> - An animation of the iterations of the optimization is included to visually explain the algorithm.</ol>
</div>

```python
import sys
sys.path.insert(0, '../../') # add the path to the AdaptiveComputing directory
import numpy as np
from ac_common import *
if utils.is_notebook():
    get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
```


Import the objective function defined in func_mask_1d.py


```python
from func_mask_1d import func_mask_1d # import each of the simulation scripts
# put each of these simulation function names in an array (without quotes). The first one is treated as the ground truth for UQ
```

Define the design parameters (inputs to the objective function)

```python
def driver_mask_1d():
    x0 = Param()
    x0.min_val = -5
    x0.max_val = 23
    params = [x0]

    # Define the options for surrogate modeling and optimization
    options = Options()
    options.animation_1d = True
    # options.plot_1d = True
    options.n_init_samp = 2 # must be >= ndim+1
    options.n_iter = 10 # number of BayesOpt iterations
    options.acq_func = 'EI'
    options.ubound_inclusive = 8
    options.mask_nans = True
    options.mask_oob_values = True

    # Perform the optimization
    import time
    t = time.time()
    x_opt, y_opt, ind_best, x_data, y_data, gpr = opt(func_mask_1d, params, options)
    t = time.time() - t
    print('Elapsed time = ', t, ' s')
    print('The minimum should be approximately [x,y] = [18.9352,-15.1251]')
    print('The minimum found is [', x_opt[0], ',', y_opt,']')
    computed_values = [x_opt[0], y_opt[0]]
    expected_values = [18.9352, -15.1251]
    tolerances = [0.2]*len(expected_values)
    return expected_values, computed_values, tolerances
```

```python
if __name__ == '__main__':
    driver_mask_1d()
```
