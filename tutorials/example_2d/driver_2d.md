
<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (Efficient Global Optimization EGO method) for optimal parameter selection for a polynomial 2D function
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 2D objective function is analytically defined as $f(x_0,x_1) = (x_0-3)^2+(x_1-4)^2$. The domain considered is $x_0 \in [0, 8]$ and $x_1 \in [0, 10]$. The global minimum of $f = 0$ occurs at $[x_0, x_1] = [3, 4]$. </ol>
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


Define the objective function


```python
# define the polynomial function
def func_2d(x):
    return pow((x[0]-3.0),2.0) + pow((x[1]-4.0),2.0)
```

Define the design parameters (inputs to the objective function)

```python
def driver_2d():
    x0 = Param()
    x0.min_val = 0
    x0.max_val = 8

    x1 = Param()
    x1.name = 'x1'
    x1.min_val = 0
    x1.max_val = 10

    params = [x0, x1]

    # Define the options for surrogate modeling and optimization
    mod_ops = ModelOptions()

    # Perform the optimization
    import time
    t = time.time()
    my_model = Model(func_2d, params, mod_ops)
    my_model.add_lhs_samples(10)
    viz_ops = VizOptions()
    viz_ops.animation_2d=True
    my_model.add_bo_samples(30,viz_ops=viz_ops)
    [x_opt, y_opt] = my_model.find_min()
    t = time.time() - t
    print('Elapsed time = ', t, ' s')
    print('The minimum should be y = 0 at the location [x0_opt, x1_opt] = [3, 4]')
    print('The minimum found is y = ', y_opt, ' at the location', x_opt)
    computed_values = [x_opt[0], x_opt[1], y_opt[0]]
    expected_values = [3.0, 4.0, 0.0]
    tolerances = [0.15]*len(expected_values)
    return expected_values, computed_values, tolerances
```

```python
if __name__ == '__main__':
    driver_2d()
```
