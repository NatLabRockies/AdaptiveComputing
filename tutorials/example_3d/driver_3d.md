
<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (Efficient Global Optimization EGO method) for optimal parameter selection for a polynomial 3D function
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 3D objective function is analytically defined as $f(x_0,x_1,x_2) = (x_0-3)^2+(x_1-4)^2+(x_2-1)^3$.
The domain considered is $x_0 \in [0, 8]$, $x_1 \in [0, 10]$, and $x_2 \in [0, 9]$.
The global minimum of $f = 0$ occurs at $[x_0, x_1, x_2] = [3, 4, 1]$. </ol>
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
def func_3d(x):
    return pow((x[0]-3.0),2.0) + pow((x[1]-4.0),2.0) + pow((x[2]-1.0),2.0)
```

Define the design parameters (inputs to the objective function)

```python
def driver_3d():
    x0 = Param()
    x0.min_val = 0
    x0.max_val = 8

    x1 = Param()
    x1.min_val = 0
    x1.max_val = 10

    x2 = Param()
    x2.min_val = 0
    x2.max_val = 9

    params = [x0, x1, x2]

    # Define the options for surrogate modeling and optimization
    mod_ops = ModelOptions()

    # Perform the optimization
    import time
    t = time.time()
    my_model = Model(func_3d, params, mod_ops)
    my_model.add_lhs_samples(10)
    viz_ops = VizOptions()
    viz_ops.plot_nd = True
    my_model.add_bo_samples(30,viz_ops=viz_ops)
    [x_opt, y_opt] = my_model.find_min()
    t = time.time() - t
    print('Elapsed time = ', t, ' s')
    print('The minimum should be y = 0 at the location [x0_opt, x1_opt, x2_opt] = [3, 4, 1]')
    print('The minimum found is y = ', y_opt, ' at the location', x_opt)
    computed_values = [x_opt[0], x_opt[1], x_opt[2], y_opt[0]]
    expected_values = [3.0, 4.0, 1.0, 0.0]
    tolerances = [0.5]*len(expected_values)
    return expected_values, computed_values, tolerances
```

```python
if __name__ == '__main__':
    driver_3d()
```
