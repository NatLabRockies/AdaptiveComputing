"""<div class="jumbotron text-left"><b>
    
This tutorial demonstrates how to pickle/upickle a model object which is useful for problems
 that use HPC and the driver program can not stay active while waiting for simulation results.
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 1D objective function is analytically defined as $f(x) = (x-3.5) sin((x-3.5)/\pi)$. The global minimum over the domain $x \in [0, 25]$ is $f\approx-15.1251$, which occurs at the parameter value of $x \approx 18.9352$. </ol>
<ol> - An animation of the iterations of the optimization is included to visually explain the algorithm.</ol>
</div>

```python"""
import pickle
import sys
sys.path.insert(0, '../../') # add the path to the AdaptiveComputing directory
import numpy as np
from ac_common import *
if utils.is_notebook():
    get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
"""```


Import the objective function defined in func_1d.py


```python"""
from func_1d import func_1d # import each of the simulation scripts
# put each of these simulation function names in an array (without quotes). The first one is treated as the ground truth for UQ
"""```

Define the design parameters (inputs to the objective function)

```python"""
def driver_pickle_1d():
    x0 = Param()
    x0.min_val = 0
    x0.max_val = 25
    params = [x0]

    # Define the options for surrogate modeling and optimization
    mod_ops = ModelOptions()

    # Perform the optimization
    import time
    t = time.time()
    my_model = Model(func_1d, params, mod_ops)
    my_model.add_lhs_samples(2)
    viz_ops = VizOptions()
    viz_ops.animation_1d=True

    # Pickle the object
    with open('data.pkl', 'wb') as file:
        pickle.dump(my_model, file)

    my_model.add_bo_samples(7,viz_ops=viz_ops)
    [x_opt, y_opt] = my_model.find_min()

    # Unpickle the object
    with open('data.pkl', 'rb') as file:
        unpickled_model = pickle.load(file)
    
    unpickled_model.add_bo_samples(7,viz_ops=viz_ops)
    [x_opt_up, y_opt_up] = my_model.find_min()
    t = time.time() - t
    print('Elapsed time = ', t, ' s')
    print('The minimum should be approximately [x,y] = [18.9352,-15.1251]')
    print('The minimum found without pickling is [', x_opt[0], ',', y_opt,']')
    print('The minimum found with pickling is [', x_opt_up[0], ',', y_opt_up,']')
    computed_values = [x_opt_up[0], y_opt_up]
    expected_values = [x_opt[0], y_opt]
    tolerances = [1e-6]*2
    return expected_values, computed_values, tolerances
"""```

```python"""
if __name__ == '__main__':
    driver_pickle_1d()
"""```"""