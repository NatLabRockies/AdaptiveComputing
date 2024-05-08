"""<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (EGO method) for optimal parameter selection for a simple 1D function
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 1D objective function is analytically defined as $f(x) = (x-3.5) sin((x-3.5)/\pi)$. The global minimum over the domain $x \in [0, 25]$ is $f\approx-15.1251$, which occurs at the parameter value of $x \approx 18.9352$. </ol>
<ol> - An animation of the iterations of the optimization is included to visually explain the algorithm.</ol>
</div>

```python"""
import sys
sys.path.insert(0, '../../') # add the path to the AdaptiveComputing directory
import numpy as np
from ac_common import *
if utils.is_notebook():
    get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
"""```


Import the objective function defined in func_mask_1d.py


```python"""
from func_mask_1d import func_mask_1d # import each of the simulation scripts
# put each of these simulation function names in an array (without quotes). The first one is treated as the ground truth for UQ
"""```

Define the design parameters (inputs to the objective function)

```python"""
def driver_mask_1d():
    x0 = Param()
    x0.min_val = -5
    x0.max_val = 23
    params = [x0]

    # Define the options for surrogate modeling and optimization
    ds_ops = DataSetOptions()
    ds_ops.ubound_inclusive = 8
    # # Mask OOBs and NaNs:
    ds_ops.exit_on_nans = False
    ds_ops.mask_nans = True
    ds_ops.exit_on_oob_values = False
    ds_ops.mask_oob_values = True
    
    # # Skip OOBs and NaNs: (exclude them from the surrogate)
    # Since using acquisition function, this reaches an OOB and then keeps requerying the same point because it's not being added to the surrogate (not masked) so the acq func max is in the skipped region everytime
    # ds_ops.exit_on_nans = False
    # ds_ops.mask_nans = False
    # ds_ops.exit_on_oob_values = False
    # ds_ops.mask_oob_values = False

    # # Throw an error for OOBs or NaNs:
    # ds_ops.exit_on_nans = True
    # ds_ops.mask_nans = False
    # ds_ops.exit_on_oob_values = True
    # ds_ops.mask_oob_values = False

    # # Throw an error for OOBs and skip NaNs:
    # ds_ops.exit_on_nans = False
    # ds_ops.mask_nans = False
    # ds_ops.exit_on_oob_values = True
    # ds_ops.mask_oob_values = False

    # # Throw an error for NaNs and skip OOBs:
    # ds_ops.exit_on_nans = True
    # ds_ops.mask_nans = False
    # ds_ops.exit_on_oob_values = False
    # ds_ops.mask_oob_values = False

    # # Throw an error for OOBs or NaNs:
    # ds_ops.exit_on_nans = True
    # ds_ops.mask_nans = True
    # ds_ops.exit_on_oob_values = True
    # ds_ops.mask_oob_values = True

    # # Throw an error for OOBs and skip NaNs:
    # ds_ops.exit_on_nans = False
    # ds_ops.mask_nans = True
    # ds_ops.exit_on_oob_values = True
    # ds_ops.mask_oob_values = True

    # # Throw an error for NaNs and skip OOBs:
    # ds_ops.exit_on_nans = True
    # ds_ops.mask_nans = True
    # ds_ops.exit_on_oob_values = False
    # ds_ops.mask_oob_values = True

    # matters if you specified oob bounds or not?
    # if exit_on_oob_values = True, must have specified oob
    # if exit_on_oob_values = False, may have specified oob (default)
    # if mask_oob_values = True, must have specified oob
    # if mask_oob_values = False, may have specified oob. (default)
        # If did specify range, then we are can ignore oobs.
        # If did not specify, then we don't need even need to check for oobs. (default)
    # if exit_on_nans = True
    # if exit_on_nans = False (default)
    # if mask_nans = True
    # if mask_nans = False, just ignores nans, does it record them or not even check? (default)

    # exit_on_oob_values = True # crash if oob value is encountered
    # exit_on_oob_values = False # behavior determined by mask_oob_values
    # exit_on_nans = True # crash if oob value is encountered
    # exit_on_nans = False # (default) behavior determined by mask_oob_values
    # question: if exit and mask are both false, will the flags be right in my code since mask is false? do I still mark these poitns as oob if they will never have a chance to be in bounds? Do I need to store the points even if they have no objective value? probably not. Well I imagine they will be in the dataset, but not in the surrogate.
    # search other tutorials that use nan or oob
    # update readme
    # mask_oob_values = True # use the surrogate to estimate a dynamic value to represent the objective here, it may or may not be in bounds and will vary as the surrogate evolves
    # mask_oob_values = False # (default) omit oob values from the surrogate

    # Perform the optimization
    import time
    t = time.time()
    my_dataset = DataSet(func_mask_1d, params, ds_ops)
    my_dataset.add_lhs_samples(2)
    viz_ops = VizOptions()
    # viz_ops.animation_1d=True
    viz_ops.plot_1d=True
    viz_ops.show_exact = True
    # use the SMT implementation of the Gaussian Process model
    from ac_common.surrogate_wrappers import SMTWrapper
    surrogate= SMTWrapper(my_dataset)
    my_dataset.add_bo_samples(10,surrogate,viz_ops=viz_ops)
    my_dataset.write_samples_csv('output_data.csv')
    [x_opt, y_opt] = my_dataset.find_min(surrogate)
    t = time.time() - t
    print('Elapsed time = ', t, ' s')
    print('The minimum should be approximately [x,y] = [18.9352,-15.1251]')
    print('The minimum found is [', x_opt[0], ',', y_opt,']')
    computed_values = [x_opt[0], y_opt[0]]
    expected_values = [18.9352, -15.1251]
    tolerances = [0.2]*len(expected_values)
    return expected_values, computed_values, tolerances
"""```

```python"""
if __name__ == '__main__':
    driver_mask_1d()
"""```"""
