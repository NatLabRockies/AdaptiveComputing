"""
<div class="jumbotron text-left"><b>
    
This tutorial demonstrates how to use AC to do Bayesian Optimization (Efficient Global Optimization EGO method) for an objective function with 3 inputs and 2 outputs.
<div>
    
Kevin Griffin
    
    January  2024

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - This tutorial is the same as example_mixed_type_read_file_3d except that the function has 2 outputs .</ol> 
<ol> - The sample points are first selected to find the minimum of the first output f0.</ol> 
<ol> - Then, sample points are selected to minimize f1.</ol> 
<ol> - $x_1$ is an integer variable, which can takes the values ${2,3,4,5,6}$.</ol> 
<ol> - $x_2$ is a categorical variable, which can take the values ${'a','b','c','d'}$.</ol> 
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
    return [pow((x[0]-5.0),2.0) + pow((x[1]-4.0),2.0) + s - 5.0, pow((x[0]-2.2),2.0) + pow((x[1]-3.0),2.0) - s + 1.0]
"""```

Define the design parameters (inputs to the objective function)

```python"""
def driver_mt_rf_3d():
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

    # Define the options for surrogate modeling and optimization
    ds_ops = DataSetOptions()

    # Perform the optimization
    import time
    t = time.time()
    my_dataset = DataSet(func_mt, params, ds_ops, n_out=2)
    my_dataset.add_file_samples('input_data.csv')
    # my_dataset.add_file_samples('input_data_parameters_only.csv')
    viz_ops = VizOptions()
    viz_ops.plot_nd=True

    # use the SMT implementation of the Gaussian Process model
    from ac_common.surrogate_wrappers import SMTWrapper
    surrogate_f0= SMTWrapper(my_dataset)

    # add samples based on EI with respect to f0 objective
    my_dataset.add_bo_samples(10,surrogate_f0,viz_ops=viz_ops)
    #my_dataset.write_samples_csv('output_data.csv')
    [x_optf0, y_optf0] = my_dataset.find_min(surrogate_f0)

    from ac_common.surrogate_wrappers import SMTWrapper
    surrogate_f1= SMTWrapper(my_dataset,i_out=1)
    # using the existing samples, we can find the minimum of f1 and it will probably be inaccurate since 
    # the samples so far have been optimizing f0.
    [x_optf1_s0, y_optf1_s0] = my_dataset.find_min(surrogate_f1)
    print('The minimum of f1 should be y = -9 at the location [x0_opt, x1_opt, x2_opt] = [2.2, 3, a]')
    print('The minimum of f1 found using initial points = ', y_optf1_s0, ' at the location [', x_optf1_s0[0],', ',x_optf1_s0[1],', ',x2.categories[int(x_optf1_s0[2])],']')
    
    # add samples based on EI with respect to f1 objective
    my_dataset.add_bo_samples(10,surrogate_f1,viz_ops=viz_ops)
    # now, after bayesian optimization with respect to the f1 objective the minimum of f1 will be more accurate
    [x_optf1_s1, y_optf1_s1] = my_dataset.find_min(surrogate_f1)
    print('The minimum of f1 should be y = -9 at the location [x0_opt, x1_opt, x2_opt] = [2.2, 3, a]')
    print('The minimum of f1 found using optimal sample points = ', y_optf1_s1, ' at the location [', x_optf1_s1[0],', ',x_optf1_s1[1],', ',x2.categories[int(x_optf1_s1[2])],']')

    my_dataset.write_samples_csv('output_data.csv')
    [x_optf0, y_optf0] = my_dataset.find_min(surrogate_f0)
    #write lhs samples to file and then make a new csv that has the outputs (not just the parameters only) and verify I can read these correctly.

    t = time.time() - t
    print('Elapsed time = ', t, ' s')
    print('The minimum of f0 should be y = 0 at the location [x0_opt, x1_opt, x2_opt] = [5, 4, b]')
    print('The minimum of f0 found is y = ', y_optf0, ' at the location [', x_optf0[0],', ',x_optf0[1],', ',x2.categories[int(x_optf0[2])],']')
    computed_values = [x_optf0[0], x_optf0[1], x_optf0[2], y_optf0[0]]
    expected_values = [5.0, 4.0, 1.0, 0.0] # Note: 1 maps to 'b' for x2
    assert(x2.categories[int(x_optf0[2])]=='b')
    tolerances = [0.1, 1e-12, 1e-12, 0.1]
    return expected_values, computed_values, tolerances
"""```

```python"""
if __name__ == '__main__':
    driver_mt_rf_3d()
"""```"""
