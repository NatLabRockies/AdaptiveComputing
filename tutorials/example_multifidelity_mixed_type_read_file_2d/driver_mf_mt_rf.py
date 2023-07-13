"""<div class="jumbotron text-left"><b>
    
This tutorial describes how to do multi-fidelity kriging for a 1D problem
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 2D objective function is analytically defined as $f(x) = (x*6 - 2)^2 * sin(x*12 - 4) + s$, where $s$ is defined as $s({'a','b','c','d'}) = {10,5,7.5,6}.
The global minimum over the domain $x \in [0, 1]$ is $f\approx -1.02074$, which occurs at the parameter value of $x \approx 0.757249$. </ol>
<ol> - The low fidelity model multiplied by 0.5 and shifted by $a*x+b$.</ol>
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

Define the simulations (objective functions) for the low and high fidelity models

```python"""
# low fidelity model
def lf_simulation(x):
    import numpy as np
    # evaluate the categorical variable
    if x[1] == 'a':
        s = 10.
    elif x[1] == 'b':
        s = 5.
    elif x[1] == 'c':
        s = 7.5
    elif x[1] == 'd':
        s = 6.
    else:
        raise Exception('Unrecognized value for categorical variable x[2]')
    return (
        0.5 * (((x[0] * 6 - 2) ** 2) * np.sin((x[0] * 6 - 2) * 2) + s)
        + (x[0] - 0.5) * 10.0 - 5
    )

# high fidelity model
def hf_simulation(x):
    import numpy as np
    # evaluate the categorical variable
    if x[1] == 'a':
        s = 10.
    elif x[1] == 'b':
        s = 5.
    elif x[1] == 'c':
        s = 7.5
    elif x[1] == 'd':
        s = 6.
    else:
        raise Exception('Unrecognized value for categorical variable x[2]')
    return ((x[0] * 6.0 - 2.0) ** 2) * np.sin((x[0] * 6.0 - 2.0) * 2.0) + s

simulations = [lf_simulation,hf_simulation]
"""```

Define the design parameters (inputs to the objective function)

```python"""
def driver_mf_mt_rf():
    x0 = Param()
    x0.min_val = 0
    x0.max_val = 1

    x1 = Param()
    x1.type = 'categorical'
    x1.categories = ['a','b','c','d']

    params = [x0, x1]

    # Define the options for surrogate modeling and optimization
    mod_ops = ModelOptions()

    # Compute the multi-fidelity model
    import time
    t = time.time()
    my_model = Model(simulations, params, mod_ops)
    # my_model.add_file_samples(['lf_input_data.csv','hf_input_data.csv'])
    my_model.add_file_samples(['lf_input_data_incomplete_y.csv','hf_input_data_incomplete_y.csv'])
    my_model.add_lhs_samples([4, 0])
    viz_ops = VizOptions()
    viz_ops.plot_2d=True
    bo_ops = BoOptions()
    bo_ops.cpu_hrs_per_sim = [1, 5]
    my_model.add_bo_samples(20,bo_ops=bo_ops,viz_ops=viz_ops)
    my_model.write_samples_csv(['lf_output_data.csv','hf_output_data.csv'])
    [x_opt, y_opt] = my_model.find_min()

    x_queries = np.array([[0,'a'],[0.3,'c'],[0.5,'b']], dtype=object)
    y_queries, _ = my_model.query(x_queries)
    print(y_queries)

    t = time.time() - t
    print('Elapsed time = ', t, ' s')
    print('The minimum should be y = -1.02074 at the location [x0, x1] = [0.757249, b]')
    print('The minimum found is y = ', y_opt, ' at the location [', x_opt[0],', ',x1.categories[int(x_opt[1])],']')
    computed_values = [x_opt[0], x_opt[1], y_opt]
    expected_values = [0.757249, 1.0, -1.02074] # Note: 1 maps to 'b' for x2
    assert(x1.categories[int(x_opt[1])]=='b')
    tolerances = [0.3, 1e-12, 0.1]
    return expected_values, computed_values, tolerances
"""```

```python"""
if __name__ == '__main__':
    driver_mf_mt_rf()
"""```"""
