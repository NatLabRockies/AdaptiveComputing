<div class="jumbotron text-left"><b>
    
This tutorial describes how to do multi-fidelity kriging for a 1D problem
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 2D objective function is analytically defined as $f(x) = (x*6 - 2)^2 * sin(x*12 - 4) + s$, where $s$ is defined as $s({'a','b','c','d'}) = {10,5,7.5,6}. The global minimum over the domain $x \in [0, 1]$ is $f\approx -1.02074$, which occurs at the parameter value of $x \approx 0.757249$. </ol>
<ol> - The low fidelity model multiplied by 0.5 and shifted by $a*x+b$.</ol>
<ol> - The optimization is programmed by calling SMT's Gaussian Process model.</ol>
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

Define the objective functions for the low and high fidelity models

```python
# low fidelity model
def lf_function(x):
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
def hf_function(x):
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

functions = [lf_function,hf_function]
```

Define the design parameters (inputs to the objective function)

```python
x0 = Param()
x0.min_val = 0
x0.max_val = 1

x1 = Param()
x1.type = 'categorical'
x1.categories = ['a','b','c','d']

params = [x0, x1]
```

Define the options for surrogate modeling and optimization

```python
options = Options()
# options.animation_1d = True
# options.plot_1d = True
options.input_data_filenames = ['low_fidelity.csv','high_fidelity.csv']
options.n_iter = 0 # zero BayesOpt iterations implies this is just design of experiments and Kriging without any iterative sample acquisition
options.acq_func = 'EI'
```

Compute the multi-fidelity model

```python
options.n_init_samp = [200, 0]
import time
t = time.time()
x_opt, y_opt, ind_best, x_data, y_data, gpr = bayes_opt(functions, params, options)
t = time.time() - t
print('Elapsed time = ', t, ' s')
print('The minimum should be y = -1.02074 at the location [x0, x1] = [0.757249, b]')
print('The minimum found is y = ', y_opt, ' at the location [', x_opt[0],', ',x1.categories[int(x_opt[1])],']')
```

```python

```
