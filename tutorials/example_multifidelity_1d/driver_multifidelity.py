"""<div class="jumbotron text-left"><b>
    
This tutorial describes how to do multi-fidelity kriging for a 1D problem
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 1D objective function is analytically defined as $f(x) = (x*6 - 2)^2 * sin(x*12 - 4)$. The global minimum over the domain $x \in [0, 1]$ is $f\approx -6.02074$, which occurs at the parameter value of $x \approx 0.757249$. </ol>
<ol> - The low fidelity model multiplied by 0.5 and shifted by $a*x+b$.</ol>
<ol> - The optimization is programmed by calling SMT's Gaussian Process model.</ol>
</div>

```python"""
AC_path = '/Users/kgriffin/codes/AdaptiveComputing'
working_dir = AC_path + '/tutorials/example_multifidelity_1d'
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


Define the objective functions for the low and high fidelity models

```python"""
# low fidelity model
def lf_function(x):
    import numpy as np
    return (
        0.5 * ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)
        + (x - 0.5) * 10.0
        - 5
    )

# high fidelity model
def hf_function(x):
    import numpy as np
    return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)

functions = [lf_function,hf_function]

"""```

Define the design parameters (inputs to the objective function)

```python"""
x0 = Param()
x0.minVal = 0
x0.maxVal = 1
params = [x0]
"""```

Define the options for surrogate modeling and optimization

```python"""
options = Options()
# options.animation_1D = True
# options.plot_1D = True
options.initial_samples = 3 # must be >= ndim+1
options.n_iter = 0 # zero BayesOpt iterations implies this is just design of experiments and Kriging without any iterative sample acquisition
options.acqFunc = 'EI'
"""```

Compute the baseline high fidelity model as a baseline

```python"""
import time
t = time.time()
x_opt, y_opt, ind_best, x_data, y_data, gpr = bayesOpt(hf_function, params, options)
t = time.time() - t
print('Elapsed time = ', t, ' s')
print('The minimum should be approximately [x,y] = [0.757249,-6.02074]')
print('The minimum found is [', x_opt[0], ',', y_opt,']')

plt.figure()
x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
plt.plot(x, hf_function(x), color="k", label="Exact function")
plt.scatter(x_data, y_data, marker="o", color="c", label="HF samples")
plt.plot(x, gpr.predict_values(x), linestyle="-.", color= 'c', label="HF under-sampled GPR")
"""```

Compute the multi-fidelity model

```python"""
options.initial_samples = [7, 3] # must be >= ndim+1
t = time.time()
x_opt, y_opt, ind_best, x_data, y_data, gpr = bayesOpt(functions, params, options)
t = time.time() - t
print('Elapsed time = ', t, ' s')
print('The minimum should be approximately [x,y] = [0.757249,-6.02074]')
print('The minimum found is [', x_opt[0], ',', y_opt,']')

x_LF = x_data[:options.initial_samples[0]]
y_LF = y_data[:options.initial_samples[0]]
plt.scatter(x_LF, y_LF, marker="*", color="r", label="LF samples")
# x_HF = x_data[options.initial_samples[0]:]
# y_HF = y_data[options.initial_samples[0]:]
# plt.scatter(x_HF, y_HF, marker="*", color="g", label="HF samples")
plt.plot(x, gpr.predict_values(x), linestyle="-.", color='r', label="MF GPR")

plt.legend(loc=0)
plt.ylim(-10, 17)
plt.xlim(-0.1, 1.1)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")

plt.show()
"""```

```python

```"""
