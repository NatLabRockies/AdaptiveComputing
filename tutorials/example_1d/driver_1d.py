"""<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (EGO method) for optimal parameter selection for a simple 1D function
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 1D objective function is analytically defined as $f(x) = (x-3.5) sin((x-3.5)/\pi)$. The global minimum over the domain $x \in [0, 25]$ is $f\approx-15.1251$, which occurs at the parameter value of $x \approx 18.9352$. </ol>
<ol> - The optimization is programmed by calling bayesOpt.py which uses SMT's Gaussian Process model.</ol>
<ol> - An animation of the iterations of the optimization is included to visually explain the algorithm. In multi-dimensional problems, visualization is more difficult. </ol>
</div>

```python"""
import numpy as np 
#%matplotlib notebook
import matplotlib.pyplot as plt
import os
os.chdir('/Users/kgriffin/codes/AdaptiveComputing/tutorials/example_1d')
import sys
sys.path.insert(0, '../../common') # add the path to the AdaptiveComputing common folder
from classes import *
from bayesOpt import *
import viz as viz
"""```


Import the objective function defined in func_1d.py


```python"""
from func_1d import func_1d # import each of the simulation scripts
# put each of these simulation function names in an array (without quotes). The first one is treated as the ground truth for UQ
"""```

Define the design parameters (inputs to the objective function)

```python"""
x1 = Param()
x1.name = 'x1'
x1.minVal = 0
x1.maxVal = 25
params = [x1]
"""```

Define the options for surrogate modeling and optimization

```python"""
options = Options()
options.animation_1D = True
options.plot_1D = False
options.initial_samples = 3 # must be >= ndim+1
options.n_iter = 15 # number of BayesOpt iterations
options.acqFunc = 'EI'
"""```

Define the options for surrogate modeling and optimization

```python"""
import time
t = time.time()
x_opt, y_opt, ind_best, x_data, y_data, gpr = bayesOpt(func_1d, params, options)
t = time.time() - t
print('Elapsed time = ', t, ' s')
print('The minimum should be approximately [x,y] = [18.9352,-15.1251]')
print('The minimum found is [', x_opt[0], ',', y_opt,']')
"""```

Plot an animation of the results


```python"""
viz.show_plots(options)
"""```


```python

```

```python

```"""
