
<div class="jumbotron text-left"><b>
    
This tutorial describes how to apply AC to the Virtual Engineering (VE) code base for doing bioreactor optimization.
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The objective function is reactor-averaged oxygen uptake rate of a biofuel synthesis reactor. </ol>
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
from ve import ve
```

Define the design parameters (inputs to the objective function)

```python
# Fraction of solids that is xylan
x0 = Param()
# x0.min_val = 0.05
##x0.min_val = 0.01 #fail
#x0.max_val = 0.7
# x0.max_val = 0.99 # fail
x0.min_val = 0.02 # surrogate bounds
x0.max_val = 0.25 # surrogate bounds

# Fraction of solids that is Glucan
x1 = Param()
# x1.min_val = 0.05
##x1.min_val = 0.01 # fail
#x1.max_val = 0.7
# x1.max_val = 0.99
x1.min_val = 0.3 # surrogate bounds
x1.max_val = 0.7 # surrogate bounds

# Porous fraction of the biomass particles
x2 = Param()
x2.min_val = 0.3
# x2.min_val = 0.01
x2.max_val = 0.99

# Initial concentration of acid
x3 = Param()
x3.min_val = 0.00001
#x3.min_val = 0.0 # fail
# x3.max_val = 0.001
x3.max_val = 0.01

# Steam temperature (C)
x4 = Param()
x4.min_val = 3.8 # steam table bounds
x4.max_val = 250.3 # steam table bounds

# Initial fraction of insoluble solids (FIS_0)
x5 = Param()
#x5.min_val = 0.5
#x5.max_val = 0.99
##x5.max_val = 1.0 fail
x5.min_val = 0.005 # surrogate bounds
x5.max_val = 0.07 # surrogate bounds

# Enzymatic Load: Ratio of the enzyme mass to the total solution mass (mg/g).
x6 = Param()
#x6.min_val = 5
#x6.max_val = 100
##x6.min_val = 0 
##x6.max_val = 1000 fail
x6.min_val = 10 # surrogate bounds
x6.max_val = 100 # surrogate bounds

# FIS_0_Target: the target value for initial fraction of insoluble solids *after* dilution (kg/kg)
x7 = Param()
x7.min_val = 0.005
x7.max_val = 0.1

params = [x0, x1, x2, x3, x4, x5, x6, x7]
```

Define the options for surrogate modeling and optimization

```python
options = Options()
#options.plot_nd = True
options.n_init_samp = 0 # must be >= ndim+1
options.n_iter = 8 # number of BayesOpt iterations
options.minimization_method = 'L-BFGS-B'
options.acq_func = 'EI'
options.n_opt_pts = 80
options.input_data_filenames = 'input_data.csv'
# options.output_data_filenames = 'output_data.csv'
```

Perform the optimization

```python
import time
t = time.time()
x_opt, y_opt, ind_best, x_data, y_data, gpr = opt(ve, params, options)
t = time.time() - t
print('Elapsed time = ', t, ' s')
print('The minimum found is y = ', y_opt, ' at the location', x_opt)
```

```python

```

