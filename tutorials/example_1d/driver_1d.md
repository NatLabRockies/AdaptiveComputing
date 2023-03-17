<div class="jumbotron text-left"><b>
    
This tutorial describes how to use AC to do Bayesian Optimization (EGO method) for optimal parameter selection for a simple 1D function
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 1D objective function is analytically defined as $f(x) = (x-3.5) sin((x-3.5)/\pi)$. The global minimum over the domain $x \in [0, 25]$ is $f\approx-15.1251$, which occurs at the parameter value of $x \approx 18.9352$. </ol>
<ol> - The optimization is programmed by calling optimalParams.py which uses SMT's Gaussian Process model.</ol>
<ol> - An animation of the iterations of the optimization is included to visually explain the algorithm. In multi-dimensional problems, visualization is more difficult. </ol>
</div>

```python
import numpy as np 
%matplotlib notebook
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../../common') # add the path to the AdaptiveComputing common folder
from classes import *
from optimalParams import *
# or could import the whole common directory
```


Import the objective function defined in func_1d.py


```python
from func_1d import func_1d # import each of the simulation scripts
# put each of these simulation function names in an array (without quotes). The first one is treated as the ground truth for UQ
```

Define the design parameters (inputs to the objective function)

```python
params = Params()
params.names = ['p1']
params.initVals = [7] # this value and end points are considered for each parameter
params.minVals = [0]
params.maxVals = [25]
```

Define the options for surrogate modeling and optimization

```python
options = Options()
options.animation = True
options.animation_dir = './movie'
options.n_iter = 15 # number of BayesOpt iterations
options.acqFunc = 'EI'
```

Define the options for surrogate modeling and optimization

```python
surrogate = optimalParams(func_1d, params, options)
print('The optimum should be approximately [x,y] = [18.9352,-15.1251]')
print('The optimum found is [' + str(float(params.optVals)) + ',' + str(float(func_1d(params.optVals))) +']')
```

Plot an animation of the results


```python
if options.animation:
    import matplotlib.image as mpimg
    import matplotlib.animation as animation
    from IPython.display import HTML
    fig = plt.figure(figsize=[10,10])
    ax = plt.gca()
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ims = []
    for k in range(options.n_iter):
        image_pt = mpimg.imread(options.animation_dir + ('/frame_%d' %k) + '.png')
        im = plt.imshow(image_pt)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims,interval=500)
    display(HTML(ani.to_jshtml()))
```


```python

```
