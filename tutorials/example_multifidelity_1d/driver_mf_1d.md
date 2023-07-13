<div class="jumbotron text-left"><b>
    
This tutorial describes how to do multi-fidelity kriging for a 1D problem
<div>
    
Kevin Griffin
    
    March  2023

<div class="alert alert-info fade in" id="d110">
<p>In this notebook, </p>
<ol> - The 1D objective function is analytically defined as $f(x) = (x*6 - 2)^2 * sin(x*12 - 4)$. The global minimum over the domain $x \in [0, 1]$ is $f\approx -6.02074$, which occurs at the parameter value of $x \approx 0.757249$. </ol>
<ol> - The low fidelity model multiplied by 0.5 and shifted by $a*x+b$.</ol>
</div>

```python
import sys
sys.path.insert(0, '../../') # add the path to the AdaptiveComputing directory
import numpy as np
from ac_common import *
if utils.is_notebook():
    get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import time
```

Define the objective functions for the low and high fidelity models

```python
# low fidelity model
def lf_simulation(x):
    import numpy as np
    return (
        0.5 * ((x[0] * 6 - 2) ** 2) * np.sin((x[0] * 6 - 2) * 2) + (x[0] - 0.5) * 10.0 - 5
    )

# high fidelity model
def hf_simulation(x):
    import numpy as np
    return ((x[0] * 6 - 2) ** 2) * np.sin((x[0] * 6 - 2) * 2)

simulations = [lf_simulation,hf_simulation]

```

Define the design parameters (inputs to the objective function)

```python
def driver_mf_1d():
    x0 = Param()
    x0.min_val = 0
    x0.max_val = 1
    params = [x0]

    # Define the options for surrogate modeling and optimization
    mod_ops = ModelOptions()
    mod_ops.perform_lower_sims = False

    # Compute the low and high fidelity models as baselines
    plt.figure()
    x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
    plt.plot(x, hf_simulation([x]), color="k", label="Exact function")

    # Compute the multi-fidelity model
    t = time.time()
    my_model = Model(simulations, params, mod_ops)
    my_model.add_lhs_samples([5, 3])
    bo_ops = BoOptions()
    bo_ops.cpu_hrs_per_sim = [1, 5]
    my_model.add_bo_samples(10,bo_ops=bo_ops)
    [x_opt, y_opt] = my_model.find_min()
    
    # Query the multifidelity GP at its high fidelity level
    x_queries_hf = np.array([[0],[0.3],[0.5]])
    y_queries_hf, _ = my_model.query(x_queries_hf,fidelity_level=1)
    print(y_queries_hf)

    # Query the multifidelity GP at its low fidelity level
    x_queries_lf = np.array([[0],[0.3],[0.5]])
    y_queries_lf, _ = my_model.query(x_queries_lf,fidelity_level=0)
    print(y_queries_lf)

    t = time.time() - t
    print('Elapsed time = ', t, ' s')
    print('The minimum should be approximately [x,y] = [0.757249,-6.02074]')
    print('The minimum found is [', x_opt[0], ',', y_opt,']')

    x_LF = my_model.x_data[0]
    y_LF = my_model.y_data[0]
    plt.scatter(x_LF, y_LF, marker="*", color="c", label="Low-fidelity samples")
    x_HF = my_model.x_data[1]
    y_HF = my_model.y_data[1]
    plt.scatter(x_HF, y_HF, marker="o", color="g", label="HF samples")
    plt.plot(x, my_model.gprs[-1].predict_values(x), linestyle="-.", color='r', label="Multi-fidelity GPR")
    sig_plus = my_model.gprs[-1].predict_values(x)+3*np.sqrt(my_model.gprs[-1].predict_variances(x))
    sig_moins = my_model.gprs[-1].predict_values(x)-3*np.sqrt(my_model.gprs[-1].predict_variances(x))
    plt.fill_between(x.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='r')

    plt.legend(loc=0)
    plt.ylim(-10, 17)
    plt.xlim(-0.1, 1.1)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

    plt.savefig('mf')
    plt.show()
    computed_values = [x_opt, y_opt]
    expected_values = [0.757249, -6.02074]
    tolerances = [0.1]*len(expected_values)
    return expected_values, computed_values, tolerances
```

```python
if __name__ == '__main__':
    driver_mf_1d()
```
