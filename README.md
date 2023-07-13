# Adaptive Computing
This repository contains the Adaptive Computing (AC) common software stack, which supports goal-based computing. As shown in the image below, the AC driver is designed to provide a simple and stable interface between: 

1. Surrogate modeling and optimization tools (which are part of the AC common software) 
2. And user-defined application-specific simulation code.

<figure align = "center"><img src="images_for_readme/ac_interface.png" alt="Trulli" style="width:80%"><figcaption align = "center"><b>This repository encompasses the blue dashed box.</b></figcaption></figure>

Application-specific code defines an objective, which may be to solve a global optimization problem or to train a surrogate model with minimal uncertainty. Then, the AC driver decides where in the design parameter space to run simulations to best achieve that objective. This process is iterative; as new data is returned from simulations, the AC driver chooses new simulations to run. 

Target applications of AC at NREL include-

* Optimizing the chemical process of converting biomass to biofuel
* Automating the training of surrogate models for Kinetic Monte Carlo simulations for materials synthesis
* Multifidelity simulations of the energy grid at the neighborhood scale

Key capabilities-

* Bayesian optimization
* Surrogate model training (multi-fidelity Gaussian process model)
* Continuous and discrete design parameters
* Uncertainty quantification

## Accessing the Repo and Installing

* Contact Marc Day or Kevin Griffin <kgriffin@nrel.gov> to be added to the Adaptive Computing github repo. You can verify that you have been added by attempting to visit this site: <https://github.com/NREL/AdaptiveComputing>
* Next you may need to setup github ssh keys if you encounter password issues when attempting to clone the repositiory as described below. To set up ssh keys, be sure to use your <github.com> account rather than your <github.nrel.gov> account (if applicable). Instructions are available [at this link](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
* See the installation instructions below.

## Package details

The `tutorials` directory contains several example programs which demonstrate the capabilities and usage. The available tutorials are listed below:

* `example_1d` finds the minimum of `f(x) ~ x sin(x)`. Animate the evolution of the Bayesian optimization and how the Gaussian Process Regression (GPR) and its uncertainty evolves with each iteration.
* `example_2d` finds the minimum of a 2D paraboloid.
* `example_3d` finds the minimum of a 3D paraboloid.
* `example_mixed_type` is similar to `example_3d` except that two of the variables are replaced with ordered integer and categorical types.
* `example_read_file` same as `example_mixed_type` except that the former reads data from a csv file instead of using pseudo-random initial sampling.
* `example_multifidelity_1d` train a GPR using high fidelity and low fidelity function evaluations. Note: this function is not iterative yet. It uses pseudo-random sampling to find the minimum.
* `example_multifidelity_mixed_type_read_file_2d` same as `example_multifidelity_1d` except the former adds a categorical variable, so it uses mixed types. Also, it reads some initial data from csv files and collects some from pseudo-random initial sampling.
* `example_mask_1d` is similar to `example_1d` except that the objective function has been modified to return `NaN` for some input arguments to emulate an objective function that is ill behaved in some region of the sample space. The example demonstrates the masking capability, which is a robustness feature that is described below.
* `example_query_1d` is similar to `example_1d` but after training the surrogate, it calls `query()` to evaluate the surrogate and dynamically run more simulations (and retrain) if the standard deviation (uncertainty) exceeds the user-specified limit.
* `example_pickle_1d` is similar to `example_1d` but it saves the model as a pickle and then unpickles it. This is useful if you need to train a model and then load into memory at a later time. It pickles the model and all simulation evaluations.

<!-- The following tutorial(s) are coming soon: -->

<!-- * `example_step_1d` uses the MSD acquistion function to place points to minimize variance estimated in the surrogate model rather than searching for a minimum of the predicted mean of the model. -->

### Explanation of the `Model` class
The workhorse of the AC package is the `Model` class. It contains the simulation sample points and the surrogate model (Gaussian Process model) which is trained on those sample points.

A model is created with the constructor

~~~{.bash}
my_model = Model(simulations, params, mod_ops)
~~~

These arguments are defined in detail below. They define the simulations (or objective functions) of interest, the sample space parameters which are the inputs to the simulations, and the surrogate model options.

Training data is provided to the model using add_lhs_samples, add_file_samples, add_bo_samples, add_batch_bo_samples, etc. methods. The surrogate model is re-trained whenever samples are added.

~~~{.bash}
my_model.add_lhs_samples(n_lhs_samp)
my_model.add_file_samples(input_data_filenames)
my_model.add_bo_samples(n_iter)
my_model.write_samples_csv(output_data_filenames)
~~~

* `add_lhs_samples` samples the simulations using Latin Hypercube Sampling. See [here for details](https://smt.readthedocs.io/en/latest/_src_docs/sampling_methods/lhs.html). 
* `add_file_samples` reads samples from a csv file. 
* `add_bo_samples` adds Bayesian Optimization samples (BO), which are described below. 
* `write_samples_csv` writes all prior samples to a csv file.

Input arguments:

| Field name | Default value |  Acceptable types |  Acceptable values | <div style="width:500px">Description</div>  |
|---|---|---|---|---|
| `input_data_filenames` | none | string, list of strings  | empty string or strings ending in `.csv`  |  file names to read existing data from. List length must equal the number of  simulations levels provided. See details of file format below. |
 `output_data_filenames` | none | string, list of strings | empty string or strings ending in `.csv`  |  file names to write final data to. This includes the data read from a file, from LHS sampling, and from Bayesian optimization. List length must equal the number of  simulations levels provided. See details of file format below. |
| `n_iter`  | none  | integer  |  positive or zero | Number of Bayesian Optimization iterations. |
| `n_lhs_samp`  | none  | integer  | >= 2 or =zero | Number of pseudo-random initial samples collected using Latin Hypercube Sampling used to initialize the Bayesian Optimization. |

The model can be queried using the query function:

~~~{.bash}
# 3 queries (x0, x1) of a two parameter model (x0 is continuous type, x1 is a categorical type)
x_queries = np.array([[0,'a'],[0.3,'c'],[0.5,'b']], dtype=object)
y_queries_mean_values, y_queries_variances = my_model.query(x_queries)
~~~

Note that since mixed type is used in this example, the queries must be stored in a numpy array of `dtype=object` so that the entries of the array can have different types.

Input arguments:

| Field name | Default value |  Acceptable types |  Acceptable values | <div style="width:500px">Description</div>  |
|---|---|---|---|---|
| `x_queries` | none | numpy array of length = # queries. Each entry is a list of simulation arguments | Bounds and argument data types are set by `Model` constructor | The locations in the sample space where the surrogate model is queried. |
| `fidelity_level` | -1 | integer | valid index for an array of length `0:n_fl` | Which fidelity level surrogate model to use for all queries. Defaults to highest level. |
| `threshold_std` | `None` (optional) | float | `> 0.0` | For each `x_query` of fidelity level`i`, if the surrogate model's standard deviation at the queried point exceeds the user-specified threshold, a simulation of fidelity level `i` is conducted at `x_query`. |

~~~{.bash}   
y_queries, y_queries_var = my_model.query(x_queries,threshold_std=0.4)
~~~

### Bayesian optimization

Bayesian optimization, is a technique for surrogate-based optimization, which is illustrated in the figure below. In this 1 parameter example, the user defines a python function which returns the value of the `x sin(x)` function. `add_bo_samples` sees this as a black-box function which it can evaluate at various parameter (`x`) values. 

``add_bo_samples`` begins with a few random function evaluations. These can be provided at specified points in the design parameter space or can be automatically selected using Latin Hypercube Sampling. The inital data points are used to train a Gaussian Process (GP) model of the black-box function. The GP model provides an estimate of the function, which is the mean of the GP (also called a Gaussian Process Regression and is essentially a smooth interpolation of the sampled points) and the variance of the GP model which estimates the uncertainty of the GPR in between sampled data points. 

<figure align = "center"><img src="images_for_readme/example_1d.png" alt="Trulli" style="width:50%"><figcaption align = "center"><b>A limited number of function evaluations are used to train a GP. We plot the GPR estimate and its confidence intervals, the sample points, and the reference exact function.</b></figcaption></figure>

So far, the Bayesian optimization has not started, we have just performed regression on some sample points. Bayesian optimization is the iterative selection of additional sample points in a way that optimizes some objective. This is stated mathematically by defining an acquisition function (details to follow). For example, below the evolution of the optimization is shown where in each iteration one sample is chosen that has the highest likelihood of having a deeper minimum than any previously sampled point.

<figure align = "center"><img src="images_for_readme/movie_1d.gif" alt="Trulli" style="width:50%"><img src="images_for_readme/example_1d.png" alt="Trulli" style="width:50%"><figcaption align = "center"><b>Left: Animation (gif) of the Bayesian Optimization algorithm for a 1 parameter function. Right: still frame illustrating that for the chosen acquisition function, the next function evaluation will be chosen at a parameter value where there is large uncertainty.</b></figcaption></figure>

### User-defined simulations
`simulations` is the name of the user-defined function that implements a simulation. The form is `f(x)`, where `x` is a list of arguments. In the example above, it was a simple function that evaluates `a*x*sin(x/b-c)`. In a multi-fidelity setting, `simulations` is a list of the user-defined functions that implement simulations which various cost and accuracy. This represents a hierarchy of simulation fidelities. More information on multi-fidelity is available further down this page. Note, all simulations must take the same arguments and return the same scalar output. The scalar output is called the quantity of interest and it is the same for all simulations.

~~~{.bash}
import high_fidelity as high_fidelity
import low_fidelity as low_fidelity
# List of user-defined function names
simulations = [high_fidelity, low_fidelity]
~~~

### Parameter list
A list of `Param` objects, each of which specifies the type and range of allowable values for one argument the user-defined functions.

~~~{.bash}
x0 = Param()
# x0.type = 'continuous' # this is the default type, so don't need to specify the type explicitly
x0.min_val = 0
x0.max_val = 8

x1 = Param()
x1.type = 'ordered'
x1.min_val = 2
x1.max_val = 6 # domain: 2,3,4,5,6.

x2 = Param()
x2.type = 'categorical'
x2.categories = ['a','b','c','d']

params = [x0, x1, x2]

# example of calling a function follows the constraints defined by params
low_fidelity(3.14, 6, 'c')
~~~

The user-defined simulations can have arguments of three different types:

* `continuous` parameters are the default type and are floating point numbers that can take any value from `min_val` to `max_val` (inclusive). The `min_val` and `max_val` fields must be specifed. 
* `ordered` parameters are integers. Such a parameter can take any value from `min_val` to `max_val` (inclusive). The `min_val` and `max_val` fields must be specifed. Use an ordered integer when the order of the discrete values has significance, that is, we expect neigboring values to have simulation output values that are correlated.
* `categorical` parameters are represent a discrete list of possibilties. Instead of specifying `min_val` and `max_val`, the `categories` field lists the discrete string values that the variable can take. This type should be used when the order of values in `categories` is arbitrary (this is what makes this type different from representing options with ordered integers).

### Options
The `Model` constructor requires the `ModelOptions` object as an argument. After initializeing the object, its default fields can be overwritten and optional fields can be set.

~~~{.bash}
mod_ops = ModelOptions()
# example of overwriting a default field:
mod_ops.deterministic = False
# example of setting an optional field:
mod_ops.lbound_inclusive = 3.5
~~~

Fields of `ModelOptions()`:

| Field name | Default value |  Acceptable types |  Acceptable values | <div style="width:500px">Description</div>  |
|---|---|---|---|---|
| `deterministic`  |  `True` | boolean | `True` or `False` |  True: random seeds for sampling are chosen deterministically so that results are reproducible. |
| `perform_lower_sims` | `True` | boolean | `True` or `False` | True: at every point in the design space where a simulation has been performed, all lower fidelity models are also simulated there too. |
| `mask_nans` | `True` | boolean | `True` or `False` | True: represent NaN simulation values with the surrogate model. See details on masking algorithm. False: NaNs trigger an error. |
| `mask_oob_values` | `True` | boolean | `True` or `False` | True: represent simulation values that are out of allowable bounds with the surrogate model. See details on masking algorithm. False: out of bounds simulation values trigger an error. |
| `lbound_inclusive`, `ubound_inclusive`, `lbound_exclusive`, `ubound_exclusive` | none | float | any | Define the allowable bounds for simulation returns. See details on masking below. |


`VizOptions()` controls plotting and is an optional argument to `add_bo_samples`. The fields of `VizOptions()` are:

| Field name | Default value |  Acceptable types |  Acceptable values | <div style="width:500px">Description</div>  |
|---|---|---|---|---|
| `animation_1d`  | `False`  | boolean  | `True` or `False` |  True: show and save a movie of the Bayesian Optimization iterations. `n_dim` must = 1. |
| `animation_2d`  | `False`  | boolean  | `True` or `False` |  True: show and save a movie of the Bayesian Optimization iterations. `n_dim` must = 2. |
| `plot_1d`  | `False`  | boolean  | `True` or `False` |  True: show and save a plot of the final result of the optimization. `n_dim` must = 1. |
| `plot_2d`  | `False`  | boolean  | `True` or `False` |  True: show and save a plot of the final result of the optimization. `n_dim` must = 2. |
| `plot_nd`  | `False`  | boolean  | `True` or `False` |  True: show and save a plot of the final result of the optimization. Plots objective function versus the n-dimensional distance in parameter from the optimal parameter value. |
| `output_dir`  | `./plots` | string | any |  All plots and animations are saved to `./output_dir/`. The directory is created if it doesn't exist. |


Bayesian optimization options are set with `BoOptions()`, which is another optional argument for `add_bo_samples`. The fields of `BoOptions` are:

| Field name | Default value |  Acceptable types |  Acceptable values | <div style="width:500px">Description</div>  |
|---|---|---|---|---|
| `acq_func`  | `'EI'`  |  string |  `'EI'`, `'LCB'`, `'SBO'`, `'MSD'` | Chose which acquisition function to use for the optimization. See descriptions below.  |
| `minimization_method` | `'SLSQP'` | string | `'SLSQP'` or `'Powell'` | See [this link](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize) for details on the available optimization methods. `SLSQP` is generally recommended, but `Powell` can be tried if there are bounds violation errors. This issue is under investigation. |
| `n_opt_pts` | 20 | integer | `>= 0` | Number of initial guesses used to sample the acquisition function to find its minimum. These samples are placed in the parameter space using Latin Hypercube Sampling. |
| `cpu_hrs_per_sim` | none | list of floats | `>0` | An estimate of the number of CPU (or cost-equivalent resource) hours required to compute a simulation at each fidelity level. The length of the list equals the number of user-defined simulations. This input is required if the number of user-defined simulations `n_fl > 1` and Bayesian optimization is used to select the next simulation point, that is `n_iter > 0`. |

<!-- | `` |  |  |  |  | -->

#### More information about acquisition functions
The following is a list of supported acquisition functions. These determine which function evaluation will be made on the present iteration. Note that `EI`, `LCB`, and `SBO` are written to find the global minimum, so the objective function should be negated if the maximum is sought.

* Set `bo_ops.acq_func = EI` to use the Expected Improvement algorithm. [Click here](https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/) for a description of the algorithm. As this acquisition function is widely used and generally recommended, it is the default value.
* `bo_ops.acq_func = LCB` to use the Lower Confidence Bound algorithm. This queries the point in the design space where the surrogate's mean minus 3 times its variance is minimal. Thus, it probes the point where the 99% confidence interval is lowest. This acquisition function can converge quickly but is not particularly robust.
* `bo_ops.acq_func = SBO` to use the Surrogate-Based Optimization algorithm. This queries the point in the design space where the surrogate's mean is minimal. This acquisition function is generally only useful for finding local minima and is not particularly robust though it can converge quickly.
* `bo_ops.acq_func = MSD` to use the Maximal Standard Deviation algorithm. This queries the point in the design space that has the largest standard deviation estimated by the surrogate model. 

#### Input and output data file format
The entries in the list `input_data_filenames` are comma separated value `.csv` files. In each file, the first N columns represent the N parameters of the design space. The last column is the value of the objective function. This column is optional in the input file. The rows represent the data points. The first row is reserved for the data type labels. These indicate the data type of each column (design parameter). So each entry is either `continuous`, `ordered`, or `categorical`. The entry in objective function column's first row should be `y`.

Example `.csv` file with two data points. This file specifies the three design parameters (which are of three different data types). The fourth column specifies corresponding objective function value, `y`.

| continuous | ordered | categorical | y |
|---|---|---|---|
| 4	| 6 | c | 7.5 |
| 7 | 3 | d | 6 |

If you would like AC to compute the objective function for you, leave out the last column or leave some entries in the last column empty. For example,

| continuous | ordered | categorical |
|---|---|---|
| 4	| 6 | c | 
| 7 | 3 | d |

#### More information on masking NaN and out of bounds (unallowable) simulation values
Anytime a simulation returns a `NaN` value, that data point will be flagged. Similarly, the user can also define a range of allowable simulation values and all simulation return values outside this range will be flagged. By default, the flagged values are then masked, to prevent them from contaminating the surrogate model (details described below). If the user prefers not to use masking, the calculation can be terminated immediately upon discovering a `NaN` or out of bounds value by setting `mod_ops.mask_nans=False` or `mod_ops.mask_oob_values=False`, respectively. 

Lower and upper bounds can be inclusive or exclusive. Upper and/or lower bounds should be omitted if semi-infinite or infinite bounds are needed. The simulation return value f(x) must obey all specified constraints in order to be classified as allowable:

* f(x) >= `lbound_inclusive`
* f(x) <= `ubound_inclusive` 
* f(x) > `lbound_exclusive`
* f(x) < `ubound_exclusive`

The surrogate model is trained using all unmasked data (non-NaN, within allowable bounds). However, just ignoring masked data would cause the Bayesian Optimization algorithm to repeatedly sample data in regions in the design space where NaNs or unallowable values are found (since the uncertainty will not reduce in these regions if returned values are always ignored). The masking algorithm is as follows.

* A precursor surrogate model is trained using all unmasked data.
* The values for masked data points are estimated using the precursor surrogate model.
* A surrogate model is retrained using unmasked data and the estimates for masked data points.
* This second surrogate model is used for evaluating the acquistion function.

This two-step training is performed at every iteration, and the precursor surrogate model is returned at the end of all iterations. This algorithm effectively fills in masked holes in the objective function using the present value of the surrogate model.  

## Multi-fidelity
The user defines a list of simulations which contains each of the fidelity levels in ascending order. The user's estimate for the cost of the fidelity levels is specified with `bo_ops.cpu_hrs_per_sim`. See the tutorial `tutorials/example_multifidelity_1d.md` for more details on use.

<figure align = "center"><img src="images_for_readme/mf.png" alt="Trulli" style="width:80%"><figcaption align = "center"><b>Comment.</b></figcaption></figure>

A bi-fidelity model is constructed using two corrections functions `rho` and `delta`, which are both functions of the design parameter(s).

~~~{.bash}
y_BF1[x] = y_0[x] rho[x] + delta[x]
~~~

These correction functions are assumed to be low order polynomials. Their coefficients are found by the least squares regression of the high fidelity data (`y_1`) to the bi-fidelity model.

This framework is used recursively to support an arbitrary level of fidelity levels. For example, if there were a third (an even higher fidelity) level (`y_2`), then `y_BF2[x] = y_BF1[x] rho1[x] + delta1[x]`. And so on for higher fidelity levels.

## Development notes


### Python and Jupyter notebooks
* Each example directory contains a driver (main) program with `.py` and `.md` extensions. These are identical except for formatting differences. The `.py` is a python script which can be run from the command line or in an IDE. The `.md` file is a Jupyter notebook. Note that with Jupytext, the markdown files can be used just like `.ipynb` notebook files, except that the output will not be saved when they are closed. This aids with version control using git.  	

### Capitalization and style
PEP8 style is used for capitalization. That is:

* Function and variable names are lowercase, with words separated by underscores as necessary to improve readability.
* Class names use UpperCamelCase.
* Constants are usually defined on a module level and written in all capital letters with underscores separating words. 

## Installing AC

### Download the source code

~~~{.bash}
git clone https://github.com/NREL/AdaptiveComputing.git
~~~

### Environment and dependencies

* Python (tested with versions 3.9, 3.10, and 3.11 on Mac and Ubuntu)
* The surrogate modeling toolbox (SMT) version 1.3 (version 2 not supported)
* IPython
* Matplotlib
* Optional: Jupyter notebooks and Jupytext

#### Option 1: Install locally using a package manager
For example, on mac

~~~{.bash}
brew install conda 
pip install smt==1.3
# optional:
pip install jupytext
~~~

#### Option 2: Install on Eagle (an NREL HPC machine)

~~~{.bash}
module load conda
~~~

## Running AC

* Change the working directory to an example directory. E.g.,

~~~{.bash}
cd tutorial/example_1d
~~~

### Option 1: Run using python

* From the command line, run

~~~{.bash}
python driver_1d.py
~~~

* For HPC, may use graphics forwarding or use (FastX)[https://www.nrel.gov/hpc/eagle-software-fastx.html] to connect to `eagle-dav.hpc.nrel.gov`

### Option 2: Run using a Jupyter notebook

* Note that github has been configured to only track `.md` files rather than `.ipynb` files. This is because markdown files work better with git's version control software. However, '.md' files do not store the output and figures from notebooks, so if you want to save this, you can save the notebook as a `.ipynb`.

#### Option 2a: Run on a local machine using a Jupyter notebook

* Launch the jupyter notebook server

~~~{.bash}
jupyter notebook
~~~

* Navigate in the GUI to open `driver_1d.md`

### Option 2b: Run using a Jupyter notebook

* See the `conda_env_instructions.md`
