# Adaptive Computing
This repository contains the Adaptive Computing (AC) common software stack, which supports goal-based computing. As shown in the image below, the AC driver is designed to provide a simple and stable interface between: 

1. surrogate modeling and optimization tools (which are part of the AC common software) 
2. and user-defined application-specific simulation code.

![](ac_interface.png)

Application-specific code defines an objective, which may be to solve a global optimization problem or to train a surrogate model with minimal uncertainty. Then, the AC driver decides where in the design parameter space to run simulations to best achieve that objective. This process is iterative; as new data is returned from simulations, the AC driver chooses new simulations to run. 

Target applications of AC at NREL include-

* Optimizing the chemical process of converting biomass to biofuel
* Automating the training of surrogate models for Kinetic Monte Carlo simulations for materials synthesis
* Multifidelity simulations of the energy grid at the neighborhood scale

Key capabilities-

* Support for multi-fidelity modeling
* Continuous and discrete design parameters
* Uncertainty quantification

## Package organization and file strucutre
* The `ac_common` directory contains the AC common software stack
* The `tutorials` directory contains several example programs which demonstrate the capabilities and usage of the methods
	* Each example directory contains a driver (main) program with `.py` and `.md` extensions. These are identical except for formatting differences. The `.py` is a python script which can be run from the command line or in an IDE. The `.md` file is a Jupyter notebook. Note that with Jupytext, the markdown files can be used just like `.ipynb` notebook files, except that the output will not be saved when they are closed. This aids with version control using git.  	 

## Installing AC

### Download the source code

~~~{.bash}
git clone https://github.nrel.gov/AC/AdaptiveComputing.git
~~~

### Environment and dependencies

* Python (haven't tested which versions will work, but I am using Python 3.9.13 from a recent conda distribution)
* The surrogate modeling toolbox (SMT)
* Optional: Jupyter notebooks and Jupytext

#### Option 1: Install using a package manager
For example, on mac

~~~{.bash}
brew install conda 
pip install smt
pip install jupytext
~~~

#### Option 2: create a conda environment
More instructions coming soon ... 

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

### Option 2: Run using a Jupyter notebook

* Launch the jupyter notebook server

~~~{.bash}
jupyter notebook
~~~

* Navigate in the GUI to open `driver_1d.md`
* Note that github has been configured to only track `.md` files rather than `.ipynb` files. This is because markdown files work better with git's version control software. However, '.md' files do not store the output and figures from notebooks, so if you want to save this, you can save the notebook as a `.ipynb`.
