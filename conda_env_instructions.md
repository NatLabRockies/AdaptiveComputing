### These instructions are a work in progress.
At this point, a working conda environment has not been produced.

#### Option 2: create a conda environment

* This guarantees the correct python version and libraries are installed
* Install conda or if you are running on an HPC machine with a conda module available, run

~~~{.bash}
    module load conda
~~~

* Navigating to the AdaptiveComputing root directory, and create the environment in the provided `environment.yaml` file.

~~~{.bash}
conda env create -f environment.yaml -n AC
~~~

* where `AC` is the name of the environment and can be chosen arbitrarily
* Activate the environment with

~~~{.bash}
conda activate AC
~~~

* If this is your first time activating a conda environment, this may fail, unless you first run `conda init <SHELL_NAME>` and open a new shell. 
* The command prompt should indicate that you are running the conda environment. You can shorten/change the way this displays in the command prompt with the command

~~~{.bash}
conda config --set env_prompt '(AC)'
~~~
* where `(AC)` will be preprended on the command prompt to indicate the AC environment has been loaded.


### Running AC interactively on HPC resources
* Load the python environment and launch Jupyter notebooks
* Note that each time you start a new shell, you need to activate the conda environment
* A) If you are running on a laptop/desktop computer (only tested on M1 mac):

~~~{.bash}
conda activate AC
jupyter notebook
~~~
* B) If you are running on an HPC machine (only tested on Eagle):
* if you are running more than just a tutorial, it would be appropriate to request a login node

~~~{.bash}
srun -A acldrd -t 1:00:00 --pty /bin/bash
~~~
* Participants of the NREL LDRD project should use the `-A acldrd` option for billing, otherwise omit it.

~~~{.bash}
module load purge
module load conda
source activate AC
~~~
* noting the distinction between using the `source` and `conda` keywords on the HPC. Finally, start the notebook without a display

~~~{.bash}
jupyter notebook --no-browser --ip=0.0.0.0 --port=8080
~~~
* In a separate terminal, run 

~~~{.bash}
ssh -L 8080:<node_name>:8080 eagle.hpc.nrel.gov
~~~
* e.g. `<node_name>` is something like `el1` for a login node on eagle or `r3i6n27` for a compute node on eagle
* where `<node_name>` is replaced by the name of the node where you started the notebook (see the command prompt of that window)
* In a browser window- connect to the second URL printed when the notebook was started. The URL should begin with `http://127.0.0.1:8080/?token=…`
* All calculations will now be done on HPC, and only the results are displayed locally

* From the Jupyter GUI, run the notebook `tutorials/example_1d/driver_1d.md`
	* Note that github only tracks `.md` files rather than `.ipynb` files. This is because markdown files work better with git's version control software. However, '.md' files do not store the output and figures from notebooks, so if you want to save this, you can save the notebook as a `.ipynb`. 
