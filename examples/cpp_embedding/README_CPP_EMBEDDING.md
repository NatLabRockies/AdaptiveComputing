# Installation

~~~{.bash}
module load conda
conda create --name AC-AMAR-3.11-v2 python=3.11
conda activate AC-AMAR-3.11-v2
pip install smt==1.3
conda install numpy IPython matplotlib
~~~

Export library path
~~~{.bash}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda-envs/AC-AMAR-3.11-v2/lib
~~~
Assuming that this is the path where `libpython3.11.so` is located

Compile c++ code
~~~{.bash}
g++ query.cpp -o query -L ~/.conda-envs/AC-AMAR-3.11-v2/lib -lpython3.11 -I ~/.conda-envs/AC-AMAR-3.11-v2/include/python3.11
~~~
Assuming `libpython3.XX.so` can be found at the library path
And Assuming `Python.h` can be found at the include path

Note: C++ program loads and executes the Python file at runtime, not during compilation. So you can change the python file after compilation.
