# func_1d.py
# For the given parameter value, return the evaluation of the 1D objective
# function (x-3.5)*sin((x-3.5)/pi)
# The argument is an array of length ndim, which in this case is 1
import numpy as np
def func_1d(x): 
    return (x[0]-3.5)*np.sin((x[0]-3.5)/(np.pi))