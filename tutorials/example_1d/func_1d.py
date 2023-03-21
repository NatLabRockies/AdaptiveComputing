# func_1d.py
# For the given parameter value, return the evaluation of the 1D objective
# function (x-3.5)*sin((x-3.5)/pi)
# The argument is an array of length ndim, then number of parameters, which for this function is 1
# The output is a single value but returned as an element of a 2d array, since SMT works with column vectors which are 2d
import numpy as np
def func_1d(paramList): 
    stats = np.atleast_2d((paramList-3.5)*np.sin((paramList-3.5)/(np.pi)))
    #stats = (paramVal[0]-3.5)*np.sin((paramVal[0]-3.5)/(np.pi))
    #stats = (paramVal-3.5)*np.sin((paramVal-3.5)/(np.pi))
    return stats 
