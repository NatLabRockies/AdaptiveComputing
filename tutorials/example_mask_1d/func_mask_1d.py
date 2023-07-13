# func_mask_1d.py
# For the given parameter value, return the evaluation of the 1D objective
# function (x-3.5)*sin((x-3.5)/pi)
# The argument is an array of length ndim, then number of parameters, which for this function is 1
# The output is a single value but returned as an element of a 2d array, since SMT works with column vectors which are 2d
import numpy as np
def func_mask_1d(x): 
    val = np.atleast_2d((x[0]-3.5)*np.sin((x[0]-3.5)/(np.pi)))
    if x[0] > 1.5 and x[0] < 5:
        val = 10 # an unallowable value
    # if x > 14 and x < 16:
    #    val = np.NaN
    # if x > 18.5 and x < 19.5:
    #     val = np.NaN
    return val
