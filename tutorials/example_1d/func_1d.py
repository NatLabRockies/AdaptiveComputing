# func_1d.py
# For the given parameter value, return the evaluation of the 1D objective
# function (x-3.5)*sin((x-3.5)/pi)
import numpy as np
def func_1d(paramVal): 
    stats = np.atleast_2d((paramVal-3.5)*np.sin((paramVal-3.5)/(np.pi)))
    return stats 
