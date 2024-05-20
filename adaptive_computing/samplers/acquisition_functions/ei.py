import numpy as np
from scipy.stats import norm

def expected_improvement(x, surrogate, f_min, fidelity_level):
    pred = surrogate.predict_values(x,fidelity_level)
    var = surrogate.predict_variances(x,fidelity_level)
    args0 = (f_min - pred)/np.sqrt(var)
    args1 = (f_min - pred)*norm.cdf(args0)
    args2 = np.sqrt(var)*norm.pdf(args0)
    if var.size == 1 and var == 0.0:  
        raise Exception('Must evaluate EI for more than one point.') #return 0.0
    ei = args1 + args2
    return -1*ei