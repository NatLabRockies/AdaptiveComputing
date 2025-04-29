import numpy as np
from scipy.stats import norm

def expected_improvement(x, surrogate, dataset, fidelity_level):
    pred = surrogate.predict_values(x,fidelity_level)
    var = surrogate.predict_variances(x,fidelity_level)
    f_min = np.min(dataset.y_data[0])
    if var.size == 1 and var == 0.0:  
        #raise Exception('Must evaluate EI for more than one point.')
        return -(f_min - pred) # if variance is zero, expected improvement is difference of the current mean and the lowest mean of prior data
    args0 = (f_min - pred)/np.sqrt(var)
    args1 = (f_min - pred)*norm.cdf(args0)
    args2 = np.sqrt(var)*norm.pdf(args0)
    ei = args1 + args2
    return -1*ei