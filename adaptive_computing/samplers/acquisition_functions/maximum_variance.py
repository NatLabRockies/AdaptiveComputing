import numpy as np
from scipy.stats import norm

def maximum_variance(x, surrogate, dataset, fidelity_level):
    return -surrogate.predict_variances(x,fidelity_level)
