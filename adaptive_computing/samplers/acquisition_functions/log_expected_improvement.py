import numpy as np
from scipy.stats import norm


def log_expected_improvement(x, surrogate, dataset, fidelity_level):
    """Negative log expected improvement, -log(EI + 1e-30), for minimization.

    Same argmin as expected_improvement, but well scaled where EI is tiny, which
    is most of the box late in a run.  The incumbent f_min is the best unmasked
    value at the fidelity being evaluated (and the surrogate's output), not at
    fidelity 0.

    The batched multi-fidelity Hero drivers (hero_batched_mf.py and its
    subclasses) use this and recover EI as exp(-value).  Do not use it with
    drivers that divide the acquisition value by a cost (active_cost_ratio.py):
    a ratio of log values has no meaning.
    """
    pred = surrogate.predict_values(x, fidelity_level)
    var = surrogate.predict_variances(x, fidelity_level)
    i_output = getattr(surrogate, 'i_output', 0)
    _, y_unmasked = dataset.get_unmasked_data(i_fidelity=fidelity_level, i_output=i_output)
    f_min = np.min(y_unmasked[:, i_output])
    if var.size == 1 and var == 0.0:
        # zero variance: EI is the improvement of the mean itself
        return -np.log(np.maximum(f_min - pred, 0.0) + 1.e-30)
    args0 = (f_min - pred) / np.sqrt(var)
    ei = (f_min - pred) * norm.cdf(args0) + np.sqrt(var) * norm.pdf(args0)
    return -np.log(np.maximum(ei, 0.0) + 1.e-30)
