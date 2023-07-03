#########################################################
# classes.py
# define some classes used throughout the package
#########################################################
class Param:
    type = 'continuous'

#########################################################    
def validate_params(params):
    import numpy as np
    params = np.atleast_1d(params)
    n_dim = len(params)
    for i in range(n_dim):
        params[i].type
        if params[i].type == 'continuous':
            if not hasattr(params[i], 'min_val'):
                raise Exception('min_val not specified for param '+str(i))
            if not hasattr(params[i], 'max_val'):
                raise Exception('max_val not specified for param '+str(i))
            if params[i].max_val <= params[i].min_val:
                raise Exception('max_val <= min_val for param '+str(i))
        elif params[i].type == 'ordered':
            if not hasattr(params[i], 'min_val'):
                raise Exception('min_val not specified for param '+str(i))
            if not hasattr(params[i], 'max_val'):
                raise Exception('max_val not specified for param '+str(i))
            if params[i].max_val <= params[i].min_val:
                raise Exception('max_val <= min_val for param '+str(i))
        elif params[i].type == 'categorical':
            if hasattr(params[i], 'min_val'):
                raise Exception('min_val should not be specified for categorical params (param '+str(i)+')')
            if hasattr(params[i], 'max_val'):
                raise Exception('max_val should not be specified for categorical params (param '+str(i)+')')
            if not hasattr(params[i], 'categories'):
                raise Exception('Categories not specified for param['+str(i)+'].')
            if len(params[i].categories) != len(set(params[i].categories)):
                raise Exception('Duplicates found in param['+str(i)+'].categories.')
            for c in params[i].categories:
                if not isinstance(c, str):
                    raise Exception('All categories of param['+str(i)+'] must be strings.')
        else:
            raise Exception('Unrecognized type for parameter '+str(i))
    return True

#########################################################
class ModelOptions:
    # set the default options
    deterministic = True # random seeds are set deterministically
    perform_lower_sims = True # if a simulation is conducted at a fidelity level, it is also run at all lower fidelity levels
    mask_nans = True # Not-a-Number values are replaced with estimates from the surrogate model for the purpose of Bayesian Optimization. Otherwise, these values are excluded from the surrogate model. 
    mask_oob_values = True # out of bounds values are replaced with estimates from the surrogate model for the purpose of Bayesian Optimization. Otherwise, these values are excluded from the surrogate model.

#########################################################
class VizOptions:
    # set the default options are below. Set boolean to true to create this vizualization
    animation_1d = False
    animation_2d = False
    animation_nd = False
    plot_1d = False
    plot_2d = False
    plot_nd = False
    output_dir = './plots'

#########################################################
class BoOptions:
    # set the default options
    acq_func = 'EI'
    minimization_method = 'SLSQP'
    n_opt_pts = 20 # number of initial guesses used to probe the acquisition function

#########################################################
