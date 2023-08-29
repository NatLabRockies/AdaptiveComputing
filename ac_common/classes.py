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
    # Check that the first n_cont_vars are all continuous and the last n_cont_vars are either ordered or categorical
    for i in range(n_dim):
        if params[i].type == 'ordered' or params[i].type == 'categorical':
            break
    for ii in range(i+1,n_dim):
        if params[ii].type != 'ordered' and params[ii].type != 'categorical':
            raise Exception('Reorder the Param objects in the list of Params so that the Params of continuous data types'+
                            ' appear in the list before the other data types.')
    return True

#########################################################
class ModelOptions:
    # set the default options
    deterministic = True # random seeds are set deterministically
    perform_lower_sims = True # if a simulation is conducted at a fidelity level, it is also run at all lower fidelity levels
    mask_nans = True # Not-a-Number values are replaced with estimates from the surrogate model for the purpose of Bayesian Optimization. Otherwise, these values are excluded from the surrogate model. 
    mask_oob_values = True # out of bounds values are replaced with estimates from the surrogate model for the purpose of Bayesian Optimization. Otherwise, these values are excluded from the surrogate model.
    use_hero = False # True: AC adds simulations to a Hero queue and (multiple) Hero workers complete the jobs asynchronously. False: simulations are run locally and serially.

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
    show_exact = False # True: evaluate the simulation at 100 uniformly spaced points and plots this curve.
    show_EI = False # True: plot a curve for the expected improvement acquisition function.

#########################################################
class BoOptions:
    # set the default options
    acq_func = 'EI'
    minimization_method = 'SLSQP'
    # Relative speed of available minimization methods:
    # SLSQP: fast iterations, fast convergence
    # L-BFGS-B: fast iterations, fast convergence
    # Powell: very slow iterations, faster convergence    
    # TNC: slow iterations, slower convergence
    # Minimization methods that should not be used:
    # CG, BFGS, Newton-CG, COBYLA: can not handle bounds. Nelder-Mead: version on Eagle can not handle bounds
    # trust-constr: warnings from approximate Hessian
    # dogleg, trust-ncg, trust-exact, trust-krylov: Jacobian required
    n_opt_pts = 20 # number of initial guesses used to probe the acquisition function

#########################################################
