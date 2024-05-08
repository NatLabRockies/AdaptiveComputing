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
class DataSetOptions:
    # set the default options
    deterministic = True # True: random seeds are set deterministically
    perform_lower_sims = True # True: if a simulation is conducted at a fidelity level, it is also run at all lower fidelity levels
    exit_on_nans = False # True: throw a ValueError if a Not-a-Number (NaN) value is returned by a simulation. False: behavior determined by mask_nans.
    mask_nans = False # True: Not-a-Number (NaN) values are replaced with estimates from the surrogate model (allows Bayesian Optimization to advance). False: NaN values are excluded from the surrogate model.
    exit_on_oob_values = False # True: throw a ValueError if an out-of-bounds (OOB) value is encountered. The user specifies the bounds (see README). False: behavior determined by mask_oob_values.
    mask_oob_values = False # True: out-of-bounds (OOB) values are replaced with estimates from the surrogate model (allows Bayesian Optimization to advance). False: OOB values are excluded from the surrogate model if user specified bounds are provided.
    # If no bounds are specified by the user, then all values will be in bounds.
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
    mixedtype_minimization = 'differential_evolution' # other option is 'sep_cont_disc'
    sep_disc_minimizer = 'brute' # if the discrete variables input space is not too large, 'brute' is typically faster than 'differential_evolution'
    sep_cont_minimizer = 'SLSQP'
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
    dont_run_sims = False # If true, Bayesian Optimization just returns masked data points instead of evaluating simulations

#########################################################
