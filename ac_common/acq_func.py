#########################################################
# acquisitionFunctions.py
# Available acquisition functions:
# - SBO (surrogate based optimization): directly using the prediction of the surrogate model ($\mu$)
# - LCB (Lower Confidence bound): using the confidence interval : $\mu -3 \times \sigma$
# - EI for expected Improvement (EGO)
# Note: be sure to add any user defined aquisition function to get_acq_func
import numpy as np 
from scipy.stats import norm
from scipy.optimize import minimize
#########################################################
# expected improvement: 
def EI(GP,points,f_min):
    pred = GP.predict_values(points)
    var = GP.predict_variances(points)
    args0 = (f_min - pred)/np.sqrt(var)
    args1 = (f_min - pred)*norm.cdf(args0)
    args2 = np.sqrt(var)*norm.pdf(args0)
    if var.size == 1 and var == 0.0:  
        raise Exception('Must evaluate EI for more than one point.') #return 0.0
    ei = args1 + args2
    return ei
#########################################################
# surrogate Based optimization: min the Surrogate model by using the mean mu
def SBO(GP,points):
    res = GP.predict_values(points)
    return res
#########################################################
# lower confidence bound optimization: minimize by using mu - 3*sigma
def LCB(GP,points):
    pred = GP.predict_values(points)
    var = GP.predict_variances(points)
    res = pred-3.*np.sqrt(var)
    return res
#########################################################
# maximal standard deviation: tries to minimize the max standard deviation
def MSD(GP,points):
    var = GP.predict_variances(points)
    res = -np.sqrt(var)
    return res
#########################################################
def get_acq_func(IC,gpr,f_min_k):
    if IC == 'EI':
        obj_k = lambda x: -EI(gpr,np.atleast_2d(x),f_min_k)[:,0]
    elif IC =='SBO':
        obj_k = lambda x: SBO(gpr,np.atleast_2d(x))
    elif IC == 'LCB':
        obj_k = lambda x: LCB(gpr,np.atleast_2d(x))
    elif IC == 'MSD':
        obj_k = lambda x: MSD(gpr,np.atleast_2d(x))
    else:
        raise Exception('Unrecognized acq_func specified.')
    return obj_k
#########################################################
# find the minimum of the acquisition function using several initial guesses and the minimize function from scipy
def minimize_acq_func(obj_k, x_start, bo_ops, xlimits_num):
    # naive random sampling:
    #x_start = np.zeros([bo_ops.n_opt_pts,n_dim])
    #for i_r in range(n_dim):
    #    x_start[:,i_r] = np.random.rand(bo_ops.n_opt_pts)*(xlimits[i_r][1]-xlimits[i_r][0])+xlimits[i_r][0]
    opt_all = np.array([])
    for i_s in range(bo_ops.n_opt_pts):
        # minimization_method values that are sometimes appropriate:
        # Powell: slow for continuous. Works for virtualEngineering. Warns initial guess not in specified bounds for mixed types
        # SLSQP: fast for continuous. works for mixed types. `x0` violates bound constraints for virtualEngineering
        # L-BFGS-B: fast for continuous. very slow for mixed types
        # TNC: bit slower than SLSQP for continuous. mixed types raises: `x0` violates bound constraints.
        # minimization_method values that should not be used:
        # CG, BFGS, Newton-CG, COBYLA: can not handle bounds. Nelder-Mead: version on Eagle can not handle bounds
        # trust-constr: warnings from approximate Hessian
        # dogleg, trust-ncg, trust-exact, trust-krylov: Jacobian required
        opt_all = np.append(opt_all,minimize(lambda x: float(obj_k(x)), x_start[i_s,:], method=bo_ops.minimization_method, bounds=xlimits_num))
    opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]] # gets only the enties of opt_all that have 'success'=True. Note: opt_all is a dictionary, so opt_all[0]['success'] is equivalent to pt_all[0].success
    obj_success = np.array([opt_i['fun'] for opt_i in opt_success]) # create an array of the function values for all of the successful optimization points
    ind_min = np.argmin(obj_success) # which initial guess was best (led to the deepest min value)
    opt = opt_success[ind_min] # the full output for the best initial guess
    x_et_k = opt['x'] # the x value at which the min occurs
    return x_et_k # note that y_et_k will be the objective function rather than the acquisition function value at this point
#########################################################