#########################################################
# acquisitionFunctions.py
# Available acquisition functions:
# - SBO (surrogate based optimization): directly using the prediction of the surrogate model ($\mu$)
# - LCB (Lower Confidence bound): using the confidence interval : $\mu -3 \times \sigma$
# - EI for expected Improvement (EGO)
# Note: be sure to add any user defined aquisition function to get_acq_func
import numpy as np 
from scipy.stats import norm
from scipy.optimize import minimize, brute, differential_evolution
#########################################################
# expected improvement: 
def EI(GP,points,f_min,fidelity_level):
    pred = GP.predict_values(points,fidelity_level)
    var = GP.predict_variances(points,fidelity_level)
    args0 = (f_min - pred)/np.sqrt(var)
    args1 = (f_min - pred)*norm.cdf(args0)
    args2 = np.sqrt(var)*norm.pdf(args0)
    if var.size == 1 and var == 0.0:  
        raise Exception('Must evaluate EI for more than one point.') #return 0.0
    ei = args1 + args2
    return ei
#########################################################
# surrogate Based optimization: min of the surrogate model by using the expected value mu
def SBO(GP,points,fidelity_level):
    res = GP.predict_values(points,fidelity_level)
    return res
#########################################################
# lower confidence bound optimization: minimize by using mu - 3*sigma
def LCB(GP,points,fidelity_level):
    pred = GP.predict_values(points,fidelity_level)
    var = GP.predict_variances(points,fidelity_level)
    res = pred-3.*np.sqrt(var)
    return res
#########################################################
# maximal standard deviation: tries to minimize the max standard deviation
def MSD(GP,points,fidelity_level):
    var = GP.predict_variances(points,fidelity_level)
    res = -np.sqrt(var)
    return res
#########################################################
def get_acq_func(IC,gpr,f_min_k,fidelity_level):
    if IC == 'EI':
        obj_k = lambda x: -EI(gpr,np.atleast_2d(x),f_min_k,fidelity_level)[:,0]
    elif IC =='SBO':
        obj_k = lambda x: SBO(gpr,np.atleast_2d(x),fidelity_level)
    elif IC == 'LCB':
        obj_k = lambda x: LCB(gpr,np.atleast_2d(x),fidelity_level)
    elif IC == 'MSD':
        obj_k = lambda x: MSD(gpr,np.atleast_2d(x),fidelity_level)
    else:
        raise Exception('Unrecognized acq_func specified.')
    return obj_k

#########################################################
# Perform a constrained optimization of the continous arguments for a specific choice
# of the integer arguments
# Input: xi is a list of integer arguments (specifies all of the ordered and categorical types)
# Output #2: the optimal choice of continous arguments (for the particular choice of input integer arguments)
# that minimizes the objective function obj_k
# Output #1: the minimum achieved for this choice of continuous and integer arguments.
def min_for_cont_vars(xc_start,xclimits_num,obj_k,bo_ops,xi):
    opt_all = np.array([])
    for i_s in range(bo_ops.n_opt_pts):
        opt_all = np.append(opt_all,minimize(lambda xf: float(obj_k(np.append(xf,xi))), xc_start[i_s], method=bo_ops.sep_cont_minimizer, bounds=xclimits_num))
    opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]] # gets only the entries of opt_all that have 'success'=True. Note: opt_all is a dictionary, so opt_all[0]['success'] is equivalent to opt_all[0].success
    obj_success = np.array([opt_i['fun'] for opt_i in opt_success]) # create an array of the function values for all of the successful optimization points
    ind_min = np.argmin(obj_success) # which initial guess was best (led to the deepest min value)
    opt = opt_success[ind_min] # the full output for the best initial guess
    xf_opt = opt['x'] # the x value at which the min occurs
    return np.min(obj_success),xf_opt

#########################################################
# Return the argmin of the objective function obj_k
# Minimization of continuous arguments uses scipy minimize
# Minimization of integer arguments uses scipy brute
def minimize_acq_func(dataset,obj_k,x_start,bo_ops):
    xc_start = x_start[:,0:dataset.n_cont_vars]
    xclimits_num = dataset.xlimits_num[0:dataset.n_cont_vars]
    xdlimits_num = dataset.xlimits_num[dataset.n_cont_vars:]
    n_disc_vars = len(xdlimits_num)
    
    # Combined optimization of continuous and discrete varaibles:
    if dataset.mixed_type and bo_ops.mixedtype_minimization == 'differential_evolution':
        DE_bounds = [0]*dataset.n_in
        integrality = [False]*dataset.n_in
        for i in range(dataset.n_in):
            DE_bounds[i] = (dataset.xlimits_num[i][0], dataset.xlimits_num[i][-1])
            integrality[i] = (dataset.params[i].type == 'ordered') or (dataset.params[i].type == 'categorical')
        x_et_k = differential_evolution(lambda x: float(obj_k(x)), DE_bounds, integrality=integrality)['x']
        # COBYLA
        # would need this new argument: unfolded_limits = sampling_opt._unfolded_xlimits
        # opt_all = np.array([])
        # bounds = unfolded_limits
        # cons = []
        # for j in range(len(bounds)):
        #     lower, upper = bounds[j]
        #     # if self.work_in_folded_space:
        #     #     if isinstance(self.xtypes[j], tuple):
        #     #         upper = int(upper - 1)
        #     l = {"type": "ineq", "fun": lambda x, lb=lower, i=j: x[i] - lb}
        #     u = {"type": "ineq", "fun": lambda x, ub=upper, i=j: ub - x[i]}
        #     cons.append(l)
        #     cons.append(u)
        # options = {"maxiter": 500, "catol": 1e-6, "tol": 1e-6, "rhobeg": 0.2}
        # for i_s in range(bo_ops.n_opt_pts):
        #     opt_all = np.append(opt_all,minimize(lambda x: float(obj_k(x)), x_start[i_s], method='COBYLA', bounds=xclimits_num,constraints=cons,options=options))
        # opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]] # gets only the entries of opt_all that have 'success'=True. Note: opt_all is a dictionary, so opt_all[0]['success'] is equivalent to opt_all[0].success
        # obj_success = np.array([opt_i['fun'] for opt_i in opt_success]) # create an array of the function values for all of the successful optimization points
        # ind_min = np.argmin(obj_success) # which initial guess was best (led to the deepest min value)
        # opt = opt_success[ind_min] # the full output for the best initial guess
        # x_et_k = opt['x'] # the x value at which the min occurs
    else: # Separate optimization methods for continuous and discrete variables:
        ranges = ()
        for i in range(dataset.n_cont_vars,dataset.n_in):
            ranges = ranges+(slice(dataset.xlimits_num[i][0], dataset.xlimits_num[i][-1]+1, 1),)
        DE_bounds = [0]*n_disc_vars
        for i in range(n_disc_vars):
            DE_bounds[i] = (xdlimits_num[i][0], xdlimits_num[i][-1])    
        if dataset.n_cont_vars == 0: # if all data types are discrete
            if bo_ops.sep_disc_minimizer == 'brute':
                x_et_k = brute(lambda x: float(obj_k(x)), ranges, finish=None)
            elif bo_ops.sep_disc_minimizer == 'differential_evolution':
                x_et_k = differential_evolution(lambda x: float(obj_k(x)), DE_bounds, integrality=[True]*n_disc_vars)['x']
            else:
                raise Exception('Unrecognized option for discrete minimization method: bo_ops.sep_disc_minimizer = '+bo_ops.sep_disc_minimizer)
        elif dataset.n_cont_vars == dataset.n_in: # if all data types are continuous
            x_et_k = min_for_cont_vars(xc_start,xclimits_num,obj_k,bo_ops,[])[1]
        else: # if there are a mixture of continuous and discrete data types
            if bo_ops.sep_disc_minimizer == 'brute':
                xi_opt = brute(lambda x: min_for_cont_vars(xc_start,xclimits_num,obj_k,bo_ops,x)[0], ranges, finish=None)
            elif bo_ops.sep_disc_minimizer == 'differential_evolution':
                xi_opt = differential_evolution(lambda x: min_for_cont_vars(xc_start,xclimits_num,obj_k,bo_ops,x)[0], DE_bounds, integrality=[True]*n_disc_vars)['x']
            else:
                raise Exception('Unrecognized option for discrete minimization method: bo_ops.sep_disc_minimizer = '+bo_ops.sep_disc_minimizer)
            xf_opt = min_for_cont_vars(xc_start,xclimits_num,obj_k,bo_ops,xi_opt)[1]
            x_et_k = np.append(xf_opt,xi_opt)
    return x_et_k # the x coordinates where the min in the acq func occurs

#########################################################
