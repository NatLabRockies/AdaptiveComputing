#########################################################
# bayesOpt.py
def bayesOpt(func, params, options):
    import numpy as np
    import viz as viz

    ndim = len(params)
    # the first dimension is the parameter space (ndim), the second min and max values
    xlimits = np.array([[params[0].minVal, params[0].maxVal]])
    for i in range(1, ndim):
        xlimits = np.append(xlimits, [[params[i].minVal, params[i].maxVal]], axis=0)

    # Set up animations
    viz.init(options,ndim)

    # Sample the objective function. This is the training data.
    from smt.sampling_methods import LHS
    #sampling = LHS(xlimits=xlimits, criterion='maximin', random_state=1) # set random seed for debugging
    sampling = LHS(xlimits=xlimits, criterion='maximin', random_state=np.random.RandomState())
    ndoe = ndim + 1
    if hasattr(options, 'initial_samples'):
        if options.initial_samples >= ndoe:
            ndoe = options.initial_samples
        else:
            raise Exception('options.initial_samples must be >= len(params) + 1 or left unspecified')
    x_data = sampling(ndoe) # 1st dimension is which sample, 2nd dimension is the parameter space
    y_data = np.zeros([ndoe,1])
    for i in range(len(x_data)):
        y_data[i] = func(x_data[i])

    # Define the Gaussian Process model (AKA Kriging model)
    from smt.surrogate_models import KPLS, KRG, KPLSK
    # The variable 'theta0' is a list of length ndim.
    gpr = KRG(theta0=[1e-2]*ndim,print_global = False) #, corr='squar_exp'

    #"""
    # Iteratively select new sample points and update the GP according to the acquisition function
    from scipy.optimize import minimize
    from scipy.optimize import fmin
    from acqFunc import EI, SBO, LCB, getAcqFunc
    n_sample = 20 # number of samples of indicator function
    sampling_st = LHS(xlimits=xlimits, criterion='maximin', random_state=np.random.RandomState())
    for k in range(options.n_iter):
        #sampling_st = LHS(xlimits=xlimits, criterion='maximin', random_state=k) # set the random seed for debugging
        f_min_k = np.min(y_data)
        gpr.set_training_values(x_data,y_data)
        gpr.train()
        obj_k = getAcqFunc(options.acqFunc,gpr,f_min_k)
        # create list of initial guesses for the minimization of the acqFunc.
        # must be in loop to get new random samples
        # 1st dim is which init_guess, 2nd dim is which param  
        # Latin Hypercube sampling:
        x_start = sampling_st(n_sample)
        # naive random sampling:
        #x_start = np.zeros([n_sample,ndim])
        #for i in range(ndim):
        #    x_start[:,i] = np.random.rand(n_sample)*(xlimits[i][1]-xlimits[i][0])+xlimits[i][0]
        opt_all = np.array([])
        for i_s in range(n_sample):
            opt_all = np.append(opt_all,minimize(lambda x: float(obj_k(x)), x_start[i_s,:], method='SLSQP', bounds=xlimits))
        opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]] # gets only the enties of opt_all that have 'success'=True. Note: opt_all is a dictionary, so opt_all[0]['success'] is equivalent to pt_all[0].success
        obj_success = np.array([opt_i['fun'] for opt_i in opt_success]) # create an array of the function values for all of the successful optimization points
        ind_min = np.argmin(obj_success) # which initial guess was best (led to the deepest min value)
        opt = opt_success[ind_min] # the full output for the best initial guess
        x_et_k = opt['x'] # the x value at which the min occurs
        y_et_k = func(x_et_k) # this is the objective function rather than the infill criterion
        y_data = np.atleast_2d(np.append(y_data,y_et_k)).T
        x_data = np.append(x_data,np.atleast_2d(x_et_k),axis=0)

        viz.animate(options,xlimits,func,gpr,x_data,y_data,f_min_k,ndoe,k)
    
    ind_best = np.argmin(y_data)
    x_opt = x_data[ind_best,:]
    y_opt = y_data[ind_best]

    viz.finalize(options,xlimits,func,gpr,x_data,y_data,f_min_k,ndoe,ind_best)
    
    return [x_opt, y_opt, ind_best, x_data, y_data, gpr]
