#########################################################
# bayesOpt.py
def bayesOpt(func_in, params, options):
    from .classes import validate_params, validate_options
    import numpy as np
    from .viz import viz_init, viz_animate, viz_finalize, viz_show_plots
    from .utils import load_existing_csv

    assert(validate_params(params))
    ndim = len(params)
    assert(validate_options(options))

    # Set up animations
    viz_init(options,ndim)

    # Define the Gaussian Process model (AKA Kriging model)
    mixedType = False
    for i in range(ndim):
        if params[i].type != 'continuous':
            mixedType = True
            break
    from smt.surrogate_models import KRG # KPLS, KRG, KPLSK    
    if mixedType:
        func = lambda x: func_in(args_str_2_enum(x,params))
    else:
        func = func_in
    if mixedType:
        from smt.applications.mixed_integer import MixedIntegerSurrogateModel
        from smt.applications.mixed_integer import (FLOAT, ORD, ENUM)
        xtypes = []
        xlimits = [] # this is the range used by func_in (may include mixed types)
        xlimits_num = [] # this is the range which converts any categoricals to integers
        for i in range(ndim):
            if params[i].type == 'continuous':
                xtypes.append(FLOAT)
                xlimits.append([params[i].minVal, params[i].maxVal])
                xlimits_num.append([params[i].minVal, params[i].maxVal])
            elif params[i].type == 'ordered':
                xtypes.append(ORD)
                xlimits.append([params[i].minVal, params[i].maxVal])
                xlimits_num.append([params[i].minVal, params[i].maxVal])
            elif params[i].type == 'categorical':
                xtypes.append((ENUM, len(params[i].categories)))
                xlimits.append(params[i].categories)
                xlimits_num.append(list(range(len(params[i].categories))))
            else:
                raise Exception('Unrecognized type for parameter '+str(i))
        #gpr = KRG(theta0=[1e-2]*ndim,print_global = False)
        gpr = KRG(print_global = False) # XXX setting theta0 was causing gpr to not train because of a dimension issue
        gpr = MixedIntegerSurrogateModel(surrogate=gpr, xtypes=xtypes, xlimits=xlimits)
        #gpr = MixedIntegerSurrogateModel(xtypes=xtypes, xlimits=xlimits, surrogate=KRG())
    else:
        xlimits = np.zeros([ndim,2]) # the first dimension is the parameter space (ndim), the second defines the bounds (min/max) for each parameter
        for i in range(ndim):    
            xlimits[i,:] = [params[i].minVal, params[i].maxVal]
        xlimits_num = xlimits
        gpr = KRG(theta0=[1e-2]*ndim,print_global = False)

    # user can provide initial function evaluations in a .csv file
    if hasattr(options, 'existing_csv_filename'):
        [x_data_file, y_data_file] = load_existing_csv(options.existing_csv_filename,params)

    # Sample the objective function. This is the training data.
    from smt.sampling_methods import LHS
    if mixedType:
        from smt.applications.mixed_integer import MixedIntegerSamplingMethod
        if options.deterministic:
            sampling = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="maximin", random_state=1)
        else:
            sampling = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="maximin", random_state=np.random.RandomState())
    else:
        if options.deterministic:
            sampling = LHS(xlimits=xlimits, criterion='maximin', random_state=1)
        else:
            sampling = LHS(xlimits=xlimits, criterion='maximin', random_state=np.random.RandomState())
    ndoe = ndim + 1 # this is the smallest sampling allowed by the SMT algorithms
    if hasattr(options, 'initial_samples'): # check if more samples were requested
        if options.initial_samples >= ndoe:
            ndoe = options.initial_samples
        elif hasattr(options, 'existing_csv_filename') and (options.initial_samples == 0):
            if len(y_data_file) >= ndoe:
                ndoe = 0
            else:
                raise Exception('0 options.initial_samples were requested, but number of provided .csv samples is < len(params) + 1')    
        else:
            raise Exception('options.initial_samples must be >= len(params) + 1, left unspecified, or set to zero if sufficient user-defined samples are provided in a .csv')
    if ndoe > 0:
        x_data = sampling(ndoe) # 1st dimension is which sample, 2nd dimension is the parameter space
        y_data = np.zeros([ndoe,1])
        for i in range(len(x_data)):
            y_data[i] = func(x_data[i])
        
    # append the sampled data to any data provided in a .csv
    if hasattr(options, 'existing_csv_filename'):
        if  ndoe > 0: # case where some data is from .csv, some is from sampling
            x_data = np.append(x_data_file,x_data,axis=0)
            y_data = np.append(y_data_file,y_data,axis=0)
            ndoe = ndoe + len(y_data_file)
        else: # case where all data is from .csv
            x_data = x_data_file
            y_data = y_data_file
            ndoe = len(y_data_file)

    # Perform the Bayesian optimization: that is, iteratively select new sample points according to the acquisition function and update the GP with the new data
    from scipy.optimize import minimize
    from .acqFunc import getAcqFunc
    n_sample = 20 # number of samples of indicator function
    if mixedType:
        sampling_st = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="maximin", random_state=np.random.RandomState())
    else:
        sampling_st = LHS(xlimits=xlimits, criterion='maximin', random_state=np.random.RandomState())
    for k in range(options.n_iter):
        if options.deterministic:
            if mixedType:
                sampling_st = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="maximin", random_state=k) # "ese"
            else:
                sampling_st = LHS(xlimits=xlimits, criterion='maximin', random_state=k)
        f_min_k = np.min(y_data)
        gpr.set_training_values(x_data,y_data)
        gpr.train()
        obj_k = getAcqFunc(options.acqFunc,gpr,f_min_k)
        x_start = sampling_st(n_sample) # 1st dim is which init_guess, 2nd dim is which param
        # naive random sampling:
        #x_start = np.zeros([n_sample,ndim])
        #for i in range(ndim):
        #    x_start[:,i] = np.random.rand(n_sample)*(xlimits[i][1]-xlimits[i][0])+xlimits[i][0]
        opt_all = np.array([])
        for i_s in range(n_sample):
            opt_all = np.append(opt_all,minimize(lambda x: float(obj_k(x)), x_start[i_s,:], method='SLSQP', bounds=xlimits_num))
        opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]] # gets only the enties of opt_all that have 'success'=True. Note: opt_all is a dictionary, so opt_all[0]['success'] is equivalent to pt_all[0].success
        obj_success = np.array([opt_i['fun'] for opt_i in opt_success]) # create an array of the function values for all of the successful optimization points
        ind_min = np.argmin(obj_success) # which initial guess was best (led to the deepest min value)
        opt = opt_success[ind_min] # the full output for the best initial guess
        x_et_k = opt['x'] # the x value at which the min occurs
        y_et_k = func(x_et_k) # this is the objective function rather than the infill criterion
        y_data = np.atleast_2d(np.append(y_data,y_et_k)).T
        x_data = np.append(x_data,np.atleast_2d(x_et_k),axis=0)

        viz_animate(options,xlimits_num,func,gpr,x_data,y_data,f_min_k,ndoe,k)
    
    ind_best = np.argmin(y_data)
    x_opt = x_data[ind_best,:]
    y_opt = y_data[ind_best]

    viz_finalize(options,xlimits_num,func,gpr,x_data,y_data,f_min_k,ndoe,ind_best)
    viz_show_plots(options)
    
    return [x_opt, y_opt, ind_best, x_data, y_data, gpr]
#########################################################
def args_str_2_enum(x,params):
    # for mixed type functions, the user defined function may have 
    # categorical args which are strings. But SMT represents these
    # as enumerated types (integers). Need to convert enumerated
    # args to strings.
    # Also, need to cast enumerataed args to ints, which shouldn't be necessary, but is helping
    ndim = len(params)
    x = x.tolist()
    for i in range(ndim):
        if params[i].type == 'ordered':
            x[i] = int(x[i]) # the int cast is needed because SMT stores ordered variables as floats
        elif params[i].type == 'categorical':
            x[i] = params[i].categories[int(x[i])] # the int cast is needed because SMT stores enums as floats
    return x