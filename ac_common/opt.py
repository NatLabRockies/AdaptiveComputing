#########################################################
# bayes_opt.py
def bayes_opt(funcs_in, params, options):
    from .classes import validate_params, validate_options
    import numpy as np
    from .viz import viz_init, viz_animate, viz_finalize, viz_show_plots
    from .utils import read_input_data
    
    # Check the number of fidelity levels
    funcs_in = np.atleast_1d(funcs_in)
    n_fl = len(funcs_in) # number of fidelity levels
    multifidelity = False
    if n_fl != 1:
        multifidelity = True
        from smt.applications.mfk import MFK
    
    assert(validate_params(params))
    n_dim = len(params)
    assert(validate_options(options,n_fl,n_dim))

    # Set up animations
    viz_init(options,n_dim)

    # Check if there are mixed types
    mixedType = False
    for i in range(n_dim):
        if params[i].type != 'continuous':
            mixedType = True
            from smt.applications.mixed_integer import MixedIntegerSurrogateModel
            from smt.applications.mixed_integer import (FLOAT, ORD, ENUM)
            from smt.applications.mixed_integer import MixedIntegerSamplingMethod
            break

    if multifidelity: # XXX this is temporary until I support these combinations of options
        if options.n_iter != 0:
            raise Exception('multifidelity modeling is currently only supported with options.n_iter = 0.')

    funcs =[]
    for i in range(n_fl):
        if mixedType:
            funcs.append(lambda x: funcs_in[i](args_str_2_enum(x,params))) 
        else:
            funcs.append(funcs_in[i])

    # Define xlimits, the domain for the design parameters
    if mixedType:
        xtypes = []
        xlimits = [] # this is the domain for the user defined funcs_in[] (which may include mixed types)
        xlimits_num = [] # this is the domain for funcs[] which assumes the categoricals and integers have been converted to continuous types
        for i in range(n_dim):
            if params[i].type == 'continuous':
                xtypes.append(FLOAT)
                xlimits.append([params[i].min_val, params[i].max_val])
                xlimits_num.append([params[i].min_val, params[i].max_val])
            elif params[i].type == 'ordered':
                xtypes.append(ORD)
                xlimits.append([params[i].min_val, params[i].max_val])
                xlimits_num.append([params[i].min_val, params[i].max_val])
            elif params[i].type == 'categorical':
                xtypes.append((ENUM, len(params[i].categories)))
                xlimits.append(params[i].categories)
                xlimits_num.append(list(range(len(params[i].categories))))
            else:
                raise Exception('Unrecognized type for parameter '+str(i)) 
    else:
        xlimits = np.zeros([n_dim,2]) # the first dimension is the parameter space (n_dim), the second defines the bounds (min/max) for each parameter
        for i in range(n_dim):    
            xlimits[i,:] = [params[i].min_val, params[i].max_val]
        xlimits_num = xlimits
        
    # Define the Gaussian Process model (AKA the Kriging model)
    from smt.surrogate_models import KRG
    if multifidelity:
        gpr = MFK(print_global = False)
    else:
        gpr = KRG(print_global = False) 
    if mixedType:
        gpr = MixedIntegerSurrogateModel(surrogate=gpr, xtypes=xtypes, xlimits=xlimits)
    
    # The initial training data is from two sources 1) pseudo-random initial sampling and 2) read from an input file.
    # n_init is the number of initial training data for each fidelity level
    # n_init[i] is the number of pseudo-random samples (options.n_init_samp[i]) plus the number of points read from an input file (n_input[i])
    n_init = options.n_init_samp
    x_data = [] # x_data is a list of length n_fl. Each entry is an n_init x n_dim np array
    y_data = [] # y_data is a list of length n_fl. Each entry is an n_init x 1 np array
    
    # part 1) pseudo-random initial sampling
    from smt.sampling_methods import LHS
    for i in range(n_fl):
        if options.n_init_samp[i] > 0: 
            assert(options.n_init_samp[i] >= n_dim + 1) # this was established by validate_options()
            rand_state = np.random.RandomState()
            if options.deterministic:
                rand_state = i*(options.n_iter+1) # ensurses the fidelity levels all have unique seeds on all optimization iterations

            if mixedType:
                sampling = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="maximin", random_state=rand_state)
            else:
                sampling = LHS(xlimits=xlimits, criterion='maximin', random_state=rand_state)
            x_data.append(sampling(options.n_init_samp[i]))
            y_data.append(np.atleast_2d(np.zeros_like(x_data[i][:,0])).T)
            for i_s in range (len(x_data[i])):
                y_data[i][i_s] =funcs[i](x_data[i][i_s,:])
        
    # part 2) read user-provided function evaluations from a .csv file
    if hasattr(options, 'input_data_filenames'):
        [x_data_file, y_data_file] = read_input_data(options.input_data_filenames,params)
        for i in range(n_fl):
            if options.n_init_samp[i] > 0: # case where some data is from .csv, some is from sampling
                filename = options.input_data_filenames[i]
                if filename != '':
                    x_data[i] = np.append(x_data_file[i],x_data[i],axis=0)
                    y_data[i] = np.append(y_data_file[i],y_data[i],axis=0)
                    n_init[i] = n_init[i] + len(y_data_file[i])
            else: # case where all data is from .csv
                x_data.append(x_data_file[i])
                y_data.append(y_data_file[i])
                n_init[i] = len(y_data_file[i])
    
    # check that there is enough initial data
    for i in range(n_fl):
        if n_init[i] < n_dim + 1:
            raise Exception('For fidelity level ' + str(i) + ', the default value of n_init_samp (the number of pseudo-random initial samples) has been overwritten and set to zero, but there are less than n_dim + 1 samples in the input file. Make sure there are n_dim + 1 samples in the input file, do not specify n_init_samp, or set n_init_samp > n_dim + 1.')

    # Perform the Bayesian optimization: that is, iteratively select new sample points according to the acquisition function and update the GP with the new data
    from scipy.optimize import minimize
    from .acq_func import get_acq_func
    n_opt_probes = 20 # number of samples of indicator function
    i = 0 # XXX fixing bayes_opt to only use the lowest fidelity level
    rand_state = np.random.RandomState()
    for k in range(options.n_iter):
        print('Beginning AC optimization iteration ' + str(k))
        if multifidelity:
            for i_f in range(n_fl-1):
                gpr.set_training_values(x_data[i_f], y_data[i_f], name=i_f) # other fidelities are accessed with names from 0 to n_fl-2 listed in order of increasing fidelity.
            gpr.set_training_values(x_data[n_fl-1], y_data[n_fl-1]) # high-fidelity dataset without name
            gpr.train()
        else:
            gpr.set_training_values(x_data[0],y_data[0])
            gpr.train()
        f_min_k = np.min(y_data)
        obj_k = get_acq_func(options.acq_func,gpr,f_min_k)
        if options.deterministic:
            rand_state = i*(options.n_iter+1)+k+1 # ensurses the fidelity levels all have unique seeds on all optimization iterations
        if mixedType:
            sampling_opt = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="maximin", random_state=rand_state)
        else:
            sampling_opt = LHS(xlimits=xlimits, criterion='maximin', random_state=rand_state)
        x_start = sampling_opt(n_opt_probes) # 1st dim is which init_guess, 2nd dim is which param
        # naive random sampling:
        #x_start = np.zeros([n_opt_probes,n_dim])
        #for i_r in range(n_dim):
        #    x_start[:,i_r] = np.random.rand(n_opt_probes)*(xlimits[i_r][1]-xlimits[i_r][0])+xlimits[i_r][0]
        opt_all = np.array([])
        for i_s in range(n_opt_probes):
            opt_all = np.append(opt_all,minimize(lambda x: float(obj_k(x)), x_start[i_s,:], method='Powell', bounds=xlimits_num))         
        opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]] # gets only the enties of opt_all that have 'success'=True. Note: opt_all is a dictionary, so opt_all[0]['success'] is equivalent to pt_all[0].success
        obj_success = np.array([opt_i['fun'] for opt_i in opt_success]) # create an array of the function values for all of the successful optimization points
        ind_min = np.argmin(obj_success) # which initial guess was best (led to the deepest min value)
        opt = opt_success[ind_min] # the full output for the best initial guess
        x_et_k = opt['x'] # the x value at which the min occurs
        y_et_k = funcs[i](x_et_k) # this is the objective function rather than the infill criterion
        y_data[i] = np.atleast_2d(np.append(y_data,y_et_k)).T
        x_data[i] = np.append(x_data[i],np.atleast_2d(x_et_k),axis=0)

        viz_animate(options,xlimits_num,funcs,gpr,x_data,y_data,n_init,k)
    
    # Update the surrogate model with the last point added 
    if multifidelity:
        for i in range(n_fl-1):
            gpr.set_training_values(x_data[i], y_data[i], name=i) # other fidelities are accessed with names from 0 to n_fl-2 listed in order of increasing fidelity.
        gpr.set_training_values(x_data[n_fl-1], y_data[n_fl-1]) # high-fidelity dataset without name
        gpr.train()
    else:
        gpr.set_training_values(x_data[0],y_data[0])
        gpr.train()

    # Find the optimal point that has been evaluated by the high fidelity model
    ind_best = np.argmin(y_data[n_fl-1])
    # # option 1: estimate the optimum using the high fidelity model
    # x_opt = x_data[n_fl-1][ind_best,:]
    # y_opt = y_data[n_fl-1][ind_best]
    # option 2: estimate the optimum using the GPR and every sampled location with any fidelity level
    x_opt = x_data[n_fl-1][ind_best,:]
    y_opt = y_data[n_fl-1][ind_best]
    for i in range(n_fl-1):
        y_min_i = np.min(gpr.predict_values(x_data[i]))
        if  y_min_i < y_opt:
            ind_best_mf = np.argmin(gpr.predict_values(x_data[i]))
            y_opt = y_min_i
            x_opt = x_data[i][ind_best_mf,:]
    # option 3: could implement a minimization on the GPR surface though this introduces additional uncertainty

    viz_finalize(options,xlimits_num,funcs,gpr,x_data,y_data,n_init,ind_best)
    viz_show_plots(options)
    
    return [x_opt, y_opt, ind_best, x_data, y_data, gpr]
#########################################################
def args_str_2_enum(x,params):
    # for mixed type functions, the user defined function may have 
    # categorical args which are strings. But SMT represents these
    # as enumerated types (integers). Need to convert enumerated
    # args to strings.
    # Also, need to cast enumerataed args to ints, which shouldn't be necessary, but is helping
    n_dim = len(params)
    x = x.tolist()
    for i in range(n_dim):
        if params[i].type == 'ordered':
            x[i] = int(x[i]) # the int cast is needed because SMT stores ordered variables as floats
        elif params[i].type == 'categorical':
            x[i] = params[i].categories[int(x[i])] # the int cast is needed because SMT stores enums as floats
    return x
