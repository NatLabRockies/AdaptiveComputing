#########################################################
# opt.py
def opt(simulations, params, options):
    from .classes import validate_params, validate_options
    import numpy as np
    from .viz import viz_init, viz_animate, viz_finalize, viz_show_plots
    from .utils import read_input_data, write_output_data
    
    # Check the number of fidelity levels
    simulations = np.atleast_1d(simulations)
    n_fl = len(simulations) # number of fidelity levels
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

    funcs =[]
    for i in range(n_fl):
        if mixedType:
            funcs.append(lambda x, i=i: simulations[i](args_str_2_enum(x,params))) 
        else:
            funcs.append(simulations[i])

    # Define xlimits, the domain for the design parameters
    if mixedType:
        xtypes = []
        xlimits = [] # this is the domain for the user defined simulations[] (which may include mixed types)
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
    
    # The initial training data is from two sources 1) pseudo-random initial sampling and 2) read from an input file.
    # n_init[i] will be the number of pseudo-random samples (options.n_init_samp[i]) plus the number of points read from an input file
    n_init = options.n_init_samp
    x_data = [np.empty([0,n_dim])]*n_fl # x_data is a list of length n_fl. Each entry will be an n_init x n_dim np array
    y_data = [np.empty([0,1])]*n_fl # y_data is a list of length n_fl. Each entry will be an n_init x 1 np array
    unmasked_data = [np.empty([0,1])]*n_fl # unmasked_data is a list of length n_fl. Each entry will be an n_init x 1 np array

    # Part 1) Pseudo-random initial sampling
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
            x_data[i] = np.append(x_data[i],sampling(options.n_init_samp[i]),axis=0)
            y_data[i] = np.append(y_data[i],np.atleast_2d(np.zeros_like(x_data[i][:,0])).T,axis=0)
            for i_s in range (len(x_data[i])):
                y_data[i][i_s] = funcs[i](x_data[i][i_s,:])
        
    # Part 2) Read user-provided function evaluations from a .csv file
    if hasattr(options, 'input_data_filenames'):
        [x_data_file, y_data_file] = read_input_data(options.input_data_filenames,params,funcs)
        for i in range(n_fl):
            filename = options.input_data_filenames[i]
            if filename != '':
                x_data[i] = np.append(x_data_file[i],x_data[i],axis=0)
                y_data[i] = np.append(y_data_file[i],y_data[i],axis=0)
                n_init[i] = n_init[i] + len(y_data_file[i])
    
    # Part 3) At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
    if options.perform_lower_sims:
        for i_fl in range(n_fl-1,0,-1): # for all levels greater than zero, counting backwards
            for j_d_upper in range(n_init[i_fl]): # for all data points in this level
                # check if this point exists on the next lowest level
                pt_exists = False 
                for j_d_lower in range(n_init[i_fl-1]): # for all data in the level below
                    if np.array_equal(x_data[i_fl][j_d_upper,:],x_data[i_fl-1][j_d_lower,:]):
                        pt_exists = True
                if not pt_exists: # add it to the lower level
                    y_et_k = funcs[i_fl-1](x_data[i_fl][j_d_upper,:])
                    y_data[i_fl-1] = np.atleast_2d(np.append(y_data[i_fl-1],y_et_k)).T
                    x_data[i_fl-1] = np.append(x_data[i_fl-1],np.atleast_2d(x_data[i_fl][j_d_upper,:]),axis=0)
                    n_init[i_fl-1] = n_init[i_fl-1] + 1

    # Check that there is enough initial data
    for i in range(n_fl):
        if n_init[i] < n_dim + 1:
            raise Exception('For fidelity level ' + str(i) + ', the default value of n_init_samp (the number of pseudo-random initial samples) has been overwritten and set to zero, but there are < n_dim + 1 samples in the input file. Make sure there are n_dim + 1 samples in the input file, do not specify n_init_samp, or set n_init_samp >= n_dim + 1.')
        if i > 0:
            if n_init[i] >= n_init[i-1]:
                print('Warning: the number of initial data for fidelity level '+str(i)+' is '+str(n_init[i])+', which is >= '+str(n_init[i-1])+', the number of initial data for (lower) fidelity level '+str(i-1)+'. This can lead to poor performance and is typically not an efficient way to initialize the Bayesian Optimization. This includes data read from files, LHS sampling, and perform_lower_sims if active.')

    # Mask data that is NaN or outside allowable bounds
    for ind_which_lvl in range(n_fl):
        unmasked_data[ind_which_lvl] = np.full([len(y_data[ind_which_lvl]),1], True)
        for i in range(len(y_data[ind_which_lvl])):
            if np.isnan(y_data[ind_which_lvl][i]):
                if options.mask_nans:
                    unmasked_data[ind_which_lvl][i] = False
                    print('NaN point found: y_data['+str(ind_which_lvl)+']['+str(i)+'] = '+str(y_data[ind_which_lvl][i])+'. Masking this point.')
                else:
                    raise Exception('NaN returned by user-defined simulation. Consider setting options.mask_nans=True to ignore NaNs.')
            else:
                oob = False
                if hasattr(options, 'lbound_inclusive'):
                    if y_data[ind_which_lvl][i]<options.lbound_inclusive:
                        oob = True
                if hasattr(options, 'ubound_inclusive'):
                    if y_data[ind_which_lvl][i]>options.ubound_inclusive:
                        oob = True
                if hasattr(options, 'lbound_exclusive'):
                    if y_data[ind_which_lvl][i]<=options.lbound_exclusive:
                        oob = True
                if hasattr(options, 'ubound_exclusive'):
                    if y_data[ind_which_lvl][i]>=options.ubound_exclusive:
                        oob = True
                if oob:
                    if options.mask_oob_values:
                        unmasked_data[ind_which_lvl][i] = False
                        print('y_data['+str(ind_which_lvl)+']['+str(i)+'] = '+str(y_data[ind_which_lvl][i])+' is out of user-specified allowable bounds. Masking this point.')
                    else:
                        raise Exception('Allowable bounds violated by return value from user-defined simulation. Consider setting options.mask_oob_values=True to ignore such values.')

    # Define the Gaussian Process model (AKA the Kriging model)
    from smt.surrogate_models import KRG
    gprs = []
    for i_fl in range(n_fl): # create at hierarchy of gprs
        if multifidelity:
            gprs.append(MFK(print_global = False))
        else:
            gprs.append(KRG(print_global = False)) 
        if mixedType:
            gprs[i_fl] = MixedIntegerSurrogateModel(surrogate=gprs[i_fl], xtypes=xtypes, xlimits=xlimits)
        
    # Perform the Bayesian optimization: that is, iteratively select new sample points according to the acquisition function and update the GP with the new data
    from .acq_func import get_acq_func, minimize_acq_func
    
    rand_state = np.random.RandomState()
    for k in range(options.n_iter):
        print('Beginning AC optimization iteration ' + str(k))

        # First, train GPRs using only the x_data[unmasked], y_data[unmasked]
        for i_fl in range(n_fl):
            for ii_fl in range(i_fl):
                gprs[i_fl].set_training_values(np.atleast_2d(x_data[ii_fl][unmasked_data[ii_fl]]).T, np.atleast_2d(y_data[ii_fl][unmasked_data[ii_fl]]).T, name=ii_fl) # other fidelities are accessed with names from 0 to n_fl-2 listed in order of increasing fidelity.
            gprs[i_fl].set_training_values(np.atleast_2d(x_data[i_fl][unmasked_data[i_fl]]).T, np.atleast_2d(y_data[i_fl][unmasked_data[i_fl]]).T) # highest-fidelity dataset does not get a name
            gprs[i_fl].train()

        # Predict the mean value at all x_data[masked] values using GPR_unmasked (and store these in y_data[masked])
        for i_fl in range(n_fl):
            for i in range(len(y_data[i_fl])):
                if not unmasked_data[i_fl][i]:
                    y_data[i_fl][i] = gprs[i_fl].predict_values(x_data[i_fl][i])

        # Retrain GPR using x_data, y_data (so this includes unmasked data and predictions at masked data locations)
        for i_fl in range(n_fl):
            for ii_fl in range(i_fl):
                gprs[i_fl].set_training_values(x_data[ii_fl], y_data[ii_fl], name=ii_fl) # other fidelities are accessed with names from 0 to n_fl-2 listed in order of increasing fidelity.
            gprs[i_fl].set_training_values(x_data[i_fl], y_data[i_fl]) # highest-fidelity dataset does not get a name
            gprs[i_fl].train()

        #af_array = []
        af_array = np.zeros([n_fl,1]) # acquisition function values for each fidelity level
        for i in range(n_fl):
            if options.deterministic:
                rand_state = i*(options.n_iter+1)+k+1 # ensurses the fidelity levels all have unique seeds on all optimization iterations. 
                # Since I am just evaluating the acquisition function on the MF GPR, I don't need the i to change the seed for different fidelity levels
                # rand_state = k+1 # ensurses the fidelity levels all have unique seeds on all optimization iterations
            if mixedType:
                sampling_opt = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="maximin", random_state=rand_state)
            else:
                sampling_opt = LHS(xlimits=xlimits, criterion='maximin', random_state=rand_state)
            x_start = sampling_opt(options.n_opt_pts) # 1st dim is which init_guess, 2nd dim is which param
            f_min_k = np.min(y_data[i])
            obj_k = get_acq_func(options.acq_func,gprs[i],f_min_k)
            x_et_k = minimize_acq_func(obj_k, x_start, options, xlimits_num)
            #af_array.append(obj_k(x_et_k)[0])
            af_array[i] = obj_k(x_et_k)
        ind_which_lvl = np.argmin(np.atleast_2d(af_array)/np.atleast_2d(options.cpu_hrs_per_sim).T)
        y_et_k = funcs[ind_which_lvl](x_et_k)
        y_data[ind_which_lvl] = np.atleast_2d(np.append(y_data[ind_which_lvl],y_et_k)).T
        unmasked_data[ind_which_lvl] = np.atleast_2d(np.append(unmasked_data[ind_which_lvl],True)).T
        x_data[ind_which_lvl] = np.append(x_data[ind_which_lvl],np.atleast_2d(x_et_k),axis=0)
        # At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
        if options.perform_lower_sims:
            for i_fl in range(ind_which_lvl):
                pt_exists = False 
                for j_d_lower in range(n_init[i_fl]): # for all data in the level below
                    if np.array_equal(x_data[ind_which_lvl][-1,:],x_data[i_fl][j_d_lower,:]):
                        pt_exists = True
                if not pt_exists: # run the sim at the current level
                    y_et_k = funcs[i_fl](x_et_k)
                    y_data[i_fl] = np.atleast_2d(np.append(y_data[i_fl],y_et_k)).T
                    unmasked_data[i_fl] = np.atleast_2d(np.append(unmasked_data[i_fl],False)).T
                    x_data[i_fl] = np.append(x_data[i_fl],np.atleast_2d(x_et_k),axis=0)
        
        # Check for NaNs and out of bounds y_data
        if np.isnan(y_data[ind_which_lvl][-1]):
            if options.mask_nans:
                unmasked_data[ind_which_lvl][-1] = False
                print('NaN point found: y_data['+str(ind_which_lvl)+']['+str(len(unmasked_data[ind_which_lvl])-1)+'] = '+str(y_data[ind_which_lvl][-1])+'. Masking this point.')
            else:
                raise Exception('NaN returned by user-defined simulation. Consider setting options.mask_nans=True to ignore NaNs.')
        else:
            oob = False
            if hasattr(options, 'lbound_inclusive'):
                if y_data[ind_which_lvl][-1]<options.lbound_inclusive:
                    oob = True
            if hasattr(options, 'ubound_inclusive'):
                if y_data[ind_which_lvl][-1]>options.ubound_inclusive:
                    oob = True
            if hasattr(options, 'lbound_exclusive'):
                if y_data[ind_which_lvl][-1]<=options.lbound_exclusive:
                    oob = True
            if hasattr(options, 'ubound_exclusive'):
                if y_data[ind_which_lvl][-1]>=options.ubound_exclusive:
                    oob = True
            if oob:
                if options.mask_oob_values:
                    unmasked_data[ind_which_lvl][-1] = False
                    print('y_data['+str(ind_which_lvl)+']['+str(len(unmasked_data[ind_which_lvl])-1)+'] = '+str(y_data[ind_which_lvl][-1])+' is out of user-specified allowable bounds. Masking this point.')
                else:
                    raise Exception('Allowable bounds violated by return value from user-defined simulation. Consider setting options.mask_oob_values=True to ignore such values.')
        
        # if options.deterministic:
        #     # rand_state = i*(options.n_iter+1)+k+1 # ensurses the fidelity levels all have unique seeds on all optimization iterations. 
        #     # Since I am just evaluating the acquisition function on the MF GPR, I don't need the i to change the seed for different fidelity levels
        #     rand_state = k+1 # ensurses the fidelity levels all have unique seeds on all optimization iterations
        # if mixedType:
        #     sampling_opt = MixedIntegerSamplingMethod(xtypes, xlimits, LHS, criterion="maximin", random_state=rand_state)
        # else:
        #     sampling_opt = LHS(xlimits=xlimits, criterion='maximin', random_state=rand_state)
        # x_start = sampling_opt(options.n_opt_pts) # 1st dim is which init_guess, 2nd dim is which param
        # f_min_k = np.min(y_data)
        # obj_k = get_acq_func(options.acq_func,gprs[-1],f_min_k)
        # x_et_k = minimize_acq_func(obj_k, x_start, options, xlimits_num)
        # if multifidelity: # decide which fidelity level to evaluate the objective on.
        #     # this is a work in progress... the algorithm is in my notes, but it has the issue that it compares variances across levels. Should be non-dimensional since the multiplicative correction function can drastically change the variance across levels
        #     # A = [];
        #     # for i_var_check in range(n_fl-1):
        #     #     A.append(gprs[i_var_check].predict_variances(x_et_k))
        #     # ind_which_lvl = 0
        #     # y_et_k = funcs[ind_which_lvl](x_et_k)
        #     # y_data[i] = np.atleast_2d(np.append(y_data,y_et_k)).T
        #     # x_data[i] = np.append(x_data[i],np.atleast_2d(x_et_k),axis=0)
        #     # for i_var_check in range(n_fl-1):
        #     #     if gprs[i_var_check + 1].predict_variances(x_et_k) > A[i_var_check]
        #     ??
        # else: # always use fidelity level 0
        #     ind_which_lvl = 0
        #     y_et_k = funcs[ind_which_lvl](x_et_k)
        #     y_data[i] = np.atleast_2d(np.append(y_data,y_et_k)).T
        #     x_data[i] = np.append(x_data[i],np.atleast_2d(x_et_k),axis=0)

        viz_animate(options,xlimits_num,funcs,gprs[-1],x_data,y_data,n_init,k)
    
    # Update the surrogate model with the last point added. This training only uses unmasked data.
    # This training only uses unmasked data
    i_fl = n_fl-1 # Only update the highest fidelity level since that is the only one that is outputted
    for ii_fl in range(i_fl):
        gprs[i_fl].set_training_values(np.atleast_2d(x_data[ii_fl][unmasked_data[ii_fl]]).T, np.atleast_2d(y_data[ii_fl][unmasked_data[ii_fl]]).T, name=ii_fl) # other fidelities are accessed with names from 0 to n_fl-2 listed in order of increasing fidelity.
    gprs[i_fl].set_training_values(np.atleast_2d(x_data[i_fl][unmasked_data[i_fl]]).T, np.atleast_2d(y_data[i_fl][unmasked_data[i_fl]]).T) # highest-fidelity dataset does not get a name
    gprs[i_fl].train()

    # Find the optimal point that has been evaluated by the high fidelity model
    ind_best = np.argmin(y_data[n_fl-1])
    # # option 1: estimate the optimum using the high fidelity model
    # x_opt = x_data[n_fl-1][ind_best,:]
    # y_opt = y_data[n_fl-1][ind_best]
    # option 2: estimate the optimum using the highest fidelity GPR and every sampled location with any fidelity level
    x_opt = x_data[n_fl-1][ind_best,:]
    y_opt = y_data[n_fl-1][ind_best]
    opt_is_masked = False
    if not unmasked_data[n_fl-1][ind_best]:
        opt_is_masked = True
    for i in range(n_fl-1):
        y_min_i = np.min(gprs[-1].predict_values(x_data[i]))
        if  y_min_i < y_opt:
            ind_best_mf = np.argmin(gprs[-1].predict_values(x_data[i]))
            y_opt = y_min_i
            x_opt = x_data[i][ind_best_mf,:]
            if not unmasked_data[i][ind_best_mf]:
                opt_is_masked = True
            else:
                opt_is_masked = False
    if opt_is_masked:
        print('Warning: the minimum value returned is in a region of masked data (the simulation returned NaN or out of allowable bounds values), so there is significant uncertainty in this solution.')
    # option 3: could implement a minimization on the GPR surface though this introduces additional uncertainty

    if hasattr(options, 'output_data_filenames'):
        write_output_data(options.output_data_filenames,params,x_data,y_data)

    viz_finalize(options,xlimits_num,funcs,gprs[-1],x_data,y_data,n_init,ind_best)
    viz_show_plots(options)
    
    return [x_opt, y_opt, ind_best, x_data, y_data, gprs[-1]]
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
