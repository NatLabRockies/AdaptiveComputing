# bo.py
import numpy as np
#########################################################
# Perform the Bayesian optimization: that is, iteratively select new sample points according to the acquisition function and update the GP with the new data
def add_bo_samples(dataset,n_iter,surrogate,bo_ops,viz_ops):
    if bo_ops is None:
        from .classes import BoOptions
        bo_ops = BoOptions()
    if viz_ops is not None:
        from .viz import viz_init, viz_finalize, viz_show_plots
        viz_init(viz_ops,dataset.n_dim) # Set up animations

    if dataset.ds_ops.use_hero:
        dataset.wait_for_workers(surrogate) # wait is recommended so that this batch uses the most up to date information

    # Check that there are enough initial samples to conduct Bayesian optimization
    for i in range(dataset.n_fl):
        if dataset.n_samp[i] <= dataset.n_dim+i:
            raise Exception("Error on fidelity level " + str(i) + ". At least n_dim+1+fidelity_level initial samples from static sampling methods are required before Bayesian Optimization can be used to perform dynamic sampling.")

    # Multi-fidelity Bayesian optimization requires the cost to be specified for each fidelity level
    if dataset.n_fl > 1:
        if not hasattr(bo_ops, 'cpu_hrs_per_sim'):
            raise Exception('In order to conduct multi-fidelity Bayesian optimization, the user must specify bo_ops.cpu_hrs_per_sim to be a list of length n_fl.')
        if len(bo_ops.cpu_hrs_per_sim) != dataset.n_fl:
            raise Exception('In order to conduct multi-fidelity Bayesian optimization, the user must specify bo_ops.cpu_hrs_per_sim to be a list of length n_fl.')
        for hrs in bo_ops.cpu_hrs_per_sim:
            if hrs <= 0:
                raise Exception('cpu_hrs_per_sim must be > 0.')
    else:
        bo_ops.cpu_hrs_per_sim = [1]

    from .acq_func import get_acq_func, minimize_acq_func
    from smt.sampling_methods import LHS
    if dataset.mixed_type:
        from smt.applications.mixed_integer import MixedIntegerSamplingMethod
    
    dataset.sync_hero_results(surrogate,viz_ops)

    # Beginning of the bayesian optimization iterations (each iteration computes a new simulation sample)
    rand_state = np.random.RandomState()
    for k in range(n_iter):
        print('Beginning AC optimization iteration ' + str(k))

        # Retrain surrogate using x_data, y_data (so this includes unmasked data and predictions at masked data locations)
        dataset.train_on_all_data(surrogate)

        af_array = np.zeros([dataset.n_fl,1]) # acquisition function min values for each fidelity level
        x_et_k_array = np.zeros([dataset.n_fl,dataset.n_dim]) # acquisition function argmin for each fidelity level
        for i in range(dataset.n_fl):
            if dataset.ds_ops.deterministic:
                # Ensurses the fidelity levels all have unique seeds on all optimization iterations. 
                # Note: that the sampler will increment the rand_state for each sample
                rand_state = int(sum(dataset.n_samp+1) + (k+1)*(i+1)*bo_ops.n_opt_pts)

            if dataset.mixed_type:
                sampling_opt = MixedIntegerSamplingMethod(dataset.xtypes, dataset.xlimits, LHS, criterion="maximin", random_state=rand_state)
            else:
                sampling_opt = LHS(xlimits=dataset.xlimits, criterion='maximin', random_state=rand_state)
            # x_start is an array of initial guess used in the search for the minimum of the acquisition function
            x_start = sampling_opt(bo_ops.n_opt_pts) # 1st dim is which init_guess, 2nd dim is which param
            f_min_k = np.min(dataset.y_data[i]) # this is an argument needed by the EI acquisition function
            obj_k = get_acq_func(bo_ops.acq_func,surrogate,f_min_k,i) # this is the acquistion function which will be minimized
            x_et_k_array[i,:] = minimize_acq_func(dataset,obj_k,x_start,bo_ops)
            af_array[i] = obj_k(x_et_k_array[i,:]) # this is the value of the acquisition at its min (note, it is not the value of the user-defined simulation at the minimum
            # print(f'x_opt = {x_et_k_array[i,:]}, obj = {af_array[i]}.')
        # We will divide af_array by the computational cost array and then take a min. So this only applies a penalty to expensive simulations if the acquisition function
        # is guaranteed to be <0. For LCB and SBO, this is not the case. So these acqusition functions are not appropriate.
        if dataset.n_fl > 1:
            if bo_ops.acq_func == 'LCB' or bo_ops.acq_func == 'SBO':
                raise Exception('SBO and LCB are not appropriate for multifidelity problems')
                # Since it is unclear how to weight these acquisition functions by cost and
                # the lower fidelity surrogates may have errors that without correcting by the bridging functions
                # will invalidate the direct comparison of acq funcs across fidelity levels
            if np.any(af_array>0.0):
                raise Exception('The method for choosing ind_which_lvl requires the acquisition function to be defined so that it is always negative.')
        ind_which_lvl = np.argmin(np.atleast_2d(af_array)/np.atleast_2d(bo_ops.cpu_hrs_per_sim).T) # chose the fidelity level with the deeper minimum when weighted by the cost of a simulation for that fidelity level
        # Option 1: Always select the location of sample point from the high fidelity model
        x_et_k = x_et_k_array[-1,:]
        # Option 2: Select the location of sample point from the fidelity level with the deepest acq func minimum
        #x_et_k = x_et_k_array[ind_which_lvl,:]
        
        # Add the chosen sample data to the surrogate model training set and retrain using only the unmasked data
        # This computes the value of the user-defined objective function at the location where the acquisition function is minimal
        # Add sample to the Hero queue instead of running it locally immediately
        if dataset.ds_ops.use_hero:
            dataset.queue_hero_sample(ind_which_lvl,x_et_k,surrogate,viz_ops=viz_ops,frame_id=k)
        # Run the simulation on the current process (blocking)
        else:
            dataset.add_xnum_sample(ind_which_lvl,x_et_k,y_eval=None,viz_ops=viz_ops,frame_id=k,surrogate=surrogate)
        
        # The comment below is for a different way of deciding which fidelity level to use for the bayesian optimization.
        # if dataset.ds_ops.deterministic:
        #     # rand_state = i*(n_iter+1)+k+1 # ensurses the fidelity levels all have unique seeds on all optimization iterations. 
        #     # Since I am just evaluating the acquisition function on the MF surrogate, I don't need the i to change the seed for different fidelity levels
        #     rand_state = k+1 # ensurses the fidelity levels all have unique seeds on all optimization iterations
        # if dataset.mixed_type:
        #     sampling_opt = MixedIntegerSamplingMethod(dataset.xtypes, dataset.xlimits, LHS, criterion="maximin", random_state=rand_state)
        # else:
        #     sampling_opt = LHS(xlimits=dataset.xlimits, criterion='maximin', random_state=rand_state)
        # x_start = sampling_opt(bo_ops.n_opt_pts) # 1st dim is which init_guess, 2nd dim is which param
        # f_min_k = np.min(dataset.y_data)
        # obj_k = get_acq_func(bo_ops.acq_func,surrogate,f_min_k,-1)
        # x_et_k = minimize_acq_func(obj_k, x_start, dataset.ds_ops, dataset.xlimits_num)
        # if dataset.multifidelity: # decide which fidelity level to evaluate the objective on.
        #     # this is a work in progress... the algorithm is in my notes, but it has the issue that it compares variances across levels.
        #     # Should be non-dimensional since the multiplicative correction function can drastically change the variance across levels
        #     # A = [];
        #     # for i_var_check in range(dataset.n_fl-1):
        #     #     A.append(surrogate.predict_variances(x_et_k,i_var_check))
        #     # ind_which_lvl = 0
        #     # y_et_k = dataset.funcs[ind_which_lvl](x_et_k)
        #     # dataset.y_data[i] = np.atleast_2d(np.append(dataset.y_data,y_et_k)).T
        #     # dataset.x_data[i] = np.append(dataset.x_data[i],np.atleast_2d(x_et_k),axis=0)
        #     # for i_var_check in range(dataset.n_fl-1):
        #     #     if surrogate.predict_variances(x_et_k,i_var_check + 1) > A[i_var_check]
        #     ??
        # else: # always use fidelity level 0
        #     ind_which_lvl = 0
        #     y_et_k = dataset.funcs[ind_which_lvl](x_et_k)
        #     dataset.y_data[i] = np.atleast_2d(np.append(dataset.y_data,y_et_k)).T
        #     dataset.x_data[i] = np.append(dataset.x_data[i],np.atleast_2d(x_et_k),axis=0)

    if viz_ops is not None:
        viz_finalize(dataset,surrogate,viz_ops,n_iter-1)
        viz_show_plots(viz_ops,n_frames=n_iter)

#########################################################
# Find the optimal point that has been evaluated by the high fidelity model
def find_min(dataset,surrogate):
    if dataset.ds_ops.use_hero:
        # dataset.wait_for_workers(surrogate)
        total_in_queue = np.sum([np.sum(arr) for arr in dataset.hero_todo])
        if total_in_queue > 0:
            print(f'Warning: {total_in_queue} Hero tasks are incomplete. Consider calling wait_for_workers(surrogate) before find_min().')
    
    # # option 1: estimate the optimum using the high fidelity model
    # ind_best = np.argmin(dataset.y_data[-1])
    # x_opt = dataset.x_data[-1][ind_best,:]
    # y_opt = dataset.y_data[-1][ind_best]
    # option 2: estimate the optimum using the highest fidelity surrogate and every sampled location with any fidelity level
    # Begin with the x_data where the highest fidelity model has been evaluated
    ind_best = np.argmin(surrogate.predict_values(dataset.x_data[-1]))
    x_opt = np.atleast_2d(dataset.x_data[-1][ind_best,:])
    y_opt = surrogate.predict_values(x_opt,-1)
    x_opt = x_opt[0]
    y_opt = y_opt[0]
    opt_is_masked = False
    if not dataset.unmasked_data[-1][ind_best]:
        opt_is_masked = True
    for i in range(dataset.n_fl-1): # check the highest fidelity surrogate model evaluated at the points where lower fidelity sims have been conducted
        y_min_i = np.min(surrogate.predict_values(dataset.x_data[i],-1))
        if  y_min_i < y_opt:
            ind_best_mf = np.argmin(surrogate.predict_values(dataset.x_data[i],-1))
            y_opt = y_min_i
            x_opt = dataset.x_data[i][ind_best_mf,:]
            if not dataset.unmasked_data[i][ind_best_mf]:
                opt_is_masked = True
            else:
                opt_is_masked = False
    if opt_is_masked:
        print('Warning: the minimum value of the surrogate is in a region of masked data (the data is a placeholder for'
              +' an incomplete Hero simulation or the simulation returned NaN or an out of allowable bounds value).')
        print(f'This masked minimum is x_opt={x_opt}, y_opt={y_opt}')
        print('Recomputing and returning minimum using only the unmasked data.')
        ind_best = np.argmin(surrogate.predict_values(dataset.x_data[-1][dataset.unmasked_data[-1].flatten()],-1)) # this index is of only the unmasked data
        orig_indices_of_unmasked = np.where(dataset.unmasked_data[-1].flatten())[0]
        ind_best = orig_indices_of_unmasked[ind_best] # this index is global index for masked and unmasked data
        x_opt = np.atleast_2d(dataset.x_data[-1][ind_best,:])
        y_opt = surrogate.predict_values(x_opt,-1)
        x_opt = x_opt[0]
        y_opt = y_opt[0]
        assert(dataset.unmasked_data[-1][ind_best])
        for i in range(dataset.n_fl-1):
            y_min_i = np.min(surrogate.predict_values(dataset.x_data[i][dataset.unmasked_data[i].flatten()],-1))
            if  y_min_i < y_opt:
                ind_best_mf = np.argmin(surrogate.predict_values(dataset.x_data[i][dataset.unmasked_data[i].flatten()],-1))
                orig_indices_of_unmasked = np.where(dataset.unmasked_data[i].flatten())[0]
                ind_best_mf = orig_indices_of_unmasked[ind_best_mf] # this index is global index for masked and unmasked data
                y_opt = y_min_i
                x_opt = dataset.x_data[i][ind_best_mf,:]
                assert(dataset.unmasked_data[i][ind_best_mf])
    # option 3: could implement a minimization on the surrogate surface though this introduces additional uncertainty, since could
    # return a point where no simulation has ever been computed yet.

    return [x_opt, y_opt]

#########################################################
# Find the optimal point that has been evaluated by the high fidelity model
def find_max(dataset,surrogate):
    if dataset.ds_ops.use_hero:
        # dataset.wait_for_workers(surrogate)
        total_in_queue = np.sum([np.sum(arr) for arr in dataset.hero_todo])
        if total_in_queue > 0:
            print(f'Warning: {total_in_queue} Hero tasks are incomplete. Consider calling wait_for_workers(surrogate) before find_max().')
    
    # # option 1: estimate the optimum using the high fidelity model
    # ind_best = np.argmax(dataset.y_data[-1])
    # x_opt = dataset.x_data[-1][ind_best,:]
    # y_opt = dataset.y_data[-1][ind_best]
    # option 2: estimate the optimum using the highest fidelity surrogate and every sampled location with any fidelity level
    # Begin with the x_data where the highest fidelity model has been evaluated
    ind_best = np.argmax(surrogate.predict_values(dataset.x_data[-1],-1))
    x_opt = np.atleast_2d(dataset.x_data[-1][ind_best,:])
    y_opt = surrogate.predict_values(x_opt,-1)
    x_opt = x_opt[0]
    y_opt = y_opt[0]
    opt_is_masked = False
    if not dataset.unmasked_data[-1][ind_best]:
        opt_is_masked = True
    for i in range(dataset.n_fl-1):
        y_max_i = np.max(surrogate.predict_values(dataset.x_data[i],-1))
        if  y_max_i < y_opt:
            ind_best_mf = np.argmax(surrogate.predict_values(dataset.x_data[i],-1))
            y_opt = y_max_i
            x_opt = dataset.x_data[i][ind_best_mf,:]
            if not dataset.unmasked_data[i][ind_best_mf]:
                opt_is_masked = True
            else:
                opt_is_masked = False
    if opt_is_masked:
        print('Warning: the maximum value of the surrogate is in a region of masked data (the data is a placeholder for'
              +' an incomplete Hero simulation or the simulation returned NaN or an out of allowable bounds value).')
        print(f'This masked maximum is x_opt={x_opt}, y_opt={y_opt}')
        print('Recomputing and returning maximum using only the unmasked data.')
        ind_best = np.argmax(surrogate.predict_values(dataset.x_data[-1][dataset.unmasked_data[-1].flatten()],-1)) # this index is of only the unmasked data
        orig_indices_of_unmasked = np.where(dataset.unmasked_data[-1].flatten())[0]
        ind_best = orig_indices_of_unmasked[ind_best] # this index is global index for masked and unmasked data
        x_opt = np.atleast_2d(dataset.x_data[-1][ind_best,:])
        y_opt = surrogate.predict_values(x_opt,-1)
        x_opt = x_opt[0]
        y_opt = y_opt[0]
        assert(dataset.unmasked_data[-1][ind_best])
        for i in range(dataset.n_fl-1):
            y_max_i = np.max(surrogate.predict_values(dataset.x_data[i][dataset.unmasked_data[i].flatten()],-1))
            if  y_max_i < y_opt:
                ind_best_mf = np.argmax(surrogate.predict_values(dataset.x_data[i][dataset.unmasked_data[i].flatten()],-1))
                orig_indices_of_unmasked = np.where(dataset.unmasked_data[i].flatten())[0]
                ind_best_mf = orig_indices_of_unmasked[ind_best_mf] # this index is global index for masked and unmasked data
                y_opt = y_max_i
                x_opt = dataset.x_data[i][ind_best_mf,:]
                assert(dataset.unmasked_data[i][ind_best_mf])
    # option 3: could implement a maximization on the surrogate surface though this introduces additional uncertainty

    return [x_opt, y_opt]