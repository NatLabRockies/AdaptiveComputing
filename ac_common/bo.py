# bo.py
import numpy as np
#########################################################
# Perform the Bayesian optimization: that is, iteratively select new sample points according to the acquisition function and update the GP with the new data
def add_bo_samples(model,n_iter,bo_ops,viz_ops):
    if bo_ops is None:
        from .classes import BoOptions
        bo_ops = BoOptions()
    if viz_ops is not None:
        from .viz import viz_init, viz_finalize, viz_show_plots
        viz_init(viz_ops,model.n_dim) # Set up animations

    # Check that there are enough initial samples to conduct Bayesian optimization
    for i in range(model.n_fl):
        if model.n_samp[i] <= model.n_dim+i:
            raise Exception("Error on fidelity level " + str(i) + ". At least n_dim+1+fidelity_level initial samples from static sampling methods are required before Bayesian Optimization can be used to perform dynamic sampling.")

    # Multi-fidelity Bayesian optimization requires the cost to be specified for each fidelity level
    if model.n_fl > 1:
        if not hasattr(bo_ops, 'cpu_hrs_per_sim'):
            raise Exception('In order to conduct multi-fidelity Bayesian optimization, the user must specify bo_ops.cpu_hrs_per_sim to be a list of length n_fl.')
        if len(bo_ops.cpu_hrs_per_sim) != model.n_fl:
            raise Exception('In order to conduct multi-fidelity Bayesian optimization, the user must specify bo_ops.cpu_hrs_per_sim to be a list of length n_fl.')
        for hrs in bo_ops.cpu_hrs_per_sim:
            if hrs <= 0:
                raise Exception('cpu_hrs_per_sim must be > 0.')
    else:
        bo_ops.cpu_hrs_per_sim = [1]

    from .acq_func import get_acq_func, minimize_acq_func
    from smt.sampling_methods import LHS
    if model.mixed_type:
        from smt.applications.mixed_integer import MixedIntegerSamplingMethod

    # Train GPRs using only the x_data[unmasked], y_data[unmasked]
    model.train_on_unmasked_data()

    # Beginning of the bayesian optimization iterations (each iteration computes a new simulation sample)
    rand_state = np.random.RandomState()
    for k in range(n_iter):
        print('Beginning AC optimization iteration ' + str(k))

        # Predict the mean value at all x_data[masked] values using GPR_unmasked (and store these in y_data[masked])
        for i_fl in range(model.n_fl):
            for i in range(len(model.y_data[i_fl])):
                if not model.unmasked_data[i_fl][i]:
                    model.y_data[i_fl][i] = model.gprs[i_fl].predict_values(np.atleast_2d(model.x_data[i_fl][i]))[0]

        # Retrain GPR using x_data, y_data (so this includes unmasked data and predictions at masked data locations)
        model.train_on_all_data()

        af_array = np.zeros([model.n_fl,1]) # acquisition function min values for each fidelity level
        x_et_k_array = np.zeros([model.n_fl,model.n_dim]) # acquisition function argmin for each fidelity level
        for i in range(model.n_fl):
            if model.mod_ops.deterministic:
                # Ensurses the fidelity levels all have unique seeds on all optimization iterations. 
                # Note: that the sampler will increment the rand_state for each sample
                rand_state = int(sum(model.n_samp+1) + (k+1)*(i+1)*bo_ops.n_opt_pts)

            if model.mixed_type:
                sampling_opt = MixedIntegerSamplingMethod(model.xtypes, model.xlimits, LHS, criterion="maximin", random_state=rand_state)
            else:
                sampling_opt = LHS(xlimits=model.xlimits, criterion='maximin', random_state=rand_state)
            # x_start is an array of initial guess used in the search for the minimum of the acquisition function
            x_start = sampling_opt(bo_ops.n_opt_pts) # 1st dim is which init_guess, 2nd dim is which param
            f_min_k = np.min(model.y_data[i]) # this is an argument needed by the EI acquisition function
            obj_k = get_acq_func(bo_ops.acq_func,model.gprs[i],f_min_k) # this is the acquistion function which will be minimized
            x_et_k_array[i,:] = minimize_acq_func(obj_k, x_start, bo_ops, model.xlimits_num)
            af_array[i] = obj_k(x_et_k_array[i,:]) # this is the value of the acquisition at its min (note, it is not the value of the user-defined simulation at the minimum
        ind_which_lvl = np.argmin(np.atleast_2d(af_array)/np.atleast_2d(bo_ops.cpu_hrs_per_sim).T) # chose the fidelity level with the deeper minimum when weighted by the cost of a simulation for that fidelity level
        # Option 1: Always select the location of sample point from the high fidelity model
        x_et_k = x_et_k_array[-1,:]
        # Option 2: Select the location of sample point from the fidelity level with the deepest acq func minimum
        #x_et_k = x_et_k_array[ind_which_lvl,:]
        
        # Add the chosen sample data to the surrogate model training set and retrain using only the unmasked data
        # This computes the value of the user-defined objective function at the location where the acquisition function is minimal
        model.add_xnum_sample(ind_which_lvl,x_et_k,y_eval=None,viz_ops=viz_ops,frame_id=k)
        
        # The comment below is for a different way of deciding which fidelity level to use for the bayesian optimization.
        # if model.mod_ops.deterministic:
        #     # rand_state = i*(n_iter+1)+k+1 # ensurses the fidelity levels all have unique seeds on all optimization iterations. 
        #     # Since I am just evaluating the acquisition function on the MF GPR, I don't need the i to change the seed for different fidelity levels
        #     rand_state = k+1 # ensurses the fidelity levels all have unique seeds on all optimization iterations
        # if model.mixed_type:
        #     sampling_opt = MixedIntegerSamplingMethod(model.xtypes, model.xlimits, LHS, criterion="maximin", random_state=rand_state)
        # else:
        #     sampling_opt = LHS(xlimits=model.xlimits, criterion='maximin', random_state=rand_state)
        # x_start = sampling_opt(bo_ops.n_opt_pts) # 1st dim is which init_guess, 2nd dim is which param
        # f_min_k = np.min(model.y_data)
        # obj_k = get_acq_func(bo_ops.acq_func,model.gprs[-1],f_min_k)
        # x_et_k = minimize_acq_func(obj_k, x_start, model.mod_ops, model.xlimits_num)
        # if model.multifidelity: # decide which fidelity level to evaluate the objective on.
        #     # this is a work in progress... the algorithm is in my notes, but it has the issue that it compares variances across levels.
        #     # Should be non-dimensional since the multiplicative correction function can drastically change the variance across levels
        #     # A = [];
        #     # for i_var_check in range(model.n_fl-1):
        #     #     A.append(model.gprs[i_var_check].predict_variances(x_et_k))
        #     # ind_which_lvl = 0
        #     # y_et_k = model.funcs[ind_which_lvl](x_et_k)
        #     # model.y_data[i] = np.atleast_2d(np.append(model.y_data,y_et_k)).T
        #     # model.x_data[i] = np.append(model.x_data[i],np.atleast_2d(x_et_k),axis=0)
        #     # for i_var_check in range(model.n_fl-1):
        #     #     if model.gprs[i_var_check + 1].predict_variances(x_et_k) > A[i_var_check]
        #     ??
        # else: # always use fidelity level 0
        #     ind_which_lvl = 0
        #     y_et_k = model.funcs[ind_which_lvl](x_et_k)
        #     model.y_data[i] = np.atleast_2d(np.append(model.y_data,y_et_k)).T
        #     model.x_data[i] = np.append(model.x_data[i],np.atleast_2d(x_et_k),axis=0)

    if viz_ops is not None:
        viz_finalize(viz_ops,model.xlimits_num,model.funcs,model.gprs[-1],model.x_data,model.y_data,n_iter-1)
        viz_show_plots(viz_ops,n_frames=n_iter)

#########################################################
# Find the optimal point that has been evaluated by the high fidelity model
def find_min(model):
    ind_best = np.argmin(model.y_data[model.n_fl-1])
    # # option 1: estimate the optimum using the high fidelity model
    # x_opt = model.x_data[model.n_fl-1][ind_best,:]
    # y_opt = model.y_data[model.n_fl-1][ind_best]
    # option 2: estimate the optimum using the highest fidelity GPR and every sampled location with any fidelity level
    x_opt = model.x_data[model.n_fl-1][ind_best,:]
    y_opt = model.y_data[model.n_fl-1][ind_best]
    opt_is_masked = False
    if not model.unmasked_data[model.n_fl-1][ind_best]:
        opt_is_masked = True
    for i in range(model.n_fl-1):
        y_min_i = np.min(model.gprs[-1].predict_values(model.x_data[i]))
        if  y_min_i < y_opt:
            ind_best_mf = np.argmin(model.gprs[-1].predict_values(model.x_data[i]))
            y_opt = y_min_i
            x_opt = model.x_data[i][ind_best_mf,:]
            if not model.unmasked_data[i][ind_best_mf]:
                opt_is_masked = True
            else:
                opt_is_masked = False
    if opt_is_masked:
        print('Warning: the minimum value returned is in a region of masked data (the simulation returned NaN or out of allowable bounds values), so there is significant uncertainty in this solution.')
    # option 3: could implement a minimization on the GPR surface though this introduces additional uncertainty

    return [x_opt, y_opt]
