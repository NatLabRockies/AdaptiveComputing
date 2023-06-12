# static_sampling.py
import numpy as np
from .utils import check_nan_oob
#########################################################
# Pseudo-random initial sampling
def add_lhs_samples(model,n_lhs_samp):
    # n_lhs_samp_min = model.n_dim + 1
    # if not hasattr(options, 'n_init_samp'):
    #     n_lhs_samp = [n_lhs_samp_min] * model.n_fl
    n_lhs_samp = np.atleast_1d(n_lhs_samp)
    if len(n_lhs_samp) != model.n_fl:
        raise Exception('Must list the number of initial samples for each function provided in funcs_in.')
    # change n_init_samp if it is less that the allowable minimum
    # for i in range(n_fl):
    #     if n_lhs_samp[i] > 0:
    #         if n_lhs_samp[i] < n_lhs_samp_min:
    #             print('Warning: The number of requested initial samples n_init_samp for fidelity level ' + str(i) + ' is being overwritten to be its minimum allowable non-zero value of ' + str(n_lhs_samp_min) + str('.'))
    #             n_lhs_samp[i] = n_lhs_samp_min

    from smt.sampling_methods import LHS
    if model.mixed_type:
        from smt.applications.mixed_integer import MixedIntegerSamplingMethod
    for i in range(model.n_fl):
        if n_lhs_samp[i] > 0: 
            # assert(n_lhs_samp[i] >= model.n_dim + 1) # this is only required if BayesOpt is used?
            rand_state = np.random.RandomState()
            if model.options.deterministic:
                # rand_state = i*(model.options.n_iter+1) # ensurses the fidelity levels all have unique seeds on all optimization iterations
                # ensurses the fidelity levels all have unique seeds on all optimization iterations. 
                rand_state = int(sum(model.n_samp)+1.0)
            
            if model.mixed_type:
                sampling = MixedIntegerSamplingMethod(model.xtypes, model.xlimits, LHS, criterion="maximin", random_state=rand_state)
            else:
                sampling = LHS(xlimits=model.xlimits, criterion='maximin', random_state=rand_state)
            x_data_lhs = sampling(n_lhs_samp[i])
            model.x_data[i] = np.append(model.x_data[i],x_data_lhs,axis=0)
            y_data_lhs = np.atleast_2d(np.zeros_like(x_data_lhs[:,0])).T
            for i_s in range (len(y_data_lhs)):
                y_data_lhs[i_s] = model.funcs[i](x_data_lhs[i_s,:])
            model.y_data[i] = np.append(model.y_data[i],y_data_lhs,axis=0)
            model.n_samp[i] = model.n_samp[i] + n_lhs_samp[i]

    perform_lower_sims(model)
    check_all_nan_oob(model)
    #retrain(model)

#########################################################
# Read user-provided function evaluations from a .csv file
def add_file_samples(model,filenames):
    from .utils import read_sample_csv
    [x_data_file, y_data_file] = read_sample_csv(model,filenames)
    for i in range(model.n_fl):
        filename = filenames[i]
        if filename != '':
            model.x_data[i] = np.append(x_data_file[i],model.x_data[i],axis=0)
            model.y_data[i] = np.append(y_data_file[i],model.y_data[i],axis=0)
            model.n_samp[i] = model.n_samp[i] + len(y_data_file[i])
    perform_lower_sims(model)
    check_all_nan_oob(model)
    #retrain(model)

#########################################################
# Mask data that is NaN or outside allowable bounds
def retrain(model):
    # Check that there is enough initial samples to train a GP model
    for i in range(model.n_fl):
        if np.count_nonzero(model.unmasked_data[i]) < model.n_dim + 1:
            raise Exception('For fidelity level ' + str(i) + ', there are '+str(np.count_nonzero(model.unmasked_data[i]))+' < n_dim+1 initial values that are non-NaN and within user-specified bounds.  Either specify more in input file or increase the number of pseudo-randomly sampled points n_init_samp.')
        if i > 0:
            if np.count_nonzero(model.unmasked_data[i]) >= np.count_nonzero(model.unmasked_data[i-1]):
                print('Warning: the number of initial data for fidelity level '+str(i)+' is '+str(np.count_nonzero(model.unmasked_data[i]))+', which is >= '+str(np.count_nonzero(model.unmasked_data[i-1]))+', the number of initial data for (lower) fidelity level '+str(i-1)+'. This can lead to poor performance and is typically not an efficient way to initialize the Bayesian Optimization. This includes data read from files, LHS sampling, and perform_lower_sims if active. Note: that only values that are non-NaN and within user-specified allowable bounds are included in these counts.')
    #XXX retrain ...

#########################################################
# At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
def perform_lower_sims(model):
    if model.options.perform_lower_sims:
        for i_fl in range(model.n_fl-1,0,-1): # for all levels greater than zero, counting backwards
            for j_d_upper in range(model.n_samp[i_fl]): # for all data points in this level
                # check if this point exists on the next lowest level
                pt_exists = False 
                for j_d_lower in range(model.n_samp[i_fl-1]): # for all data in the level below
                    if np.array_equal(model.x_data[i_fl][j_d_upper,:],model.x_data[i_fl-1][j_d_lower,:]):
                        pt_exists = True
                if not pt_exists: # add it to the lower level
                    y_et_k = model.funcs[i_fl-1](model.x_data[i_fl][j_d_upper,:])
                    model.y_data[i_fl-1] = np.atleast_2d(np.append(model.y_data[i_fl-1],y_et_k)).T
                    model.x_data[i_fl-1] = np.append(model.x_data[i_fl-1],np.atleast_2d(model.x_data[i_fl][j_d_upper,:]),axis=0)
                    model.n_samp[i_fl-1] = model.n_samp[i_fl-1] + 1

#########################################################
# Mask data that is NaN or outside allowable bounds
def check_all_nan_oob(model):
    for ind_which_lvl in range(model.n_fl):
        model.unmasked_data[ind_which_lvl] = np.full([len(model.y_data[ind_which_lvl]),1], True)
        for i in range(len(model.y_data[ind_which_lvl])):
            model.unmasked_data[ind_which_lvl][i] = check_nan_oob(model.y_data[ind_which_lvl][i],model.options)



