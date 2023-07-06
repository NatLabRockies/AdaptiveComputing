# static_sampling.py
import numpy as np
from .utils import check_nan_oob
#########################################################
# Pseudo-random initial sampling
def add_lhs_samples(model,n_lhs_samp):
    n_lhs_samp = np.atleast_1d(n_lhs_samp)
    if len(n_lhs_samp) != model.n_fl:
        raise Exception('Must list the number of initial samples for each function provided in funcs_in.')

    from smt.sampling_methods import LHS
    if model.mixed_type:
        from smt.applications.mixed_integer import MixedIntegerSamplingMethod
    for i in range(model.n_fl):
        if n_lhs_samp[i] > 0: 
            if n_lhs_samp[i] <= 1:
                raise Exception('LatinHypercubeSampler requires n_lhs_samp ==0 or >=2')
            rand_state = np.random.RandomState()
            if model.mod_ops.deterministic:
                # Ensurses the fidelity levels all have unique seeds.
                # Note: that the sampler will increment the rand_state for each sample
                rand_state = int(sum(model.n_samp)+1.0 + sum(n_lhs_samp[:i]))
            
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
    retrain(model)

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
    retrain(model)

#########################################################
# Update the surrogate model with the last point added. This training only uses unmasked data (i.e. not-NaN and within the user-specified allowable bounds).
def retrain(model):
    # Check that there are enough samples to train a GP model
    for i in range(model.n_fl):
        if np.count_nonzero(model.unmasked_data[i]) < model.n_dim + 1 + i:
            raise Exception('For fidelity level ' + str(i) + ', there are '+str(np.count_nonzero(model.unmasked_data[i]))+
                            ' < n_dim+1+fidelity_level initial values that are non-NaN and within user-specified bounds.'+
                            ' Either specify more in input file or increase the number of pseudo-randomly sampled points'+
                            ' n_lhs_samp.')
        # print a warning if there are more higher fidelity samples than lower fidelity samples
        if i > 0:
            if np.count_nonzero(model.unmasked_data[i]) >= np.count_nonzero(model.unmasked_data[i-1]):
                print('Warning: the number of initial data for fidelity level '+str(i)+' is '+
                      str(np.count_nonzero(model.unmasked_data[i]))+
                      ', which is >= '+str(np.count_nonzero(model.unmasked_data[i-1]))+
                      ', the number of initial data for (lower) fidelity level '+str(i-1)+
                      '. This can lead to poor performance and is typically not an efficient way to '+
                      'initialize the Bayesian Optimization. This includes data read from files, LHS'+
                      ' sampling, and perform_lower_sims if active. Note: that only values that are '+
                      'non-NaN and within user-specified allowable bounds are included in these counts.')
    
    for i_fl in range(model.n_fl):
        for ii_fl in range(i_fl):
            model.gprs[i_fl].set_training_values(model.x_data[ii_fl][model.unmasked_data[ii_fl].flatten()],
                                                 model.y_data[ii_fl][model.unmasked_data[ii_fl].flatten()], name=ii_fl)
            # Note: other fidelities are accessed with names from 0 to n_fl-2 listed in order of increasing fidelity.
        model.gprs[i_fl].set_training_values(model.x_data[i_fl][model.unmasked_data[i_fl].flatten()],
                                             model.y_data[i_fl][model.unmasked_data[i_fl].flatten()])
        # Note: highest-fidelity dataset does not get a name        
        model.gprs[i_fl].train()

#########################################################
# At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
def perform_lower_sims(model):
    if model.mod_ops.perform_lower_sims:
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
            model.unmasked_data[ind_which_lvl][i] = check_nan_oob(model.y_data[ind_which_lvl][i],model.mod_ops)



