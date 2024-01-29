# static_sampling.py
import numpy as np
from decimal import Decimal
from .utils import check_nan_oob
from .dataset import num_to_native
#########################################################
# Pseudo-random initial sampling
def add_lhs_samples(dataset,n_lhs_samp,surrogate):
    n_lhs_samp = np.atleast_1d(n_lhs_samp)
    if len(n_lhs_samp) != dataset.n_fl:
        raise Exception('Must provide a list of length n fidelity levels specifying number of initial samples for each fidelity level.')
    from smt.sampling_methods import LHS
    if dataset.mixed_type:
        from smt.applications.mixed_integer import MixedIntegerSamplingMethod
    for i in range(dataset.n_fl):
        if n_lhs_samp[i] > 0: 
            if n_lhs_samp[i] <= 1:
                raise Exception('LatinHypercubeSampler requires n_lhs_samp ==0 or >=2')
            rand_state = np.random.RandomState()
            if dataset.ds_ops.deterministic:
                # Ensurses the fidelity levels all have unique seeds.
                # Note: that the sampler will increment the rand_state for each sample
                rand_state = int(sum(dataset.n_samp)+1.0 + sum(n_lhs_samp[:i]))
            
            if dataset.mixed_type:
                sampling = MixedIntegerSamplingMethod(dataset.xtypes, dataset.xlimits, LHS, criterion="maximin", random_state=rand_state)
            else:
                sampling = LHS(xlimits=dataset.xlimits, criterion='maximin', random_state=rand_state)
            x_data_lhs = sampling(n_lhs_samp[i])
            
            if dataset.ds_ops.use_hero:
                for i_s in range (n_lhs_samp[i]):
                    dataset.queue_hero_sample(i,x_data_lhs[i_s,:],surrogate=None)
            else:
                for i_s in range (n_lhs_samp[i]):
                    dataset.add_xnum_sample(i,x_data_lhs[i_s,:],surrogate=None)
                    # Note: surrogate=None since will perform a single training below, outside the loop

    if dataset.ds_ops.use_hero:
        print('Note: LHS with use_hero=True is blocking. Since non-blocking relies on the surrogate already having enough samples to do masking. If you write these outputs to a file, next time you can use add_file_samples instead of add_lhs_samples')
        dataset.wait_for_workers(surrogate=None)
    if surrogate is not None:
        dataset.train_on_unmasked_data(surrogate)

#########################################################
# Read user-provided function evaluations from a .csv file
def add_file_samples(dataset,filenames,surrogate):
    import csv
    filenames = np.atleast_1d(filenames)
    if len(filenames) != dataset.n_fl:
        raise Exception('If any filenames are provided, the length of the list of filenames must equal len(simulations). Use empty quotes as an entry in the list if no data should be loaded for a fidelity level.')
    for filename in filenames:
        if not filename.endswith('.csv'):
            if filename != '':
                raise Exception('csv filename must end in .csv or be an empty string')

    for f in range(dataset.n_fl):
        filename = filenames[f]
        if filename == '':
            print('No input data file specified for fidelity level ' + str(f) + '. Skipping read_input_data for this level.')
        else:
            with open(filename, newline='', encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile, delimiter=',') # , quotechar='|'
                a = []
                for row in reader:
                    if (len(row) != dataset.n_in+dataset.n_out) and (len(row) != dataset.n_in):
                        raise Exception('Number of columns in csv must be either the number of parameters (n_in) or the number of outputs (n_out) plus n_in.')
                    a.append(row)
            n_samples = len(a) - 1 # first row is header
            if n_samples < 1:
                raise Exception('There is less than 1 row of data (not counting the header) in the csv for fidelity level ' + str(f) + '. Use an empty string as the file name instead.')
            
            # a is a list of lists
            for i in range(n_samples):
                x_cur = np.zeros([dataset.n_in])
                for j in range(dataset.n_in):
                    if a[0][j] == 'categorical':
                        x_cur[j] = dataset.params[j].categories.index(a[i+1][j])
                    elif a[0][j] == 'continuous':
                        x_cur[j] = float(a[i+1][j])
                    elif a[0][j] == 'ordered':
                        x_cur[j] = int(float(a[i+1][j]))
                    else:
                        raise Exception('Unrecognized type for parameter '+str(i))    
                if len(a[i+1]) == dataset.n_in: # if the user has not specified any objective function values
                    if dataset.ds_ops.use_hero:
                        dataset.queue_hero_sample(f,x_cur,surrogate=None)
                    else:
                        dataset.add_xnum_sample(f,x_cur,surrogate=None) # Note: surrogate=None since will perform a single training below, outside the loop
                elif any(not s.strip() for s in a[i+1][dataset.n_in:]): # elif the user has specified some objective function values, but not all of the obj values for the present row
                    if dataset.ds_ops.use_hero:
                        dataset.queue_hero_sample(f,x_cur,surrogate=None)
                    else:
                        dataset.add_xnum_sample(f,x_cur,surrogate=None)
                else: # the user has specified the parameters and the corresponding objective function evaluations 
                    dataset.add_xnum_sample(f,x_cur,y_eval=[float(s) for s in a[i+1][dataset.n_in:]],surrogate=None)

    if dataset.ds_ops.use_hero:
        print('Note: LHS with use_hero=True is blocking. Since non-blocking relies on the surrogate already having enough samples to do masking. If you write these outputs to a file, next time you can use add_file_samples instead of add_lhs_samples')
        dataset.wait_for_workers(surrogate=None)
    if surrogate is not None:
        dataset.train_on_unmasked_data(surrogate)

#########################################################
# Convert any categorical entries in x_query to be numbers   
def native_to_num(dataset,x_eval_native):
    x_eval_num = np.zeros([1,dataset.n_in])
    for j in range(dataset.n_in):
        if dataset.params[j].type == 'categorical':
            x_eval_num[0,j] = dataset.params[j].categories.index(x_eval_native[j])
        elif (dataset.params[j].type == 'continuous') or (dataset.params[j].type == 'ordered'):
            x_eval_num[0,j] = x_eval_native[j]
        else:
            raise Exception('Unrecognized type for parameter '+str(j)) 
    return x_eval_num

#########################################################
# Add a sim to the dataset, retrain the surrogate model using all unmasked data, and update the masked data using the new surrogate
# The x_eval_num argument has variable types converted to floats as SMT expects
def add_xnum_sample(dataset,fidelity_level,x_eval_num,y_eval,viz_ops,frame_id,surrogate):
    if y_eval is None:
        # y_eval = dataset.funcs[fidelity_level](x_eval_num)
        y_eval = dataset.eval_xnum(fidelity_level,x_eval_num)
    dataset.x_data[fidelity_level] = np.append(dataset.x_data[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
    dataset.y_data[fidelity_level] = np.append(dataset.y_data[fidelity_level],np.atleast_2d(y_eval),axis=0)
    dataset.n_samp[fidelity_level] += 1

    # Check for NaNs and out of bounds y_data
    dataset.unmasked_data[fidelity_level] = np.append(dataset.unmasked_data[fidelity_level],np.full((1,dataset.n_out), True, dtype=bool),axis=0)
    for i_o in range(dataset.n_out):
        dataset.unmasked_data[fidelity_level][-1,i_o] = check_nan_oob(dataset.y_data[fidelity_level][-1,i_o],dataset.ds_ops)
    dataset.hero_todo[fidelity_level] = np.atleast_2d(np.append(dataset.hero_todo[fidelity_level],False)).T # don't add this point to the hero queue
    dataset.hero_task_id[fidelity_level] = np.atleast_2d(np.append(dataset.hero_task_id[fidelity_level],'None')).T
    
    # At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
    if dataset.ds_ops.perform_lower_sims:        
        pt_exists = False 
        im1_fl = fidelity_level-1
        if im1_fl >= 0:    
            for j_d_lower in range(dataset.n_samp[im1_fl]): # for all data in the level below
                if np.array_equal(dataset.x_data[fidelity_level][-1,:],dataset.x_data[im1_fl][j_d_lower,:]):
                    pt_exists = True
            if not pt_exists: # add the sim at the im1_fl level
                dataset.add_xnum_sample(im1_fl,x_eval_num,surrogate=surrogate)

    # Visualize the next point to add and the surrogate that has not yet been trained on this point
    if viz_ops is not None:
        from .viz import viz_animate
        viz_animate(dataset,surrogate,viz_ops,frame_id)

    if surrogate is not None:
        dataset.train_on_unmasked_data(surrogate)

#########################################################
# Add a sim to the dataset and retrain the surrogate model using all unmasked data.
# Unlike add_xnum_sample, this is a non-blocking operation. The sim is added to a hero
# queue of simulations to run and the function returns, storing a temporary result
# in the mean time.
# The x_eval_num argument has variable types converted to floats as SMT expects
def queue_hero_sample(dataset,fidelity_level,x_eval_num,surrogate,viz_ops,frame_id):
    dataset.x_data[fidelity_level] = np.append(dataset.x_data[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
    # set a placeholder value using the surrogate's prediction
    if surrogate is not None:
        y_eval = surrogate.predict_values(np.atleast_2d(x_eval_num),fidelity_level)[0]
        dataset.y_data[fidelity_level] = np.atleast_2d(np.append(dataset.y_data[fidelity_level], y_eval)).T
    else:
        dataset.y_data[fidelity_level] = np.atleast_2d(np.append(dataset.y_data[fidelity_level], np.full((1,dataset.n_out), np.NaN))).T
    dataset.n_samp[fidelity_level] += 1

    # Mark the data as masked
    dataset.unmasked_data[fidelity_level] = np.atleast_2d(np.append(dataset.unmasked_data[fidelity_level],np.full((1,dataset.n_out), False, dtype=bool))).T
    # Mark the data as in the hero queue
    dataset.hero_todo[fidelity_level] = np.atleast_2d(np.append(dataset.hero_todo[fidelity_level],True)).T
    x_eval_native = num_to_native(x_eval_num,dataset.params)
    x_eval_str = [str(variable) for variable in x_eval_native]
    # Since Hero only allows string arguments, convert args to strings and append the variable types
    for j in range(dataset.n_in):
        if dataset.params[j].type == 'categorical':
            x_eval_str[j] += '_categorical'
        elif dataset.params[j].type == 'continuous':
            x_eval_str[j] += '_continuous'
        elif dataset.params[j].type == 'ordered':
            x_eval_str[j] += '_ordered'
        else:
            raise Exception('Unrecognized type for parameter '+str(j)) 
    task_id = dataset.hero_objs[fidelity_level].put_tasks([{"name": "test_"+str(fidelity_level)+"_"+str(dataset.n_samp[fidelity_level]), "args": x_eval_str}])[0]
    dataset.hero_task_id[fidelity_level] = np.atleast_2d(np.append(dataset.hero_task_id[fidelity_level],task_id)).T

    # At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
    if dataset.ds_ops.perform_lower_sims:        
        pt_exists = False 
        im1_fl = fidelity_level-1
        if im1_fl >= 0:    
            for j_d_lower in range(dataset.n_samp[im1_fl]): # for all data in the level below
                if np.array_equal(dataset.x_data[fidelity_level][-1,:],dataset.x_data[im1_fl][j_d_lower,:]):
                    pt_exists = True
            if not pt_exists: # add the sim at the im1_fl level
                dataset.queue_hero_sample(im1_fl,x_eval_num,surrogate)

#########################################################
# Collect data from any sims in the hero queue that have finished. Retrain the surrogate if it is provided.
def sync_hero_results(dataset,surrogate,viz_ops):
    # if viz_ops is not None:
    #     from .viz import viz_animate
    for i_fl in range(dataset.n_fl):
        for i in range(dataset.n_samp[i_fl]):
            if dataset.hero_todo[i_fl][i] == True:
                task_id = dataset.hero_task_id[i_fl][i][0]
                task_data = dataset.hero_objs[i_fl].get_task(task_id)
                if task_data["status"] == 'complete':
                    print(f'Found a complete task = {task_id} with simulation result = {task_data["results_s3"]}')
                    y_eval = float(task_data["results"]["objective"])
                    dataset.y_data[i_fl][i,:] = np.atleast_2d(y_eval)
                    # Check for NaNs and out of bounds y_data
                    for i_o in range(dataset.n_out):
                        dataset.unmasked_data[i_fl][i,i_o] = check_nan_oob(y_eval[i_o],dataset.ds_ops)
                    dataset.hero_todo[i_fl][i] = False
                    # Visualize the next point to add and the surrogate that has not yet been trained on this point
                    # if viz_ops is not None:
                    #     viz_animate(dataset,surrogate,viz_ops,frame_id)

    if surrogate is not None:
        dataset.train_on_unmasked_data(surrogate)

#########################################################
# Wait for workers to complete all tasks in Hero queues.
def wait_for_workers(dataset,surrogate,viz_ops):
    print('Wait until workers complete all tasks in all hero queues.')
    while True:
        total_in_queue = np.sum([np.sum(arr) for arr in dataset.hero_todo])
        if total_in_queue > 0:
            print(f'Number of remaining tasks in hero queues = {total_in_queue}')
            dataset.hero_objs[0].wait(1)
            sync_hero_results(dataset,surrogate,viz_ops)
        else:
            print('Workers are done. All Hero queues are empty.')
            break

# #########################################################
# # Add a sim to the dataset and retrain the surrogate model
# # The x_eval_native argument has the data types defined by the use-defined list of Params
# # If y_eval is not speicified, a simulation is conducted
# def add_xnative_sim(dataset,fidelity_level,x_eval_native,y_eval=None,surrogate=None):
#     x_eval_num = native_to_num(x_eval_native)
#     dataset.add_xnum_sample(fidelity_level,x_eval_num,y_eval,surrogate)

#########################################################
# Check if the x_eval_native follows the user specified bounds in the list of Params
# The x_eval_native argument has the data types defined by the usef-defined list of Params
def bounds_check_xnative(dataset,x_eval_native):
    for j in range(dataset.n_in):
        if dataset.params[j].type == 'categorical':
            assert(isinstance(x_eval_native[j], str))
            if x_eval_native[j] not in dataset.params[j].categories:
                raise Exception('Parameter x' + str(j) + ' = '+str(x_eval_native[j])+' is not a valid value for categorical parameter.')
        elif (dataset.params[j].type == 'continuous') or (dataset.params[j].type == 'ordered'):
            assert(isinstance(x_eval_native[j], (int,float)))
            if x_eval_native[j] < dataset.params[j].min_val or x_eval_native[j] > dataset.params[j].max_val:
                raise Exception('Out of bounds value of parameter x' + str(j) + ' = '+str(x_eval_native[j])+' .')
        else:
            raise Exception('Unrecognized type for parameter x'+str(j))

#########################################################
# Train the surrogate model using x_data and y_data. This training only uses unmasked data (i.e. not-NaN and within the user-specified allowable bounds).
def train_on_unmasked_data(dataset,surrogate):
    # Check that there are enough samples to train a GP model
    for i in range(dataset.n_fl):
        if np.count_nonzero(dataset.unmasked_data[i][:,surrogate.i_out]) < dataset.n_in + 1 + i:
            raise Exception('For fidelity level ' + str(i) + ', there are '+str(np.count_nonzero(dataset.unmasked_data[i][:,surrogate.i_out]))+
                            ' < n_in+1+fidelity_level simulation sample values that are non-NaN and within user-specified bounds.'+
                            ' Either specify more in input file or increase the number of pseudo-randomly sampled points'+
                            ' n_lhs_samp before training a surrogate dataset.')
        # print a warning if there are more higher fidelity samples than lower fidelity samples
        if i > 0:
            if np.count_nonzero(dataset.unmasked_data[i][:,surrogate.i_out]) >= np.count_nonzero(dataset.unmasked_data[i-1][:,surrogate.i_out]):
                print('Warning: the number of simulation samples for fidelity level '+str(i)+' is '+
                      str(np.count_nonzero(dataset.unmasked_data[i][:,surrogate.i_out]))+
                      ', which is >= '+str(np.count_nonzero(dataset.unmasked_data[i-1][:,surrogate.i_out]))+
                      ', the number of simulation samples for (lower) fidelity level '+str(i-1)+
                      '. This can lead to poor performance and is typically not an efficient way to '+
                      'initialize the Bayesian Optimization. This includes data read from files, LHS'+
                      ' sampling, and perform_lower_sims if active. Note: that only values that are '+
                      'non-NaN and within user-specified allowable bounds are included in these counts.')
    
    # Extract the unmasked data
    x_data_unmasked = []
    y_data_unmasked = []
    for i_fl in range(dataset.n_fl):
        x_data_unmasked.append(dataset.x_data[i_fl][dataset.unmasked_data[i_fl][:,surrogate.i_out].flatten()])
        y_data_unmasked.append(dataset.y_data[i_fl][dataset.unmasked_data[i_fl][:,surrogate.i_out].flatten()])
    surrogate.train(x_data_unmasked, y_data_unmasked)

    # Update the values for the masked data
    # Predict the mean value at all x_data[masked] values using surrogate_unmasked (and store these in y_data[masked])
    for i_fl in range(dataset.n_fl):
        for i in range(dataset.n_samp[i_fl]):
            if not dataset.unmasked_data[i_fl][i,surrogate.i_out]:
                dataset.y_data[i_fl][i,surrogate.i_out] = surrogate.predict_values(np.atleast_2d(dataset.x_data[i_fl][i]), i_fl)[0]

#########################################################
# Train the surrogate model using x_data and y_data. This training uses masked and unmasked data.
def train_on_all_data(dataset,surrogate):
    # Check that there are enough samples to train a GP model
    for i in range(dataset.n_fl):
        if dataset.n_samp[i] < dataset.n_in + 1 + i:
            raise Exception('For fidelity level ' + str(i) + ', there are '+str(dataset.n_samp[i])+
                            ' < n_in+1+fidelity_level simulation samples.'+
                            ' Either specify more in input file or increase the number of pseudo-randomly sampled points'+
                            ' n_lhs_samp before training a surrogate dataset.')
        # print a warning if there are more higher fidelity samples than lower fidelity samples
        if i > 0:
            if dataset.n_samp[i] >= dataset.n_samp[i-1]:
                print('Warning: the number of simulation samples for fidelity level '+str(i)+' is '+
                      str(dataset.n_samp[i])+
                      ', which is >= '+str(dataset.n_samp[i-1])+
                      ', the number of simulation samples for (lower) fidelity level '+str(i-1)+
                      '. This can lead to poor performance and is typically not an efficient way to '+
                      'initialize the Bayesian Optimization. This includes data read from files, LHS'+
                      ' sampling, and perform_lower_sims if active. Note: that only values that are '+
                      'non-NaN and within user-specified allowable bounds are included in these counts.')
                
    surrogate.train(dataset.x_data, dataset.y_data)

# #########################################################
# # At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
# def check_all_lower_sims(dataset):
#     if dataset.ds_ops.perform_lower_sims:
#         for i_fl in range(dataset.n_fl-1,0,-1): # for all levels greater than zero, counting backwards
#             for j_d_upper in range(dataset.n_samp[i_fl]): # for all data points in this level
#                 # check if this point exists on the next lowest level
#                 pt_exists = False 
#                 for j_d_lower in range(dataset.n_samp[i_fl-1]): # for all data in the level below
#                     if np.array_equal(dataset.x_data[i_fl][j_d_upper,:],dataset.x_data[i_fl-1][j_d_lower,:]):
#                         pt_exists = True
#                 if not pt_exists: # add it to the lower level
#                     y_eval = dataset.funcs[i_fl-1](dataset.x_data[i_fl][j_d_upper,:])
#                     dataset.y_data[i_fl-1] = np.atleast_2d(np.append(dataset.y_data[i_fl-1],y_eval)).T
#                     dataset.x_data[i_fl-1] = np.append(dataset.x_data[i_fl-1],np.atleast_2d(dataset.x_data[i_fl][j_d_upper,:]),axis=0)
#                     dataset.n_samp[i_fl-1] = dataset.n_samp[i_fl-1] + 1

# #########################################################
# # Mask data that is NaN or outside allowable bounds
# def check_all_nan_oob(dataset):
#     for ind_which_lvl in range(dataset.n_fl):
#         dataset.unmasked_data[ind_which_lvl] = np.full([dataset.n_samp[ind_which_lvl],1], True)
#         for i in range(dataset.n_samp[ind_which_lvl]):
#             dataset.unmasked_data[ind_which_lvl][i,surrogate.i_out] = check_nan_oob(dataset.y_data[ind_which_lvl][i,surrogate.i_out],dataset.ds_ops)
#         # don't put these points in the hero queue
#         dataset.hero_todo[ind_which_lvl] = np.full([dataset.n_samp[ind_which_lvl],1], False)
#         dataset.hero_task_id[ind_which_lvl] = np.full([dataset.n_samp[ind_which_lvl],1], 'None')



