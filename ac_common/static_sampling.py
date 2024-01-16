# static_sampling.py
import numpy as np
from decimal import Decimal
from .utils import check_nan_oob
from .model import num_to_native
#########################################################
# Pseudo-random initial sampling
def add_lhs_samples(model,n_lhs_samp):
    n_lhs_samp = np.atleast_1d(n_lhs_samp)
    if len(n_lhs_samp) != model.n_fl:
        raise Exception('Must provide a list of length n fidelity levels specifying number of initial samples for each fidelity level.')
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
            
            if model.mod_ops.use_hero:
                for i_s in range (n_lhs_samp[i]):
                    model.queue_hero_sample(i,x_data_lhs[i_s,:],y_sur=False)
            else:
                for i_s in range (n_lhs_samp[i]):
                    model.add_xnum_sample(i,x_data_lhs[i_s,:],train=False)
                    # Note: train=False since will perform a single training below, outside the loop

    if model.mod_ops.use_hero:
        print('Note: LHS with use_hero=True is blocking. Since non-blocking relies on the surrogate already having enough samples to do masking. If you write these outputs to a file, next time you can use add_file_samples instead of add_lhs_samples')
        model.wait_for_workers(train=False)
    model.train_on_unmasked_data()

#########################################################
# Read user-provided function evaluations from a .csv file
def add_file_samples(model,filenames):
    import csv
    filenames = np.atleast_1d(filenames)
    if len(filenames) != model.n_fl:
        raise Exception('If any filenames are provided, the length of the list of filenames must equal len(simulations). Use empty quotes as an entry in the list if no data should be loaded for a fidelity level.')
    for filename in filenames:
        if not filename.endswith('.csv'):
            if filename != '':
                raise Exception('csv filename must end in .csv or be an empty string')

    for f in range(model.n_fl):
        filename = filenames[f]
        if filename == '':
            print('No input data file specified for fidelity level ' + str(f) + '. Skipping read_input_data for this level.')
        else:
            with open(filename, newline='', encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile, delimiter=',') # , quotechar='|'
                a = []
                for row in reader:
                    if (len(row) != model.n_dim+1) and (len(row) != model.n_dim):
                        raise Exception('Number of columns in csv must be >= the number of parameters (n_dim) and <= n_dim+1.')
                    a.append(row)
            n_samples = len(a) - 1 # first row is header
            if n_samples < 1:
                raise Exception('There is less than 1 row of data (not counting the header) in the csv for fidelity level ' + str(f) + '. Use an empty string as the file name instead.')
            
            # a is a list of lists
            for i in range(n_samples):
                x_cur = np.zeros([model.n_dim])
                for j in range(model.n_dim):
                    if a[0][j] == 'categorical':
                        x_cur[j] = model.params[j].categories.index(a[i+1][j])
                    elif a[0][j] == 'continuous':
                        x_cur[j] = float(a[i+1][j])
                    elif a[0][j] == 'ordered':
                        x_cur[j] = int(float(a[i+1][j]))
                    else:
                        raise Exception('Unrecognized type for parameter '+str(i))    
                if len(a[i+1]) == model.n_dim: # if the user has not specified any objective function values
                    if model.mod_ops.use_hero:
                        model.queue_hero_sample(f,x_cur,y_sur=False)
                    else:
                        model.add_xnum_sample(f,x_cur,train=False) # Note: train=False since will perform a single training below, outside the loop
                elif a[i+1][model.n_dim] == '': # elif the user has specified some objective function values, but not the present row's objective function value
                    if model.mod_ops.use_hero:
                        model.queue_hero_sample(f,x_cur,y_sur=False)
                    else:
                        model.add_xnum_sample(f,x_cur,train=False)
                else: # the user has specified the parameters and the corresponding objective function evaluations
                    if model.mod_ops.use_hero:
                        model.queue_hero_sample(f,x_cur,y_sur=False)
                    else:
                        model.add_xnum_sample(f,x_cur,y_eval=float(a[i+1][model.n_dim]),train=False)

    if model.mod_ops.use_hero:
        print('Note: LHS with use_hero=True is blocking. Since non-blocking relies on the surrogate already having enough samples to do masking. If you write these outputs to a file, next time you can use add_file_samples instead of add_lhs_samples')
        model.wait_for_workers(train=False)
    model.train_on_unmasked_data()

#########################################################
# Convert any categorical entries in x_query to be numbers   
def native_to_num(model,x_eval_native):
    x_eval_num = np.zeros([1,model.n_dim])
    for j in range(model.n_dim):
        if model.params[j].type == 'categorical':
            x_eval_num[0,j] = model.params[j].categories.index(x_eval_native[j])
        elif (model.params[j].type == 'continuous') or (model.params[j].type == 'ordered'):
            x_eval_num[0,j] = x_eval_native[j]
        else:
            raise Exception('Unrecognized type for parameter '+str(j)) 
    return x_eval_num

#########################################################
# Add a sim to the model's training set, retrain the model using all unmasked data, and update the masked data using the new surrogate
# The x_eval_num argument has variable types converted to floats as SMT expects
def add_xnum_sample(model,fidelity_level,x_eval_num,y_eval,viz_ops,frame_id,train):
    if y_eval is None:
        # y_eval = model.funcs[fidelity_level](x_eval_num)
        y_eval = model.eval_xnum(fidelity_level,x_eval_num)
    model.x_data[fidelity_level] = np.append(model.x_data[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
    model.y_data[fidelity_level] = np.atleast_2d(np.append(model.y_data[fidelity_level], y_eval)).T
    model.n_samp[fidelity_level] += 1

    # Check for NaNs and out of bounds y_data
    model.unmasked_data[fidelity_level] = np.atleast_2d(np.append(model.unmasked_data[fidelity_level],True)).T
    model.unmasked_data[fidelity_level][-1] = check_nan_oob(model.y_data[fidelity_level][-1],model.mod_ops)
    model.hero_todo[fidelity_level] = np.atleast_2d(np.append(model.hero_todo[fidelity_level],False)).T # don't add this point to the hero queue
    model.hero_task_id[fidelity_level] = np.atleast_2d(np.append(model.hero_task_id[fidelity_level],'None')).T
    
    # At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
    if model.mod_ops.perform_lower_sims:        
        pt_exists = False 
        im1_fl = fidelity_level-1
        if im1_fl >= 0:    
            for j_d_lower in range(model.n_samp[im1_fl]): # for all data in the level below
                if np.array_equal(model.x_data[fidelity_level][-1,:],model.x_data[im1_fl][j_d_lower,:]):
                    pt_exists = True
            if not pt_exists: # add the sim at the im1_fl level
                model.add_xnum_sample(im1_fl,x_eval_num,train=train)

    # Visualize the next point to add and the surrogate that has not yet been trained on this point
    if viz_ops is not None:
        from .viz import viz_animate
        viz_animate(model,viz_ops,frame_id)

    if train:
        model.train_on_unmasked_data()

#########################################################
# Add a sim to the model's training set and retrain the model using all unmasked data.
# Unlike add_xnum_sample, this is a non-blocking operation. The sim is added to a hero
# queue of simulations to run and the function returns, storing a temporary result
# in the mean time.
# The x_eval_num argument has variable types converted to floats as SMT expects
def queue_hero_sample(model,fidelity_level,x_eval_num,y_sur,viz_ops,frame_id):
    model.x_data[fidelity_level] = np.append(model.x_data[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
    # set a placeholder value using the surrogate's prediction
    assert(isinstance(y_sur, bool))
    if y_sur:
        y_eval = model.surrogate.predict_values(np.atleast_2d(x_eval_num),fidelity_level)[0]
        model.y_data[fidelity_level] = np.atleast_2d(np.append(model.y_data[fidelity_level], y_eval)).T
    else:
        model.y_data[fidelity_level] = np.atleast_2d(np.append(model.y_data[fidelity_level], np.NaN)).T
    model.n_samp[fidelity_level] += 1

    # Mark the data as masked
    model.unmasked_data[fidelity_level] = np.atleast_2d(np.append(model.unmasked_data[fidelity_level],False)).T
    # Mark the data as in the hero queue
    model.hero_todo[fidelity_level] = np.atleast_2d(np.append(model.hero_todo[fidelity_level],True)).T
    x_eval_native = num_to_native(x_eval_num,model.params)
    x_eval_str = [str(variable) for variable in x_eval_native]
    # Since Hero only allows string arguments, convert args to strings and append the variable types
    for j in range(model.n_dim):
        if model.params[j].type == 'categorical':
            x_eval_str[j] += '_categorical'
        elif model.params[j].type == 'continuous':
            x_eval_str[j] += '_continuous'
        elif model.params[j].type == 'ordered':
            x_eval_str[j] += '_ordered'
        else:
            raise Exception('Unrecognized type for parameter '+str(j)) 
    task_id = model.hero_objs[fidelity_level].put_tasks([{"name": "test_"+str(fidelity_level)+"_"+str(model.n_samp[fidelity_level]), "args": x_eval_str}])[0]
    model.hero_task_id[fidelity_level] = np.atleast_2d(np.append(model.hero_task_id[fidelity_level],task_id)).T

    # At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
    if model.mod_ops.perform_lower_sims:        
        pt_exists = False 
        im1_fl = fidelity_level-1
        if im1_fl >= 0:    
            for j_d_lower in range(model.n_samp[im1_fl]): # for all data in the level below
                if np.array_equal(model.x_data[fidelity_level][-1,:],model.x_data[im1_fl][j_d_lower,:]):
                    pt_exists = True
            if not pt_exists: # add the sim at the im1_fl level
                model.queue_hero_sample(im1_fl,x_eval_num)

#########################################################
# Collect data from any sims in the hero queue that have finished. Retrain the surrogate.
def sync_hero_results(model,viz_ops,train):
    # if viz_ops is not None:
    #     from .viz import viz_animate
    for i_fl in range(model.n_fl):
        for i in range(model.n_samp[i_fl]):
            if model.hero_todo[i_fl][i] == True:
                task_id = model.hero_task_id[i_fl][i][0]
                task_data = model.hero_objs[i_fl].get_task(task_id)
                if task_data["status"] == 'complete':
                    print(f'Found a complete task = {task_id} with simulation result = {task_data["results_s3"]}')
                    y_eval = float(task_data["results"]["objective"])
                    model.y_data[i_fl][i] = np.atleast_2d(y_eval)
                    # Check for NaNs and out of bounds y_data
                    model.unmasked_data[i_fl][i] = check_nan_oob(y_eval,model.mod_ops)
                    model.hero_todo[i_fl][i] = False
                    # Visualize the next point to add and the surrogate that has not yet been trained on this point
                    # if viz_ops is not None:
                    #     viz_animate(model,viz_ops,frame_id)

    if train:
        model.train_on_unmasked_data()

#########################################################
# Wait for workers to complete all tasks in Hero queues.
def wait_for_workers(model,viz_ops,train):
    print('Wait until workers complete all tasks in all hero queues.')
    while True:
        total_in_queue = np.sum([np.sum(arr) for arr in model.hero_todo])
        if total_in_queue > 0:
            print(f'Number of remaining tasks in hero queues = {total_in_queue}')
            model.hero_objs[0].wait(1)
            sync_hero_results(model,viz_ops,train)
        else:
            print('Workers are done. All Hero queues are empty.')
            break

# #########################################################
# # Add a sim to the model's training set and retrain the model
# # The x_eval_native argument has the data types defined by the use-defined list of Params
# # If y_eval is not speicified, a simulation is conducted
# def add_xnative_sim(model,fidelity_level,x_eval_native,y_eval=None):
#     x_eval_num = native_to_num(x_eval_native)
#     model.add_xnum_sample(fidelity_level,x_eval_num,y_eval)

#########################################################
# Check if the x_eval_native follows the user specified bounds in the list of Params
# The x_eval_native argument has the data types defined by the usef-defined list of Params
def bounds_check_xnative(model,x_eval_native):
    for j in range(model.n_dim):
        if model.params[j].type == 'categorical':
            assert(isinstance(x_eval_native[j], str))
            if x_eval_native[j] not in model.params[j].categories:
                raise Exception('Parameter x' + str(j) + ' = '+str(x_eval_native[j])+' is not a valid value for categorical parameter.')
        elif (model.params[j].type == 'continuous') or (model.params[j].type == 'ordered'):
            assert(isinstance(x_eval_native[j], (int,float)))
            if x_eval_native[j] < model.params[j].min_val or x_eval_native[j] > model.params[j].max_val:
                raise Exception('Out of bounds value of parameter x' + str(j) + ' = '+str(x_eval_native[j])+' .')
        else:
            raise Exception('Unrecognized type for parameter x'+str(j))

#########################################################
# Train the surrogate model using x_data and y_data. This training only uses unmasked data (i.e. not-NaN and within the user-specified allowable bounds).
def train_on_unmasked_data(model):
    # Check that there are enough samples to train a GP model
    for i in range(model.n_fl):
        if np.count_nonzero(model.unmasked_data[i]) < model.n_dim + 1 + i:
            raise Exception('For fidelity level ' + str(i) + ', there are '+str(np.count_nonzero(model.unmasked_data[i]))+
                            ' < n_dim+1+fidelity_level simulation sample values that are non-NaN and within user-specified bounds.'+
                            ' Either specify more in input file or increase the number of pseudo-randomly sampled points'+
                            ' n_lhs_samp before training a surrogate model.')
        # print a warning if there are more higher fidelity samples than lower fidelity samples
        if i > 0:
            if np.count_nonzero(model.unmasked_data[i]) >= np.count_nonzero(model.unmasked_data[i-1]):
                print('Warning: the number of simulation samples for fidelity level '+str(i)+' is '+
                      str(np.count_nonzero(model.unmasked_data[i]))+
                      ', which is >= '+str(np.count_nonzero(model.unmasked_data[i-1]))+
                      ', the number of simulation samples for (lower) fidelity level '+str(i-1)+
                      '. This can lead to poor performance and is typically not an efficient way to '+
                      'initialize the Bayesian Optimization. This includes data read from files, LHS'+
                      ' sampling, and perform_lower_sims if active. Note: that only values that are '+
                      'non-NaN and within user-specified allowable bounds are included in these counts.')
    
    # Extract the unmasked data
    x_data_unmasked = []
    y_data_unmasked = []
    for i_fl in range(model.n_fl):
        x_data_unmasked.append(model.x_data[i_fl][model.unmasked_data[i_fl].flatten()])
        y_data_unmasked.append(model.y_data[i_fl][model.unmasked_data[i_fl].flatten()])
    model.surrogate.train(x_data_unmasked, y_data_unmasked)

    # Update the values for the masked data
    # Predict the mean value at all x_data[masked] values using surrogate_unmasked (and store these in y_data[masked])
    for i_fl in range(model.n_fl):
        for i in range(len(model.y_data[i_fl])):
            if not model.unmasked_data[i_fl][i]:
                model.y_data[i_fl][i] = model.surrogate.predict_values(np.atleast_2d(model.x_data[i_fl][i]), i_fl)[0]

#########################################################
# Train the surrogate model using x_data and y_data. This training uses masked and unmasked data.
def train_on_all_data(model):
    # Check that there are enough samples to train a GP model
    for i in range(model.n_fl):
        if len(model.y_data[i]) < model.n_dim + 1 + i:
            raise Exception('For fidelity level ' + str(i) + ', there are '+str(len(model.y_data[i]))+
                            ' < n_dim+1+fidelity_level simulation samples.'+
                            ' Either specify more in input file or increase the number of pseudo-randomly sampled points'+
                            ' n_lhs_samp before training a surrogate model.')
        # print a warning if there are more higher fidelity samples than lower fidelity samples
        if i > 0:
            if len(model.y_data[i]) >= len(model.y_data[i-1]):
                print('Warning: the number of simulation samples for fidelity level '+str(i)+' is '+
                      str(len(model.y_data[i]))+
                      ', which is >= '+str(len(model.y_data[i-1]))+
                      ', the number of simulation samples for (lower) fidelity level '+str(i-1)+
                      '. This can lead to poor performance and is typically not an efficient way to '+
                      'initialize the Bayesian Optimization. This includes data read from files, LHS'+
                      ' sampling, and perform_lower_sims if active. Note: that only values that are '+
                      'non-NaN and within user-specified allowable bounds are included in these counts.')
                
    model.surrogate.train(model.x_data, model.y_data)

# #########################################################
# # At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
# def check_all_lower_sims(model):
#     if model.mod_ops.perform_lower_sims:
#         for i_fl in range(model.n_fl-1,0,-1): # for all levels greater than zero, counting backwards
#             for j_d_upper in range(model.n_samp[i_fl]): # for all data points in this level
#                 # check if this point exists on the next lowest level
#                 pt_exists = False 
#                 for j_d_lower in range(model.n_samp[i_fl-1]): # for all data in the level below
#                     if np.array_equal(model.x_data[i_fl][j_d_upper,:],model.x_data[i_fl-1][j_d_lower,:]):
#                         pt_exists = True
#                 if not pt_exists: # add it to the lower level
#                     y_eval = model.funcs[i_fl-1](model.x_data[i_fl][j_d_upper,:])
#                     model.y_data[i_fl-1] = np.atleast_2d(np.append(model.y_data[i_fl-1],y_eval)).T
#                     model.x_data[i_fl-1] = np.append(model.x_data[i_fl-1],np.atleast_2d(model.x_data[i_fl][j_d_upper,:]),axis=0)
#                     model.n_samp[i_fl-1] = model.n_samp[i_fl-1] + 1

# #########################################################
# # Mask data that is NaN or outside allowable bounds
# def check_all_nan_oob(model):
#     for ind_which_lvl in range(model.n_fl):
#         model.unmasked_data[ind_which_lvl] = np.full([len(model.y_data[ind_which_lvl]),1], True)
#         for i in range(len(model.y_data[ind_which_lvl])):
#             model.unmasked_data[ind_which_lvl][i] = check_nan_oob(model.y_data[ind_which_lvl][i],model.mod_ops)
#         # don't put these points in the hero queue
#         model.hero_todo[ind_which_lvl] = np.full([len(model.y_data[ind_which_lvl]),1], False)
#         model.hero_task_id[ind_which_lvl] = np.full([len(model.y_data[ind_which_lvl]),1], 'None')



