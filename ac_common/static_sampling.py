# static_sampling.py
import numpy as np
from decimal import Decimal
from .utils import check_skip_vec
from .utils import check_unmasked
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

    # Check if the data should be skipped because it is unmasked and is either NaN or out of bounds
    # Skipped data is not added to x_data, y_data
    if check_skip_vec(np.atleast_1d(y_eval),dataset.ds_ops,dataset.n_out):
        dataset.x_data_skipped[fidelity_level] = np.append(dataset.x_data_skipped[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
        dataset.y_data_skipped[fidelity_level] = np.append(dataset.y_data_skipped[fidelity_level],np.atleast_2d(y_eval),axis=0)
        dataset.n_samp_skipped[fidelity_level] += 1
        
    else: # add to x_data and y_data and use masking if needed
        dataset.x_data[fidelity_level] = np.append(dataset.x_data[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
        dataset.y_data[fidelity_level] = np.append(dataset.y_data[fidelity_level],np.atleast_2d(y_eval),axis=0)
        dataset.n_samp[fidelity_level] += 1
        # Check for NaNs and out of bounds y_data
        dataset.unmasked_data[fidelity_level] = np.append(dataset.unmasked_data[fidelity_level],np.full((1,dataset.n_out), True, dtype=bool),axis=0)
        for i_o in range(dataset.n_out):
            dataset.unmasked_data[fidelity_level][-1,i_o] = check_unmasked(dataset.y_data[fidelity_level][-1,i_o],dataset.ds_ops)
        # the function has been evaluated, so no need to add to the hero queue
        dataset.hero_todo[fidelity_level] = np.atleast_2d(np.append(dataset.hero_todo[fidelity_level],False)).T
        dataset.hero_task_id[fidelity_level] = np.atleast_2d(np.append(dataset.hero_task_id[fidelity_level],'None')).T
        
        # Visualize the next point to be added and visualize the surrogate that has not yet been trained on this point
        if viz_ops is not None:
            from .viz import viz_animate
            viz_animate(dataset,surrogate,viz_ops,frame_id)
        
    # At every point in the design space where a simulation is performed, recursively compute all lower fidelity level simulations there too
    if dataset.ds_ops.perform_lower_sims:        
        pt_exists = False 
        im1_fl = fidelity_level-1
        if im1_fl >= 0:    
            for j_d_lower in range(dataset.n_samp[im1_fl]): # for all data in the level below
                if np.array_equal(dataset.x_data[fidelity_level][-1,:],dataset.x_data[im1_fl][j_d_lower,:]):
                    pt_exists = True
            if not pt_exists: # add the sim at the im1_fl level
                dataset.add_xnum_sample(im1_fl,x_eval_num,surrogate=surrogate)

    if surrogate is not None:
        dataset.train_on_unmasked_data(surrogate)

def add_xnum_sample_6d(dataset,fidelity_level,x_eval_num,y_eval,viz_ops,frame_id,surrogate0,surrogate1,surrogate2,surrogate3,surrogate4,surrogate5):
    if y_eval is None:
        # y_eval = dataset.funcs[fidelity_level](x_eval_num)
        y_eval = dataset.eval_xnum(fidelity_level,x_eval_num)

    # Check if the data should be skipped because it is unmasked and is either NaN or out of bounds
    # Skipped data is not added to x_data, y_data
    if check_skip_vec(np.atleast_1d(y_eval),dataset.ds_ops,dataset.n_out):
        dataset.x_data_skipped[fidelity_level] = np.append(dataset.x_data_skipped[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
        dataset.y_data_skipped[fidelity_level] = np.append(dataset.y_data_skipped[fidelity_level],np.atleast_2d(y_eval),axis=0)
        dataset.n_samp_skipped[fidelity_level] += 1

    else: # add to x_data and y_data and use masking if needed
        print("Added unmasked data point {} --> {}".format(np.asarray(x_eval_num).astype(np.float32),np.asarray(y_eval).astype(np.float32)))
        dataset.x_data[fidelity_level] = np.append(dataset.x_data[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
        dataset.y_data[fidelity_level] = np.append(dataset.y_data[fidelity_level],np.atleast_2d(y_eval),axis=0)
        dataset.n_samp[fidelity_level] += 1
        # Check for NaNs and out of bounds y_data
        dataset.unmasked_data[fidelity_level] = np.append(dataset.unmasked_data[fidelity_level],np.full((1,dataset.n_out), True, dtype=bool),axis=0)
        for i_o in range(dataset.n_out):
            dataset.unmasked_data[fidelity_level][-1,i_o] = check_unmasked(dataset.y_data[fidelity_level][-1,i_o],dataset.ds_ops)
        # the function has been evaluated, so no need to add to the hero queue
        dataset.hero_todo[fidelity_level] = np.atleast_2d(np.append(dataset.hero_todo[fidelity_level],False)).T
        dataset.hero_task_id[fidelity_level] = np.atleast_2d(np.append(dataset.hero_task_id[fidelity_level],'None')).T

        # Visualize the next point to be added and visualize the surrogate that has not yet been trained on this point
        if viz_ops is not None:
            from .viz import viz_animate
            viz_animate(dataset,surrogate,viz_ops,frame_id)

    # At every point in the design space where a simulation is performed, recursively compute all lower fidelity level simulations there too
    if dataset.ds_ops.perform_lower_sims:
        pt_exists = False
        im1_fl = fidelity_level-1
        if im1_fl >= 0:
            for j_d_lower in range(dataset.n_samp[im1_fl]): # for all data in the level below
                if np.array_equal(dataset.x_data[fidelity_level][-1,:],dataset.x_data[im1_fl][j_d_lower,:]):
                    pt_exists = True
            if not pt_exists: # add the sim at the im1_fl level
                dataset.add_xnum_sample_6d(im1_fl,x_eval_num,surrogate0=surrogate0,surrogate1=surrogate1,surrogate2=surrogate2,surrogate3=surrogate3,surrogate4=surrogate4,surrogate5=surrogate5)

    surrogates = [surrogate0,surrogate1,surrogate2,surrogate3,surrogate4,surrogate5]
    for i in range(len(surrogates)):
        surrogate = surrogates[i]
        dataset.train_on_unmasked_data(surrogate)
    print("Trained 6 surrogates on new data.")

#########################################################
# Add a sim to the dataset and retrain the surrogate model using all unmasked data.
# Unlike add_xnum_sample, instead of running the sim locally, but it in the Hero task queue.
# Options control if this is a blocking or non-blocking operation. In the non-blocking case,
# options control if masking is used (temporary value).
# The x_eval_num argument has variable types converted to floats as SMT expects
def queue_hero_sample(dataset,fidelity_level,x_eval_num,surrogate,viz_ops,frame_id):
    # Prepare the sample input arguments
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

    # Add the sample to the Hero task queue
    if not dataset.ds_ops.hero_blocking and dataset.ds_ops.hero_masking: # don't store the sample in the persistent memory but rather use a temporary buffer (until the hero task completes and the output is collected)
        task_id = dataset.hero_objs_unallocated[fidelity_level].put_tasks([{"name": "test_"+str(fidelity_level)+"_"+str(dataset.n_samp_unallocated[fidelity_level]+1), "args": x_eval_str}])[0]
        dataset.n_samp_unallocated[fidelity_level] += 1
        dataset.x_data_unallocated[fidelity_level] = np.append(dataset.x_data_unallocated[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
        dataset.hero_task_id_unallocated[fidelity_level] = np.atleast_2d(np.append(dataset.hero_task_id_unallocated[fidelity_level],task_id)).T
    else: # allocate memory for this data point (either for the masked value or the final value if blocking)
        task_id = dataset.hero_objs[fidelity_level].put_tasks([{"name": "test_"+str(fidelity_level)+"_"+str(dataset.n_samp[fidelity_level]+1), "args": x_eval_str}])[0]
        
        if dataset.ds_ops.hero_masking: # set a placeholder value using the surrogate's prediction
            assert(surrogate is not None)
            y_eval = surrogate.predict_values(np.atleast_2d(x_eval_num),fidelity_level)[0]
            # Mark the data as masked
            dataset.unmasked_data[fidelity_level] = np.atleast_2d(np.append(dataset.unmasked_data[fidelity_level],np.full((1,dataset.n_out), False, dtype=bool))).T
            dataset.hero_todo[fidelity_level] = np.atleast_2d(np.append(dataset.hero_todo[fidelity_level],True)).T
            
        else: # wait for Hero worker to complete the simulation
            assert(dataset.ds_ops.hero_blocking)
            print('Wait until worker completes this blocking task.')
            while True:
                task_data = dataset.hero_objs[i_fl].get_task(task_id)
                if task_data["status"] == 'complete':
                    print(f'Task {task_id} completed with simulation result = {task_data["results_s3"]}')
                    y_eval_str = task_data["results"]["objective"]
                    # Parsing the string representation into a NumPy array
                    y_eval = np.fromstring(objective_str[1:-1], dtype=float, sep=' ')
                    break
                dataset.hero_objs[0].wait(1)

            # Check for NaNs and out of bounds y_data
            dataset.unmasked_data[i_fl] = np.append(dataset.unmasked_data[i_fl],np.full((1,dataset.n_out), True, dtype=bool),axis=0)
            for i_o in range(dataset.n_out):
                dataset.unmasked_data[i_fl][-1,i_o] = check_unmasked(dataset.y_data[i_fl][-1,i_o],dataset.ds_ops)
            dataset.hero_todo[fidelity_level] = np.atleast_2d(np.append(dataset.hero_todo[fidelity_level],False)).T
            
        dataset.n_samp[fidelity_level] += 1
        dataset.x_data[fidelity_level] = np.append(dataset.x_data[fidelity_level],np.atleast_2d(x_eval_num),axis=0)
        dataset.y_data[fidelity_level] = np.append(dataset.y_data[fidelity_level],np.atleast_2d(y_eval),axis=0)
        dataset.hero_task_id[fidelity_level] = np.atleast_2d(np.append(dataset.hero_task_id[fidelity_level],task_id)).T
        if surrogate is not None and dataset.ds_ops.hero_blocking: # if the simulation has been evaluated, retrain the surrogate if provided
            dataset.train_on_unmasked_data(surrogate)
        
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
# Similar to add_xnum_sample, but it does not run or queue the sim.
# Instead, this function just stores this point as a masked sample and stores
# the value of the surrogate model as the correpsonding y value.
# The x_eval_num argument has variable types converted to floats as SMT expects
def mask_xnum_sample(dataset,fidelity_level,x_eval_num,surrogate,viz_ops,frame_id):
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
    
    # At every point in the design space where a simulation is performed, compute all lower fidelity level simulations there too
    if dataset.ds_ops.perform_lower_sims:        
        pt_exists = False 
        im1_fl = fidelity_level-1
        if im1_fl >= 0:    
            for j_d_lower in range(dataset.n_samp[im1_fl]): # for all data in the level below
                if np.array_equal(dataset.x_data[fidelity_level][-1,:],dataset.x_data[im1_fl][j_d_lower,:]):
                    pt_exists = True
            if not pt_exists: # add the sim at the im1_fl level
                dataset.mask_xnum_sample(im1_fl,x_eval_num,surrogate)
        
    # Visualize the next point to add and the surrogate that has not yet been trained on this point
    if viz_ops is not None:
        from .viz import viz_animate
        viz_animate(dataset,surrogate,viz_ops,frame_id)

def mask_xnum_sample_6d(dataset,fidelity_level,x_eval_num,surrogate0,surrogate1,surrogate2,surrogate3,surrogate4,surrogate5,viz_ops=None,frame_id=None):
    #print("Adding masked to x_data: {}".format(x_eval_num))
    dataset.x_data[fidelity_level] = np.append(dataset.x_data[fidelity_level],np.atleast_2d(x_eval_num),axis=0)

    surrogates = [surrogate0,surrogate1,surrogate2,surrogate3,surrogate4,surrogate5]

    y_eval = np.zeros(len(surrogates))
    for i in range(len(y_eval)):
        surrogate = surrogates[i]
        y_eval[i] = surrogate.predict_values(np.atleast_2d(x_eval_num),fidelity_level)[0][0]
    dataset.y_data[fidelity_level] = np.append(dataset.y_data[fidelity_level],np.atleast_2d(y_eval),axis=0)
    #dataset.y_data[fidelity_level] = np.atleast_2d(np.append(dataset.y_data[fidelity_level], y_eval)).T
    dataset.n_samp[fidelity_level] += 1

    dataset.unmasked_data[fidelity_level] = np.append(dataset.unmasked_data[fidelity_level],np.full((1,dataset.n_out), False, dtype=bool),axis=0)
    #dataset.unmasked_data[fidelity_level] = np.atleast_2d(np.append(dataset.unmasked_data[fidelity_level],np.full((1,dataset.n_out), False, dtype=bool))).T

    if dataset.ds_ops.perform_lower_sims:
        pt_exists = False
        im1_fl = fidelity_level-1
        if im1_fl >= 0:
            for j_d_lower in range(dataset.n_samp[im1_fl]): # for all data in the level below
                if np.array_equal(dataset.x_data[fidelity_level][-1,:],dataset.x_data[im1_fl][j_d_lower,:]):
                    pt_exists = True
            if not pt_exists: # add the sim at the im1_fl level
                dataset.mask_xnum_sample_6d(im1_fl,x_eval_num,surrogate0,surrogate1,surrogate2,surrogate3,surrogate4,surrogate5)

    # Not sure how to do this for multiple surrogates
    if viz_ops is not None:
        from .viz import viz_animate
        viz_animate(dataset,surrogate0,viz_ops,frame_id)

def overwrite_data(dataset,fidelity_level, x_eval_num, y_eval, surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5):
    largest_index = -1
    x_eval_num = np.asarray(x_eval_num)
    print("Looking for {}".format(x_eval_num.astype(np.float32)))
    for i in range(len(dataset.x_data[fidelity_level])):
        print("Row {} is {} --> {}".format(i,dataset.x_data[fidelity_level][i].astype(np.float32),dataset.y_data[fidelity_level][i].astype(np.float32)))
        if np.array_equal(dataset.x_data[fidelity_level][i].astype(np.float32),x_eval_num.astype(np.float32)):
            largest_index = i 
    
    if largest_index != -1:
        print("Deleting masked x and y data at index {}".format(largest_index))
        dataset.x_data[fidelity_level] = np.delete(dataset.x_data[fidelity_level], largest_index, axis=0)
        dataset.y_data[fidelity_level] = np.delete(dataset.y_data[fidelity_level], largest_index, axis=0)
        dataset.unmasked_data[fidelity_level] = np.delete(dataset.unmasked_data[fidelity_level], largest_index, axis=0)
        dataset.n_samp[fidelity_level] -= 1
    else:
        print("Error: could not find x_eval_num")

    add_xnum_sample_6d(dataset,fidelity_level,x_eval_num,y_eval,viz_ops=None,frame_id=None,surrogate0=surrogate0,surrogate1=surrogate1,surrogate2=surrogate2,surrogate3=surrogate3,surrogate4=surrogate4,surrogate5=surrogate5)
    #print(np.shape(y_eval))
    #print(np.shape(dataset.y_data[fidelity_level]))
    #print(dataset.unmasked_data[fidelity_level])
    #print(np.shape(x_eval_num))
    #print(np.shape(dataset.x_data[fidelity_level]))

#########################################################
# Collect data from any sims in the hero queue that have finished. Retrain the surrogate if it is provided.
def sync_hero_results(dataset,surrogate,viz_ops):
    # if viz_ops is not None:
    #     from .viz import viz_animate

    # check for Hero tasks that have been tracked in x_data and y_data
    for i_fl in range(dataset.n_fl):
        for i in range(dataset.n_samp[i_fl]):
            if dataset.hero_todo[i_fl][i] == True:
                task_id = dataset.hero_task_id[i_fl][i][0]
                task_data = dataset.hero_objs[i_fl].get_task(task_id)
                if task_data["status"] == 'complete':
                    print(f'Found a complete task = {task_id} with simulation result = {task_data["results_s3"]}')
                    y_eval_str = task_data["results"]["objective"]
                    # Parsing the string representation into a NumPy array
                    y_eval = np.fromstring(objective_str[1:-1], dtype=float, sep=' ')
                    #y_eval = float(task_data["results"]["objective"]) # this only works for a scalar
                    dataset.y_data[i_fl][i,:] = np.atleast_2d(y_eval)
                    # Check for NaNs and out of bounds y_data
                    for i_o in range(dataset.n_out):
                        dataset.unmasked_data[i_fl][i,i_o] = check_unmasked(y_eval[i_o],dataset.ds_ops)
                    dataset.hero_todo[i_fl][i] = False
                    
                    # Visualize the next point to add and the surrogate that has not yet been trained on this point
                    # if viz_ops is not None:
                    #     viz_animate(dataset,surrogate,viz_ops,frame_id)

    # check for Hero tasks that have not been allocated in x_data, y_data, etc.
    for i_fl in range(dataset.n_fl):
        for i in range(dataset.n_samp_unallocated[i_fl]):
            task_id = dataset.hero_task_id_unallocated[i_fl][i][0]
            task_data = dataset.hero_objs_unallocated[i_fl].get_task(task_id)
            if task_data["status"] == 'complete':
                print(f'Found a complete task = {task_id} with simulation result = {task_data["results_s3"]}')
                y_eval_str = task_data["results"]["objective"]
                # Parsing the string representation into a NumPy array
                y_eval = np.fromstring(objective_str[1:-1], dtype=float, sep=' ')
                #y_eval = float(task_data["results"]["objective"]) # this only works for a scalar

                # Allocate memory in the persistent data structures
                dataset.n_samp[i_fl] += 1
                dataset.x_data[i_fl] = np.append(dataset.x_data[i_fl],np.atleast_2d(dataset.x_data_unallocated[i_fl][i]),axis=0)
                dataset.y_data[i_fl] = np.append(dataset.y_data[i_fl],np.atleast_2d(y_eval),axis=0)
                dataset.hero_todo[i_fl] = np.atleast_2d(np.append(dataset.hero_todo[i_fl],False)).T
                dataset.hero_task_id[i_fl] = np.atleast_2d(np.append(dataset.hero_task_id[i_fl],dataset.hero_task_id_unallocated[i_fl][i])).T
                # Check for NaNs and out of bounds y_data
                dataset.unmasked_data[i_fl] = np.append(dataset.unmasked_data[i_fl],np.full((1,dataset.n_out), True, dtype=bool),axis=0)
                for i_o in range(dataset.n_out):
                    dataset.unmasked_data[i_fl][-1,i_o] = check_unmasked(dataset.y_data[i_fl][-1,i_o],dataset.ds_ops)

                # Remove sample from the temporary data structures
                n_samp_unallocated[i_fl] -= 1
                dataset.x_data_unallocated[i_fl] = np.delete(dataset.x_data_unallocated[i_fl], i)
                dataset.hero_task_id_unallocated[i_fl] = np.delete(dataset.hero_task_id_unallocated[i_fl], i)

                # Visualize the next point to add and the surrogate that has not yet been trained on this point
                # if viz_ops is not None:
                #     viz_animate(dataset,surrogate,viz_ops,frame_id)
                    
    if surrogate is not None:
        dataset.train_on_unmasked_data(surrogate)

#########################################################
# Wait for workers to complete all tasks in Hero queues. And retrain the surrogate if it is provided.
def wait_for_workers(dataset,surrogate,viz_ops):
    print('Wait until workers complete all tasks in all hero queues.')
    while True:
        sync_hero_results(dataset,viz_ops)
        # Determine how many Hero tasks are outstanding (not finished or not processed yet). First just counting those with x_data and y_data allocated.
        total_in_queue = np.sum([np.sum(arr) for arr in dataset.hero_todo])
        # Also count those that are not yet allocated in x_data and y_data arrays.
        total_in_queue += np.sum(n_samp_unallocated)
        if total_in_queue == 0:
            print('Workers are done. All Hero queues are empty.')
            break
        else:
            print(f'Number of remaining tasks in hero queues = {total_in_queue}')
            dataset.hero_objs[0].wait(1)
    
    if surrogate is not None:
        dataset.train_on_unmasked_data(surrogate)

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
def train_on_all_data(dataset,surrogate,update_masked):
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

    if update_masked:
        dataset.train_on_unmasked_data(surrogate) # this updates the predictions at masked data locations since they will be used by the next step
        # if the unmasked data is already up to date, this step could be skipped for computational efficiency
    surrogate.train(dataset.x_data, dataset.y_data)
