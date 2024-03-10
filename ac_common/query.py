# query.py
import numpy as np
#########################################################
# Query the highest level multifidelity GP at a numpy array of points specified by the sample space coordinates x_queries
# Queries the highest fidelity level (-1) unless a different level is specified
# Each x_query is a list of variables with each having the data type specified by user's list of Params
# Returns y_queries, y_queries_var
def query(dataset,surrogate,x_queries,fidelity_level,threshold_std,threshold_std_mean,threshold_std_tv):
    if len(x_queries.shape) == 1: # if x_queries is a 1d array
        x_queries = x_queries[:, np.newaxis]
    assert(x_queries.shape[1]==dataset.n_in)
    n_queries = x_queries.shape[0]

    if threshold_std is not None:
        assert(threshold_std > 0.0)
    if threshold_std_mean is not None:
        assert(threshold_std_mean > 0.0)
    if threshold_std_tv is not None:
        assert(threshold_std_tv > 0.0)

    y_queries = np.zeros([n_queries,1])
    y_queries_var = np.zeros([n_queries,1])
    x_queries_num = np.zeros([n_queries,dataset.n_in])

    for i in range(n_queries):
        # Bounds checking for the queries
        dataset.bounds_check_xnative(x_queries[i,:])

        # Convert any categorical entries in x_query to be numbers
        x_queries_num[i,:] = dataset.native_to_num(x_queries[i,:])  

        # Evaluate the surrogate model
        y_queries[i] = surrogate.predict_values(np.atleast_2d(x_queries_num[i]),fidelity_level)[0]
        y_queries_var[i] = surrogate.predict_variances(np.atleast_2d(x_queries_num[i]),fidelity_level)[0]

        # Run simulation at all points where the measured standard deviation >= user-specified threshold value
        if threshold_std is not None:
            if np.sqrt(y_queries_var[i]) >= threshold_std:
                # conduct a simulation and retrain the GPR using this data
                dataset.add_xnum_sample(fidelity_level,x_queries_num[i],surrogate=surrogate)

        # Run simulation at all points where the std/mean >= user-specified threshold value
        if threshold_std_mean is not None:
            if np.sqrt(y_queries_var[i]) >= threshold_std_mean*y_queries[i]:
                # conduct a simulation and retrain the GPR using this data
                dataset.add_xnum_sample(fidelity_level,x_queries_num[i],surrogate=surrogate)

        # Run simulation at all points where the std/total_variation >= user-specified threshold value
        if threshold_std_tv is not None:
            total_variation = dataset.find_max(surrogate)[1][0] - dataset.find_min(surrogate)[1][0]
            if np.sqrt(y_queries_var[i]) >= threshold_std_tv*total_variation:
                # conduct a simulation and retrain the GPR using this data
                dataset.add_xnum_sample(fidelity_level,x_queries_num[i],surrogate=surrogate)
        
    # Re-evaluate the surrogate model because some new simulations may have been conducted
    if (threshold_std is not None) or (threshold_std_mean is not None) or (threshold_std_tv is not None):
        for i in range(n_queries):    
            y_queries[i] = surrogate.predict_values(np.atleast_2d(x_queries_num[i]),fidelity_level)[0]
            y_queries_var[i] = surrogate.predict_variances(np.atleast_2d(x_queries_num[i]),fidelity_level)[0]

    return y_queries, y_queries_var

def query_cpp(dataset,surrogate,x_queries,fidelity_level,threshold_std,threshold_std_mean,threshold_std_tv):
    if len(x_queries.shape) == 1: # if x_queries is a 1d array
        x_queries = x_queries[:, np.newaxis]
    assert(x_queries.shape[1]==dataset.n_in)
    n_queries = x_queries.shape[0]

    if threshold_std is not None:
        assert(threshold_std > 0.0)
    if threshold_std_mean is not None:
        assert(threshold_std_mean > 0.0)
    if threshold_std_tv is not None:
        assert(threshold_std_tv > 0.0)

    y_queries = np.zeros([n_queries,1])
    y_queries_var = np.zeros([n_queries,1])
    x_queries_num = np.zeros([n_queries,dataset.n_in])

    for i in range(n_queries):
        # Bounds checking for the queries
        dataset.bounds_check_xnative(x_queries[i,:])

        # Convert any categorical entries in x_query to be numbers
        x_queries_num[i,:] = dataset.native_to_num(x_queries[i,:])  

        # Evaluate the surrogate model
        y_queries[i] = surrogate.predict_values(np.atleast_2d(x_queries_num[i]),fidelity_level)[0]
        y_queries_var[i] = surrogate.predict_variances(np.atleast_2d(x_queries_num[i]),fidelity_level)[0]

        # Run simulation at all points where the measured standard deviation >= user-specified threshold value
        if threshold_std is not None:
            if np.sqrt(y_queries_var[i]) >= threshold_std:
                # conduct a simulation and retrain the GPR using this data
                return None

        # Run simulation at all points where the std/mean >= user-specified threshold value
        if threshold_std_mean is not None:
            if np.sqrt(y_queries_var[i]) >= threshold_std_mean*y_queries[i]:
                # conduct a simulation and retrain the GPR using this data
                return None

        # Run simulation at all points where the std/total_variation >= user-specified threshold value
        if threshold_std_tv is not None:
            total_variation = dataset.find_max(surrogate)[1][0] - dataset.find_min(surrogate)[1][0]
            if np.sqrt(y_queries_var[i]) >= threshold_std_tv*total_variation:
                # conduct a simulation and retrain the GPR using this data
                return None            

    return y_queries

    
#########################################################
