# query.py
import numpy as np
#########################################################
# Query the highest level multifidelity GP at a numpy array of points specified by the sample space coordinates x_queries
# Queries the highest fidelity level (-1) unless a different level is specified
# Each x_query is a list of variables with each having the data type specified by user's list of Params
# Returns y_queries, y_queries_var
def query(model,x_queries,fidelity_level,threshold_std):
    if len(x_queries.shape) == 1: # if x_queries is a 1d array
        x_queries = x_queries[:, np.newaxis]
    assert(x_queries.shape[1]==model.n_dim)
    n_queries = x_queries.shape[0]

    if threshold_std is not None:
        assert(threshold_std > 0.0)

    y_queries = np.zeros([n_queries,1])
    y_queries_var = np.zeros([n_queries,1])
    x_queries_num = np.zeros([n_queries,model.n_dim])

    for i in range(n_queries):
        # Bounds checking for the queries
        model.bounds_check_xnative(x_queries[i,:])

        # Convert any categorical entries in x_query to be numbers
        x_queries_num[i,:] = model.native_to_num(x_queries[i,:])  

        # Evaluate the surrogate model
        y_queries[i] = model.gprs[fidelity_level].predict_values(np.atleast_2d(x_queries_num[i]))[0]
        y_queries_var[i] = model.gprs[fidelity_level].predict_variances(np.atleast_2d(x_queries_num[i]))[0]

        # Run simulation at all points where the measured standard deviation >= user-specified threshold value
        if threshold_std is not None:
            if np.sqrt(y_queries_var[i]) >= threshold_std:
                # conduct a simulation and retrain the GPR using this data
                model.add_xnum_sample(fidelity_level,x_queries_num[i])
        
    # Re-evaluate the surrogate model because some new simulations may have been conducted
    if threshold_std is not None:
        for i in range(n_queries):    
            y_queries[i] = model.gprs[fidelity_level].predict_values(x_queries_num[i])
            y_queries_var[i] = model.gprs[fidelity_level].predict_variances(x_queries_num[i])

    return y_queries, y_queries_var

#########################################################
