# query.py
import numpy as np
#########################################################
# Query the highest level multifidelity GP at a numpy array of points specified by the sample space coordinates x_queries
# Queries the highest fidelity level (-1) unless a different level is specified
# Returns y_queries, y_queries_var
def query(model,x_queries,fidelity_level=-1):
    if len(x_queries.shape) == 1: # if x_queries is a 1d array
        x_queries = x_queries[:, np.newaxis]
    assert(x_queries.shape[1]==model.n_dim)
    n_queries = x_queries.shape[0]

    # XXX this could be added but isn't necessary since the surrogate model is defined for all x

    y_queries = np.zeros([n_queries,1])
    y_queries_var = np.zeros([n_queries,1])

    for i in range(n_queries):
        # Bounds checking for the queries
        for j in range(model.n_dim):
            if model.params[j].type == 'categorical':
                if x_queries[i,j] not in model.params[j].categories:
                    raise Exception('Query ' + str(j) + ' of parameter x' + str(i) + ' = '+str(x_queries[i,j])+' is not a valid value for categorical parameter.')
            elif (model.params[j].type == 'continuous') or (model.params[j].type == 'ordered'):
                if x_queries[i,j] < model.params[j].min_val or x_queries[i,j] > model.params[j].max_val:
                    raise Exception('Query ' + str(j) + ' is out of bounds with the value of parameter x' + str(i) + ' = '+str(x_queries[i,j])+' .')
            else:
                raise Exception('Unrecognized type for parameter x'+str(i))

        # Convert any categorical entries in x_query to be numbers
        x_query_num = np.zeros([1,model.n_dim])
        for j in range(model.n_dim):
            if model.params[j].type == 'categorical':
                x_query_num[0,j] = model.params[j].categories.index(x_queries[i,j])
            elif (model.params[j].type == 'continuous') or (model.params[j].type == 'ordered'):
                x_query_num[0,j] = x_queries[i,j]
            else:
                raise Exception('Unrecognized type for parameter '+str(i))   
        
        # Evaluate the surrogate model
        y_queries[i] = model.gprs[fidelity_level].predict_values(x_query_num)
        y_queries_var[i] = model.gprs[fidelity_level].predict_variances(x_query_num)

    return y_queries, y_queries_var

#########################################################
