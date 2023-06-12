# query.py
import numpy as np
#########################################################
# Query the highest level multifidelity GP at a numpy array of points specified by the sample space coordinates x_queries
# Returns y_queries
def query(model,x_queries):
    assert(x_queries.shape[1]==model.n_dim)
    n_queries = x_queries.shape[0]

    # XXX could add optional argument for which fidelity level to query. Default to highest level.

    # bounds checking for the queries
    # XXX this could be added but isn't necessary

    y_queries = np.zeros([n_queries,1])

    for i in range(n_queries):
        # convert any categorical entries in x_query to be numbers
        x_query_num = np.zeros([1,model.n_dim])
        for j in range(model.n_dim):
            if model.params[j].type == 'categorical':
                x_query_num[0,j] = model.params[j].categories.index(x_queries[i,j])
            elif (model.params[j].type == 'continuous') or (model.params[j].type == 'ordered'):
                x_query_num[0,j] = x_queries[i,j]
            else:
                raise Exception('Unrecognized type for parameter '+str(i))   
        y_queries[i] = model.gprs[-1].predict_values(x_query_num)
    return y_queries

#########################################################
