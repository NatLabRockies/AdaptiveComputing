#########################################################
# classes.py
#########################################################
#class Param:
class Param:
    type = 'continuous'
#########################################################    
def validate_params(params):
    import numpy as np
    params = np.atleast_1d(params)
    n_dim = len(params)
    for i in range(n_dim):
        params[i].type
        if params[i].type == 'continuous':
            if not hasattr(params[i], 'min_val'):
                raise Exception('min_val not specified for param '+str(i))
            if not hasattr(params[i], 'max_val'):
                raise Exception('max_val not specified for param '+str(i))
            if params[i].max_val <= params[i].min_val:
                raise Exception('max_val <= min_val for param '+str(i))
        elif params[i].type == 'ordered':
            if not hasattr(params[i], 'min_val'):
                raise Exception('min_val not specified for param '+str(i))
            if not hasattr(params[i], 'max_val'):
                raise Exception('max_val not specified for param '+str(i))
            if params[i].max_val <= params[i].min_val:
                raise Exception('max_val <= min_val for param '+str(i))
        elif params[i].type == 'categorical':
            if hasattr(params[i], 'min_val'):
                raise Exception('min_val should not be specified for categorical params (param '+str(i)+')')
            if hasattr(params[i], 'max_val'):
                raise Exception('max_val should not be specified for categorical params (param '+str(i)+')')
            if not hasattr(params[i], 'categories'):
                raise Exception('Categories not specified for param['+str(i)+'].')
            if len(params[i].categories) != len(set(params[i].categories)):
                raise Exception('Duplicates found in param['+str(i)+'].categories.')
            for c in params[i].categories:
                if not isinstance(c, str):
                    raise Exception('All categories of param['+str(i)+'] must be strings.')
        else:
            raise Exception('Unrecognized type for parameter '+str(i))
    return True
#########################################################
class Options:
    def __new__(cls): # make this class a singleton
        if not hasattr(cls, 'instance'):
            cls.instance = super(Options, cls).__new__(cls)
        return cls.instance
    # set the default options
    # surrogateModel = 'KRG'
    minimization_method = 'SLSQP'
    deterministic = True # random seeds are set deterministically
    acq_func = 'EI'
    animation_1d = False
    animation_2d = False
    animation_nd = False
    plot_1d = False
    plot_2d = False
    plot_nd = False
    output_dir = './plots'
    n_iter = 15 # number of optimization iterations in opt.opt
    n_opt_pts = 20 # number of initial guesses used to probe the acquisition function
    # n_init_samp defaults to n_dim+1. It can be set larger but not smaller.
#########################################################    
def validate_options(options,n_fl,n_dim):
    import numpy as np
    if hasattr(options, 'input_data_filenames'):
        options.input_data_filenames = np.atleast_1d(options.input_data_filenames)
    n_init_samp_min = n_dim + 1
    if not hasattr(options, 'n_init_samp'):
        options.n_init_samp = [n_init_samp_min] * n_fl
    options.n_init_samp = np.atleast_1d(options.n_init_samp)
    if len(options.n_init_samp) != n_fl:
        raise Exception('Must list the number of initial samples for each function provided in funcs_in.')
    for i in range(n_fl):
        if options.n_init_samp[i] > 0:
            if options.n_init_samp[i] < n_init_samp_min:
                print('Warning: The number of requested initial samples n_init_samp for fidelity level ' + str(i) + ' is being overwrittent to be its minimum allowable non-zero value of ' + str(n_init_samp_min) + str('.'))
                options.n_init_samp[i] = n_init_samp_min
    if hasattr(options, 'input_data_filenames'):
        if len(options.input_data_filenames) != n_fl:
            raise Exception('If any filenames are provided, must give a separate csv for each the functions provided in funcs_in. Use empty quotes if no data should be loaded for a fidelity level.')
        for filename in options.input_data_filenames:
            if not filename.endswith('.csv'):
                if filename != '':
                    raise Exception('csv filename must end in .csv or be an empty string')
        
    supported_acq_funcs = ['EI','LCB','SBO','MSD']
    try:
        supported_acq_funcs.index(options.acq_func)
    except ValueError:
        raise Exception('Unrecognized acq_func specified.')

    return True
#########################################################
### test code
if __name__ == "__main__":
    print('Checking validate_params:')
    x0 = Param(); x0.type = 'continuous'; x0.min_val = 0; x0.max_val = 8
    x1 = Param(); x1.type = 'ordered'; x1.min_val = 2; x1.max_val = 6
    x2 = Param(); x2.type = 'categorical'; x2.categories = ['a','b','c','d']
    params = [x0, x1, x2]
    print('This is a valid set of parameters:',validate_params(params))

    params[2].categories = ['a','b','c','b']
    try:
        validate_params(params)
    except Exception:
        print('Correctly identified a duplicate entry in categories.')
        params[2].categories = ['a','b','c','d']
    else:
        raise Exception('Did not identify a duplicate entry in categories.')

    params[2].categories = ['a','b','c',1]
    try:
        validate_params(params)
    except Exception:
        print('Correctly identified a nonstring entry in categories.')
        params[2].categories = ['a','b','c','d']
    else:
        raise Exception('Did not identify a nonstring entry in categories.')
    
    print('Checking that Options is a singleton:')
    singleton = Options()
    new_singleton = Options()
    print(singleton is new_singleton)

    print('Checking validate_options:')
    options = Options()
    options.plot_nd = True
    options.input_data_filenames = 'existing_data.csv'
    options.n_init_samp = 4 # must be >= n_dim+1, left unspecified, or set to zero if sufficient samples are provided in a .csv
    options.n_iter = 25 # number of BayesOpt iterations
    options.acq_func = 'EI'
    print('This is a valid set of options:',validate_options(options,1,3))

    options.input_data_filenames = 'existing_data.txt'
    try:
        validate_options(options,1,3)
    except Exception:
        print('Correctly identified missing csv extension.')
        options.input_data_filenames = 'existing_data.csv'
    else:
        raise Exception('Did not identify a missing csv extension.')

    options.acq_func = 'XX'
    try:
        validate_options(options,1,3)
    except Exception:
        print('Correctly identified unrecognized acq_func.')
        options.acq_func = 'EI'
    else:
        raise Exception('Did not identify unrecognized acq_func.')
    options.input_data_filenames = 'existing_data.csv'

    options.input_data_filenames = ['','existing_data.csv','']
    options.n_init_samp = [2,4]
    try:
        validate_options(options,3,3)
    except Exception:
        print('Correctly identified missing entry in initial samples.')
        options.n_init_samp = [2,4,5]
    else:
        raise Exception('Did not identify missing entry in initial samples.')
    
    options.input_data_filenames = ['','existing_data.csvf','']
    try:
        validate_options(options,3,3)
    except Exception:
        print('Correctly identified filename that is neither an empty string nor an a .csv.')
        options.n_init_samp = [2,4,5]
    else:
        raise Exception('Failed to identify bad file extension.')
    



#########################################################
