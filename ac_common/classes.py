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
            if not hasattr(params[i], 'minVal'):
                raise Exception('minVal not specified for param '+str(i))
            if not hasattr(params[i], 'maxVal'):
                raise Exception('maxVal not specified for param '+str(i))
        elif params[i].type == 'ordered':
            if not hasattr(params[i], 'minVal'):
                raise Exception('minVal not specified for param '+str(i))
            if not hasattr(params[i], 'maxVal'):
                raise Exception('maxVal not specified for param '+str(i))
        elif params[i].type == 'categorical':
            if hasattr(params[i], 'minVal'):
                raise Exception('minVal should not be specified for categorical params (param '+str(i)+')')
            if hasattr(params[i], 'maxVal'):
                raise Exception('maxVal should not be specified for categorical params (param '+str(i)+')')
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
    deterministic = True # random seeds are set deterministically
    acqFunc = 'EI'
    animation_1D = False
    animation_2D = False
    animation_ND = False
    plot_1D = False
    plot_2D = False
    plot_ND = False
    output_dir = './plots'
    n_iter = 15 # number of BayesOpt iterations
    # n_init_samp defaults to n_dim+1. It can be set larger but not smaller.
#########################################################    
def validate_options(options,n_fl,n_dim):
    import numpy as np
    if hasattr(options, 'existing_csv_filenames'):
        options.existing_csv_filenames = np.atleast_1d(options.existing_csv_filenames)
    n_init_samp_min = n_dim + 1
    if not hasattr(options, 'n_init_samp'):
        options.n_init_samp = [n_init_samp_min] * n_fl
    options.n_init_samp = np.atleast_1d(options.n_init_samp)
    if len(options.n_init_samp) != n_fl:
        raise Exception('Must list the number of initial samples for each function provided in funcs_in.')
    for i in range(n_fl):
        if options.n_init_samp[i] > 0:
            if options.n_init_samp[i] < n_init_samp_min:
                options.n_init_samp[i] = n_init_samp_min
    if hasattr(options, 'existing_csv_filenames'):
        if len(options.existing_csv_filenames) != n_fl:
            raise Exception('If any filenames are provided, must give a separate csv for each the functions provided in funcs_in. Use empty quotes if no data should be loaded for a fidelity level.')
        for filename in options.existing_csv_filenames:
            if not filename.endswith('.csv'):
                if filename != '':
                    raise Exception('csv filename must end in .csv or be an empty string')
        
    supported_acqFuns = ['EI','LCB','SBO','MSD']
    try:
        supported_acqFuns.index(options.acqFunc)
    except ValueError:
        raise Exception('Unrecognized acqFunc specified.')

    return True
#########################################################
### test code
if __name__ == "__main__":
    print('Checking validate_params:')
    x0 = Param(); x0.type = 'continuous'; x0.minVal = 0; x0.maxVal = 8
    x1 = Param(); x1.type = 'ordered'; x1.minVal = 2; x1.maxVal = 6
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
    options.plot_ND = True
    options.existing_csv_filenames = 'existing_data.csv'
    options.n_init_samp = 4 # must be >= n_dim+1, left unspecified, or set to zero if sufficient samples are provided in a .csv
    options.n_iter = 25 # number of BayesOpt iterations
    options.acqFunc = 'EI'
    print('This is a valid set of options:',validate_options(options,1,3))

    options.existing_csv_filenames = 'existing_data.txt'
    try:
        validate_options(options,1,3)
    except Exception:
        print('Correctly identified missing csv extension.')
        options.existing_csv_filenames = 'existing_data.csv'
    else:
        raise Exception('Did not identify a missing csv extension.')

    options.acqFunc = 'XX'
    try:
        validate_options(options,1,3)
    except Exception:
        print('Correctly identified unrecognized acqFunc.')
        options.acqFunc = 'EI'
    else:
        raise Exception('Did not identify unrecognized acqFunc.')
    options.existing_csv_filenames = 'existing_data.csv'

    options.existing_csv_filenames = ['','existing_data.csv','']
    options.n_init_samp = [2,4]
    try:
        validate_options(options,3,3)
    except Exception:
        print('Correctly identified missing entry in initial samples.')
        options.n_init_samp = [2,4,5]
    else:
        raise Exception('Did not identify missing entry in initial samples.')
    
    options.existing_csv_filenames = ['','existing_data.csvf','']
    try:
        validate_options(options,3,3)
    except Exception:
        print('Correctly identified filename that is neither an empty string nor an a .csv.')
        options.n_init_samp = [2,4,5]
    else:
        raise Exception('Failed to identify bad file extension.')
    



#########################################################