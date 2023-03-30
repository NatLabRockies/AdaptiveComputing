#########################################################
# classes.py
#########################################################
#class Param:
class Param:
    type = 'continuous'
#########################################################    
def validate_params(params):
    ndim = len(params)
    for i in range(ndim):
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
    # surrogateModel = 'gaussianProcess'
    deterministic = True # random seeds are set deterministically
    acqFunc = 'EI'
    animation_1D = False
    animation_2D = False
    animation_ND = False
    plot_1D = False
    plot_2D = False
    plot_ND = False
    #output_dir = '~/codes/AdaptiveComputing/tutorials/example_1d/plots'
    output_dir = './plots'
    n_iter = 15 # number of BayesOpt iterations
    # initial_samples defaults to ndim+1. It can be set larger but not smaller.
#########################################################    
def validate_options(options):
    if hasattr(options, 'existing_csv_filename'):
        if not options.existing_csv_filename.endswith('.csv'):
            raise Exception('csv name must end in .csv')
    supported_acqFuns = ['EI','LCB','SBO','MSD']
    try:
        supported_acqFuns.index(options.acqFunc)
    except ValueError:
        raise Exception('Unrecognized acqFunc specified.')
    
    return True
#########################################################
### test code
if __name__ == "__main__":
    print('Checking that Options is a singleton:')
    singleton = Options()
    new_singleton = Options()
    print(singleton is new_singleton)

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
    
    print('Checking validate_options:')
    options = Options()
    options.plot_ND = True
    options.existing_csv_filename = 'existing_data.csv'
    options.initial_samples = 0 # must be >= ndim+1, left unspecified, or set to zero if sufficient samples are provided in a .csv
    options.n_iter = 25 # number of BayesOpt iterations
    options.acqFunc = 'EI'
    print('This is a valid set of options:',validate_options(options))

    options.existing_csv_filename = 'existing_data.txt'
    try:
        validate_options(options)
    except Exception:
        print('Correctly identified missing csv extension.')
        options.existing_csv_filename = 'existing_data.csv'
    else:
        raise Exception('Did not identify a missing csv extension.')

    options.acqFunc = 'XX'
    try:
        validate_options(options)
    except Exception:
        print('Correctly identified unrecognized acqFunc.')
        options.acqFunc = 'EI'
    else:
        raise Exception('Did not identify unrecognized acqFunc.')
    options.existing_csv_filename = 'existing_data.csv'

#########################################################