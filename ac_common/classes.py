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
                raise Exception('categories not specified for param '+str(i))
        else:
            raise Exception('Unrecognized type for parameter '+str(i))
    return
#########################################################
class Options:
    def __new__(cls): # make this class a singleton
        if not hasattr(cls, 'instance'):
            cls.instance = super(Options, cls).__new__(cls)
        return cls.instance
    # set the default options
    surrogateModel = 'gaussianProcess'
    acquFunc = 'EI'
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
### test code
if __name__ == "__main__":
    print('Checking that Params is a singleton: ')
    singleton = Params()
    new_singleton = Params()
    print(singleton is new_singleton)

    print('Checking that Options is a singleton: ')
    singleton = Options()
    new_singleton = Options()
    print(singleton is new_singleton)
#########################################################