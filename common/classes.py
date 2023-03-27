#########################################################
# classes.py
#########################################################
#class Param:
class Param:
    name = ''
    minVal = 0
    maxVal = 1
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
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
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