# Model.py
import numpy as np
from .classes import validate_params

class Model:
    def __init__(self, simulations, params, options):
        # Check the number of fidelity levels
        self.simulations = np.atleast_1d(simulations)
        self.n_fl = len(self.simulations) # number of fidelity levels
        self.multifidelity = False
        if self.n_fl != 1:
            self.multifidelity = True

        self.params = params
        assert(validate_params(self.params))
        self.n_dim = len(self.params)

        self.options = options
        
        # Check if there are mixed types
        self.mixed_type = False
        for i in range(self.n_dim):
            if self.params[i].type != 'continuous':
                self.mixed_type = True
                break

        self.funcs =[]
        for i in range(self.n_fl):
            if self.mixed_type:
                func_int = lambda x, i=i: self.simulations[i](args_str_2_enum(x,self.params))
                self.funcs.append(lambda v, func_int=func_int: catch_valerr(func_int,v))
            else:
                func_int = lambda x, i=i: self.simulations[i](x)
                self.funcs.append(lambda v, func_int=func_int: catch_valerr(func_int,v))

        # Define xlimits, the domain for the design parameters
        if self.mixed_type:
            from smt.applications.mixed_integer import (FLOAT, ORD, ENUM)
            self.xtypes = []
            self.xlimits = [] # this is the domain for the user defined simulations[] (which may include mixed types)
            self.xlimits_num = [] # this is the domain for self.funcs[] which assumes the categoricals and integers have been converted to continuous types
            for i in range(self.n_dim):
                if self.params[i].type == 'continuous':
                    self.xtypes.append(FLOAT)
                    self.xlimits.append([self.params[i].min_val, self.params[i].max_val])
                    self.xlimits_num.append([self.params[i].min_val, self.params[i].max_val])
                elif self.params[i].type == 'ordered':
                    self.xtypes.append(ORD)
                    self.xlimits.append([self.params[i].min_val, self.params[i].max_val])
                    self.xlimits_num.append([self.params[i].min_val, self.params[i].max_val])
                elif self.params[i].type == 'categorical':
                    self.xtypes.append((ENUM, len(self.params[i].categories)))
                    self.xlimits.append(self.params[i].categories)
                    self.xlimits_num.append(list(range(len(self.params[i].categories))))
                else:
                    raise Exception('Unrecognized type for parameter '+str(i)) 
        else:
            self.xlimits = np.zeros([self.n_dim,2]) # the first dimension is the parameter space (self.n_dim), the second defines the bounds (min/max) for each parameter
            for i in range(self.n_dim):    
                self.xlimits[i,:] = [self.params[i].min_val, self.params[i].max_val]
            self.xlimits_num = self.xlimits
        
        # declare model data
        # n_samp[i] will be incremented for each Latin hypercube sample, sample read from an input file, or Bayesian optimization iteration performed.
        self.n_samp = np.zeros(self.n_fl).astype(int)
        # the sample-space coordinates:
        self.x_data = [np.empty([0,self.n_dim])]*self.n_fl # x_data is a list of length n_fl. Each entry will be an n_samp x n_dim np array
        # the function values at these coordiantes:
        self.y_data = [np.empty([0,1])]*self.n_fl # y_data is a list of length n_fl. Each entry will be an n_samp x 1 np array
        # boolean indicating if the data is non-NaN and within user-specified bounds
        self.unmasked_data = [np.empty([0,1])]*self.n_fl # unmasked_data is a list of length n_fl. Each entry will be an n_samp x 1 np array

        # set upt the GPs Gaussian Process models (AKA the Kriging model)
        from smt.surrogate_models import KRG
        if self.multifidelity:
            from smt.applications.mfk import MFK
        if self.mixed_type:
            from smt.applications.mixed_integer import MixedIntegerSurrogateModel
        self.gprs = []
        for i_fl in range(self.n_fl): # create at hierarchy of gprs
            if self.multifidelity and i_fl > 0:
                self.gprs.append(MFK(print_global = False))
            else:
                self.gprs.append(KRG(print_global = False)) 
            if self.mixed_type:
                self.gprs[i_fl] = MixedIntegerSurrogateModel(surrogate=self.gprs[i_fl], xtypes=self.xtypes, xlimits=self.xlimits)

    def retrain(self):
        from ac_common.static_sampling import retrain
        retrain(self)
    
    def add_lhs_samples(self,n_lhs_samp):
        from ac_common.static_sampling import add_lhs_samples
        add_lhs_samples(self,n_lhs_samp)
    
    def add_file_samples(self,filenames):
        from ac_common.static_sampling import add_file_samples
        add_file_samples(self,filenames)

    def add_bo_samples(self,n_iter,bo_ops=None,ani_ops=None):
        from ac_common.bo import add_bo_samples
        add_bo_samples(self,n_iter,bo_ops,ani_ops)

    def find_min(self):
        from ac_common.bo import find_min
        return find_min(self)
    
    def write_samples_csv(self,filenames):
        from ac_common.utils import write_samples_csv
        return write_samples_csv(self,filenames)
    
    def query(self,x_queries,fidelity_level=-1):
        from ac_common.query import query
        return query(self,x_queries,fidelity_level)

#########################################################
def catch_valerr(func,x):
    try:
        val = func(x)
    except ValueError:
        print('Caught a ValueError in user-defined simulation and setting to NaN.')
        val = np.NaN
    return val
#########################################################
def args_str_2_enum(x,params):
    # for mixed type functions, the user defined function may have 
    # categorical args which are strings. But SMT represents these
    # as enumerated types (integers). Need to convert enumerated
    # args to strings.
    # Also, need to cast enumerataed args to ints, which shouldn't be necessary, but is helping
    n_dim = len(params)
    x = x.tolist()
    for i in range(n_dim):
        if params[i].type == 'ordered':
            x[i] = int(x[i]) # the int cast is needed because SMT stores ordered variables as floats
        elif params[i].type == 'categorical':
            x[i] = params[i].categories[int(x[i])] # the int cast is needed because SMT stores enums as floats
    return x