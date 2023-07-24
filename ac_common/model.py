# Model.py
import numpy as np
from .classes import validate_params

class Model:
    def __init__(self, simulations, params, mod_ops):
        # Check the number of fidelity levels
        self.simulations = np.atleast_1d(simulations)
        self.n_fl = len(self.simulations) # number of fidelity levels
        self.multifidelity = False
        if self.n_fl != 1:
            self.multifidelity = True

        self.params = params
        assert(validate_params(self.params))
        self.n_dim = len(self.params)

        # Store the number of continous parameters
        self.n_cont_vars = 0
        for i in range(self.n_dim):
            if params[i].type == 'continuous':
                self.n_cont_vars += 1

        self.mod_ops = mod_ops
        
        # Check if there are mixed types
        self.mixed_type = False
        for i in range(self.n_dim):
            if self.params[i].type != 'continuous':
                self.mixed_type = True
                break

        self.funcs =[]
        for i in range(self.n_fl):
            self.funcs.append(ComposedFunction(self.simulations[i],self.params))

        # Define xlimits, the domain for the design parameters
        if self.mixed_type:
            from smt.applications.mixed_integer import (FLOAT, ORD, ENUM)
            self.xtypes = []
            self.xlimits = [] # this is the domain for the user defined simulations[] (which may include mixed types)
            self.xlimits_num = [] # this is the domain for self.funcs[] which assumes the categoricals and integers have been converted to continuous types. Categoricals are a list of floats.
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

    def train_on_unmasked_data(self):
        from ac_common.static_sampling import train_on_unmasked_data
        train_on_unmasked_data(self)
    
    def train_on_all_data(self):
        from ac_common.static_sampling import train_on_all_data
        train_on_all_data(self)
    
    def add_lhs_samples(self,n_lhs_samp):
        from ac_common.static_sampling import add_lhs_samples
        add_lhs_samples(self,n_lhs_samp)
    
    def add_file_samples(self,filenames):
        from ac_common.static_sampling import add_file_samples
        add_file_samples(self,filenames)

    def add_bo_samples(self,n_iter,bo_ops=None,viz_ops=None):
        from ac_common.bo import add_bo_samples
        add_bo_samples(self,n_iter,bo_ops,viz_ops)

    def add_xnum_sample(self,fidelity_level,x_eval_num,y_eval=None,viz_ops=None,frame_id=None):
        from ac_common.static_sampling import add_xnum_sample
        add_xnum_sample(self,fidelity_level,x_eval_num,y_eval,viz_ops,frame_id)

    def native_to_num(self,x_eval_native):
        from ac_common.static_sampling import native_to_num
        return native_to_num(self,x_eval_native)

    def bounds_check_xnative(self,x_eval_native):
        from ac_common.static_sampling import bounds_check_xnative
        bounds_check_xnative(self,x_eval_native) 

    def find_min(self):
        from ac_common.bo import find_min
        return find_min(self)
    
    def write_samples_csv(self,filenames):
        from ac_common.utils import write_samples_csv
        return write_samples_csv(self,filenames)
    
    def query(self,x_queries,fidelity_level=-1,threshold_std=None):
        from ac_common.query import query
        return query(self,x_queries,fidelity_level,threshold_std)

#########################################################
# In order to pickle the Model object, no local functions
# can be defined inside the Model, so funcs is set using
# class functions. In fact, composite class functions.
# simulation_i is the user-defined implementation of the simulations
# catch_valerr catches ValueErrors that may occur in g
# num_to_native converts the arguments of the SMT data types to the user-defined data types
class ComposedFunction:
    def __init__(self, simulation_i, params):
        self.simulation_i = simulation_i
        self.params = params

    def __call__(self, x):
        return self.catch_valerr(self.simulation_i, num_to_native(x, self.params))
    
    # Wrap error catching around the user-defined simulation functions.
    def catch_valerr(self,func,x):
        try:
            val = func(x)
        except ValueError:
            print('Caught a ValueError in user-defined simulation and setting to NaN.')
            val = np.NaN
        return val

#########################################################
# For mixed type functions, need to convert SMT's internal representation
# of the data to be compatible with the data types in user-defined functions
# SMT stores ordered types as floats, so this function casts them to ints
# SMT stores categorical types as floats, so this function converts them to strings
def num_to_native(x,params):
    n_dim = len(params)
    #x = x.tolist()
    x_return = [] #np.empty_like(x, dtype=object)
    for i in range(n_dim):
        if params[i].type == 'ordered':
            x_return.append(x[i].astype(int)) # the int cast is needed because SMT stores ordered variables as floats
        elif params[i].type == 'categorical':
            x_return.append(params[i].categories[x[i].astype(int)]) # the int cast is needed because SMT stores enums as floats
        elif params[i].type == 'continuous':
            x_return.append(x[i])
        else:
            raise Exception("Unrecognized parameter type.")
    return x_return
#########################################################
