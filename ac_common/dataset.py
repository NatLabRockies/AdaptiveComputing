# dataset.py
import numpy as np
from .classes import validate_params

class DataSet:
    def __init__(self, simulations, params, ds_ops, n_out=1):
        self.n_out = n_out # dimension of the output space (number of simulation outputs)
        # Check the number of fidelity levels
        self.simulations = np.atleast_1d(simulations)
        self.n_fl = len(self.simulations) # number of fidelity levels
        print(f'Number of fidelity levels detected from len(simulations) = {self.n_fl}')
        self.multifidelity = False
        if self.n_fl != 1:
            self.multifidelity = True

        self.params = params
        assert(validate_params(self.params))
        self.n_in = len(self.params)

        # Store the number of continous parameters
        self.n_cont_vars = 0
        for i in range(self.n_in):
            if params[i].type == 'continuous':
                self.n_cont_vars += 1

        self.ds_ops = ds_ops
        if self.ds_ops.use_hero:
            # Hero setup ...
            # Set Hero environment variables
            import os
            os.environ["HERO_PROJECT"] = "adaptive-computing"
            os.environ["HERO_QUEUE"] = "queue-1"
            os.environ["HERO_CLIENT_ID"] = "f4om7c738a1um7fgjao6msve7"
            os.environ["HERO_CLIENT_SECRET"] = "mbk361rg0eedkd6k34t5cujukl19clbv50qnteqi829gnpufkde"
            os.environ["HERO_QUEUE_VISIBILITY_TIMEOUT"] = "60"
            os.environ["HERO_DATABASE_PASSWORD"] = "8fc2a2e2-ed9e-413d-996a-72da94e11c5c"
            # Add the path to hero
            import sys
            sys.path.insert(0, '/projects/acldrd/kgriffin/hero/')
            from hero import Hero
            from hero.api.task import COMPLETE, READY, CLAIMED, FAILED
            self.hero_objs = np.array([], dtype=object)
            for i_fl in range(self.n_fl):
                self.hero_objs = np.append(self.hero_objs,Hero(queue=str(i_fl)))
                self.hero_objs[i_fl].clear_tasks()
        
        # Check if there are mixed types
        self.mixed_type = False
        for i in range(self.n_in):
            if self.params[i].type != 'continuous':
                self.mixed_type = True
                break

        # self.funcs =[]
        # for i in range(self.n_fl):
        #     self.funcs.append(ComposedFunction(self.simulations[i],self.params))

        # Define xlimits, the domain for the design parameters
        if self.mixed_type:
            from smt.applications.mixed_integer import (FLOAT, ORD, ENUM)
            self.xtypes = []
            self.xlimits = [] # this is the domain for the user defined simulations[] (which may include mixed types)
            self.xlimits_num = [] # this is the domain with the categoricals and integers converted to continuous types. Categoricals are a list of floats.
            for i in range(self.n_in):
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
            self.xlimits = np.zeros([self.n_in,2]) # the first dimension is the parameter space (self.n_in), the second defines the bounds (min/max) for each parameter
            for i in range(self.n_in):    
                self.xlimits[i,:] = [self.params[i].min_val, self.params[i].max_val]
            self.xlimits_num = self.xlimits
            self.xtypes = ['float_type']*self.n_in
        
        # declare DataSet data
        # n_samp[i] will be incremented for each Latin hypercube sample, sample read from an input file, or Bayesian optimization iteration performed.
        self.n_samp = np.zeros(self.n_fl).astype(int)
        # the sample-space coordinates:
        self.x_data = [np.empty([0,self.n_in])]*self.n_fl # x_data is a list of length n_fl. Each entry will be an n_samp[i_fl] x n_in np array
        # the function values at these coordiantes:
        self.y_data = [np.empty([0,self.n_out])]*self.n_fl # y_data is a list of length n_fl. Each entry will be an n_samp[i_fl] x n_out np array
        # boolean indicating if the sample's output is non-NaN and within user-specified bounds
        self.unmasked_data = [np.empty([0,self.n_out],dtype=bool)]*self.n_fl # unmasked_data is a list of length n_fl. Each entry will be an n_samp[i_fl] x n_out np array
        # boolean indicating if the sample evaluation is queued to be run by a Hero worker
        self.hero_todo = [np.empty([0,1],dtype=bool)]*self.n_fl # hero_todo is a list of length n_fl. Each entry will be an n_samp[i_fl] x 1 np array
        # string generated by hero to identify each simulation run's inputs and outputs
        self.hero_task_id = [np.empty([0,1])]*self.n_fl # hero_task_id is a list of length n_fl. Each entry will be an n_samp[i_fl] x 1 np array

    def train_on_unmasked_data(self,surrogate):
        from ac_common.static_sampling import train_on_unmasked_data
        train_on_unmasked_data(self,surrogate)
    
    def train_on_all_data(self,surrogate):
        from ac_common.static_sampling import train_on_all_data
        train_on_all_data(self,surrogate)
    
    def add_lhs_samples(self,n_lhs_samp,surrogate=None):
        from ac_common.static_sampling import add_lhs_samples
        add_lhs_samples(self,n_lhs_samp,surrogate)
    
    def add_file_samples(self,filenames,surrogate=None):
        from ac_common.static_sampling import add_file_samples
        add_file_samples(self,filenames,surrogate)

    def add_bo_samples(self,n_iter,surrogate,bo_ops=None,viz_ops=None):
        from ac_common.bo import add_bo_samples
        add_bo_samples(self,n_iter,surrogate,bo_ops,viz_ops)

    def add_xnum_sample(self,fidelity_level,x_eval_num,y_eval=None,viz_ops=None,frame_id=None,surrogate=None):
        from ac_common.static_sampling import add_xnum_sample
        add_xnum_sample(self,fidelity_level,x_eval_num,y_eval,viz_ops,frame_id,surrogate)

    def queue_hero_sample(self,fidelity_level,x_eval_num,surrogate=None,viz_ops=None,frame_id=None):
        from ac_common.static_sampling import queue_hero_sample
        queue_hero_sample(self,fidelity_level,x_eval_num,surrogate,viz_ops,frame_id)
    
    def sync_hero_results(self,surrogate=None,viz_ops=None):
        from ac_common.static_sampling import sync_hero_results
        sync_hero_results(self,surrogate,viz_ops)

    def wait_for_workers(self,surrogate=None,viz_ops=None):
        from ac_common.static_sampling import wait_for_workers
        wait_for_workers(self,surrogate,viz_ops)

    def native_to_num(self,x_eval_native):
        from ac_common.static_sampling import native_to_num
        return native_to_num(self,x_eval_native)

    def bounds_check_xnative(self,x_eval_native):
        from ac_common.static_sampling import bounds_check_xnative
        bounds_check_xnative(self,x_eval_native) 

    def find_min(self,surrogate):
        from ac_common.bo import find_min
        return find_min(self,surrogate)
    
    def find_max(self,surrogate):
        from ac_common.bo import find_max
        return find_max(self,surrogate)
    
    def write_samples_csv(self,filenames):
        from ac_common.utils import write_samples_csv
        return write_samples_csv(self,filenames)
    
    def query(self,surrogate,x_queries,fidelity_level=-1,threshold_std=None,threshold_std_mean=None,threshold_std_tv=None):
        from ac_common.query import query
        return query(self,surrogate,x_queries,fidelity_level,threshold_std,threshold_std_mean,threshold_std_tv)
    
    #########################################################
    # This runs a simulation without using Hero. So it runs on the AC main process and is blocking.
    def eval_xnum(self,fidelity_level,x_num):
        x_native = num_to_native(x_num, self.params)
        try:
            y_eval = self.simulations[fidelity_level](x_native)
        except ValueError:
            print('Caught a ValueError in user-defined simulation and setting to NaN.')
            y_eval = np.NaN
        return y_eval

#########################################################
# In order to pickle the DataSet object, no local functions
# can be defined inside the DataSet, so funcs is set using
# class functions. In fact, composite class functions.
# simulation_i is the user-defined implementation of the simulations
# catch_valerr catches ValueErrors that may occur in g
# num_to_native converts the arguments of the SMT data types to the user-defined data types
# class ComposedFunction:
#     def __init__(self, simulation_i, params):
#         self.simulation_i = simulation_i
#         self.params = params

#     def __call__(self, x):
#         return self.catch_valerr(self.simulation_i, num_to_native(x, self.params))
    
#     # Wrap error catching around the user-defined simulation functions.
#     def catch_valerr(self,func,x):
#         try:
#             val = func(x)
#         except ValueError:
#             print('Caught a ValueError in user-defined simulation and setting to NaN.')
#             val = np.NaN
#         return val

#########################################################
# For mixed type functions, need to convert SMT's internal representation
# of the data to be compatible with the data types in user-defined functions
# SMT stores ordered types as floats, so this function casts them to ints
# SMT stores categorical types as floats, so this function converts them to strings
def num_to_native(x,params):
    n_in = len(params)
    #x = x.tolist()
    x_return = [] #np.empty_like(x, dtype=object)
    for i in range(n_in):
        if params[i].type == 'ordered':
            x_return.append(x[i].astype(int)) # the int cast is needed because SMT stores ordered variables as floats
        elif params[i].type == 'categorical':
            x_return.append(params[i].categories[x[i].astype(int)]) # the int cast is needed because SMT stores enums as floats
        elif params[i].type == 'continuous':
            x_return.append(x[i])
        else:
            raise Exception("Unrecognized parameter type.")
    return x_return
