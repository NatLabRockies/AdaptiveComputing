from smt.applications.mixed_integer import (FLOAT, ORD, ENUM)
import numpy as np

class DatasetBase():
    def __init__(self, params, n_fidelity=1):
        self.n_fl = n_fidelity
        self.params = params
        assert(validate_params(self.params))

        # Store the number of continous parameters
        self.n_cont_vars = 0
        for i in range(self.n_in):
            if params[i].type == 'continuous':
                self.n_cont_vars += 1

        # Check if there are mixed types
        self.mixed_type = False
        for i in range(self.n_in):
            if self.params[i].type != 'continuous':
                self.mixed_type = True
                break
        self._set_xlimits()

    def _set_xlimits(self):

        # Define xlimits, the domain for the design parameters
        if self.mixed_type:
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

    @property
    def x_data(self):
        raise(NotImplementedError)

    @property
    def y_data(self):
        raise(NotImplementedError)

    def add_samples(self):
        raise(NotImplementedError)