import numpy as np

class SurrogateModelBase:    
    def __init__(self, n_fl, multifidelity, mixed_type):
        # initialize variables needed by all derived classes
        self.n_fl = n_fl
        self.multifidelity = multifidelity
        self.mixed_type = mixed_type
        pass

    def train(self, x_data, y_data):
        raise NotImplementedError("train method must be implemented in derived classes")

    def predict_values(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_values method must be implemented in derived classes")

    def predict_variances(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_variances method must be implemented in derived classes")

# Implement surrogate modeling using the Surrogate Modeling Toolbox (SMT)
class SMTWrapper(SurrogateModelBase):
    def __init__(self, n_fl, multifidelity, mixed_type, xlimits, xtypes):
        # Call the constructor of the base class
        super().__init__(n_fl, multifidelity, mixed_type)
        
        # Initialize SMT-specific surrogate model
        # set upt the GPs Gaussian Process models (AKA the Kriging model)
        from smt.surrogate_models import KRG
        if multifidelity:
            from smt.applications.mfk import MFK
        if mixed_type:
            from smt.applications.mixed_integer import MixedIntegerSurrogateModel
        self.surrogate_model = []
        for i_fl in range(self.n_fl): # create at hierarchy of GPs
            if self.multifidelity and i_fl > 0:
                self.surrogate_model.append(MFK(print_global = False))
            else:
                self.surrogate_model.append(KRG(print_global = False)) 
            if self.mixed_type:
                self.surrogate_model[i_fl] = MixedIntegerSurrogateModel(surrogate=self.surrogate_model[i_fl], xtypes=xtypes, xlimits=xlimits)

    def train(self, x_data, y_data):
        #self.surrogate_model.set_training_values(x_data, y_data)
        for i_fl in range(self.n_fl):
            # Set the training values for all levels below the current fidelity level by indicating the name field
            # Note: other fidelities are accessed with names from 0 to n_fl-2 listed in order of increasing fidelity.
            for ii_fl in range(i_fl):
                self.surrogate_model[i_fl].set_training_values(x_data[ii_fl],y_data[ii_fl], name=ii_fl)
            # Set the training values for the current fidelity level
            # Note: the name field should not be specified for the current fidelity level    
            self.surrogate_model[i_fl].set_training_values(x_data[i_fl],y_data[i_fl])
            
            # Update the surrogate model using this data
            self.surrogate_model[i_fl].train()

    def predict_values(self, x_data, fidelity_level=-1):
        return self.surrogate_model[fidelity_level].predict_values(np.atleast_2d(x_data))

    def predict_variances(self, x_data, fidelity_level=-1):
        return self.surrogate_model[fidelity_level].predict_variances(np.atleast_2d(x_data))    
        
# To implement a new type of surrogate model, copy the preceding SMTWrapper code,
# implement the methods using function calls to your desired
# surrogate modeling library.
