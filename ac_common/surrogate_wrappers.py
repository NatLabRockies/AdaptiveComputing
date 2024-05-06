import numpy as np

class SurrogateModelBase:    
    def __init__(self, dataset, i_out=0):
        # initialize variables needed by all derived classes
        self.n_fl = dataset.n_fl
        self.multifidelity = dataset.multifidelity
        self.mixed_type = dataset.mixed_type
        self.i_out = i_out # indicates which output variable to model
        pass

    def train(self, x_data, y_data):
        raise NotImplementedError("train method must be implemented in derived classes")

    def predict_values(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_values method must be implemented in derived classes")

    def predict_variances(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_variances method must be implemented in derived classes")

# Implement surrogate modeling using the Surrogate Modeling Toolbox (SMT)
class SMTWrapper(SurrogateModelBase):
    def __init__(self, dataset, i_out=0):
        # Call the constructor of the base class
        super().__init__(dataset, i_out)
        
        # Initialize SMT-specific surrogate model
        # set upt the GPs Gaussian Process models (AKA the Kriging model)
        from smt.surrogate_models import KRG
        if self.multifidelity:
            from smt.applications.mfk import MFK
        if self.mixed_type:
            from smt.applications.mixed_integer import MixedIntegerSurrogateModel
        self.surrogate_model = []
        for i_fl in range(self.n_fl): # create at hierarchy of GPs
            if self.multifidelity and i_fl > 0:
                self.surrogate_model.append(MFK(print_global = False))
            else:
                self.surrogate_model.append(KRG(print_global = False)) 
            if self.mixed_type:
                self.surrogate_model[i_fl] = MixedIntegerSurrogateModel(surrogate=self.surrogate_model[i_fl], xtypes=dataset.xtypes, xlimits=dataset.xlimits)
        #dataset.train_on_unmasked_data(self)

    def train(self, x_data, y_data):
        for i_fl in range(self.n_fl):
            # Set the training values for all levels below the current fidelity level by indicating the name field
            # Note: other fidelities are accessed with names from 0 to n_fl-2 listed in order of increasing fidelity.
            for ii_fl in range(i_fl):
                self.surrogate_model[i_fl].set_training_values(x_data[ii_fl],y_data[ii_fl][:,self.i_out], name=ii_fl)
            # Set the training values for the current fidelity level
            # Note: the name field should not be specified for the current fidelity level    )
            self.surrogate_model[i_fl].set_training_values(x_data[i_fl],y_data[i_fl][:,self.i_out])
            
            # Update the surrogate model using this data
            self.surrogate_model[i_fl].train()

    def predict_values(self, x_data, fidelity_level=-1):
        return self.surrogate_model[fidelity_level].predict_values(np.atleast_2d(x_data))

    def predict_variances(self, x_data, fidelity_level=-1):
        return self.surrogate_model[fidelity_level].predict_variances(np.atleast_2d(x_data))    


# Implement surrogate modeling using the Surrogate Modeling Toolbox (SMT)
class ConstrainedSMTWrapper(SMTWrapper):
    def __init__(self, dataset, constraint_func, i_out=0):
        # Call the constructor of the base class
        super().__init__(dataset, i_out)
        self.constraint_func = constraint_func
        
    def predict_constraint(self, x_data, fidelity_level=-1):
        return self.constraint_func(x_data)


# # To implement a new type of surrogate model, copy the following code and
# # implement the methods using function calls to your desired surrogate modeling library.
# class ExampleWrapper(SurrogateModelBase):
#     # n_fl: number of fidelity levels
#     # multifidelity: boolean indicates if more than 1 fidelity level
#     # mixed_type: boolean indicates if non-float types are used (enumerated or ordered types)
#     def __init__(self, dataset, i_out=0, ...):
#         # Call the constructor of the base class
#         super().__init__(dataset, i_out)
        
#         # Initialize the surrogate model. Might use additional arguments
#         # self.surrogate_model = ...
#         pass
    
#         # Then, you likely want to train the surrogate model using the existing dataset in the dataset
#         dataset.train_on_unmasked_data(self)

#     # Train the surrogate model using the provided samples space coordinates x_data and the function values y_data
#     # x_data: is the location in the sample space of the training data. It is a list of length n_fl.
#     # Each entry is a numpy array of size n_samp[i_fl] (number of samples for the given fidelity level) x n_in (dimension size of the sample space)
#     # y_data: is the function values for the training data. It is a list of length n_fl.
#     # Each entry is a numpy array of size n_samp[i_fl] (number of samples for the given fidelity level) x n_out (dimension size of the output space, but only 
#     # the i_out direction should be used by the surrogate model since surrogate models only support one output right now)
#     def train(self, x_data, y_data):
#         # Example code: self.surrogate_model.add_training_point(x_data[0:n_fl][0:n_samp,0:n_in], y_data[0:n_fl][0:n_samp,i_out])
#         pass

#     # Evaluate the expected value of the surrogate model on specified fidelity level at the specified x coordinates
#     # x_data: is the location in the sample space of the training data. It is a list of length n_fl.
#     # Each entry is a numpy array of size n_samp[i_fl] (number of samples for the given fidelity level) x n_in (dimension size of the sample space)
#     # fidelity_level: interger in the range 0,..., n_fl-1 indicating which fidelity level to evaluate
#     def predict_values(self, x_data, fidelity_level=-1):
#         # Example code: self.surrogate_model.eval(x_data[0:n_fl][0:n_samp,0:n_in],fidelity_level=0) 
#         pass

#     # Evaluate the variance of the surrogate model on specified fidelity level at the specified x coordinates
#     # x_data: is the location in the sample space of the training data. It is a list of length n_fl.
#     # Each entry is a numpy array of size n_samp[i_fl] (number of samples for the given fidelity level) x n_in (dimension size of the sample space)
#     # fidelity_level: interger in the range 0,..., n_fl-1 indicating which fidelity level to evaluate
#     def predict_variances(self, x_data, fidelity_level=-1):
#         # Example code: self.surrogate_model.eval_variance(x_data[0:n_fl][0:n_samp,0:n_in],fidelity_level=0) 
#         pass
