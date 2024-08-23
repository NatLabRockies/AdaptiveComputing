
class SurrogateModelBase:    
    def __init__(self, dataset, i_out=0):
        # initialize variables needed by all derived classes
        self.n_fl = dataset.n_fl
        self.multifidelity = dataset.multifidelity
        self.mixed_type = dataset.mixed_type
        self.i_out = i_out # indicates which output variable to model
        self.threshold_std_dyn = 1
        pass

    def train(self, x_data, y_data):
        raise NotImplementedError("train method must be implemented in derived classes")

    def predict_values(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_values method must be implemented in derived classes")

    def predict_variances(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_variances method must be implemented in derived classes")


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
