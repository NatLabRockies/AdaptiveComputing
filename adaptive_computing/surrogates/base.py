import numpy as np

class SurrogateModelBase:    
    def __init__(self, dataset):
        # initialize variables needed by all derived classes
        self.n_fl = dataset.n_fl
        self.multifidelity = dataset.multifidelity
        self.mixed_type = dataset.mixed_type
        self.nan_behavior = dataset.nan_behavior

    def _validate_data_i(self, x_data, y_data, fidelity_level=-1):
        if self.nan_behavior == 'mask_replace' and np.any(np.isnan(y_data)) :
            
            y_pred  = self.predict_values(x_data)
            y_data[np.isnan(y_data)] = y_pred[np.isnan(y_data)]

        return x_data, y_data
    
    def _validate_data(self, x_data, y_data):
        for i in range(self.n_fl):
            x_data[i], y_data[i] = self._validate_data_i(x_data[i], y_data[i],i)
        return x_data, y_data

    def train(self, x_data, y_data):
        raise NotImplementedError("train method must be implemented in derived classes")

    def predict_values(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_values method must be implemented in derived classes")

    def predict_variances(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_variances method must be implemented in derived classes")

