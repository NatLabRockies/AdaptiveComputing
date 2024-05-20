
class SurrogateModelBase:    
    def __init__(self, dataset):
        # initialize variables needed by all derived classes
        self.n_fl = dataset.n_fl
        self.multifidelity = dataset.multifidelity
        self.mixed_type = dataset.mixed_type

    def train(self, x_data, y_data):
        raise NotImplementedError("train method must be implemented in derived classes")

    def predict_values(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_values method must be implemented in derived classes")

    def predict_variances(self, x_data, fidelity_level=-1):
        raise NotImplementedError("predict_variances method must be implemented in derived classes")

