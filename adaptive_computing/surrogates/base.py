import numpy as np

class SurrogateModelBase:
    """
    A base class for surrogate models.
    
    Attributes:
        n_fidelity (int): Number of fidelity levels.
        multifidelity (bool): Indicates if the model is multifidelity.
        mixed_type (bool): Indicates if the dataset has mixed types.
        nan_behavior (str): Behavior when encountering NaN values.
        
    Methods:
        _validate_data_i(x_data, y_data, fidelity_level): Validates and processes data for a specific fidelity level.
        _validate_data(x_data, y_data): Validates and processes data for all fidelity levels.
        train(x_data, y_data): Trains the surrogate model (must be implemented in derived classes).
        predict_values(x_data, fidelity_level): Predicts values using the surrogate model (must be implemented in derived classes).
        predict_variances(x_data, fidelity_level): Predicts variances using the surrogate model (must be implemented in derived classes).
    """
    
    def __init__(self, dataset):
        """
        Initializes the SurrogateModelBase with the dataset.
        
        Args:
            dataset (DatasetBase): The dataset to used to setup surrogate.
        """
        # initialize variables needed by all derived classes
        self.n_fidelity = dataset.n_fidelity
        self.multifidelity = dataset.multifidelity
        self.mixed_type = dataset.mixed_type
        self.nan_behavior = dataset.nan_behavior

    def _validate_data_i(self, x_data, y_data, fidelity_level=-1):
        """
        Validates and processes data for a specific fidelity level.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            y_data (N samples, N output dimension): Output data.
            fidelity_level (int): The fidelity level. Defaults to -1.
        
        Returns:
            tuple: The validated input and output data.
        """
        if self.nan_behavior == 'mask_replace' and np.any(np.isnan(y_data)):
            y_pred = self.predict_values(x_data)
            y_data[np.isnan(y_data)] = y_pred[np.isnan(y_data)]

        return x_data, y_data
    
    def _validate_data(self, x_data, y_data):
        """
        Validates and processes data for all fidelity levels.
        
        Args:
            x_data (list): List of input data arrays for each fidelity level.
                Each element of list has shape (N samples, N input dimension)
            y_data (list): List of output data arrays for each fidelity level.
                Each element of list has shape (N samples, N output dimension)
        
        Returns:
            tuple: The validated input and output data arrays.
        """
        for i in range(self.n_fidelity):
            x_data[i], y_data[i] = self._validate_data_i(x_data[i], y_data[i], i)
        return x_data, y_data

    def train(self, x_data, y_data):
        """
        Trains the surrogate model.
        
        Args:
            x_data (list): List of input data arrays for each fidelity level.
                Each element of list has shape (N samples, N input dimension)
            y_data (list): List of output data arrays for each fidelity level.
                Each element of list has shape (N samples, N output dimension)
        
        Raises:
            NotImplementedError: If the method is not implemented in derived classes.
        """
        raise NotImplementedError("train method must be implemented in derived classes")

    def predict_values(self, x_data, fidelity_level=-1):
        """
        Predicts values using the surrogate model.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): The fidelity level. Defaults to -1.
        
        Returns:
            np.ndarray: The predicted values.
        
        Raises:
            NotImplementedError: If the method is not implemented in derived classes.
        """
        raise NotImplementedError("predict_values method must be implemented in derived classes")

    def predict_variances(self, x_data, fidelity_level=-1):
        """
        Predicts variances using the surrogate model.
        
        Args:
            x_data (N samples, N input dimension): Input data.
            fidelity_level (int): The fidelity level. Defaults to -1.
        
        Returns:
            np.ndarray: The predicted variances.
        
        Raises:
            NotImplementedError: If the method is not implemented in derived classes.
        """
        raise NotImplementedError("predict_variances method must be implemented in derived classes")
