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
        train(dataset): Trains the surrogate model using a dataset with automatic masking support.
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

    def _train_impl(self, x_data, y_data):
        """
        Internal training implementation that derived classes should override.
        This method receives only the validated, unmasked training data.
        
        Args:
            x_data (list): List of input data arrays for each fidelity level (unmasked only).
            y_data (list): List of output data arrays for each fidelity level (unmasked only).
        
        Raises:
            NotImplementedError: If the method is not implemented in derived classes.
        """
        raise NotImplementedError("_train_impl method must be implemented in derived classes")

    def train(self, dataset):
        """
        Trains the surrogate model using a dataset with automatic masking support.
        
        Always filters out masked data points to ensure only valid data is used for training.
        This provides a safe, consistent interface that respects data validity constraints.
        
        Args:
            dataset (DatasetBase): Dataset object with x_data, y_data, and masking information.
        """
        # Extract unmasked data from dataset
        # For single-output surrogates (SMT_GP, SOOGO_GP), filter by specific output dimension
        # For multi-output surrogates (TFMELT_*), require all outputs to be valid
        try:
            if hasattr(self, 'i_output'):
                # Single-output surrogate: filter by specific output dimension
                x_data, y_data = dataset.get_unmasked_data(i_output=self.i_output)
                filtering_msg = f"output dimension {self.i_output}"
            else:
                # Multi-output surrogate: require all outputs valid
                x_data, y_data = dataset.get_unmasked_data()
                filtering_msg = "all output dimensions"
            
            total_samples = sum(len(x) for x in x_data)
            original_samples = sum(len(x) for x in dataset.x_data)
            print(f"Training surrogate on {total_samples} unmasked samples (filtered from {original_samples} total samples, using {filtering_msg})")
        except Exception as e:
            raise ValueError(f"Could not extract unmasked data from dataset: {e}")
        
        # Validate the data
        x_data, y_data = self._validate_data(x_data, y_data)
        
        # Check if we have any data to train on
        total_samples = sum(len(x) for x in x_data)
        if total_samples == 0:
            raise ValueError("No training data available after masking. Cannot train surrogate model.")
        
        # Call the implementation-specific training method
        self._train_impl(x_data, y_data)

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
