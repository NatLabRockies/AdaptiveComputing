class SamplerBase():
    """
    A base class for sampling from a dataset.
    
    Attributes:
        mixed_type (bool): Whether the dataset has mixed types.
        ndim (int): The number of input dimensions.
        x_limits (np.ndarray): The limits for each input dimension.
        x_types (list): The types of each input dimension.
        
    Methods:
        get_sample(N_samples): Abstract method to get samples. Must be implemented in subclasses.
    """
    
    def __init__(self, dataset):
        """
        Initializes the SamplerBase with the dataset's attributes.
        
        Args:
            dataset (DatasetBase): The dataset to sample from.
        """
        self.mixed_type = dataset.mixed_type
        self.ndim = dataset.n_in
        self.x_limits = dataset.x_limits
        self.x_types = dataset.x_types

    def get_sample(self, N_samples=1):
        """
        Abstract method to get samples. Must be implemented in subclasses.
        
        Args:
            N_samples (int): The number of samples to generate. Defaults to 1.
        
        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Must implement get_sample method")
