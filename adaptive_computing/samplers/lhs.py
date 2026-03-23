from adaptive_computing.samplers import SamplerBase
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod

class LHSSampler(SamplerBase):
    """
    A class for sampling points using Latin Hypercube Sampling (LHS).
    
    Attributes:
        _rand_seed (int): Random seed for deterministic behavior.
        
    Methods:
        _check_sample(N_samples): Checks if the number of samples is valid.
        get_sample(N_samples): Generates samples using LHS.
    """
    
    def __init__(self, dataset, rand_seed=-1):
        """
        Initializes the LHSSampler with the dataset and random seed.
        
        Args:
            dataset (DatasetBase): The dataset used to setup sampling limits.
            rand_seed (int): Random seed for deterministic behavior. Defaults to -1 for never repeating samples.
        """
        if rand_seed == -1:
            self._rand_seed = sum(array.shape[0] for array in dataset._x_data)
        else:
            self._rand_seed = rand_seed
        self.dataset = dataset  # Store dataset reference for mixed-type support  
        super().__init__(dataset)

    def _check_sample(self, N_samples):
        """
        Checks if the number of samples is valid.
        
        Args:
            N_samples (int): The number of samples to generate.
        
        Raises:
            Exception: If the number of samples is 1, which is invalid for Latin Hypercube Sampling.
        """
        if N_samples == 1:
            raise Exception('LatinHypercubeSampler requires n_lhs_samp == 0 or >= 2')

    def get_sample(self, N_samples=1):
        """
        Generates samples using Latin Hypercube Sampling (LHS).
        
        Args:
            N_samples (int): The number of samples to generate. Defaults to 1.
        
        Returns:
            x (N samples, N input dimension): The generated samples.
        
        Raises:
            Exception: If the number of samples is 1, which is invalid for Latin Hypercube Sampling.
        """
        self._check_sample(N_samples)

        if self.mixed_type:
            sampling = MixedIntegerSamplingMethod(self.x_types,
                                                  self.x_limits, 
                                                  LHS, criterion="maximin", 
                                                  seed=self._rand_seed)
        else:
            # Use regular LHS for continuous variables only
            sampling = LHS(xlimits=self.x_limits, 
                           criterion='maximin', 
                           seed=self._rand_seed)
        
        x = sampling(N_samples)
        
        # Post-process mixed-type samples to ensure proper data types
        if self.mixed_type:
            x = self._process_mixed_type_samples(x)
        
        # increment the random seed, so that future samples are not duplicates
        if self._rand_seed is not None:
            self._rand_seed += N_samples
        
        return x

    def _process_mixed_type_samples(self, x):
        """
        Post-process samples for mixed-type variables to ensure proper data types.
        
        Args:
            x (np.ndarray): Raw samples from LHS sampler
            
        Returns:
            np.ndarray: Processed samples with correct data types
        """
        import numpy as np
        x_processed = x.copy()
        
        for i, param in enumerate(self.dataset.params):
            if param.type == 'ordered':
                # Round to nearest integer and clamp to bounds
                x_processed[:, i] = np.round(x_processed[:, i]).astype(int)
                x_processed[:, i] = np.clip(x_processed[:, i], param.min_val, param.max_val)
            elif param.type == 'categorical':
                # Round to nearest integer (category index) and clamp to valid range
                x_processed[:, i] = np.round(x_processed[:, i]).astype(int)
                x_processed[:, i] = np.clip(x_processed[:, i], 0, len(param.categories) - 1)
            # continuous variables need no processing
        
        return x_processed
