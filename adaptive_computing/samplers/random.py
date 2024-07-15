from adaptive_computing.samplers import SamplerBase
from smt.sampling_methods import Random

class RandomSampler(SamplerBase):
    """
    A class for sampling points using random sampling.
    
    Attributes:
        _rand_seed (int): Random seed for deterministic behavior.
        
    Methods:
        get_sample(N_samples): Generates samples using random sampling.
    """
    
    def __init__(self, dataset, rand_seed=-1):
        """
        Initializes the RandomSampler with the dataset and random seed.
        
        Args:
            dataset (DatasetBase): The dataset used to setup sampling limits.
            rand_seed (int): Random seed for deterministic behavior. Defaults to -1 for never repeating samples.
        """
        if rand_seed == -1:
            self._rand_seed = sum(array.shape[0] for array in dataset._x_data)
        else:
            self._rand_seed = rand_seed
        super().__init__(dataset)

    def get_sample(self, N_samples=1):
        """
        Generates samples using random sampling.
        
        Args:
            N_samples (int): The number of samples to generate. Defaults to 1.
        
        Returns:
            x (N samples, N input dimension): The generated samples.
        """
        sampling = Random(xlimits=self.x_limits)
        
        x = sampling(N_samples)
        # increment the random seed, so that future samples are not duplicates
        if self._rand_seed is not None:
            self._rand_seed += N_samples
        
        return x
