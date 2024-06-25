from adaptive_computing.samplers import SamplerBase
from smt.sampling_methods import Random

class RandomSampler(SamplerBase):
    def __init__(self,dataset,
                 rand_seed=-1): # set rand_seed=None for non-determinstic behavior
        if rand_seed == -1:
            self._rand_seed = sum(array.shape[0] for array in dataset._x_data)
        else:
            self._rand_seed = rand_seed
        super().__init__(dataset)

    def get_sample(self, N_samples=1):
        
        sampling = Random(xlimits=self.x_limits)
        
        x = sampling(N_samples)
        # increment the random seed, so that future samples are not duplicates
        if self._rand_seed is not None:
            self._rand_seed += N_samples
        
        return x
