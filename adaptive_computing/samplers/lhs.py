from adaptive_computing.samplers import SamplerBase
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod

class LHSSampler(SamplerBase):
    def __init__(self,dataset,
                 rand_seed=-1): # set rand_seed=None for non-determinstic behavior
        if rand_seed == -1:
            self._rand_seed = sum(array.shape[0] for array in dataset._x_data)
        else:
            self._rand_seed = rand_seed
        super().__init__(dataset)

    def _check_sample(self, N_samples):
        if N_samples == 1:
            raise Exception('LatinHypercubeSampler requires n_lhs_samp ==0 or >=2')


    def get_sample(self, N_samples=1):
        self._check_sample(N_samples)

        if self.mixed_type:
            sampling = MixedIntegerSamplingMethod(self.x_types,
                                                  self.x_limits, 
                                                  LHS, criterion="maximin", 
                                                  random_state=self._rand_seed)
        else:
            sampling = LHS(xlimits=self.x_limits, 
                           criterion='maximin', 
                           random_state=self._rand_seed)
        
        x = sampling(N_samples)
        # increment the random seed, so that future samples are not duplicates
        if self._rand_seed is not None:
            self._rand_seed += N_samples
        
        return x
