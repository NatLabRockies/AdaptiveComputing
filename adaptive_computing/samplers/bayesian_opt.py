from adaptive_computing.samplers import SamplerBase

class BayesianSampler(SamplerBase):
    def __init__(self,dataset, acquistion_function, rand_seed=None):
        self._rand_seed = rand_seed
        self.acq_func = acquistion_function
        super().__init__(dataset)

    def get_sample(self, surrogate, dataset, N_samples=1):
        pass