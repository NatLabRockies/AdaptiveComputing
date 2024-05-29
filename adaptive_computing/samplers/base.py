
class SamplerBase():
    def __init__(self, dataset):
        self.mixed_type = dataset.mixed_type
        self.ndim = dataset.n_in
        self.x_limits = dataset.x_limits
        self.x_types = dataset.x_types

    def get_sample(self, N_samples=1):
        raise NotImplementedError("Must implement get sample method")