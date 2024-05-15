
class SamplerBase():
    def __init__(self, dataset):
        self.mixed_type = dataset.mixed_type
        self.ndim = dataset.n_in
        self.xlimits = dataset.xlimits
        self.xtypes = dataset.xtypes

    def get_sample(self, N_samples=1):
        raise NotImplementedError("Must implement get sample method")