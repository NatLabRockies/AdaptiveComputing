from adaptive_computing.datasets import DatasetBase

class HeroDataset(DatasetBase):
    def __init__(self, params, n_fidelity=1, data_repo=None):
        super().__init__(params, n_fidelity=n_fidelity)

        self.data_repo = data_repo

    @property
    def x_data(self):
        # somehow load the data from data repo
        pass

    @property
    def y_data(self):
        #somehow load the data from the data repo
        pass
