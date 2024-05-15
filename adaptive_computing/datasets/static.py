import numpy as np
from adaptive_computing.datasets import BaseDataset


class StaticDataset(BaseDataset):
    def __init__(self, params, n_fidelity=1, n_out=1):
        """
        Dataset where all values are static and can be stored
        in memory as numpy arrays.
        """
        super().__init__(params, n_fidelity)
        self.n_out = n_out

        self._x_data = [np.empty([0,self.n_in])]*self.n_fl # x_data is a list of length n_fl. Each entry will be an n_samp[i_fl] x n_in np array
        # the function values at these coordinates:
        self._y_data = [np.empty([0,self.n_out])]*self.n_fl # y_data is a list of length n_fl. Each entry will be an n_samp[i_fl] x n_out np array

    @property
    def x_data(self):
        return self._x_data
    
    @property
    def y_data(self):
        return self._y_data
    
    def add_samples(self, x_data, y_data, n_fidelity):
        
        self._x_data[n_fidelity] = np.concat(self._x_data[n_fidelity],
                                             x_data,axis=0)
        self._y_data[n_fidelity] = np.concat(self._y_data[n_fidelity],
                                             y_data,axis=0)
