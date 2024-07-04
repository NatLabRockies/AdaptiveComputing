from adaptive_computing.datasets import DatasetBase
from adaptive_computing.surrogates import SurrogateModelBase, surrogate_initializer
from adaptive_computing.samplers import LHSSampler, BayesianSampler
from adaptive_computing.samplers.acquisition_functions import expected_improvement
from adaptive_computing.evaluators import BaseEvaluator
from adaptive_computing.drivers import ActiveLoopDriver
import numpy as np

class ActiveLoopDriverCostRatio(ActiveLoopDriver):
    """
    Active learning loop driver for adaptive sampling using 
    surrogate models with cost ratio optimization.

    Inherits from ActiveLoopDriver.

    Methods:
        get_next_sample():
            Retrieves the next sample to evaluate using cost ratio optimization.
    """

    def get_next_sample(self):
        """
        Retrieves the next sample to evaluate using cost ratio optimization 
        based on objective function values divided by fidelity costs.

        Returns:
            np.ndarray: Next sample to evaluate.
            int: Fidelity level index for the sample.
        """
        x_samples = np.zeros((self.n_fl, 1, self.dataset.n_in))
        objs = np.zeros(self.n_fl)

        # Iterate over fidelity levels
        for f_i in range(self.n_fl):
            x = self.sampler.get_sample(self.surrogate, self.dataset, f_i)
            y = self.sampler.acq_func(x, self.surrogate, self.dataset, f_i)

            x_samples[f_i] = x
            objs[f_i] = y

        # Determine fidelity level based on minimum objective function value divided by fidelity costs
        fi_eval = np.argmin(objs / self.fidelity_costs)
        x = x_samples[fi_eval]

        return x, fi_eval
