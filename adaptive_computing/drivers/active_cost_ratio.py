from adaptive_computing.datasets import DatasetBase
from adaptive_computing.surrogates import SurrogateModelBase, surrogate_initializer
from adaptive_computing.samplers import LHSSampler, BayesianSampler
from adaptive_computing.samplers.acquisition_functions import expected_improvement
from adaptive_computing.evaluators import BaseEvaluator
from adaptive_computing.drivers import ActiveLoopDriver
import numpy as np

class ActiveLoopDriverCostRatio(ActiveLoopDriver):

    def get_next_sample(self):
        x_samples = np.zeros((self.n_fl, 1, self.dataset.n_in))
        objs = np.zeros(self.n_fl)
        for f_i in range(self.n_fl):
            x = self.sampler.get_sample(self.surrogate, self.dataset, f_i)
            y = self.sampler.acq_func(x, self.surrogate, self.dataset, f_i)

            x_samples[f_i] = x
            objs[f_i] = y

        fi_eval = np.argmin(objs/self.fidelity_costs)
        x = x_samples[fi_eval]

        return x, fi_eval


  