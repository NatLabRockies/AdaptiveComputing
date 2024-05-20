from adaptive_computing.datasets import DatasetBase
from adaptive_computing.surrogates import SurrogateModelBase, surrogate_initializer
from adaptive_computing.samplers import LHSSampler, BayesianSampler
from adaptive_computing.samplers.acquisition_functions import expected_improvement

import numpy as np

class ActiveLoopDriverMF():
    def __init__(self,simulations, fidelity_costs, params, surrogate=None, dataset=None):
        self.params = params

        self.fidelity_costs = fidelity_costs
        self.evaluate_sample_mf = simulations
        self.n_fl = len(simulations)

        if dataset is None:
            self.dataset = DatasetBase(params, n_fidelity=self.n_fl)

        if isinstance(surrogate, SurrogateModelBase):
            self.surrogate = surrogate
        else:
            self.surrogate = surrogate_initializer(surrogate, 
                                                   self.dataset)
            
        self.init_sampler = LHSSampler(self.dataset)
        self.sampler = BayesianSampler(self.dataset, 
                                       expected_improvement)

        self._bopt_initialized = False
        
        self.evaluate_sample = lambda x, n_fidelity: \
            self.evaluate_sample_mf[n_fidelity](x)

    def _initialize_fidelity(self, n_fidelity, N_samples_init=3):
        x = self.init_sampler.get_sample(N_samples=N_samples_init)
        y = self.evaluate_sample(x,n_fidelity=n_fidelity)

        self.dataset.add_samples(x,y, n_fidelity=n_fidelity)

        

    def initialize(self, N_samples_init=3):
        for f_i in range(self.n_fl):
            self._initialize_fidelity(f_i, N_samples_init=N_samples_init)
        self.surrogate.train(self.dataset.x_data,
                        self.dataset.y_data)
        self._bopt_initialized = True

    def step(self):
        
        x_samples = np.zeros((self.n_fl, 1, self.dataset.n_in))
        objs = np.zeros(self.n_fl)
        for f_i in range(self.n_fl):
            x = self.sampler.get_sample(self.surrogate, self.dataset, f_i)
            y = self.sampler.acq_func(x, self.surrogate, self.dataset, f_i)

            x_samples[f_i] = x
            objs[f_i] = y

        fi_eval = np.argmin(objs/self.fidelity_costs)
        x = x_samples[fi_eval]
        y = self.evaluate_sample(x, fi_eval)

        self.dataset.add_samples(x,y, n_fidelity=fi_eval)
        self.surrogate.train(self.dataset.x_data,
                             self.dataset.y_data)


    def run(self, N_steps=None):
        if not self._bopt_initialized:
            self.initialize()

        for i in range(N_steps):
            self.step()
        