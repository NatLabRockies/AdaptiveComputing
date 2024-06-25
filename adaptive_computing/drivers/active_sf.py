from adaptive_computing.datasets import DatasetBase
from adaptive_computing.surrogates import SurrogateModelBase, surrogate_initializer
from adaptive_computing.samplers import LHSSampler, BayesianSampler
from adaptive_computing.samplers.acquisition_functions import expected_improvement
from adaptive_computing.evaluators import BaseEvaluator
class ActiveLoopDriverSF():
    def __init__(self,simulation, params, surrogate=None, dataset=None,
                 nan_behavior='fail'):
        
        self.params = params

        if dataset is None:
            self.dataset = DatasetBase(params)

        self.evaluator = BaseEvaluator(simulation,
                            n_in=len(self.params))

        if isinstance(surrogate, SurrogateModelBase):
            self.surrogate = surrogate
        else:
            self.surrogate = surrogate_initializer(surrogate, 
                                                   self.dataset)
            
        self.init_sampler = LHSSampler(self.dataset)
        self.sampler = BayesianSampler(self.dataset, 
                                       expected_improvement)

        self._bopt_initialized = False

        self.nan_behavior =  nan_behavior

    def initialize(self, N_samples_init=3):
        x = self.init_sampler.get_sample(N_samples=N_samples_init)
        y = self.evaluator.evaluate_points(x)
        self.dataset.add_samples(x,y, n_fidelity=0)
        
        self.surrogate.train(self.dataset.x_data,
                             self.dataset.y_data)

        self._bopt_initialized = True

    def step(self):

        x = self.sampler.get_sample(self.surrogate, self.dataset)
        y = self.evaluator.evaluate_points(x)
        self.dataset.add_samples(x,y, n_fidelity=0)
        self.surrogate.train(self.dataset.x_data,
                             self.dataset.y_data)


    def run(self, N_steps=None):
        if not self._bopt_initialized:
            self.initialize()

        for i in range(N_steps):
            self.step()

    @property
    def nan_behavior(self):
        return self._nan_behavior
    
    @nan_behavior.setter
    def nan_behavior(self, nan_behavior):
        self._nan_behavior = nan_behavior
        self.dataset.nan_behavior = nan_behavior

    def add_points(self, points):
        for x in points:
            y = self.evaluate_sample(x)
            self.dataset.add_samples(x,y, n_fidelity=0, surrogate=self.surrogate)
