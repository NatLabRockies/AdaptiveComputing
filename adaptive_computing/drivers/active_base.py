from adaptive_computing.datasets import DatasetBase
from adaptive_computing.surrogates import SurrogateModelBase, surrogate_initializer
from adaptive_computing.samplers import LHSSampler, BayesianSampler
from adaptive_computing.samplers.acquisition_functions import expected_improvement
from adaptive_computing.evaluators import BaseEvaluator
from adaptive_computing.drivers.query_validators import get_query_validator
import numpy as np

class ActiveLoopDriver():
    def __init__(self,simulations, params, surrogate=None, dataset=None,
                 nan_behavior='fail', fidelity_costs=None):
        
        self.params = params

        if dataset is None:
            self.dataset = DatasetBase(params)
        
        self.n_fl = len(simulations)
        self.evaluators = [BaseEvaluator(simulation, n_in=len(self.params)) for
                           simulation in simulations]
        
        self.fidelity_costs = fidelity_costs

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

    def get_next_sample(self, f_i=0):
        x = self.sampler.get_sample(self.surrogate, self.dataset, f_i)
        return x, f_i

    def step(self):

        x, fi_eval = self.get_next_sample()
        y = self.evaluate_sample(x, fi_eval)

        self.dataset.add_samples(x,y, n_fidelity=fi_eval)
        self.surrogate.train(self.dataset.x_data,
                             self.dataset.y_data)


    def run(self, N_steps=None):
        if not self._bopt_initialized:
            self.initialize()

        for i in range(N_steps):
            self.step()

    def add_points(self, points):
        for x in points:
            y = self.evaluate_sample(x, n_fidelity=0)
            self.dataset.add_samples(x,y, n_fidelity=0)

    def evaluate_sample(self, points, n_fidelity):
        return self.evaluators[n_fidelity].evaluate_points(points)
    

    def query(self, points, error_criterion, **kwargs):
        points = np.asarray(points)
        values = np.zeros((points.shape[0],1))
        validator = get_query_validator(criterion=error_criterion)

        for i in range(points.shape[0]):
            surrogate_value = self.surrogate.predict_values(points[[i]])
            surrogate_variance = self.surrogate.predict_variances(points[[i]])
            valid = validator(surrogate_value,surrogate_variance, **kwargs)

            if not valid:
                print(f"Error too large for point {points[i]}, running simulation")
                y = self.evaluate_sample(points[[i]], n_fidelity=0)
                self.dataset.add_samples(points[[i]],y, n_fidelity=0)

                self.surrogate.train(self.dataset.x_data,
                                    self.dataset.y_data)
                
                values[i] = y
            else:
                values[i] = surrogate_value

        return values


    @property
    def nan_behavior(self):
        return self._nan_behavior
    
    @nan_behavior.setter
    def nan_behavior(self, nan_behavior):
        self._nan_behavior = nan_behavior
        self.dataset.nan_behavior = nan_behavior