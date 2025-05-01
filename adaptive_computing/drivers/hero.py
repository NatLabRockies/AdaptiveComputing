from adaptive_computing.drivers import ActiveLoopDriver
from adaptive_computing.datasets import HeroDataset
from time import sleep

import numpy as np

class ActiveLoopDriverHero(ActiveLoopDriver):
    def __init__(self, simulations, params, machine_names, surrogate=None, dataset=None,
                 nan_behavior='fail', fidelity_costs=None, acq_func='expected_improvement', blocking=False):
        self.use_hero = True
        if dataset is None:
            dataset = HeroDataset(params, machine_names, n_fidelity=1, blocking=blocking)
        self.dataset = dataset
        if blocking:
            retrain = True
        else:
            retrain = False # only retrain when wait hero_wait_for_data_and_train is called
        super().__init__(simulations, params, surrogate=surrogate, dataset=self.dataset,
                         nan_behavior=nan_behavior, fidelity_costs=fidelity_costs, acq_func=acq_func, retrain=retrain)

        for sim_i in simulations:
            assert(sim_i is None) # since the user has opted to use Hero, simulations should be set to a list of Nones of length n_fidelity and the definition of the simulations should be implemented in the worker script.
        self.evaluators = None
        
    def _initialize_fidelity(self, i_fidelity, N_samples_init=3):
        """
        Initializes a fidelity level with initial samples. No placeholder values for y_data are provided since these are random samples typically used to initialize the surrogate.

        Args:
            i_fidelity (int): Fidelity level index.
            N_samples_init (int, optional): Number of initial samples to generate. Defaults to 3.
        """
        x = self.init_sampler.get_sample(N_samples=N_samples_init)
        y = None
        self.dataset.add_samples(x, y, i_fidelity=i_fidelity)
    
    def evaluate_sample(self, points, i_fidelity):
        """
        return the surrogate's prediction value as a placeholder, since real output will be computed by a hero worker.
        The task is created and the hero_todo list and masking list are updated by add_samples

        Args:
            points (N samples, N input dimension): Sample points to evaluate.
            i_fidelity (int): Fidelity level index.

        Returns:
            y (N samples, N Output dimension): Estimated values.
        """     
        return self.surrogate.predict_values(points)
    
    def hero_wait_for_data_and_train(self):
        self.dataset.hero_wait_for_data()
        self.surrogate.train(self.dataset.x_data,
                             self.dataset.y_data)          

    