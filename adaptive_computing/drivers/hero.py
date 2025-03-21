from adaptive_computing.drivers import ActiveLoopDriver
from adaptive_computing.datasets import HeroDataset
from adaptive_computing.evaluators import BaseEvaluator
from time import sleep

import numpy as np

class ActiveLoopDriverHero(ActiveLoopDriver):
    def __init__(self, simulations, fidelity_costs, params, surrogate=None, dataset=None, nonblocking=False):
        self.use_hero = True
        if dataset is None:
            dataset = HeroDataset(params, n_fidelity=self.n_fidelity, 
                                        nonblocking=nonblocking)
        self.dataset = dataset
        self.nonblocking = nonblocking
        super().__init__(simulations, fidelity_costs, params, surrogate=surrogate, dataset=self.dataset)

        assert(simulations is None) # since the user has opted to use Hero, simulations should be set to None and the definition of the simulations should be implemented in the worker script.
        self.evaluators = None
        
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
    
    def _initialize_fidelity(self, i_fidelity, N_samples_init=3):
        x = self.init_sampler.get_sample(N_samples=N_samples_init)
        self.add_sample_queue(x,i_fidelity=i_fidelity)
    
    def step(self):
        self.surrogate.train(self.dataset.x_data,
                             self.dataset.y_data)
       
        x, fi_eval = self.get_next_sample()
        self.add_sample_queue(x, fi_eval)

    def run(self):
        if not self._bopt_initialized:
            self.initialize()

        # Wait until initialized
        self.dataset.hero_wait_for_data()

        while True:
            # check if there are available resources
            if not self.nonblocking:
                self.dataset.hero_wait_for_data()
            else:
                for i_fl in range(len(self.dataset.n_fidelity)):
                    self.dataset.hero_sync_avail_data(i_fl)
            # add tasks to queue
            self.step()

    def _validate_point(self, point):
        """
        Validates a single point to ensure it has the correct number of input dimensions.
        
        Args:
            point (N input dimension): A 1D array representing a point to validate.
        
        Raises:
            ValueError: If the point does not have the correct number of input dimensions.
        """
        n_in = len(self.params)
        if point.shape[0] != self.n_in:
            print("Point has incorrect number of input dimensions")
            print(f"Point shape: {point.shape}")
            print(f"Target input dimension: [{self.n_in}]")
            raise ValueError
            
    
        
