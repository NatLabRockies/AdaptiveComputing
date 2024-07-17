from adaptive_computing.drivers import ActiveLoopDriver
from adaptive_computing.datasets import HeroDataset
from time import sleep

import numpy as np

class ActiveLoopDriverMFNonBlock(ActiveLoopDriver):
    def __init__(self, simulations, fidelity_costs, params, surrogate=None, dataset=None):
        super().__init__(simulations, fidelity_costs, params, surrogate=surrogate)

        # setup hero stuff
        self.data_repo = None
        self.task_queue = None

        # Setup some notion of available resources
        self.resources = None

        if dataset is None:
            self.dataset = HeroDataset(params, n_fidelity=self.n_fl, 
                                       data_repo=self.data_repo)

    def add_sample_queue(self, x, fidelity):
        #add sample to hero queue
        pass
    
    @property
    def resources_available(self):
        # somehow check if we have any resources waiting that 
        # can take new tasks
        return False

    def _initialize_fidelity(self, n_fidelity, N_samples_init=3):
        x = self.init_sampler.get_sample(N_samples=N_samples_init)
        self.add_sample_queue(x,n_fidelity=n_fidelity)
        
    
    def step(self):
        self.surrogate.train(self.dataset.x_data,
                             self.dataset.y_data)
       
        x, fi_eval = self.get_next_sample()
        self.add_sample_queue(x, fi_eval)

    def run(self):
        if not self._bopt_initialized:
            self.initialize()

        # Wait until initialized

        while True:
            # check if there are available resources
            if self.resources_available:
                # If there are, add tasks to queue
                self.step()
            else:
                # If not, wait a minute until checking again
                sleep(60)
            
    
        