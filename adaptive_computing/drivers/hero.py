from adaptive_computing.drivers import ActiveLoopDriver
from adaptive_computing.datasets import HeroDataset
from time import sleep

import numpy as np

class ActiveLoopDriverHero(ActiveLoopDriver):
    def __init__(self, simulations, params, machine_names, output_field_path, surrogate=None, dataset=None,
                 nan_behavior='fail', fidelity_costs=None, acq_func='expected_improvement', blocking=False, 
                 task_formatter=None):
        self.use_hero = True
        if dataset is None:
            dataset = HeroDataset(params, machine_names, output_field_path, n_fidelity=1, blocking=blocking, 
                                task_formatter=task_formatter, nan_behavior=nan_behavior)
        self.dataset = dataset
        if blocking:
            retrain = True
        else:
            retrain = False # only retrain when wait hero_wait_for_data_and_train is called
        super().__init__(simulations, params, surrogate=surrogate, dataset=self.dataset,
                         nan_behavior=nan_behavior, fidelity_costs=fidelity_costs, acq_func=acq_func, retrain=retrain)

        for sim_i in simulations:
            assert(sim_i is None) # since the user has opted to use Hero, simulations should be set to a list of Nones of length n_fidelity and the definition of the simulations should be implemented in the manager script.
        self.evaluators = None
        
    def _initialize_fidelity(self, i_fidelity, N_samples_init=3):
        """
        Initializes a fidelity level by queuing random LHS samples in the Hero task system.

        Args:
            i_fidelity (int): Fidelity level index.
            N_samples_init (int, optional): Number of initial samples to generate. Defaults to 3.
        """
        x = self.init_sampler.get_sample(N_samples=N_samples_init)
        self.dataset.add_samples(x, i_fidelity=i_fidelity)

    def add_samples(self, points, i_fidelity=0):
        """
        Queues input points in the Hero task system for asynchronous evaluation.
        Non-blocking: returns immediately after creating the Hero tasks.
        Call hero_wait_for_data_and_train() to wait for results.

        Args:
            points (list or np.ndarray): Points to queue for evaluation.
            i_fidelity (int): Fidelity level index.
        """
        for x in points:
            x = np.atleast_2d(x)
            self.dataset.add_samples(x, i_fidelity)

    def step(self):
        """
        Executes one step of the active learning loop: selects the next sample
        using the acquisition function and queues it as a Hero task.
        """
        x, fi_eval = self.get_next_sample()
        self.dataset.add_samples(x, i_fidelity=fi_eval)
        if self.retrain:
            self.surrogate.train(self.dataset)
    
    def hero_wait_for_data_and_train(self):
        self.dataset.hero_wait_for_data()
        self.surrogate.train(self.dataset)
    
    def hero_update_avail_data_and_train(self):
        for i_fl in range(self.dataset.n_fidelity):
            self.dataset.hero_update_avail_data(i_fl)
        self.surrogate.train(self.dataset)

    
