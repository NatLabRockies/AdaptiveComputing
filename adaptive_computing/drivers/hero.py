from adaptive_computing.drivers import ActiveLoopDriver
from adaptive_computing.datasets import HeroDataset
from time import sleep

import numpy as np

class ActiveLoopDriverHero(ActiveLoopDriver):
    def __init__(self, simulations, params, machine_names, output_field_path, surrogate=None, dataset=None,
                 nan_behavior='fail', fidelity_costs=None, acq_func='expected_improvement', blocking=False,
                 task_formatter=None, inline_manager=None):
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

        # Optional LocalHPCManager for the noSSH workflow (controller runs on the
        # same node as the scheduler).  When set, query() calls
        # inline_manager.run_until_done() between task submission and waiting for
        # results.  Not needed when a background manager daemon is already running
        # (e.g. the SSH+tmux approach in hero_HPC_managers).
        # Not serialized to pickle — set this attribute after loading a saved driver:
        #   ac_driver = pickle.load(f)
        #   ac_driver.inline_manager = manager
        self.inline_manager = inline_manager

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

    def query(self, points, error_criterion, threshold):
        """Query the surrogate; submit Hero tasks in parallel for all high-variance points.

        Overrides ActiveLoopDriver.query() to use Hero task submission instead of
        local evaluators (self.evaluators is None for Hero drivers).

        Computes surrogate variance at every query point, then submits all points
        above the threshold as a single batch of Hero tasks.  The scheduler runs
        those jobs in parallel (wall time = slowest single job), the surrogate is
        retrained once on all new results, and final predictions are returned.

        For the noSSH workflow, set self.inline_manager to a LocalHPCManager
        instance (or assign it after loading from pickle) so that run_until_done()
        is called automatically between task submission and the Hero wait:

            ac_driver = pickle.load(f)
            ac_driver.inline_manager = manager
            y = ac_driver.query(x_queries, 'absolute_variance', threshold)

        For the SSH+tmux workflow, leave inline_manager as None — the background
        manager daemon processes the tasks while hero_wait_for_data_and_train() waits.

        Args:
            points:          Query points, shape (N, n_inputs).
            error_criterion: Must be 'absolute_variance'.
            threshold:       Points with predicted variance above this value are
                             evaluated via Hero rather than the surrogate alone.

        Returns:
            np.ndarray: Final surrogate predictions at all query points, shape (N, 1).
        """
        assert error_criterion == 'absolute_variance', \
            f"Hero driver query only supports 'absolute_variance', got '{error_criterion}'"

        x = np.asarray(points)
        variances = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            variances[i] = self.surrogate.predict_variances(x[[i]])

        high_var_mask = variances[:, 0] > threshold
        n_high = int(np.sum(high_var_mask))

        if n_high > 0:
            print(f"Queuing {n_high} Hero task(s) for points with variance above threshold:")
            for i in np.where(high_var_mask)[0]:
                print(f"  x={x[i]}, variance={variances[i, 0]:.2e}")
            self.add_samples(x[high_var_mask], i_fidelity=0)
            inline_manager = getattr(self, 'inline_manager', None)
            if inline_manager is not None:
                inline_manager.run_until_done(i_fidelity=0)
            self.hero_wait_for_data_and_train()
        else:
            print("All query points are below the variance threshold; using surrogate only.")

        return self.surrogate.predict_values(x)

    def hero_wait_for_data_and_train(self):
        self.dataset.hero_wait_for_data()
        self.surrogate.train(self.dataset)

    def hero_update_avail_data_and_train(self):
        for i_fl in range(self.dataset.n_fidelity):
            self.dataset.hero_update_avail_data(i_fl)
        self.surrogate.train(self.dataset)
