from adaptive_computing.datasets import DatasetBase
from adaptive_computing.surrogates import SurrogateModelBase, surrogate_initializer
from adaptive_computing.samplers import LHSSampler, BayesianSampler
from adaptive_computing.samplers import acquisition_functions
from adaptive_computing.evaluators import BaseEvaluator
from adaptive_computing.drivers.query_validators import get_query_validator
import numpy as np

class ActiveLoopDriver:
    """
    Active learning loop driver for adaptive sampling using surrogate models.

    Attributes:
        params (list): List of parameters for the simulations.
        dataset (DatasetBase): Dataset object storing input-output data.
        n_fidelity (int): Number of fidelity levels.
        evaluators (list): List of evaluators for each simulation.
        fidelity_costs (dict or None): Dictionary specifying costs associated with each fidelity level.
        surrogate (SurrogateModelBase): Surrogate model for prediction and optimization.
        init_sampler (LHSSampler): Initial sampler for sampling initial points.
        sampler (BayesianSampler): Bayesian sampler for selecting next samples.
        _bopt_initialized (bool): Flag indicating if the Bayesian optimization loop is initialized.
        nan_behavior (str): Behavior for handling NaN values in the dataset.

    Methods:
        __init__(simulations, params, surrogate=None, dataset=None, nan_behavior='fail', fidelity_costs=None):
            Initializes the ActiveLoopDriver with simulations, parameters, optional surrogate and dataset, nan behavior, and fidelity costs.
        _initialize_fidelity(i_fidelity, N_samples_init=3):
            Initializes a fidelity level with initial samples.
        initialize(N_samples_init=3):
            Initializes all fidelity levels with initial samples and trains the surrogate model.
        get_next_sample(i_fidelity=0):
            Retrieves the next sample to evaluate using the Bayesian sampler.
        step():
            Executes one step of the active learning loop: gets the next sample, evaluates it, and updates the dataset and surrogate.
        run(N_steps=None):
            Runs the active learning loop for a specified number of steps.
        add_points(points):
            Adds additional points to the dataset for evaluation.
        evaluate_sample(points, i_fidelity):
            Evaluates a sample using the corresponding evaluator.
        query(points, error_criterion, **kwargs):
            Queries the surrogate model for predictions and validates them against an error criterion.
    """

    def __init__(self, simulations, params, surrogate=None, dataset=None,
                 nan_behavior='fail', fidelity_costs=None, acq_func='expected_improvement', retrain=True):
        """
        Initializes the ActiveLoopDriver.

        Args:
            simulations (list): List of simulation functions to evaluate.
            params (list): List of parameters for the simulations.
            surrogate (SurrogateModelBase or str, optional): Surrogate model or string identifier for initializing surrogate. Defaults to None.
            dataset (DatasetBase, optional): Dataset object for storing input-output data. Defaults to None.
            nan_behavior (str, optional): Behavior for handling NaN values ('fail', 'mask_ignore'). Defaults to 'fail'.
            fidelity_costs (dict or None, optional): Dictionary specifying costs associated with each fidelity level. Defaults to None.
        """
        self.retrain = retrain
        self.params = params

        self.n_fidelity = len(simulations)
        if dataset is None:
            dataset = DatasetBase(params, n_fidelity=self.n_fidelity)
        self.dataset = dataset

        if simulations is not None:
            self.evaluators = [BaseEvaluator(simulation, n_in=len(self.params)) for simulation in simulations]
        else:
            assert(self.use_hero) # since the user has opted to use Hero, simulations should be set to None and the definition of the simulations should be implemented in the manager script. If Hero is not used, then simulations should not be specified by the user as a list of python functions

        self.fidelity_costs = fidelity_costs

        if isinstance(surrogate, SurrogateModelBase):
            self.surrogate = surrogate
        else:
            self.surrogate = surrogate_initializer(surrogate, 
                                                   self.dataset)
            
        self.init_sampler = LHSSampler(self.dataset)
        try:
            acq_function = acquisition_functions.acq_func_map[acq_func]
            self.sampler = BayesianSampler(self.dataset, acq_function)
        except KeyError:
            raise ValueError(f"Unsupported acquisition function type: {acq_func}")

        self._bopt_initialized = False

        self.nan_behavior = nan_behavior

    def _initialize_fidelity(self, i_fidelity, N_samples_init=3):
        """
        Initializes a fidelity level with initial samples.

        Args:
            i_fidelity (int): Fidelity level index.
            N_samples_init (int, optional): Number of initial samples to generate. Defaults to 3.
        """
        x = self.init_sampler.get_sample(N_samples=N_samples_init)
        y = self.evaluate_sample(x, i_fidelity=i_fidelity)
        self.dataset.add_samples(x, y, i_fidelity=i_fidelity)

    def initialize(self, N_samples_init=3):
        """
        Initializes all fidelity levels with initial samples and trains the surrogate model.

        Args:
            N_samples_init (int, optional): Number of initial samples to generate. Defaults to 3.
        """
        for i_fidelity in range(self.n_fidelity):
            self._initialize_fidelity(i_fidelity, N_samples_init=N_samples_init)
        if self.retrain:
            self.surrogate.train(self.dataset)
        self._bopt_initialized = True

    def get_next_sample(self, i_fidelity=0):
        """
        Retrieves the next sample to evaluate using the Bayesian sampler.

        Args:
            i_fidelity (int, optional): Fidelity level index. Defaults to 0.

        Returns:
            tuple: Next sample and its fidelity level index.
        """
        x = self.sampler.get_sample(self.surrogate, self.dataset, i_fidelity)
        return x, i_fidelity

    def step(self):
        """
        Executes one step of the active learning loop: gets the next sample, evaluates it,
        and updates the dataset and surrogate model.
        """
        x, fi_eval = self.get_next_sample()
        y = self.evaluate_sample(x, fi_eval)
        self.dataset.add_samples(x, y, i_fidelity=fi_eval)
        if self.retrain:
            self.surrogate.train(self.dataset)

    def run(self, N_steps=None):
        """
        Runs the active learning loop for a specified number of steps.

        Args:
            N_steps (int, optional): Number of steps to run. Defaults to None (runs indefinitely).
        """
        if not self._bopt_initialized:
            self.initialize()

        if N_steps is None:
            N_steps = np.inf

        for i in range(N_steps):
            self.step()

    def add_points(self, points, i_fidelity=0):
        """
        Adds additional points to the dataset for evaluation.

        Args:
            points (list or np.ndarray): Points to add to the dataset.
        """
        for x in points:
            x = np.atleast_2d(x)
            y = self.evaluate_sample(x, i_fidelity)
            self.dataset.add_samples(x, y, i_fidelity)

    def evaluate_sample(self, points, i_fidelity):
        """
        Evaluates a sample using the corresponding evaluator.

        Args:
            points (N samples, N input dimension): Sample points to evaluate.
            i_fidelity (int): Fidelity level index.

        Returns:
            y (N samples, N Output dimension): Evaluated values.
        """
        return self.evaluators[i_fidelity].evaluate_points(points)

    def query(self, points, error_criterion, threshold):
        """
        Queries the surrogate model for predictions and validates them against an error criterion.

        Args:
            points (N samples, N input dimension): Points to query.
            error_criterion (str): Error criterion for validation.
            arg: Additional argument for the error criterion threshold.

        Returns:
            np.ndarray: Predicted values.
        """
        points = np.asarray(points)
        values = np.zeros((points.shape[0], 1))

        # naive implementation: perform evaluations in a loop if they exceed the threshold
        # This involves only one pass through the data, but may not run the most informative points first,
        # and fails to use the most up to date surrogate for points that are evaluated before the last retraining.
        # validator = get_query_validator(criterion=error_criterion)
        # for i in range(points.shape[0]):
        #     surrogate_value = self.surrogate.predict_values(points[[i]])
        #     surrogate_variance = self.surrogate.predict_variances(points[[i]])
        #     valid = validator(surrogate_value, surrogate_variance, threshold)

        #     if not valid:
        #         print(f"Variance exceeds threshold for x={points[i]}, running simulation and retraining.")
        #         y = self.evaluate_sample(points[[i]], i_fidelity=0)
        #         self.dataset.add_samples(points[[i]], y, i_fidelity=0)
        #         if self.retrain:
        #             self.surrogate.train(self.dataset)
        #         # Note: do not return the simulation value. Instead, reevaluate the updated surrogate.
        #         surrogate_value = self.surrogate.predict_values(points[[i]])
        #     values[i] = surrogate_value

        # alternatate implementation: perform evaluations on the highest variance points first
        assert error_criterion == 'absolute_variance' #'percent_variance' is not supported in this implementation
        surrogate_variances = np.zeros((points.shape[0], 1))
        for i in range(points.shape[0]):
            surrogate_variances[i] = self.surrogate.predict_variances(points[[i]])

        # perform the simulations with variance exceeding the threshold starting with the highest variance
        while np.max(surrogate_variances) > threshold:
            i = np.argmax(surrogate_variances)
            print(f"Variance exceeds threshold for x={points[i]}, running simulation and retraining.")
            y = self.evaluate_sample(points[[i]], i_fidelity=0)
            self.dataset.add_samples(points[[i]], y, i_fidelity=0)
            if self.retrain:
                self.surrogate.train(self.dataset)
            # don't allow the same point to be evaluated again
            surrogate_variances[i] = 0
            # reevalute the variance of all points not set to zero
            for j in range(points.shape[0]):
                if surrogate_variances[j] != 0:
                    surrogate_variances[j] = self.surrogate.predict_variances(points[[j]])

        # reevalute the surrogate for all points
        for i in range(points.shape[0]):
            values[i] = self.surrogate.predict_values(points[[i]])

        return values

    @property
    def nan_behavior(self):
        """
        Getter for nan_behavior attribute.

        Returns:
            str: Current nan_behavior attribute value.
        """
        return self._nan_behavior
    
    @nan_behavior.setter
    def nan_behavior(self, nan_behavior):
        """
        Setter for nan_behavior attribute.

        Args:
            nan_behavior (str): New nan_behavior attribute value.
        """
        self._nan_behavior = nan_behavior
        self.dataset.nan_behavior = nan_behavior
