from adaptive_computing.samplers import SamplerBase
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod

from scipy.optimize import minimize, brute, differential_evolution
import numpy as np
from copy import deepcopy

class BayesianSampler(SamplerBase):
    """
    A class for sampling points using Bayesian optimization.
    
    Attributes:
        _rand_seed (int): Random seed for deterministic behavior.
        acq_func (callable): Acquisition function for the Bayesian optimization.
        n_eval_pts (int): Number of evaluation points.
        ranges (tuple): Ranges for the sampler.
        x_limits (np.ndarray): Limits for the input dimensions.
        opt_method (str): Optimization method.
        _minimizer (callable): Minimization function for continuous variables.
        
    Methods:
        get_sample(surrogate, dataset, i_fidelity, N_samples): Generates samples using the surrogate model and dataset.
        minimize_acq_func(surrogate, dataset, i_fidelity): Minimizes the acquisition function.
        _min_cont_vars(xstart, obj_k): Minimizes the objective function for continuous variables.
    """
    
    def __init__(self, dataset, acquisition_function, rand_seed=-1, n_eval_pts=30):
        """
        Initializes the BayesianSampler with the dataset, acquisition function, and other parameters.
        
        Args:
            dataset (DatasetBase): The dataset used to setup sampling limits.
            acquisition_function (callable): The acquisition function for Bayesian optimization.
            rand_seed (int): Random seed for deterministic behavior. Defaults to -1 for never repeating samples.
            n_eval_pts (int): Number of evaluation points. Defaults to 30.
        
        Raises:
            NotImplementedError: If the dataset has mixed types.
        """
        self._rand_seed = rand_seed
        self.acq_func = acquisition_function
        self.n_eval_pts = n_eval_pts
        self.ranges = dataset._sampler_ranges
        self.x_limits = dataset.x_limits
        self.opt_method = 'SLSQP'

        if dataset.mixed_type:
            raise NotImplementedError
        else:
            self._minimizer = self._min_cont_vars
            
        super().__init__(dataset)

    def get_sample(self, surrogate, dataset, i_fidelity=0, N_samples=1):
        """
        Generates samples using the surrogate model and dataset.
        
        Args:
            surrogate (Surrogate): The surrogate model.
            dataset (Dataset): The dataset to sample from.
            i_fidelity (int): The fidelity level. Defaults to 0.
            N_samples (int): The number of samples to generate. Defaults to 1.
        
        Returns:
            x samples (N samples, N input dimension): The generated samples.
        """
        tmp_dataset = deepcopy(dataset)
        tmp_surrogate = deepcopy(surrogate)

        x_samples = []

        for i_sample in range(N_samples):
            tmp_surrogate.train(tmp_dataset.x_data, tmp_dataset.y_data)
            
            x_est = self.minimize_acq_func(tmp_surrogate, tmp_dataset, i_fidelity=i_fidelity)  
            y_est = tmp_surrogate.predict_values(x_est)
            tmp_dataset.add_samples(x_est, y_est, i_fidelity=i_fidelity)
            x_samples.append(x_est)

        return np.concatenate(x_samples, axis=0)

    def minimize_acq_func(self, surrogate, dataset, i_fidelity=0):
        """
        Minimizes the acquisition function using the surrogate model and dataset.
        
        Args:
            surrogate (object): The surrogate model.
            dataset (DatasetBase): The dataset to sample from.
            i_fidelity (int): The fidelity level. Defaults to 0.
        
        Returns:
            np.ndarray: The point that minimizes the acquisition function.
        
        Raises:
            NotImplementedError: If the dataset has mixed types.

        Note:
            cannot create the sampler in the constructor (must be done here) 
            since the random_state may need to be updated with each call of minimize_acq_func
        """
        if dataset.mixed_type:
            raise NotImplementedError
        else:
            # increment the random seed, so that future samples are not duplicates
            random_state = self._rand_seed
            if self._rand_seed == -1:
                random_state = sum(array.shape[0] for array in dataset._x_data) * self.n_eval_pts
            self.init_sample = LHS(xlimits=dataset.x_limits,
                                   criterion='maximin',
                                   random_state=random_state)

        xstart = self.init_sample(self.n_eval_pts)
        obj_k = lambda x: self.acq_func(x, surrogate, dataset, i_fidelity)
        x = self._min_cont_vars(xstart, obj_k)
        return np.array([x])
    
    def _min_cont_vars(self, xstart, obj_k):
        """
        Minimizes the objective function for continuous variables.
        
        Args:
            xstart (np.ndarray): The starting points for optimization.
            obj_k (callable): The objective function to minimize.
        
        Returns:
            np.ndarray: The optimized values.
        """
        opt_all = np.array([])
        for i_s in range(self.n_eval_pts):
            opt_all = np.append(opt_all, minimize(lambda xf: float(obj_k(np.append(xf, []))),
                                                  xstart[i_s], 
                                                  method=self.opt_method,
                                                  bounds=self.x_limits))
        opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]]  # gets only the entries of opt_all that have 'success'=True
        obj_success = np.array([opt_i['fun'] for opt_i in opt_success])  # create an array of the function values for all of the successful optimization points
        ind_min = np.argmin(obj_success)  # which initial guess was best (led to the deepest min value)
        opt = opt_success[ind_min]  # the full output for the best initial guess
        xf_opt = opt['x']  # the x value at which the min occurs
        return xf_opt
