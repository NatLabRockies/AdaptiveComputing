from adaptive_computing.samplers import SamplerBase
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod

# Optional Hero import
try:
    from adaptive_computing.datasets import HeroDataset
except ImportError:
    HeroDataset = None

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
        self.dataset = dataset  # Store dataset reference for mixed-type optimization
        self.ranges = dataset._sampler_ranges
        self.x_limits = dataset.x_limits
        self.opt_method = 'SLSQP'

        if dataset.mixed_type:
            self._minimizer = self._min_mixed_smt2x
            print("Using SMT 2.x mixed-type optimization for Bayesian sampler")
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
        # if dataset is instance HeroDataset:
        #     tmp_dataset = DatasetBase(dataset.params, n_fidelity=dataset.n_fidelity)
        #     for i_fidelity in dataset.n_fidelity:
        #         tmp_dataset.add_samples(dataset.x_data,dataset.y_data,i_fidelity=i_fidelity)
        #     # XXX Note that any masked data will be added as unmasked. But for non-Hero this is not the behavior. Not sure which is best
        # else:
        tmp_dataset = deepcopy(dataset)
        
        tmp_surrogate = deepcopy(surrogate)

        x_samples = []

        for i_sample in range(N_samples):
            tmp_surrogate.train(tmp_dataset.x_data, tmp_dataset.y_data)
            
            x_est = self.minimize_acq_func(tmp_surrogate, tmp_dataset, i_fidelity=i_fidelity)
            
            # Post-process mixed-type samples to ensure proper data types
            if self.mixed_type:
                x_est = self._process_mixed_type_samples(x_est)
                
            y_est = tmp_surrogate.predict_values(x_est)
            if HeroDataset and isinstance(dataset, HeroDataset):
                tmp_dataset.add_samples_nohero(x_est, y_est, i_fidelity=i_fidelity)
            else:
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
            # Use SMT 2.x mixed-type sampling
            from smt.applications.mixed_integer import MixedIntegerSamplingMethod
            from smt.sampling_methods import LHS as LHS_mixed
            
            # Use iteration number to ensure different random seeds each step
            iteration_number = sum(array.shape[0] for array in dataset._x_data)
            if self._rand_seed == -1:
                random_state = iteration_number * 1000 + 42  # Use iteration-based seed
            else:
                random_state = self._rand_seed + iteration_number  # Add iteration to base seed
                
            # Use SMT 2.x design space for initialization
            self.init_sample = MixedIntegerSamplingMethod(
                LHS_mixed,
                dataset.design_space,
                criterion='maximin',
                seed=random_state
            )
        else:
            # Use regular LHS for continuous variables only
            from smt.sampling_methods import LHS as LHS_cont
            # Use iteration number to ensure different random seeds each step
            iteration_number = sum(array.shape[0] for array in dataset._x_data)
            if self._rand_seed == -1:
                random_state = iteration_number * 1000 + 42  # Use iteration-based seed
            else:
                random_state = self._rand_seed + iteration_number  # Add iteration to base seed
            self.init_sample = LHS_cont(xlimits=dataset.x_limits,
                                   criterion='maximin',
                                   seed=random_state)

        xstart = self.init_sample(self.n_eval_pts)
        obj_k = lambda x: self.acq_func(x, surrogate, dataset, i_fidelity)
        
        # Use appropriate optimization method based on variable types
        if dataset.mixed_type:
            x = self._min_mixed_smt2x(xstart, obj_k)
        else:
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
            opt_all = np.append(opt_all, minimize(lambda xf: obj_k(xf).item() if hasattr(obj_k(xf), 'item') else float(obj_k(xf)),
                                                  xstart[i_s], 
                                                  method=self.opt_method,
                                                  bounds=self.x_limits))
        opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]]  # gets only the entries of opt_all that have 'success'=True
        obj_success = np.array([opt_i['fun'] for opt_i in opt_success])  # create an array of the function values for all of the successful optimization points
        ind_min = np.argmin(obj_success)  # which initial guess was best (led to the deepest min value)
        opt = opt_success[ind_min]  # the full output for the best initial guess
        xf_opt = opt['x']  # the x value at which the min occurs
        return xf_opt

    def _min_mixed_smt2x(self, xstart, obj_k):
        """
        Minimizes the objective function for mixed-type variables using brute force for discrete variables.
        
        Args:
            xstart (np.ndarray): The starting points for optimization.
            obj_k (callable): The objective function to minimize.
        
        Returns:
            np.ndarray: The optimized values.
        """
        import numpy as np
        from itertools import product
        from scipy.optimize import minimize
        
        # Separate continuous and discrete variable indices
        continuous_indices = []
        discrete_indices = []
        discrete_values = []  # List of possible values for each discrete variable
        
        for i, param in enumerate(self.dataset.params):
            if param.type == 'continuous':
                continuous_indices.append(i)
            elif param.type == 'ordered':
                discrete_indices.append(i)
                # For ordered variables, enumerate all possible integer values
                discrete_values.append(list(range(param.min_val, param.max_val + 1)))
            elif param.type == 'categorical':
                discrete_indices.append(i)
                # For categorical variables, enumerate all category indices
                discrete_values.append(list(range(len(param.categories))))
        
        best_x = None
        best_obj = np.inf
        
        if len(discrete_indices) > 0:
            # Use brute force enumeration for discrete variables
            discrete_combinations = list(product(*discrete_values))
            
            # Limit the number of combinations to avoid exponential explosion
            max_combinations = 1000
            if len(discrete_combinations) > max_combinations:
                # Sample a subset of combinations
                indices = np.random.choice(len(discrete_combinations), max_combinations, replace=False)
                discrete_combinations = [discrete_combinations[i] for i in indices]
            
            for discrete_combo in discrete_combinations:
                # For each combination of discrete values, optimize over continuous variables
                if len(continuous_indices) > 0:
                    # Create objective function with fixed discrete variables
                    def cont_obj(x_cont):
                        x_full = np.zeros(len(self.dataset.params))
                        x_full[continuous_indices] = x_cont
                        for j, idx in enumerate(discrete_indices):
                            x_full[idx] = discrete_combo[j]
                        return obj_k(x_full.reshape(1, -1))[0]
                    
                    # Optimize over continuous variables
                    continuous_bounds = [self.dataset.params[i].limits for i in continuous_indices]
                    
                    best_cont_obj = np.inf
                    best_cont_x = None
                    
                    # Try multiple starting points for continuous optimization
                    for start_pt in xstart[:min(self.n_eval_pts, len(xstart))]:  # Use n_eval_pts starting points
                        x_cont_start = start_pt[continuous_indices]
                        
                        try:
                            result = minimize(
                                cont_obj,
                                x_cont_start,
                                method='L-BFGS-B',
                                bounds=continuous_bounds,
                                options={'ftol': 1e-9, 'gtol': 1e-6}
                            )
                            
                            if result.success and result.fun < best_cont_obj:
                                best_cont_obj = result.fun
                                best_cont_x = result.x
                        except:
                            continue
                    
                    if best_cont_x is not None:
                        # Construct full solution
                        x_candidate = np.zeros(len(self.dataset.params))
                        x_candidate[continuous_indices] = best_cont_x
                        for j, idx in enumerate(discrete_indices):
                            x_candidate[idx] = discrete_combo[j]
                        
                        # Add small jitter to continuous variables to prevent SMT duplicate warnings
                        # while maintaining optimization precision
                        jitter_scale = 1e-8
                        for cont_idx in continuous_indices:
                            param = self.dataset.params[cont_idx]
                            range_size = param.max - param.min
                            jitter = np.random.uniform(-jitter_scale, jitter_scale) * range_size
                            x_candidate[cont_idx] += jitter
                            # Ensure we stay within bounds
                            x_candidate[cont_idx] = np.clip(x_candidate[cont_idx], param.min, param.max)
                        
                        obj_val = best_cont_obj
                    else:
                        # Fallback: use starting point continuous values
                        x_candidate = np.zeros(len(self.dataset.params))
                        x_candidate[continuous_indices] = xstart[0][continuous_indices]
                        for j, idx in enumerate(discrete_indices):
                            x_candidate[idx] = discrete_combo[j]
                        
                        obj_val = cont_obj(x_candidate[continuous_indices])
                        
                else:
                    # Only discrete variables - just evaluate the objective
                    x_candidate = np.array(discrete_combo, dtype=float)
                    obj_val = obj_k(x_candidate.reshape(1, -1))[0]
                
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = x_candidate
                    
        else:
            # Only continuous variables - use standard optimization
            return self._min_cont_vars(xstart, obj_k)
        
        return best_x if best_x is not None else xstart[0]

    def _min_cont_vars(self, xstart, obj_k):
        """
        Optimization for continuous variables only using gradient-based methods.
        """
        from scipy.optimize import minimize
        import numpy as np
        
        bounds = [param.limits for param in self.dataset.params if param.type == 'continuous']
        
        best_x = None
        best_obj = np.inf
        
        # Try multiple starting points
        for x_start in xstart[:min(10, len(xstart))]:
            try:
                result = minimize(
                    lambda x: obj_k(x.reshape(1, -1))[0],
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds
                )
                
                if result.success and result.fun < best_obj:
                    best_obj = result.fun
                    best_x = result.x
            except:
                continue
        
        return best_x if best_x is not None else xstart[0]

    def _min_mixed_fallback(self, xstart, obj_k):
        """
        Fallback mixed-type optimization using simple evaluation.
        """
        best_x = None
        best_obj = np.inf
        
        for x_candidate in xstart:
            try:
                obj_val = obj_k(x_candidate.reshape(1, -1))
                if obj_val[0] < best_obj:
                    best_obj = obj_val[0]
                    best_x = x_candidate
            except:
                continue
                
        return best_x if best_x is not None else xstart[0]
    def _process_mixed_type_samples(self, x):
        """
        Post-process samples for mixed-type variables to ensure proper data types.
        
        Args:
            x (np.ndarray): Raw samples 
            
        Returns:
            np.ndarray: Processed samples with correct data types
        """
        import numpy as np
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        x_processed = x.copy()
        
        for i, param in enumerate(self.dataset.params):
            if param.type == 'ordered':
                # Round to nearest integer and clamp to bounds
                x_processed[:, i] = np.round(x_processed[:, i]).astype(int)
                x_processed[:, i] = np.clip(x_processed[:, i], param.min_val, param.max_val)
            elif param.type == 'categorical':
                # Round to nearest integer (category index) and clamp to valid range
                x_processed[:, i] = np.round(x_processed[:, i]).astype(int)
                x_processed[:, i] = np.clip(x_processed[:, i], 0, len(param.categories) - 1)
            # continuous variables need no processing
        
        return x_processed