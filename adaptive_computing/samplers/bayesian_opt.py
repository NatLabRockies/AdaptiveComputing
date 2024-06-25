from adaptive_computing.samplers import SamplerBase
from smt.sampling_methods import LHS
from smt.applications.mixed_integer import MixedIntegerSamplingMethod

from scipy.optimize import minimize, brute, differential_evolution
import numpy as np
from copy import deepcopy

class BayesianSampler(SamplerBase):
    def __init__(self,dataset, acquistion_function, 
                 rand_seed=-1, # set rand_seed=None for non-determinstic behavior, -1 to never repeat samples, and some specific integer for testing
                 n_eval_pts=30):
        self._rand_seed = rand_seed
        self.acq_func = acquistion_function
        self.n_eval_pts = n_eval_pts
        self.ranges = dataset._sampler_ranges
        self.x_limits = dataset.x_limits
        self.opt_method = 'SLSQP'

        if dataset.mixed_type:
            raise(NotImplementedError)
        else:
            self._minimizer = self._min_cont_vars
            
        super().__init__(dataset)

    def get_sample(self, surrogate, dataset,n_fidelity=0, N_samples=1):
        
        tmp_dataset = deepcopy(dataset)
        tmp_surrogate = deepcopy(surrogate)

        x_samples = []

        for i_sample in range(N_samples):
            tmp_surrogate.train(tmp_dataset.x_data, tmp_dataset.y_data)
            
            x_est = self.minimize_acq_func(tmp_surrogate, tmp_dataset, n_fidelity=n_fidelity)  
            y_est = tmp_surrogate.predict_values(x_est)
            tmp_dataset.add_samples(x_est, y_est, n_fidelity=n_fidelity)
            x_samples.append(x_est)

        return np.concatenate(x_samples,axis=0)


    def minimize_acq_func(self, surrogate, dataset, n_fidelity=0):
        # cannot create the sampler in the constructor (must be done here) since the random_state may need to be updated with each call of minimize_acq_func
        if dataset.mixed_type:
            raise(NotImplementedError)
        else:
            # increment the random seed, so that future samples are not duplicates
            random_state = self._rand_seed
            if self._rand_seed == -1:
                random_state = sum(array.shape[0] for array in dataset._x_data)*self.n_eval_pts
            self.init_sample = LHS(xlimits= dataset.x_limits,
                                   criterion='maximin',
                                   random_state=random_state)

        xstart = self.init_sample(self.n_eval_pts)
        obj_k = lambda x: self.acq_func(x,surrogate, dataset, n_fidelity)
        x = self._min_cont_vars(xstart, obj_k)
        return np.array([x])
    
    def _min_cont_vars(self,xstart, obj_k):
        opt_all = np.array([])
        for i_s in range(self.n_eval_pts):
            opt_all = np.append(opt_all,minimize(lambda xf: float(obj_k(np.append(xf,[]))),
                                                 xstart[i_s], 
                                                 method=self.opt_method,
                                                 bounds=self.x_limits))
        opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]] # gets only the entries of opt_all that have 'success'=True. Note: opt_all is a dictionary, so opt_all[0]['success'] is equivalent to opt_all[0].success
        obj_success = np.array([opt_i['fun'] for opt_i in opt_success]) # create an array of the function values for all of the successful optimization points
        ind_min = np.argmin(obj_success) # which initial guess was best (led to the deepest min value)
        opt = opt_success[ind_min] # the full output for the best initial guess
        xf_opt = opt['x'] # the x value at which the min occurs
        return xf_opt

