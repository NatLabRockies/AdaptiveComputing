import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverMF
import numpy as np

def func_lf(x):
    return (x-3)**2

def func_hf(x):
    return (x-3)**2+0.1*np.sin(x)

class CustomMFDriver(ActiveLoopDriverMF):

    def step(self):
        """
        Designing a workflow that always runs all lower
        fidelity models if sampling a higher fidelity model point.
        """
        x_samples = np.zeros((self.n_fl, 1, self.dataset.n_in))
        objs = np.zeros(self.n_fl)
        for f_i in range(self.n_fl):
            x = self.sampler.get_sample(self.surrogate, self.dataset, f_i)
            y = self.sampler.acq_func(x, self.surrogate, self.dataset, f_i)

            x_samples[f_i] = x
            objs[f_i] = y

        fi_eval = np.argmin(objs/self.fidelity_costs)
        x = x_samples[fi_eval]

        for f_i in range(0, fi_eval):
            y = self.evaluate_sample(x, f_i)
            self.dataset.add_known_samples(x,y, n_fidelity=f_i)

def bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = CustomMFDriver(simulations=[func_lf,
                                            func_hf],
                                   fidelity_costs=[1,10],
                                   params=params,
                                   surrogate='SMT_GP')
    
    ac_driver.run(N_steps = 10)

    return ac_driver

if __name__ == "__main__":
    bayesian_1d_sf()
