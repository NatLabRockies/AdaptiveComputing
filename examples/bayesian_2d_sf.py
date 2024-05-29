from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverSF
from adaptive_computing.surrogates import SMTWrapper

def func_2d(x):
    # x will have shape (n_samples, n_in,)
    y = (x[...,0]-3)**2+(x[...,0]-6.2)**2
    
    # y must have shape (n_samples, 1)
    y = y.reshape(-1,1)

    return y

def example_bayesian_2d_sf():

    params = [ContinuousVariable(min=0, max=10),
              ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverSF(simulation=func_2d,
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 50)



if __name__ == "__main__":
    example_bayesian_2d_sf()