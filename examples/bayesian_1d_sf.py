from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverSF
from adaptive_computing.surrogates import SMTWrapper

def func_1d(x):
    return (x-3)**2

def example_bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverSF(simulation=func_1d,
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 10)



if __name__ == "__main__":
    example_bayesian_1d_sf()