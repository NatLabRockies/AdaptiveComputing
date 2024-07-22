import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriver
from adaptive_computing.surrogates import SMTWrapper

def func_1d(x):
    return (x-3)**2

def bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriver(simulations=[func_1d],
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 10)

    # plot the result
    plt.figure(figsize=(10, 6))
    plt.scatter(ac_driver.dataset.x_data[0], ac_driver.dataset.y_data[0], marker='o', linestyle='-', color='b')
    plt.xlabel('x_data')
    plt.ylabel('y_data')
    plt.show()

    return ac_driver

if __name__ == "__main__":
    bayesian_1d_sf()
