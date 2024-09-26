import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriver
import numpy as np

def func_1d(x):
    return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)#(x-3)**2

def bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=1)]

    ac_driver = ActiveLoopDriver(simulations=[func_1d],
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 7)

    # plot the result
    plt.figure(figsize=(10, 6))
    # plot the discrete samples
    plt.scatter(ac_driver.dataset.x_data[0], ac_driver.dataset.y_data[0], marker='o', color='b', label='Samples')
    # plot the high fidelity function
    x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)
    plt.plot(x, func_1d(x), color="k", label="Exact function")
    # plot the continuous surrogate model
    plt.plot(x, ac_driver.surrogate.predict_values(x), linestyle="-.", color='r', label='Surrogate model')
    # plot uncertainty bounds
    sig_plus = ac_driver.surrogate.predict_values(x)+3*np.sqrt(ac_driver.surrogate.predict_variances(x))
    sig_moins = ac_driver.surrogate.predict_values(x)-3*np.sqrt(ac_driver.surrogate.predict_variances(x))
    plt.fill_between(x.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='r')
    plt.legend(loc=0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return ac_driver

if __name__ == "__main__":
    bayesian_1d_sf()
