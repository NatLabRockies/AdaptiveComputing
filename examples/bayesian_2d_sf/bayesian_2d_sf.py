import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverSF
from adaptive_computing.surrogates import SMTWrapper

def func_2d(x):
    # x will have shape (n_samples, n_in,)
    y = (x[...,0]-3)**2 + (x[...,1]-6.2)**2 + 1.4
    
    # y must have shape (n_samples, 1)
    y = y.reshape(-1,1)

    return y

def bayesian_2d_sf():

    params = [ContinuousVariable(min=0, max=10),
              ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverSF(simulation=func_2d,
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 50)

    # plot the result
    # Create a grid of points
    x = np.linspace(0, 10, 400)
    y = np.linspace(0, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = func_2d(np.stack([X, Y], axis=-1))

    # Plot the contour
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z.reshape(X.shape), levels=50, cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label('y') 
    x_data = ac_driver.dataset.x_data[0]
    sc = plt.scatter(x_data[:, 1], x_data[:, 0], c=np.arange(x_data.shape[0]), cmap='rainbow', edgecolor='k')
    plt.colorbar(sc, label='Iteration')
    plt.xlabel('x_0')
    plt.ylabel('x_1')
    plt.show()

    return ac_driver

if __name__ == "__main__":
    bayesian_2d_sf()
