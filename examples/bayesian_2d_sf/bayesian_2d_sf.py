import numpy as np
import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriver

def func_2d(x):
    # x will have shape (n_samples, n_in,)
    y = (x[...,0]-3)**2 + (x[...,1]-6.2)**2 + 1.4
    
    # y must have shape (n_samples, 1)
    y = y.reshape(-1,1)

    return y

def bayesian_2d_sf(test_mode=False):
    """
    2D Bayesian optimization example.
    
    Objective function: f(x0,x1) = (x0-3)^2 + (x1-6.2)^2 + 1.4
    Global minimum: f=1.4 at [x0,x1] = [3, 6.2]
    
    Note: Full optimization mode takes a couple minutes to complete due to 
          Gaussian Process training on 50 samples.
    
    Args:
        test_mode (bool): If True, use minimal computational load for testing
    """
    import os
    import sys
    
    # Detect if running in test environment to use reduced computational load
    is_testing = (
        test_mode or                                      # Manual override
        'pytest' in sys.modules or                        # pytest is imported
        'PYTEST_CURRENT_TEST' in os.environ or           # set by pytest
        any('pytest' in arg for arg in sys.argv) or      # 'pytest' in command line
        any('test' in arg.lower() for arg in sys.argv)   # any 'test' in args
    )
    
    if is_testing:
        print("Detected testing environment - using reduced computational load")
    else:
        print("Running in full optimization mode - this will take a couple minutes to complete...")

    params = [ContinuousVariable(min=0, max=10),
              ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriver(simulations=[func_2d],
                                   params=params,
                                   surrogate='SMT')
    
    if is_testing:
        # Fast testing configuration: minimal samples for pytest
        ac_driver.run(N_steps=8)  # Only 8 steps for speed
    else:
        # Full optimization configuration
        ac_driver.run(N_steps=50)  # Full 50 steps for better results

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
