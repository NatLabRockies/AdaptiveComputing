import matplotlib
import os

# Try different backends for interactive display
if os.environ.get('DISPLAY'):
    try:
        matplotlib.use('TkAgg')  # Try TkAgg first (more commonly available)
        print("Using TkAgg backend for interactive display")
    except ImportError:
        try:
            matplotlib.use('Qt5Agg')  # Fallback to Qt5
            print("Using Qt5Agg backend for interactive display") 
        except ImportError:
            try:
                matplotlib.use('GTK3Agg')  # Fallback to GTK3
                print("Using GTK3Agg backend for interactive display")
            except ImportError:
                print("No interactive backend available, using Agg")
                matplotlib.use('Agg')
else:
    print("No DISPLAY environment variable detected, using Agg backend")
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriver
import numpy as np

print(f"Using matplotlib backend: {matplotlib.get_backend()}")

def func_1d(x):
    return ((x * 6 - 2) ** 2) * np.sin((x * 6 - 2) * 2)#(x-3)**2

def bayesian_1d_sf():

    params = [ContinuousVariable(min=0, max=1)]

    ac_driver = ActiveLoopDriver(simulations=[func_1d],
                                   params=params,
                                   surrogate='TFMELT_BNN')                                   
    
    ac_driver.initialize(N_samples_init=13)
    ac_driver.run(N_steps = 2)

    # Switch to full mode for better uncertainty estimates in plotting
    ac_driver.surrogate.enable_full_mode()

    # plot the result
    plt.figure(figsize=(10, 6))
    # plot the discrete samples
    plt.scatter(ac_driver.dataset.x_data[0], ac_driver.dataset.y_data[0], marker='o', color='b', label='Samples')
    # plot the high fidelity function
    x = np.linspace(0, 1, 101, endpoint=True).reshape(-1, 1)  # Reasonable resolution for plotting
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
    plt.title('Bayesian 1D Surrogate with TF_MELT BNN')
    plt.savefig('bayesian_1d_tf_melt_result.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'bayesian_1d_tf_melt_result.png'")
    
    # Try to show plot interactively if backend supports it
    if matplotlib.get_backend() != 'Agg':
        try:
            plt.show()
            print("Plot displayed interactively")
        except Exception as e:
            print(f"Interactive display failed ({e}) - plot saved to file only")
    else:
        print("Using non-interactive backend - plot saved to file only")

    return ac_driver

if __name__ == "__main__":
    bayesian_1d_sf()
