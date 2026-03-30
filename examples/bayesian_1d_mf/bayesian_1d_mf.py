import matplotlib
import os

# Ensure appropriate backend for plotting
def set_matplotlib_backend():
    if os.environ.get('DISPLAY'):
        # Try backends in order of preference for X11 forwarding
        backends_to_try = ['TkAgg', 'Qt5Agg', 'GTK3Agg']
        for backend in backends_to_try:
            try:
                matplotlib.use(backend, force=True)
                # Test if backend actually works
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                plt.close(fig)
                print(f"Using interactive matplotlib backend: {backend}")
                return
            except (ImportError, Exception):
                continue
        
        # If no GUI backends work, fall back to Agg
        print("No working interactive backends, using Agg backend")
        matplotlib.use('Agg')
    else:
        print("No DISPLAY detected, using Agg backend")
        matplotlib.use('Agg')

set_matplotlib_backend()

import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverCostRatio
import numpy as np

def func_lf(x):
    return (x-3)**2


def func_hf(x):
    return (x-3)**2+0.1*np.sin(x)

def bayesian_1d_mf():

    params = [ContinuousVariable(min=0, max=10)]

    ac_driver = ActiveLoopDriverCostRatio(simulations=[func_lf,
                                                func_hf],
                                   fidelity_costs=[1,10],
                                   params=params,
                                   surrogate='SMT_GP')
    
    ac_driver.run(N_steps = 10)

    # plot the result
    plt.figure(figsize=(10, 6))
    plt.scatter(ac_driver.dataset.x_data[0], ac_driver.dataset.y_data[0], marker='o', color='b', label='Low fidelity')
    plt.scatter(ac_driver.dataset.x_data[1], ac_driver.dataset.y_data[1], marker='s', color='r', label='High fidelity')
    plt.xlabel('x_data')
    plt.ylabel('y_data')
    plt.title('Bayesian 1D Multi-Fidelity Optimization')
    plt.legend()
    plt.savefig('bayesian_1d_mf_result.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'bayesian_1d_mf_result.png'")
    
    # Try to show plot if backend supports it
    try:
        plt.show()
    except Exception as e:
        print(f"Interactive display failed ({e}), but plot was saved")
    
    return ac_driver

if __name__ == "__main__":
    bayesian_1d_mf()
