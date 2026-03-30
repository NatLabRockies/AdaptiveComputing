import numpy as np
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
from adaptive_computing.datasets import ContinuousVariable, OrderedVariable, CategoricalVariable
from adaptive_computing.drivers import ActiveLoopDriver

def func_3d_mixedtypes(x):
    # x will have shape (n_samples, n_in) where n_in=3
    # x[...,0]: continuous variable [0,8]
    # x[...,1]: ordered integer variable [2,6] 
    # x[...,2]: categorical variable {'a','b','c','d'} encoded as {0,1,2,3}
    
    # Handle the categorical variable encoding
    # In the current implementation, categorical variables are encoded as indices
    # Map indices to categorical values: {0:'a', 1:'b', 2:'c', 3:'d'}
    # Then map to s values: {'a':10, 'b':5, 'c':7.5, 'd':6}
    s_values = [10.0, 5.0, 7.5, 6.0]  # corresponds to ['a','b','c','d']
    
    # For vectorized operations, handle both single samples and batch samples
    if x.ndim == 1:
        # Single sample case
        s = s_values[int(x[2])]
        y = (x[0] - 5.0)**2 + (x[1] - 4.0)**2 + s - 5.0
        return np.array([[y]])
    else:
        # Batch samples case
        batch_size = x.shape[0]
        y = np.zeros(batch_size)
        for i in range(batch_size):
            s = s_values[int(x[i, 2])]
            y[i] = (x[i, 0] - 5.0)**2 + (x[i, 1] - 4.0)**2 + s - 5.0
        
        # y must have shape (n_samples, 1)
        return y.reshape(-1, 1)

def bayesian_3d_sf_mixedtypes(test_mode=False):
    """
    3D Bayesian optimization with mixed data types:
    - x0: continuous variable [0,8] 
    - x1: ordered integer variable [2,6]
    - x2: categorical variable ['a','b','c','d']
    
    Objective function: f(x0,x1,x2) = (x0-5)^2+(x1-4)^2+s(x2)-5.0
    where s({'a','b','c','d'}) = {10,5,7.5,6}
    Global minimum: f=0 at [x0,x1,x2] = [5,4,'b']
    
    Note: Full optimization mode takes several minutes to complete due to 
          Gaussian Process training on 35 samples (20 initial + 15 Bayesian steps).
    
    Args:
        test_mode (bool): If True, use minimal computational load for testing
    """
    import os
    
    # Detect if running in test environment to use reduced computational load
    import sys
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
        print("Running in full optimization mode - this will take several minutes to complete...")
    
    params = [ContinuousVariable(min=0, max=8),           # x0: continuous [0,8]
              OrderedVariable(min_val=2, max_val=6),      # x1: ordered integer [2,6]
              CategoricalVariable(categories=['a','b','c','d'])]  # x2: categorical

    ac_driver = ActiveLoopDriver(simulations=[func_3d_mixedtypes],
                                   params=params,
                                   surrogate='SMT_GP')
    
    if is_testing:
        # Fast testing configuration: minimal samples for pytest  
        ac_driver.initialize(N_samples_init=6)   # Just 6 initial samples for speed
        ac_driver.run(N_steps=2)                 # Only 2 Bayesian steps
    else:
        # Full optimization configuration: better coverage for interactive use
        # Start with ~20 LHS samples (1 per discrete combination), then 15 Bayesian optimization steps
        # Note: 20 LHS samples are not guaranteeded to sample all discrete combinations since LHS tries to cover the volume uniformly, but it should cover a good portion of discrete combinations.
        # For guaranteed coverage, we would need to create a custom initial sample set that includes all discrete combinations.
        ac_driver.initialize(N_samples_init=20)  # Initialize with 20 LHS samples
        ac_driver.run(N_steps=10)                # Then do 15 Bayesian optimization steps

    # Display optimization results
    x_data = ac_driver.dataset.x_data[0]
    y_data = ac_driver.dataset.y_data[0]
    
    # Find the best solution
    best_idx = np.argmin(y_data)
    x_opt = x_data[best_idx]
    y_opt = y_data[best_idx]
    
    print('\nOptimization Results:')
    print('====================')
    print('Expected minimum: y = 0 at [x0, x1, x2] = [5, 4, "b"]')
    categories = ["a","b","c","d"]
    print(f'Found minimum: y = {y_opt[0]:.6f} at [x0, x1, x2] = [{x_opt[0]:.3f}, {x_opt[1]:.0f}, "{categories[int(x_opt[2])]}"]')
    
    # Plot the sampling history for the continuous dimensions
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot x0 vs iteration
    ax1.plot(x_data[:, 0], 'o-', label='x0 (continuous)')
    ax1.axhline(y=5, color='r', linestyle='--', label='Optimal x0=5')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('x0')
    ax1.set_title('x0 vs Iteration')
    ax1.legend()
    ax1.grid(True)
    
    # Plot x1 vs iteration  
    ax2.plot(x_data[:, 1], 'o-', label='x1 (ordered)', color='orange')
    ax2.axhline(y=4, color='r', linestyle='--', label='Optimal x1=4')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('x1')
    ax2.set_title('x1 vs Iteration')
    ax2.legend()
    ax2.grid(True)
    
    # Plot categorical variable selection
    categories = ['a', 'b', 'c', 'd']
    cat_values = [categories[int(x)] for x in x_data[:, 2]]
    ax3.scatter(range(len(cat_values)), x_data[:, 2], c=x_data[:, 2], cmap='tab10')
    ax3.axhline(y=1, color='r', linestyle='--', label='Optimal x2="b"')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('x2 (categorical index)')
    ax3.set_title('x2 vs Iteration')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(categories)
    ax3.legend()
    ax3.grid(True)
    
    # Plot objective function value vs iteration
    ax4.plot(y_data[:, 0], 'o-', label='Objective value', color='green')
    ax4.axhline(y=0, color='r', linestyle='--', label='Global minimum=0')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('f(x)')
    ax4.set_title('Objective Function vs Iteration')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.suptitle('Bayesian 3D Mixed-Types Optimization Results', y=1.02)
    plt.savefig('bayesian_3d_mixedtypes_result.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'bayesian_3d_mixedtypes_result.png'")
    
    # Try to show plot if backend supports it
    try:
        plt.show()
    except Exception as e:
        print(f"Interactive display failed ({e}), but plot was saved")

    return ac_driver

if __name__ == "__main__":
    bayesian_3d_sf_mixedtypes()
