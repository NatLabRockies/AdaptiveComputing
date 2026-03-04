import numpy as np
import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable, OrderedVariable
from adaptive_computing.drivers import ActiveLoopDriver

def func_2d_mixedtypes_debug(x):
    # x will have shape (n_samples, n_in) where n_in=2
    # x[...,0]: continuous variable [0,8]
    # x[...,1]: ordered integer variable [2,6] 
    # Fixed: x2='b' (s=5) for debugging
    
    # Objective: f(x0,x1) = (x0-5)^2 + (x1-4)^2 + 5-5 = (x0-5)^2 + (x1-4)^2
    # Global minimum: f=0 at [x0,x1] = [5,4]
    
    # For vectorized operations, handle both single samples and batch samples
    if x.ndim == 1:
        # Single sample case
        y = (x[0] - 5.0)**2 + (x[1] - 4.0)**2
        return np.array([[y]])
    else:
        # Batch samples case
        batch_size = x.shape[0]
        y = np.zeros(batch_size)
        for i in range(batch_size):
            y[i] = (x[i, 0] - 5.0)**2 + (x[i, 1] - 4.0)**2
        return y.reshape(-1, 1)

def bayesian_2d_mixedtypes_debug():
    """
    Debug script with continuous x0 and ordered x1 to test mixed-type optimization.
    
    Variables:
    - x0: continuous variable [0,8] 
    - x1: ordered integer variable [2,6]
    
    Objective function: f(x0,x1) = (x0-5)^2 + (x1-4)^2
    Global minimum: f=0 at [x0,x1] = [5,4]
    """

    params = [ContinuousVariable(min=0, max=8),           # x0: continuous [0,8]
              OrderedVariable(min_val=2, max_val=6)]      # x1: ordered integer [2,6]

    ac_driver = ActiveLoopDriver(simulations=[func_2d_mixedtypes_debug],
                                   params=params,
                                   surrogate='SMT')
    
    ac_driver.run(N_steps = 15)  # Increased from 5 to 15 for better convergence

    # Display optimization results
    x_data = ac_driver.dataset.x_data[0]
    y_data = ac_driver.dataset.y_data[0]
    
    print(x_data)

    # Find the best solution
    best_idx = np.argmin(y_data)
    x_opt = x_data[best_idx]
    y_opt = y_data[best_idx]
    
    print('\nOptimization Results (Mixed-Type x0+x1 Debug):')
    print('==============================================')
    print('Expected minimum: y = 0 at [x0, x1] = [5.0, 4]')
    print(f'Found minimum: y = {y_opt[0]:.6f} at [x0, x1] = [{x_opt[0]:.6f}, {x_opt[1]:.0f}]')
    print(f'Error in x0: {abs(x_opt[0] - 5.0):.6f}')
    print(f'Error in x1: {abs(x_opt[1] - 4.0):.6f}')
    
    # Debug: Print all data points to see duplicates
    print('\nFull sampling history:')
    print('Iteration | x0       | x1 | y')
    print('-' * 30)
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        print(f'{i:8d} | {x[0]:8.6f} | {x[1]:2.0f} | {y[0]:8.6f}')
    
    # Check for exact duplicates
    unique_points = np.unique(x_data, axis=0)
    print(f'\nTotal points: {len(x_data)}')
    print(f'Unique points: {len(unique_points)}')
    print(f'Duplicates: {len(x_data) - len(unique_points)}')
    
    # Count points per x1 value
    for x1_val in [2, 3, 4, 5, 6]:
        count = np.sum(x_data[:, 1] == x1_val)
        print(f'Points with x1={x1_val}: {count}')
    
    # Plot the sampling history
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot x0 vs iteration
    ax1.plot(x_data[:, 0], 'o-', label='x0 (continuous)', markersize=6)
    ax1.axhline(y=5, color='r', linestyle='--', label='Optimal x0=5', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('x0')
    ax1.set_title('x0 vs Iteration')
    ax1.legend()
    ax1.grid(True)
    
    # Plot x1 vs iteration  
    ax2.plot(x_data[:, 1], 'o-', label='x1 (ordered)', color='orange', markersize=6)
    ax2.axhline(y=4, color='r', linestyle='--', label='Optimal x1=4', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('x1')
    ax2.set_title('x1 vs Iteration')
    ax2.legend()
    ax2.grid(True)
    
    # Plot y vs iteration
    ax3.plot(y_data[:, 0], 'o-', label='Objective value', color='green', markersize=6)
    ax3.axhline(y=0, color='r', linestyle='--', label='Optimal y=0', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Objective value')
    ax3.set_title('Objective Value vs Iteration')
    ax3.legend()
    ax3.grid(True)
    ax3.set_yscale('log')  # Log scale to see convergence better
    
    # 2D scatter plot showing exploration in x0-x1 space
    scatter = ax4.scatter(x_data[:, 0], x_data[:, 1], c=y_data[:, 0], cmap='viridis', s=50)
    ax4.plot(5, 4, 'r*', markersize=15, label='True optimum (5,4)')
    ax4.plot(x_opt[0], x_opt[1], 'rx', markersize=10, label=f'Found optimum ({x_opt[0]:.3f},{x_opt[1]:.0f})')
    ax4.set_xlabel('x0 (continuous)')
    ax4.set_ylabel('x1 (ordered)')
    ax4.set_title('Exploration in x0-x1 Space')
    ax4.legend()
    ax4.grid(True)
    plt.colorbar(scatter, ax=ax4, label='Objective value')
    
    plt.tight_layout()
    plt.savefig('bayesian_2d_mixedtypes_debug.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f'\nSampling history saved to bayesian_2d_mixedtypes_debug.png')
    print(f'Total evaluations: {len(x_data)}')

if __name__ == "__main__":
    bayesian_2d_mixedtypes_debug()