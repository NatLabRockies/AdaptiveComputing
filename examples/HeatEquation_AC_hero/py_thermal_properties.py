
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable, DatasetBase
from adaptive_computing.drivers import ActiveLoopDriver

# Check environment variable from C++ (defaults to "1" / Auto if not set)
enable_gpu_env = os.environ.get("AC_ENABLE_GPU_SURROGATE", "1")

if enable_gpu_env == "0":
    USE_GPU_SURROGATE = False
    print("GPU surrogate disabled by application environment.")
else:
    # Try to import GPU surrogate, fall back to standard SMT if not available
    try:
        import cupy
        from gpu_surrogate_wrapper import GPUSMTGP
        # Check if a GPU is actually available
        if cupy.cuda.runtime.getDeviceCount() > 0:
            USE_GPU_SURROGATE = True
            print("GPU detected. Using CuPy-accelerated surrogate.")
        else:
            USE_GPU_SURROGATE = False
            print("CuPy installed but no GPU detected. Using standard SMT surrogate.")
    except (ImportError, Exception) as e:
        USE_GPU_SURROGATE = False
        print(f"GPU surrogate not available ({e}). Using standard SMT surrogate.")

def func_ThermalConductivity(x):
    return  16 + 0.01 * (x-300)

#def func_SpecificHeatCapacity(x):
#    return  500 + 0.1 * (x-300)

#def func_Density(x):
#    return  7910 - 0.4 * (x-300);


def initialize_driver():
    params = [ContinuousVariable(min=300, max=800)]
    
    if USE_GPU_SURROGATE:
        # Initialize Dataset and GPU Surrogate manually
        dataset = DatasetBase(params, n_fidelity=1)
        surrogate = GPUSMTGP(dataset=dataset)
        
        ac_driver = ActiveLoopDriver(simulations=[func_ThermalConductivity],
                                     params=params,
                                     surrogate=surrogate,
                                     dataset=dataset)
    else:
        # Use standard SMT
        ac_driver = ActiveLoopDriver(simulations=[func_ThermalConductivity],
                                     params=params,
                                     surrogate='SMT')
                                     
    ac_driver.initialize()
    return ac_driver

def print_data(ac_driver):
    print(f"x_data = {ac_driver.dataset.x_data[0]}")
    print(f"y_data = {ac_driver.dataset.y_data[0]}")
    return

def get_kriging_params(ac_driver):
    """
    Extracts Kriging model parameters for GPU implementation.
    Returns a dictionary with all necessary arrays and scalars.
    """
    import numpy as np
    
    # Access the underlying SMT KRG model
    # ac_driver -> surrogate (SMTGP) -> surrogate_model[0] (KRG)
    krg = ac_driver.surrogate.surrogate_model[0]
    
    # Extract parameters
    # Note: SMT KRG structure might vary, this assumes the structure found in inspection
    
    params = {}
    
    # Normalized training points
    params['X_norma'] = krg.X_norma.flatten()
    
    # Weights (gamma) and Mean Offset (beta)
    opt_par = krg.optimal_par
    params['gamma'] = opt_par['gamma'].flatten()
    params['beta'] = opt_par['beta'].flatten()[0] # Assuming scalar beta (poly0)
    
    # Hyperparameters (theta)
    params['theta'] = krg.optimal_theta.flatten()
    
    # Normalization parameters
    params['X_offset'] = krg.X_offset.flatten()
    params['X_scale'] = krg.X_scale.flatten()
    params['y_mean'] = krg.y_mean.flatten()
    params['y_std'] = krg.y_std.flatten()
    
    return params

if __name__ == '__main__':
    ac_driver = initialize_driver()
    print_data(ac_driver)
    x_queries = [[320],[350],[400]]
    print(f"x_queries = {x_queries}")
    y_queries = ac_driver.query(x_queries, 'absolute_variance', 0.1)
    print(f"y_queries = {y_queries}")
    print_data(ac_driver)
    y_queries = ac_driver.query(x_queries, 'absolute_variance', 0.1)
    print(f"y_queries = {y_queries}")
    print_data(ac_driver)
    # expect that the second time, no simulations are launched and the outputs are the same

