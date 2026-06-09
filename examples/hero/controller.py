# Simplified Hero tutorial demonstrating ActiveLoopDriverHero pattern
# This follows the same pattern as the rental car async demo for clean Hero integration
import numpy as np
import sys
import os

# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriverHero

def compute_conductivity(x):
    """
    Simple test function: conductivity = temperature^2 / 1000
    This matches what our worker.py computes.
    """
    return x * x / 1000.0

def create_task_formatter():
    """
    Create task formatter for the local worker.
    Converts x_data to the format expected by worker.py.
    """
    def format_task(x_data):
        # x_data is a single input array, e.g., [1.5]
        temperature = x_data[0]  # Extract temperature value
        
        # Create task metadata in the format worker.py expects
        return {
            'x_data': [temperature],  # Worker reads temperature from here
            'y_data': None,           # Worker will fill this in
            'slurm_job_id': {'local': -1},  # Required by Hero queue monitoring
            'running': {'local': False}      # Required by Hero queue monitoring
        }
    
    return format_task

def main():
    """
    Simplified Hero demo using ActiveLoopDriverHero pattern.
    """
    print("=== Simplified Hero Tutorial with ActiveLoopDriverHero ===")
    print("This demo shows clean Hero integration using the driver pattern.")
    print()
    
    # Define parameter space: temperature from 0.7 to 2.0
    params = [ContinuousVariable(min=0.7, max=2.0)]
    
    # Setup machine names for local processing
    machine_names = ['local']
    
    # Create task formatter for our simple worker
    task_formatter = create_task_formatter()
    
    # Create ActiveLoopDriverHero with Hero configuration
    print("Creating ActiveLoopDriverHero with Hero configuration...")
    ac_driver = ActiveLoopDriverHero(
        simulations=[None],  # No direct simulation function, using Hero workers
        params=params,
        machine_names=machine_names,
        output_field_path='y_data',  # Where worker puts the result
        surrogate='SMT_GP',  # Use single-output Gaussian Process
        acq_func='expected_improvement',  # Acquisition function for active learning
        blocking=False,      # Non-blocking Hero processing
        nan_behavior='mask_ignore',  # Handle failed Hero tasks gracefully
        task_formatter=task_formatter  # Convert x_data to worker format
    )
    
    # Initialize with Latin Hypercube samples
    print("\\nStep 1: Initializing with 3 LHS samples...")
    ac_driver.initialize(N_samples_init=3)

    # Wait for all Hero tasks to complete before analyzing results
    print("\\nStep 2: Waiting for all Hero tasks to complete...")
    ac_driver.hero_wait_for_data_and_train()
    
    # Show initial data state
    print("\\nInitial data state:")
    dataset = ac_driver.dataset
    print(f"Total samples: {len(dataset.x_data[0])}")
    
    # Get unmasked (completed) data using unified interface
    x_unmasked, y_unmasked = dataset.get_unmasked_data(0)
    print(f"Completed samples: {len(x_unmasked)}")
    print(f"Sample data: {list(zip(x_unmasked.flatten(), y_unmasked.flatten()))}")
    
    # Run 5 Bayesian optimization steps
    print("\\nStep 3: Running 5 Bayesian optimization iterations...")
    ac_driver.run(N_steps=5)
    
    # Wait for all Hero tasks to complete before analyzing results
    print("\\nStep 4: Waiting for all Hero tasks to complete...")
    ac_driver.hero_wait_for_data_and_train()
    
    # Show final results
    print("\\nFinal results:")
    x_unmasked, y_unmasked = dataset.get_unmasked_data(0)
    print(f"Total completed samples: {len(x_unmasked)}")
    print("All temperature -> conductivity mappings:")
    for temp, cond in zip(x_unmasked.flatten(), y_unmasked.flatten()):
        expected = compute_conductivity(temp)
        print(f"  Temperature: {temp:.2f} -> Conductivity: {cond:.6f} (expected: {expected:.6f})")
    
    # Find best sample
    best_idx = np.argmin(y_unmasked)  # Minimize conductivity
    best_temp = x_unmasked[best_idx, 0]
    best_cond = y_unmasked[best_idx, 0]
    print(f"\\nBest sample found:")
    print(f"  Temperature: {best_temp:.3f} -> Conductivity: {best_cond:.6f}")

if __name__ == "__main__":
    main()