import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the model directory to path
sys.path.append('/home/kgriffin/AdaptiveComputing_1.0/AdaptiveComputing/examples/rental_car/model-aeroportal-rental-car')

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.drivers import ActiveLoopDriver
from model import rental_car_model, UtilityRate, RentalCarSOC, RentalCarSolution, RentalCarDemand

def bilinear_interpolation(x, x1, x2, y1, y2, f11, f12, f21, f22):
    """
    Perform bilinear interpolation.
    
    Args:
        x, y: target point coordinates
        x1, x2, y1, y2: grid boundaries 
        f11, f12, f21, f22: function values at grid corners
        
    Returns:
        interpolated value
    """
    y = x  # In our case, we're doing 1D interpolation for each variable separately
    
    # First interpolate in x direction
    f1 = f11 * (x2 - x) / (x2 - x1) + f21 * (x - x1) / (x2 - x1)
    f2 = f12 * (x2 - x) / (x2 - x1) + f22 * (x - x1) / (x2 - x1)
    
    # Then interpolate in y direction  
    f = f1 * (y2 - y) / (y2 - y1) + f2 * (y - y1) / (y2 - y1)
    
    return f

def map_continuous_to_discrete(value, allowed_values):
    """
    Map a continuous value to the nearest discrete values for interpolation.
    
    Args:
        value: continuous input value
        allowed_values: list of allowed discrete values
        
    Returns:
        (lower_val, upper_val, lower_idx, upper_idx, weight)
    """
    allowed_values = np.array(sorted(allowed_values))
    
    # Clamp value to range
    value = np.clip(value, allowed_values[0], allowed_values[-1])
    
    # Find surrounding values
    if value <= allowed_values[0]:
        return allowed_values[0], allowed_values[0], 0, 0, 1.0
    elif value >= allowed_values[-1]:
        return allowed_values[-1], allowed_values[-1], -1, -1, 1.0
    else:
        # Find the two surrounding values
        upper_idx = np.searchsorted(allowed_values, value)
        lower_idx = upper_idx - 1
        
        lower_val = allowed_values[lower_idx]
        upper_val = allowed_values[upper_idx]
        
        # Calculate interpolation weight
        weight = (value - lower_val) / (upper_val - lower_val)
        
        return lower_val, upper_val, lower_idx, upper_idx, weight

def map_utility_rate(value):
    """Map 0-1 value to UtilityRate enum."""
    if value < 0.5:
        return UtilityRate.MODERATE
    else:
        return UtilityRate.AGGRESSIVE

def map_solution(value):
    """Map 0-4 value to RentalCarSolution enum."""
    solutions = [
        RentalCarSolution.GRID,
        RentalCarSolution.STORAGE_025, 
        RentalCarSolution.STORAGE_050,
        RentalCarSolution.STORAGE_075,
        RentalCarSolution.STORAGE_100
    ]
    idx = int(np.round(np.clip(value, 0, 4)))
    return solutions[idx]

def func_4d_hero(x):
    """
    4D function for rental car cost optimization with all parameters varying.
    Uses continuous variable interpolation for SOOGO surrogate.
    
    Args:
        x: Input array with shape (4,) or (n_samples, 4)
           x[...,0] = utility_rate (continuous: 0-1) 
           x[...,1] = solution (continuous: 0-4)
           x[...,2] = demand (continuous: 1000-10000)
           x[...,3] = soc_mean (continuous: 25-55)
    
    Returns:
        y: Output representing 'cost'
    """
    # Handle batch input
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_samples = x.shape[0]
    results = []
    
    # Define the mappings for discrete variables
    utility_rate_map = [UtilityRate.MODERATE, UtilityRate.AGGRESSIVE]
    solution_map = [
        RentalCarSolution.GRID,
        RentalCarSolution.STORAGE_025, 
        RentalCarSolution.STORAGE_050,
        RentalCarSolution.STORAGE_075,
        RentalCarSolution.STORAGE_100
    ]
    
    demand_values = [10, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    soc_values = [25, 35, 45, 55]
    
    for i in range(n_samples):
        # Extract continuous values
        utility_val = x[i, 0]    # 0-1
        solution_val = x[i, 1]   # 0-4
        demand_val = x[i, 2]     # 1000-10000
        soc_val = x[i, 3]        # 25-55
        
        # Map continuous values to discrete enum values using interpolation
        
        # Utility rate: threshold at 0.5
        utility_rate = map_utility_rate(utility_val)
        
        # Solution: round to nearest integer and map
        solution = map_solution(solution_val)
        
        # Demand: interpolate between nearest values
        demand_low, demand_high, _, _, demand_weight = map_continuous_to_discrete(demand_val, demand_values)
        
        # SOC: interpolate between nearest values  
        soc_low, soc_high, _, _, soc_weight = map_continuous_to_discrete(soc_val, soc_values)
        
        try:
            configs = {}  # Empty config as required by the function
            
            # We need to evaluate at the corners of the demand x soc interpolation grid
            costs = []
            for demand_corner, soc_corner in [(demand_low, soc_low), (demand_low, soc_high), 
                                            (demand_high, soc_low), (demand_high, soc_high)]:
                try:
                    result = rental_car_model(
                        utility_rate=utility_rate,
                        return_soc=RentalCarSOC(int(soc_corner)),
                        number_of_daily_evs=RentalCarDemand(int(demand_corner)),
                        solution=solution,
                        config=configs
                    )
                    cost = extract_cost_from_result(result)
                    costs.append(cost)
                except Exception as e:
                    print(f"Error calling rental_car_model for corner ({demand_corner}, {soc_corner}): {e}")
                    costs.append(100000.0)
            
            # Bilinear interpolation over demand x soc grid
            if demand_high != demand_low and soc_high != soc_low:
                # Full bilinear interpolation
                cost_00, cost_01, cost_10, cost_11 = costs
                
                # Interpolate in demand direction first
                cost_0 = cost_00 * (1 - demand_weight) + cost_10 * demand_weight
                cost_1 = cost_01 * (1 - demand_weight) + cost_11 * demand_weight
                
                # Then interpolate in soc direction
                interpolated_cost = cost_0 * (1 - soc_weight) + cost_1 * soc_weight
                
            elif demand_high != demand_low:
                # Linear interpolation in demand only
                cost_0, cost_1 = costs[0], costs[2]  # Same soc, different demand
                interpolated_cost = cost_0 * (1 - demand_weight) + cost_1 * demand_weight
                
            elif soc_high != soc_low:
                # Linear interpolation in soc only  
                cost_0, cost_1 = costs[0], costs[1]  # Same demand, different soc
                interpolated_cost = cost_0 * (1 - soc_weight) + cost_1 * soc_weight
                
            else:
                # Exact match
                interpolated_cost = costs[0]
                
            results.append(interpolated_cost)
            
        except Exception as e:
            print(f"Error calling rental_car_model for sample {i}: {e}")
            results.append(100000.0)  # Default high cost
    
    if len(results) == 1:
        return results[0]
    else:
        return np.array(results)

def extract_cost_from_result(result):
    """
    Extract cost value from rental_car_model result.
    
    Args:
        result: Result dictionary from rental_car_model
        
    Returns:
        float: Cost value
    """
    try:
        if isinstance(result, dict):
            if 'metadata' in result and 'Task' in result['metadata'] and 'response' in result['metadata']['Task']:
                response = result['metadata']['Task']['response']
                if 'cost' in response:
                    return float(response['cost'])
        
        return 100000.0  # Fallback value
        
    except Exception as e:
        print(f"Error extracting cost: {e}")
        return 100000.0

def compute_variance_statistics_over_continuous_space(ac_driver, n_samples=100):
    """
    Compute the average and maximum predicted variance over a sampled continuous input space.
    
    Args:
        ac_driver: Active learning driver with trained surrogate
        n_samples: Number of random samples to use for variance estimation
        
    Returns:
        tuple: (average_variance, maximum_variance) across sampled points
    """
    # Generate random samples across the 4D continuous space
    np.random.seed(42)  # For reproducible variance estimates
    random_samples = np.random.rand(n_samples, 4)
    
    # Scale to proper ranges
    random_samples[:, 0] = random_samples[:, 0]  # utility_rate: 0-1
    random_samples[:, 1] = random_samples[:, 1] * 4  # solution: 0-4
    random_samples[:, 2] = random_samples[:, 2] * 9000 + 1000  # demand: 1000-10000
    random_samples[:, 3] = random_samples[:, 3] * 30 + 25  # soc_mean: 25-55
    
    print(f"  Computing variance for {n_samples} random continuous samples...")
    
    try:
        # Get predicted variances from the surrogate
        variances = ac_driver.surrogate.predict_variances(random_samples)
        average_variance = np.mean(variances)
        maximum_variance = np.max(variances)
        print(f"  Average variance: {average_variance:.2e}")
        print(f"  Maximum variance: {maximum_variance:.2e}")
        return average_variance, maximum_variance
    except Exception as e:
        print(f"  Error computing variance: {e}")
        return 0.0, 0.0

def AC_surrogate_hero(n_bayes_opt=10, n_initial=20):
    """
    AC version using SOOGO surrogate with iterative uncertainty quantification visualization.
    Tracks average variance reduction over a sampled continuous 4D space.
    
    Args:
        n_bayes_opt: Number of Bayesian optimization iterations to run
        n_initial: Number of initial LHS samples to use
    """
    
    # Define the 4 input parameters to match rental car model
    # Scaling ranges to match the enum bounds
    params = [
        ContinuousVariable(min=0, max=1),        # utility_rate (0=Moderate, 1=Aggressive)
        ContinuousVariable(min=0, max=4),        # solution (0=Grid, 4=Storage-100)  
        ContinuousVariable(min=1000, max=10000), # demand (RentalCarDemand range)
        ContinuousVariable(min=25, max=55)       # soc_mean (RentalCarSOC range)
    ]

    # Create the ActiveLoopDriver with SOOGO surrogate and maximum variance acquisition
    ac_driver = ActiveLoopDriver(simulations=[func_4d_hero],
                                params=params,
                                surrogate='SOOGO',
                                acq_func='maximum_variance')
    
    print(f"4D continuous space: utility_rate×solution×demand×soc_mean")
    print(f"Running {n_bayes_opt} Bayesian optimization iterations...")
    print("Tracking average variance reduction across continuous input space")
    print("-" * 60)
    
    # Initialize with LHS samples
    print(f"\nInitializing with {n_initial} LHS samples...")
    ac_driver.initialize(N_samples_init=n_initial)
    initial_samples = len(ac_driver.dataset.x_data[0])
    print(f"Initial samples: {initial_samples}")
    
    # Track variance over iterations
    iterations = []
    average_variances = []
    maximum_variances = []
    total_evaluations = []
    
    # Compute initial variance after LHS sampling
    print(f"\nIteration 0 (after initial LHS):")
    initial_avg_variance, initial_max_variance = compute_variance_statistics_over_continuous_space(ac_driver)
    iterations.append(0)
    average_variances.append(initial_avg_variance)
    maximum_variances.append(initial_max_variance)
    total_evaluations.append(len(ac_driver.dataset.x_data[0]))
    
    # Run iterative Bayesian optimization
    for i in range(1, n_bayes_opt + 1):
        print(f"\nIteration {i}:")
        print(f"  Running 1 Bayesian optimization step...")
        
        # Run one step of Bayesian optimization
        ac_driver.run(N_steps=1)
        
        # Compute average and maximum variance over sampled space
        avg_variance, max_variance = compute_variance_statistics_over_continuous_space(ac_driver)
        
        # Store results
        iterations.append(i)
        average_variances.append(avg_variance)
        maximum_variances.append(max_variance)
        total_evaluations.append(len(ac_driver.dataset.x_data[0]))
        
        print(f"  Total evaluations so far: {total_evaluations[-1]}")

    # Print results summary
    total_samples = len(ac_driver.dataset.x_data[0])
    print(f"\n{'='*60}")
    print(f"Iterative Bayesian optimization completed!")
    print(f"Total HERO evaluations: {total_samples}")
    print(f"Initial samples: {initial_samples}")
    print(f"Bayesian optimization samples: {total_samples - initial_samples}")
    print(f"Focus: uncertainty quantification via maximum variance acquisition")
    print(f"Surrogate: SOOGO with continuous variables")
    print(f"Average uncertainty reduction: {average_variances[0]:.2e} → {average_variances[-1]:.2e}")
    avg_reduction_percent = (average_variances[0] - average_variances[-1]) / average_variances[0] * 100
    print(f"Average variance reduction: {avg_reduction_percent:.1f}%")
    print(f"Maximum uncertainty reduction: {maximum_variances[0]:.2e} → {maximum_variances[-1]:.2e}")
    max_reduction_percent = (maximum_variances[0] - maximum_variances[-1]) / maximum_variances[0] * 100
    print(f"Maximum variance reduction: {max_reduction_percent:.1f}%")
    
    # Create the uncertainty quantification plot
    print("\nCreating uncertainty quantification visualization...")
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Sample distribution in 4D space (project to 2D) with legend
    plt.subplot(1, 3, 1)
    x_samples = ac_driver.dataset.x_data[0]
    y_samples = ac_driver.dataset.y_data[0]
    
    # Project 4D to 2D: demand vs cost, colored by utility rate
    
    # Separate by sample type and utility rate (using continuous utility rate threshold)
    initial_moderate_x, initial_moderate_y = [], []
    initial_aggressive_x, initial_aggressive_y = [], []
    adaptive_moderate_x, adaptive_moderate_y = [], []
    adaptive_aggressive_x, adaptive_aggressive_y = [], []
    
    # Separate by sample type and utility rate
    for i, (x, y) in enumerate(zip(x_samples, y_samples)):
        utility_val, solution_val, demand_val, soc_val = x
        
        if i < initial_samples:  # Initial samples
            if utility_val < 0.5:  # Moderate utility (< 0.5)
                initial_moderate_x.append(demand_val)
                initial_moderate_y.append(y)
            else:  # Aggressive utility (>= 0.5)
                initial_aggressive_x.append(demand_val)
                initial_aggressive_y.append(y)
        else:  # Adaptive samples
            if utility_val < 0.5:  # Moderate utility (< 0.5)
                adaptive_moderate_x.append(demand_val)
                adaptive_moderate_y.append(y)
            else:  # Aggressive utility (>= 0.5)
                adaptive_aggressive_x.append(demand_val)
                adaptive_aggressive_y.append(y)
    
    # Plot with different markers and colors
    plt.scatter(initial_moderate_x, initial_moderate_y, c='lightblue', marker='^', s=60, 
               edgecolor='blue', linewidth=1, alpha=0.7, label='Initial (Moderate)')
    plt.scatter(initial_aggressive_x, initial_aggressive_y, c='lightcoral', marker='^', s=60, 
               edgecolor='red', linewidth=1, alpha=0.7, label='Initial (Aggressive)')
    plt.scatter(adaptive_moderate_x, adaptive_moderate_y, c='blue', marker='o', s=60, 
               edgecolor='darkblue', linewidth=1, alpha=0.9, label='Adaptive (Moderate)')
    plt.scatter(adaptive_aggressive_x, adaptive_aggressive_y, c='red', marker='o', s=60, 
               edgecolor='darkred', linewidth=1, alpha=0.9, label='Adaptive (Aggressive)')
    
    plt.xlabel('Daily EV Demand')
    plt.ylabel('Cost ($)')
    plt.title('Sample Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize='small')
    
    # Plot 2: Relative average variance
    plt.subplot(1, 3, 2)
    if len(average_variances) > 1:
        relative_avg_variances = [avg_var / average_variances[0] for avg_var in average_variances]
        plt.plot(iterations, relative_avg_variances, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Bayesian Optimization Iteration')
        plt.ylabel('Relative Average Variance')
        plt.title('Relative Average Variance Reduction')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
    
    # Plot 3: Relative maximum variance  
    plt.subplot(1, 3, 3)
    if len(maximum_variances) > 1:
        relative_max_variances = [max_var / maximum_variances[0] for max_var in maximum_variances]
        plt.plot(iterations, relative_max_variances, 'r-s', linewidth=2, markersize=6)
        plt.xlabel('Bayesian Optimization Iteration')
        plt.ylabel('Relative Maximum Variance')
        plt.title('Relative Maximum Variance Reduction')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    # Print variance reduction summary
    print(f"\nUncertainty Quantification Results:")
    for i, (iter_num, avg_var, max_var, n_evals) in enumerate(zip(iterations, average_variances, maximum_variances, total_evaluations)):
        if i == 0:
            print(f"  Initial (after LHS): avg={avg_var:.2e}, max={max_var:.2e}, {n_evals} evaluations")
        else:
            avg_reduction = (average_variances[0] - avg_var) / average_variances[0] * 100
            max_reduction = (maximum_variances[0] - max_var) / maximum_variances[0] * 100
            print(f"  Iteration {iter_num}: avg={avg_var:.2e} ({avg_reduction:.1f}% reduction), max={max_var:.2e} ({max_reduction:.1f}% reduction), {n_evals} evaluations")
    
    return ac_driver

if __name__ == "__main__":
    import sys
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parse command line arguments
    n_bayes_opt = 15  # Default Bayesian optimization iterations
    n_initial = 10    # Default initial samples
    
    if '--n-bayes-opt' in sys.argv:
        try:
            idx = sys.argv.index('--n-bayes-opt')
            n_bayes_opt = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print(f"Warning: Invalid --n-bayes-opt argument, using default ({n_bayes_opt})")
            
    if '--n-initial' in sys.argv:
        try:
            idx = sys.argv.index('--n-initial')
            n_initial = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print(f"Warning: Invalid --n-initial argument, using default ({n_initial})")
    
    print("Running AC surrogate with SOOGO and HERO rental car model...")
    print("UNCERTAINTY QUANTIFICATION FOCUS: Average variance tracking over 4D space")
    print("4D input: [utility_rate, solution, demand, soc_mean] - continuous variables")
    print("  - utility_rate: continuous (0-1)")
    print("  - solution: continuous (0-4)")
    print("  - demand: continuous (1000-10000)")
    print("  - soc_mean: continuous (25-55)")
    print("1D output: [cost] via HERO backend")
    print("Note: Each evaluation calls the actual HERO rental car simulation")
    print("Advantage: SOOGO surrogate with continuous variables + global uncertainty tracking!")
    print(f"Running {n_bayes_opt} Bayesian iterations (use --n-bayes-opt N to change)")
    print(f"Using {n_initial} initial samples (use --n-initial N to change)")
    print(f"Configuration: {n_initial} initial LHS + {n_bayes_opt} Bayesian = {n_initial + n_bayes_opt} total evaluations")
    print("="* 80)
    
    ac_driver = AC_surrogate_hero(n_bayes_opt=n_bayes_opt, n_initial=n_initial)