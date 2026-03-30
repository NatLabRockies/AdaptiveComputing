import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the model directory to path
sys.path.append('/home/kgriffin/AdaptiveComputing_1.0/AdaptiveComputing/examples/rental_car/model-aeroportal-rental-car')

from adaptive_computing.datasets import ContinuousVariable, OrderedVariable, CategoricalVariable
from adaptive_computing.drivers import ActiveLoopDriver
from model import rental_car_model, UtilityRate, RentalCarSOC, RentalCarSolution, RentalCarDemand

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

def func_rental_car_smt(x):
    """
    Rental car function using SMT surrogate with proper mixed types support.
    
    Args:
        x: Input array with shape (n_samples, 4)
           x[...,0] = utility_rate (categorical: 0='Moderate', 1='Aggressive') 
           x[...,1] = solution (ordered: 0-4 integer indices)
           x[...,2] = demand (ordered: 0-12 integer indices)
           x[...,3] = soc_mean (ordered: 0-3 integer indices)
    
    Returns:
        y: Output array with shape (n_samples,) representing 'cost'
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
    
    demand_map = [
        RentalCarDemand.DEMAND_00010,
        RentalCarDemand.DEMAND_00100,
        RentalCarDemand.DEMAND_00500,
        RentalCarDemand.DEMAND_01000,
        RentalCarDemand.DEMAND_02000,
        RentalCarDemand.DEMAND_03000,
        RentalCarDemand.DEMAND_04000,
        RentalCarDemand.DEMAND_05000,
        RentalCarDemand.DEMAND_06000,
        RentalCarDemand.DEMAND_07000,
        RentalCarDemand.DEMAND_08000,
        RentalCarDemand.DEMAND_09000,
        RentalCarDemand.DEMAND_10000
    ]
    
    soc_map = [
        RentalCarSOC.SOC_25,
        RentalCarSOC.SOC_35,
        RentalCarSOC.SOC_45,
        RentalCarSOC.SOC_55
    ]
    
    for i in range(n_samples):
        # Extract discrete indices directly from mixed types
        utility_rate_idx = int(x[i, 0])  # Categorical variable already encoded as integer
        solution_idx = int(x[i, 1])      # Ordered variable as integer
        demand_idx = int(x[i, 2])        # Ordered variable as integer  
        soc_idx = int(x[i, 3])           # Ordered variable as integer
        
        # Map indices to enum values
        utility_rate = utility_rate_map[utility_rate_idx]
        solution = solution_map[solution_idx] 
        demand = demand_map[demand_idx]
        soc = soc_map[soc_idx]
        
        try:
            configs = {}  # Empty config as required by the function
            
            # Direct call to rental_car_model - no interpolation needed!
            result = rental_car_model(
                utility_rate=utility_rate,
                return_soc=soc,
                number_of_daily_evs=demand,
                solution=solution,
                config=configs
            )
            
            cost = extract_cost_from_result(result)
            results.append(cost)
            
        except Exception as e:
            print(f"Error calling rental_car_model for sample {i}: {e}")
            results.append(100000.0)  # Default high cost
    
    return np.array(results)

def compute_variance_statistics_over_space(ac_driver):
    """
    Compute the average and maximum predicted variance over the entire 4D discrete input space.
    
    Returns:
        tuple: (average_variance, maximum_variance) across all 520 discrete combinations
    """
    # Generate all possible discrete combinations
    utility_indices = [0, 1]  # Moderate, Aggressive
    solution_indices = list(range(5))  # 0-4
    demand_indices = list(range(13))   # 0-12
    soc_indices = list(range(4))       # 0-3
    
    all_combinations = []
    for util_idx in utility_indices:
        for sol_idx in solution_indices:
            for dem_idx in demand_indices:
                for soc_idx in soc_indices:
                    all_combinations.append([util_idx, sol_idx, dem_idx, soc_idx])
    
    X_all = np.array(all_combinations)
    print(f"  Computing variance for {len(X_all)} discrete combinations...")
    
    try:
        # Get predicted variances from the surrogate
        variances = ac_driver.surrogate.predict_variances(X_all)
        average_variance = np.mean(variances)
        maximum_variance = np.max(variances)
        print(f"  Average variance: {average_variance:.2e}")
        print(f"  Maximum variance: {maximum_variance:.2e}")
        return average_variance, maximum_variance
    except Exception as e:
        print(f"  Error computing variance: {e}")
        return 0.0, 0.0

def AC_surrogate_smt(n_bayes_opt=10, n_initial=20):
    """
    AC version using SMT surrogate with iterative uncertainty quantification visualization.
    Tracks average variance reduction over the entire 4D discrete space.
    
    Args:
        n_bayes_opt: Number of Bayesian optimization iterations to run
        n_initial: Number of initial LHS samples to use
    """
    
    # Define the 4 input parameters using proper mixed types
    params = [
        CategoricalVariable(categories=['Moderate', 'Aggressive']),  # utility_rate: categorical
        OrderedVariable(min_val=0, max_val=4),                      # solution: ordered indices 0-4
        OrderedVariable(min_val=0, max_val=12),                     # demand: ordered indices 0-12
        OrderedVariable(min_val=0, max_val=3)                       # soc_mean: ordered indices 0-3
    ]

    # Create the ActiveLoopDriver with SMT surrogate and maximum variance acquisition
    ac_driver = ActiveLoopDriver(simulations=[func_rental_car_smt],
                                params=params,
                                surrogate='SMT',
                                acq_func='maximum_variance')
    
    print(f"4D discrete space: 2×5×13×4 = {2*5*13*4} total combinations")
    print(f"Running {n_bayes_opt} Bayesian optimization iterations...")
    print("Tracking average variance reduction across entire input space")
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
    initial_avg_variance, initial_max_variance = compute_variance_statistics_over_space(ac_driver)
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
        
        # Compute average and maximum variance over entire space
        avg_variance, max_variance = compute_variance_statistics_over_space(ac_driver)
        
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
    print(f"Surrogate: SMT with native mixed types (categorical + ordered variables)")
    print(f"Average uncertainty reduction: {average_variances[0]:.2e} → {average_variances[-1]:.2e}")
    avg_reduction_percent = (average_variances[0] - average_variances[-1]) / average_variances[0] * 100
    print(f"Average variance reduction: {avg_reduction_percent:.1f}%")
    print(f"Maximum uncertainty reduction: {maximum_variances[0]:.2e} → {maximum_variances[-1]:.2e}")
    max_reduction_percent = (maximum_variances[0] - maximum_variances[-1]) / maximum_variances[0] * 100
    print(f"Maximum variance reduction: {max_reduction_percent:.1f}%")
    
    # Create the uncertainty quantification plot
    print("\\nCreating uncertainty quantification visualization...")
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Sample distribution in 4D space (project to 2D) with legend
    plt.subplot(1, 3, 1)
    x_samples = ac_driver.dataset.x_data[0]
    y_samples = ac_driver.dataset.y_data[0]
    
    # Project 4D to 2D: demand vs cost, colored by utility rate
    demand_labels = ['10', '100', '500', '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000']
    
    # Plot initial samples first
    initial_moderate_x, initial_moderate_y = [], []
    initial_aggressive_x, initial_aggressive_y = [], []
    adaptive_moderate_x, adaptive_moderate_y = [], []
    adaptive_aggressive_x, adaptive_aggressive_y = [], []
    
    # Separate by sample type and utility rate
    for i, (x, y) in enumerate(zip(x_samples, y_samples)):
        utility_idx, solution_idx, demand_idx, soc_idx = x
        demand_val = int(demand_labels[int(demand_idx)])
        
        if i < initial_samples:  # Initial samples
            if utility_idx == 0:
                initial_moderate_x.append(demand_val)
                initial_moderate_y.append(y)
            else:
                initial_aggressive_x.append(demand_val)
                initial_aggressive_y.append(y)
        else:  # Adaptive samples
            if utility_idx == 0:
                adaptive_moderate_x.append(demand_val)
                adaptive_moderate_y.append(y)
            else:
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
    print(f"\\nUncertainty Quantification Results:")
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
    
    print("Running AC surrogate with SMT and HERO rental car model...")
    print("UNCERTAINTY QUANTIFICATION FOCUS: Average variance tracking over 4D space")
    print("4D input: [utility_rate, solution, demand, soc_mean] - native mixed types")
    print("  - utility_rate: categorical ('Moderate', 'Aggressive')")
    print("  - solution: ordered integer (0-4 indices)")
    print("  - demand: ordered integer (0-12 indices)")
    print("  - soc_mean: ordered integer (0-3 indices)")
    print("1D output: [cost] via HERO backend")
    print("Note: Each evaluation calls the actual HERO rental car simulation")
    print("Advantage: SMT surrogate with native mixed types + global uncertainty tracking!")
    print(f"Running {n_bayes_opt} Bayesian iterations (use --n-bayes-opt N to change)")
    print(f"Using {n_initial} initial samples (use --n-initial N to change)")
    print(f"Configuration: {n_initial} initial LHS + {n_bayes_opt} Bayesian = {n_initial + n_bayes_opt} total evaluations")
    print("=" * 80)
    
    ac_driver = AC_surrogate_smt(n_bayes_opt=n_bayes_opt, n_initial=n_initial)