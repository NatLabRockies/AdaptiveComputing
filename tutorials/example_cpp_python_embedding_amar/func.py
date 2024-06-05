
import sys
sys.path.insert(0, '../../') # add the path to the AdaptiveComputing directory
from ac_common import *    
import numpy as np
import matplotlib.pyplot as plt
import math

# define the microscale (Kinetic Monte Carlo) simulation
# this function takes Temperature, Pressure, and compositions as input
# returns the flux on the surface
def func_4d(x):
    
    #return  math.sin((x[0] - 3.5) * ((x[0] - 3.5)) + x[1] + x[2] + x[3])
    return (x[0]-3.5)*((x[0]-3.5)) + x[1] + x[2] + x[3]
"""```

Define the design parameters (inputs to the objective function)

```python"""
def init_dataset():
    

    T = Param() # Temperature
    T.type = 'continuous'
    T.min_val = 0
    T.max_val = 300
    P = Param() # Pressure
    P.type = 'continuous'
    P.min_val = 0
    P.max_val = 100
    x0= Param() # composition of species 0
    x0.type = 'continuous'
    x0.min_val = 0
    x0.max_val = 1
    x1= Param() # composition of species 1
    x1.type = 'continuous'
    x1.min_val = 0
    x1.max_val = 1

    params = [T, P, x0, x1]

    # Define the options for surrogate modeling and optimization
    ds_ops = DataSetOptions()
    my_dataset = DataSet(func_4d, params, ds_ops)
    my_dataset.add_lhs_samples(10) # >= the number of input arguments of func_4d + 1 (=5) 
    return my_dataset

def init_surrogate(my_dataset):
    # use the SMT implementation of the Gaussian Process model
    from ac_common.surrogate_wrappers import SMTWrapper
    surrogate= SMTWrapper(my_dataset)
    return surrogate

def if_query(my_dataset, surrogate, x_queries, threshold_std_mean):
    # Query with a std/mean threshold. Conducts simulations if the standard deviation is too high.
    import numpy as np
    y_queries = my_dataset.query_cpp(surrogate,x_queries,threshold_std_mean=threshold_std_mean)
    return y_queries
    #return expected_values, computed_values, tolerances
    
def add_xnum_sample(my_dataset, fidelity_level, x_eval_num, y_eval):
    print("Re-evaluating HF model")
    #viz_ops = VizOptions()
    #viz_ops.plot_nd = True
    viz_ops = None # workaround to avoid vizops on a surrogate=None
    my_dataset.add_xnum_sample(fidelity_level, x_eval_num, y_eval = y_eval, viz_ops = viz_ops)

def dynamic_if_query(my_dataset, surrogate, x_queries, time_ratio, computer_budget_ratio):
    y_queries = my_dataset.dynamic_query_cpp(surrogate, x_queries, time_ratio = time_ratio, computer_budget_ratio = computer_budget_ratio)
    return y_queries

def print_stmt(lf, hf, cpu_elapsed):
    print("HF Eval: ", hf, " LF Eval: ", lf)
    print("CPU Elapsed: ", cpu_elapsed)
    #plot_graphs()
    
def write_output(iter, computer_budget_ratio, time_ratio, threshold_std_dyn, hf_count, lf_count):
    #write results to csv
    import os.path
    csvfilename = 'output.csv'
    row = [iter, computer_budget_ratio, time_ratio, threshold_std_dyn, hf_count, lf_count]        
    import csv
    with open(csvfilename, 'a', newline='', encoding='utf-8') as fd:
        csvwriter = csv.writer(fd, delimiter=',')
        csvwriter.writerow(row)
def plot_graphs():
    #Plot graphs         row = [computer_budget_ratio, time_ratio, threshold_std_dyn, np.sqrt(y_queries_var[i])[0]]
    comp_budget = []
    time_ratio = []
    threshold = []
    var = []
    lf = []
    hf = []
    iter = []
    import csv
    with open(r'C:\Users\JanelleDomantay\Documents\GitHub\AdaptiveComputing\tutorials\example_cpp_python_embedding_amar\output.csv', 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        
        for row in plots:
            if len(row) > 1:
                iter.append(row[0])
                comp_budget.append(float(row[1]))
                time_ratio.append(float(row[2]))
                threshold.append(float(row[3]))                
                hf.append(float(row[4]))
                lf.append(float(row[5]))

    
                
    x = np.linspace(0, 1, 6400)  
    plt.figure()
    #plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
    plt.plot([0,1], [0, 1],  color = 'b', ls="--", label = "x = y")   
    plt.plot(time_ratio, comp_budget, color = 'r', label = "CPU vs Time")     
    plt.ylabel("CPU Elapsed")
    plt.xlabel("Time")
    
    
    
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(time_ratio, threshold, color = 'g')
    plt.ylabel("Threshold")    
    plt.xlabel("Time")

    plt.plot(hf, lf, color = 'r', ls="-", label = "High-Fidelity vs Low-Fidelity")    
    plt.xlabel("High-Fidelity")
    plt.ylabel("Low-Fidelity")
    plt.legend()
    plt.show()
    
    print("DONE")
 
if __name__ == '__main__':
    print("Calling func.py")

