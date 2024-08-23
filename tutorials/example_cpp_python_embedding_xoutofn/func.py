
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
    return [(x[0]-3.5)*((x[0]-3.5)) + x[1] + x[2] + x[3],
            (x[6] + x[4])*(x[0] - x[7]),
            (x[5] + x[4])*(x[0] + x[2] - x[1]),
            (x[6]  - x[1] + x[4])*(x[0] - x[2]),
            (x[5] - x[4]*x[3] + x[2])*(x[0] + x[3]),
            (x[5] + x[6])*(x[0] - x[4])]
"""```

Define the design parameters (inputs to the objective function)

```python"""
def init_dataset():

    x0 = Param()
    x0.type = 'continuous'
    x0.min_val = -5
    x0.max_val = 9

    # Use an ordered integer when the order of the discrete values has significance,
    # that is, we expect neigboring values to have objective function values that are correlated
    x1 = Param()
    x1.type = 'continuous'
    x1.min_val = -5
    x1.max_val = 9

    x2 = Param()
    x2.type = 'continuous'
    x2.min_val = -5
    x2.max_val = 9

    x3 = Param()
    x3.type = 'continuous'
    x3.min_val = -5
    x3.max_val = 9

    x4 = Param()
    x4.type = 'continuous'
    x4.min_val = -5
    x4.max_val = 9

    x5 = Param()
    x5.type = 'continuous'
    x5.min_val = -5
    x5.max_val = 9
  
    x6 = Param()
    x6.type = 'continuous'
    x6.min_val = -5
    x6.max_val = 9

    x7 = Param()
    x7.type = 'continuous'
    x7.min_val = -5
    x7.max_val = 9

    # Use categorical type if the order of the categories is arbitrary.
    #x2 = Param()
    #x2.type = 'categorical'
    #x2.categories = ['a','b','c','d']

    params = [x0, x1, x2, x3, x4, x5, x6, x7]
    # Define the options for surrogate modeling and optimization
    ds_ops = DataSetOptions()
    my_dataset = DataSet(func_4d, params, ds_ops, n_out=6)
    my_dataset.add_file_samples('input_data.csv') # >= the number of input arguments of func_4d + 1 (=5) 
    return my_dataset

def init_surrogate(my_dataset, out_index):
    # use the SMT implementation of the Gaussian Process model
    from ac_common.surrogates import SMTWrapper
    surrogate= SMTWrapper(my_dataset, i_out=out_index)
    return surrogate

def if_query(my_dataset, surrogate, x_queries, threshold_std_mean):
    # Query with a std/mean threshold. Conducts simulations if the standard deviation is too high.
    import numpy as np
    y_queries = my_dataset.query_cpp(surrogate,x_queries,threshold_std_mean=threshold_std_mean)
    return y_queries
    #return expected_values, computed_values, tolerances

def if_query_6d(my_dataset, surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5, x_queries, threshold_std):
    # Query with a std/mean threshold. Conducts simulations if the standard deviation is too high.
    import numpy as np
    y_queries = my_dataset.query_cpp_6d(surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5,x_queries,threshold_std=threshold_std)
    return y_queries

def get_variance(my_dataset, surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5, x_queries):
    import numpy as np
    variance = my_dataset.get_variance(surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5, x_queries)
    return variance

def add_xnum_sample(my_dataset, fidelity_level, x_eval_num, y_eval, surrogate):
    my_dataset.add_xnum_sample(fidelity_level, x_eval_num, y_eval = y_eval, surrogate=surrogate)    

def add_xnum_sample_6d(my_dataset, fidelity_level, x_eval_num, y_eval, surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5):
    my_dataset.add_xnum_sample_6d(fidelity_level, x_eval_num, y_eval = y_eval, surrogate0=surrogate0, surrogate1=surrogate1, surrogate2=surrogate2, surrogate3=surrogate3, surrogate4=surrogate4, surrogate5=surrogate5)

def overwrite_data(my_dataset, fidelity_level, x_eval_num, y_eval, surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5):
    my_dataset.overwrite_data(fidelity_level, x_eval_num, y_eval, surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5)

def mask_xnum_sample_6d(my_dataset, fidelity_level, x_eval_num, surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5):
    my_dataset.mask_xnum_sample_6d(fidelity_level, x_eval_num, surrogate0=surrogate0, surrogate1=surrogate1, surrogate2=surrogate2, surrogate3=surrogate3, surrogate4=surrogate4, surrogate5=surrogate5)

def train_on_all_data_6d(my_dataset,surrogate0, surrogate1, surrogate2, surrogate3, surrogate4, surrogate5):
    my_dataset.train_on_all_data(surrogate0,update_masked=False)
    my_dataset.train_on_all_data(surrogate1,update_masked=False)
    my_dataset.train_on_all_data(surrogate2,update_masked=False)
    my_dataset.train_on_all_data(surrogate3,update_masked=False)
    my_dataset.train_on_all_data(surrogate4,update_masked=False)
    my_dataset.train_on_all_data(surrogate5,update_masked=False)

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

