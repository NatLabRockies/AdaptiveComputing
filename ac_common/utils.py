# utils.py
#########################################################
# return a Boolean indicating if the main program is a Jupyter notebook
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
#########################################################
# # read a csv file and return the contents
def read_input_data(filenames,params,funcs):
    import numpy as np
    import csv

    n_dim = len(params)
    n_fl = len(filenames)

    x_data = []
    y_data = []
    for f in range(n_fl):
        filename = filenames[f]
        if filename == '':
            print('No input data file specified for fidelity level ' + str(f) + '. Skipping read_input_data for this level.')
            x_data.append([])
            y_data.append([])
        else:
            with open(filename, newline='', encoding='utf-8-sig') as csvfile:
                reader = csv.reader(csvfile, delimiter=',') # , quotechar='|'
                a = []
                for row in reader:
                    if (len(row) != n_dim+1) and (len(row) != n_dim):
                        raise Exception('Rows of csv must have length equal to n_dim or n_dim+1.')
                    a.append(row)
            n_samples = len(a) - 1 # first row is header
            if n_samples < 1:
                raise Exception('There is less than 1 row of data (not counting the header) in the csv for fidelity level ' + str(f) + '. Use an empty string as the file name instead.')
            x_data.append(np.zeros([n_samples,n_dim]))
            y_data.append(np.zeros([n_samples,1]))
            
            # move the data from a list of lists to a 2d np array
            for i in range(n_samples):
                for j in range(n_dim):
                    if a[0][j] == 'categorical':
                        x_data[f][i,j] = params[j].categories.index(a[i+1][j])
                    elif (a[0][j] == 'continuous') or (a[0][j] == 'ordered'):
                        x_data[f][i,j] = a[i+1][j]
                    else:
                        raise Exception('Unrecognized type for parameter '+str(i))    
                if len(a[i+1]) == n_dim: # if the user has not specified any objective function values
                    y_data[f][i,0] = funcs[f](x_data[f][i,:])
                elif a[i+1][n_dim] == '': # elif the user has specified some objective function values, but not the present row's objective function value
                    y_data[f][i,0] = funcs[f](x_data[f][i,:])
                else: # fir the user has specified the parameters and the corresponding objective function evaluations
                    y_data[f][i,0] = a[i+1][n_dim]

    return [x_data, y_data]
#########################################################
# # read a csv file and return the contents
def write_output_data(filenames,params,x_data,y_data):
    import numpy as np
    import csv

    n_dim = len(params)
    n_fl = len(filenames)

    for f in range(n_fl):
        filename = filenames[f]
        if filename == '':
            print('No ouput data file specified for fidelity level ' + str(f) + '. Skipping write_output_data for this level.')
        else:
            with open(filename,'w', encoding='utf-8-sig') as csvfile: #, newline=''
                writer = csv.writer(csvfile, delimiter=',') # , quotechar='|'
                row = []
                for i_p in range(n_dim):
                    row.append(params[i_p].type)
                row.append('y')
                writer.writerow(row)
                for i_d in range(len(y_data[f])):
                    row = []
                    for i_p in range(n_dim):
                        if params[i_p].type == 'categorical':
                            row.append(params[i_p].categories[int(x_data[f][i_d][i_p])])
                        else:
                            row.append(x_data[f][i_d][i_p])
                    row.append(y_data[f][i_d][0])
                    writer.writerow(row)
                
            
    return
#########################################################
### test code
if __name__ == "__main__":
    from classes import Param
    AC_path = '../'
    working_dir = AC_path + '/tutorials/example_read_file'
    import os
    os.chdir(working_dir)

    print('Testing read_input_data with one file: ')
    x0 = Param(); x0.type = 'continuous'; x0.min_val = 0; x0.max_val = 8
    x1 = Param(); x1.type = 'ordered'; x1.min_val = 2; x1.max_val = 6
    x2 = Param(); x2.type = 'categorical'; x2.categories = ['a','b','c','d']
    params = [x0, x1, x2]
    filenames = ['existing_data.csv']
    [x_data, y_data] = read_input_data(filenames,params)

    print('Testing read_input_data with two files: ')
    #functions = [is_notebook, is_notebook, is_notebook] # this an array of arbitrary functions
    filenames = ['','existing_data.csv','existing_data.csv']
    [x_data, y_data] = read_input_data(filenames,params)

#########################################################