# utils.py
import numpy as np
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
def write_samples_csv(model,filenames):
    import numpy as np
    import csv

    # validate filenames end with .csv
    filenames = np.atleast_1d(filenames)
    if len(filenames) != model.n_fl:
        raise Exception('Must give a list of file names of len(simulations). Use empty quotes for list entries corresponding to fidelity levels that you want to skip writing data for.')
    for filename in filenames:
        if not filename.endswith('.csv'):
            if filename != '':
                raise Exception('csv filename must end in .csv or be an empty string')

    for f in range(model.n_fl):
        filename = filenames[f]
        if filename == '':
            print('No ouput data file specified for fidelity level ' + str(f) + '. Skipping write_output_data for this level.')
        else:
            with open(filename,'w', encoding='utf-8-sig') as csvfile: #, newline=''
                writer = csv.writer(csvfile, delimiter=',') # , quotechar='|'
                row = []
                for i_p in range(model.n_dim):
                    row.append(model.params[i_p].type)
                row.append('y')
                writer.writerow(row)
                for i_d in range(len(model.y_data[f])):
                    row = []
                    for i_p in range(model.n_dim):
                        if model.params[i_p].type == 'categorical':
                            row.append(model.params[i_p].categories[int(model.x_data[f][i_d][i_p])])
                        else:
                            row.append(model.x_data[f][i_d][i_p])
                    row.append(model.y_data[f][i_d][0])
                    writer.writerow(row)
                
            
    return

#########################################################
def check_nan_oob(y,mod_ops):
    unmasked = True
    if np.isnan(y):
        if mod_ops.mask_nans:
            unmasked = False
            print('NaN point found. Masking this point.')
        else:
            raise Exception('NaN returned by user-defined simulation. Consider setting mod_ops.mask_nans=True to ignore NaNs.')
    else:
        oob = False
        if hasattr(mod_ops, 'lbound_inclusive'):
            if y<mod_ops.lbound_inclusive:
                oob = True
        if hasattr(mod_ops, 'ubound_inclusive'):
            if y>mod_ops.ubound_inclusive:
                oob = True
        if hasattr(mod_ops, 'lbound_exclusive'):
            if y<=mod_ops.lbound_exclusive:
                oob = True
        if hasattr(mod_ops, 'ubound_exclusive'):
            if y>=mod_ops.ubound_exclusive:
                oob = True
        if oob:
            if mod_ops.mask_oob_values:
                unmasked = False
                print('Data is out of user-specified allowable bounds. Masking this point.')
            else:
                raise Exception('Allowable bounds violated by return value from user-defined simulation. Consider setting mod_ops.mask_oob_values=True to ignore such values.')
    return unmasked

#########################################################
### test code
# if __name__ == "__main__":
#     from classes import Param
#     AC_path = '../'
#     working_dir = AC_path + '/tutorials/example_read_file'
#     import os
#     os.chdir(working_dir)

#     print('Testing read_input_data with one file: ')
#     x0 = Param(); x0.type = 'continuous'; x0.min_val = 0; x0.max_val = 8
#     x1 = Param(); x1.type = 'ordered'; x1.min_val = 2; x1.max_val = 6
#     x2 = Param(); x2.type = 'categorical'; x2.categories = ['a','b','c','d']
#     params = [x0, x1, x2]
#     filenames = ['existing_data.csv']
#     [x_data, y_data] = read_input_data(filenames,params)

#     print('Testing read_input_data with two files: ')
#     #functions = [is_notebook, is_notebook, is_notebook] # this an array of arbitrary functions
#     filenames = ['','existing_data.csv','existing_data.csv']
#     [x_data, y_data] = read_input_data(filenames,params)

#########################################################
