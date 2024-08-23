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
def write_samples_csv(dataset,filenames):
    import numpy as np
    import csv

    # validate filenames end with .csv
    filenames = np.atleast_1d(filenames)
    if len(filenames) != dataset.n_fl:
        raise Exception('Must give a list of file names of len(simulations). Use empty quotes for list entries corresponding to fidelity levels that you want to skip writing data for.')
    for filename in filenames:
        if not filename.endswith('.csv'):
            if filename != '':
                raise Exception('csv filename must end in .csv or be an empty string')

    for i_fl in range(dataset.n_fl):
        filename = filenames[i_fl]
        if filename == '':
            print('No ouput data file specified for fidelity level ' + str(i_fl) + '. Skipping write_output_data for this level.')
        else:
            with open(filename,'w', encoding='utf-8-sig') as csvfile: #, newline=''
                writer = csv.writer(csvfile, delimiter=',') # , quotechar='|'
                row = []
                for i_p in range(dataset.n_in):
                    row.append(dataset.params[i_p].type)
                for i_o in range(dataset.n_out):
                    row.append('y'+str(i_o))
                writer.writerow(row)
                for i_s in range(dataset.n_samp[i_fl]):
                    row = []
                    for i_p in range(dataset.n_in):
                        if dataset.params[i_p].type == 'categorical':
                            row.append(dataset.params[i_p].categories[int(dataset.x_data[i_fl][i_s][i_p])])
                        else:
                            row.append(dataset.x_data[i_fl][i_s][i_p])
                    for i_o in range(dataset.n_out):
                        row.append(dataset.y_data[i_fl][i_s][i_o])
                    writer.writerow(row)
                
            
    return

#########################################################
# Return True if y is NaN.
# Otherwise, return False.
def check_nan(y,ds_ops):
    is_nan = False
    if np.isnan(y):
        is_nan = True
    if is_nan and ds_ops.exit_on_nans:
        raise ValueError('NaN returned by user-defined simulation. Exiting because ds_ops.exit_on_nans = True. See README on setting ds_ops.mask_nans.')
    return is_nan

#########################################################
# Returns True if y is out of user-specified bounds (OOB).
# Otherwise, returns False.
def check_oob(y,ds_ops):
    is_oob = False
    if hasattr(ds_ops, 'lbound_inclusive'):
        if y<ds_ops.lbound_inclusive:
            is_oob = True
    if hasattr(ds_ops, 'ubound_inclusive'):
        if y>ds_ops.ubound_inclusive:
            is_oob = True
    if hasattr(ds_ops, 'lbound_exclusive'):
        if y<=ds_ops.lbound_exclusive:
            is_oob = True
    if hasattr(ds_ops, 'ubound_exclusive'):
        if y>=ds_ops.ubound_exclusive:
            is_oob = True
    if is_oob and ds_ops.exit_on_oob_values:
        raise ValueError('User-defined allowable bounds violated by return value from user-defined simulation. Exiting because ds_ops.exit_on_oob_values=True. See README on setting ds_ops.mask_oob_values.')
    return is_oob

#########################################################
# Returns True if the point should be skipped because the output is unmasked and any of the output values are either NaN or OOB.
# Otherwise, returns False.
def check_skip_vec(y_vec,ds_ops,n_out):
    for i_o in range(n_out):
        # check if should skip this point due to NaN value(s)
        if (check_nan(y_vec[i_o],ds_ops) and not ds_ops.mask_nans):
            print('NaN point found. Skipping this point since ds_ops.mask_nans=False. Warning: if using an acqusition function, skipping might lead to continued requerying of the same point. Consider using masking.')
            return True
        
        # check if should skip this point due to OOB value(s)
        if (check_oob(y_vec[i_o],ds_ops) and not ds_ops.mask_oob_values):
            print('Data is out of user-specified allowable bounds. Skipping this point since ds_ops.mask_oob_values=False. Warning: if using an acqusition function, skipping might lead to continued requerying of the same point. Consider using masking.')
            return True
         
    return False
    
#########################################################
def check_unmasked(y,ds_ops):
    unmasked = True
    # check if should mask this point due to NaN value
    if check_nan(y,ds_ops):
        assert(ds_ops.mask_nans) # this code should not be called if you have NaNs that you want to skip.
        unmasked = False
        print('NaN point found. Masking this point since ds_ops.mask_nans=True.')

    # check if should mask this point due to OOB value
    if check_oob(y,ds_ops):
        assert(ds_ops.mask_oob_values) # this code should not be called if you have OOBs that you want to skip.
        unmasked = False
        print('Data is out of user-specified allowable bounds. Masking this point since ds_ops.mask_oob_values=True.')
        
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
