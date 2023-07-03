# pretreat.py
# This function only completes the pretreat and enzymatic hydrolysis steps of the
# virtualEngineering biofuels synthesis reaction
# The code is a modified version of virtual_engineering_notebook.md

#from ipywidgets import *

def ve(x_input): 
    from ipywidgets import widgets
    #from IPython.display import HTML, clear_output
    import os
    import sys
    import numpy as np
    prev_dir = os.getcwd()
    #print('Initial working directory: '),
    #print(prev_dir)
    ve_dir = '../../../VirtualEngineering/'
    #sys.path.insert(0, ve_dir) # add the path to the AdaptiveComputing common folder
    # add path for no-CFD EH model
    sys.path.append(os.path.join(ve_dir, "submodules/CEH_EmpiricalModel/src/core/"))
    sys.path.insert(0,'') # this tells python to search where ever the current code is located
    os.chdir(ve_dir)
    #print('Changed working directory to: '),
    #print(os.getcwd())
    notebook_dir = os.getcwd()
    # imports from vebio modules
    import vebio.WidgetFunctions as wf
    from vebio.FileModifiers import write_file_with_replacements
    from vebio.Utilities import get_host_computer, yaml_to_dict, dict_to_yaml
    from vebio.RunFunctions import run_pretreatment, run_enzymatic_hydrolysis, run_bioreactor
    # See if we're running on Eagle or on a laptop
    hpc_run = get_host_computer()

    # set the feedstock and pretreatment options
    fs_options = wf.WidgetCollection()
    fs_options.xylan_solid_fraction = widgets.BoundedFloatText(value = x_input[0])
    fs_options.glucan_solid_fraction = widgets.BoundedFloatText(value = x_input[1])
    fs_options.initial_porosity = widgets.BoundedFloatText(value = x_input[2])
    pt_options = wf.WidgetCollection()
    pt_options.initial_acid_conc = widgets.BoundedFloatText(value = x_input[3]) 
    pt_options.steam_temperature = widgets.BoundedFloatText(value = x_input[4])
    pt_options.steam_temperature.scaling_fn = lambda C : C + 273.15 # conversion from C to K
    pt_options.initial_solid_fraction = widgets.BoundedFloatText(value = x_input[5])
    pt_options.final_time = widgets.BoundedFloatText(value = 8.3, max = 1440, min = 1, )
    pt_options.final_time.scaling_fn = lambda s : 60.0 * s # Conversion from minutes to seconds
    pt_options.show_plots = widgets.Checkbox(value = False)

    ## set the enzymatic hydrolysis options
    eh_options = wf.WidgetCollection()
    # ...
    eh_options.model_type = widgets.RadioButtons(options = ['Lignocellulose Model', 'CFD Surrogate', 'CFD Simulation'],value = 'CFD Surrogate')
    eh_options.lambda_e = widgets.BoundedFloatText(value = x_input[6]) # Enzymatic Load: Ratio of the enzyme mass to the total solution mass (mg/g).
    # Conversion from mg/g to kg/kg
    eh_options.lambda_e.scaling_fn = lambda e : 0.001 * e
    eh_options.fis_0 = widgets.BoundedFloatText(value = x_input[7]) # FIS_0 Target: the target value for initial fraction of insoluble solids *after* dilution (kg/kg)
    eh_options.t_final = widgets.BoundedFloatText(value = 24.0)
    eh_options.show_plots = widgets.Checkbox(value = False)
    
    ## set the bioreactor options
    br_options = wf.WidgetCollection()
    br_options.model_type = widgets.RadioButtons(options = ['CFD Surrogate', 'CFD Simulation'],value = 'CFD Surrogate')
    br_options.t_final = widgets.BoundedFloatText(value = 100.0)

    print('inputs = ' + str(x_input) )

    # Set global paths and files for communication between operations
    params_filename = 'virteng_params.yaml'
    # Run the pretreatment model
    run_pretreatment(notebook_dir, params_filename, fs_options, pt_options)
    
    # Run the enzymatic hydrolysis model
    run_enzymatic_hydrolysis(notebook_dir, params_filename, eh_options, hpc_run)
    
    # Run the bioreactor model
    run_bioreactor(notebook_dir, params_filename, br_options, hpc_run)
    
    ve_params = yaml_to_dict(params_filename)
    #xG0 = ve_params['pretreatment_output']['X_G']
    #xX0 = ve_params['pretreatment_output']['X_X']
    our = ve_params['bioreactor_output']['our']
    #stats = [xG0, xX0] # If this file is read after pretreatment, these are pretreatment outputs and EH inputs (hence the zero)
    #print(stats)
    print('y = ' + str(-our))
    os.chdir(prev_dir)
    #print('Changed working directory back to: '),
    #print(os.getcwd())
    #return stats 
    return -our # return negative so that it AC maximizes
    
### test code
if __name__ == "__main__":
    ve([ 0.263, 0.4, 0.8, 0.0001, 150, 0.745, 30.0, 0.05 ])
