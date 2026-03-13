# Run all examples
# To run this script, call "pytest" from the "AdaptiveComputing/" directory
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def test_bayesian_1d_sf(monkeypatch):
    dir_name = 'bayesian_1d_sf'
    py_name = 'bayesian_1d_sf'
    ac_driver = run_example(monkeypatch,dir_name,py_name)

    # check the minimum of the surrogate model
    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    computed_output = [x_opt[0], y_opt]

    # compare expected and computed outputs
    expected_output = [0.757249, -6.02074] # analytical solution = [x_min, y_min]
    tolerances = [0.1, 0.1]
    output_validator(expected_output, computed_output, tolerances)

    return

def test_bayesian_2d_sf(monkeypatch):
    dir_name = 'bayesian_2d_sf'
    py_name = 'bayesian_2d_sf'
    ac_driver = run_example(monkeypatch,dir_name,py_name)

    # check the minimum of the surrogate model
    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    computed_output = [x_opt[0], x_opt[1], y_opt]

    # compare expected and computed outputs
    expected_output = [3.0, 6.2, 1.4] # analytical solution = [x0_min, x1_min, y_min]
    tolerances = [0.2, 0.2, 0.1]
    output_validator(expected_output, computed_output, tolerances)

    return

def test_bayesian_1d_mf(monkeypatch):
    dir_name = 'bayesian_1d_mf'
    py_name = 'bayesian_1d_mf'
    ac_driver = run_example(monkeypatch,dir_name,py_name)

    # check the minimum of the surrogate model
    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    computed_output = [x_opt[0], y_opt]

    # compare expected and computed outputs
    expected_output = [3.0, 0.0] # analytical solution = [x_min, y_min]
    tolerances = [0.1, 0.1]
    output_validator(expected_output, computed_output, tolerances)

    return

"""
def test_custom_mf_workflow(monkeypatch):
    dir_name = 'custom_mf_workflow'
    py_name = 'custom_mf_workflow'
    ac_driver = run_example(monkeypatch,dir_name,py_name)

    # check the minimum of the surrogate model
    i_opt = np.argmin(ac_driver.dataset.y_data[0])
    x_opt = ac_driver.dataset.x_data[0][i_opt,:]
    y_opt = np.min(ac_driver.dataset.y_data[0])
    computed_output = [x_opt[0], y_opt]

    # compare expected and computed outputs
    expected_output = [3.0, 0.0] # analytical solution = [x_min, y_min]
    tolerances = [0.1, 0.1]
    output_validator(expected_output, computed_output, tolerances)

    return
"""
# 1st arg is always monkeypatch
# 2nd arg is subdirectory inside examples where the .py is located
# 3rd arg is the name of the .py file for the example
def run_example(monkeypatch,dir_name,py_name):
    monkeypatch.setattr(plt, 'show', lambda: None) # close all plots
    initial_wd = os.getcwd()
    print(initial_wd)
    
    # Find the repository root (where setup.py or environment.yaml exists)
    import pathlib
    current_path = pathlib.Path(initial_wd)
    repo_root = current_path
    while repo_root.parent != repo_root:
        if (repo_root / 'setup.py').exists() or (repo_root / 'environment.yaml').exists():
            break
        repo_root = repo_root.parent
    
    examples_dir = repo_root / 'examples' / dir_name
    os.chdir(str(examples_dir))
    print(os.getcwd())
    print('Testing ' + dir_name + '/' + py_name + '.py:')
    #sys.path.insert(0, '.') # add the path to the current directory. For some reason this doesn't work when multiple tests are run in parallel
    sys.path.insert(0, '../'+dir_name) # add the path to the current directory
    import importlib
    mod = importlib.import_module(py_name)
    example_func = getattr(mod, py_name)
    # call the example code and return its output
    output = example_func()
    os.chdir(initial_wd) # return to the initial working directory
    return output

# 1st arg is a 1d array of expected outputs
# 2nd arg is a 1d array of computed (actual) outputs
# 3rd arg is a 1d array of the allowed difference between expected and computed outputs
def output_validator(expected_output, computed_output, tolerances):
    assert len(expected_output) == len(computed_output)
    assert len(expected_output) == len(tolerances)
    print(f'The expected output = {expected_output}')
    print(f'The computed output = {computed_output}')
    for i in range(len(expected_output)):
        assert abs(expected_output[i] - computed_output[i]) < tolerances[i]
    print('Test passed!')
    return
