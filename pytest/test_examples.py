# Run all examples
# To run this script, call "pytest" from the "AdaptiveComputing/" directory
import os
import sys
import matplotlib.pyplot as plt

def test_bayesian_1d_sf(monkeypatch):
    # 2nd arg is subdirectory inside examples where the .py is located
    # 3rd arg is the name of the .py file for the example
    example_tester(monkeypatch,'bayesian_1d_sf','bayesian_1d_sf')
    return

def example_tester(monkeypatch,dir_name,py_name):
    monkeypatch.setattr(plt, 'show', lambda: None) # close all plots
    initial_wd = os.getcwd()
    print(os.getcwd())
    os.chdir('./examples/' + dir_name)
    print(os.getcwd())
    print('Testing ' + dir_name + '/' + py_name + '.py:')
    #sys.path.insert(0, '.') # add the path to the current directory. For some reason this doesn't work when multiple tests are run in parallel
    sys.path.insert(0, '../'+dir_name) # add the path to the current directory
    import importlib
    mod = importlib.import_module(py_name)

    driver = getattr(mod, py_name)
    expected_values, computed_values, tolerances = driver()
    assert len(expected_values) == len(computed_values)
    assert len(expected_values) == len(tolerances)
    for i in range(len(expected_values)):
        assert abs(expected_values[i] - computed_values[i]) < tolerances[i]
    print('Test ' + dir_name + '/' + py_name + '.py passed!')
    os.chdir(initial_wd) # return to the initial working directory
    print(os.getcwd())
    
    return

