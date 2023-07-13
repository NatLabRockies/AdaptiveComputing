# Run all tutorials
# To run this script, call "pytest" from the "AdaptiveComputing/" directory
import os
import sys
import matplotlib.pyplot as plt

def test_example_pickle_1d(monkeypatch):
    tutorial_tester(monkeypatch,'example_pickle_1d','driver_pickle_1d')
    return

def test_example_query_1d(monkeypatch):
    tutorial_tester(monkeypatch,'example_query_1d','driver_query_1d')
    return

def test_example_1d(monkeypatch):
    tutorial_tester(monkeypatch,'example_1d','driver_1d')
    return

def test_example_2d(monkeypatch):
    tutorial_tester(monkeypatch,'example_2d','driver_2d')
    return

def test_example_3d(monkeypatch):
    tutorial_tester(monkeypatch,'example_3d','driver_3d')
    return

def test_example_mask_1d(monkeypatch):
    tutorial_tester(monkeypatch,'example_mask_1d','driver_mask_1d')
    return

def test_example_mixed_type_3d(monkeypatch):
    tutorial_tester(monkeypatch,'example_mixed_type_3d','driver_mixed_type_3d')
    return

def test_example_mixed_type_read_file_3d(monkeypatch):
    tutorial_tester(monkeypatch,'example_mixed_type_read_file_3d','driver_mt_rf_3d')
    return

def test_example_multifidelity_1d(monkeypatch):
    tutorial_tester(monkeypatch,'example_multifidelity_1d','driver_mf_1d')
    return

def test_example_multifidelity_mixed_type_read_file_2d(monkeypatch):
    tutorial_tester(monkeypatch,'example_multifidelity_mixed_type_read_file_2d','driver_mf_mt_rf')
    return

def tutorial_tester(monkeypatch,dir_name,py_name):
    monkeypatch.setattr(plt, 'show', lambda: None) # close all plots
    initial_wd = os.getcwd()
    print(os.getcwd())
    os.chdir('./tutorials/' + dir_name)
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

#if __name__ == '__main__':
#    test_example_1d(monkeypatch)
#    test_example_2d(monkeypatch)

