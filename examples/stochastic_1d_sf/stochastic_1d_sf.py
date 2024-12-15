import matplotlib.pyplot as plt
from adaptive_computing.datasets import ContinuousVariable
#from adaptive_computing.drivers import ActiveLoopDriverCostRatio
from adaptive_computing.datasets import StochasticDataset
import numpy as np

def func_lf(x):
    noise = np.random.normal(0, 0.2)  # Mean = 0, std = 0.2
    return (x-3)**2 + noise


def func_hf(x):
    noise = np.random.normal(0, 0.05)  # Mean = 0, std = 0.05
    return (x-3)**2+0.1*np.sin(x) + noise

# Helper function to compare arrays of arrays
def arrays_of_arrays_equal(arr1, arr2):
    """
    Compare two arrays of arrays (or nested structures) for equality.
    Handles nested np.ndarrays or lists containing np.ndarrays.
    """
    if len(arr1) != len(arr2):
        return False

    for a1, a2 in zip(arr1, arr2):
        # If both are arrays with dtype=object, compare element-wise recursively
        if isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray) and a1.dtype == object and a2.dtype == object:
            if not arrays_of_arrays_equal(a1, a2):
                return False
        # If both are arrays, compare directly
        elif isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
            if not np.array_equal(a1, a2):
                return False
        # For scalar-like values, compare directly
        elif not np.array_equal(np.array(a1), np.array(a2)):
            return False
    return True

def stochastic_1d_mf():

    # 4 input dimensions, each has box constraint on it range of acceptable values
    params = [ContinuousVariable(min=0, max=10), ContinuousVariable(min=0, max=10), ContinuousVariable(min=0, max=10), ContinuousVariable(min=0, max=10)]

    # 1 fidelity level, 2 output dimensions
    dataset = StochasticDataset(params, n_fidelity=1, n_out=2)

    # Test: add a sample containint NaNs
    x_data = np.array([[0.1, 1.1, 2.1, np.NaN],
                   [0.2, 1.2, 2.2, 3.2],
                   [0.4, 1.4, 2.4, 3.4],
                   [0.3, 1.3, 2.3, 3.3]])
    y_data = np.array([[np.array([-1.0, 0.0, 1.0]), np.array([0, 0, 0])],
                    [np.array([2.0, 3.0]), np.array([0, np.NaN])],
                    [np.array([2.0, 3.0]), np.array([np.NaN, np.NaN])],
                    [np.array([-5.0, -6.0, -5.5, -6.5]), np.array([np.NaN, 0, 0, 0])]],
                    dtype=object)
    dataset.add_sample(x_data, y_data, 0)
    assert len(dataset.x_data[0]) == 2
    expected_x_data = np.array([[0.2, 1.2, 2.2, 3.2],
                                [0.3, 1.3, 2.3, 3.3]])
    assert arrays_of_arrays_equal(dataset.x_data[0], expected_x_data)
    assert len(dataset.y_data[0]) == 2
    expected_y_data = np.array([[np.array([2.0]), np.array([0.0])],
                                [np.array([-6.0, -5.5, -6.5]), np.array([0, 0, 0])]],
                            dtype=object)
    assert arrays_of_arrays_equal(dataset.y_data[0], expected_y_data)

    # Test: Higher-dimensional input; should trigger a ValueError
    x_data = np.random.rand(10, 5, 2)  # 3D array
    y_data = np.empty((10, 2), dtype=object)
    try:
        dataset.add_sample(x_data, y_data, 0)
    except ValueError:
        pass  # Expected

    # Test: Using list instead of numpy arrays; should trigger a TypeError
    x_data = [[1.0, 2.0], [3.0, 4.0]]  # List instead of NumPy array
    y_data = [[[], []], [[], []]]
    try:
        dataset.add_sample(x_data, y_data, 0)
    except TypeError:
        pass  # Expected

    # Test: Add a sample with all output dimensions having the same number of entries. This tests if the 2d array of arrays is collapsing to a 3d array.
    x_data = np.array([[0.1, 1.1, 2.1, 3.1]])
    y_data = np.array([[np.array([-1.0, 0.0, 1.0]), np.array([0, 0, 0])]], dtype=object)
    dataset.add_sample(x_data, y_data, 0)
    assert len(dataset.x_data[0]) == 3  # 1 new sample added

    # Test: Single NaN in x_data should ignore the whole sample
    x_data = np.array([[np.NaN, 1.1, 2.1, 3.1]])
    y_data = np.array([[np.array([-1.0, 0.0, 1.0]), np.array([0, 0, 0])]], dtype=object)
    dataset.add_sample(x_data, y_data, 0)
    assert len(dataset.x_data[0]) == 3  # No new samples added

    # Test: Single NaN in y_data should ignore only that stochastic sample
    x_data = np.array([[0.4, 1.4, 2.4, 3.4]])
    y_data = np.array([[np.array([-1.0, 0.0, 1.0]), np.array([np.NaN, 0, 0])]], dtype=object)
    dataset.add_sample(x_data, y_data, 0)
    assert len(dataset.x_data[0]) == 4
    assert arrays_of_arrays_equal(dataset.x_data[0][3], np.array([0.4, 1.4, 2.4, 3.4]))
    expected_y_data_2 = np.array([np.array([0.0, 1.0]), np.array([0.0, 0.0])], dtype=object)
    assert arrays_of_arrays_equal(dataset.y_data[0][3], expected_y_data_2)

    # Test: All NaN in y_data (of at least one output dimension) should ignore the whole sample
    x_data = np.array([[0.5, 1.5, 2.5, 3.5]])
    y_data = np.array([[np.array([0.5, 1.5, 2.5, 3.5]),np.array([np.NaN, np.NaN, np.NaN])]], dtype=object)
    dataset.add_sample(x_data, y_data, 0)
    assert len(dataset.x_data[0]) == 4  # No new samples added

    # Test: If different output dimensions (for the same sample) have a different number of stochastic samples. Should throw an error
    x_data = np.array([[0.1, 1.1, 2.1, 3.1]])
    y_data = np.array([[np.array([-1.0, 0.0, 1.0]), np.array([0, 0, 0, 4])]], dtype=object)
    try:
        dataset.add_sample(x_data, y_data, 0)
        assert False, "Expected error for inconsistent number of stochastic samples"
    except ValueError as e:
        pass  # Expected. Output dimensions have inconsistent number of stochastic samples

    # Test: If inconsistent number of x and y samples, should throw an error
    x_data = np.array([[0.2, 1.2, 2.2, 3.2],
                       [0.3, 1.3, 2.3, 3.3]])
    y_data = np.array([[[-1., 0., 1.],         [0, 0, 0]],
                       [[2., 3.],             [0, np.NaN]],
                       [[-5., -6., -5.5, -6.5], [np.NaN, 0, 0, 0]]], dtype=object)
    try:
        dataset.add_sample(x_data, y_data, 0)
        assert False, "Expected error for inconsistent number of x and y samples"
    except ValueError as e:
        pass # Expected. Number of x_samples and y_samples do not match
    
    # Test: add more stochastic samples to an existing sample
    y_data = np.array([[np.array([2.0, 3.0]), np.array([-7.7, -8.7])]], dtype=object)
    dataset.add_more_sample(2, y_data, 0)
    assert len(dataset._y_data[0][2][0]) == 5  # Original samples (3) + new samples (2)
    assert len(dataset._y_data[0][2][1]) == 5  # Original samples (3) + new samples (2)
    assert np.array_equal(dataset._y_data[0][2][0][-2:], [2.0, 3.0])  # New samples added to first output dim
    assert np.array_equal(dataset._y_data[0][2][1][-2:], [-7.7, -8.7])  # New samples added to second output dim

    # Test: add the non-NaN stochatics samples. Should add one sample
    y_data = np.array([[np.array([2.0, 3.0]), np.array([-7.8, np.NaN])]], dtype=object)
    dataset.add_more_sample(2, y_data, 0)
    assert len(dataset._y_data[0][2][0]) == 6  # One valid sample added to the first output dim
    assert len(dataset._y_data[0][2][1]) == 6  # One valid sample added to the second output dim
    assert dataset._y_data[0][2][0][-1] == 2.0  # Only valid element appended to first output dim
    assert dataset._y_data[0][2][1][-1] == -7.8  # Only valid element appended to second output dim

    # Test: add the non-NaN stochatics samples. should add no samples
    y_data = np.array([[np.array([2.0, 3.0]), np.array([np.NaN, np.NaN])]], dtype=object)
    dataset.add_more_sample(2, y_data, 0)
    assert len(dataset._y_data[0][2][0]) == 6  # No new samples added
    assert len(dataset._y_data[0][2][1]) == 6  # No new samples added

    print("Stochastic database tests complete!")

    return

if __name__ == "__main__":
    stochastic_1d_mf()
