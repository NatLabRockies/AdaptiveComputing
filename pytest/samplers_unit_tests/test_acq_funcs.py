# To run this script, call "pytest" from the "AdaptiveComputing/" directory
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ..test_examples import output_validator

from adaptive_computing.datasets import ContinuousVariable
from adaptive_computing.datasets import DatasetBase
from adaptive_computing.surrogates import SMTGP
from adaptive_computing.samplers import BayesianSampler
from adaptive_computing.samplers.acquisition_functions import expected_improvement
from adaptive_computing.samplers.acquisition_functions import maximum_variance

def test_acq_func_minimum_variance():
    params = [ContinuousVariable(min=0, max=4)]
    dataset = DatasetBase(params)
    # should this work? Should warn not to do this? Add_samples is misleading if can only do 1 sample.
    # Maybe with different array structure this would work, since this ends up being 1 sample that is 4d by 4d.
    # could print message that to clarify how AC is interpretting this input (dimension and number of samples)
    #x_inputs = np.atleast_2d(np.array([0.0, 1.0, 3.0, 4.0]))
    #y_inputs = np.atleast_2d(np.array([0.0, 1.0, 1.0, 0.0]))
    # Should this work? Or fail with a more helpful error message
    #dataset.add_samples(0, 0, 0)
    dataset.add_samples(np.atleast_2d(np.array([0])), np.atleast_2d(np.array([0])), 0)
    dataset.add_samples(np.atleast_2d(np.array([1])), np.atleast_2d(np.array([1])), 0)
    dataset.add_samples(np.atleast_2d(np.array([3])), np.atleast_2d(np.array([1])), 0)
    dataset.add_samples(np.atleast_2d(np.array([4])), np.atleast_2d(np.array([0])), 0)
    sampler = BayesianSampler(dataset, maximum_variance)
    #sampler = BayesianSampler(dataset, expected_improvement)
    surrogate = SMTGP(dataset)
    sample = sampler.get_sample(surrogate, dataset)

    computed_output = [sample[0][0]]

    # compare expected and computed outputs
    expected_output = [2.0]
    tolerances = [0.1]
    output_validator(expected_output, computed_output, tolerances)

    return

if __name__ == "__main__":
    test_acq_func_minimum_variance()
