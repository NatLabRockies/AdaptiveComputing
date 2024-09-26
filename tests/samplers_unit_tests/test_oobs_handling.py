# To run this script, call "pytest" from the "AdaptiveComputing/" directory
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# add the path to the adaptive_computing module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from adaptive_computing.datasets import DatasetBase
from adaptive_computing.datasets import ContinuousVariable

import pytest

def test_oob_handling():
    # Fail is default behaviour
    ds = DatasetBase([ContinuousVariable(min=0, max=10)],
                     n_fidelity=1,
                     )
    assert ds.nan_behavior == 'fail'
    ds.add_samples([[1.0]], [[1.0]],0)

    with pytest.raises(ValueError) as e_info:
        ds.add_samples([[1.0]],[[np.nan]],0)

    # Ignore and mask
    ds = DatasetBase([ContinuousVariable(min=0, max=10)],
                     n_fidelity=1,
                     nan_behavior='mask_ignore'
                     )

    assert ds.nan_behavior == 'mask_ignore'
    ds.add_samples([[1.0]], [[1.0]],0)
    ds.add_samples([[1.0]],[[np.nan]],0)

    assert (ds.x_data[0].shape == (1,1))  
    assert (ds.y_data[0].shape == (1,1))

    # fail on OOB
    ds = DatasetBase([ContinuousVariable(min=0, max=10)],
                     n_fidelity=1,
                     oob_behavior='fail',
                     y_bounds=(0,1)
                     )

    ds.add_samples([[1.0]], [[1.0]],0)
    with pytest.raises(ValueError) as e_info:
        ds.add_samples([[1.0]],[[5.0]],0)
    return

if __name__ == "__main__":
    test_oob_handling()
